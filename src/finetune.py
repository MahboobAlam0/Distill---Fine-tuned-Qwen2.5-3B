"""
src/finetune.py
---------------
QLoRA fine-tuning of Llama-3.2-3B on Spider Text-to-SQL.

Hardware : Free Kaggle GPU (2x T4, 30GB) or Colab A100 (40GB)
Method   : QLoRA — 4-bit quantization + LoRA adapters
Time     : ~2 hours on Kaggle T4 x2

Run:
    python src/finetune.py

Outputs:
    checkpoints/final/        ← LoRA adapter weights (not full model)
    checkpoints/final/README  ← merge instructions
"""

import json
import os
import math

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
    EarlyStoppingCallback,
)

# ── Config ────────────────────────────────────────────────────
BASE_MODEL     = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR     = "checkpoints"
TRAIN_PATH     = "data/train.jsonl"
VAL_PATH       = "data/val.jsonl"
MAX_LENGTH     = 512
BATCH_SIZE     = 4       # per device
GRAD_ACCUM     = 4       # effective batch = 16
EPOCHS         = 3
LR             = 2e-4
WARMUP_RATIO   = 0.05
LORA_R         = 16      # LoRA rank
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
SEED           = 42

# Target modules for Llama-3 architecture
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

INSTRUCTION_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are a Text-to-SQL assistant. Convert natural language questions "
    "to correct SQL queries based on the given schema.<|eot_id|>\n"
    "<|start_header_id|>user<|end_header_id|>\n"
    "{prompt}<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "{completion}<|eot_id|>"
)


# ── Data loading ──────────────────────────────────────────────
def load_jsonl(path: str) -> list:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_and_tokenize(examples, tokenizer):
    """Format examples into Llama-3 chat template and tokenize."""
    input_ids_list, attention_masks, labels_list = [], [], []

    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        full_text = INSTRUCTION_TEMPLATE.format(
            prompt=prompt,
            completion=completion,
        )
        tokenized = tokenizer(
            full_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )
        input_ids = tokenized["input_ids"]

        # Build labels: mask the prompt tokens with -100 (only train on completion)
        prompt_text = INSTRUCTION_TEMPLATE.format(
            prompt=prompt,
            completion="",
        )
        prompt_tokenized = tokenizer(prompt_text, max_length=MAX_LENGTH, truncation=True)
        prompt_len = len(prompt_tokenized["input_ids"])

        labels = [-100] * prompt_len + input_ids[prompt_len:]

        input_ids_list.append(input_ids)
        attention_masks.append(tokenized["attention_mask"])
        labels_list.append(labels)

    return {
        "input_ids":      input_ids_list,
        "attention_mask": attention_masks,
        "labels":         labels_list,
    }


# ── Model loading ─────────────────────────────────────────────
def load_model_and_tokenizer():
    print(f"Loading base model: {BASE_MODEL}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 — better than int4
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,     # nested quantization saves ~0.4GB
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"    # important for causal LM

    return model, tokenizer


# ── LoRA config ───────────────────────────────────────────────
def apply_lora(model):
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Training ──────────────────────────────────────────────────
def train():
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)

    # Load and tokenize datasets
    print("Preparing datasets...")
    train_raw = load_jsonl(TRAIN_PATH)
    val_raw   = load_jsonl(VAL_PATH)

    train_ds = Dataset.from_list(train_raw)
    val_ds   = Dataset.from_list(val_raw)

    tokenize_fn = lambda x: format_and_tokenize(x, tokenizer)

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(tokenize_fn,   batched=True, remove_columns=val_ds.column_names)

    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        fp16=False,
        bf16=True,                          # bfloat16 more stable than fp16
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",                   # set to "wandb" if you want tracking
        seed=SEED,
        dataloader_num_workers=2,
        group_by_length=True,               # speeds up training by reducing padding
        optim="paged_adamw_8bit",           # memory-efficient optimizer for QLoRA
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training...")
    trainer.train()

    # Save final adapter
    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"LoRA adapter saved → {final_path}")

    # Log final eval loss
    metrics = trainer.evaluate()
    perplexity = math.exp(metrics["eval_loss"])
    print(f"\nFinal eval loss : {metrics['eval_loss']:.4f}")
    print(f"Perplexity      : {perplexity:.2f}")

    return final_path


if __name__ == "__main__":
    train()