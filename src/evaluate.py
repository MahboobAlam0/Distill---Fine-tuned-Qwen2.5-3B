"""
src/evaluate.py
---------------
Evaluates fine-tuned model on sql-create-context validation split.

Metrics:
  - Exact Match Accuracy  : predicted SQL == gold SQL (string normalised)
  - BLEU Score            : token-level n-gram overlap with gold SQL
  - Execution Accuracy    : predicted SQL produces same result as gold SQL
                            (schema-seeded in-memory SQLite; meaningful for
                            structural correctness, not data correctness)

Run:
    python src/evaluate.py --model_path checkpoints/final_v2
    python src/evaluate.py --model_path checkpoints/final_v2 --baseline --max_samples 300
"""

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

nltk.download("punkt", quiet=True)

VAL_PATH   = "data/val.jsonl"
MAX_NEW    = 200
BATCH_SIZE = 16

INSTRUCTION_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a Text-to-SQL assistant. Convert natural language questions "
    "to correct SQL queries based on the given schema.<|im_end|>\n"
    "<|im_start|>user\n"
    "{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

_SMOOTHER = SmoothingFunction().method1


# ── SQL helpers ───────────────────────────────────────────────
def normalise_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.lower().strip().rstrip(";")).strip()


def sql_tokens(sql: str) -> list:
    """Tokenise SQL on whitespace + operators for BLEU scoring."""
    return re.findall(r"\w+|[^\w\s]", normalise_sql(sql))


def bleu_score(pred: str, gold: str) -> float:
    ref  = sql_tokens(gold)
    hyp  = sql_tokens(pred)
    if not hyp:
        return 0.0
    return sentence_bleu([ref], hyp, smoothing_function=_SMOOTHER)


def extract_schema(prompt: str) -> str:
    match = re.search(r"Schema:\n(.*?)\n\nQuestion:", prompt, re.DOTALL)
    return match.group(1).strip() if match else ""


def exec_sql(sql: str, schema: str = "") -> Optional[list]:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys = OFF")
        cursor = conn.cursor()
        if schema:
            cursor.executescript(schema)
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return sorted(result)
    except Exception:
        return None


# ── Inference ─────────────────────────────────────────────────
def load_model(model_path: str, base_model: str, is_finetuned: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if is_finetuned else base_model,
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,
        device_map={"": 0},
    )

    if is_finetuned:
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        print(f"Loaded fine-tuned model from {model_path}")
    else:
        model = base
        print("Loaded base model (no fine-tuning)")

    model.eval()
    return model, tokenizer


def _extract_sql(text: str) -> str:
    """Pull a clean SQL statement out of verbose / markdown-wrapped output."""
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().split(";")[0].strip() + ";"
    m = re.search(r"```\s*(SELECT|INSERT|UPDATE|DELETE|WITH)\b(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return (m.group(1) + m.group(2)).strip().split(";")[0].strip() + ";"
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE)\b", line, re.IGNORECASE):
            return line.split(";")[0].strip() + ";"
    return text.split(";")[0].strip() + ";"


def generate_sql(prompts: list, model, tokenizer) -> list:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, out in enumerate(outputs):
        prompt_len = inputs["input_ids"].shape[1]
        generated  = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        results.append(_extract_sql(generated))
    return results


# ── Evaluation ────────────────────────────────────────────────
@dataclass
class EvalStats:
    total:        int   = 0
    exact_match:  int   = 0
    exec_correct: int   = 0
    bleu_sum:     float = 0.0
    by_difficulty: dict = field(default_factory=lambda: {
        "easy":   {"total": 0, "em": 0, "bleu": 0.0},
        "medium": {"total": 0, "em": 0, "bleu": 0.0},
        "hard":   {"total": 0, "em": 0, "bleu": 0.0},
        "extra":  {"total": 0, "em": 0, "bleu": 0.0},
    })


def evaluate_model(model_path: str, base_model: str, is_finetuned: bool,
                   max_samples: int = 0) -> EvalStats:
    model, tokenizer = load_model(model_path, base_model, is_finetuned)

    val_data = []
    with open(VAL_PATH, encoding="utf-8") as f:
        for line in f:
            val_data.append(json.loads(line))

    if max_samples > 0:
        val_data = val_data[:max_samples]

    stats = EvalStats(total=len(val_data))

    for i in tqdm(range(0, len(val_data), BATCH_SIZE), desc="Evaluating"):
        batch       = val_data[i : i + BATCH_SIZE]
        prompts     = [INSTRUCTION_TEMPLATE.format(prompt=ex["prompt"]) for ex in batch]
        predictions = generate_sql(prompts, model, tokenizer)

        for ex, pred in zip(batch, predictions):
            gold   = ex["completion"]
            diff   = ex.get("difficulty", "")

            em   = normalise_sql(pred) == normalise_sql(gold)
            bleu = bleu_score(pred, gold)

            if em:
                stats.exact_match += 1
            stats.bleu_sum += bleu

            schema      = extract_schema(ex["prompt"])
            pred_result = exec_sql(pred, schema)
            gold_result = exec_sql(gold, schema)
            if (pred_result is not None and gold_result is not None
                    and pred_result == gold_result):
                stats.exec_correct += 1

            if diff in stats.by_difficulty:
                stats.by_difficulty[diff]["total"] += 1
                if em:
                    stats.by_difficulty[diff]["em"] += 1
                stats.by_difficulty[diff]["bleu"] += bleu

    return stats


def print_results(label: str, stats: EvalStats):
    em   = stats.exact_match  / stats.total * 100
    exc  = stats.exec_correct / stats.total * 100
    bleu = stats.bleu_sum     / stats.total * 100
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(f"  Total examples     : {stats.total}")
    print(f"  Exact Match Acc.   : {em:.2f}%")
    print(f"  BLEU Score         : {bleu:.2f}%")
    print(f"  Execution Acc.*    : {exc:.2f}%")
    print(f"  (* schema-only SQLite, no real data rows)")
    filled = {k: v for k, v in stats.by_difficulty.items() if v["total"] > 0}
    if filled:
        print(f"\n  By Difficulty (Exact Match | BLEU):")
        for diff, v in filled.items():
            acc  = v["em"]   / v["total"] * 100
            bavg = v["bleu"] / v["total"] * 100
            print(f"    {diff:<8}: EM={acc:.1f}%  BLEU={bavg:.1f}%  ({v['em']}/{v['total']})")
    print(f"{'='*52}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  default="checkpoints/final_v2",
                        help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--base_model",  default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model name/path")
    parser.add_argument("--baseline",    action="store_true",
                        help="Also evaluate base model for comparison")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit to first N samples (0 = all)")
    args = parser.parse_args()

    if args.baseline:
        base_stats = evaluate_model(args.model_path, args.base_model,
                                    is_finetuned=False, max_samples=args.max_samples)
        print_results("BASE MODEL (Qwen2.5-3B-Instruct, no fine-tuning)", base_stats)

    ft_stats = evaluate_model(args.model_path, args.base_model,
                               is_finetuned=True, max_samples=args.max_samples)
    print_results("FINE-TUNED MODEL (QLoRA)", ft_stats)

    if args.baseline:
        em_delta   = (ft_stats.exact_match  - base_stats.exact_match)  / base_stats.total * 100
        bleu_delta = (ft_stats.bleu_sum     - base_stats.bleu_sum)     / base_stats.total * 100
        exc_delta  = (ft_stats.exec_correct - base_stats.exec_correct) / base_stats.total * 100
        print(f"\n{'='*52}")
        print(f"  Improvement (fine-tuned vs base)")
        print(f"{'='*52}")
        print(f"  Exact Match  : {em_delta:+.2f}pp")
        print(f"  BLEU Score   : {bleu_delta:+.2f}pp")
        print(f"  Exec Acc.*   : {exc_delta:+.2f}pp")
        print(f"{'='*52}")


if __name__ == "__main__":
    main()
