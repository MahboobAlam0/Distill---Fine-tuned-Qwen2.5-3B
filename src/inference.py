"""
src/inference.py
----------------
Loads the fine-tuned model and generates SQL from natural language.
Used by both the FastAPI backend and Streamlit app.
"""

import os
import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL    = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_PATH  = os.environ.get("ADAPTER_PATH", "checkpoints/final_v2")
MAX_NEW       = 200

PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a Text-to-SQL assistant. Convert natural language questions "
    "to correct SQL queries based on the given schema.<|im_end|>\n"
    "<|im_start|>user\n"
    "Given the database schema and a natural language question, write the correct SQL query.\n\n"
    "Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class Text2SQLModel:
    def __init__(self):
        self._model     = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return

        print(f"Loading tokenizer from {ADAPTER_PATH}...")
        self._tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        self._tokenizer.pad_token    = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        print(f"Loading base model: {BASE_MODEL}")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float16,
            device_map={"": 0},
        )

        print(f"Loading LoRA adapter: {ADAPTER_PATH}")
        self._model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        self._model = self._model.merge_and_unload()   # merge for faster inference
        self._model.eval()
        print("Model ready.")

    def generate(self, schema: str, question: str) -> dict:
        self._load()

        prompt = PROMPT_TEMPLATE.format(schema=schema, question=question)

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._model.device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = self._tokenizer.decode(
            outputs[0][prompt_len:],
            skip_special_tokens=True,
        )

        # Clean up: take only first SQL statement
        sql = self._clean_sql(generated)

        return {
            "question": question,
            "sql":      sql,
            "raw":      generated,
        }

    def _clean_sql(self, text: str) -> str:
        m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip().split(";")[0].strip() + ";"
        m = re.search(r"```\s*(SELECT|INSERT|UPDATE|DELETE|WITH)\b(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            return (m.group(1) + m.group(2)).strip().split(";")[0].strip() + ";"
        for line in text.splitlines():
            line = line.strip()
            if re.match(r"^(SELECT|INSERT|UPDATE|DELETE|WITH)\b", line, re.IGNORECASE):
                return line.split(";")[0].strip() + ";"
        return text.split(";")[0].strip() + ";"


# Singleton
_model_instance: Text2SQLModel = None


def get_model() -> Text2SQLModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = Text2SQLModel()
    return _model_instance