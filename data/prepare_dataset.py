"""
data/prepare_dataset.py
-----------------------
Downloads b-mc2/sql-create-context dataset and formats it for fine-tuning.
Every example has a real CREATE TABLE schema — 78,577 total examples.

Dataset : b-mc2/sql-create-context
Source  : https://huggingface.co/datasets/b-mc2/sql-create-context
License : CC BY 4.0

Run:
    python data/prepare_dataset.py
Output:
    data/train.jsonl  (28,000 examples)
    data/val.jsonl    (7,858 examples)
"""

import json
import os
from datasets import load_dataset

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

INSTRUCTION = (
    "Given the database schema and a natural language question, "
    "write the correct SQL query.\n\n"
    "Schema:\n{schema}\n\n"
    "Question: {question}\n\n"
    "SQL:"
)


def save_jsonl(examples: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} examples -> {path}")


def main():
    print("Loading sql-create-context dataset...")
    ds = load_dataset("b-mc2/sql-create-context", split="train")
    print(f"Total examples: {len(ds)}")

    # 90/10 train/val split (val: 78,577*0.1 ≈ 7,858, matches README)
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"].shuffle(seed=42).select(range(28000))  # README: 28,000 train examples
    val_ds   = split["test"]

    train_data, val_data = [], []

    for row in train_ds:
        train_data.append({
            "prompt":     INSTRUCTION.format(schema=row["context"], question=row["question"]),
            "completion": row["answer"].strip(),
        })

    for row in val_ds:
        val_data.append({
            "prompt":     INSTRUCTION.format(schema=row["context"], question=row["question"]),
            "completion": row["answer"].strip(),
        })

    # Show sample to verify real schema
    print(f"\nSample prompt:\n{train_data[0]['prompt'][:400]}\n...")
    print(f"\nSample SQL: {train_data[0]['completion']}")

    save_jsonl(train_data, os.path.join(OUT_DIR, "train.jsonl"))
    save_jsonl(val_data,   os.path.join(OUT_DIR, "val.jsonl"))

    print(f"\nDataset summary:")
    print(f"  Train : {len(train_data)} examples")
    print(f"  Val   : {len(val_data)} examples")


if __name__ == "__main__":
    main()
