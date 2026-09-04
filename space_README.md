---
title: Distill — Fine-tuned Qwen2.5-3B for Text-to-SQL
emoji: ⚗️
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: true
license: mit
short_description: Natural language → SQL using QLoRA fine-tuned Qwen2.5-3B (13× improvement over base)
---

# ⚗️ Distill

QLoRA fine-tuned **Qwen2.5-3B-Instruct** for Text-to-SQL.  
Type a question in plain English, get a valid SQL query back.

| Metric | Base Model | Distill |
|---|---|---|
| Exact Match | 2.33% | **30.67%** (+13×) |
| BLEU Score | 51.87% | **59.74%** |
| Execution Acc. | — | **55.03%** |

**Model:** [mahboobalam0/distill-text2sql](https://huggingface.co/mahboobalam0/distill-text2sql)  
**Code:** [GitHub](https://github.com/MahboobAlam0/Distill---Fine-tuned-Qwen2.5-3B)
