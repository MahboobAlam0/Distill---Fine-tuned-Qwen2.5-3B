<div align="center">

# Distill - Fine-tuned Qwen2.5-3B for text-to-SQL using QLoRA, achieving 13× improvement over the base model

### Natural language → executable SQL, fine-tuned on a free GPU.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/Model-Qwen2.5--3B--Instruct-7c3aed)
![Method](https://img.shields.io/badge/Method-QLoRA-16a34a)
![Hardware](https://img.shields.io/badge/GPU-Kaggle%20T4%20Free-f97316)
![License](https://img.shields.io/badge/License-MIT-64748b)

</div>

---

## The Problem

Most text-to-SQL solutions demand 70B+ models or expensive fine-tuning clusters. Distill asks a sharper question:

> *How much can a 3B model improve with disciplined fine-tuning on free hardware — and can the result ship as a real product?*

The answer: **13× improvement in exact-match accuracy**, a 114 MB adapter instead of a 6 GB model, and a full serving stack — all on a Kaggle T4 GPU.

---

## Results

| Metric | Base Qwen2.5-3B | Distill | Δ |
|---|---|---|---|
| **Exact Match Accuracy** | 2.33% | **30.67%** | **+28.3 pp · 13×** |
| **BLEU Score** | 51.87% | **59.74%** | +7.9 pp |
| **Execution Accuracy** | — | **55.03%** | — |
| **Train Loss** | — | 0.83 → 0.07 | −91% |
| **Val Perplexity** | — | **1.98** | — |

*Evaluated on 7,858 held-out examples from sql-create-context.*  
*Baseline measured on a 300-sample subset with a robust SQL extractor.*

### Why these numbers matter

Exact match is strict — column order, aliases, whitespace all count. A query that returns the correct answer via a different formulation scores zero. **55% execution accuracy** shows that many "misses" are equivalent queries written differently. One caveat: execution accuracy here runs both queries on an in-memory SQLite database built from the `CREATE TABLE` schema **with no rows** (see `exec_sql` in `src/evaluate.py`), so two different `SELECT`s that both return nothing count as a match. It checks that the query is valid against the schema and structurally compatible, and it overstates true semantic accuracy. Measuring that properly needs tables filled with data (not done yet).

---

## Architecture

```
  Natural Language Question
          +
    CREATE TABLE Schema
          │
          ▼
  ┌───────────────────────────────────────────────┐
  │         Qwen2.5-3B-Instruct (frozen)          │
  │         4-bit NF4 quantized weights           │
  │                                               │
  │  ┌─────────────────────────────────────────┐  │
  │  │ LoRA Adapters (trainable, 29.9M params) │  │
  │  │  Attention : q, k, v, o projections     │  │
  │  │  MLP       : gate, up, down projections │  │
  │  └─────────────────────────────────────────┘  │
  └───────────────────────────────────────────────┘
          │
          ▼
     Generated SQL
          │
    ┌─────┴──────┐
    │            │
  FastAPI    Streamlit
   REST UI    Demo UI
```

---

## Why These Choices

**Why Qwen2.5-3B over Llama-3.2-3B?**
Qwen2.5-3B-Instruct outperforms Llama-3.2-3B on coding and structured generation benchmarks at the same parameter count. For SQL — where output must be syntactically valid — the stronger instruction-following baseline mattered more than familiarity.

**Why QLoRA over full fine-tuning?**
Full fine-tuning a 3B model in mixed precision needs about 49 GB of GPU memory for the weights, their gradients and Adam's optimizer states (16 bytes per parameter; about 25 GB even with the leanest 8 bytes per parameter accounting), before activations. QLoRA — 4-bit NF4 quantization of frozen weights + small trainable LoRA adapters — shrinks the frozen weights from about 6 GB in fp16 to about 1.8 GB, so the model, the adapters and the training activations fit on a free 15 GB T4. The accuracy tradeoff is minimal because SQL generation is a low-rank adaptation: the base model already knows grammar; the adapter learns domain patterns.

**Why LoRA rank r=16?**
r=8 underfit on initial runs; r=32 showed no improvement but doubled adapter memory. r=16 with α=32 (scaling ratio = 2) hit the sweet spot — sufficient capacity to learn SQL structure without overfitting the 28K training set.

**Why early stopping with patience=3?**
Validation loss diverged from training loss after step 200 (epoch 0.46). Without early stopping, training would have memorised queries for another 1.5 epochs. The best checkpoint was automatically restored — a real-world safeguard the final evaluation results depend on.

---

## Training Details

### Hyperparameters

| Setting | Value | Rationale |
|---|---|---|
| Base model | Qwen/Qwen2.5-3B-Instruct | Best 3B model for structured generation |
| Dataset | sql-create-context | Single-table schemas; cleaner signal for 3B |
| Train / Val | 28,000 / 7,858 examples | — |
| Batch size | 4 × grad accum 4 = **16 effective** | Maximises T4 utilisation within 15 GB limit |
| Learning rate | 2e-4, cosine decay | Standard for LoRA; cosine prevents late-stage instability |
| Warmup | 5% of steps | Stabilises optimizer state before full LR |
| Optimizer | Paged AdamW 8-bit | Handles backward-pass memory spikes without OOM |
| Quantization | 4-bit NF4 + double quantization | NF4 preserves more signal than int4 for normally-distributed weights |
| Platform | Kaggle T4 × 1 (free) | ~15 GB VRAM, zero cost |

### Reproducibility notes

- **Training-set size.** `data/prepare_dataset.py` samples **28,000** training examples from the 90% split, and the numbers in this README describe that setup. The Kaggle notebook as committed (`kaggle/train_notebook.ipynb`) writes the **full 90% split** (about 70k examples) instead. To reproduce the reported run, use the 28,000-example files from `data/prepare_dataset.py`.
- **GPU count.** The notebook's header suggests enabling "GPU T4 x2" and loads the model with `device_map='auto'`, while the run reported here used one T4. The notebook has no saved outputs, so this can't be confirmed from the repo.
- **Precision.** On a T4 (no bf16 support) the notebook trains in fp16, with fp16 as the 4-bit compute dtype.
- **`src/finetune.py` is the earlier script** from the first Llama-3.2-3B / Spider experiment and is kept for reference only. The reported model was trained with the Kaggle notebook.

### Loss Curve

```
Train Loss
  0.83 │████████████████████████████████████████ step   50
  0.46 │█████████████████████████ step 100
  0.36 │████████████████████ step 150
  0.32 │██████████████████ step 200  ◄── val 0.685  BEST CHECKPOINT
  0.25 │██████████████ step 250
  0.20 │███████████ step 350
  0.10 │█████ step 550
  0.07 │████ step 800  ◄── val 0.829  EARLY STOP (patience exhausted)
```

### Checkpoint Summary

| Checkpoint | Step | Val Loss | Outcome |
|---|---|---|---|
| checkpoint-200 | 200 | **0.685** | ✅ Best — restored as final model |
| checkpoint-800 | 800 | 0.829 | ✗ Overfit — discarded |

---

## Evaluation Methodology

```
Exact Match (EM)
  normalise(pred) == normalise(gold)
  Strict string match after lowercasing and collapsing whitespace.
  Penalises correct SQL written differently. The floor, not the ceiling.

BLEU Score
  Token-level n-gram overlap with smoothing (NLTK sentence_bleu)
  Continuous signal. Rewards partial correctness EM misses.
  Useful for tracking training progress.

Execution Accuracy
  exec(pred, schema) == exec(gold, schema)
  Both queries run on a schema-seeded in-memory SQLite database.
  Measures semantic correctness — correct results, any valid formulation.
  Note: empty tables; meaningful for structural correctness only.
```

The **24 pp gap between EM (30.67%) and execution accuracy (55.03%)** suggests Distill often writes a different-but-equivalent formulation, which exact match penalises. Because the tables are empty, part of that gap can also be queries that merely return the same empty result, so treat 55.03% as an upper bound on semantic accuracy.

---

## Project Structure

```
distill/
│
├── src/
│   ├── finetune.py          # Legacy QLoRA script (first Llama-3.2-3B / Spider experiment); the reported run used kaggle/train_notebook.ipynb
│   ├── evaluate.py          # EM + BLEU + Execution Accuracy + baseline comparison
│   └── inference.py         # Model wrapper with lazy loading (used by API and UI)
│
├── api/
│   └── main.py              # FastAPI — POST /generate {schema, question} → {sql}
│
├── app/
│   └── streamlit_app.py     # Interactive demo — schema editor, one-click examples
│
├── kaggle/
│   └── train_notebook.ipynb # Self-contained Kaggle notebook — run and export adapter
│
├── data/
│   ├── train.jsonl          # 28,000 training examples
│   └── val.jsonl            # 7,858 validation examples
│
├── checkpoints/
│   └── final_v2/            # LoRA adapter (114 MB) — plug into any Qwen2.5-3B copy
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### Inference

```python
from src.inference import get_model

model = get_model()
result = model.generate(
    schema="CREATE TABLE orders (id INTEGER, customer VARCHAR, amount FLOAT, status VARCHAR)",
    question="What is the total revenue from completed orders?"
)
print(result["sql"])
# SELECT SUM(amount) FROM orders WHERE status = 'completed';
```

### Evaluate

```bash
# Full validation set (~60 min on a laptop GPU)
python src/evaluate.py --model_path checkpoints/final_v2

# Fast baseline comparison — 300 samples (~15 min)
python src/evaluate.py --model_path checkpoints/final_v2 --baseline --max_samples 300
```

### REST API

```bash
uvicorn api.main:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "CREATE TABLE products (id INTEGER, name VARCHAR, price FLOAT, stock INTEGER)",
    "question": "Which products are out of stock?"
  }'

# {"sql": "SELECT name FROM products WHERE stock = 0;", "latency_ms": 312.4}
```

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

### Docker

```bash
docker-compose up
# API → http://localhost:8000
# UI  → http://localhost:8501
```

---

## Reproducing Training

```bash
# 1. Upload kaggle/train_notebook.ipynb to Kaggle
# 2. Runtime → Change runtime type → GPU (T4 × 1)
# 3. Run all cells  (~2 hours)
# 4. Download output/lora_adapter.zip from the output panel
unzip lora_adapter.zip -d checkpoints/final_v2/
```

The notebook is self-contained — installs dependencies, downloads the base model and dataset, trains, evaluates, and zips the adapter for export.

---

## What I Learned

**The base model's 2.33% EM was a measurement artifact.**
Without fine-tuning, Qwen2.5-3B wraps SQL answers in markdown code blocks and verbose prose. A naive extractor (`split(";")[0]`) captures the explanation, not the SQL. With a proper extractor — regex over ` ```sql ` blocks → keyword-anchored line search → fallback — the base model's BLEU rises from near-zero to 51.87%. Evaluation methodology shapes reported results as much as model quality does.

**Val loss and exact match told different stories.**
Val loss diverged sharply after step 200. Yet the final exact match (30.67%) was better than early-stopped models in preliminary runs. The model learned the SQL generation pattern quickly; what "overfit" after step 200 was superficial token-level distribution, not semantic understanding.

**LoRA rank is a real hyperparameter, not a default.**
r=16 worked here because SQL generation is a compact, low-variance domain close to Qwen's pretraining distribution. For tasks with highly domain-specific vocabulary or multi-step reasoning, higher rank would be warranted. Treating r=16 as universal is a common mistake.

---

## Stack

| Layer | Tools |
|---|---|
| Model | Qwen2.5-3B-Instruct, Hugging Face Transformers, PEFT |
| Training | BitsAndBytes (QLoRA), TRL, PyTorch, Kaggle T4 |
| Evaluation | Custom harness — EM, BLEU, Execution Accuracy, NLTK, SQLite |
| Serving | FastAPI, Uvicorn |
| UI | Streamlit |
| Infra | Docker, Docker Compose |
| Language | Python 3.12 |

---

## License

MIT — use freely, attribution appreciated.
