"""
api/main.py
-----------
FastAPI service for Text-to-SQL inference.

Endpoints:
  POST /generate   -- convert natural language to SQL
  GET  /health     -- liveness check

Run:
  uvicorn api.main:app --reload --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import get_model

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = get_model()
    model._load()    # warm up on startup
    yield


app = FastAPI(
    title="Distill API",
    description="QLoRA fine-tuned Qwen2.5-3B for natural language to SQL generation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    question: str = Field(
        ..., min_length=5, max_length=500,
        example="Find all employees who earn more than 50000.",
    )
    schema: str = Field(
        ..., min_length=5, max_length=2000,
        example="CREATE TABLE employees (id INT, name TEXT, salary INT, dept_id INT);",
    )


class GenerateResponse(BaseModel):
    question:   str
    sql:        str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model:  str


# ── Routes ───────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model="Distill — Qwen2.5-3B QLoRA")


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()
    try:
        result = model.generate(schema=req.schema, question=req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency = round((time.perf_counter() - t0) * 1000, 1)

    return GenerateResponse(
        question=result["question"],
        sql=result["sql"],
        latency_ms=latency,
    )