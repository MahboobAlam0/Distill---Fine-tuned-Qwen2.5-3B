"""
app/streamlit_app.py
--------------------
Streamlit UI for Text-to-SQL inference.

Run:
  streamlit run app/streamlit_app.py
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.inference import get_model

st.set_page_config(page_title="Distill", page_icon="⚗️", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("⚗️ Distill")
    st.caption("Natural language → SQL")
    st.markdown("---")
    st.markdown("**Base Model:** Qwen2.5-3B-Instruct")
    st.markdown("**Method:** QLoRA · 4-bit NF4 · LoRA r=16")
    st.markdown("**Dataset:** sql-create-context · 28K examples")
    st.markdown("---")
    st.markdown("**Results (vs base model):**")
    st.markdown("- Exact Match: 2.33% → **30.67%** (+28.3 pp)")
    st.markdown("- BLEU Score: 51.87% → **59.74%** (+7.9 pp)")
    st.markdown("- Execution Acc: **55.03%**")
    st.markdown("---")
    st.caption("Run `python src/evaluate.py --baseline --max_samples 300` to reproduce.")

# ── Load model ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading fine-tuned model...")
def load_model():
    m = get_model()
    m._load()
    return m

model = load_model()

# ── Main UI ──────────────────────────────────────────────────
st.title("⚗️ Distill")
st.caption("QLoRA fine-tuned Qwen2.5-3B · Natural language → SQL")

# Example schemas
EXAMPLES = {
    "E-commerce": {
        "schema": (
            "CREATE TABLE customers (id INT, name TEXT, email TEXT, city TEXT);\n"
            "CREATE TABLE orders (id INT, customer_id INT, total FLOAT, status TEXT, created_at DATE);\n"
            "CREATE TABLE products (id INT, name TEXT, price FLOAT, category TEXT, stock INT);"
        ),
        "questions": [
            "Find all customers from New York who placed more than 3 orders.",
            "What is the total revenue per product category?",
            "List the top 5 most expensive products still in stock.",
        ],
    },
    "HR": {
        "schema": (
            "CREATE TABLE employees (id INT, name TEXT, salary FLOAT, dept_id INT, hire_date DATE);\n"
            "CREATE TABLE departments (id INT, name TEXT, manager_id INT, budget FLOAT);"
        ),
        "questions": [
            "Find employees hired in the last 6 months earning above 60000.",
            "Which department has the highest average salary?",
            "List all employees whose salary exceeds their department budget divided by 10.",
        ],
    },
    "Hospital": {
        "schema": (
            "CREATE TABLE patients (id INT, name TEXT, age INT, gender TEXT, diagnosis TEXT);\n"
            "CREATE TABLE appointments (id INT, patient_id INT, doctor_id INT, date DATE, status TEXT);\n"
            "CREATE TABLE doctors (id INT, name TEXT, specialty TEXT, experience_years INT);"
        ),
        "questions": [
            "How many patients were diagnosed with diabetes this year?",
            "Find doctors with more than 10 years experience in cardiology.",
            "List patients who had more than 2 cancelled appointments.",
        ],
    },
}

# Domain selector
domain = st.selectbox("Choose a domain:", list(EXAMPLES.keys()))
ex = EXAMPLES[domain]

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Database Schema")
    schema = st.text_area("Edit schema:", value=ex["schema"], height=200)

with col2:
    st.markdown("#### Example Questions")
    for q in ex["questions"]:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["question"] = q

question = st.text_input(
    "Natural language question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. Find all customers from New York who placed more than 3 orders.",
)

if st.button("⚡ Generate SQL", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not schema.strip():
        st.warning("Please provide a schema.")
    else:
        with st.spinner("Generating SQL..."):
            import time
            t0 = time.perf_counter()
            result = model.generate(schema=schema, question=question)
            latency = round((time.perf_counter() - t0) * 1000)

        st.markdown("---")
        st.markdown("#### Generated SQL")
        st.code(result["sql"], language="sql")
        st.caption(f"⏱️ Latency: {latency} ms")