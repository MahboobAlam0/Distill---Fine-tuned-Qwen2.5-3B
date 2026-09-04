"""
tests/test_pipeline.py
-----------------------
Unit tests for data processing, SQL normalisation, and inference.

Run:
    pytest tests/ -v --cov=src
"""

import os
import sys
import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate import normalise_sql, exec_sql
from src.inference import Text2SQLModel


# ── normalise_sql ─────────────────────────────────────────────
class TestNormaliseSQL:
    def test_lowercases(self):
        assert normalise_sql("SELECT * FROM Users") == "select * from users"

    def test_strips_trailing_semicolon(self):
        assert normalise_sql("SELECT id FROM t;") == "select id from t"

    def test_collapses_whitespace(self):
        assert normalise_sql("SELECT  id   FROM   t") == "select id from t"

    def test_equivalent_queries_match(self):
        q1 = "SELECT name FROM employees WHERE salary > 50000;"
        q2 = "select name  from employees  where salary > 50000"
        assert normalise_sql(q1) == normalise_sql(q2)


# ── exec_sql ──────────────────────────────────────────────────
class TestExecSQL:
    def test_valid_sql_returns_result(self):
        result = exec_sql("SELECT 1 + 1")
        assert result == [(2,)]

    def test_invalid_sql_returns_none(self):
        result = exec_sql("SELEC broken sql @@##")
        assert result is None

    def test_create_and_query(self):
        # Use temp file DB to test real DDL
        import sqlite3, tempfile, os
        db = tempfile.mktemp(suffix=".db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE t (id INT, val TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
        conn.commit()
        conn.close()
        result = exec_sql("SELECT * FROM t ORDER BY id", db_path=db)
        assert result == [(1, "a"), (2, "b")]
        os.remove(db)


# ── Text2SQLModel ─────────────────────────────────────────────
class TestText2SQLModel:
    def test_clean_sql_extracts_select(self):
        m = Text2SQLModel()
        raw = "Here is the SQL:\nSELECT name FROM users WHERE age > 18;"
        assert "SELECT" in m._clean_sql(raw).upper()
        assert m._clean_sql(raw).strip().endswith(";")

    def test_clean_sql_handles_no_semicolon(self):
        m = Text2SQLModel()
        raw = "SELECT id FROM orders"
        result = m._clean_sql(raw)
        assert result.endswith(";")

    def test_clean_sql_takes_first_statement(self):
        m = Text2SQLModel()
        raw = "SELECT a FROM t1; SELECT b FROM t2;"
        result = m._clean_sql(raw)
        assert result.count(";") == 1

    @patch("src.inference.PeftModel")
    @patch("src.inference.AutoModelForCausalLM")
    @patch("src.inference.AutoTokenizer")
    def test_generate_returns_dict(self, mock_tok_cls, mock_model_cls, mock_peft_cls):
        import torch

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.eos_token     = "<eos>"
        mock_tok.eos_token_id  = 0
        mock_tok.return_value  = {
            "input_ids":      torch.zeros((1, 10), dtype=torch.long),
            "attention_mask": torch.ones((1, 10), dtype=torch.long),
        }
        mock_tok.decode.return_value = "SELECT name FROM employees WHERE salary > 50000;"
        mock_tok_cls.from_pretrained.return_value = mock_tok

        # Mock model
        mock_base = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_base
        mock_peft = MagicMock()
        mock_peft_cls.from_pretrained.return_value = mock_peft
        mock_merged = MagicMock()
        mock_peft.merge_and_unload.return_value = mock_merged
        mock_merged.device = torch.device("cpu")
        mock_merged.generate.return_value = torch.zeros((1, 20), dtype=torch.long)

        model = Text2SQLModel()
        model._model     = mock_merged
        model._tokenizer = mock_tok

        result = model.generate(
            schema="CREATE TABLE employees (id INT, name TEXT, salary FLOAT);",
            question="Find all employees earning more than 50000.",
        )
        assert "question" in result
        assert "sql" in result
        assert isinstance(result["sql"], str)
        assert result["sql"].endswith(";")


# ── Data formatting ───────────────────────────────────────────
class TestDataPrep:
    def test_jsonl_round_trip(self):
        data = [
            {"prompt": "Schema: t\nQuestion: q", "completion": "SELECT * FROM t;", "db_id": "test", "difficulty": "easy"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
            path = f.name

        loaded = []
        with open(path) as f:
            for line in f:
                loaded.append(json.loads(line))

        assert len(loaded) == 1
        assert loaded[0]["completion"] == "SELECT * FROM t;"
        os.remove(path)