"""
HuggingFace Spaces entry point — redirects to app/streamlit_app.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

# Set model sources to HF Hub (overrides local defaults in inference.py)
os.environ.setdefault("BASE_MODEL",   "Qwen/Qwen2.5-3B-Instruct")
os.environ.setdefault("ADAPTER_PATH", "mahboobalam0/distill-text2sql")

from app.streamlit_app import *
