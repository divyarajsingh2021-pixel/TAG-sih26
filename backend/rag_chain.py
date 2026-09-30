"""
backend/rag_chain.py
---------------------
RAG pipeline: semantic search over ARGO profiles + Groq LLM for answers.
Lightweight & robust for cloud deployment (pure NumPy fallback without heavy torch requirement).
"""

from __future__ import annotations

import json
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from groq import Groq

INDEX_DIR = Path("data/faiss_index")
DATA_PATH = Path("data/processed/argo_indian_ocean.parquet")

_model = None
_summaries: Optional[list] = None
_embeddings: Optional[np.ndarray] = None
_df: Optional[pd.DataFrame] = None


def _load_resources():
    global _model, _summaries, _embeddings, _df

    if _summaries is None:
        summary_path = INDEX_DIR / "summaries.json"
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                _summaries = json.load(f)
        else:
            _summaries = []

    if _embeddings is None:
        emb_path = INDEX_DIR / "embeddings.npy"
        if emb_path.exists():
            try:
                _embeddings = np.load(str(emb_path))
            except Exception:
                _embeddings = None

    if _df is None and DATA_PATH.exists():
        try:
            _df = pd.read_parquet(DATA_PATH)
        except Exception:
            _df = None

    # Try loading sentence-transformers if available
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _model = False


def _keyword_search(query: str, top_k: int = 5) -> list[dict]:
    """Fast keyword/BM25-style search over ARGO summaries without torch/transformers."""
    if not _summaries:
        return []
    
    words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
    if not words:
        return _summaries[:top_k]
    
    scores = []
    for s in _summaries:
        text = s.get("text", "").lower()
        score = sum(text.count(w) * (3 if w in ["arabian", "bengal", "salinity", "temperature", "sst", "depth"] else 1) for w in words)
        scores.append(score)
    
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [_summaries[i] for i in ranked_indices if scores[i] > 0] or _summaries[:top_k]


def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """Find most relevant ARGO profiles for a query."""
    _load_resources()
    if not _summaries:
        return []

    # If neural embedding model is loaded, use vector search
    if _model and _embeddings is not None:
        try:
            query_vec = _model.encode([query], convert_to_numpy=True).astype("float32")
            norms = np.linalg.norm(_embeddings, axis=1, keepdims=True) + 1e-9
            normed = _embeddings / norms
            qnorm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
            sims = normed @ qnorm.T
            top_idx = np.argsort(sims[:, 0])[::-1][:top_k]
            return [_summaries[i] for i in top_idx]
        except Exception:
            pass

    # Lightweight fallback
    return _keyword_search(query, top_k=top_k)


def get_data_summary(query: str) -> str:
    """Get a structured data summary relevant to the query."""
    _load_resources()
    if _df is None:
        return "No data available."

    df = _df
    summary_parts = []
    ql = query.lower()

    # Filter by region if mentioned
    for region in ["arabian sea", "bay of bengal", "indian ocean"]:
        if region in ql:
            filtered = df[df["region"].str.lower() == region]
            if len(filtered) > 0:
                df = filtered
                summary_parts.append(f"Region: {region.title()}")
            break

    # Filter by month if mentioned
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    for month_name, month_num in month_map.items():
        if month_name in ql:
            df = df[df["month"] == month_num]
            summary_parts.append(f"Month: {month_name.title()}")
            break

    surface = df[df["depth_m"] == 0]
    if len(surface) == 0:
        surface = df

    stats = {
        "total_profiles": df["profile_id"].nunique(),
        "total_floats": df["float_id"].nunique(),
        "avg_sst": round(surface["temperature_c"].mean(), 2),
        "avg_salinity": round(surface["salinity_psu"].mean(), 2),
        "lat_range": f"{surface['latitude'].min():.1f} to {surface['latitude'].max():.1f}",
        "lon_range": f"{surface['longitude'].min():.1f} to {surface['longitude'].max():.1f}",
    }
    summary_parts.append(str(stats))
    return " | ".join(summary_parts)


def answer_query(query: str, groq_api_key: str) -> str:
    """Full RAG pipeline: retrieve context + generate answer with Groq LLM."""
    results = semantic_search(query, top_k=5)
    data_summary = get_data_summary(query)

    context = "\n".join([r.get("text", "") for r in results]) if results else "No specific profiles found."

    system_prompt = """You are FloatChat (Outrage AI), an expert oceanography assistant specializing in ARGO float data 
from the Indian Ocean. Answer questions about ocean temperature, salinity, float trajectories, 
and oceanographic patterns using the provided context. Be precise with numbers and scientific.
If you don't have enough data to answer accurately, say so clearly."""

    user_prompt = f"""Context from ARGO database:
{context}

Data statistics:
{data_summary}

User question: {query}

Provide a clear, scientific answer based on the ARGO data above."""

    try:
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI response: {str(e)}"
