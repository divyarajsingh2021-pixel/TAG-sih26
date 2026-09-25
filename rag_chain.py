"""
backend/rag_chain.py
---------------------
RAG pipeline: semantic search over ARGO profiles + Groq LLM for answers.
"""

from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer
from groq import Groq

INDEX_DIR = Path("data/faiss_index")
DATA_PATH = Path("data/processed/argo_indian_ocean.parquet")

_model: Optional[SentenceTransformer] = None
_summaries: Optional[list] = None
_embeddings: Optional[np.ndarray] = None
_df: Optional[pd.DataFrame] = None


def _load_resources():
    global _model, _summaries, _embeddings, _df

    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    if _summaries is None:
        summary_path = INDEX_DIR / "summaries.json"
        if summary_path.exists():
            with open(summary_path) as f:
                _summaries = json.load(f)
        else:
            _summaries = []

    if _embeddings is None:
        emb_path = INDEX_DIR / "embeddings.npy"
        if emb_path.exists():
            _embeddings = np.load(str(emb_path))
        elif _summaries:
            texts = [s["text"] for s in _summaries]
            _embeddings = _model.encode(texts, batch_size=64)
            _embeddings = np.array(_embeddings, dtype="float32")

    if _df is None and DATA_PATH.exists():
        _df = pd.read_parquet(DATA_PATH)


def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """Find most relevant ARGO profiles for a query."""
    _load_resources()
    if not _summaries or _embeddings is None:
        return []

    query_vec = _model.encode([query], convert_to_numpy=True).astype("float32")

    try:
        import faiss
        index_path = INDEX_DIR / "argo.index"
        if index_path.exists():
            index = faiss.read_index(str(index_path))
            _, indices = index.search(query_vec, top_k)
            return [_summaries[i] for i in indices[0] if i < len(_summaries)]
    except ImportError:
        pass

    # Numpy fallback cosine similarity
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True) + 1e-9
    normed = _embeddings / norms
    qnorm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    sims = normed @ qnorm.T
    top_idx = np.argsort(sims[:, 0])[::-1][:top_k]
    return [_summaries[i] for i in top_idx]


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
    # Retrieve relevant profiles
    results = semantic_search(query, top_k=5)
    data_summary = get_data_summary(query)

    context = "\n".join([r["text"] for r in results]) if results else "No specific profiles found."

    system_prompt = """You are FloatChat, an expert oceanography assistant specializing in ARGO float data 
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
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM error: {str(e)}. Please check your Groq API key."
