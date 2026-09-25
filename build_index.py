"""
scripts/build_index.py
-----------------------
Builds a FAISS vector index from ARGO profile summaries.
Run once after downloading data.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_PATH = Path("data/processed/argo_indian_ocean.parquet")
INDEX_DIR = Path("data/faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def build_profile_summaries(df: pd.DataFrame) -> list[dict]:
    """Convert dataframe rows into text summaries for embedding."""
    summaries = []
    for profile_id, group in df.groupby("profile_id"):
        surface = group[group["depth_m"] == 0].iloc[0] if len(group[group["depth_m"] == 0]) > 0 else group.iloc[0]
        deep = group[group["depth_m"] == group["depth_m"].max()].iloc[0]

        text = (
            f"ARGO float {surface['float_id']} profile {profile_id} "
            f"measured on {str(surface['date'])[:10]} "
            f"at latitude {surface['latitude']:.2f}, longitude {surface['longitude']:.2f} "
            f"in the {surface['region']}. "
            f"Sea surface temperature: {surface['temperature_c']:.1f}°C, "
            f"surface salinity: {surface['salinity_psu']:.2f} PSU. "
            f"At {deep['depth_m']}m depth: temperature {deep['temperature_c']:.1f}°C, "
            f"salinity {deep['salinity_psu']:.2f} PSU."
        )
        summaries.append({
            "profile_id": profile_id,
            "float_id": surface["float_id"],
            "region": surface["region"],
            "date": str(surface["date"])[:10],
            "latitude": surface["latitude"],
            "longitude": surface["longitude"],
            "text": text,
        })
    return summaries


def build_index():
    print("Loading ARGO data...")
    if not DATA_PATH.exists():
        print("Data not found. Running data generator first...")
        import sys
        sys.path.append(".")
        from scripts.download_argo import main as gen_data
        gen_data()

    df = pd.read_parquet(DATA_PATH)

    print("Building profile summaries...")
    summaries = build_profile_summaries(df)
    print(f"  {len(summaries)} profiles to index")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")
    texts = [s["text"] for s in summaries]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")

    print("Building FAISS index...")
    try:
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, str(INDEX_DIR / "argo.index"))
        print(f"  FAISS index saved ({index.ntotal} vectors)")
    except ImportError:
        print("  FAISS not available — using numpy fallback")
        np.save(str(INDEX_DIR / "embeddings.npy"), embeddings)

    # Save metadata
    with open(INDEX_DIR / "summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"  Metadata saved → {INDEX_DIR / 'summaries.json'}")
    print("Index build complete!")


if __name__ == "__main__":
    build_index()
