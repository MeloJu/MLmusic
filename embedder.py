"""Embedder: create sentence embeddings using Hugging Face (sentence-transformers)
and store them in a local Chroma collection.

Usage:
  - Ensure dependencies are installed: `python -m pip install -r requirements.txt`
    - Run: `python embedder.py --csv itunes_songs.csv --chroma_dir ./chroma_db --collection songs`

The embedding metadata will include: title, track, time, keywords, artist, style, vibe
"""
from typing import List, Dict, Optional
import os
import argparse
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError("sentence-transformers is required. Install via requirements.txt")

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def compose_embed_text(title: str = "", artist: str = "", keywords: str = "", style: str = "", vibe: str = "") -> str:
    """Compose embedding text from individual fields. Can be reused in query interface."""
    parts = [str(title), str(artist), str(keywords), str(style), str(vibe)]
    return " | ".join([p for p in parts if p])


def extract_metadatas(df: pd.DataFrame, cols=None) -> List[Dict]:
    """Extract metadata records from DataFrame for specified columns."""
    if cols is None:
        cols = ["title", "track", "time", "keywords", "artist", "style", "vibe"]
    return df[cols].fillna("").to_dict(orient="records")


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def ensure_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has the required metadata fields. If some are missing,
    attempt to map common iTunes fields to the required names.
    Required fields: title, track, time, keywords, artist, style, vibe
    """
    df = df.copy()
    # Map common iTunes fields
    if "title" not in df.columns:
        if "trackName" in df.columns:
            df["title"] = df["trackName"]
        elif "collectionName" in df.columns:
            df["title"] = df["collectionName"]
        else:
            df["title"] = ""

    if "track" not in df.columns:
        if "trackId" in df.columns:
            df["track"] = df["trackId"].astype(str)
        else:
            # fallback to index-based id
            df["track"] = df.index.astype(str)

    if "time" not in df.columns:
        if "trackTimeMillis" in df.columns:
            # convert millis to seconds
            df["time"] = (df["trackTimeMillis"] / 1000).fillna(0)
        else:
            df["time"] = ""

    # Optional fields: keywords, artist, style, vibe
    if "keywords" not in df.columns:
        df["keywords"] = df.get("primaryGenreName", "")

    if "artist" not in df.columns:
        df["artist"] = df.get("artistName", "")

    if "style" not in df.columns:
        df["style"] = ""

    if "vibe" not in df.columns:
        df["vibe"] = ""

    # Compose embedding text using the extracted function (reusable in query interface)
    def compose_row_text(row):
        return compose_embed_text(
            title=row.get("title", ""),
            artist=row.get("artist", ""),
            keywords=row.get("keywords", ""),
            style=row.get("style", ""),
            vibe=row.get("vibe", "")
        )

    df["_embed_text"] = df.apply(compose_row_text, axis=1)
    return df


def get_embeddings(texts: List[str], model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> List[List[float]]:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings


def save_to_chroma(embeddings: List[List[float]], metadatas: List[Dict], ids: List[str], chroma_dir: str, collection_name: str, batch_size: int = 500):
    """Save embeddings and metadata to Chroma using upsert with batching.
    
    Args:
        embeddings: List of embedding vectors (floats)
        metadatas: List of metadata dicts (one per embedding)
        ids: List of unique IDs (one per embedding)
        chroma_dir: Directory to persist Chroma DB
        collection_name: Name of the collection
        batch_size: Number of items per batch (default 500 to avoid memory spikes)
    """
    if chromadb is None:
        raise ImportError("chromadb is not installed. Add it to requirements.txt and install.")

    # Chroma has changed APIs across versions; prefer PersistentClient when available.
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
    except Exception:
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=chroma_dir))

    # create or get collection
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)

    # upsert in batches to avoid memory spikes and handle duplicates safely
    total = len(ids)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_ids = ids[start:end]
        batch_emb = embeddings[start:end]
        batch_meta = metadatas[start:end]
        
        print(f"  Upserting batch {start // batch_size + 1} ({start}-{end}/{total})...")
        collection.upsert(ids=batch_ids, embeddings=batch_emb, metadatas=batch_meta)
    
    # Newer Chroma persists automatically; older versions need persist().
    print("Persisting to disk...")
    persist = getattr(client, "persist", None)
    if callable(persist):
        client.persist()
    return True


def run(csv_path: str, chroma_dir: str, collection_name: str, model_name: str):
    df = load_csv(csv_path)
    df = ensure_fields(df)

    texts = df["_embed_text"].astype(str).tolist()
    ids = df["track"].astype(str).tolist()

    print(f"Generating embeddings for {len(texts)} records using {model_name}...")
    embeddings = get_embeddings(texts, model_name=model_name)

    metadatas = extract_metadatas(df)

    print(f"Saving to Chroma at {chroma_dir} in collection '{collection_name}'...")
    save_to_chroma(embeddings, metadatas, ids, chroma_dir, collection_name)
    print("Done!")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV with records")
    p.add_argument("--chroma_dir", default="./chroma_db", help="Chroma persist directory")
    p.add_argument("--collection", default="songs", help="Chroma collection name")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-Transformers model name")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.csv, args.chroma_dir, args.collection, args.model)
