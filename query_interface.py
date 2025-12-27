"""Query interface: test recommendations from Chroma collection using similarity search.

Usage:
  python query_interface.py --chroma_dir ./chroma_db --collection songs
"""
from typing import List, Dict, Any
import argparse
import shlex

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
    """Compose embedding text from individual fields (same as in embedder.py)"""
    parts = [str(title), str(artist), str(keywords), str(style), str(vibe)]
    return " | ".join([p for p in parts if p])


def get_embedding(text: str, model_name: str = DEFAULT_MODEL) -> List[float]:
    """Generate a single embedding for a query text"""
    model = SentenceTransformer(model_name)
    embedding = model.encode([text])[0]
    return embedding.tolist()


def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load the embedding model once for interactive usage."""
    return SentenceTransformer(model_name)


def query_chroma(query_embedding: List[float], chroma_dir: str, collection_name: str, n_results: int = 5) -> Dict:
    """Query Chroma collection by similarity"""
    if chromadb is None:
        raise ImportError("chromadb is not installed.")

    try:
        client = chromadb.PersistentClient(path=chroma_dir)
    except Exception:
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=chroma_dir))
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error: collection '{collection_name}' not found in {chroma_dir}")
        return {"results": []}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "distances"]
    )
    return results


def format_results(results: Dict) -> str:
    """Pretty-print query results"""
    if not results.get("metadatas") or not results["metadatas"][0]:
        return "No results found."
    
    output = []
    for i, metadata in enumerate(results["metadatas"][0]):
        distance = results["distances"][0][i] if i < len(results["distances"][0]) else "N/A"
        # Chroma uses Euclidean distance; lower is more similar
        similarity = 1 / (1 + distance)  # rough conversion to similarity score 0-1
        
        output.append(f"\n--- Result {i+1} (Similarity: {similarity:.3f}) ---")
        output.append(f"Title: {metadata.get('title', 'N/A')}")
        output.append(f"Artist: {metadata.get('artist', 'N/A')}")
        output.append(f"Keywords: {metadata.get('keywords', 'N/A')}")
        output.append(f"Style: {metadata.get('style', 'N/A')}")
        output.append(f"Vibe: {metadata.get('vibe', 'N/A')}")
        output.append(f"Time (s): {metadata.get('time', 'N/A')}")
        output.append(f"Track ID: {metadata.get('track', 'N/A')}")
    
    return "\n".join(output)


def diversify_results_by_artist(results: Dict[str, Any], top_k: int, max_per_artist: int = 1) -> Dict[str, Any]:
    """Post-process Chroma results to diversify by artist.

    Chroma returns lists shaped like: results['metadatas'][0], results['distances'][0].
    We keep order (best-first) while enforcing at most `max_per_artist` per artist.
    """
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []

    filtered_meta: List[Dict[str, Any]] = []
    filtered_dist: List[float] = []

    per_artist: Dict[str, int] = {}
    for i, meta in enumerate(metadatas):
        artist = str((meta or {}).get("artist", "") or "").strip().lower() or "__unknown__"
        per_artist.setdefault(artist, 0)
        if per_artist[artist] >= max_per_artist:
            continue

        per_artist[artist] += 1
        filtered_meta.append(meta)
        if i < len(distances):
            filtered_dist.append(distances[i])
        else:
            filtered_dist.append(float("nan"))

        if len(filtered_meta) >= top_k:
            break

    # Keep the same overall shape as Chroma
    return {"metadatas": [filtered_meta], "distances": [filtered_dist]}


def interactive_query(
    chroma_dir: str,
    collection_name: str,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 5,
    diverse_artists: bool = True,
    max_per_artist: int = 1,
    fetch_multiplier: int = 5,
):
    """Interactive loop for querying recommendations"""
    print("\n" + "="*70)
    print("CHROMA MUSIC RECOMMENDER - Interactive Query Interface")
    print("="*70)
    print("\nInstruções:")
    print("  1. Enter 'title=<title> artist=<artist> keywords=<keywords> style=<style> vibe=<vibe>'")
    print("  2. Or simpler: 'title=<title> artist=<artist>'")
    print("  3. Type 'quit' to exit")
    print("  4. Type 'help' for examples")
    print("\nDica:")
    print("  - Por padrão, já tenta sugerir artistas diferentes (max 1 por artista).")
    print("  - Para desligar: --no_diverse_artists")
    print("="*70 + "\n")

    model = get_model(model_name)

    def embed_query(text: str) -> List[float]:
        emb = model.encode([text])[0]
        return emb.tolist()

    def parse_kv_or_free_text(user_text: str) -> Dict:
        """Parse key=value pairs with support for quoted values.

        Examples:
          title="Bohemian Rhapsody" artist=Queen
          keywords=rock vibe=energetic

        If no key=value appears, treat the entire input as free text under 'free'.
        """
        query_fields = {"title": "", "artist": "", "keywords": "", "style": "", "vibe": "", "free": ""}
        try:
            tokens = shlex.split(user_text, posix=True)
        except Exception:
            tokens = user_text.split()

        has_kv = any("=" in t for t in tokens)
        if not has_kv:
            query_fields["free"] = user_text.strip()
            return query_fields

        for token in tokens:
            if "=" not in token:
                continue
            key, val = token.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key in query_fields:
                query_fields[key] = val
        return query_fields

    while True:
        user_input = input("Enter query (or 'quit'/'help'): ").strip()
        
        if user_input.lower() == "quit":
            print("Bye!")
            break
        
        if user_input.lower() == "help":
            print("\n--- Example Queries ---")
            print('title="Bohemian Rhapsody" artist=Queen')
            print('title="Shape of You" artist="Ed Sheeran" keywords=pop style=modern vibe=upbeat')
            print("keywords=rock vibe=energetic")
            print("lofi chill study beats")
            print("")
            continue

        query_fields = parse_kv_or_free_text(user_input)

        if query_fields.get("free"):
            query_text = query_fields["free"]
        else:
            if not any(query_fields[k] for k in ["title", "artist", "keywords", "style", "vibe"]):
                print("No valid fields provided.")
                continue
            query_text = compose_embed_text(
                title=query_fields.get("title", ""),
                artist=query_fields.get("artist", ""),
                keywords=query_fields.get("keywords", ""),
                style=query_fields.get("style", ""),
                vibe=query_fields.get("vibe", ""),
            )
        
        print("\nGenerating embedding for query...")
        print(f"Query text: {query_text}\n")
        
        try:
            query_embedding = embed_query(query_text)
            print("Querying Chroma...\n")
            # Fetch more candidates, then optionally diversify by artist.
            n_fetch = max(top_k, top_k * max(1, fetch_multiplier))
            results = query_chroma(query_embedding, chroma_dir, collection_name, n_results=n_fetch)
            if diverse_artists:
                results = diversify_results_by_artist(results, top_k=top_k, max_per_artist=max_per_artist)
            formatted = format_results(results)
            print(formatted)
        except Exception as e:
            print(f"Error: {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chroma_dir", default="./chroma_db", help="Chroma persist directory")
    p.add_argument("--collection", default="songs", help="Chroma collection name")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-Transformers model name")
    p.add_argument("--top_k", type=int, default=5, help="How many recommendations to return")
    # Default ON: better UX (avoids returning 5 results from the same artist).
    p.add_argument(
        "--diverse_artists",
        dest="diverse_artists",
        action="store_true",
        default=True,
        help="Diversify results by artist (default: enabled)",
    )
    p.add_argument(
        "--no_diverse_artists",
        dest="diverse_artists",
        action="store_false",
        help="Disable artist diversification",
    )
    p.add_argument(
        "--max_per_artist",
        type=int,
        default=1,
        help="Max results per artist when --diverse_artists is enabled",
    )
    p.add_argument(
        "--fetch_multiplier",
        type=int,
        default=5,
        help="Fetch top_k * multiplier candidates before diversifying",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    interactive_query(
        args.chroma_dir,
        args.collection,
        model_name=args.model,
        top_k=args.top_k,
        diverse_artists=args.diverse_artists,
        max_per_artist=args.max_per_artist,
        fetch_multiplier=args.fetch_multiplier,
    )
