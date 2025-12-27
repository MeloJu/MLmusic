"""Basic UI for the MLMusics recommender.

Run:
  streamlit run app.py

This UI queries the local ChromaDB collection created by embedder.py.
"""

from __future__ import annotations

from typing import Any, Dict, List

import sys

import streamlit as st

try:
    from query_interface import (
        DEFAULT_MODEL,
        compose_embed_text,
        diversify_results_by_artist,
        format_results,
        query_chroma,
    )
except Exception as e:
    st.error(
        "Falha ao importar dependências do projeto.\n\n"
        "Isso normalmente acontece quando o Streamlit está rodando com o Python errado (ex.: Conda `base` / system Python) "
        "em vez da `.venv` do projeto.\n\n"
        "Rode assim:\n"
        "- `./.venv/Scripts/python.exe -m streamlit run app.py`\n"
        "- ou `./.venv/Scripts/python.exe run_ui.py`\n"
    )
    st.code(f"sys.executable = {sys.executable}")
    raise

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    st.error(
        "`sentence-transformers` não está disponível neste Python. "
        "Instale as dependências na `.venv` e rode a UI com o Python da `.venv`."
    )
    st.code(f"sys.executable = {sys.executable}")
    raise


st.set_page_config(page_title="MLMusics", page_icon="🎵", layout="centered")


@st.cache_resource
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def to_table(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []

    rows: List[Dict[str, Any]] = []
    for i, meta in enumerate(metadatas):
        dist = distances[i] if i < len(distances) else None
        sim = (1 / (1 + dist)) if isinstance(dist, (int, float)) else None
        rows.append(
            {
                "similarity": round(sim, 3) if sim is not None else None,
                "title": (meta or {}).get("title", ""),
                "artist": (meta or {}).get("artist", ""),
                "keywords": (meta or {}).get("keywords", ""),
                "style": (meta or {}).get("style", ""),
                "vibe": (meta or {}).get("vibe", ""),
                "time_s": (meta or {}).get("time", ""),
                "track": (meta or {}).get("track", ""),
            }
        )
    return rows


st.title("MLMusics")
st.caption("Recomendação por embeddings + busca por similaridade (ChromaDB)")

with st.sidebar:
    st.subheader("Config")
    st.caption(f"Python: {sys.executable}")
    chroma_dir = st.text_input("Chroma dir", value="./chroma_db")
    collection = st.text_input("Collection", value="songs")
    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5)

    st.divider()
    st.subheader("Diversificação")
    diverse_artists = st.checkbox("Artistas diferentes", value=True)
    max_per_artist = st.slider("Máx por artista", min_value=1, max_value=5, value=1)
    fetch_multiplier = st.slider("Fetch multiplier", min_value=1, max_value=20, value=8)

st.subheader("Consulta")
mode = st.radio("Modo", ["Texto livre", "Campos (title/artist/etc.)"], horizontal=True)

query_text = ""
if mode == "Texto livre":
    query_text = st.text_input("Digite sua busca", value="lofi chill study beats")
else:
    c1, c2 = st.columns(2)
    with c1:
        title = st.text_input("title", value="Hello")
        keywords = st.text_input("keywords", value="")
        vibe = st.text_input("vibe", value="")
    with c2:
        artist = st.text_input("artist", value="Adele")
        style = st.text_input("style", value="")

    query_text = compose_embed_text(title=title, artist=artist, keywords=keywords, style=style, vibe=vibe)

run = st.button("Recomendar")

if run:
    if not query_text.strip():
        st.error("Preencha a consulta.")
    else:
        with st.spinner("Gerando embedding e consultando o Chroma..."):
            model = load_model(model_name)
            emb = model.encode([query_text])[0].tolist()
            n_fetch = max(top_k, top_k * max(1, fetch_multiplier))
            results = query_chroma(emb, chroma_dir, collection, n_results=n_fetch)
            if diverse_artists:
                results = diversify_results_by_artist(results, top_k=top_k, max_per_artist=max_per_artist)

        st.success("Pronto!")
        st.write("Query:")
        st.code(query_text)

        st.write("Resultados (tabela):")
        st.dataframe(to_table(results), use_container_width=True)

        st.write("Resultados (texto):")
        st.text(format_results(results))
