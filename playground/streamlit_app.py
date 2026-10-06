"""Pop song retrieval and grounded answer generation."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb
import streamlit as st
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from openai import APIStatusError, OpenAI, RateLimitError

from playground.rag_answer import generate_answer
from playground.song_search import search_songs


COLLECTION_PATH = Path(os.getenv("COLLECTION_PATH", ROOT / "my_collection_1")).resolve()
COLLECTION_NAME = "my_collection_1"
MISTRAL_URL = "https://api.mistral.ai/v1"
MISTRAL_MODEL = "ministral-8b-2512"


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=str(COLLECTION_PATH))
    return client.get_collection(name=COLLECTION_NAME)


@st.cache_resource
def load_embedder():
    return ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])


st.set_page_config(page_title="Pop RAG Search", page_icon="🎵")
st.title("Pop RAG Search")
st.write("Describe a mood or theme. The app finds matching lyric chunks, then uses them to recommend songs.")

query = st.text_input("What kind of song are you looking for?", placeholder="e.g. songs about a painful breakup")
limit = st.slider("Number of songs", min_value=1, max_value=10, value=3)
mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
if not mistral_key:
    try:
        mistral_key = str(st.secrets.get("MISTRAL_API_KEY") or "").strip()
    except FileNotFoundError:
        pass
if mistral_key:
    st.caption("Answers are generated from the retrieved lyrics with Mistral.")
else:
    st.caption("Answer generation is temporarily unavailable; retrieval still works.")

if st.button("Search and answer", type="primary"):
    st.session_state["songs"] = []
    st.session_state["answer"] = None
    st.session_state["search_error"] = False
    if not query.strip():
        st.warning("Enter a search phrase first.")
    else:
        try:
            with st.spinner("Finding matching songs…"):
                st.session_state["songs"] = search_songs(
                    load_collection(), query, limit, embedder=load_embedder()
                )
        except Exception:
            st.session_state["search_error"] = True
            st.error("Search is temporarily unavailable. Please try again later.")

        if st.session_state["songs"]:
            if mistral_key:
                try:
                    with st.spinner("Writing a grounded answer…"):
                        client = OpenAI(
                            api_key=mistral_key,
                            base_url=MISTRAL_URL,
                            timeout=20,
                            max_retries=0,
                        )
                        st.session_state["answer"] = generate_answer(
                            client,
                            query.strip(),
                            st.session_state["songs"],
                            model=MISTRAL_MODEL,
                        )
                except RateLimitError:
                    st.error("Mistral is busy or its usage limit was reached. Please try again later.")
                except APIStatusError:
                    st.error("Mistral could not generate an answer. Please try again later.")
                except Exception:
                    st.error("Answer generation failed. Please try again later.")
            else:
                st.info("Answer generation is temporarily unavailable. Retrieved evidence is shown below.")

answer = st.session_state.get("answer")
if answer:
    st.subheader("Answer")
    st.write(answer["overview"])
    for number, recommendation in enumerate(answer["recommendations"], start=1):
        song = recommendation["song"]
        st.markdown(f"**{number}. {song['title']} — {song['artist']}**")
        st.write(recommendation["reason"])
        citation = ", ".join(part for part in (
            song["title"], song["artist"], song["album"], song["year"]
        ) if part)
        st.caption(f"Source: {citation}")

songs = st.session_state.get("songs", [])
if songs:
    with st.expander("Retrieved evidence"):
        for number, song in enumerate(songs, start=1):
            st.write(f"{number}. {song['title']} — {song['artist']}: {song['excerpt']}")
elif "songs" in st.session_state and not st.session_state["search_error"] and query.strip():
    st.info("No matching songs found. Try another phrase.")
