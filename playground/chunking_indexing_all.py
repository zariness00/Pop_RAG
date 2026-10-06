"""Prepare lyric chunks and build the local Chroma search index."""

import glob
import hashlib
import os
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = Path(os.getenv("COLLECTION_PATH", ROOT / "my_collection_1")).resolve()
COLLECTION_NAME = "my_collection_1"
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)


def _clean(value):
    return "" if pd.isna(value) else str(value).strip()


def split_lyrics_from_csv(csv_file_path):
    """Return chunks with song metadata, skipping incomplete or duplicate rows."""
    df = pd.read_csv(csv_file_path)
    if not {"Artist", "Title", "Lyric"}.issubset(df.columns):
        raise ValueError("CSV must contain Artist, Title, and Lyric columns")

    chunks = []
    seen_songs = set()
    for _, row in df.iterrows():
        artist = _clean(row["Artist"])
        title = _clean(row["Title"])
        lyrics = _clean(row["Lyric"])
        if not (artist and title and lyrics):
            continue

        album = _clean(row.get("Album", ""))
        year_value = row.get("Year", "")
        try:
            year = int(float(year_value)) if _clean(year_value) else ""
        except (TypeError, ValueError):
            year = ""

        fingerprint = (artist.casefold(), title.casefold(), album.casefold(), year, lyrics)
        if fingerprint in seen_songs:
            continue
        seen_songs.add(fingerprint)

        for index, content in enumerate(SPLITTER.split_text(lyrics), start=1):
            chunks.append(Document(page_content=content, metadata={
                "artist": artist,
                "song_title": title,
                "song_album": album,
                "song_year": year,
                "chunk_index": index,
            }))
    return chunks


def build_index_records(chunks):
    """Build stable, unique IDs for repeatable upserts."""
    records = []
    seen_ids = set()
    for chunk in chunks:
        metadata = chunk.metadata
        identity = "\0".join(str(metadata.get(field, "")) for field in
                             ("artist", "song_title", "song_album", "song_year", "chunk_index"))
        identity += "\0" + chunk.page_content
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        records.append((chunk_id, chunk.page_content, metadata))
    return records


def main():
    csv_folder = Path(os.getenv("CSV_FOLDER", ROOT / "datasets" / "Cleaned_csvs"))
    csv_paths = sorted(glob.glob(str(csv_folder / "*.csv")))
    if not csv_paths:
        raise SystemExit(f"No CSVs found in {csv_folder}")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embedder = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    total = 0
    for csv_path in csv_paths:
        try:
            records = build_index_records(split_lyrics_from_csv(csv_path))
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            print(f"Skipping {csv_path}: {exc}")
            continue
        if not records:
            continue
        ids, documents, metadatas = zip(*records)
        collection.upsert(
            ids=list(ids), documents=list(documents), metadatas=list(metadatas),
            embeddings=embedder(list(documents)),
        )
        total += len(records)
        print(f"Indexed {len(records)} chunks from {Path(csv_path).name}")
    print(f"Indexed {total} chunks into {CHROMA_PATH}")


if __name__ == "__main__":
    main()
