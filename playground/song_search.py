"""Pure formatting and retrieval helpers for the public demo."""


def metadata_text(value):
    """Treat CSV missing-value markers in the existing index as empty text."""
    text = str(value).strip() if value is not None else ""
    return "" if text.casefold() == "nan" else text


def search_songs(collection, query, limit=3, embedder=None):
    """Return up to ``limit`` distinct songs in Chroma relevance order."""
    if not isinstance(query, str):
        raise ValueError("Query must be text")
    query = query.strip()
    if not query:
        return []
    if not 1 <= limit <= 10:
        raise ValueError("Limit must be between 1 and 10")

    count = collection.count()
    if count == 0:
        return []
    query_arg = (
        {"query_embeddings": embedder([query[:300]])}
        if embedder is not None else {"query_texts": [query[:300]]}
    )
    result = collection.query(
        **query_arg, n_results=min(count, limit * 15),
        include=["documents", "metadatas"],
    )
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]

    songs = []
    seen = set()
    for chunk_id, document, metadata in zip(ids, documents, metadatas):
        metadata = metadata if isinstance(metadata, dict) else {}
        artist = metadata_text(metadata.get("artist")) or "Unknown artist"
        title = metadata_text(metadata.get("song_title")) or "Unknown title"
        key = (artist.casefold(), title.casefold())
        if key in seen:
            continue
        seen.add(key)
        excerpt = " ".join(str(document or "").split())
        songs.append({
            "id": str(chunk_id),
            "artist": artist,
            "title": title,
            "album": metadata_text(metadata.get("song_album")),
            "year": metadata_text(metadata.get("song_year")),
            "context": excerpt[:500],
            "excerpt": excerpt[:180] + ("…" if len(excerpt) > 180 else ""),
        })
        if len(songs) == limit:
            break
    return songs
