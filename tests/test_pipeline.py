import csv
import json
from types import SimpleNamespace

import pytest

from playground.chunking_indexing_all import build_index_records, split_lyrics_from_csv
from playground.rag_answer import generate_answer
from playground.song_search import search_songs


def write_csv(tmp_path, rows, fieldnames=("Artist", "Title", "Album", "Year", "Lyric")):
    path = tmp_path / "songs.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_preprocessing_splits_long_lyrics_and_preserves_metadata(tmp_path):
    path = write_csv(tmp_path, [{
        "Artist": "Artist", "Title": "Song", "Album": "Album", "Year": "2024.0",
        "Lyric": "A line about love. " * 60,
    }])
    chunks = split_lyrics_from_csv(path)
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 500 for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert chunks[0].metadata == {
        "artist": "Artist", "song_title": "Song", "song_album": "Album",
        "song_year": 2024, "chunk_index": 1,
    }


def test_preprocessing_skips_empty_and_missing_rows(tmp_path):
    path = write_csv(tmp_path, [
        {"Artist": "A", "Title": "Empty", "Lyric": ""},
        {"Artist": "", "Title": "Untitled", "Lyric": "some words"},
        {"Artist": "A", "Title": "Valid", "Lyric": "some words"},
    ])
    chunks = split_lyrics_from_csv(path)
    assert len(chunks) == 1
    assert chunks[0].metadata["song_title"] == "Valid"
    assert chunks[0].metadata["song_album"] == ""
    assert chunks[0].metadata["song_year"] == ""


def test_preprocessing_rejects_missing_required_columns(tmp_path):
    path = write_csv(tmp_path, [{"Artist": "A", "Title": "Song"}], ("Artist", "Title"))
    with pytest.raises(ValueError, match="Lyric"):
        split_lyrics_from_csv(path)


def test_preprocessing_skips_duplicate_rows(tmp_path):
    song = {"Artist": "A", "Title": "Song", "Lyric": "some words"}
    path = write_csv(tmp_path, [song, song])
    assert len(split_lyrics_from_csv(path)) == 1


def test_index_records_have_stable_distinct_ids(tmp_path):
    path = write_csv(tmp_path, [{
        "Artist": "A", "Title": "Song", "Lyric": "Some words. " * 100,
    }])
    chunks = split_lyrics_from_csv(path)
    first = build_index_records(chunks)
    assert first == build_index_records(chunks)
    assert len({record[0] for record in first}) == len(first)


class FakeCollection:
    def __init__(self, result):
        self.result = result
        self.queries = []

    def count(self):
        return len(self.result.get("ids", [[]])[0])

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return self.result


def test_retrieval_returns_expected_fields_and_order():
    collection = FakeCollection({
        "ids": [["1", "2"]], "documents": [["love and loss", "hope and joy"]],
        "metadatas": [[
            {"artist": "A", "song_title": "First", "song_album": "One", "song_year": 2020},
            {"artist": "B", "song_title": "Second"},
        ]],
    })
    songs = search_songs(collection, " breakup ", 2)
    assert [song["title"] for song in songs] == ["First", "Second"]
    assert songs[0] == {
        "id": "1", "artist": "A", "title": "First", "album": "One",
        "year": "2020", "context": "love and loss", "excerpt": "love and loss",
    }
    assert collection.queries[0]["query_texts"] == ["breakup"]


def test_retrieval_deduplicates_song_chunks_and_handles_missing_metadata():
    collection = FakeCollection({
        "ids": [["1", "2", "3"]], "documents": [["first", "repeat", None]],
        "metadatas": [[
            {"artist": "A", "song_title": "Song"},
            {"artist": "A", "song_title": "Song"},
            None,
        ]],
    })
    songs = search_songs(collection, "mood", 3)
    assert len(songs) == 2
    assert songs[1]["title"] == "Unknown title"
    assert songs[1]["excerpt"] == ""


def test_retrieval_cleans_nan_values_from_existing_index_metadata():
    collection = FakeCollection({
        "ids": [["1"]], "documents": [["lyrics"]],
        "metadatas": [[{
            "artist": "Dua Lipa", "song_title": "I’m Free",
            "song_album": "nan", "song_year": float("nan"),
        }]],
    })
    songs = search_songs(collection, "freedom")
    assert songs[0]["album"] == ""
    assert songs[0]["year"] == ""


def test_retrieval_handles_empty_query_and_collection():
    collection = FakeCollection({"ids": [[]]})
    assert search_songs(collection, "  ") == []
    assert search_songs(collection, "mood") == []
    assert collection.queries == []


def test_retrieval_uses_local_embeddings_when_provided():
    collection = FakeCollection({
        "ids": [["1"]], "documents": [["words"]],
        "metadatas": [[{"artist": "A", "song_title": "Song"}]],
    })
    search_songs(collection, "mood", embedder=lambda queries: [[0.1, 0.2]])
    assert collection.queries[0]["query_embeddings"] == [[0.1, 0.2]]
    assert "query_texts" not in collection.queries[0]


@pytest.mark.parametrize("query,limit", [(None, 3), ("mood", 0), ("mood", 11)])
def test_retrieval_rejects_invalid_input(query, limit):
    with pytest.raises(ValueError):
        search_songs(FakeCollection({"ids": [[]]}), query, limit)


def fake_openai_client(content, calls):
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_generation_uses_retrieved_chunk_and_trusted_citation():
    song = {
        "id": "1", "artist": "A", "title": "Song", "album": "Album",
        "year": "2024", "context": "about heartbreak", "excerpt": "about heartbreak",
    }
    calls = []
    client = fake_openai_client(json.dumps({
        "overview": "A song about loss.",
        "recommendations": [{"reason": "It reflects heartbreak."}],
    }), calls)
    answer = generate_answer(client, "sad songs", [song], model="ministral-8b-2512")
    assert answer["recommendations"][0]["song"] is song
    assert answer["overview"] == "A song about loss."
    assert json.loads(calls[0]["messages"][1]["content"])["search_results"][0]["matching_chunk"] == "about heartbreak"
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
    recommendation_schema = calls[0]["response_format"]["json_schema"]["schema"]["properties"]["recommendations"]
    assert recommendation_schema["minItems"] == 1
    assert recommendation_schema["maxItems"] == 1
    assert calls[0]["model"] == "ministral-8b-2512"
    assert calls[0]["max_tokens"] == 700


def test_generation_rejects_missing_recommendations():
    songs = [
        {
            "id": str(number), "artist": "A", "title": f"Song {number}", "album": "",
            "year": "", "context": "words", "excerpt": "words",
        }
        for number in range(1, 4)
    ]
    client = fake_openai_client(json.dumps({
        "overview": "Overview",
        "recommendations": [{"reason": "Reason one"}, {"reason": "Reason two"}],
    }), [])
    with pytest.raises(ValueError, match="one recommendation per song"):
        generate_answer(client, "query", songs)
