# Pop RAG Search

<img src="assets/sabrina.png" alt="Pop RAG Search cover" width="180" align="left">

FYI: This project began as part of a university course in Information Retrieval. I later prepared it for public deployment as a portfolio project.

<br clear="left">

## How it works
1. The Streamlit app retrieves relevant lyric chunks from a local Chroma index
2. Mistral uses the retrieved evidence to generate structured song recommendations with citations
3. The Mistral API key is stored securely in Streamlit secrets. Retrieved evidence remains available if answer generation is temporarily unavailable.

[View the project presentation](Zarina_B_IR_PRESENTATION.pdf)

Have fun!
P.S I used Codex for helping me to run tests and deploy!
## To run locally

Use Python 3.12:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run playground/streamlit_app.py
```

The app reads the committed Chroma collection in `my_collection_1/`. Its first search may take longer while Chroma downloads its default embedding model. To test generation locally, set `MISTRAL_API_KEY` in your environment. Retrieval alone does not need a key.

## Tests and indexing

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

The tests cover chunking, metadata, missing input, duplicate rows, stable index IDs, and retrieval output. To build a fresh local collection from `datasets/Cleaned_csvs/`:

```bash
COLLECTION_PATH=./my_collection_rebuilt .venv/bin/python playground/chunking_indexing_all.py
COLLECTION_PATH=./my_collection_rebuilt .venv/bin/streamlit run playground/streamlit_app.py
```

The committed collection was built with older IDs. Index into a fresh directory to avoid mixing those entries with newly generated IDs.
