import chromadb
from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv("api_key.env")
collection_path = os.getenv("COLLECTION_PATH", str(Path(__file__).resolve().parents[1] / "my_collection_1"))
COLLECTION_NAME  = "my_collection_1"

client     = chromadb.PersistentClient(path= collection_path)
collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata = {"hnsw:space": "cosine"})
def get_completion(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY to use local generation")
    response = OpenAI(api_key=api_key, timeout=20, max_retries=0).chat.completions.create(
        model= "gpt-4o-mini",
        messages=[
            {"role":"system", "content": "You're a helpful assistant who retrieves information from external sources and presents them to the user."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=350,
    )
    return response.choices[0].message.content

# A helper function that takes a user query and returns a well-formatted, pseudo-XML string of the search results
def populate_rag_query(query, n_results=1):
    search_results = collection.query(query_texts=[query], n_results=n_results)
    result_str = ""
    for idx, result in enumerate(search_results["documents"][0]):
        metadata = search_results["metadatas"][0][idx]
        formatted_result = f"""<SEARCH RESULT>
        <DOCUMENT>{result}</DOCUMENT>
        <METADATA>
        <ARTIST>{metadata.get('artist', '')}</ARTIST>
        <TITLE>{metadata.get('song_title', '')}</TITLE>
        <ALBUM>{metadata.get('song_album', '')}</ALBUM>
        <YEAR>{metadata.get('song_year', '')}</YEAR>
        <CHUNK_IDX>{metadata.get('chunk_index', '')}</CHUNK_IDX>
        </METADATA>
        </SEARCH RESULT>"""
        result_str += formatted_result
    return result_str

def make_rag_prompt(query, results):
    return f"""<INSTRUCTIONS>
    <EXAMPLE CITATION>
    Answer to the user query in your own words, drawn from the search results. 
    AND - "Direct quote from source material backing up the claim" - [Source: Song Title, Artist, Album, Year]
    </EXAMPLE CITATION>
    When you finish outlining, please output a **numbered list** of the recommended songs
    </INSTRUCTIONS>
    <USER QUERY>
    {query}
    </USER QUERY>
    <SEARCH RESULTS>
    {results}
    </SEARCH RESULTS>

    Your answer:"""
# all_data = collection.get()                   # returns all documents
# all_ids  = all_data["ids"]                    # list of every ID in the collection
# print(f"Total IDs: {len(all_ids)}")
# print(all_ids[1900:1967])


"""Decoupling-- 
Retrieval model will focus entirely on finding the most relevant search results, 
Generation model can focus on generating "good" responses based on the retrieved data"""

def get_prev_next_chunks(chunk_index: int):
    
    prev_chunk = collection.get(where={"chunk_index": {"$eq": chunk_index - 1}})
    next_chunk = collection.get(where={"chunk_index": {"$eq": chunk_index + 1}})
    return prev_chunk, next_chunk

def expanded_search_results(original_chunk):
    original_chunk_idx = original_chunk["metadatas"][0]["chunk_index"]
    prev_chunk, next_chunk = get_prev_next_chunks(original_chunk_idx)
    result_str = ""
    for chunk in [prev_chunk, original_chunk, next_chunk]:
        if len(chunk["metadatas"])>0:
            meta= chunk["metadatas"][0]
            formatted_result = f"""<SEARCH RESULT>
            <DOCUMENT>{chunk["documents"][0]}</DOCUMENT>
            <METADATA>
            <ARTIST>{meta.get("artist", "")}</ARTIST>
            <TITLE>{meta.get("song_title", "")}</TITLE>
            <ALBUM>{meta.get("song_album", "")}</ALBUM>
            <YEAR>{meta.get("song_year", "")}</YEAR>
            <CHUNK_IDX>{meta.get("chunk_index", "")}</CHUNK_IDX>
            </METADATA>
            </SEARCH RESULT>"""
            result_str += formatted_result
    return result_str


def make_decoupled_rag_prompt(query, n_results=1):
    search_results = collection.query(
        query_texts=[query],
        n_results=n_results
        )
    total_result_str = ""
    for doc_text, metadata in zip(
        search_results["documents"][0],
        search_results["metadatas"][0]
    ):
        chunk = {
            "documents": [doc_text],
            "metadatas": [metadata]
        }

        expanded_result = expanded_search_results(chunk)
        total_result_str += expanded_result
        rag_prompt = make_rag_prompt(query, total_result_str)
    return rag_prompt

if __name__ == "__main__":
    prompt_1 = make_decoupled_rag_prompt("song about being brokenhearted", n_results=3)
    print(get_completion(prompt_1))
