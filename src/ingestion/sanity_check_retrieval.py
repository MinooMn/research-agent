"""
Quick sanity check: embed a test query, search the FAISS index,
print which paper/chunk each result came from. Not a formal test —
just a manual check that retrieval returns plausible results.
"""

import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

INDEX_PATH = "data/index/faiss.index"
CHUNKS_PATH = "data/chunked_corpus/chunks.jsonl"
TOP_K = 5

# Load chunks (row order must match what was embedded/indexed)
chunks = []
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

# Load index + model
index = faiss.read_index(INDEX_PATH)
model = SentenceTransformer("all-MiniLM-L6-v2")


def search(query: str, top_k: int = TOP_K):
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)

    print(f"\nQuery: {query!r}\n")
    for rank, (row, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        chunk = chunks[row]
        preview = chunk["chunk_text"][:200].replace("\n", " ")
        print(f"[{rank}] dist={dist:.3f} | {chunk['chunk_id']}")
        print(f"    {preview}...\n")


if __name__ == "__main__":
    test_queries = [
        "What is self-reflection in retrieval-augmented generation?",
        "How does query decomposition improve multi-hop retrieval?",
    ]
    for q in test_queries:
        search(q)
