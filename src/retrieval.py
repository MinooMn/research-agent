import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDER_MODEL, INDEX_PATH, CORPUS_CHUNKS_PATH

model = SentenceTransformer(EMBEDDER_MODEL)
index = faiss.read_index(INDEX_PATH)

# Load chunks (row order must match what was embedded/indexed)
chunks = []
with open(CORPUS_CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))


def search(query: str, top_k: int = 5) -> list[dict]:
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)

    result = []
    for rank, (row, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        chunk = chunks[row]
        chunk_data = {
            "rank": rank,
            "dist": f"{dist:.3f}",
            "chunk_id": chunk["chunk_id"],
            "chunk": chunk["chunk_text"],
        }
        result.append(chunk_data)

    return result
