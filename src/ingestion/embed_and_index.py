import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from config import EMBEDDER_MODEL, INDEX_PATH, CORPUS_CHUNKS_PATH

os.makedirs(INDEX_PATH, exist_ok=True)

model = SentenceTransformer(EMBEDDER_MODEL)

chunks = []
with open(CORPUS_CHUNKS_PATH) as f:
    for line in f:
        chunks.append(json.loads(line))

texts = [c["chunk_text"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings.astype(np.float32))

faiss.write_index(index, f"{INDEX_PATH}/faiss.index")
