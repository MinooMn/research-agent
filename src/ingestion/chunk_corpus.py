import os
import json

RAW_DATA_DIR = "data/raw"
CHUNKED_OUTPUT_PATH = "data/chunked_corpus/chunks.jsonl"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
STRIDE = CHUNK_SIZE - CHUNK_OVERLAP

papers = os.listdir(RAW_DATA_DIR)
with open(CHUNKED_OUTPUT_PATH, "w", encoding="utf-8") as out_f:
    for p in papers:
        # Load text
        p_path = os.path.join(RAW_DATA_DIR, p)
        with open(p_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Chunk
        tokens = text.split()
        for i in range(len(tokens)):
            chunk_start = i * STRIDE
            chunk_end = min(chunk_start + CHUNK_SIZE, len(tokens))
            chunk = tokens[chunk_start:chunk_end]
            chunk_text = " ".join(chunk)

            # Save chunk as JSON
            arxiv_id = os.path.splitext(p)[0]
            chunk_data = {
                "arxiv_id": arxiv_id,
                "chunk_index": i,
                "chunk_id": f"{arxiv_id}_{i}",
                "chunk_text": chunk_text,
            }

            out_f.write(json.dumps(chunk_data) + "\n")

            if chunk_end == len(tokens):
                break
