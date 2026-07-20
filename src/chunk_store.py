import json
from config import CORPUS_CHUNKS_PATH


def load_chunks(path: str = CORPUS_CHUNKS_PATH) -> dict[str, str]:
    """Returns {chunk_id: chunk_text} for all chunks."""
    chunks = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c["chunk_text"]
    return chunks
