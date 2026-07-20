import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Copy .env.example to .env and add your key."
    )

# Default settings
GROQ_MODEL_DEFAULT = "llama-3.3-70b-versatile"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 5
CRITIC_MODEL = "llama-3.1-8b-instant"

# Paths
INDEX_PATH = "data/index/faiss.index"
CORPUS_CHUNKS_PATH = "data/chunked_corpus/chunks.jsonl"
