import requests
import io
import os
import time
from pypdf import PdfReader
from src.corpus_metadata import ARXIV_IDS


def fetch_paper(arxiv_id: str, out_dir: str = "data/raw"):
    os.makedirs(out_dir, exist_ok=True)
    try:
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        reader = PdfReader(io.BytesIO(resp.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        with open(f"{out_dir}/{arxiv_id}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"FAILD {arxiv_id}: e")
        return False


for id in ARXIV_IDS:
    fetch_paper(id)
    time.sleep(1)
