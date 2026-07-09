import requests
from pypdf import PdfReader
import io

arxiv_ids = [
    "2506.16988",
    "2506.10844",
    "2602.03442",
    "2601.21912",
    "2606.22681",
    "2502.01142",
    "2404.00610",
    "2310.11511",
    "2501.14342",
    "2501.09136",
]


def fetch_paper(arxiv_id: str, out_dir: str = "data/raw"):
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(f"{out_dir}/{arxiv_id}.txt", "w", encoding="utf-8") as f:
        f.write(text)


for id in arxiv_ids:
    fetch_paper(id)
