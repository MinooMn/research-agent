import json

from src.chunk_store import load_chunks, load_chunks_ordered


def _write_chunks_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_chunks_returns_dict_keyed_by_chunk_id(tmp_path):
    records = [
        {"chunk_id": "1234.56789_0", "chunk_text": "first chunk"},
        {"chunk_id": "1234.56789_1", "chunk_text": "second chunk"},
    ]
    path = tmp_path / "chunks.jsonl"
    _write_chunks_jsonl(path, records)

    chunks = load_chunks(str(path))

    assert chunks == {
        "1234.56789_0": "first chunk",
        "1234.56789_1": "second chunk",
    }


def test_load_chunks_ordered_preserves_file_order_and_full_records(tmp_path):
    records = [
        {"chunk_id": "1234.56789_1", "chunk_text": "second chunk"},
        {"chunk_id": "1234.56789_0", "chunk_text": "first chunk"},
    ]
    path = tmp_path / "chunks.jsonl"
    _write_chunks_jsonl(path, records)

    chunks = load_chunks_ordered(str(path))

    assert chunks == records
