import json

from src.llm_client import generate
from src.llm_json import clean_json
from src.corpus_metadata import PAPER_TITLES, ARXIV_IDS
from src.chunk_store import load_chunks_ordered

OUTPUT_PATH = "data/eval/gold_questions_draft.jsonl"

# Hand-picked chunk indices per paper: 2 intro/abstract chunks + 1
# results-oriented chunk (selected by keyword scoring + manual inspection
# confirming genuine results content, not reference-list contamination).
# Used for single_hop and independent_2hop drafting.
SEED_CHUNKS = {
    "2310.11511": [0, 1, 21],
    "2404.00610": [0, 1, 10],
    "2501.09136": [0, 1, 42],
    "2501.14342": [0, 1, 34],
    "2502.01142": [0, 1, 33],
    "2506.10844": [0, 1, 11],
    "2506.16988": [0, 1, 11],
    "2601.21912": [0, 1, 18],
    "2602.03442": [0, 1, 11],
    "2606.22681": [0, 1, 34],
}

INDEPENDENT_2HOP_PAIRS = [
    ("2602.03442", "2601.21912"),
    ("2502.01142", "2404.00610"),
    ("2601.21912", "2502.01142"),
    ("2506.16988", "2506.10844"),
    ("2606.22681", "2501.14342"),
]

# For dependent_2hop only: hand-verified chunks where one paper specifically
# names and critiques/compares against the other -- generic SEED_CHUNKS did
# not contain this. Overrides the default seed lookup on a per-paper, per-pair basis.
DEPENDENT_SEED_OVERRIDES = {
    ("2602.03442", "2506.16988"): {
        "2602.03442": [10],  # A-RAG vs RAGentA head-to-head results table
        "2506.16988": [11],  # RAGentA's own stated overhead limitation
    },
    ("2404.00610", "2310.11511"): {
        "2404.00610": [11],  # RQ-RAG explicitly comparing to Self-RAG
        "2310.11511": [0, 1],  # Self-RAG's own stated approach, for context
    },
}
DEPENDENT_2HOP_PAIRS = list(DEPENDENT_SEED_OVERRIDES.keys())


def get_excerpt(
    arxiv_id: str, all_chunks: list[dict], indices: list[int] = None
) -> str:
    """
    Concatenate chunks for one paper into a single excerpt block.
    If `indices` is given, use those chunk_index values instead of SEED_CHUNKS.
    """
    wanted_indices = set(indices) if indices is not None else set(SEED_CHUNKS[arxiv_id])
    matches = [
        c
        for c in all_chunks
        if c["arxiv_id"] == arxiv_id and c["chunk_index"] in wanted_indices
    ]
    matches.sort(key=lambda c: c["chunk_index"])
    return "\n\n".join(c["chunk_text"] for c in matches)


DRAFT_PROMPT_TEMPLATE = """You are helping build a gold evaluation set for a RAG system testing corpus.

{instruction}

Base your question and reference_answer STRICTLY on the excerpt(s) below.
Do NOT use outside knowledge about these methods, and do NOT invent numbers,
benchmark names, or claims that are not explicitly present in the excerpts.
If the excerpts don't contain enough material for a good question of this
type, respond with {{"question": null, "reference_answer": null}} instead of
guessing.

{excerpts}

Respond with ONLY a JSON object, no other text:
{{"question": "...", "reference_answer": "..."}}

The reference_answer should be a brief (1-3 sentence) factual answer drawn
directly from the excerpt(s) above.
"""

SINGLE_HOP_INSTRUCTION = """Draft ONE single-hop factual question about the paper
"{title}" (arXiv {arxiv_id}), answerable entirely from the excerpt below --
e.g. about its method name, a specific reported metric, or its core mechanism.
Do not reference any other paper."""

INDEPENDENT_2HOP_INSTRUCTION = """Draft ONE question comparing "{title_a}"
(arXiv {id_a}) and "{title_b}" (arXiv {id_b}), using ONLY the excerpts below.
The question must require combining a fact from EACH paper's excerpt
independently -- e.g. comparing reported performance or retrieval strategy.
Each half should be answerable on its own from its respective excerpt,
without needing the other paper's answer first."""

DEPENDENT_2HOP_INSTRUCTION = """Draft ONE question about "{title_a}" (arXiv
{id_a}) and "{title_b}" (arXiv {id_b}), using ONLY the excerpts below. The
excerpts were specifically chosen because one paper names and critiques or
compares against the other -- use that specific relationship. The question
must require the SPECIFIC answer to the first part in order to answer the
second part -- e.g. "what limitation does the excerpt show paper A has (or
that paper B claims about paper A), and how does paper B's specific mechanism
address THAT limitation." Only draft this if the excerpts actually support a
real, specific dependency; otherwise return nulls as instructed above."""


def draft_single_hop(arxiv_id: str, qid: str, all_chunks: list[dict]) -> dict | None:
    excerpt = get_excerpt(arxiv_id, all_chunks)
    instruction = SINGLE_HOP_INSTRUCTION.format(
        title=PAPER_TITLES[arxiv_id], arxiv_id=arxiv_id
    )
    excerpts_block = f"Excerpt from {PAPER_TITLES[arxiv_id]}:\n{excerpt}"
    raw = generate(
        DRAFT_PROMPT_TEMPLATE.format(instruction=instruction, excerpts=excerpts_block)
    )
    parsed = clean_json(raw)
    if not parsed.get("question"):
        return None
    return {
        "id": qid,
        "category": "single_hop",
        "question": parsed["question"],
        "reference_answer": parsed["reference_answer"],
        "source_chunk_ids": [],
        "source_papers": [arxiv_id],
    }


def draft_pair(
    id_a: str,
    id_b: str,
    qid: str,
    category: str,
    instruction_template: str,
    all_chunks: list[dict],
    seed_override: dict[str, list[int]] = None,
) -> dict | None:
    indices_a = seed_override[id_a] if seed_override else None
    indices_b = seed_override[id_b] if seed_override else None
    excerpt_a = get_excerpt(id_a, all_chunks, indices_a)
    excerpt_b = get_excerpt(id_b, all_chunks, indices_b)

    instruction = instruction_template.format(
        title_a=PAPER_TITLES[id_a], id_a=id_a, title_b=PAPER_TITLES[id_b], id_b=id_b
    )
    excerpts_block = (
        f"Excerpt from {PAPER_TITLES[id_a]}:\n{excerpt_a}\n\n"
        f"Excerpt from {PAPER_TITLES[id_b]}:\n{excerpt_b}"
    )
    raw = generate(
        DRAFT_PROMPT_TEMPLATE.format(instruction=instruction, excerpts=excerpts_block)
    )
    parsed = clean_json(raw)
    if not parsed.get("question"):
        return None
    return {
        "id": qid,
        "category": category,
        "question": parsed["question"],
        "reference_answer": parsed["reference_answer"],
        "source_chunk_ids": [],
        "source_papers": [id_a, id_b],
    }


def main():
    all_chunks = load_chunks_ordered()
    questions = []
    counter = 1
    skipped = 0

    for arxiv_id in ARXIV_IDS:
        try:
            q = draft_single_hop(arxiv_id, f"sh_{counter:03d}", all_chunks)
            if q:
                questions.append(q)
                print(f"[single_hop] {q['question']}")
            else:
                skipped += 1
                print(f"[single_hop] SKIPPED (insufficient excerpt) -- {arxiv_id}")
        except Exception as e:
            print(f"FAILED single_hop for {arxiv_id}: {e}")
        counter += 1

    for id_a, id_b in INDEPENDENT_2HOP_PAIRS:
        try:
            q = draft_pair(
                id_a,
                id_b,
                f"i2h_{counter:03d}",
                "independent_2hop",
                INDEPENDENT_2HOP_INSTRUCTION,
                all_chunks,
            )
            if q:
                questions.append(q)
                print(f"[independent_2hop] {q['question']}")
            else:
                skipped += 1
                print(
                    f"[independent_2hop] SKIPPED (insufficient excerpt) -- {id_a}/{id_b}"
                )
        except Exception as e:
            print(f"FAILED independent_2hop for {id_a}/{id_b}: {e}")
        counter += 1

    for id_a, id_b in DEPENDENT_2HOP_PAIRS:
        try:
            q = draft_pair(
                id_a,
                id_b,
                f"d2h_{counter:03d}",
                "dependent_2hop",
                DEPENDENT_2HOP_INSTRUCTION,
                all_chunks,
                seed_override=DEPENDENT_SEED_OVERRIDES[(id_a, id_b)],
            )
            if q:
                questions.append(q)
                print(f"[dependent_2hop] {q['question']}")
            else:
                skipped += 1
                print(
                    f"[dependent_2hop] SKIPPED (insufficient excerpt) -- {id_a}/{id_b}"
                )
        except Exception as e:
            print(f"FAILED dependent_2hop for {id_a}/{id_b}: {e}")
        counter += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    by_cat = {}
    for q in questions:
        by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1

    print(
        f"\nWrote {len(questions)} draft questions to {OUTPUT_PATH} ({skipped} skipped)"
    )
    print("Category breakdown:", by_cat)


if __name__ == "__main__":
    main()
