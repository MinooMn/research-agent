import json
import re

from src.llm_client import generate
from src.chunk_store import load_chunks
from config import CRITIC_MODEL

CRITIC_PROMPT_TEMPLATE = """Does this source text support this claim?
If the claim contains multiple parts, and ANY part is not supported by the
source text, mark it as "unsupported" -- do not mark a claim as "supported"
just because part of it matches.
Answer supported / unsupported / contradicted, with one sentence of reasoning.

Your output MUST ONLY be a structured JSON like this:
{{"verdict": "supported" | "unsupported" | "contradicted", "reasoning": "..."}}

Source text: {source_text}
Claim: {claim}"""


def extract_claims(answer_text: str) -> list[dict]:
    """
    Split answer_text into (claim, chunk_id) pairs based on inline [chunk_id]
    citation markers. Each citation marker closes out the claim text that
    precedes it. Repeated citations of the same chunk_id produce separate
    entries, one per claim -- claims are NOT grouped/deduplicated by chunk_id.
    """
    claims = []
    start = 0
    for m in re.finditer(r"\[([\w.]+_\d+)\]", answer_text):
        claim_text = answer_text[start : m.start()].strip()
        if claim_text:
            claims.append({"claim": claim_text, "chunk_id": m.group(1)})
        start = m.end()
    return claims


def check_faithfulness(answer_text: str, cited_chunks: dict[str, str]) -> dict:
    """
    cited_chunks: {chunk_id: chunk_text}, keyed WITHOUT brackets -- must match
    the chunk_id format extracted by extract_claims().

    Returns:
    {
        "claims": [
            {"claim": ..., "chunk_id": ..., "verdict": ..., "reasoning": ...},
            ...
        ],
        "faithfulness_score": float
    }
    """
    extracted = extract_claims(answer_text)

    critic_results = []
    for item in extracted:
        chunk_id, claim = item["chunk_id"], item["claim"]

        if chunk_id not in cited_chunks:
            # Defensive: don't silently skip -- a missing source chunk is a
            # real data problem (mismatched chunk_id format, or the retriever
            # cited a chunk that wasn't actually passed in), and should be
            # visible rather than swallowed.
            critic_results.append(
                {
                    "claim": claim,
                    "chunk_id": chunk_id,
                    "verdict": "error",
                    "reasoning": f"No source text found for chunk_id '{chunk_id}' in cited_chunks.",
                }
            )
            continue

        raw_output = generate(
            model=CRITIC_MODEL,
            prompt=CRITIC_PROMPT_TEMPLATE.format(
                source_text=cited_chunks[chunk_id], claim=claim
            ),
        )

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            critic_results.append(
                {
                    "claim": claim,
                    "chunk_id": chunk_id,
                    "verdict": "error",
                    "reasoning": f"Critic returned unparseable output: {raw_output!r}",
                }
            )
            continue

        critic_results.append(
            {
                "claim": claim,
                "chunk_id": chunk_id,
                "verdict": parsed.get("verdict"),
                "reasoning": parsed.get("reasoning"),
            }
        )

    supported = sum(1 for r in critic_results if r["verdict"] == "supported")
    scoreable = [r for r in critic_results if r["verdict"] != "error"]
    faithfulness_score = supported / len(scoreable) if scoreable else 0.0

    return {"claims": critic_results, "faithfulness_score": faithfulness_score}


if __name__ == "__main__":
    answer_text = """A-RAG's hierarchical retrieval interface addresses the identified limitation of \
RAGentA by allowing the agent to make autonomous judgments and precisely read the most relevant \
content [2602.03442_13]. This design enables the model to selectively read only the most relevant \
chunks in full, avoiding the noise introduced by irrelevant content [2602.03442_13]. However, \
the context does not provide a direct comparison or explanation of how A-RAG's hierarchical retrieval \
interface specifically addresses the limitations of RAGentA, so more information would be needed \
to fully answer the question."""

    all_chunks = load_chunks()
    cited_chunks = {"2602.03442_13": all_chunks["2602.03442_13"]}

    result = check_faithfulness(answer_text, cited_chunks)
    for c in result["claims"]:
        print(f"[{c['verdict']}] {c['claim']}")
        print(f"   chunk_id: {c['chunk_id']}")
        print(f"   reasoning: {c['reasoning']}\n")
    print("Faithfulness score:", result["faithfulness_score"])
