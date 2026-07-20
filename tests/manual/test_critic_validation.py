"""
Day 8 validation: does the critic correctly distinguish
supported / unsupported / contradicted on deliberately hand-crafted cases?

Run this directly: python test_critic_validation.py
"""

from src.agents.critic import check_faithfulness
from src.chunk_store import load_chunks

all_chunks = load_chunks()

# --------------------------------------------------------------------------
# Case 1: CONTRADICTED -- a real chunk paired with a deliberately wrong number.
# Real reported number: F1 51.6 on PopQA.
# --------------------------------------------------------------------------
case_1_chunk_id = "2601.21912_22"
case_1_answer = (
    f"ProRAG achieves an F1 score of 85 on the PopQA dataset " f"[{case_1_chunk_id}]."
)
case_1_cited_chunks = {case_1_chunk_id: all_chunks[case_1_chunk_id]}

# --------------------------------------------------------------------------
# Case 2: UNSUPPORTED -- same real chunk, but a claim about a topic the
# chunk plausibly doesn't address at all.
# --------------------------------------------------------------------------
case_2_chunk_id = "2601.21912_22"
case_2_answer = (
    f"ProRAG was developed by a team at Stanford University in 2024 "
    f"[{case_2_chunk_id}]."
)
case_2_cited_chunks = {case_2_chunk_id: all_chunks[case_2_chunk_id]}

# --------------------------------------------------------------------------
# Case 3: SUPPORTED control -- reuse a known-good case from earlier testing.
# --------------------------------------------------------------------------
case_3_chunk_id = "2602.03442_13"
case_3_answer = (
    f"A-RAG's design enables the model to selectively read only the most "
    f"relevant chunks in full, avoiding the noise introduced by irrelevant "
    f"content [{case_3_chunk_id}]."
)
case_3_cited_chunks = {case_3_chunk_id: all_chunks[case_3_chunk_id]}


def run_case(label: str, answer: str, cited_chunks: dict):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"Answer text: {answer}")
    print(f"\nSource chunk text ({list(cited_chunks.keys())[0]}):")
    print(list(cited_chunks.values())[0][:400] + "...\n")

    result = check_faithfulness(answer, cited_chunks)
    for c in result["claims"]:
        print(f"VERDICT: [{c['verdict']}]")
        print(f"  claim: {c['claim']}")
        print(f"  reasoning: {c['reasoning']}")
    print(f"\nFaithfulness score: {result['faithfulness_score']}")


if __name__ == "__main__":
    run_case(
        "CASE 1: Expected CONTRADICTED (wrong number)",
        case_1_answer,
        case_1_cited_chunks,
    )
    run_case(
        "CASE 2: Expected UNSUPPORTED (off-topic claim)",
        case_2_answer,
        case_2_cited_chunks,
    )
    run_case(
        "CASE 3: Expected SUPPORTED (known-good control)",
        case_3_answer,
        case_3_cited_chunks,
    )
