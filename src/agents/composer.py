from src.llm_client import generate

COMPOSE_PROMPT_TEMPLATE = """You are answering the following question by combining
verified sub-answers gathered from a research corpus.

Original question: {original_question}

Sub-answers (each labeled with its verification status):
{annotated_subanswers}

Instructions:
- Prioritize claims labeled [VERIFIED]. Treat [UNVERIFIED -- LOW CONFIDENCE] claims
  with caution -- you may still use them if no better information is available, but
  do not state them as confidently as verified claims.
- Synthesize a single, coherent final answer to the ORIGINAL question above --
  do not just restate the sub-answers separately.
- If the verified evidence is insufficient to fully answer the original question,
  say so explicitly rather than filling the gap with unverified claims.

Final answer:"""


def annotate_sub_result(sub_result: dict) -> str:
    """
    sub_result: {sub_question, answer, chunk_ids, critic_result}
    critic_result is the dict returned by check_faithfulness(), i.e.
    {"claims": [...], "faithfulness_score": float}

    Returns a single formatted block: the sub-question, its answer, and a
    verification label based on the aggregate faithfulness_score of its claims.
    """
    critic_result = sub_result.get("critic_result") or {}
    score = critic_result.get("faithfulness_score", 0.0)

    if score >= 0.75:
        label = "[VERIFIED]"
    elif score > 0:
        label = "[PARTIALLY VERIFIED -- some claims unsupported]"
    else:
        label = "[UNVERIFIED -- LOW CONFIDENCE]"

    return (
        f"{label}\n"
        f"Sub-question: {sub_result['sub_question']}\n"
        f"Answer: {sub_result['answer']}"
    )


def compose(original_question: str, sub_results: list[dict]) -> dict:
    """
    sub_results: list of {sub_question, answer, chunk_ids, critic_result}

    Returns:
    {
        "final_answer": str,
        "sub_results": sub_results,  # passed through unchanged, for logging
        "avg_faithfulness_score": float  # average across sub_results' critic scores
    }
    """
    annotated_blocks = [annotate_sub_result(sr) for sr in sub_results]
    annotated_subanswers = "\n\n".join(annotated_blocks)

    prompt = COMPOSE_PROMPT_TEMPLATE.format(
        original_question=original_question,
        annotated_subanswers=annotated_subanswers,
    )
    final_answer = generate(prompt)

    scores = [
        (sr.get("critic_result") or {}).get("faithfulness_score", 0.0)
        for sr in sub_results
    ]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "final_answer": final_answer,
        "sub_results": sub_results,
        "avg_faithfulness_score": avg_score,
    }


if __name__ == "__main__":
    # Minimal manual test with hand-built sub_results (no live planner/retriever/
    # critic call needed here -- full pipeline integration test comes with
    # multi_agent_pipeline.py).
    fake_sub_results = [
        {
            "sub_question": "What limitation does RAGentA's own paper state about its four-agent design?",
            "answer": "RAGentA's four-agent design introduces substantial computational overhead [2506.16988_11].",
            "chunk_ids": ["2506.16988_11"],
            "critic_result": {"faithfulness_score": 1.0, "claims": []},
        },
        {
            "sub_question": "How does A-RAG's architecture compare in reported results?",
            "answer": "A-RAG (Full) outperforms RAGentA across benchmarks [2602.03442_10].",
            "chunk_ids": ["2602.03442_10"],
            "critic_result": {"faithfulness_score": 1.0, "claims": []},
        },
    ]

    original_question = (
        "RAGentA's own paper describes its four-agent design as introducing "
        "substantial computational overhead. How does A-RAG's architecture, "
        "given its reported head-to-head results against RAGentA, address this "
        "overhead concern while still outperforming it?"
    )

    result = compose(original_question, fake_sub_results)
    print("Final answer:", result["final_answer"])
    print("Avg faithfulness score:", result["avg_faithfulness_score"])
