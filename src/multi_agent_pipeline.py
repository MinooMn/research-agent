"""
Full multi-agent pipeline: planner -> retriever -> critic -> composer.

For a given question:
1. Planner decomposes it into up to 2 sub-questions.
2. Each sub-question is independently retrieved and answered (retriever agent).
3. Each sub-answer's cited claims are checked for faithfulness (critic agent).
4. The composer synthesizes a final answer from the sub-answers, weighted by
   their critic-verified faithfulness.
"""

from src.agents.planner import plan
from src.agents.retriever import retrieve_and_answer
from src.agents.critic import check_faithfulness
from src.agents.composer import compose
from src.chunk_store import load_chunks


def run_multi_agent(question: str) -> dict:
    """
    Returns:
    {
        "question": str,
        "sub_questions": list[str],
        "sub_results": [
            {"sub_question", "answer", "chunk_ids", "critic_result"}, ...
        ],
        "final_answer": str,
        "avg_faithfulness_score": float,
    }
    """
    sub_questions = plan(question)

    all_chunks = load_chunks()  # {chunk_id: chunk_text}, for critic lookups

    sub_results = []
    for sub_q in sub_questions:
        retriever_output = retrieve_and_answer(sub_q)
        # retriever_output: {"sub_question", "answer", "chunk_ids"}

        cited_chunks = {
            cid: all_chunks[cid]
            for cid in retriever_output["chunk_ids"]
            if cid in all_chunks
        }
        critic_result = check_faithfulness(retriever_output["answer"], cited_chunks)

        sub_results.append(
            {
                "sub_question": retriever_output["sub_question"],
                "answer": retriever_output["answer"],
                "chunk_ids": retriever_output["chunk_ids"],
                "critic_result": critic_result,
            }
        )

    composed = compose(question, sub_results)

    return {
        "question": question,
        "sub_questions": sub_questions,
        "sub_results": sub_results,
        "final_answer": composed["final_answer"],
        "avg_faithfulness_score": composed["avg_faithfulness_score"],
    }


if __name__ == "__main__":
    test_questions = [
        # single-hop control
        "What embedding model does the RAGentA paper use?",
        # real 2-hop from the gold set
        (
            "RAGentA's own paper describes its four-agent design as introducing "
            "substantial computational overhead. How does A-RAG's architecture, "
            "given its reported head-to-head results against RAGentA, address "
            "this overhead concern while still outperforming it?"
        ),
    ]

    for q in test_questions:
        print("=" * 70)
        print("QUESTION:", q)
        result = run_multi_agent(q)
        print("\nSub-questions:", result["sub_questions"])
        for sr in result["sub_results"]:
            print(
                f"\n  [{sr['critic_result']['faithfulness_score']:.2f}] {sr['sub_question']}"
            )
            print(f"    -> {sr['answer'][:200]}...")
        print("\nFINAL ANSWER:", result["final_answer"])
        print("AVG FAITHFULNESS:", result["avg_faithfulness_score"])
        print()
