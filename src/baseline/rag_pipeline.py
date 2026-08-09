from src.retrieval import search
from src.llm_client import generate
from config import RETRIEVAL_TOP_K

RAG_PROMPT_TEMPLATE = """Answer the question using ONLY the retrieved context below. \
Cite the chunk_id for every claim you make, in square brackets immediately after \
the claim, like this: "X does Y [chunk_id]." \
If the context doesn't contain enough information to answer, say so explicitly.

Question: {query}

Retrieved Context:
{context}

Answer (with inline [chunk_id] citations):"""


def answer_question(query: str) -> dict:
    retrieved = search(query, top_k=RETRIEVAL_TOP_K)
    context = "\n\n".join(f"[{c['chunk_id']}] {c['chunk']}" for c in retrieved)
    prompt = RAG_PROMPT_TEMPLATE.format(query=query, context=context)

    answer = generate(prompt)

    return {
        "query": query,
        "answer": answer,
        "chunk_ids": [c["chunk_id"] for c in retrieved],
    }


if __name__ == "__main__":
    question = """
    Which of the papers in this corpus cite Self-RAG, and what do they each say its main limitation is?
    """
    result = answer_question(question)
    print("Question: ", question)
    print("Answer:", result["answer"])
    print("Sources:", result["chunk_ids"])
