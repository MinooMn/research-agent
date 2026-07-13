from src.retrieval import search
from src.llm_client import generate
from config import RETRIEVAL_TOP_K

RAG_PROMPT_TEMPLATE = """Answer the question using ONLY the retrieved context below. \
If the context doesn't contain enough information to answer, say so explicitly.

Question: {query}

Retrieved Context:
{context}

Answer:"""


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
    result = answer_question("what is multi-hop question answering?")
    print("Answer:", result["answer"])
    print("Sources:", result["chunk_ids"])
