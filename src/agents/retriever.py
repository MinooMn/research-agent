import re
from src.retrieval import search
from src.llm_client import generate
from src.agents.planner import plan
from config import RETRIEVAL_TOP_K

RETRIEVER_PROMPT_TEMPLATE = """Answer the sub-question using ONLY the retrieved context below.
Cite the chunk_id for every claim you make, in square brackets immediately after the claim,
like this: "X does Y [chunk_id]." If the context is insufficient, say so explicitly.

Sub-question: {sub_question}

Retrieved Context:
{context}

Answer (with inline [chunk_id] citations):"""


def extract_cited_chunk_ids(answer_text: str) -> list[str]:
    return re.findall(r"\[([\w.]+_\d+)\]", answer_text)


def retrieve_and_answer(sub_question: str) -> dict:
    retrieved = search(sub_question, top_k=RETRIEVAL_TOP_K)
    context = "\n\n".join(f"[{c['chunk_id']}] {c['chunk']}" for c in retrieved)
    prompt = RETRIEVER_PROMPT_TEMPLATE.format(
        sub_question=sub_question, context=context
    )

    answer = generate(prompt)
    cited_ids = extract_cited_chunk_ids(answer)
    # deduplicate chunk IDs
    cited_ids = list(dict.fromkeys(cited_ids))

    return {
        "sub_question": sub_question,
        "answer": answer,
        "chunk_ids": cited_ids if cited_ids else [c["chunk_id"] for c in retrieved],
    }


if __name__ == "__main__":
    question = "What limitation of RAGentA does A-RAG identify, and how does A-RAG's hierarchical retrieval interface address it?"
    sub_questions = plan(question)

    print("Question: ", question)
    print("Sub-questions: ", sub_questions)
    for q in sub_questions:
        print(retrieve_and_answer(q))
