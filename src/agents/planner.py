from src.llm_client import generate
from src.llm_json import clean_json

PLANNER_PROMPT_TEMPLATE = """
Your role is to decompose a question into sub-questions ONLY if answering it
requires combining information from separate, distinct sources. Most questions
do NOT need decomposition.

Rules:
- Maximum 2 sub-questions. Never produce more than 2.
- If the question can be answered by looking up a single fact or concept, return
  it UNCHANGED as a single sub-question. Do not split single-concept questions
  into "define term A" + "define term B" — that is NOT genuine decomposition.
- Only decompose when the question explicitly requires comparing, combining, or
  relating information across two distinct things (e.g. two papers, two methods,
  a cause and its effect in a different source).

Examples:
Question: "What embedding model does the RAGentA paper use?"
{{"sub-questions": ["What embedding model does the RAGentA paper use?"]}}

Question: "What is self-reflection in retrieval-augmented generation?"
{{"sub-questions": ["What is self-reflection in retrieval-augmented generation?"]}}

Question: "How does A-RAG's retrieval approach compare to RAGentA's, and which performs better?"
{{"sub-questions": ["What is A-RAG's retrieval approach and reported performance?", "What is RAGentA's retrieval approach and reported performance?"]}}

** Output format
You MUST ONLY return a structured JSON containing the sub-questions.
e.i., {{"sub-questions": ["...", "..."]}}

QUESTION: {question}
"""


def plan(question: str) -> list[str]:
    model_output = generate(PLANNER_PROMPT_TEMPLATE.format(question=question))
    output_json = clean_json(model_output)
    sub_questions = output_json.get("sub-questions")
    if not sub_questions:
        print("No sub-questions found. Model output was: ", model_output)
        return
    return sub_questions


if __name__ == "__main__":
    question = "What retrieval strategy does ProRAG use, and how does its reported performance compare to DeepRAG's?"
    print("Question: ", question)
    print("Sub-questions: ", plan(question))
