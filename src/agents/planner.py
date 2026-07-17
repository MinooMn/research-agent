import json
import re
from src.llm_client import generate

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


def clean_and_parse_json(model_output: str) -> dict:
    # 1. Try to extract content inside ```json ... ``` or ``` ... ```
    markdown_regex = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(markdown_regex, model_output)

    if match:
        json_string = match.group(1).strip()
    else:
        # 2. Fallback: If no backticks, find the first '{' and last '}'
        # This strips out any leading/trailing conversational text
        start_idx = model_output.find("{")
        end_idx = model_output.rfind("}")

        if start_idx != -1 and end_idx != -1:
            json_string = model_output[start_idx : end_idx + 1].strip()
        else:
            json_string = model_output.strip()

    # 3. Parse the cleaned string
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # Handle cases where the JSON itself is structurally broken
        print(f"Failed to parse JSON: {e}")
        print(json_string)
        raise


def plan(question: str) -> list[str]:
    model_output = generate(PLANNER_PROMPT_TEMPLATE.format(question=question))
    output_json = clean_and_parse_json(model_output)
    sub_questions = output_json.get("sub-questions")
    if not sub_questions:
        print("No sub-questions found. Model output was: ", model_output)
        return
    return sub_questions


if __name__ == "__main__":
    question = "What retrieval strategy does ProRAG use, and how does its reported performance compare to DeepRAG's?"
    print("Question: ", question)
    print("Sub-questions: ", plan(question))
