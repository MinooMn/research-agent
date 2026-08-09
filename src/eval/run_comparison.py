"""
Runs all gold questions through both the baseline and multi-agent pipelines,
scores each for faithfulness and correctness, tracks latency, and logs results
to MLflow. Checkpointed to a JSONL file so a rate-limit failure partway through
doesn't lose completed work -- rerun the script and it skips already-done
(question_id, pipeline) pairs.
"""

import json
import time
import os

import mlflow

from src.llm_client import generate
from src.llm_json import clean_json
from src.chunk_store import load_chunks
from src.baseline.rag_pipeline import answer_question
from src.multi_agent_pipeline import run_multi_agent
from src.agents.critic import check_faithfulness

GOLD_PATH = "data/eval/gold_questions_verified.jsonl"
CHECKPOINT_PATH = "results/comparison_checkpoint.jsonl"
RESULTS_MD_PATH = "results/comparison_results.md"

CORRECTNESS_JUDGE_PROMPT = """You are grading whether a generated answer is
factually correct compared to a reference answer, for the same question.

Question: {question}

Reference answer: {reference_answer}

Generated answer: {generated_answer}

Does the generated answer convey the same key facts as the reference answer,
even if worded differently? Minor omissions of secondary detail are OK; the
core claim(s) must match. If the generated answer explicitly declines to
answer or says the evidence is insufficient, mark it as incorrect (it did not
successfully answer the question), not as an error.

Respond with ONLY a JSON object:
{{"correct": true|false, "reasoning": "..."}}
"""


def judge_correctness(
    question: str, reference_answer: str, generated_answer: str
) -> dict:
    prompt = CORRECTNESS_JUDGE_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
    )
    raw = generate(prompt)
    try:
        return clean_json(raw)
    except json.JSONDecodeError:
        return {"correct": None, "reasoning": f"Judge output unparseable: {raw!r}"}


def load_gold_questions() -> list[dict]:
    questions = []
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def load_checkpoint() -> dict:
    """Returns {(question_id, pipeline): result_dict} for already-completed runs."""
    done = {}
    if not os.path.exists(CHECKPOINT_PATH):
        return done
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            done[(entry["question_id"], entry["pipeline"])] = entry
    return done


def append_checkpoint(entry: dict):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def run_baseline_scored(q: dict, all_chunks: dict) -> dict:
    start = time.time()
    result = answer_question(q["question"])
    latency = time.time() - start

    cited_chunks = {
        cid: all_chunks[cid] for cid in result["chunk_ids"] if cid in all_chunks
    }
    faithfulness = check_faithfulness(result["answer"], cited_chunks)
    correctness = judge_correctness(
        q["question"], q["reference_answer"], result["answer"]
    )

    return {
        "question_id": q["id"],
        "question": q["question"],
        "category": q["category"],
        "pipeline": "baseline",
        "answer": result["answer"],
        "faithfulness_score": faithfulness["faithfulness_score"],
        "correct": correctness.get("correct"),
        "correctness_reasoning": correctness.get("reasoning"),
        "latency_seconds": latency,
    }


def run_multi_agent_scored(q: dict) -> dict:
    start = time.time()
    result = run_multi_agent(q["question"])
    latency = time.time() - start

    correctness = judge_correctness(
        q["question"], q["reference_answer"], result["final_answer"]
    )

    return {
        "question_id": q["id"],
        "question": q["question"],
        "category": q["category"],
        "pipeline": "multi_agent",
        "answer": result["final_answer"],
        "faithfulness_score": result["avg_faithfulness_score"],
        "correct": correctness.get("correct"),
        "correctness_reasoning": correctness.get("reasoning"),
        "latency_seconds": latency,
        "trace": {
            "sub_questions": result["sub_questions"],
            "sub_results": result[
                "sub_results"
            ],  # includes per-sub-answer chunk_ids + critic claims
        },
    }


def main():
    questions = load_gold_questions()
    all_chunks = load_chunks()
    done = load_checkpoint()

    mlflow.set_experiment("baseline_vs_multiagent")

    with mlflow.start_run():
        for q in questions:
            for pipeline_name, run_fn in [
                ("baseline", lambda: run_baseline_scored(q, all_chunks)),
                ("multi_agent", lambda: run_multi_agent_scored(q)),
            ]:
                key = (q["id"], pipeline_name)
                if key in done:
                    print(f"SKIP (already done): {q['id']} / {pipeline_name}")
                    continue

                print(f"Running: {q['id']} / {pipeline_name} ...")
                try:
                    entry = run_fn()
                except Exception as e:
                    print(f"  FAILED: {e}")
                    entry = {
                        "question_id": q["id"],
                        "question": q["question"],
                        "category": q["category"],
                        "pipeline": pipeline_name,
                        "answer": None,
                        "faithfulness_score": None,
                        "correct": None,
                        "correctness_reasoning": f"Pipeline error: {e}",
                        "latency_seconds": None,
                    }

                append_checkpoint(entry)
                done[key] = entry

                mlflow.log_metric(
                    f"{pipeline_name}_faithfulness_{q['id']}",
                    entry["faithfulness_score"] or 0.0,
                )
                mlflow.log_metric(
                    f"{pipeline_name}_latency_{q['id']}",
                    entry["latency_seconds"] or 0.0,
                )
                mlflow.log_dict(entry, f"traces/{q['id']}_{pipeline_name}.json")

        write_results_table(done, questions)


def write_results_table(done: dict, questions: list[dict]):
    os.makedirs("results", exist_ok=True)

    categories = ["single_hop", "independent_2hop", "dependent_2hop"]
    lines = ["# Baseline vs Multi-Agent Comparison Results\n"]

    # Per-question table
    lines.append("## Per-question results\n")
    lines.append("| ID | Category | Pipeline | Faithfulness | Correct | Latency (s) |")
    lines.append("|---|---|---|---|---|---|")
    for q in questions:
        for pipeline_name in ["baseline", "multi_agent"]:
            entry = done.get((q["id"], pipeline_name))
            if not entry:
                continue
            f_score = entry["faithfulness_score"]
            f_str = f"{f_score:.2f}" if f_score is not None else "ERROR"
            correct_str = (
                str(entry["correct"]) if entry["correct"] is not None else "ERROR"
            )
            lat_str = (
                f"{entry['latency_seconds']:.1f}"
                if entry["latency_seconds"] is not None
                else "-"
            )
            lines.append(
                f"| {q['id']} | {q['category']} | {pipeline_name} | {f_str} | {correct_str} | {lat_str} |"
            )

    # Aggregate by category and pipeline
    lines.append("\n## Aggregate results by category\n")
    lines.append(
        "| Category | Pipeline | Avg Faithfulness | Accuracy | Avg Latency (s) | N |"
    )
    lines.append("|---|---|---|---|---|---|")
    for cat in categories:
        for pipeline_name in ["baseline", "multi_agent"]:
            entries = [
                done[(q["id"], pipeline_name)]
                for q in questions
                if q["category"] == cat
                and (q["id"], pipeline_name) in done
                and done[(q["id"], pipeline_name)]["faithfulness_score"] is not None
            ]
            if not entries:
                continue
            avg_f = sum(e["faithfulness_score"] for e in entries) / len(entries)
            correct_entries = [e for e in entries if e["correct"] is not None]
            accuracy = (
                sum(1 for e in correct_entries if e["correct"]) / len(correct_entries)
                if correct_entries
                else 0.0
            )
            avg_lat = sum(e["latency_seconds"] for e in entries) / len(entries)
            lines.append(
                f"| {cat} | {pipeline_name} | {avg_f:.2f} | {accuracy:.2f} | {avg_lat:.1f} | {len(entries)} |"
            )

    # Overall
    lines.append("\n## Overall\n")
    lines.append("| Pipeline | Avg Faithfulness | Accuracy | Avg Latency (s) | N |")
    lines.append("|---|---|---|---|---|")
    for pipeline_name in ["baseline", "multi_agent"]:
        entries = [
            done[(q["id"], pipeline_name)]
            for q in questions
            if (q["id"], pipeline_name) in done
            and done[(q["id"], pipeline_name)]["faithfulness_score"] is not None
        ]
        if not entries:
            continue
        avg_f = sum(e["faithfulness_score"] for e in entries) / len(entries)
        correct_entries = [e for e in entries if e["correct"] is not None]
        accuracy = (
            sum(1 for e in correct_entries if e["correct"]) / len(correct_entries)
            if correct_entries
            else 0.0
        )
        avg_lat = sum(e["latency_seconds"] for e in entries) / len(entries)
        lines.append(
            f"| {pipeline_name} | {avg_f:.2f} | {accuracy:.2f} | {avg_lat:.1f} | {len(entries)} |"
        )

    with open(RESULTS_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nResults table written to {RESULTS_MD_PATH}")


if __name__ == "__main__":
    main()
