# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Multi-agent RAG research project: implements and empirically compares a naive
single-shot RAG baseline against a planner→retriever→critic→composer multi-agent
pipeline, over a fixed corpus of 10 arXiv papers on agentic RAG. The goal is
measuring whether decomposition + self-verification actually improves answer
faithfulness/correctness — see `README.md` for the full problem statement, corpus
list, pipeline breakdown, and (important) the "Known limitations" section, which
documents real, already-diagnosed failure modes (retrieval misses, critic being
overly generous on compound claims, synthesis errors across sources, etc.). Read
that section before "fixing" behavior that may be an intentional empirical finding
rather than a bug.

Status: M1–M3 complete, M4 (gold eval + comparison harness) in progress. Milestone
progress and per-agent findings are logged in `notes/*.md` — check these before
re-investigating something that may already be documented there.

## Setup & commands

```
pip install -r requirements.txt
cp .env.example .env   # then add GROQ_API_KEY (free tier: https://console.groq.com)
```

There is no build step (pure Python) and no linter/formatter configured. There is
no pytest suite — `tests/manual/` scripts are standalone, run directly and read
manually rather than asserted on:

```
python tests/manual/test_critic_validation.py
```

Run individual pipelines/agents (each module has a `__main__` smoke test):

```
python src/baseline/rag_pipeline.py       # baseline single-shot RAG
python -m src.multi_agent_pipeline         # full planner/retriever/critic/composer pipeline
python src/agents/planner.py
python src/agents/retriever.py
python src/agents/critic.py
python src/agents/composer.py
```

Run the full baseline-vs-multi-agent comparison over the gold question set
(checkpointed/resumable — safe to rerun after a rate-limit failure, skips
already-completed `(question_id, pipeline)` pairs):

```
python -m src.eval.run_comparison
mlflow ui   # view logged faithfulness/correctness/latency metrics, from repo root
```

Note: `src/eval/run_comparison.py` currently hardcodes `questions = questions[:3]`
in `main()` (line ~141) — a temporary dev-time cap while M4 is finished, not the
full 24-question gold set. Remove/adjust this when actually running the complete
evaluation.

Rebuilding the corpus from scratch (not normally needed — `data/` is committed for
reproducibility):

```
python src/ingestion/fetch_corpus.py
python src/ingestion/chunk_corpus.py
python src/ingestion/embed_and_index.py
python src/ingestion/sanity_check_retrieval.py   # manual retrieval spot-check
```

## Architecture

**Module resolution**: code is run from the repo root as `python -m src....` or
`python src/.../file.py`, and imports use `src.`-prefixed absolute paths (e.g.
`from src.llm_client import generate`) alongside a root-level `from config import
...`. There's no package install — this only works when the working directory is
the repo root, which is also why `config.py` sits at the top level rather than
inside `src/`.

**Data flow**: `data/raw/{arxiv_id}.txt` (full paper text) → chunked into
`data/chunked_corpus/chunks.jsonl` (`{chunk_id, chunk_text}`, `chunk_id` format
`{arxiv_id}_{index}`) → embedded into `data/index/faiss.index` (row order matches
`chunks.jsonl` order — the two files must stay in sync). `src/chunk_store.py` and
`src/retrieval.py` are the single source of truth for reading chunks / querying the
index; other modules always go through them rather than touching the files
directly.

**Citation format is the load-bearing contract between components.** Every
generation prompt (baseline and retriever agent) instructs the model to cite
`[chunk_id]` inline after each claim. `src/agents/critic.py::extract_claims()`
parses answer text by splitting on those `[chunk_id]` markers — text before each
marker becomes one claim attributed to that chunk. This is why the baseline and
retriever-agent prompts must produce the same citation format: it's what lets one
critic (`check_faithfulness()`) score both pipelines identically. If you change a
prompt's citation instructions, check both call sites and `extract_claims()`'s
regex (`\[([\w.]+_\d+)\]`).

**Multi-agent pipeline** (`src/multi_agent_pipeline.py::run_multi_agent()`):
planner splits a question into ≤2 sub-questions → each sub-question is
independently retrieved+answered (retriever agent, own FAISS search + generation
call) → critic scores each sub-answer's claims for faithfulness against the actual
cited chunk text → composer synthesizes a final answer, weighting sub-answers by
their critic score (`>=0.75` "VERIFIED", `>0` "PARTIALLY VERIFIED", else
"UNVERIFIED — LOW CONFIDENCE") and is explicitly instructed to admit gaps rather
than fill them with unverified claims. Sub-questions are answered independently,
not sequentially — there's no mechanism for sub-question 2 to see sub-question 1's
answer, which is a known weak point for true dependent multi-hop questions (see
README "Known limitations").

**Two LLM roles, two models**: `src/llm_client.py::generate()` wraps the Groq API
(OpenAI-compatible client) with exponential-backoff retry on 429s.
`GROQ_MODEL_DEFAULT` (`llama-3.3-70b-versatile`, from `config.py`) is used for
planning/retrieval/composition/generation; `CRITIC_MODEL`
(`llama-3.1-8b-instant`) is used specifically for critic judgments — deliberately
a different, smaller model. `src/llm_json.py::clean_json()` is the shared
tolerant-JSON-extraction helper for parsing structured LLM output (strips markdown
code fences, falls back to first-`{`/last-`}` slicing); `critic.py` parses its own
JSON directly instead since its prompt is constrained to bare JSON output.

**Evaluation harness** (`src/eval/run_comparison.py`): runs every gold question
(`data/eval/gold_questions_verified.jsonl`) through both pipelines, scores
faithfulness (shared critic) and correctness (a separate LLM-judge prompt against
each question's `reference_answer` — an answer that declines/hedges is graded
incorrect, not errored), logs per-question and aggregate-by-category metrics to
MLflow, and writes `results/comparison_results.md`. Progress is checkpointed to
`results/comparison_checkpoint.jsonl` keyed by `(question_id, pipeline)` so a
rate-limit failure mid-run doesn't lose completed work — rerunning the script
resumes rather than restarts.

**Gold question categories** (`data/eval/gold_questions_verified.jsonl`):
`single_hop`, `independent_2hop` (two sub-answers looked up independently and
combined), `dependent_2hop` (sub-answer 2 depends on sub-answer 1's result — the
pipeline's weakest case, see above). Questions were drafted grounded in real chunk
excerpts (not paper titles alone — an earlier title-only approach fabricated
statistics, see `notes/m4_question_drafting_observations.md`) and manually
verified against source chunks.
