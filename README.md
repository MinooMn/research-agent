# Multi-Agent Research Analyst: Planner–Retriever–Critic RAG

## Problem

Plain single-shot RAG (retrieve top-k chunks, generate an answer) struggles with
multi-hop questions that require combining evidence from multiple sources, and
offers no built-in signal for whether an answer is actually grounded in retrieved
evidence versus hallucinated. This project implements and empirically evaluates a
multi-agent pipeline — planner, retriever, critic, composer — over a corpus of
recent (2023–2026) papers on agentic RAG, measuring whether decomposition and
self-verification improve answer faithfulness over a naive RAG baseline.

This is not a novel architecture — it reimplements patterns published in Self-RAG,
RQ-RAG, and related work (see corpus below) — the contribution here is a controlled,
measured comparison against a baseline, not a new method.

## Corpus

10 papers on agentic/multi-agent RAG (2023–2026), chosen so that genuine cross-paper
references exist (e.g. later papers explicitly benchmark against earlier ones),
enabling real multi-hop evaluation questions rather than artificial ones.

| Paper                                                            | arXiv ID   |
| ---------------------------------------------------------------- | ---------- |
| RAGentA: Multi-Agent RAG for faithful, attributed answers        | 2506.16988 |
| mRAG: planner/searcher/reasoner multi-agent framework            | 2506.10844 |
| A-RAG: Scaling Agentic RAG via Hierarchical Retrieval Interfaces | 2602.03442 |
| ProRAG: Process-Supervised RL for RAG                            | 2601.21912 |
| Grounded Delta Planning: Efficient Multi-step RAG                | 2606.22681 |
| DeepRAG: Thinking to Retrieve Step by Step                       | 2502.01142 |
| RQ-RAG: Learning to Refine Queries for RAG                       | 2404.00610 |
| Self-RAG: Learning to Retrieve, Generate, and Critique           | 2310.11511 |
| Chain-of-Retrieval Augmented Generation                          | 2501.14342 |
| Agentic Retrieval-Augmented Generation: A Survey                 | 2501.09136 |

Full text (not just abstracts) is fetched via the arXiv PDF endpoint and stored as
plain text in `data/raw/`, to support fine-grained chunk-level retrieval rather than
whole-paper matching.

## Status

🚧 Work in progress. Corpus ingestion, chunking, retrieval, baseline RAG, and the
full multi-agent pipeline (planner/retriever/critic/composer) are complete and
integration-tested. The baseline-vs-multi-agent comparison harness is built and
checkpointed; running the full 24-question gold set is in progress. Next: add
intermediate-trace logging (sub-questions, retrieved chunks, critic verdicts) so
individual results are debuggable, then finish the full run and write up results.

## Roadmap

- [x] M1: Corpus ingestion + baseline single-shot RAG
- [x] M2: Planner + retriever agents
- [x] M3: Critic agent + faithfulness scoring
- [ ] M4: Gold evaluation set + comparison harness _(in progress — gold set verified,
      full pipeline wired, comparison harness built and running)_
- [ ] M5: CI, Streamlit demo, final polish

## Pipeline

**Ingestion:**

1. `src/ingestion/fetch_corpus.py` — pulls full-text PDFs via the arXiv PDF
   endpoint, extracts text (`pypdf`), saves to `data/raw/{arxiv_id}.txt`.
2. `src/ingestion/chunk_corpus.py` — overlapping chunks (~300 words, ~50 overlap),
   written to `data/chunked_corpus/chunks.jsonl`. 448 chunks across the corpus.
3. `src/ingestion/embed_and_index.py` — embeds with
   `sentence-transformers/all-MiniLM-L6-v2`, builds a FAISS `IndexFlatL2` index.
4. `src/ingestion/sanity_check_retrieval.py` — manual retrieval sanity check.

**Baseline RAG (single-shot):**

- `src/retrieval.py` / `src/chunk_store.py` — shared retrieval and chunk-loading
  modules (single source of truth for reading `chunks.jsonl`).
- `src/llm_client.py` — generic Groq API wrapper (`llama-3.3-70b-versatile`
  default), with retry/backoff on rate limits.
- `src/baseline/rag_pipeline.py` — retrieve → grounding-instructed prompt with
  inline `[chunk_id]` citations → generate → return `{query, answer, chunk_ids}`.
  Citation format matches the retriever agent's, so both pipelines can be scored
  with the identical faithfulness metric (critical fix — see Known Limitations).

**Planner + Retriever + Critic + Composer agents (M2/M3):**

- `src/agents/planner.py` — decomposes a question into up to 2 sub-questions.
- `src/agents/retriever.py` — per-sub-question grounded answering with inline
  `[chunk_id]` citations.
- `src/agents/critic.py` — claim-level faithfulness judging
  (`llama-3.1-8b-instant`). Validated against deliberately fabricated claims.
- `src/agents/composer.py` — synthesizes a final answer from sub-answers,
  weighting by critic-verified faithfulness; explicitly flags when verified
  evidence is insufficient rather than filling gaps with unverified claims.
- `src/multi_agent_pipeline.py` — wires all four agents into one callable
  pipeline (`run_multi_agent()`), integration-tested end-to-end.
- See `notes/m2_agent_observations.md`, `notes/m3_critic_observations.md`,
  `notes/m4_pipeline_integration_observations.md` for detailed findings,
  including a traced real-data example of the critic/composer correctly
  refusing to state a claim traced to a misattributed source chunk.

**Gold evaluation set (M4):**

- `src/corpus_metadata.py`, `src/llm_json.py` — shared canonical corpus metadata
  and LLM JSON-output parsing, used across ingestion and evaluation code.
- `src/eval/draft_gold_questions.py` — drafts candidate gold questions grounded in
  real chunk excerpts (an earlier title-only version fabricated statistics —
  see `notes/m4_question_drafting_observations.md`).
- `data/eval/gold_questions_verified.jsonl` — 24 questions across three categories
  (10 single_hop, 12 independent_2hop, 2 dependent_2hop), each manually verified
  against real source chunks, with corrected `source_chunk_ids`.
- `src/eval/run_comparison.py` — runs all gold questions through both pipelines,
  scores faithfulness (via the shared critic) and correctness (LLM-judge against
  the reference answer), tracks latency, checkpoints to
  `results/comparison_checkpoint.jsonl` (resumable across rate-limit failures),
  and logs metrics to MLflow. Produces `results/comparison_results.md`.

**Known limitations (to revisit in later milestones):**

- Retrieval sometimes surfaces bibliography/reference-list chunks.
- Retrieval is pure vector similarity (FAISS), no keyword/exact-match component.
- Chunking uses whitespace word counts as a token-count proxy, not the embedding
  model's actual tokenizer.
- Sub-questions in the retriever agent are answered independently, not
  sequentially — weakens performance on true dependent multi-hop questions. The
  planner itself was also found to sometimes decompose on a question's surface
  structure rather than preserving the specific dependency between its parts.
- The critic can be overly generous on compound sentences, and doesn't reliably
  distinguish "contradicted" from "unsupported."
- LLM synthesis across multiple sources is unreliable even when given correct,
  relevant material from both sources — observed independently in the baseline
  pipeline, the retriever agent, and the question-drafting process. This is the
  central empirical finding motivating this project.
- **A real retrieval miss was traced during pipeline integration testing**: a
  question about RAGentA's embedding model retrieved a topically-similar but
  wrong-paper chunk instead of the chunk containing the correct answer (which
  does exist in the corpus). The critic/composer correctly refused to state the
  wrong answer with confidence, but the deeper retrieval-quality issue remains
  unaddressed. See `notes/m4_pipeline_integration_observations.md`.
- **Faithfulness scoring initially could not be fairly compared across
  pipelines**: the baseline's original prompt didn't require inline chunk_id
  citations, so the critic (which extracts claims from citation markers) scored
  it 0.0 by construction, not because it was actually unfaithful. Fixed by
  aligning the baseline's citation format with the retriever agent's.
- Early comparison runs show faithfulness and correctness can diverge sharply —
  e.g. a multi-agent answer scored fully faithful (every claim individually
  grounded) while still reaching an incorrect conclusion, because the composed
  synthesis of two true sub-claims was itself wrong. This is treated as an
  important, intentional finding, not noise: faithfulness and correctness are
  measuring genuinely different things.

## Setup

1. Clone the repo and `cd` into it.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your [Groq API key](https://console.groq.com)
   (free, no credit card required):
   ```
   cp .env.example .env
   ```
4. The corpus, chunks, and FAISS index are already committed under `data/` for
   reproducibility — no need to re-run ingestion unless rebuilding from scratch.
5. Test the baseline pipeline, agents, and full multi-agent pipeline:
   ```
   python src/baseline/rag_pipeline.py
   python -m src.multi_agent_pipeline
   ```
6. Run the full comparison (resumable — safe to rerun after a rate-limit failure):
   ```
   python -m src.eval.run_comparison
   ```
   View tracked results with `mlflow ui` from the repo root.

## Tech Stack

- Python 3.11+
- `sentence-transformers` (embeddings, local/free)
- FAISS (vector index, local/free)
- Groq API (LLM generation — free tier, `llama-3.3-70b-versatile` / `llama-3.1-8b-instant`)
- MLflow (experiment tracking, local)
- Streamlit (demo UI — planned, M5)
