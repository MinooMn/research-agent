# Multi-Agent Research Analyst: Planner–Retriever–Critic RAG

## Live Demo

Try it: **[research-agent-comparison.streamlit.app](https://research-agent-comparison.streamlit.app)**

Ask a question against a fixed corpus of 10 arXiv papers on agentic RAG, and
compare a single-shot RAG baseline against a planner → retriever → critic →
composer multi-agent pipeline. Try a specific question about one of the papers
in the corpus (see below) rather than a generic question — the system is
deliberately grounded and will decline to answer questions the corpus doesn't
cover.

> Note: the app is hosted on Streamlit Community Cloud's free tier and sleeps
> after periods of inactivity. If you see a "Zzzz" wake-up screen, click
> "Yes, get this app back up!" — it takes ~30-60 seconds to cold-start.

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

> **Note:** Originally built on Llama 3.1/3.3 via Groq; migrated to
> GPT-OSS-20B/120B following Groq's model deprecation announcement. The results
> below and the M4 analysis notes reflect the previous (Llama) models.

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

Full text is fetched via the arXiv PDF endpoint and stored as
plain text in `data/raw/`, to support fine-grained chunk-level retrieval rather than
whole-paper matching.

## Results

Baseline and multi-agent pipelines were run against a 24-question gold evaluation
set (single-hop, independent 2-hop, and dependent 2-hop questions), scoring
faithfulness (claim-level, critic-verified) and correctness (LLM-judged against
reference answers) for each. Manual review covered some of the questions in
depth (see `notes/m4_final_comparison_analysis.md`); the full run and per-question
table are in `results/comparison_results.md`.

**Headline finding:** the multi-agent architecture did not show a clear
faithfulness or correctness advantage over the single-shot baseline on this
corpus and question set. On independent 2-hop questions, both pipelines
frequently retrieved the same top chunks and reached similar conclusions —
suggesting the planner's sub-questions often weren't diverging enough in wording
to meaningfully change what got retrieved, so decomposition wasn't adding the
targeting benefit the architecture is designed to provide.

A few things did work as intended and are worth calling out specifically:

- The critic/composer correctly refused to state an unverified claim in a
  traced case where retrieval surfaced a topically-similar but wrong-paper
  chunk — a real, concrete example of the faithfulness-checking mechanism doing
  its job (see `notes/m4_pipeline_integration_observations.md`).
- Faithfulness and correctness were observed to diverge meaningfully in at
  least one case — an answer scored fully faithful (every individual claim
  grounded) while still reaching an incorrect conclusion, because the
  _synthesis_ of two true sub-claims was itself wrong. This is a genuinely
  useful distinction the two-metric design was built to surface.

Several confounds limit how precisely this comparison should be read — a
citation-parsing bug in the critic (documented, not fixed), gold-set questions
whose answers can appear in multiple differently-worded locations per paper
(confusing the correctness judge), and composer language that sometimes hedges
more than the underlying evidence warrants. These are detailed in
`notes/m4_final_comparison_analysis.md`.

## Roadmap

- [x] M1: Corpus ingestion + baseline single-shot RAG
- [x] M2: Planner + retriever agents
- [x] M3: Critic agent + faithfulness scoring
- [x] M4: Gold evaluation set + comparison harness
- [x] M5: CI, Streamlit demo, final polish

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
  modules.
- `src/llm_client.py` — generic Groq API wrapper (`openai/gpt-oss-120b`
  default).
- `src/baseline/rag_pipeline.py` — retrieve → grounding-instructed prompt with
  inline `[chunk_id]` citations → generate.

**Planner + Retriever + Critic + Composer agents (M2/M3):**

- `src/agents/planner.py` — decomposes a question into up to 2 sub-questions.
- `src/agents/retriever.py` — per-sub-question grounded answering with inline
  `[chunk_id]` citations.
- `src/agents/critic.py` — claim-level faithfulness judging
  (`openai/gpt-oss-20b`). Validated against deliberately fabricated claims.
- `src/agents/composer.py` — synthesizes a final answer from sub-answers,
  weighting by critic-verified faithfulness; explicitly flags when verified
  evidence is insufficient rather than filling gaps with unverified claims.
- `src/multi_agent_pipeline.py` — wires all four agents into one callable
  pipeline (`run_multi_agent()`), integration-tested end-to-end.
- See `notes/m2_agent_observations.md`, `notes/m3_critic_observations.md`,
  `notes/m4_pipeline_integration_observations.md` for detailed findings.

**Gold evaluation set (M4):**

- `src/corpus_metadata.py`, `src/llm_json.py` — shared canonical corpus metadata
  and LLM JSON-output parsing, used across ingestion and evaluation code.
- `src/eval/draft_gold_questions.py` — drafts candidate gold questions grounded in
  real chunk excerpts. See `notes/m4_question_drafting_observations.md`.
- `data/eval/gold_questions_verified.jsonl` — 24 questions across three categories
  (10 single_hop, 12 independent_2hop, 2 dependent_2hop), each manually verified
  against real source chunks, with corrected `source_chunk_ids`.
- `src/eval/run_comparison.py` — runs gold questions through both pipelines,
  scores faithfulness (via the shared critic) and correctness (LLM-judge against
  the reference answer), tracks latency, checkpoints to
  `results/comparison_checkpoint.jsonl` (resumable across rate-limit failures),
  and logs metrics to MLflow. Produces `results/comparison_results.md`.
  See Results above and `notes/m4_final_comparison_analysis.md` for findings.

**Testing & CI (M5):**

- `tests/unit/` — unit tests for pure-logic functions (`extract_claims`,
  `clean_json`, chunk loading).
- `.github/workflows/ci.yml` — runs the test suite on every push/PR to `main`.

**Demo (M5):**

- `app/streamlit_app.py` — Streamlit UI for interactively querying either
  pipeline and comparing answers, citations, and faithfulness scores. Deployed
  to Streamlit Community Cloud (see Live Demo above).

**Known limitations:**

- Retrieval sometimes surfaces bibliography/reference-list chunks.
- Retrieval is pure vector similarity (FAISS), no keyword/exact-match component.
- Chunking uses whitespace word counts as a token-count proxy, not the embedding
  model's actual tokenizer.
- Sub-questions in the retriever agent are answered independently, not
  sequentially — weakens performance on true dependent multi-hop questions. The
  planner itself was also found to sometimes decompose on a question's surface
  structure rather than preserving the specific dependency between its parts.
- The critic can be overly generous on compound sentences, doesn't reliably
  distinguish "contradicted" from "unsupported," and has a known citation-parsing
  bug where back-to-back multi-citations can produce spurious empty "claims"
  that deflate faithfulness scores. Not fixed — see `notes/m4_final_comparison_analysis.md`.
- LLM synthesis across multiple sources is unreliable even when given correct,
  relevant material from both sources — observed independently in the baseline
  pipeline, the retriever agent, and the question-drafting process. This is the
  central empirical finding motivating this project.
- The correctness judge can penalize answers drawn from a different (but still
  valid) passage than the gold reference chunk, since some facts appear in
  multiple places per paper with different wording — not every "incorrect"
  label reflects an actual pipeline failure.
- The demo's LLM backbone was migrated (Groq deprecated the originally-used
  Llama models); the reported comparison results reflect the original models,
  not the current demo's models.

## Future Work

- Fix the critic's citation-parsing edge cases and rerun the full comparison.
- Build a chunk-overlap-aware correctness judge, rather than pure text
  comparison against a single gold reference chunk.
- Prompt the planner to diverge sub-question wording more explicitly from the
  original question, to test whether that changes retrieval targeting.
- Explore hybrid (vector + keyword) retrieval, given the field's broader 2026
  trend toward agentic, tool-driven search over pure vector RAG.

## Setup

1. Clone the repo and `cd` into it.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your [Groq API key](https://console.groq.com)
   (free):
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
7. Run the Streamlit app locally:
   ```
   streamlit run app/streamlit_app.py
   ```
8. Run tests:
   ```
   python -m pytest tests/unit -v
   ```

## Tech Stack

- Python 3.10+
- `sentence-transformers` (embeddings, local/free)
- FAISS (vector index, local/free)
- Groq API (LLM generation — free tier, `openai/gpt-oss-120b` / `openai/gpt-oss-20b`)
- MLflow (experiment tracking, local)
- Streamlit (demo UI, deployed to Streamlit Community Cloud)
- pytest + GitHub Actions (CI)
