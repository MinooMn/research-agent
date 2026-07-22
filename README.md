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

🚧 Work in progress. Corpus ingestion, chunking, retrieval, baseline single-shot RAG,
and planner/retriever/critic agents are complete. Currently building the gold
evaluation question set (M4) — draft complete, verification against full papers
next, followed by the baseline-vs-multi-agent comparison run.

## Roadmap

- [x] M1: Corpus ingestion + baseline single-shot RAG
- [x] M2: Planner + retriever agents
- [x] M3: Critic agent + faithfulness scoring
- [ ] M4: Gold evaluation set + comparison harness _(in progress — questions drafted, verification next)_
- [ ] M5: CI, Streamlit demo, final polish

## Pipeline

**Ingestion:**

1. `src/ingestion/fetch_corpus.py` — pulls full-text PDFs for the 10 papers above via
   the arXiv PDF endpoint, extracts text (`pypdf`), saves to `data/raw/{arxiv_id}.txt`.
2. `src/ingestion/chunk_corpus.py` — splits each paper into overlapping chunks
   (chunk size ~300 words, overlap ~50 words, sliding window) and writes them to
   `data/chunked_corpus/chunks.jsonl`. Currently 448 chunks across the 10-paper corpus.
3. `src/ingestion/embed_and_index.py` — embeds each chunk with
   `sentence-transformers/all-MiniLM-L6-v2` and builds a FAISS `IndexFlatL2` index.
4. `src/ingestion/sanity_check_retrieval.py` — manual sanity check confirming
   retrieval surfaces the correct source paper/section for test queries.

**Baseline RAG (single-shot):**

- `src/retrieval.py` / `src/chunk_store.py` — shared retrieval and chunk-loading
  modules used across the whole pipeline (single source of truth for reading
  `chunks.jsonl`, in both ordered-list and id-keyed dict forms).
- `src/llm_client.py` — thin, generic wrapper around the Groq API
  (`llama-3.3-70b-versatile` by default).
- `src/baseline/rag_pipeline.py` — retrieve → grounding-instructed prompt →
  generate → return `{query, answer, chunk_ids}`.
- See `notes/day4_baseline_observations.md`: two distinct baseline failure modes
  identified — **retrieval coverage failure** and **synthesis failure**.

**Planner + Retriever + Critic agents (M2/M3):**

- `src/agents/planner.py` — decomposes a question into up to 2 sub-questions,
  with few-shot examples preventing over-decomposition of single-hop questions.
- `src/agents/retriever.py` — per-sub-question grounded answering with inline
  `[chunk_id]` citations.
- `src/agents/critic.py` — claim-level faithfulness judging (`llama-3.1-8b-instant`),
  splitting answers into individual cited claims and verifying each against its
  source chunk. Validated against deliberately fabricated claims (Day 8) — correctly
  scores fabricated claims as unfaithful, though it doesn't yet reliably distinguish
  "contradicted" from "unsupported" as separate categories.
- See `notes/m2_agent_observations.md` and `notes/m3_critic_observations.md` for
  detailed findings, including a known critic limitation with compound-claim
  over-generosity.

**Gold evaluation question set (M4, in progress):**

- `src/corpus_metadata.py` — canonical single source of truth for paper IDs/titles,
  used by both ingestion and evaluation code (previously duplicated).
- `src/llm_json.py` — shared LLM JSON-output parser (markdown-fence stripping,
  brace-boundary fallback), used by the planner and question drafter.
- `src/eval/draft_gold_questions.py` — drafts candidate gold questions across three
  categories: single_hop (10), independent_2hop (5), dependent_2hop (2). Grounded in
  real chunk excerpts (not paper titles alone — an initial ungrounded version
  fabricated statistics not present in the corpus). Dependent-hop questions required
  hand-authoring rather than LLM drafting: even given correct dual-source excerpts,
  the drafting model consistently defaulted to single-source framing rather than
  genuine cross-paper synthesis — the same "synthesis failure" pattern observed
  independently in the baseline (Day 4) and retriever agent (M2). See
  `notes/m4_question_drafting_observations.md` for the full account.
- Currently 17 draft questions; category counts to be expanded (target ~25-30) and
  all reference answers verified against full paper text before use in the
  comparison harness.

**Known limitations (to revisit in later milestones):**

- Retrieval sometimes surfaces bibliography/reference-list chunks. Not yet filtered.
- Retrieval is pure vector similarity (FAISS), no keyword/exact-match component.
- Chunking uses whitespace word counts as a token-count proxy, not the embedding
  model's actual tokenizer.
- Sub-questions in the retriever agent are answered independently, not
  sequentially — weakens performance on true dependent multi-hop questions.
- The critic can be overly generous on compound sentences, and doesn't reliably
  distinguish "contradicted" from "unsupported."
- **LLM synthesis across multiple sources is unreliable even when given correct,
  relevant material from both sources** — observed independently in the baseline
  pipeline, the retriever agent, and the question-drafting process. This is
  arguably the central empirical finding motivating this whole project, and will
  be the main axis of the M4 baseline-vs-multi-agent comparison.

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
5. Test the baseline RAG pipeline:
   ```
   python src/baseline/rag_pipeline.py
   ```
6. Test the planner + retriever + critic agents:
   ```
   python src/agents/planner.py
   python src/agents/retriever.py
   python src/agents/critic.py
   ```

## Tech Stack

- Python 3.11+
- `sentence-transformers` (embeddings, local/free)
- FAISS (vector index, local/free)
- Groq API (LLM generation — free tier, `llama-3.3-70b-versatile` / `llama-3.1-8b-instant`)
- MLflow (experiment tracking — planned, M4)
- Streamlit (demo UI — planned, M5)
