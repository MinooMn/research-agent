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
and planner + retriever agents are complete. Next: critic agent + faithfulness
scoring (M3).

## Roadmap

- [x] M1: Corpus ingestion + baseline single-shot RAG
- [x] M2: Planner + retriever agents
- [ ] M3: Critic agent + faithfulness scoring
- [ ] M4: Gold evaluation set + comparison harness
- [ ] M5: CI, Streamlit demo, final polish

## Pipeline

**Ingestion:**

1. `src/ingestion/fetch_corpus.py` — pulls full-text PDFs for the 10 papers above via
   the arXiv PDF endpoint, extracts text (`pypdf`), saves to `data/raw/{arxiv_id}.txt`.
2. `src/ingestion/chunk_corpus.py` — splits each paper into overlapping chunks
   (chunk size ~300 words, overlap ~50 words, sliding window) and writes them to
   `data/chunked_corpus/chunks.jsonl`, one JSON object per chunk:
   `{arxiv_id, chunk_index, chunk_id, chunk_text}`. Currently 448 chunks across the
   10-paper corpus.
3. `src/ingestion/embed_and_index.py` — embeds each chunk with
   `sentence-transformers/all-MiniLM-L6-v2` and builds a FAISS `IndexFlatL2` index
   (exact search — sufficient at this corpus size). Index saved to
   `data/index/faiss.index`; row order corresponds directly to line order in
   `chunks.jsonl`.
4. `src/ingestion/sanity_check_retrieval.py` — manual sanity check confirming
   retrieval surfaces the correct source paper/section for test queries.

**Baseline RAG (single-shot):**

- `src/retrieval.py` — shared retrieval module: embeds a query, searches the FAISS
  index, returns top-k chunks with `chunk_id`, distance, and text.
- `src/llm_client.py` — thin, generic wrapper around the Groq API
  (`llama-3.3-70b-versatile` by default). No knowledge of retrieval/RAG — reused by
  the baseline pipeline and all agents.
- `src/baseline/rag_pipeline.py` — retrieve top-k chunks → grounding-instructed
  prompt (answer only from context, say so if insufficient) → generate → return
  `{query, answer, chunk_ids}`.
- See `notes/day4_baseline_observations.md` for manual multi-hop test results
  against the baseline: two distinct failure modes identified — **retrieval
  coverage failure** (top-k search skews toward one source even when a question
  spans two) and **synthesis failure** (relevant chunks from multiple sources are
  retrieved, but the model doesn't reliably combine them, defaulting to refusal).

**Planner + Retriever agents (M2):**

- `src/agents/planner.py` — decomposes a question into up to 2 sub-questions using
  the LLM, with a structured-JSON output contract (parsed defensively — strips
  markdown code fences before parsing). Prompt uses explicit few-shot examples
  showing both single-hop (unsplit) and 2-hop (decomposed) cases; an earlier
  version without worked examples over-decomposed simple single-concept questions,
  fixed by adding the examples (see `notes/m2_agent_observations.md`).
- `src/agents/retriever.py` — for each sub-question independently: retrieves top-k
  chunks, generates a grounded answer with **inline `[chunk_id]` citations per
  claim** (rather than a single citation list produced after the fact, so each
  claim is traceable to a specific source at generation time — this also gives the
  upcoming critic agent claim-level granularity to verify against). Cited chunk_ids
  are extracted via regex and deduplicated; falls back to all retrieved chunk_ids
  if the model doesn't cite inline.

**Known limitations (to revisit in later milestones):**

- Retrieval sometimes surfaces bibliography/reference-list chunks, since reference
  entries lexically overlap with query terms (paper titles, author names) without
  containing explanatory content. Not yet filtered — candidate fix for M4/M5.
- Retrieval is pure vector similarity (FAISS) with no keyword/exact-match component.
  A production system would likely use hybrid search (vector + BM25); out of scope
  here given corpus size, but noted as a "future work" item.
- Chunking uses whitespace-based word counts as a proxy for token count, not the
  embedding model's actual tokenizer — close enough at this chunk size, but not exact.
- **Sub-questions are retrieved and answered independently, not sequentially** — a
  later sub-question's retrieval/prompt has no knowledge of an earlier sub-question's
  answer. This is fine for independently-answerable multi-hop questions (e.g.
  comparing two papers' reported metrics) but weakens performance on questions with
  true sequential dependency (e.g. "what limitation does A identify in B, and how
  does A's mechanism address it" — the second half genuinely needs the first half's
  specific answer to evaluate properly). Documented via manual testing rather than
  fixed, given project time constraints; see `notes/m2_agent_observations.md`. The
  M4 gold eval set will tag questions as "independent" vs. "dependent" multi-hop to
  surface this gap explicitly rather than average it away.
- The model's confidence-hedging is inconsistent within single answers (observed
  hedging at the start of an answer, stating claims plainly in the middle, then
  re-hedging at the end) — a good target for the critic agent to probe in M3.

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
   reproducibility — no need to re-run ingestion unless rebuilding from scratch:
   ```
   python src/ingestion/fetch_corpus.py
   python src/ingestion/chunk_corpus.py
   python src/ingestion/embed_and_index.py
   ```
5. Test the baseline RAG pipeline:
   ```
   python src/baseline/rag_pipeline.py
   ```
6. Test the planner + retriever agents:
   ```
   python src/agents/planner.py
   python src/agents/retriever.py
   ```

## Tech Stack

- Python 3.11+
- `sentence-transformers` (embeddings, local/free)
- FAISS (vector index, local/free)
- Groq API (LLM generation — free tier, `llama-3.3-70b-versatile`)
- MLflow (experiment tracking — planned, M4)
- Streamlit (demo UI — planned, M5)
