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

🚧 Work in progress. Corpus ingestion, chunking, retrieval, and a working baseline
single-shot RAG pipeline (retrieval + generation) are complete. Next: planner and
retriever agents (M2).

## Roadmap

- [x] M1: Corpus ingestion + baseline single-shot RAG
- [ ] M2: Planner + retriever agents
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
   `sentence-transformers/all-MiniLM-L6-v2` (chosen for speed on CPU-only hardware,
   zero cost, and being the standard default for this kind of RAG prototype) and
   builds a FAISS `IndexFlatL2` index (exact search — more than sufficient at this
   corpus size). Index saved to `data/index/faiss.index`; row order corresponds
   directly to line order in `chunks.jsonl`.
4. `src/ingestion/sanity_check_retrieval.py` — manual sanity check: embeds a test
   query, retrieves top-k chunks, prints source paper/chunk for each. Verified
   retrieval correctly surfaces the most relevant paper/section for test queries.

**Baseline RAG (single-shot):**

- `src/retrieval.py` — shared retrieval module: embeds a query, searches the FAISS
  index, returns top-k chunks with `chunk_id`, distance, and text. Used by both the
  baseline pipeline and (later) the retriever agent.
- `src/llm_client.py` — thin, generic wrapper around the Groq API
  (`llama-3.3-70b-versatile` by default). Takes a prompt, returns generated text.
  Deliberately has no knowledge of retrieval/RAG — kept generic so the planner and
  critic agents (M2/M3) can reuse it for non-retrieval LLM calls.
- `src/baseline/rag_pipeline.py` — orchestrates the baseline pipeline: retrieve
  top-k chunks for a question → build a grounding-instructed prompt (explicitly
  told to answer only from context and say so if context is insufficient) → call
  `llm_client.generate()` → return `{query, answer, chunk_ids}`. Manually verified:
  correctly answers grounded questions, and correctly declines to answer when the
  retrieved context doesn't cover the question (e.g. a generic "what are
  transformers?" query outside the corpus's actual content).

**Known limitations (to revisit in later milestones):**

- Retrieval sometimes surfaces bibliography/reference-list chunks, since reference
  entries lexically overlap with query terms (paper titles, author names) without
  containing explanatory content. Not yet filtered — candidate fix for M4/M5.
- Retrieval is pure vector similarity (FAISS) with no keyword/exact-match component.
  A production system would likely use hybrid search (vector + BM25); out of scope
  here given corpus size, but noted as a "future work" item.
- Chunking uses whitespace-based word counts as a proxy for token count, not the
  embedding model's actual tokenizer — close enough at this chunk size, but not exact.

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
   reproducibility — no need to re-run ingestion unless you want to rebuild it from
   scratch:
   ```
   python src/ingestion/fetch_corpus.py
   python src/ingestion/chunk_corpus.py
   python src/ingestion/embed_and_index.py
   ```
5. Test the baseline RAG pipeline:
   ```
   python src/baseline/rag_pipeline.py
   ```

## Tech Stack

- Python 3.11+
- `sentence-transformers` (embeddings, local/free)
- FAISS (vector index, local/free)
- Groq API (LLM generation — free tier, `llama-3.3-70b-versatile`)
- MLflow (experiment tracking — planned, M4)
- Streamlit (demo UI — planned, M5)
