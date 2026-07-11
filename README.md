# Multi-Agent Research Analyst: Planner–Retriever–Critic RAG

## Problem

Plain single-shot RAG (retrieve top-k chunks, generate an answer) struggles with
multi-hop questions that require combining evidence from multiple sources, and
offers no built-in signal for whether an answer is actually grounded in retrieved
evidence versus hallucinated. This project implements and empirically evaluates a
multi-agent pipeline — planner, retriever, critic, composer — over a corpus of
recent (2023–2026) papers on agentic RAG, measuring whether decomposition and
self-verification improve answer faithfulness over a naive RAG baseline.

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

## Roadmap

- [x] M1: Corpus ingestion + baseline single-shot RAG
- [ ] M2: Planner + retriever agents
- [ ] M3: Critic agent + faithfulness scoring
- [ ] M4: Gold evaluation set + comparison harness
- [ ] M5: CI, Streamlit demo, final polish

## Pipeline

(To be completed)

**Known limitations (to revisit in later milestones):**

- Retrieval sometimes surfaces bibliography/reference-list chunks, since reference
  entries lexically overlap with query terms (paper titles, author names) without
  containing explanatory content. Not yet filtered — candidate fix for M4/M5.
- Retrieval is pure vector similarity (FAISS) with no keyword/exact-match component.
  A production system would likely use hybrid search (vector + BM25); out of scope
  here given corpus size, but noted as a "future work" item.

## Tech Stack

- Python 3.10+
- `sentence-transformers` (embeddings, local/free)
- FAISS (vector index, local/free)
- Groq API (LLM generation — free tier)
- MLflow (experiment tracking — planned, M4)
- Streamlit (demo UI — planned, M5)
