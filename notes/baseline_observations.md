# Baseline RAG — Multi-Hop Observations

Manual test of `src/baseline/rag_pipeline.py` (single-shot retrieval + generation,
top_k=5) against questions requiring synthesis across multiple papers in the corpus.
Goal: establish early, informal evidence of baseline failure modes before building
the multi-agent (planner/retriever/critic) pipeline, to compare against later in M4.

**Caveat:** these are informal spot checks with unverified reference answers (not
read against the full papers in detail) — useful for spotting failure _patterns_,
not a substitute for the gold evaluation set in M4.

---

## Q1: "What earlier method does RQ-RAG build on, and what specific limitation of

that method does it try to address?"

**Answer:** Refused — "context does not contain enough information."

**Sources retrieved:** `2602.03442_11`, `2602.03442_21`, `2404.00610_12`,
`2602.03442_14`, `2404.00610_13`

**Observation:** Retrieval _did_ surface RQ-RAG chunks (`2404.00610_12`, `_13`), so
this is not a retrieval failure. The model refused despite having at least some
relevant material. **Failure mode: synthesis/reasoning failure, not retrieval
failure.** Open question (not yet checked): whether chunks `_12`/`_13` actually
state RQ-RAG's predecessor/limitation explicitly, or only imply it — worth
verifying by hand before drawing a stronger conclusion.

---

## Q2: "How does A-RAG's approach to retrieval differ from RAGentA's, and which one

reports better performance?"

**Answer:** Partially answered. Refused the "how do approaches differ" half, but
correctly answered "which performs better" — citing that A-RAG (Naive and Full)
outperforms RAGentA across benchmarks, with specific chunk citations.

**Sources retrieved:** `2310.11511_22`, `2502.01142_16`, `2602.03442_9`,
`2506.16988_9`, `2602.03442_10`

**Observation:** Best result of the day. Retrieval pulled chunks from **both**
target papers (`2602.03442` = A-RAG, `2506.16988` = RAGentA), and the model
successfully synthesized a factual/quantitative comparison (performance) across
them. It failed only on the more conceptual half of the question (mechanistic
differences in retrieval approach). **Suggests baseline can handle cross-paper
comparison when the comparison is a concrete reported number, but struggles with
cross-paper conceptual/mechanistic comparison.**

---

## Q3: "What is the relationship between DeepRAG's step-wise retrieval and RQ-RAG's

query refinement — are they solving the same problem differently, or different
problems?"

**Answer:** Refused — discusses DeepRAG and RQ-RAG "separately" but does not
compare them.

**Sources retrieved:** `2404.00610_12`, `2502.01142_16`, `2502.01142_30`,
`2502.01142_2`, `2502.01142_17`

**Observation:** Retrieval imbalance — 4 of 5 chunks are from DeepRAG
(`2502.01142`), only 1 from RQ-RAG (`2404.00610`). The query didn't retrieve
balanced coverage of both papers, so the model wasn't given a fair basis for
comparison. **Failure mode: retrieval imbalance/coverage failure, distinct from
Q1's pure synthesis failure** — top-k similarity search skewed toward one paper
even though the question explicitly asks about two.

---

## Q4: "Which of the papers in this corpus cite Self-RAG, and what do they each say

its main limitation is?"

**Answer:** Refused entirely — "does not provide a list of papers that cite it or
their respective comments on its limitations."

**Sources retrieved:** `2506.16988_13`, `2606.22681_22`, `2310.11511_47`,
`2310.11511_26`, `2501.09136_9`

**Observation:** Retrieval found Self-RAG itself plus three other papers
(`2506.16988`, `2606.22681`, `2501.09136`) — reasonable multi-source coverage. Yet
the model refused completely, with no partial credit given even for what material
was available. More complete refusal than Q2's partial success, despite seemingly
comparable retrieval quality. **Failure mode: synthesis failure, more severe than
Q1 — suggests refusal behavior may not scale gracefully with the number of
sources/sub-claims involved (2-paper comparison partially worked in Q2; a
broader "which of N papers" question failed outright here).**

---

## Summary pattern across all 4 questions

Single-shot RAG (top_k=5) shows two distinct, separable failure modes on multi-hop
questions:

1. **Retrieval coverage failure** (Q3): top-k similarity search does not
   guarantee balanced retrieval across all papers/sources a question spans —
   it can skew heavily toward one source even when the question explicitly
   requires two.
2. **Synthesis failure** (Q1, Q4, and partially Q2): even when relevant chunks
   from multiple sources _are_ retrieved, the model does not reliably combine
   them into an answer — it defaults to conservative refusal rather than
   hallucinating, but this means genuine multi-hop questions frequently go
   unanswered even when the underlying evidence was available in context.

Both failure modes are plausible targets for the planner/retriever/critic pipeline
to address: the planner's decomposition should mitigate (1) by issuing separate,
targeted retrieval per sub-question instead of one blended query; the composer
step should mitigate (2) by combining independently-verified sub-answers instead
of asking the model to synthesize everything in one pass.

These four Q&A pairs are not part of the formal gold evaluation set (M4) but serve
as early qualitative evidence motivating the multi-agent design.
