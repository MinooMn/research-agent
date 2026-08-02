# Multi-Agent Pipeline — Integration Test Observations

First end-to-end test of the full pipeline (`src/multi_agent_pipeline.py`):
planner -> retriever -> critic -> composer, run on 2 real questions (1 single-hop
control, 1 dependent_2hop from the gold set).

## Case 1: retrieval attribution error, correctly caught by the critic/composer

**Question:** "What embedding model does the RAGentA paper use?"

**Retriever's answer:** "The RAGentA paper uses Qwen3-Embedding-0.6B
[2602.03442_10]."

**Critic:** scored 0.0 faithfulness (unsupported).

**Composer:** correctly refused to state the claim as fact, explicitly noting the
information is unverified.

**Root cause, traced by hand:** chunk `2602.03442_10` is from **A-RAG's paper**,
not RAGentA's. The chunk is A-RAG's own experimental setup description: "all
methods [in A-RAG's benchmark comparison, including RAGentA] utilize
Qwen3-Embedding-0.6B... for dense retrieval." This is A-RAG's authors describing
the embedding model _they_ used to benchmark RAGentA alongside other baselines --
not a claim from RAGentA's own paper about what RAGentA itself was built with.

**Checked whether the real answer was even retrievable:** yes -- RAGentA's own
paper states its actual embedding model plainly, in chunk `2506.16988_5`:
"semantic retrieval is powered by dense embeddings using the intfloat/e5-base-v2
model (E5)," combined with BM25 in a tunable hybrid (alpha=0.35). This chunk was
never surfaced by retrieval. So this is not just an attribution ambiguity between
two plausible sources -- it's a genuine **retrieval miss**: the correct answer
existed in the corpus and was retrievable in principle, but top-k vector search
returned a topically-similar wrong-paper chunk instead of the chunk that actually
answers the question. The critic/composer layer did its job correctly (refusing
to state the wrong answer with confidence), but the deeper problem is one stage
upstream, in retrieval quality itself, not just citation/attribution.

**Why this finding matters:** this is a concrete, real example of exactly the
failure mode faithfulness-checking is designed to catch. A naive single-shot RAG
system would very plausibly have stated "RAGentA uses Qwen3-Embedding-0.6B" as
fact with no way to detect the misattribution. Here, the critic didn't diagnose
_why_ the claim was wrong (it doesn't reason about cross-paper attribution
specifically, just claim-vs-chunk support), but it correctly detected that the
chunk didn't actually support the claim being attributed to it, and the composer
correctly refused to present the claim with confidence. This is a good, concrete,
traceable example to cite in the final README/portfolio writeup as evidence the
architecture delivers on its core thesis.

**Not fixed / out of scope:** improving retrieval to disambiguate "chunk mentions
paper X" from "chunk is _from_ paper X and describes X's own claims" would need
either metadata-aware retrieval filtering or a more sophisticated retriever
prompt instructing it to check the chunk's own `arxiv_id` against the paper the
question is actually about. Worth noting as a specific, well-understood future
improvement rather than a vague "retrieval could be better" statement.

## Case 2: dependent_2hop (d2h_023), real gold-set question

**Sub-questions from the real planner:**

1. "How does A-RAG's architecture address computational overhead?"
2. "What are the reported head-to-head results of A-RAG compared to RAGentA?"

As noted separately (see planner/composer interaction note above), sub-Q1
decomposed on surface structure rather than preserving the specific dependency
on RAGentA's stated overhead -- consistent with the recurring cross-pipeline
synthesis-failure pattern documented in `notes/m2_agent_observations.md` and
`notes/m4_question_drafting_observations.md`.

Sub-Q1 scored 1.0 faithfulness (well-grounded, A-RAG's own context-efficiency
claims). Sub-Q2 scored 0.0 -- likely due to imprecise metric-name attribution
("LLM-Acc and Contain-Acc") rather than a misattribution error like Case 1.

**Composer's final answer:** explicitly separated what was verified (A-RAG's
context-efficiency mechanism) from what wasn't (the specific head-to-head
comparison), and explicitly stated it could not fully connect the two -- again,
correct, honest behavior rather than forcing an unsupported causal link between
A-RAG's mechanism and its reported outperformance of RAGentA.

## Overall takeaway

Both integration test cases show the pipeline behaving exactly as intended:
epistemically honest under weak or misattributed grounding, rather than
confidently synthesizing an answer that outruns its evidence. This is the core,
measurable value proposition motivating the whole project, and these two traced
examples (especially Case 1's misattribution) are strong concrete evidence for
the final README and comparison writeup, independent of the aggregate M4
faithfulness score comparison still to come.
