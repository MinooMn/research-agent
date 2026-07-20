# Critic Agent — Observations (M3)

## Compound-claim over-generosity

Manual spot-check of `check_faithfulness()` against a real retriever output (not
just trusting the critic's own verdict) surfaced a real limitation.

**Test case:** claim extracted from a retriever answer about A-RAG's hierarchical
retrieval interface:

> "A-RAG's hierarchical retrieval interface addresses the identified limitation of
> RAGentA by allowing the agent to make autonomous judgments and precisely read the
> most relevant content" [chunk_id: 2602.03442_13]

**Critic verdict:** `supported`, with reasoning citing the part of the source chunk
about "progressive information acquisition design allows the agent to make
autonomous judgments and precisely read the most relevant content."

**Manual review of the actual source chunk (2602.03442_13):** the chunk is from
A-RAG's own ablation/results section (comparing "A-RAG Full" vs. "w/o Chunk Read").
It does support the _second half_ of the claim (autonomous judgment, precise
reading of relevant content) almost verbatim. However, the chunk **never mentions
RAGentA or any limitation of RAGentA at all** — so the _first half_ of the claim
("...addresses the identified limitation of RAGentA...") is entirely unsupported by
this source. The critic marked the whole sentence "supported" based on the
well-matching second half, without noticing the unsupported first half.

**Root cause:** claim splitting is sentence-level (split on inline `[chunk_id]`
citation boundaries), not sub-clause-level. A single sentence can contain multiple
distinct assertions, and the critic's judgment appears to anchor on whichever part
matches most strongly, rather than checking each sub-assertion independently.

**Mitigation attempted:** added an explicit instruction to the critic prompt:
"If the claim contains multiple parts, and ANY part is not supported by the source
text, mark it as unsupported — do not mark a claim as supported just because part
of it matches." Re-ran the identical test case. **Result: verdict did not change**
— still marked "supported." The guardrail did not fix the issue on this example.

**Decision:** not pursuing further fixes (e.g. sub-clause-level claim splitting) —
out of scope for the project timeline. Documenting as a known, real limitation of
the faithfulness metric rather than silently accepting an inflated score. This
should be disclosed explicitly in the final README's limitations/future-work
section: the faithfulness score as implemented likely somewhat over-estimates true
faithfulness on compound-sentence claims, and a more rigorous implementation would
split claims at the sub-clause level rather than the citation-boundary/sentence
level.

**Why this is still a reasonable v1, despite the flaw:** the metric is being
applied identically to both the baseline and multi-agent pipelines in the M4
comparison, so this bias is likely present in both conditions rather than
favoring one over the other — a relative comparison between the two pipelines
remains meaningful even though the absolute faithfulness numbers are probably
somewhat inflated for both.

---

## Deliberately unfaithful test cases

Ran 3 hand-crafted cases against `check_faithfulness()`: a wrong-number claim
(real chunk, fabricated F1 score), an off-topic claim (real chunk, fabricated
fact about ProRAG's origin), and a known-good control (reused from Day 7).

**Results:** control case correctly scored `supported` (1.0). Both fabricated
cases correctly scored `0.0` faithfulness -- the critic did not mark either as
falsely "supported." However, both were labeled `unsupported` rather than one
being labeled `contradicted` -- the critic isn't reliably distinguishing "source
says otherwise" from "source doesn't address this at all," despite the wrong-
number case being a clear contradiction (source states 51.6, claim states 85).

**Verdict:** core validation requirement met -- the critic correctly flags
fabricated/unfaithful claims and does not silently mark them as supported. The
contradicted-vs-unsupported category distinction is a known, minor limitation,
likely fixable with few-shot examples in the critic prompt, but not pursued
further given project time constraints. `faithfulness_score` (based on
supported-vs-not) is unaffected by this category confusion, since both
"unsupported" and "contradicted" count identically as not-supported in the
current scoring formula.
