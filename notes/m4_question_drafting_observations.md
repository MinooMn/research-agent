# Gold Question Drafting — Observations

## v1: LLM drafting from titles alone fabricated content

First drafting pass gave the LLM only paper titles, no real text. Result:
several questions/reference answers contained fabricated statistics (e.g. a
"Natural Questions" benchmark result that doesn't appear anywhere in this
corpus) and generic textbook-style mechanism descriptions not grounded in the
actual papers. Rejected wholesale rather than spot-fixed.

## v2: grounding in real excerpts fixed fabrication, but exposed a synthesis gap

Rewrote drafting to inject real chunk text (2 intro/abstract chunks + 1
results-oriented chunk per paper, selected via keyword scoring then manually
verified to exclude reference-list contamination). This fixed the fabrication
problem entirely -- v2's single_hop and independent_2hop questions all check
out against real reported numbers and mechanisms.

However, the first dependent_2hop batch (using the same generic seed chunks)
revealed a design flaw: intro/abstract chunks contain each paper's OWN
generic motivation ("existing RAG methods fail to X"), not a specific named
critique of the paired paper. Result: "dependent" questions weren't actually
testing cross-paper dependency -- they read as single-paper motivation
statements with an unused second paper attached.

## v3: targeted excerpt overrides for dependent pairs -- partial fix

Found real cross-paper critique material by grepping each paper's chunks for
the other paper's name and manually reading context:

- A-RAG (2602.03442) vs RAGentA (2506.16988): a genuine head-to-head results
  table exists (chunk 2602.03442_10), and RAGentA's own paper states its
  four-agent design's overhead limitation (chunk 2506.16988_11).
- RQ-RAG (2404.00610) vs Self-RAG (2310.11511): RQ-RAG explicitly compares
  training-data efficiency against Self-RAG in its own text (chunk
  2404.00610_11) -- genuine, specific, real numbers (150k vs ~40k training
  examples, 1.9% improvement).
- DeepRAG (2502.01142) vs Self-RAG: dropped this pair entirely. DeepRAG's
  only mention of Self-RAG is a one-line related-work name-drop with no
  stated limitation or comparison -- insufficient material to force a
  genuine dependent question from.

Re-ran drafting with these targeted excerpts substituted in for the two
remaining dependent pairs. **Result: still not fully fixed.** Even given the
correct dual-source excerpts, the LLM drafter continued to default to
whichever paper's framing was more self-contained -- one drafted question
described "Standard RAG" instead of A-RAG (effectively ignoring the A-RAG
excerpt), and the other described "existing RAG methods" generically instead
of RQ-RAG specifically (ignoring the RQ-RAG-specific comparison in favor of
Self-RAG's own self-description).

**Decision:** did not pursue further prompt iteration. Hand-authored the final
2 dependent_2hop questions directly from the verified excerpts instead --
faster and more reliable than continuing to debug LLM drafting behavior for a
category of only 2 questions.

## Takeaway

This is the same qualitative pattern observed with the baseline RAG pipeline
back in early testing (Day 4) and with the retriever agent (M2): even when
given correct, relevant source material from multiple documents, the model
tends to synthesize from the single most self-contained source rather than
genuinely combining both. This recurring pattern across three different parts
of the pipeline (baseline generation, retriever sub-answers, and now question
drafting) is a stronger, more specific finding than any one instance alone --
worth stating explicitly in the final README as a consistent, cross-cutting
limitation of single-pass LLM synthesis over multiple sources, not a one-off
prompt engineering issue.
