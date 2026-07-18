# Planner & Retriever Agent — Observations (M2)

Manual testing notes from building and testing the planner (`src/agents/planner.py`)
and retriever (`src/agents/retriever.py`) agents. These are qualitative findings from
spot-checking behavior during development — not part of the formal gold evaluation
set (M4), but they surface real design tradeoffs worth documenting and carrying into
the critic agent (M3) and final README.

---

## Planner: initial over-decomposition, fixed with few-shot examples

First version of the planner prompt stated "don't force decomposition where none is
needed" as a bare instruction, with no worked example. Result: it over-decomposed
genuinely single-hop questions, e.g. splitting "What is self-reflection in
retrieval-augmented generation?" into two sub-questions ("What is RAG?" + "What is
self-reflection?") even though the original question is a single coherent concept
lookup.

**Fix:** added explicit few-shot examples to the prompt showing single-hop questions
returned unsplit, alongside a genuine 2-hop comparison example. After the fix, the
planner correctly returned single-hop questions unchanged and capped multi-hop
decomposition at 2 sub-questions (per the project's 2-hop scope limit) across every
test case tried afterward.

**Takeaway:** stating a negative constraint ("don't do X") without a positive worked
example of the desired behavior is unreliable with these models — a demonstrated
example of "here is what NOT-decomposing looks like" was necessary, not just an
instruction.

---

## Retriever: inconsistent confidence-signaling within a single answer

Tested question: _"What limitation of RAGentA does A-RAG identify, and how does
A-RAG's hierarchical retrieval interface address it?"_

The retriever's answer to the first sub-question ("What limitation of RAGentA does
A-RAG identify?") opened by stating the context did _not_ explicitly contain the
answer, then went on to describe a specific, plausible limitation (four-agent design
overhead, limited value of RAGentA's second-stage retrieval) in detail, then closed
by saying the limitation "can be inferred" rather than stating it as established
fact.

**Observation:** the model's confidence hedging is not consistent within a single
generated answer — it hedges at the start, answers substantively in the middle, then
re-hedges at the end. This matters directly for the critic agent's design: a critic
that only checks whether a _final_ claim is grounded may miss that the model itself
was uncertain partway through constructing that claim. Worth considering whether the
critic should evaluate the answer's expressed confidence against its actual
groundedness, not just whether the end claim matches source text.

**Separately, on a different test case:** the retriever's answer for a ProRAG
question included an explicit self-flagged gap ("the exact details of ProRAG's
retrieval strategy are not fully explained in the provided context") after otherwise
answering confidently — a case of _good_, honest incompleteness-flagging. So the
model is clearly capable of appropriate hedging; it's just inconsistent about when
it does so within a single generation. This inconsistency itself, rather than a
uniform lack of honesty, is the interesting finding.

---

## Architectural finding: sub-questions are answered independently, not sequentially

Same RAGentA/A-RAG test case above. The second sub-question ("How does A-RAG's
hierarchical retrieval interface address the identified limitation of RAGentA?") was
retrieved and answered with no knowledge of what the first sub-question's answer
actually found. Its answer describes A-RAG's retrieval mechanism in general terms
and explicitly admits: "the context does not provide a direct comparison or
explanation of how A-RAG's hierarchical retrieval interface specifically addresses
the limitations of RAGentA."

**Root cause:** the current retriever design retrieves and answers each sub-question
independently and in parallel — no intermediate answer is threaded into a later
sub-question's retrieval or prompt context. This is fine for sub-questions that are
independently answerable (e.g. "what is X's performance" + "what is Y's
performance"), but breaks down for genuinely _sequential/dependent_ sub-questions,
where the second sub-question can only be properly answered in light of the first
one's specific result (a critique-and-solution question like this one is a clear
example: you need to know _which specific limitation_ was identified before you can
judge whether a given mechanism actually addresses _that_ limitation).

**Decision:** not fixing this in M2. This is being treated as a documented,
deliberate scope boundary rather than a bug to chase down, given the project's time
budget. Two possible mitigations exist for future work: (a) thread prior
sub-answers into later sub-questions' retrieval/prompt context, turning this into a
genuinely sequential pipeline instead of parallel sub-question answering, or (b)
rely on the composer step (M3) to attempt cross-sub-answer synthesis after the fact.
Either way, expect this pipeline to underperform specifically on sequentially
dependent multi-hop questions relative to "independently answerable" multi-hop
questions — this distinction should be reflected in how the M4 gold eval set is
categorized (e.g. tagging questions as "independent 2-hop" vs. "dependent 2-hop")
so the comparison results can show this gap explicitly rather than averaging it away.

---

## Retriever: duplicate chunk_ids in citation extraction

Minor implementation note: initial `extract_cited_chunk_ids()` did not deduplicate
repeated citations (a chunk cited to support multiple sentences appeared multiple
times in the returned `chunk_ids` list). Fixed by deduplicating while preserving
citation order. Noted here only because it's a reminder to check for this same
pattern in the critic agent's output in M3, where accurate counts of distinct
supporting sources may matter for scoring.
