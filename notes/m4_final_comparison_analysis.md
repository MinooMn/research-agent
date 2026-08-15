# M4 Comparison Analysis — Manual Review of the Results

> **Note:** This analysis is based on results produced with the original Llama
> 3.1/3.3 models via Groq, prior to the migration to GPT-OSS-20B/120B following
> Groq's model deprecation announcement. See `README.md`.

Manual review of `results/comparison_checkpoint.jsonl` across the first 13 gold
questions (both pipelines). Decision made not to further tune the system or rerun
the full comparison after these findings, as the primary goal of this project was
an exploratory comparison of the two pipelines and identification of each one's
limitations, rather than optimizing for the best possible metric.

## Headline finding: baseline and multi-agent show similar behavior on 2-hop questions

On independent_2hop questions reviewed so far (i2h_011, i2h_012, i2h_013), the
two pipelines retrieved largely the same top chunks and produced similarly
correct/incorrect answers -- where one pipeline was wrong, the other usually was
too, and likewise for correct cases. **The multi-agent architecture did not show
a clear empirical advantage over the baseline in this evaluation.**

Likely mechanistic explanation: the planner's sub-questions often closely
resemble the wording of the original question rather than diverging enough to
target meaningfully different retrieval results. If sub-question retrieval
converges on the same chunks the baseline's single query would have found
anyway, decomposition provides no retrieval-targeting benefit -- the
theoretical advantage of the architecture (disambiguated, targeted per-hop
retrieval) isn't materializing in practice on this corpus and question set.

This is treated as a legitimate, informative negative result, not a failure of
the project -- see "What this means for the project" below.

## Other findings

**Single-hop questions sometimes get needlessly decomposed, causing wrong
answers (sh_001).** Despite the planner's few-shot fix (M2) reducing
over-decomposition on clearly single-concept questions, some phrasing patterns
still trigger unnecessary 2-hop treatment (see below).

**Retrieval quality issues occur in both systems, not just one** (sh_002,
sh_003, sh_006, sh_007, sh_009) -- bad retrieval is a shared, systemic
weakness rather than something the multi-agent architecture avoids.

**Decomposition doesn't always hurt** -- sh_002 was correctly answered by the
multi-agent pipeline despite being single-hop, showing the failure mode above
is inconsistent, not universal.

**Composer overuses epistemic hedging language** ("based on the information,"
"not fully provided") even in cases with reasonably strong grounding (sh_002).
This may be systematically deflating the correctness judge's scoring of
multi-agent answers independent of whether the underlying information was
actually adequate -- a bias worth naming, since it means low correctness
scores for multi-agent don't always mean the evidence was insufficient, only
that the composer chose to hedge.

**Critic faithfulness scoring has a real, unfixed measurement bug affecting
some scores downward artificially (sh_008 and others).** Two related citation-
parsing failure modes were identified in `extract_claims()`:

1. Multiple separate `[chunk_id]` citations placed back-to-back (e.g.
   `[id1], [id2]`) produce empty/punctuation-only "claims" for the gap between
   them, which get sent to the critic and fail as unparseable, logged as
   `"verdict": "error"`.
2. Multiple chunk_ids inside a single bracket (e.g. `[id1, id2]`) don't match
   the citation regex at all, so no claim boundary is created there.

Either failure can make a well-grounded answer appear less faithful than it
actually is, purely due to citation formatting, not actual ungroundedness.
Decision made not to fix. A fix and full rerun were considered out of scope
given project time constraints and the marginal value of chasing a cleaner
number over documenting the limitation honestly. Reported faithfulness scores
should be read as a noisy underestimate in cases with multi-citation clusters,
not a precise measurement.

**Single-hop vs. 2-hop categorization is genuinely ambiguous for some question
phrasings**, particularly ones using comparative language ("improvement,"
"compared to") without necessarily requiring two independent sources. This
appears to be a real difficulty for humans too, not just the planner -- some
of these questions could arguably belong in either category. Treated as an
inherent limitation of the categorization scheme, not a drafting error.

**The correctness judge can penalize answers that are reasonable but drawn
from different (correct) source text than the gold reference chunks.** Several
questions concern facts (strategies, general improvements) that appear in
multiple places in a paper with different wording. When the retriever surfaces
a different-but-equally-valid passage than the one the gold answer was written
from, the LLM judge may mark the resulting answer "incorrect" even though it's
not actually wrong -- just differently sourced. **Not all "incorrect" labels in
the results indicate a real system failure; some reflect gold-set/judge
brittleness.** This is an eval methodology limitation, not a pipeline flaw.

## What this means for the project

The core, honest conclusion from this evaluation: **this implementation of the
planner-retriever-critic-composer architecture did not demonstrate a clear
faithfulness or correctness advantage over a simple single-shot RAG baseline
on this corpus and question set**, and several confounds (citation-parsing
measurement noise, eval-set chunk-matching brittleness, composer hedging bias)
make the comparison less precise than ideal. Rather than optimize further to
try to produce a cleaner "multi-agent wins" result, this is being reported as
the actual finding, with the confounds documented so a reader can weigh the
result appropriately.
