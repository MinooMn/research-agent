"""
extract_claims() is the parser that turns [chunk_id]-cited answer text into
(claim, chunk_id) pairs for the critic to score. Two of the cases below
document known-bad-but-accepted parsing behavior described in
notes/m4_final_comparison_analysis.md ("Critic faithfulness scoring has a
real, unfixed measurement bug...") rather than asserting the "correct"
behavior -- these are regression tests for the documented status quo, not
a spec for how it should work.
"""

from src.agents.critic import extract_claims


def test_single_claim_single_citation():
    text = "ProRAG achieves an F1 of 85 on the PopQA dataset [2601.21912_22]."
    assert extract_claims(text) == [
        {
            "claim": "ProRAG achieves an F1 of 85 on the PopQA dataset",
            "chunk_id": "2601.21912_22",
        }
    ]


def test_multiple_claims_split_on_each_citation():
    text = (
        "A-RAG uses hierarchical retrieval [2602.03442_13]. "
        "It avoids noise from irrelevant chunks [2602.03442_14]."
    )
    assert extract_claims(text) == [
        {"claim": "A-RAG uses hierarchical retrieval", "chunk_id": "2602.03442_13"},
        {
            "claim": ". It avoids noise from irrelevant chunks",
            "chunk_id": "2602.03442_14",
        },
    ]


def test_no_citations_returns_empty_list():
    assert extract_claims("This answer has no citations at all.") == []


def test_back_to_back_citations_produce_punctuation_only_claim():
    """Known parsing gap: [id1], [id2] back-to-back leaves only ", " as the
    "claim" text for the second citation, which survives .strip() as "," and
    gets sent to the critic as a bogus claim (see m4 final comparison notes,
    finding #1)."""
    text = "This is supported by two sources [2601.21912_22], [2602.03442_13]."
    assert extract_claims(text) == [
        {
            "claim": "This is supported by two sources",
            "chunk_id": "2601.21912_22",
        },
        {"claim": ",", "chunk_id": "2602.03442_13"},
    ]


def test_multiple_ids_in_one_bracket_are_not_matched():
    """Known parsing gap: [id1, id2] doesn't match the single-chunk-id citation
    regex at all, so no claim boundary is created there and the citation is
    silently dropped (see m4 final comparison notes, finding #2)."""
    text = "Both benchmarks show gains [2601.21912_22, 2602.03442_13]."
    assert extract_claims(text) == []
