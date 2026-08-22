import streamlit as st
import sys
import os

# Add repo root to sys.path so `src.*` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASELINE = "Baseline (single-shot RAG)"
MULTI_AGENT = "Multi-Agent (planner -> retriever -> critic -> composer)"

st.set_page_config(page_title="Agentic RAG: Baseline vs Multi-Agent", page_icon="🔎")

st.title("Baseline vs Multi-Agent RAG")
st.caption(
    "Ask a question against a fixed corpus of 10 arXiv papers on agentic RAG, "
    "and compare a single-shot RAG baseline against a planner -> retriever -> "
    "critic -> composer multi-agent pipeline. Faithfulness score = share of "
    "cited claims the critic verified against the actual retrieved source text."
)
st.info(
    "This is a RAG system answering questions about a fixed corpus of 10 arXiv "
    "papers on agentic RAG — try asking about specific methods like Self-RAG, "
    "RQ-RAG, or A-RAG"
)


@st.cache_resource(show_spinner="Loading embedding model and FAISS index (first run only)...")
def load_pipeline():
    # Imported lazily so the title/caption above render immediately -- these
    # imports pull in src.retrieval, which loads the SentenceTransformer model
    # and FAISS index at import time. st.cache_resource means this only runs
    # once per app process, not on every rerun/button click.
    from src.agents.critic import check_faithfulness
    from src.baseline.rag_pipeline import answer_question
    from src.chunk_store import load_chunks
    from src.multi_agent_pipeline import run_multi_agent

    return check_faithfulness, answer_question, load_chunks, run_multi_agent


check_faithfulness, answer_question, load_chunks, run_multi_agent = load_pipeline()

question = st.text_input(
    "Question",
    placeholder="e.g. What embedding model does the RAGentA paper use?",
)
pipeline_choice = st.selectbox("Pipeline", [BASELINE, MULTI_AGENT])
run_clicked = st.button("Run", type="primary", disabled=not question.strip())


def faithfulness_label(score: float) -> str:
    if score >= 0.75:
        return "VERIFIED"
    if score > 0:
        return "PARTIALLY VERIFIED"
    return "UNVERIFIED -- LOW CONFIDENCE"


def show_faithfulness(score: float):
    st.metric("Faithfulness score", f"{score:.2f}")
    label = faithfulness_label(score)
    if label == "VERIFIED":
        st.success(label)
    elif label == "PARTIALLY VERIFIED":
        st.warning(label)
    else:
        st.error(label)


if run_clicked:
    spinner_msg = (
        "Running pipeline... (first run also loads the embedding model, "
        "so it may take a bit longer)"
    )
    with st.spinner(spinner_msg):
        try:
            if pipeline_choice == BASELINE:
                result = answer_question(question)
                all_chunks = load_chunks()
                cited_chunks = {
                    cid: all_chunks[cid]
                    for cid in result["chunk_ids"]
                    if cid in all_chunks
                }
                faithfulness = check_faithfulness(result["answer"], cited_chunks)
                answer = result["answer"]
                chunk_ids = result["chunk_ids"]
                score = faithfulness["faithfulness_score"]
                trace = None
            else:
                result = run_multi_agent(question)
                answer = result["final_answer"]
                chunk_ids = sorted(
                    {cid for sr in result["sub_results"] for cid in sr["chunk_ids"]}
                )
                score = result["avg_faithfulness_score"]
                trace = result["sub_results"]
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
        else:
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Citations")
            st.write(
                ", ".join(f"`{cid}`" for cid in chunk_ids) if chunk_ids else "_none_"
            )

            show_faithfulness(score)

            if trace is not None:
                with st.expander(
                    "Pipeline trace: sub-questions & per-answer faithfulness"
                ):
                    for sr in trace:
                        sub_score = sr["critic_result"]["faithfulness_score"]
                        st.markdown(
                            f"**{sr['sub_question']}** "
                            f"— faithfulness {sub_score:.2f} ({faithfulness_label(sub_score)})"
                        )
                        st.write(sr["answer"])
                        st.caption(
                            "Sources: " + ", ".join(f"`{c}`" for c in sr["chunk_ids"])
                        )
