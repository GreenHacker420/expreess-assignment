from typing import List
import streamlit as st

from frontend.utils.api import run_rag_query, submit_feedback
from frontend.utils.helpers import format_doc_title
from frontend.utils.icons import (
    URI_CHAT,
    URI_CHECK,
    URI_CLOCK,
    URI_REFRESH,
    URI_SPARKLE,
)

SAMPLE_QUESTIONS: List[str] = [
    "How do I declare a path parameter with type annotations in FastAPI?",
    "Can query parameters have default values in FastAPI?",
    "What is the difference between File and UploadFile in FastAPI?",
    "How do I define a request body using Pydantic BaseModel?",
    "How do I use Depends() for dependency injection in FastAPI?",
]


def render_qa_tab():
    """Render the Ask Assistant interactive Q&A tab."""
    title_row_html = f"""
    <div class="section-title-row">
        <img src="{URI_CHAT}" width="20" height="20" alt="Chat" />
        <h3 style="margin: 0; color: #086288; font-size: 18px;">Technical Q&A Assistant</h3>
    </div>
    """
    st.html(title_row_html)
    st.caption("Ask questions about the indexed FastAPI documentation. The assistant evaluates document relevance and self-corrects with query rewrites if context is insufficient.")

    # Sample Questions Row
    st.markdown("<p style='font-size: 13px; font-weight: 600; color: #086288; margin-bottom: 6px;'>Quick Sample Questions:</p>", unsafe_allow_html=True)
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for i, q in enumerate(SAMPLE_QUESTIONS):
        short_label = q.split("?")[0].replace("How do I ", "").replace("What is the ", "")
        if cols[i].button(short_label[:22] + "...", key=f"sample_{i}", width="stretch"):
            st.session_state["active_question"] = q
            st.rerun()

    # Question Input
    user_query = st.text_area(
        "Enter your technical question:",
        value=st.session_state.get("active_question", ""),
        height=95,
        placeholder="e.g. How do I declare a path parameter with type annotations in FastAPI?",
        label_visibility="collapsed",
    )

    col_btn, col_clear, _ = st.columns([1.5, 1, 5.5])
    ask_clicked = col_btn.button("Submit Question", type="primary", width="stretch")
    if col_clear.button("Clear", width="stretch"):
        st.session_state["active_question"] = ""
        st.session_state["last_query_response"] = None
        st.rerun()

    if ask_clicked:
        if not user_query.strip():
            st.warning("Please enter a technical question to ask.")
        else:
            with st.spinner("Analyzing query intent, retrieving chunks, and grading relevance..."):
                res = run_rag_query(user_query.strip())
                if res:
                    st.session_state["last_query_response"] = res
                    st.session_state["last_query_latency"] = res.get("_latency", 0.0)
                    st.session_state["feedback_status"] = None
                st.session_state["active_question"] = user_query.strip()
                st.rerun()

    # Display Query Results
    res = st.session_state.get("last_query_response")
    latency = st.session_state.get("last_query_latency", 0.0)

    if res:
        st.markdown("<hr style='border: none; border-top: 1px solid #D8ECF8; margin: 24px 0;' />", unsafe_allow_html=True)

        # Metrics Row
        query_type = res.get("query_type", "how-to")
        retries = res.get("retry_count", 0)
        retrieval_attempts = res.get("retrieval_attempts", 1)
        status = res.get("status", "answered")
        citations = res.get("citations", [])

        retry_text = "Direct Match (0 Retries)" if retries == 0 else f"Self-Corrected ({retries} Rewrite{'s' if retries > 1 else ''})"

        metrics_html = f"""
        <div class="metric-pills">
            <span class="metric-pill pill-type">
                <img src="{URI_SPARKLE}" width="14" height="14" alt="" />
                Intent: {query_type.upper()}
            </span>
            <span class="metric-pill pill-retries">
                <img src="{URI_REFRESH}" width="14" height="14" alt="" />
                {retry_text}
            </span>
            <span class="metric-pill pill-latency">
                <img src="{URI_CLOCK}" width="14" height="14" alt="" />
                Latency: {latency}s ({retrieval_attempts} retrieval{'s' if retrieval_attempts > 1 else ''})
            </span>
            <span class="metric-pill pill-status">
                <img src="{URI_CHECK}" width="14" height="14" alt="" />
                Status: {status.title()} ({len(citations)} Citations)
            </span>
        </div>
        """
        st.html(metrics_html)

        # Grounded Answer Container
        with st.container(border=True):
            st.markdown("<h4 style='color: #086288; margin-top: 0;'>Answer</h4>", unsafe_allow_html=True)
            st.markdown(res.get("answer", ""))

        # Citations Section
        if citations:
            st.markdown("<h4 style='color: #086288; margin-top: 20px;'>Grounded Source Citations</h4>", unsafe_allow_html=True)
            for cit in citations:
                raw_t = cit.get("title", "Document")
                clean_t = format_doc_title(raw_t)
                clean_sec = format_doc_title(cit.get("section", "General"))
                cit_html = f"""
                <div class="citation-box">
                    <span class="citation-tag">[{cit.get('label', 'S1')}]</span>
                    <strong style="color: #086288;">{clean_t}</strong> — <em>{clean_sec}</em>
                    <div style="font-size: 11.5px; color: #4D8DBA; margin-top: 4px;">
                        Chunk ID: <code>{cit.get('chunk_id', '')}</code> · Source: <a href="{cit.get('source', '')}" target="_blank" style="color: #086288; font-weight: 500;">{cit.get('source', '')}</a>
                    </div>
                </div>
                """
                st.html(cit_html)
        else:
            st.info("No specific document chunks were cited for this response.")

        # User Feedback Section
        st.markdown("<h5 style='color: #086288; margin-top: 20px;'>Was this answer helpful?</h5>", unsafe_allow_html=True)
        fb_col1, fb_col2, _ = st.columns([1.2, 1.2, 5.6])
        query_id = res.get("query_id", "")
        fb_status = st.session_state.get("feedback_status")

        if fb_status:
            st.success(f"Feedback recorded: {fb_status.title()}! Thank you.")
        else:
            if fb_col1.button("Helpful", key="fb_up", width="stretch"):
                if submit_feedback(query_id, "up", "Helpful answer"):
                    st.session_state["feedback_status"] = "helpful"
                    st.rerun()

            if fb_col2.button("Irrelevant", key="fb_down", width="stretch"):
                if submit_feedback(query_id, "down", "Needs improvement"):
                    st.session_state["feedback_status"] = "needs improvement"
                    st.rerun()
