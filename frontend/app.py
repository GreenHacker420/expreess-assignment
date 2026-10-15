import base64
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

# Vector SVG favicon
FAVICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.svg")

st.set_page_config(
    page_title="Technical Documentation Assistant · LangGraph RAG",
    page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
HTTP_TIMEOUT: float = 60.0

SAMPLE_QUESTIONS = [
    "How do I declare a path parameter with type annotations in FastAPI?",
    "Can query parameters have default values in FastAPI?",
    "What is the difference between File and UploadFile in FastAPI?",
    "How do I define a request body using Pydantic BaseModel?",
    "How do I use Depends() for dependency injection in FastAPI?",
]


# -----------------------------------------------------------------------------
# Base64 Vector SVGs (Curated with #086288 and #4D8DBA strokes)
# -----------------------------------------------------------------------------
def b64_svg(svg_code: str) -> str:
    encoded = base64.b64encode(svg_code.strip().encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


URI_LOGO = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#086288" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/><polyline points="10 7 14 11 10 15"/></svg>"""
)

URI_SPARKLE = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#086288" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3L12 3z"/></svg>"""
)

URI_DOC = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#086288" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>"""
)

URI_UPLOAD = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#086288" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>"""
)

URI_GRAPH = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#086288" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="3"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="18" r="3"/><line x1="9" y1="6" x2="15" y2="6"/><line x1="6" y1="9" x2="6" y2="15"/><line x1="18" y1="9" x2="18" y2="15"/><line x1="8.5" y1="8.5" x2="15.5" y2="15.5"/></svg>"""
)

URI_CHECK = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#086288" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>"""
)

URI_CLOCK = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4D8DBA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>"""
)

URI_REFRESH = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#086288" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>"""
)

URI_CODE = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#086288" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>"""
)

URI_CHAT = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#086288" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>"""
)

URI_LAYERS = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#4D8DBA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>"""
)


# -----------------------------------------------------------------------------
# Clean Ocean Azure & Warm Cream CSS Design System
# -----------------------------------------------------------------------------
OCEAN_CREAM_CSS = """
<style>
/* Clean Soft Oceanic Background */
.stApp {
    background-color: #F5F9FC !important;
    background-image: none !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #0E2A38 !important;
}

.main .block-container {
    max-width: 1200px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
}

/* Top Header Card */
.top-header-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #FFFFFF;
    border: 1px solid #D8ECF8;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(8, 98, 136, 0.05);
}

.top-brand {
    display: flex;
    align-items: center;
    gap: 16px;
}

.top-brand-icon {
    width: 48px;
    height: 48px;
    background: #EBF5FB;
    border: 1px solid #97D7FF;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.top-brand-title {
    font-size: 20px;
    font-weight: 700;
    color: #086288;
    margin: 0;
    line-height: 1.25;
}

.top-brand-subtitle {
    font-size: 13px;
    color: #4D8DBA;
    margin: 3px 0 0 0;
}

.header-badges {
    display: flex;
    align-items: center;
    gap: 10px;
}

.status-pill-blue {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #EBF5FB;
    border: 1px solid #97D7FF;
    color: #086288;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 12.5px;
    font-weight: 600;
}

.status-pill-warm {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #FFEFC4;
    border: 1px solid #F5DC9A;
    color: #614605;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 12.5px;
    font-weight: 600;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #086288;
}

.status-dot-green {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #2E8B57;
}

/* Tabs Navigation Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent !important;
    border-bottom: 2px solid #D8ECF8 !important;
    padding-bottom: 4px;
    margin-bottom: 24px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D8ECF8 !important;
    border-radius: 8px !important;
    padding: 8px 22px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #4D8DBA !important;
    transition: all 0.15s ease-in-out;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #EBF5FB !important;
    border-color: #97D7FF !important;
    color: #086288 !important;
}

.stTabs [aria-selected="true"] {
    background-color: #086288 !important;
    border-color: #086288 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Metric Badges Row */
.metric-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 16px 0;
}

.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 12.5px;
    font-weight: 600;
}

.pill-type {
    background: #EBF5FB;
    border: 1px solid #97D7FF;
    color: #086288;
}

.pill-retries {
    background: #FFEFC4;
    border: 1px solid #F3DC9B;
    color: #614605;
}

.pill-latency {
    background: #FFFFFF;
    border: 1px solid #D8ECF8;
    color: #4D8DBA;
}

.pill-status {
    background: #EBF7EE;
    border: 1px solid #BCE5C8;
    color: #1B6338;
}

/* Citation Card */
.citation-box {
    background: #FFFFFF;
    border: 1px solid #D8ECF8;
    border-left: 4px solid #086288;
    border-radius: 8px;
    padding: 12px 18px;
    margin-top: 10px;
    font-size: 13px;
    color: #0E2A38;
    box-shadow: 0 1px 4px rgba(8, 98, 136, 0.03);
}

.citation-tag {
    display: inline-block;
    background: #FFEFC4;
    color: #614605;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11.5px;
    margin-right: 8px;
    border: 1px solid #F3DC9B;
}

/* Document Grid Cards */
.doc-card-container {
    background: #FFFFFF;
    border: 1px solid #D8ECF8;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    box-shadow: 0 2px 8px rgba(8, 98, 136, 0.04);
}

.doc-card-container:hover {
    border-color: #97D7FF;
    box-shadow: 0 4px 14px rgba(8, 98, 136, 0.08);
}

.doc-badge-pill {
    background: #FFEFC4;
    color: #614605;
    font-size: 11.5px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 9999px;
    border: 1px solid #F3DC9B;
    display: inline-block;
}

.doc-chunk-pill {
    background: #EBF5FB;
    color: #086288;
    font-size: 11.5px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 9999px;
    border: 1px solid #97D7FF;
    display: inline-block;
}

/* Architecture Node Flow */
.flow-node {
    background: #FFFFFF;
    border: 1px solid #D8ECF8;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(8, 98, 136, 0.04);
    height: 100%;
}

.flow-node-title {
    font-weight: 700;
    color: #086288;
    font-size: 14.5px;
    margin-bottom: 6px;
}

.flow-node-desc {
    font-size: 12px;
    color: #4D8DBA;
    line-height: 1.5;
}

/* Primary and Outline Buttons */
.stButton button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    transition: all 0.15s ease-in-out !important;
}

.stButton button[kind="primary"] {
    background-color: #086288 !important;
    color: #FFFFFF !important;
    border: 1px solid #086288 !important;
}

.stButton button[kind="primary"]:hover {
    background-color: #4D8DBA !important;
    border-color: #4D8DBA !important;
    color: #FFFFFF !important;
}

.stButton button[kind="secondary"] {
    background-color: #FFFFFF !important;
    color: #086288 !important;
    border: 1px solid #D8ECF8 !important;
}

.stButton button[kind="secondary"]:hover {
    background-color: #EBF5FB !important;
    border-color: #97D7FF !important;
    color: #086288 !important;
}

/* Text area styling */
.stTextArea textarea {
    border: 1px solid #D8ECF8 !important;
    border-radius: 10px !important;
    background-color: #FFFFFF !important;
    color: #0E2A38 !important;
    font-size: 14px !important;
}

.stTextArea textarea:focus {
    border-color: #086288 !important;
    box-shadow: 0 0 0 2px rgba(8, 98, 136, 0.15) !important;
}

/* Section Header Row */
.section-title-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}
</style>
"""

st.html(OCEAN_CREAM_CSS)


# -----------------------------------------------------------------------------
# Clean Title Helper
# -----------------------------------------------------------------------------
def format_doc_title(raw_title: str) -> str:
    """Format markdown titles by stripping out bracket tags like '{ #path-params }'."""
    clean = re.sub(r"\{.*?\}", "", raw_title).strip()
    return clean if clean else raw_title


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "active_question" not in st.session_state:
    st.session_state["active_question"] = ""

if "last_query_response" not in st.session_state:
    st.session_state["last_query_response"] = None

if "last_query_latency" not in st.session_state:
    st.session_state["last_query_latency"] = 0.0

if "feedback_status" not in st.session_state:
    st.session_state["feedback_status"] = None


# -----------------------------------------------------------------------------
# API Helper Functions
# -----------------------------------------------------------------------------
def get_backend_health() -> Optional[Dict[str, Any]]:
    """Fetch health and vector store status from FastAPI."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/health", timeout=3.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_indexed_documents() -> List[Dict[str, Any]]:
    """Fetch all indexed documents from the corpus."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/documents", timeout=5.0)
        if resp.status_code == 200:
            return resp.json().get("documents", [])
    except Exception:
        pass
    return []


def run_rag_query(question: str) -> Optional[Dict[str, Any]]:
    """Submit user query to LangGraph RAG pipeline."""
    try:
        t0 = time.time()
        resp = httpx.post(
            f"{BACKEND_URL}/query",
            json={"question": question},
            timeout=HTTP_TIMEOUT,
        )
        latency = round(time.time() - t0, 2)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state["last_query_response"] = data
            st.session_state["last_query_latency"] = latency
            st.session_state["feedback_status"] = None
            return data
    except Exception as exc:
        st.error(f"Error connecting to backend: {exc}")
    return None


def submit_feedback(query_id: str, rating: str, comment: Optional[str] = None) -> bool:
    """Submit thumbs up/down feedback."""
    try:
        resp = httpx.post(
            f"{BACKEND_URL}/feedback",
            json={"query_id": query_id, "rating": rating, "comment": comment or ""},
            timeout=5.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


def upload_document(uploaded_file) -> Optional[Dict[str, Any]]:
    """Upload a document to /ingest."""
    try:
        files = {"files": (uploaded_file.name, uploaded_file.getvalue(), "text/markdown")}
        resp = httpx.post(
            f"{BACKEND_URL}/ingest",
            files=files,
            timeout=30.0,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        st.error(f"Upload failed: {exc}")
    return None


# -----------------------------------------------------------------------------
# Top Header Banner
# -----------------------------------------------------------------------------
health = get_backend_health()
if health and health.get("status") == "healthy":
    doc_count = health.get("document_count", 5)
    chunk_count = health.get("chunk_count", 101)
    status_markup = f"""
    <div class="header-badges">
        <div class="status-pill-blue">
            <span class="status-dot"></span>
            ChromaDB Ready · {doc_count} Docs · {chunk_count} Chunks
        </div>
        <div class="status-pill-warm">
            <span class="status-dot-green"></span>
            LangGraph Active
        </div>
    </div>
    """
else:
    status_markup = """
    <div class="status-pill-blue" style="background: #FEE8E8; border-color: #F8B4B4; color: #9A2828;">
        <span class="status-dot" style="background: #D45B7A;"></span>
        Connecting to Backend...
    </div>
    """

header_html = f"""
<div class="top-header-banner">
    <div class="top-brand">
        <div class="top-brand-icon">
            <img src="{URI_LOGO}" width="26" height="26" alt="RAG Assistant Logo" />
        </div>
        <div>
            <h1 class="top-brand-title">Express Analytics · Documentation Assistant</h1>
            <p class="top-brand-subtitle">Self-Corrective LangGraph RAG Assistant with ChromaDB & FastAPI</p>
        </div>
    </div>
    {status_markup}
</div>
"""
st.html(header_html)


# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab_qa, tab_docs, tab_ingest, tab_arch = st.tabs([
    "Ask Assistant",
    "Document Corpus",
    "Ingest Documents",
    "LangGraph Workflow",
])


# =============================================================================
# TAB 1: ASK ASSISTANT (Interactive Q&A)
# =============================================================================
with tab_qa:
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
        value=st.session_state["active_question"],
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
                run_rag_query(user_query.strip())
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
        fb_col1, fb_col2, fb_col3 = st.columns([1.2, 1.2, 5.6])
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


# =============================================================================
# TAB 2: DOCUMENT CORPUS (ChromaDB Indexed Technical Docs)
# =============================================================================
with tab_docs:
    docs_title_html = f"""
    <div class="section-title-row">
        <img src="{URI_DOC}" width="20" height="20" alt="Docs" />
        <h3 style="margin: 0; color: #086288; font-size: 18px;">Indexed Technical Corpus</h3>
    </div>
    """
    st.html(docs_title_html)
    st.caption("Technical documents indexed in ChromaDB using RecursiveCharacterTextSplitter (800 target chunk size, 150 overlap).")

    docs = get_indexed_documents()
    if not docs:
        st.info("Loading documents from backend...")
    else:
        # Display in a clean, structured 2-column card grid
        c1, c2 = st.columns(2)
        for idx, doc in enumerate(docs):
            target_col = c1 if idx % 2 == 0 else c2
            doc_id = doc.get("document_id", "")
            raw_title = doc.get("title", doc_id)
            title = format_doc_title(raw_title)
            source = doc.get("source", "")
            chunk_count = doc.get("chunk_count", 0)
            updated_at = doc.get("updated_at", "")[:10]

            with target_col:
                with st.container(border=True):
                    card_header_html = f"""
                    <div style="display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="width: 36px; height: 36px; background: #EBF5FB; border: 1px solid #97D7FF; border-radius: 8px; display: flex; align-items: center; justify-content: center;">
                                <img src="{URI_DOC}" width="18" height="18" alt="Doc" />
                            </div>
                            <div>
                                <h4 style="margin: 0; color: #086288; font-size: 16px;">{title}</h4>
                                <span style="font-size: 11px; color: #4D8DBA;">{doc_id}</span>
                            </div>
                        </div>
                        <span class="doc-badge-pill">{chunk_count} Chunks</span>
                    </div>
                    """
                    st.html(card_header_html)
                    st.caption(f"**Source:** [{source}]({source}) · **Indexed:** {updated_at}")
                    
                    if st.button("Ask a question about this doc", key=f"ask_doc_{doc_id}", width="stretch"):
                        st.session_state["active_question"] = f"Explain how {title} works in FastAPI with a code example."
                        st.rerun()


# =============================================================================
# TAB 3: INGEST DOCUMENTS (Upload & Indexing)
# =============================================================================
with tab_ingest:
    ingest_title_html = f"""
    <div class="section-title-row">
        <img src="{URI_UPLOAD}" width="20" height="20" alt="Ingest" />
        <h3 style="margin: 0; color: #086288; font-size: 18px;">Ingest New Technical Documentation</h3>
    </div>
    """
    st.html(ingest_title_html)
    st.caption("Upload technical documentation files (Markdown .md or Text .txt) to index them into ChromaDB and SQLite.")

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Select technical documentation file to ingest:",
            type=["md", "txt"],
            help="Files are split into 800-token chunks with 150-token overlap and embedded with all-MiniLM-L6-v2.",
        )

        if uploaded_file is not None:
            st.markdown(f"""
            <div style="background: #EBF5FB; border: 1px solid #97D7FF; border-radius: 8px; padding: 10px 14px; margin: 12px 0;">
                <strong style="color: #086288;">File Ready:</strong> <code>{uploaded_file.name}</code> ({uploaded_file.size} bytes)
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Ingest Document to ChromaDB", type="primary"):
                with st.spinner("Processing document chunks, generating embeddings, and storing in ChromaDB..."):
                    result = upload_document(uploaded_file)
                    if result:
                        st.success(f"Successfully ingested {uploaded_file.name}! Indexed {result.get('total_chunks', 0)} chunks.")
                        time.sleep(1)
                        st.rerun()


# =============================================================================
# TAB 4: LANGGRAPH WORKFLOW ARCHITECTURE
# =============================================================================
with tab_arch:
    arch_title_html = f"""
    <div class="section-title-row">
        <img src="{URI_GRAPH}" width="20" height="20" alt="Workflow" />
        <h3 style="margin: 0; color: #086288; font-size: 18px;">LangGraph Self-Corrective RAG Architecture</h3>
    </div>
    """
    st.html(arch_title_html)
    st.caption("A deterministic StateGraph implementing intent classification, semantic retrieval, document relevance grading, and self-corrective query refinement.")

    # Workflow Visual Nodes
    n1, n2, n3, n4 = st.columns(4)
    with n1:
        st.markdown(f"""
        <div class="flow-node">
            <div class="flow-node-title">1. Query Analysis</div>
            <div class="flow-node-desc">Categorizes query intent (how-to, conceptual, api_reference) and rewrites search phrasing for optimal vector search.</div>
        </div>
        """, unsafe_allow_html=True)
    with n2:
        st.markdown(f"""
        <div class="flow-node">
            <div class="flow-node-title">2. Chroma Retrieval</div>
            <div class="flow-node-desc">Performs top-k semantic similarity search against ChromaDB, preserving document and chunk metadata intact.</div>
        </div>
        """, unsafe_allow_html=True)
    with n3:
        st.markdown(f"""
        <div class="flow-node">
            <div class="flow-node-title">3. Relevance Grading</div>
            <div class="flow-node-desc">Evaluates each retrieved chunk using structured LLM output (yes/no). Filters out noise before generation.</div>
        </div>
        """, unsafe_allow_html=True)
    with n4:
        st.markdown(f"""
        <div class="flow-node">
            <div class="flow-node-title">4. Grounded Generation</div>
            <div class="flow-node-desc">Synthesizes a factual answer citing retrieved source chunks explicitly [S1], [S2] without hallucinating.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Self-Correction Explanation with Vector SVG Icon
    with st.container(border=True):
        routing_header_html = f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <img src="{URI_REFRESH}" width="18" height="18" alt="Self-Correction" />
            <h4 style="margin: 0; color: #086288;">Self-Correction & Conditional Routing</h4>
        </div>
        """
        st.html(routing_header_html)
        st.markdown("""
        - **Document Grading Outcome:**
          - **If relevant documents found** &rarr; Routes directly to `generate`.
          - **If NO relevant documents found & `retry_count < MAX_RETRIES (2)`** &rarr; Routes to `rewrite_query` which reformulates the query and re-retrieves.
          - **If `retry_count >= MAX_RETRIES`** &rarr; Gracefully returns an honest response indicating insufficient context, avoiding hallucinations.
        - **Bonus Self-RAG Groundedness:** Verifies the final response against the retrieved chunks to eliminate hallucinations.
        """)

    with st.container(border=True):
        schema_header_html = f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <img src="{URI_CODE}" width="18" height="18" alt="Schema" />
            <h4 style="margin: 0; color: #086288;">Typed Graph State Schema</h4>
        </div>
        """
        st.html(schema_header_html)
        st.code("""class GraphState(TypedDict):
    question: str
    transformed_query: Optional[str]
    query_type: Optional[str]
    documents: List[Document]
    relevant_documents: List[Document]
    generation: Optional[str]
    citations: List[CitationSchema]
    retry_count: int
    web_fallback_used: bool
    status: Literal["answered", "insufficient_context"]""", language="python")
