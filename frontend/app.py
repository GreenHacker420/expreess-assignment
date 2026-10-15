"""Technical Documentation Assistant — Express Analytics Take-Home Assignment.

A self-corrective Retrieval-Augmented Generation (RAG) assistant for technical
documentation built with LangGraph, ChromaDB, and FastAPI.

Strictly styled with the Pastel Dreamland Adventure palette:
- Solid Clean Background: #FEF6FB (Zero gradients)
- Pink Orchid: #cdb4db
- Pastel Petal: #ffc8dd
- Blush Pop: #ffafcc
- Icy Blue: #bde0fe
- Sky Blue: #a2d2ff
- Deep Plum Text: #2B1540 / #5B3F85
"""

import base64
import os
import time
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

st.set_page_config(
    page_title="Technical Documentation Assistant · LangGraph RAG",
    page_icon="📖",
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
# Base64 Vector SVGs (Crisp Vector Icons Across All Browsers)
# -----------------------------------------------------------------------------
def b64_svg(svg_code: str) -> str:
    encoded = base64.b64encode(svg_code.strip().encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


URI_LOGO = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#5B3F85" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/><polyline points="10 7 14 11 10 15"/></svg>"""
)

URI_SPARKLE = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#5B3F85" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3L12 3z"/></svg>"""
)

URI_DOC = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4A7CC9" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>"""
)

URI_UPLOAD = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#5B3F85" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>"""
)

URI_GRAPH = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4A7CC9" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="3"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="18" r="3"/><line x1="9" y1="6" x2="15" y2="6"/><line x1="6" y1="9" x2="6" y2="15"/><line x1="18" y1="9" x2="18" y2="15"/><line x1="8.5" y1="8.5" x2="15.5" y2="15.5"/></svg>"""
)

URI_CHECK = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2E7D32" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>"""
)

URI_CITE = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#5B3F85" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>"""
)

URI_INFO = b64_svg(
    """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#4A7CC9" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>"""
)


# -----------------------------------------------------------------------------
# Pastel Dreamland Adventure CSS (Solid Clean Background, Zero Gradients)
# -----------------------------------------------------------------------------
PASTEL_DREAMLAND_CSS = """
<style>
/* Clean flat background */
.stApp {
    background-color: #FEF6FB !important;
    background-image: none !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #2B1540 !important;
}

.main .block-container {
    max-width: 1200px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}

/* Header container */
.top-header-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #FFFFFF;
    border: 1px solid #CDB4DB;
    border-radius: 16px;
    padding: 18px 24px;
    margin-bottom: 24px;
    box-shadow: 0 2px 8px rgba(91, 63, 133, 0.04);
}

.top-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.top-brand-icon {
    width: 44px;
    height: 44px;
    background: #FCE8F3;
    border: 1px solid #FFC8DD;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.top-brand-title {
    font-size: 20px;
    font-weight: 700;
    color: #2B1540;
    margin: 0;
    line-height: 1.2;
}

.top-brand-subtitle {
    font-size: 13px;
    color: #6B5EA8;
    margin: 2px 0 0 0;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #E8F4FD;
    border: 1px solid #BDE0FE;
    color: #1A5490;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 12.5px;
    font-weight: 600;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #3A8C6E;
}

/* Tab navigation styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent !important;
    border-bottom: 1px solid #E2D5ED !important;
    padding-bottom: 4px;
    margin-bottom: 20px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2D5ED !important;
    border-radius: 9999px !important;
    padding: 8px 18px !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: #5B3F85 !important;
    transition: all 0.15s ease-in-out;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #FCE8F3 !important;
    border-color: #FFC8DD !important;
    color: #2B1540 !important;
}

.stTabs [aria-selected="true"] {
    background-color: #CDB4DB !important;
    border-color: #B39CC8 !important;
    color: #2B1540 !important;
    font-weight: 700 !important;
}

/* Card styling */
.rag-card {
    background: #FFFFFF;
    border: 1px solid #E2D5ED;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 6px rgba(91, 63, 133, 0.03);
}

.rag-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}

.rag-card-title {
    font-size: 16px;
    font-weight: 700;
    color: #2B1540;
    margin: 0;
}

/* Metric badges row */
.metric-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 14px 0;
}

.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 8px;
    font-size: 12.5px;
    font-weight: 600;
}

.pill-type {
    background: #E8F4FD;
    border: 1px solid #BDE0FE;
    color: #1A5490;
}

.pill-retries {
    background: #FCE8F3;
    border: 1px solid #FFC8DD;
    color: #8B5BA6;
}

.pill-latency {
    background: #F0E8FB;
    border: 1px solid #CDB4DB;
    color: #5B3F85;
}

.pill-grounded {
    background: #EAF7EE;
    border: 1px solid #B8E4C9;
    color: #226B43;
}

/* Citation card */
.citation-box {
    background: #FBF8FD;
    border: 1px solid #E6DCF0;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 10px;
    font-size: 13px;
    color: #3D2258;
}

.citation-tag {
    display: inline-block;
    background: #CDB4DB;
    color: #2B1540;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 11px;
    margin-right: 6px;
}

/* Node flow badge */
.flow-node {
    background: #FFFFFF;
    border: 1px solid #CDB4DB;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}

.flow-node-title {
    font-weight: 700;
    color: #2B1540;
    font-size: 14px;
    margin-bottom: 4px;
}

.flow-node-desc {
    font-size: 12px;
    color: #6B5EA8;
    line-height: 1.4;
}

/* Primary buttons */
.stButton button {
    border-radius: 9999px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    transition: all 0.15s ease-in-out !important;
}

/* Text area styling */
.stTextArea textarea {
    border: 1px solid #CDB4DB !important;
    border-radius: 12px !important;
    background-color: #FFFFFF !important;
    color: #2B1540 !important;
}

.stTextArea textarea:focus {
    border-color: #8B5BA6 !important;
    box-shadow: 0 0 0 2px rgba(139, 91, 166, 0.2) !important;
}
</style>
"""

st.html(PASTEL_DREAMLAND_CSS)


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
    <div class="status-pill">
        <span class="status-dot"></span>
        ChromaDB Ready · {doc_count} Docs · {chunk_count} Chunks
    </div>
    """
else:
    status_markup = """
    <div class="status-pill" style="background: #FEE8E8; border-color: #F8B4B4; color: #9A2828;">
        <span class="status-dot" style="background: #D45B7A;"></span>
        Connecting to Backend...
    </div>
    """

header_html = f"""
<div class="top-header-banner">
    <div class="top-brand">
        <div class="top-brand-icon">
            <img src="{URI_LOGO}" width="24" height="24" alt="RAG Assistant Logo" />
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
    "💬 Ask Assistant",
    "📚 Document Corpus",
    "📥 Ingest Documents",
    "⚡ LangGraph Workflow",
])


# =============================================================================
# TAB 1: ASK ASSISTANT (Interactive Q&A)
# =============================================================================
with tab_qa:
    st.markdown("### Technical Q&A Assistant")
    st.caption("Ask questions about the indexed FastAPI technical documentation. The assistant uses a self-corrective LangGraph pipeline to retrieve, grade, and cite source chunks.")

    # Sample Questions Row
    st.markdown("**Quick Sample Questions:**")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for i, q in enumerate(SAMPLE_QUESTIONS):
        short_label = q.split("?")[0].replace("How do I ", "").replace("What is the ", "")
        if cols[i].button(short_label[:24] + "...", key=f"sample_{i}", width="stretch"):
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

    col_btn, col_clear = st.columns([1, 5])
    ask_clicked = col_btn.button("✨ Ask Assistant", type="primary", width="stretch")
    if col_clear.button("Clear", width="content"):
        st.session_state["active_question"] = ""
        st.session_state["last_query_response"] = None
        st.rerun()

    if ask_clicked:
        if not user_query.strip():
            st.warning("Please enter a technical question to ask.")
        else:
            with st.spinner("Analyzing query, retrieving chunks, and verifying relevance..."):
                run_rag_query(user_query.strip())
                st.session_state["active_question"] = user_query.strip()
                st.rerun()

    # Display Query Results
    res = st.session_state.get("last_query_response")
    latency = st.session_state.get("last_query_latency", 0.0)

    if res:
        st.markdown("---")
        
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
                <img src="{URI_GRAPH}" width="14" height="14" alt="" />
                {retry_text}
            </span>
            <span class="metric-pill pill-latency">
                ⏱️ Latency: {latency}s ({retrieval_attempts} retrieval{'s' if retrieval_attempts > 1 else ''})
            </span>
            <span class="metric-pill pill-grounded">
                <img src="{URI_CHECK}" width="14" height="14" alt="" />
                Status: {status.title()} ({len(citations)} Citations)
            </span>
        </div>
        """
        st.html(metrics_html)

        # Grounded Answer Container
        with st.container(border=True):
            st.markdown("#### Answer")
            st.markdown(res.get("answer", ""))

        # Citations Section
        if citations:
            st.markdown("#### Grounded Citations")
            for cit in citations:
                cit_html = f"""
                <div class="citation-box">
                    <span class="citation-tag">[{cit.get('label', 'S1')}]</span>
                    <strong>{cit.get('title', 'Document')}</strong> — <em>{cit.get('section', 'General')}</em>
                    <div style="font-size: 11.5px; color: #6B5EA8; margin-top: 4px;">
                        Chunk ID: <code>{cit.get('chunk_id', '')}</code> · Source: <a href="{cit.get('source', '')}" target="_blank" style="color: #4A7CC9;">{cit.get('source', '')}</a>
                    </div>
                </div>
                """
                st.html(cit_html)
        else:
            st.info("No specific document chunks were cited for this response.")

        # User Feedback Section
        st.markdown("##### Was this answer helpful?")
        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])
        query_id = res.get("query_id", "")
        
        fb_status = st.session_state.get("feedback_status")

        if fb_status:
            st.success(f"Feedback recorded: {fb_status.title()}! Thank you.")
        else:
            if fb_col1.button("👍 Helpful", key="fb_up", width="stretch"):
                if submit_feedback(query_id, "up", "Helpful answer"):
                    st.session_state["feedback_status"] = "helpful"
                    st.rerun()

            if fb_col2.button("👎 Irrelevant", key="fb_down", width="stretch"):
                if submit_feedback(query_id, "down", "Needs improvement"):
                    st.session_state["feedback_status"] = "needs improvement"
                    st.rerun()


# =============================================================================
# TAB 2: DOCUMENT CORPUS (ChromaDB Indexed Technical Docs)
# =============================================================================
with tab_docs:
    st.markdown("### Indexed Technical Corpus")
    st.caption("These documents are stored in ChromaDB and indexed into semantic chunks using `RecursiveCharacterTextSplitter`.")

    docs = get_indexed_documents()
    if not docs:
        st.info("Loading documents from backend...")
    else:
        for doc in docs:
            doc_id = doc.get("document_id", "")
            title = doc.get("title", doc_id).replace("{ #", "").replace(" }", "")
            source = doc.get("source", "")
            chunk_count = doc.get("chunk_count", 0)
            updated_at = doc.get("updated_at", "")[:10]

            with st.container(border=True):
                c_info, c_action = st.columns([5, 2])
                with c_info:
                    st.markdown(f"#### 📄 {title}")
                    st.caption(f"**Source:** [{source}]({source}) · **Chunks:** {chunk_count} · **Updated:** {updated_at}")
                with c_action:
                    if st.button("Ask about this", key=f"ask_doc_{doc_id}", width="stretch"):
                        st.session_state["active_question"] = f"Explain how {title} works in FastAPI with an example."
                        st.rerun()


# =============================================================================
# TAB 3: INGEST DOCUMENTS (Upload & Indexing)
# =============================================================================
with tab_ingest:
    st.markdown("### Ingest New Technical Documentation")
    st.caption("Upload technical documentation files (Markdown `.md` or Text `.txt`) to index them into ChromaDB and SQLite.")

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Select technical documentation file to ingest:",
            type=["md", "txt"],
            help="Files will be parsed, split into 800-token chunks with 150-token overlap, and embedded via all-MiniLM-L6-v2.",
        )

        if uploaded_file is not None:
            st.write(f"**Filename:** `{uploaded_file.name}` ({uploaded_file.size} bytes)")
            if st.button("🚀 Ingest Document to ChromaDB", type="primary"):
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
    st.markdown("### LangGraph Self-Corrective RAG Architecture")
    st.caption("This assistant implements a deterministic, self-reflective StateGraph with document relevance grading and query rewriting.")

    # Workflow Visual Nodes
    n1, n2, n3, n4 = st.columns(4)
    with n1:
        st.markdown("""
        <div class="flow-node">
            <div class="flow-node-title">1. Query Analysis</div>
            <div class="flow-node-desc">Categorizes query intent (how-to, conceptual, api_reference) and rewrites search phrasing for optimal vector search.</div>
        </div>
        """, unsafe_allow_html=True)
    with n2:
        st.markdown("""
        <div class="flow-node">
            <div class="flow-node-title">2. Chroma Retrieval</div>
            <div class="flow-node-desc">Performs top-k semantic similarity search against ChromaDB, preserving document & chunk metadata intact.</div>
        </div>
        """, unsafe_allow_html=True)
    with n3:
        st.markdown("""
        <div class="flow-node">
            <div class="flow-node-title">3. Relevance Grading</div>
            <div class="flow-node-desc">Evaluates each retrieved chunk using structured LLM output (yes/no). Filters out noise before generation.</div>
        </div>
        """, unsafe_allow_html=True)
    with n4:
        st.markdown("""
        <div class="flow-node">
            <div class="flow-node-title">4. Grounded Generation</div>
            <div class="flow-node-desc">Synthesizes a factual answer citing retrieved source chunks explicitly [S1], [S2] without hallucinating.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Self-Correction Explanation
    with st.container(border=True):
        st.markdown("#### 🔄 Self-Correction & Conditional Routing")
        st.markdown("""
        - **Document Grading Outcome:**
          - **If relevant documents found** &rarr; Routes directly to `generate`.
          - **If NO relevant documents found & `retry_count < MAX_RETRIES (2)`** &rarr; Routes to `rewrite_query` which reformulates the query and re-retrieves.
          - **If `retry_count >= MAX_RETRIES`** &rarr; Gracefully returns an honest response indicating insufficient context, avoiding hallucinations.
        - **Bonus Self-RAG Groundedness:** Verifies the final response against the retrieved chunks to eliminate hallucinations.
        """)

    with st.container(border=True):
        st.markdown("#### 📐 Typed Graph State Schema")
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
