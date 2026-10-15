import os
import streamlit as st

from frontend.components.corpus_tab import render_corpus_tab
from frontend.components.header import render_header
from frontend.components.ingest_tab import render_ingest_tab
from frontend.components.qa_tab import render_qa_tab
from frontend.components.workflow_tab import render_workflow_tab
from frontend.styles.theme import CUSTOM_CSS
from frontend.utils.api import get_backend_health

FAVICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.svg")

st.set_page_config(
    page_title="Technical Documentation Assistant · LangGraph RAG",
    page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply sleek modern Ocean Azure theme & fixed tab styles
st.html(CUSTOM_CSS)

# Initialize Session State Variables
if "active_question" not in st.session_state:
    st.session_state["active_question"] = ""

if "last_query_response" not in st.session_state:
    st.session_state["last_query_response"] = None

if "last_query_latency" not in st.session_state:
    st.session_state["last_query_latency"] = 0.0

if "feedback_status" not in st.session_state:
    st.session_state["feedback_status"] = None

# Render Top Brand Header with Live Health / Chroma Status
health = get_backend_health()
render_header(health)

# Render Application Tabs
tab_qa, tab_docs, tab_ingest, tab_arch = st.tabs([
    "Ask Assistant",
    "Document Corpus",
    "Ingest Documents",
    "LangGraph Workflow",
])

with tab_qa:
    render_qa_tab()

with tab_docs:
    render_corpus_tab()

with tab_ingest:
    render_ingest_tab()

with tab_arch:
    render_workflow_tab()
