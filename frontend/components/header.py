"""Top Branding and Status Header Component."""

from typing import Any, Dict, Optional
import streamlit as st

from frontend.utils.icons import URI_LOGO


def render_header(health: Optional[Dict[str, Any]]):
    """Render top brand header with system health status pills."""
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
