"""Backend API client for FastAPI endpoints."""

import os
import time
from typing import Any, Dict, List, Optional
import httpx

import streamlit as st

def _resolve_backend_url() -> str:
    """Resolve backend URL from Streamlit secrets or OS environment."""
    try:
        if "BACKEND_URL" in st.secrets:
            return str(st.secrets["BACKEND_URL"]).rstrip("/")
    except Exception:
        pass
    return os.getenv("BACKEND_URL", "https://express-rag.nstsdc.org").rstrip("/")

BACKEND_URL: str = _resolve_backend_url()
HTTP_TIMEOUT: float = 60.0



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
            data["_latency"] = latency
            return data
    except Exception:
        pass
    return None


def submit_feedback(query_id: str, rating: str, comment: Optional[str] = None) -> bool:
    """Submit thumbs up/down feedback."""
    try:
        resp = httpx.post(
            f"{BACKEND_URL}/feedback",
            json={"query_id": query_id, "rating": rating, "comment": comment or ""},
            timeout=5.0,
        )
        return resp.is_success
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
        if resp.is_success:
            return resp.json()
    except Exception:
        pass
    return None


def ingest_url(url: str, title: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Ingest a document from URL via /ingest/url."""
    try:
        resp = httpx.post(
            f"{BACKEND_URL}/ingest/url",
            json={"url": url, "title": title or None},
            timeout=30.0,
        )
        if resp.is_success:
            return resp.json()
    except Exception:
        pass
    return None
