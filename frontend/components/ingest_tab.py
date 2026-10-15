"""Document Ingestion Tab Component."""

import time
import streamlit as st

from frontend.utils.api import ingest_url, upload_document
from frontend.utils.icons import URI_DOC, URI_UPLOAD


def render_ingest_tab():
    """Render the Ingest Documents tab for file upload and URL indexing."""
    ingest_title_html = f"""
    <div class="section-title-row">
        <img src="{URI_UPLOAD}" width="20" height="20" alt="Ingest" />
        <h3 style="margin: 0; color: #086288; font-size: 18px;">Ingest New Technical Documentation</h3>
    </div>
    """
    st.html(ingest_title_html)
    st.caption("Add documentation to ChromaDB using semantic chunking (800-token chunks with 150 overlap) via file upload or external URL.")

    # Ingestion Mode Selection via Segmented Control / Radio
    mode = st.radio(
        "Ingestion Source:",
        options=["File Upload (.md / .txt)", "External Web / GitHub URL"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "File Upload (.md / .txt)":
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
                        else:
                            st.error("Failed to ingest document. Ensure the backend server is reachable.")

    else:
        with st.container(border=True):
            st.markdown("<p style='font-size: 13.5px; color: #4D8DBA; margin-bottom: 12px;'>Enter a direct URL to a Markdown or plain text document (e.g. GitHub raw URL, docs page):</p>", unsafe_allow_html=True)
            doc_url = st.text_input(
                "Document URL:",
                placeholder="https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/security/index.md",
            )
            custom_title = st.text_input("Optional Document Title:", placeholder="e.g. FastAPI Security & OAuth2")

            if st.button("Fetch and Ingest from URL", type="primary"):
                if not doc_url.strip():
                    st.warning("Please provide a valid document URL.")
                else:
                    with st.spinner("Fetching document from URL and indexing chunks into ChromaDB..."):
                        result = ingest_url(doc_url.strip(), custom_title.strip() or None)
                        if result:
                            st.success(f"Successfully fetched and indexed {result.get('total_chunks', 0)} chunks from URL!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to fetch or index document from URL. Verify the URL is publicly reachable.")
