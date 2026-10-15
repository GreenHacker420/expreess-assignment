import time
import streamlit as st

from frontend.utils.api import upload_document
from frontend.utils.icons import URI_UPLOAD


def render_ingest_tab():
    """Render the Ingest Documents tab for file upload and ChromaDB indexing."""
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
                    else:
                        st.error("Failed to ingest document. Ensure the backend server is reachable.")
