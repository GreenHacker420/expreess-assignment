import streamlit as st

from frontend.utils.api import get_indexed_documents
from frontend.utils.helpers import format_doc_title
from frontend.utils.icons import URI_DOC


def render_corpus_tab():
    """Render the Document Corpus tab displaying indexed documents in a 2-column grid."""
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
                        <span class="doc-badge-pill" style="background: #FFEFC4; color: #614605; font-size: 11.5px; font-weight: 700; padding: 3px 10px; border-radius: 9999px; border: 1px solid #F3DC9B;">
                            {chunk_count} Chunks
                        </span>
                    </div>
                    """
                    st.html(card_header_html)
                    st.caption(f"**Source:** [{source}]({source}) · **Indexed:** {updated_at}")

                    if st.button("Ask a question about this doc", key=f"ask_doc_{doc_id}", width="stretch"):
                        st.session_state["active_question"] = f"Explain how {title} works in FastAPI with a code example."
                        st.rerun()
