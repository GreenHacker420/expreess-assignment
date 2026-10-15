import streamlit as st

from frontend.utils.icons import URI_CODE, URI_GRAPH, URI_REFRESH


def render_workflow_tab():
    """Render the LangGraph architecture and state schema tab."""
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
            <div class="flow-node-desc">Performs top-k semantic similarity search against ChromaDB, preserving document and chunk metadata intact.</div>
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
    query_id: str
    original_question: str
    retrieval_query: str
    query_type: str
    query_history: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    relevant_chunks: List[Dict[str, Any]]
    grades: List[Dict[str, Any]]
    retry_count: int
    max_retries: int
    retrieval_attempts: int
    answer: str
    citations: List[Dict[str, Any]]
    status: Literal["answered", "insufficient_context"]""", language="python")
