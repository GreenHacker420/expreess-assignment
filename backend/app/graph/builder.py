"""LangGraph workflow assembly and StateGraph compilation."""

from typing import Any, Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.app.graph.nodes import RAGNodeFactory, route_after_grading
from backend.app.graph.state import GraphState
from backend.app.services.embeddings import EmbeddingsService
from backend.app.services.llm import LLMService
from backend.app.services.vector_store import VectorStoreService


def build_rag_graph(
    llm: LLMService,
    embeddings: EmbeddingsService,
    vector_store: VectorStoreService,
    top_k: int = 5,
    grading_concurrency: int = 3,
    checkpointer: Optional[Any] = None,
):
    """Build and compile the self-corrective RAG StateGraph workflow."""
    factory = RAGNodeFactory(
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store,
        top_k=top_k,
        grading_concurrency=grading_concurrency,
    )

    workflow = StateGraph(GraphState)

    # Register Nodes
    workflow.add_node("analyze_query", factory.analyze_query)
    workflow.add_node("retrieve", factory.retrieve)
    workflow.add_node("grade_documents", factory.grade_documents)
    workflow.add_node("rewrite_query", factory.rewrite_query)
    workflow.add_node("generate", factory.generate)
    workflow.add_node("fallback", factory.fallback)

    # Register Direct Edges
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    # Register Conditional Edge after grade_documents
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )

    # Compile with checkpointer support for state persistence and time-travel debugging
    return workflow.compile(checkpointer=checkpointer)
