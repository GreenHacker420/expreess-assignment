"""Unit tests for LangGraph StateGraph, nodes, retry routing, and error boundaries."""

import pytest
from backend.app.graph.builder import build_rag_graph
from backend.app.graph.nodes import route_after_grading
from backend.app.graph.state import create_initial_state
from backend.app.services.llm import LLMTimeoutError, LLMUpstreamError, MockLLMService


@pytest.mark.asyncio
async def test_relevant_chunks_route_to_generation(mock_embeddings, mock_vector_store):
    # Setup 1 stored chunk
    mock_vector_store.add_chunks(
        chunk_ids=["chunk_01"],
        documents=["FastAPI path parameters allow URL pattern matching."],
        metadatas=[{"chunk_id": "chunk_01", "title": "Path Params", "section": "Basics", "source": "path.md"}],
        embeddings=[mock_embeddings.embed_query("query")],
    )

    llm = MockLLMService(default_relevant=True, generation_labels=["S1"])
    graph = build_rag_graph(llm, mock_embeddings, mock_vector_store)

    state = create_initial_state("How do path params work?")
    result = await graph.ainvoke(state)

    assert result["status"] == "answered"
    assert result["retrieval_attempts"] == 1
    assert result["retry_count"] == 0
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "chunk_01"


@pytest.mark.asyncio
async def test_irrelevant_chunks_trigger_rewriting_until_exhausted(mock_embeddings, mock_vector_store):
    # Setup stored chunk that will be judged irrelevant
    mock_vector_store.add_chunks(
        chunk_ids=["chunk_irr"],
        documents=["Completely unrelated content about cooking recipes."],
        metadatas=[{"chunk_id": "chunk_irr", "title": "Recipes", "section": "Food", "source": "recipes.md"}],
        embeddings=[mock_embeddings.embed_query("recipes")],
    )

    # Grader will mark irrelevant (relevant=False)
    llm = MockLLMService(default_relevant=False)
    graph = build_rag_graph(llm, mock_embeddings, mock_vector_store)

    state = create_initial_state("How do path parameters work in FastAPI?", max_retries=2)
    result = await graph.ainvoke(state)

    # Invariant: Exhausted retries return fallback with status insufficient_context
    assert result["status"] == "insufficient_context"
    # Invariant: Exactly 3 total retrieval attempts (1 initial + 2 rewrites)
    assert result["retrieval_attempts"] == 3
    assert result["retry_count"] == 2
    assert len(result["citations"]) == 0
    assert "unable to answer" in result["answer"].lower()


@pytest.mark.asyncio
async def test_new_query_resets_counters_and_context(mock_embeddings, mock_vector_store):
    state1 = create_initial_state("First question", max_retries=2)
    state1["retry_count"] = 2
    state1["retrieval_attempts"] = 3
    state1["relevant_chunks"] = [{"some": "old_data"}]

    # A new request starts completely fresh
    state2 = create_initial_state("Second question", max_retries=2)
    assert state2["retry_count"] == 0
    assert state2["retrieval_attempts"] == 0
    assert len(state2["relevant_chunks"]) == 0
    assert state2["query_id"] != state1["query_id"]


@pytest.mark.asyncio
async def test_mixed_grading_filters_correctly(mock_embeddings, mock_vector_store):
    mock_vector_store.add_chunks(
        chunk_ids=["c_rel", "c_irr"],
        documents=["FastAPI query parameters guide.", "Random unrelated paragraph."],
        metadatas=[
            {"chunk_id": "c_rel", "title": "Query Params", "section": "Basics", "source": "qp.md"},
            {"chunk_id": "c_irr", "title": "Irrelevant", "section": "Random", "source": "irr.md"},
        ],
        embeddings=[mock_embeddings.embed_query("q"), mock_embeddings.embed_query("irr")],
    )

    llm = MockLLMService()
    # Explicitly mark c_rel as True, c_irr as False
    llm.chunk_relevance_map = {"c_rel": True, "c_irr": False}

    graph = build_rag_graph(llm, mock_embeddings, mock_vector_store)
    state = create_initial_state("How do query parameters work?")
    result = await graph.ainvoke(state)

    assert result["status"] == "answered"
    assert len(result["relevant_chunks"]) == 1
    assert result["relevant_chunks"][0]["chunk_id"] == "c_rel"


@pytest.mark.asyncio
async def test_invalid_citation_label_triggers_safe_fallback(mock_embeddings, mock_vector_store):
    mock_vector_store.add_chunks(
        chunk_ids=["chunk_1"],
        documents=["FastAPI dependency injection system."],
        metadatas=[{"chunk_id": "chunk_1", "title": "Deps", "section": "Intro", "source": "dep.md"}],
        embeddings=[mock_embeddings.embed_query("deps")],
    )

    # Model hallucinates a label that wasn't provided, e.g. [S99]
    llm = MockLLMService(default_relevant=True, generation_labels=["S99"])
    graph = build_rag_graph(llm, mock_embeddings, mock_vector_store)

    state = create_initial_state("How do dependencies work?")
    result = await graph.ainvoke(state)

    # Invariant: Unknown citation label triggers safe fallback
    assert result["status"] == "insufficient_context"
    assert len(result["citations"]) == 0


@pytest.mark.asyncio
async def test_uncited_factual_answer_triggers_safe_fallback(mock_embeddings, mock_vector_store):
    mock_vector_store.add_chunks(
        chunk_ids=["chunk_1"],
        documents=["FastAPI request body guide."],
        metadatas=[{"chunk_id": "chunk_1", "title": "Body", "section": "Intro", "source": "body.md"}],
        embeddings=[mock_embeddings.embed_query("body")],
    )

    # Model generates answer but provides NO citation labels
    llm = MockLLMService(default_relevant=True, generation_labels=[])
    graph = build_rag_graph(llm, mock_embeddings, mock_vector_store)

    state = create_initial_state("How do request bodies work?")
    result = await graph.ainvoke(state)

    # Invariant: Reject uncited factual answers
    assert result["status"] == "insufficient_context"
    assert len(result["citations"]) == 0


@pytest.mark.asyncio
async def test_provider_timeout_is_treated_as_operational_error(mock_embeddings, mock_vector_store):
    mock_vector_store.add_chunks(
        chunk_ids=["chunk_1"],
        documents=["FastAPI request files guide."],
        metadatas=[{"chunk_id": "chunk_1", "title": "Files", "section": "Intro", "source": "files.md"}],
        embeddings=[mock_embeddings.embed_query("files")],
    )

    llm = MockLLMService(simulate_timeout=True)
    graph = build_rag_graph(llm, mock_embeddings, mock_vector_store)

    state = create_initial_state("How do request files work?")

    # Invariant: Do NOT convert provider failures into false document-irrelevance results
    with pytest.raises(LLMTimeoutError):
        await graph.ainvoke(state)


def test_route_after_grading_logic():
    # Any relevant chunks -> generate
    s1 = {"relevant_chunks": [{"id": 1}], "retry_count": 0, "max_retries": 2}
    assert route_after_grading(s1) == "generate"

    # No relevant chunks and retries < max -> rewrite_query
    s2 = {"relevant_chunks": [], "retry_count": 0, "max_retries": 2}
    assert route_after_grading(s2) == "rewrite_query"

    s3 = {"relevant_chunks": [], "retry_count": 1, "max_retries": 2}
    assert route_after_grading(s3) == "rewrite_query"

    # No relevant chunks and retries >= max -> fallback
    s4 = {"relevant_chunks": [], "retry_count": 2, "max_retries": 2}
    assert route_after_grading(s4) == "fallback"
