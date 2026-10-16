"""Unit tests for LangGraph StateGraph, nodes, retry routing, and error boundaries."""

import pytest
from backend.app.graph.builder import build_rag_graph
from backend.app.graph.nodes import route_after_grading
from backend.app.graph.state import create_initial_state
from backend.app.services.llm import (
    GenerationOutput,
    GradeOutput,
    LLMTimeoutError,
    LLMUpstreamError,
    MockLLMService,
)


class FakeWebSearch:
    def __init__(self, results):
        self.results = results
        self.queries = []

    async def search(self, query: str, max_results: int = 5):
        self.queries.append(query)
        return self.results[:max_results]


class MismatchedGradeIdLLM(MockLLMService):
    async def grade_chunk(self, question: str, chunk_id: str, chunk_text: str) -> GradeOutput:
        del question, chunk_id, chunk_text
        return GradeOutput(
            chunk_id="model-returned-the-wrong-id", relevant=True, reason="Relevant content."
        )


@pytest.mark.asyncio
async def test_relevant_chunks_route_to_generation(mock_embeddings, mock_vector_store):
    # Setup 1 stored chunk
    mock_vector_store.add_chunks(
        chunk_ids=["chunk_01"],
        documents=["FastAPI path parameters allow URL pattern matching."],
        metadatas=[
            {
                "chunk_id": "chunk_01",
                "title": "Path Params",
                "section": "Basics",
                "source": "path.md",
            }
        ],
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
async def test_inline_citations_are_resolved_even_if_structured_list_is_partial(
    mock_embeddings, mock_vector_store
):
    texts = ["Path parameters come from the URL.", "Query parameters come from the query string."]
    mock_vector_store.add_chunks(
        chunk_ids=["path", "query"],
        documents=texts,
        metadatas=[
            {"chunk_id": key, "title": key, "section": "Basics", "source": f"{key}.md"}
            for key in ("path", "query")
        ],
        embeddings=[mock_embeddings.embed_query(text) for text in texts],
    )

    class PartialCitationLLM(MockLLMService):
        async def generate_answer(self, question, labeled_context):
            del question, labeled_context
            return GenerationOutput(
                answer="Path values are read from the URL [S1]. Query values are read from the query string [S2].",
                cited_labels=["S1"],
            )

    graph = build_rag_graph(PartialCitationLLM(), mock_embeddings, mock_vector_store)
    result = await graph.ainvoke(create_initial_state("Explain path and query parameters"))

    assert result["status"] == "answered"
    assert {citation["label"] for citation in result["citations"]} == {"S1", "S2"}


@pytest.mark.asyncio
async def test_grader_id_mismatch_does_not_discard_relevant_chunk(
    mock_embeddings, mock_vector_store
):
    mock_vector_store.add_chunks(
        chunk_ids=["stable_chunk_id"],
        documents=["FastAPI path parameters are declared in route paths."],
        metadatas=[
            {
                "chunk_id": "stable_chunk_id",
                "title": "Path Params",
                "section": "Basics",
                "source": "path.md",
            }
        ],
        embeddings=[mock_embeddings.embed_query("path params")],
    )
    graph = build_rag_graph(
        MismatchedGradeIdLLM(generation_labels=["S1"]), mock_embeddings, mock_vector_store
    )

    result = await graph.ainvoke(create_initial_state("How do path params work?"))

    assert result["status"] == "answered"
    assert result["relevant_chunks"][0]["chunk_id"] == "stable_chunk_id"


@pytest.mark.asyncio
async def test_hybrid_retrieval_finds_each_part_of_compound_question(
    mock_embeddings, mock_vector_store
):
    source_docs = [
        ("path", "FastAPI path parameter item_id is declared in the route path."),
        ("query", "FastAPI query parameter q is declared as a function parameter."),
        ("body", "Pydantic BaseModel defines the JSON request body."),
    ]
    mock_vector_store.add_chunks(
        chunk_ids=[item[0] for item in source_docs],
        documents=[item[1] for item in source_docs],
        metadatas=[
            {
                "chunk_id": cid,
                "title": cid,
                "section": "Guide",
                "source": f"fastapi-{cid}.md",
                "document_id": f"fastapi-{cid}",
            }
            for cid, _ in source_docs
        ],
        embeddings=[mock_embeddings.embed_query(text) for _, text in source_docs],
    )
    graph = build_rag_graph(
        MockLLMService(default_relevant=True, generation_labels=["S1", "S2", "S3"]),
        mock_embeddings,
        mock_vector_store,
        top_k=3,
    )

    result = await graph.ainvoke(
        create_initial_state(
            "Can one endpoint combine a path parameter, a query parameter, and a Pydantic BaseModel request body?"
        )
    )

    assert {chunk["chunk_id"] for chunk in result["retrieved_chunks"]} == {"path", "query", "body"}
    assert result["status"] == "answered"


@pytest.mark.asyncio
async def test_irrelevant_chunks_trigger_rewriting_until_exhausted(
    mock_embeddings, mock_vector_store
):
    # Setup stored chunk that will be judged irrelevant
    mock_vector_store.add_chunks(
        chunk_ids=["chunk_irr"],
        documents=["Completely unrelated content about cooking recipes."],
        metadatas=[
            {"chunk_id": "chunk_irr", "title": "Recipes", "section": "Food", "source": "recipes.md"}
        ],
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
async def test_web_fallback_runs_after_corpus_retries_and_cites_url(
    mock_embeddings, mock_vector_store
):
    web = FakeWebSearch(
        [
            {
                "title": "FastAPI External Docs",
                "url": "https://fastapi.tiangolo.com/tutorial/",
                "content": "FastAPI supports dependency injection for shared logic.",
            }
        ]
    )
    llm = MockLLMService(default_relevant=True, generation_labels=["S1"])
    graph = build_rag_graph(llm, mock_embeddings, mock_vector_store, web_search=web)

    result = await graph.ainvoke(
        create_initial_state("How does dependency injection work?", max_retries=0)
    )

    assert len(web.queries) == 1
    assert result["status"] == "answered"
    assert result["web_fallback_used"] is True
    assert result["citations"][0]["source"] == "https://fastapi.tiangolo.com/tutorial/"


@pytest.mark.asyncio
async def test_web_fallback_disabled_fails_closed_without_loop(mock_embeddings, mock_vector_store):
    graph = build_rag_graph(
        MockLLMService(default_relevant=False), mock_embeddings, mock_vector_store
    )

    result = await graph.ainvoke(create_initial_state("Unknown topic", max_retries=0))

    assert result["status"] == "insufficient_context"
    assert result["web_search_attempted"] is True
    assert result["web_fallback_used"] is False
    assert result["retrieval_attempts"] == 1


@pytest.mark.asyncio
async def test_retrieval_pass_preserves_relevant_evidence_from_previous_pass(
    mock_embeddings, mock_vector_store
):
    graph = build_rag_graph(
        MockLLMService(default_relevant=True, generation_labels=["S1"]),
        mock_embeddings,
        mock_vector_store,
    )
    state = create_initial_state("Combine details across documentation", max_retries=0)
    state["relevant_chunks"] = [
        {
            "chunk_id": "earlier-evidence",
            "text": "Previously graded useful context.",
            "source": "first.md",
            "title": "First source",
            "section": "Overview",
            "chunk_index": 0,
            "distance": 0.0,
        }
    ]

    result = await graph.ainvoke(state)

    assert result["status"] == "answered"
    assert result["citations"][0]["chunk_id"] == "earlier-evidence"


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
        metadatas=[
            {"chunk_id": "chunk_1", "title": "Deps", "section": "Intro", "source": "dep.md"}
        ],
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
        metadatas=[
            {"chunk_id": "chunk_1", "title": "Body", "section": "Intro", "source": "body.md"}
        ],
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
        metadatas=[
            {"chunk_id": "chunk_1", "title": "Files", "section": "Intro", "source": "files.md"}
        ],
        embeddings=[mock_embeddings.embed_query("files")],
    )

    llm = MockLLMService(simulate_timeout=True)
    graph = build_rag_graph(llm, mock_embeddings, mock_vector_store)

    state = create_initial_state("How do request files work?")

    # Invariant: Do NOT convert provider failures into false document-irrelevance results
    with pytest.raises(LLMTimeoutError):
        await graph.ainvoke(state)


@pytest.mark.asyncio
async def test_grounding_failure_causes_one_regeneration_then_accepts(
    mock_embeddings, mock_vector_store
):
    mock_vector_store.add_chunks(
        chunk_ids=["grounded_chunk"],
        documents=["FastAPI uses Depends to declare a dependency."],
        metadatas=[{"title": "Dependencies", "section": "Depends", "source": "dependencies.md"}],
        embeddings=[mock_embeddings.embed_query("dependency")],
    )
    llm = MockLLMService()
    llm.grounding_failures_remaining = 1
    graph = build_rag_graph(llm, mock_embeddings, mock_vector_store)

    result = await graph.ainvoke(create_initial_state("How do I declare a dependency?"))

    assert result["status"] == "answered"
    assert result["generation_attempts"] == 2
    assert result["grounding_result"]["supported"] is True


@pytest.mark.asyncio
async def test_persistently_ungrounded_answer_fails_closed(mock_embeddings, mock_vector_store):
    mock_vector_store.add_chunks(
        chunk_ids=["ungrounded_chunk"],
        documents=["FastAPI uses Depends to declare a dependency."],
        metadatas=[{"title": "Dependencies", "section": "Depends", "source": "dependencies.md"}],
        embeddings=[mock_embeddings.embed_query("dependency")],
    )
    llm = MockLLMService(grounding_supported=False)
    graph = build_rag_graph(llm, mock_embeddings, mock_vector_store)

    result = await graph.ainvoke(create_initial_state("How do I declare a dependency?"))

    assert result["status"] == "insufficient_context"
    assert result["generation_attempts"] == 2
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_incomplete_generation_triggers_bounded_crag_retrieval(
    mock_embeddings, mock_vector_store
):
    mock_vector_store.add_chunks(
        chunk_ids=["incomplete_chunk"],
        documents=["FastAPI supports path parameters."],
        metadatas=[{"title": "Paths", "section": "Overview", "source": "paths.md"}],
        embeddings=[mock_embeddings.embed_query("path")],
    )
    graph = build_rag_graph(
        MockLLMService(has_sufficient_context=False), mock_embeddings, mock_vector_store
    )

    result = await graph.ainvoke(
        create_initial_state("Explain this multi-part question", max_retries=1)
    )

    assert result["status"] == "insufficient_context"
    assert result["retry_count"] == 1
    assert result["retrieval_attempts"] == 2
    assert result["generation_attempts"] == 0


@pytest.mark.asyncio
async def test_crag_retry_can_recover_after_first_incomplete_draft(
    mock_embeddings, mock_vector_store
):
    from backend.app.services.llm import GenerationOutput

    class RecoveringLLM(MockLLMService):
        def __init__(self):
            super().__init__()
            self.generation_calls = 0

        async def generate_answer(self, question, labeled_context):
            del question, labeled_context
            self.generation_calls += 1
            return GenerationOutput(
                answer="FastAPI supports this behavior. [S1]",
                cited_labels=["S1"],
                has_sufficient_context=self.generation_calls > 1,
            )

    mock_vector_store.add_chunks(
        chunk_ids=["recover_chunk"],
        documents=["FastAPI supports path parameters."],
        metadatas=[{"title": "Paths", "section": "Overview", "source": "paths.md"}],
        embeddings=[mock_embeddings.embed_query("path")],
    )
    llm = RecoveringLLM()
    graph = build_rag_graph(llm, mock_embeddings, mock_vector_store, top_k=1)

    result = await graph.ainvoke(create_initial_state("How do paths work?", max_retries=1))

    assert result["status"] == "answered"
    assert result["retry_count"] == 1
    assert result["generation_attempts"] == 1


@pytest.mark.asyncio
async def test_query_decomposition_merges_distinct_document_evidence(mock_embeddings):
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda

    class PerQueryVectorStore:
        def __init__(self):
            self.calls = 0

        def as_retriever(self, embeddings, top_k=5):
            del embeddings, top_k

            def retrieve(_query):
                self.calls += 1
                chunk_id = f"aspect_{self.calls}"
                return [
                    Document(
                        page_content=f"Evidence for aspect {self.calls}.",
                        metadata={
                            "chunk_id": chunk_id,
                            "title": chunk_id,
                            "section": "Overview",
                            "source": f"{chunk_id}.md",
                        },
                    )
                ]

            return RunnableLambda(retrieve)

    vector_store = PerQueryVectorStore()
    llm = MockLLMService(retrieval_queries=["path parameter validation", "file upload handling"])
    graph = build_rag_graph(llm, mock_embeddings, vector_store, top_k=1)

    result = await graph.ainvoke(
        create_initial_state("Compare path parameters and file upload handling")
    )

    assert vector_store.calls == 3  # primary query and two focused searches
    assert {chunk["chunk_id"] for chunk in result["retrieved_chunks"]} == {
        "aspect_1",
        "aspect_2",
        "aspect_3",
    }


def test_route_after_grading_logic():
    # Any relevant chunks -> generate
    s1 = {"relevant_chunks": [{"id": 1}], "retry_count": 0, "max_retries": 2}
    assert route_after_grading(s1) == "generate"

    # No relevant chunks and retries < max -> rewrite_query
    s2 = {"relevant_chunks": [], "retry_count": 0, "max_retries": 2}
    assert route_after_grading(s2) == "rewrite_query"

    s3 = {"relevant_chunks": [], "retry_count": 1, "max_retries": 2}
    assert route_after_grading(s3) == "rewrite_query"

    # No relevant chunks after corpus retries -> one optional web attempt
    s4 = {"relevant_chunks": [], "retry_count": 2, "max_retries": 2}
    assert route_after_grading(s4) == "web_search"
    s5 = {**s4, "web_search_attempted": True}
    assert route_after_grading(s5) == "fallback"


@pytest.mark.asyncio
async def test_sqlite_checkpointer_preserves_conversation_history(mock_embeddings, mock_vector_store):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    saver_cm = AsyncSqliteSaver.from_conn_string(":memory:")
    checkpointer = await saver_cm.__aenter__()
    await checkpointer.setup()

    mock_vector_store.add_chunks(
        chunk_ids=["chunk_1"],
        documents=["Path parameters are declared with format strings in FastAPI."],
        metadatas=[
            {
                "chunk_id": "chunk_1",
                "title": "Path Params",
                "section": "Path",
                "source": "path.md",
            }
        ],
        embeddings=[mock_embeddings.embed_query("query")],
    )

    try:
        llm = MockLLMService(default_relevant=True, generation_labels=["S1"])
        graph = build_rag_graph(
            llm=llm,
            embeddings=mock_embeddings,
            vector_store=mock_vector_store,
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "session-test-mem"}}

        # Turn 1
        state_t1 = create_initial_state("How do path parameters work in FastAPI?")
        res_t1 = await graph.ainvoke(state_t1, config=config)
        assert res_t1["status"] == "answered"
        assert len(res_t1["chat_history"]) == 2  # user + assistant

        # Turn 2: follow-up referencing "them"
        state_t2 = create_initial_state("Can I add validation to them?")
        res_t2 = await graph.ainvoke(state_t2, config=config)
        assert res_t2["status"] == "answered"
        # 4 messages across 2 turns preserved in thread state
        assert len(res_t2["chat_history"]) == 4
        assert res_t2["chat_history"][0]["role"] == "user"
        assert res_t2["chat_history"][1]["role"] == "assistant"
        assert res_t2["chat_history"][2]["role"] == "user"
        assert res_t2["chat_history"][3]["role"] == "assistant"

        # Verify query analysis contextualized the query incorporating Turn 1
        assert "path parameters" in res_t2["retrieval_query"].lower()
    finally:
        await saver_cm.__aexit__(None, None, None)

