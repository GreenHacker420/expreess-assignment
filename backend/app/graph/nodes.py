"""Implementation of all LangGraph nodes for the self-corrective RAG pipeline."""

import asyncio
import logging
import re
from typing import Any, Dict, List

from backend.app.graph.state import CitationRecord, GradeRecord, GraphState
from backend.app.services.embeddings import EmbeddingsService
from backend.app.services.llm import LLMOperationalError, LLMService
from backend.app.services.vector_store import VectorStoreService

logger = logging.getLogger("graph_nodes")

FALLBACK_ANSWER = (
    "I am unable to answer this question based on the indexed FastAPI technical documentation. "
    "The available documentation does not contain sufficient verified evidence to support a factual answer."
)


class RAGNodeFactory:
    """Factory creating bound graph node functions with injected services."""

    def __init__(
        self,
        llm: LLMService,
        embeddings: EmbeddingsService,
        vector_store: VectorStoreService,
        top_k: int = 5,
        grading_concurrency: int = 3,
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.top_k = top_k
        self.grading_concurrency = grading_concurrency
        self._grade_semaphore = asyncio.Semaphore(grading_concurrency)

    async def analyze_query(self, state: GraphState) -> Dict[str, Any]:
        """Node 1: Rewrites the question for retrieval and classifies intent."""
        question = state["original_question"]
        logger.info("[query_id=%s] Node: analyze_query for: '%s'", state["query_id"], question)

        analysis = await self.llm.analyze_query(question)
        retrieval_query = analysis.transformed_query.strip()
        query_type = analysis.query_type

        return {
            "retrieval_query": retrieval_query,
            "query_type": query_type,
            "query_history": [retrieval_query],
        }

    async def retrieve(self, state: GraphState) -> Dict[str, Any]:
        """Node 2: Executes top-k similarity search against vector store."""
        query = state["retrieval_query"]
        attempts = state["retrieval_attempts"] + 1
        logger.info(
            "[query_id=%s] Node: retrieve (attempt %d) for query: '%s'",
            state["query_id"], attempts, query
        )

        query_emb = self.embeddings.embed_query(query)
        raw_results = self.vector_store.search(query_emb, top_k=self.top_k)

        retrieved_chunks = []
        for r in raw_results:
            meta = r.get("metadata", {})
            retrieved_chunks.append({
                "chunk_id": r.get("chunk_id", ""),
                "text": r.get("text", ""),
                "source": meta.get("source", ""),
                "title": meta.get("title", ""),
                "section": meta.get("section", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "distance": r.get("distance", 0.0),
            })

        logger.info(
            "[query_id=%s] Retrieved %d chunks from vector store.",
            state["query_id"], len(retrieved_chunks)
        )

        # Invariant: Replace retrieved_chunks, clear old grades and relevant_chunks
        return {
            "retrieved_chunks": retrieved_chunks,
            "relevant_chunks": [],
            "grades": [],
            "retrieval_attempts": attempts,
        }

    async def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """Node 3: Evaluates retrieved chunks for factual relevance using structured LLM outputs."""
        question = state["original_question"]
        chunks = state["retrieved_chunks"]
        logger.info(
            "[query_id=%s] Node: grade_documents evaluating %d chunks.",
            state["query_id"], len(chunks)
        )

        if not chunks:
            return {"relevant_chunks": [], "grades": []}

        async def _grade_single(chunk: Dict[str, Any]) -> GradeRecord:
            async with self._grade_semaphore:
                out = await self.llm.grade_chunk(
                    question=question,
                    chunk_id=chunk["chunk_id"],
                    chunk_text=chunk["text"],
                )
                return GradeRecord(
                    chunk_id=out.chunk_id,
                    relevant=out.relevant,
                    reason=out.reason,
                )

        # Concurrently grade chunks up to semaphore limit
        grade_results: List[GradeRecord] = await asyncio.gather(
            *[_grade_single(c) for c in chunks]
        )

        grades_dict = [g.model_dump() for g in grade_results]
        relevance_by_id = {g.chunk_id: g.relevant for g in grade_results}

        relevant_chunks = [c for c in chunks if relevance_by_id.get(c["chunk_id"], False)]

        logger.info(
            "[query_id=%s] Grading complete: %d/%d chunks judged relevant.",
            state["query_id"], len(relevant_chunks), len(chunks)
        )

        return {
            "grades": grades_dict,
            "relevant_chunks": relevant_chunks,
        }

    async def rewrite_query(self, state: GraphState) -> Dict[str, Any]:
        """Node 4: Produces meaning-preserving alternative query when retrieval yields no relevant chunks."""
        orig_q = state["original_question"]
        history = state["query_history"]
        new_retry_count = state["retry_count"] + 1

        logger.info(
            "[query_id=%s] Node: rewrite_query (retry_count %d/%d)",
            state["query_id"], new_retry_count, state["max_retries"]
        )

        rewritten = await self.llm.rewrite_query(orig_q, history)
        rewritten = rewritten.strip()

        return {
            "retrieval_query": rewritten,
            "retry_count": new_retry_count,
            "query_history": history + [rewritten],
        }

    async def generate(self, state: GraphState) -> Dict[str, Any]:
        """Node 5: Synthesizes factual answer citing explicit source labels."""
        orig_q = state["original_question"]
        rel_chunks = state["relevant_chunks"]
        logger.info(
            "[query_id=%s] Node: generate using %d relevant chunks.",
            state["query_id"], len(rel_chunks)
        )

        if not rel_chunks:
            return await self.fallback(state)

        # Label contexts [S1], [S2], ...
        label_map: Dict[str, Dict[str, Any]] = {}
        context_blocks: List[str] = []

        for idx, chunk in enumerate(rel_chunks, start=1):
            label = f"S{idx}"
            label_map[label] = chunk
            context_blocks.append(
                f"[{label}] Document: {chunk['title']} | Section: {chunk['section']}\n"
                f"{chunk['text']}"
            )

        labeled_context = "\n\n---\n\n".join(context_blocks)

        gen_output = await self.llm.generate_answer(orig_q, labeled_context)

        # Invariant: If model indicates insufficient context, fallback safely
        if not gen_output.has_sufficient_context or not gen_output.answer.strip():
            logger.warning("[query_id=%s] Model signaled insufficient context.", state["query_id"])
            return await self.fallback(state)

        # Invariant: Reject uncited factual answers
        cited_labels = [lbl.strip().strip("[]") for lbl in gen_output.cited_labels]
        # Also scan answer for cited labels in text if list was omitted
        if not cited_labels:
            found_in_text = re.findall(r"\[(S\d+)\]", gen_output.answer)
            cited_labels = list(dict.fromkeys(found_in_text))

        if not cited_labels:
            logger.warning("[query_id=%s] Rejecting uncited factual answer.", state["query_id"])
            return await self.fallback(state)

        # Invariant: Reject unknown labels or fabricated identifiers
        citations: List[Dict[str, Any]] = []
        for lbl in cited_labels:
            if lbl not in label_map:
                logger.warning("[query_id=%s] Unknown citation label '%s'. Triggering fallback.", state["query_id"], lbl)
                return await self.fallback(state)
            c = label_map[lbl]
            citations.append({
                "label": lbl,
                "chunk_id": c["chunk_id"],
                "title": c["title"],
                "section": c["section"],
                "source": c["source"],
            })

        return {
            "answer": gen_output.answer.strip(),
            "citations": citations,
            "status": "answered",
        }

    async def fallback(self, state: GraphState) -> Dict[str, Any]:
        """Node 6: Graceful deterministic fallback when evidence is insufficient."""
        logger.info("[query_id=%s] Node: fallback executed.", state["query_id"])
        return {
            "answer": FALLBACK_ANSWER,
            "citations": [],
            "status": "insufficient_context",
        }


def route_after_grading(state: GraphState) -> str:
    """Conditional routing function executed after grade_documents."""
    has_relevant = len(state.get("relevant_chunks", [])) > 0
    retries = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if has_relevant:
        return "generate"
    if retries < max_retries:
        return "rewrite_query"
    return "fallback"
