"""Standalone CLI runner to debug queries directly against the LangGraph RAG workflow."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add repository root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.core.config import get_settings
from backend.app.graph.builder import build_rag_graph
from backend.app.graph.state import create_initial_state
from backend.app.services.database import SQLiteDatabaseService
from backend.app.services.embeddings import GeminiEmbeddingsService
from backend.app.services.llm import create_llm_service
from backend.app.services.vector_store import ChromaVectorStoreService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("debug_query")


async def run_query(question: str) -> None:
    settings = get_settings()

    print(f"\n{'='*70}\nQuery: {question}\n{'='*70}\n")

    logger.info("Initializing services...")
    embeddings = GeminiEmbeddingsService(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_embedding_model,
    )
    vector_store = ChromaVectorStoreService(settings.chroma_dir, settings.chroma_collection)
    database = SQLiteDatabaseService(settings.db_path)
    llm = create_llm_service(settings)

    doc_count = database.count_documents()
    chunk_count = vector_store.count_chunks()
    logger.info("Corpus status: %d documents in SQLite, %d chunks in ChromaDB", doc_count, chunk_count)
    if chunk_count == 0:
        logger.warning("Vector store is empty! Did you run 'uv run python scripts/ingest.py'?")

    logger.info("Building LangGraph workflow...")
    app_graph = build_rag_graph(
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store,
        top_k=settings.top_k,
        grading_concurrency=settings.grading_concurrency,
    )

    initial_state = create_initial_state(question, max_retries=settings.max_retries)

    logger.info("Invoking graph...")
    final_state = await app_graph.ainvoke(initial_state)

    print(f"\n{'-'*70}")
    print(f"Status:             {final_state['status'].upper()}")
    print(f"Query Type:         {final_state['query_type']}")
    print(f"Retrieval Attempts: {final_state['retrieval_attempts']}")
    print(f"Retries:            {final_state['retry_count']}")
    print(f"Query History:      {final_state['query_history']}")
    print(f"Retrieved Chunks:   {len(final_state['retrieved_chunks'])}")
    print(f"Relevant Chunks:    {len(final_state['relevant_chunks'])}")
    print(f"{'-'*70}\n")

    print(f"Answer:\n{final_state['answer']}\n")

    if final_state["citations"]:
        print("Citations:")
        for c in final_state["citations"]:
            print(f"  [{c['label']}] {c['title']} > {c['section']} ({c['source']})")
    else:
        print("No citations.")

    print(f"\n{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug a natural language query against the RAG graph.")
    parser.add_argument("question", nargs="?", default="How do I declare optional query parameters in FastAPI?")
    args = parser.parse_args()

    asyncio.run(run_query(args.question))


if __name__ == "__main__":
    main()
