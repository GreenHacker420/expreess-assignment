"""FastAPI application factory, lifecycle, and middleware configuration."""

from contextlib import asynccontextmanager
import logging
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.documents import router as documents_router
from backend.app.api.feedback import router as feedback_router
from backend.app.api.ingest import router as ingest_router
from backend.app.api.query import router as query_router
from backend.app.core.config import Settings, get_settings
from backend.app.graph.builder import build_rag_graph
from backend.app.schemas import HealthResponse
from backend.app.services.database import SQLiteDatabaseService
from backend.app.services.embeddings import GeminiEmbeddingsService
from backend.app.services.ingestion import IngestionService
from backend.app.services.llm import create_llm_service
from backend.app.services.vector_store import ChromaVectorStoreService

logger = logging.getLogger("backend_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context initializing services on startup."""
    if not hasattr(app.state, "settings") or app.state.settings is None:
        app.state.settings = get_settings()

    settings: Settings = app.state.settings
    settings.ensure_directories()

    # Lazy-initialize production services if not already injected (e.g., during tests)
    if not hasattr(app.state, "database"):
        app.state.database = SQLiteDatabaseService(settings.db_path)

    if not hasattr(app.state, "embeddings"):
        app.state.embeddings = GeminiEmbeddingsService(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_embedding_model,
        )

    if not hasattr(app.state, "vector_store"):
        app.state.vector_store = ChromaVectorStoreService(
            persist_dir=settings.chroma_dir,
            collection_name=settings.chroma_collection,
        )

    if not hasattr(app.state, "ingestion"):
        app.state.ingestion = IngestionService(
            embeddings=app.state.embeddings,
            vector_store=app.state.vector_store,
            database=app.state.database,
            target_chunk_tokens=settings.chunk_size,
            overlap_tokens=settings.chunk_overlap,
        )

    if not hasattr(app.state, "llm"):
        app.state.llm = create_llm_service(settings)

    if not hasattr(app.state, "graph"):
        app.state.graph = build_rag_graph(
            llm=app.state.llm,
            embeddings=app.state.embeddings,
            vector_store=app.state.vector_store,
            top_k=settings.top_k,
            grading_concurrency=settings.grading_concurrency,
        )

    logger.info("Application initialized successfully.")
    yield
    logger.info("Application shutting down.")


def create_app(custom_settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure a FastAPI application instance."""
    settings = custom_settings or get_settings()

    app = FastAPI(
        title="RAG Technical Documentation Assistant",
        description="Self-Corrective RAG Technical Documentation Assistant built with LangGraph and FastAPI",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.settings = settings

    origins = settings.cors_origins if settings.cors_origins else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check():
        db = getattr(app.state, "database", None)
        vstore = getattr(app.state, "vector_store", None)
        doc_count = db.count_documents() if db else 0
        chunk_count = vstore.count_chunks() if vstore else 0
        v_status = "ready" if vstore is not None else "uninitialized"
        return HealthResponse(
            status="healthy",
            vector_store=v_status,
            document_count=doc_count,
            chunk_count=chunk_count,
        )

    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(documents_router)
    app.include_router(feedback_router)

    return app


app = create_app()
