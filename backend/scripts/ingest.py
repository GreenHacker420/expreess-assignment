"""CLI command to index documents into ChromaDB and SQLite registry."""

import argparse
import logging
import sys
from pathlib import Path

# Add repository root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.core.config import get_settings
from backend.app.services.database import SQLiteDatabaseService
from backend.app.services.embeddings import SentenceTransformerEmbeddingsService
from backend.app.services.ingestion import IngestionService
from backend.app.services.vector_store import ChromaVectorStoreService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cli_ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest technical documentation into ChromaDB and SQLite.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=get_settings().manifest_path,
        help="Path to corpus manifest JSON file.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional single Markdown or text file to ingest.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear vector store collection before indexing.",
    )

    args = parser.parse_args()
    settings = get_settings()

    logger.info("Initializing embedding model: %s", settings.embedding_model)
    embeddings = SentenceTransformerEmbeddingsService(settings.embedding_model)

    logger.info("Initializing vector store: %s (collection: %s)", settings.chroma_dir, settings.chroma_collection)
    vector_store = ChromaVectorStoreService(settings.chroma_dir, settings.chroma_collection)

    logger.info("Initializing SQLite database: %s", settings.db_path)
    database = SQLiteDatabaseService(settings.db_path)

    if args.reset:
        logger.warning("Resetting vector collection '%s' and SQLite document registry...", settings.chroma_collection)
        vector_store.reset()
        with database._get_connection() as conn:
            conn.execute("DELETE FROM documents;")

    ingestion = IngestionService(
        embeddings=embeddings,
        vector_store=vector_store,
        database=database,
        target_chunk_tokens=settings.chunk_size,
        overlap_tokens=settings.chunk_overlap,
    )

    if args.file:
        file_path = Path(args.file)
        logger.info("Ingesting single file: %s", file_path)
        res = ingestion.ingest_file(file_path)
        logger.info("Ingestion result: %s (doc_id=%s, chunks=%d)", res.status, res.document_id, res.chunk_count)
    else:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            logger.error("Manifest not found: %s. Run scripts/fetch_corpus.py first.", manifest_path)
            sys.exit(1)
        logger.info("Ingesting corpus from manifest: %s", manifest_path)
        results = ingestion.ingest_manifest(manifest_path)
        total_chunks = sum(r.chunk_count for r in results)
        logger.info("Batch ingestion finished: %d documents processed, %d total chunks active.", len(results), total_chunks)
        for r in results:
            print(f"  [{r.status.upper()}] {r.title} ({r.document_id}) -> {r.chunk_count} chunks")


if __name__ == "__main__":
    main()
