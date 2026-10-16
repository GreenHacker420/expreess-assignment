"""CLI command to index documents into ChromaDB and SQLite registry."""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import dotenv_values

# Add repository root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.core.config import Settings, get_settings
from backend.app.services.database import SQLiteDatabaseService
from backend.app.services.embeddings import GeminiEmbeddingsService
from backend.app.services.ingestion import IngestionService
from backend.app.services.vector_store import ChromaVectorStoreService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cli_ingest")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest technical documentation into ChromaDB and SQLite."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to corpus manifest JSON file.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Read provider credentials from this file, overriding runtime key variables.",
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
    if args.env_file:
        values = dotenv_values(args.env_file)
        api_key = values.get("GEMINI_API_KEY") or values.get("GOOGLE_API_KEY")
        if not api_key:
            parser.error(f"No GEMINI_API_KEY or GOOGLE_API_KEY found in {args.env_file}.")
        settings = Settings(_env_file=args.env_file, gemini_api_key=api_key)
    else:
        settings = get_settings()
    manifest_path_value = args.manifest or settings.manifest_path

    logger.info("Initializing embedding model: %s", settings.gemini_embedding_model)
    embeddings = GeminiEmbeddingsService(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_embedding_model,
        output_dimensionality=settings.gemini_embedding_dimensions,
    )

    logger.info(
        "Initializing vector store: %s (collection: %s)",
        settings.chroma_dir,
        settings.chroma_collection,
    )
    vector_store = ChromaVectorStoreService(settings.chroma_dir, settings.chroma_collection)

    logger.info("Initializing SQLite database: %s", settings.db_path)
    database = SQLiteDatabaseService(settings.db_path)

    if args.reset:
        logger.warning(
            "Resetting vector collection '%s' and SQLite document registry...",
            settings.chroma_collection,
        )
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
        logger.info(
            "Ingestion result: %s (doc_id=%s, chunks=%d)",
            res.status,
            res.document_id,
            res.chunk_count,
        )
    else:
        manifest_path = Path(manifest_path_value)
        if not manifest_path.exists():
            logger.error(
                "Manifest not found: %s. Run scripts/fetch_corpus.py first.", manifest_path
            )
            sys.exit(1)
        logger.info("Ingesting corpus from manifest: %s", manifest_path)
        results = ingestion.ingest_manifest(manifest_path)
        total_chunks = sum(r.chunk_count for r in results)
        logger.info(
            "Batch ingestion finished: %d documents processed, %d total chunks active.",
            len(results),
            total_chunks,
        )
        for r in results:
            print(f"  [{r.status.upper()}] {r.title} ({r.document_id}) -> {r.chunk_count} chunks")


if __name__ == "__main__":
    main()
