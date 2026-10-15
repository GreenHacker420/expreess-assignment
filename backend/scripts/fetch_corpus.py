"""Script to fetch verified official FastAPI documentation corpus and generate manifest."""

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_corpus")

# Approved official FastAPI documentation pages
APPROVED_CORPUS: List[Dict[str, str]] = [
    {
        "document_id": "fastapi-path-params",
        "title": "Path Parameters",
        "source_url": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/path-params.md",
        "docs_page_url": "https://fastapi.tiangolo.com/tutorial/path-params/",
        "filename": "path_params.md",
    },
    {
        "document_id": "fastapi-query-params",
        "title": "Query Parameters",
        "source_url": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/query-params.md",
        "docs_page_url": "https://fastapi.tiangolo.com/tutorial/query-params/",
        "filename": "query_params.md",
    },
    {
        "document_id": "fastapi-request-body",
        "title": "Request Body",
        "source_url": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/body.md",
        "docs_page_url": "https://fastapi.tiangolo.com/tutorial/body/",
        "filename": "request_body.md",
    },
    {
        "document_id": "fastapi-dependencies",
        "title": "Dependencies - First Steps",
        "source_url": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/dependencies/index.md",
        "docs_page_url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
        "filename": "dependencies.md",
    },
    {
        "document_id": "fastapi-request-files",
        "title": "Request Files",
        "source_url": "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/request-files.md",
        "docs_page_url": "https://fastapi.tiangolo.com/tutorial/request-files/",
        "filename": "request_files.md",
    },
]

LICENSE_ATTRIBUTION = {
    "project": "FastAPI",
    "author": "Sebastián Ramírez",
    "license": "MIT",
    "license_url": "https://github.com/fastapi/fastapi/blob/master/LICENSE",
    "copyright": "Copyright (c) 2018 Sebastián Ramírez",
}


def fetch_corpus(
    output_dir: Path = Path("backend/data/corpus"),
    manifest_path: Path = Path("backend/data/corpus_manifest.json"),
    max_size_bytes: int = 1024 * 1024,  # 1 MB limit per document
    timeout_seconds: float = 15.0,
) -> bool:
    """Fetch approved documents and build manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []

    logger.info("Starting corpus fetch for %d approved FastAPI documents...", len(APPROVED_CORPUS))

    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        for entry in APPROVED_CORPUS:
            doc_id = entry["document_id"]
            title = entry["title"]
            url = entry["source_url"]
            target_file = output_dir / entry["filename"]

            logger.info("Fetching '%s' from %s", title, url)
            try:
                response = client.get(url, headers={"User-Agent": "FastAPI-RAG-Assistant/1.0"})
                response.raise_for_status()

                content_bytes = response.content
                if len(content_bytes) > max_size_bytes:
                    raise ValueError(f"Content exceeds size limit ({len(content_bytes)} > {max_size_bytes})")

                # Verify valid UTF-8
                text_content = content_bytes.decode("utf-8")

                # Compute content hash
                content_hash = hashlib.sha256(content_bytes).hexdigest()

                # Save file locally
                target_file.write_text(text_content, encoding="utf-8")

                manifest_records.append({
                    "document_id": doc_id,
                    "title": title,
                    "source_url": url,
                    "docs_page_url": entry["docs_page_url"],
                    "local_path": str(target_file),
                    "size_bytes": len(content_bytes),
                    "content_hash": content_hash,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "attribution": LICENSE_ATTRIBUTION,
                })
                logger.info("Successfully saved '%s' -> %s (%d bytes)", title, target_file, len(content_bytes))

            except Exception as e:
                logger.error("Failed to fetch %s: %s", title, e)
                return False

    manifest_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_documents": len(manifest_records),
        "license_attribution": LICENSE_ATTRIBUTION,
        "documents": manifest_records,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    logger.info("Corpus manifest written to %s", manifest_path)
    return True


if __name__ == "__main__":
    success = fetch_corpus()
    if not success:
        sys.exit(1)
    print("Corpus fetch completed successfully.")
