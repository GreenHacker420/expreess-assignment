"""Document ingestion service with heading-aware, code-block-preserving chunking."""

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

from backend.app.services.database import DatabaseService
from backend.app.services.embeddings import EmbeddingsService
from backend.app.services.vector_store import VectorStoreService

logger = logging.getLogger("ingestion")


class ChunkRecord(NamedTuple):
    chunk_id: str
    document_id: str
    text: str
    source: str
    title: str
    section: str
    chunk_index: int
    content_hash: str


class IngestionResult(NamedTuple):
    document_id: str
    source: str
    title: str
    chunk_count: int
    status: str  # "indexed", "skipped", "updated"


class IngestionService:
    """Reusable, thread-safe ingestion service for technical documentation.

    Transaction & Recovery Limitations:
        SQLite and ChromaDB do not participate in a shared two-phase transaction.
        To minimize partial-state risks, new chunks and embeddings are computed
        and prepared in memory BEFORE altering persistent state. Obsolete chunks
        are then purged from ChromaDB and replaced, followed immediately by updating
        SQLite. If a fatal crash occurs mid-write, re-running ingestion restores
        full consistency.
    """

    def __init__(
        self,
        embeddings: EmbeddingsService,
        vector_store: VectorStoreService,
        database: DatabaseService,
        target_chunk_tokens: int = 800,
        overlap_tokens: int = 150,
    ):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.database = database
        self.target_chunk_tokens = target_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self._lock = threading.Lock()

    @staticmethod
    def normalize_content(content: str) -> str:
        """Normalize line endings without corrupting code block indentation."""
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        # Strip trailing whitespace on lines, but preserve leading tabs/spaces
        lines = [line.rstrip() for line in content.split("\n")]
        return "\n".join(lines).strip()

    @staticmethod
    def extract_title(content: str, fallback_title: str) -> str:
        """Extract top H1 header or use fallback."""
        match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
        return fallback_title

    def _split_into_sections(self, content: str, default_title: str) -> List[Dict[str, str]]:
        """Split markdown into sections based on headings (# , ## , ### )."""
        lines = content.split("\n")
        sections: List[Dict[str, str]] = []
        current_section = default_title
        current_lines: List[str] = []

        heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$")

        for line in lines:
            m = heading_pattern.match(line)
            if m:
                if current_lines:
                    text = "\n".join(current_lines).strip()
                    if text:
                        sections.append({"section": current_section, "text": text})
                    current_lines = []
                current_section = m.group(2).strip()
            current_lines.append(line)

        if current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                sections.append({"section": current_section, "text": text})

        return sections or [{"section": default_title, "text": content}]

    def _split_section_into_chunks(
        self, section_name: str, section_text: str, doc_title: str
    ) -> List[str]:
        """Split section into token-budgeted chunks, keeping code blocks intact where possible."""
        header_prefix = f"[{doc_title} > {section_name}]\n\n"
        full_candidate = header_prefix + section_text
        if self.embeddings.count_tokens(full_candidate) <= self.target_chunk_tokens:
            return [full_candidate]

        blocks = re.split(r"(```[\s\S]*?```)", section_text)
        units: List[str] = []
        for block in blocks:
            b = block.strip()
            if not b:
                continue
            if b.startswith("```") and b.endswith("```"):
                units.append(b)
            else:
                paragraphs = [p.strip() for p in b.split("\n\n") if p.strip()]
                units.extend(paragraphs)

        chunks: List[str] = []
        current_units: List[str] = []

        max_budget = int(self.target_chunk_tokens * 1.15)

        def _safe_add_chunk(chunk_content: str) -> None:
            """Ensure chunk strictly obeys token budget, splitting words if needed."""
            tokens = self.embeddings.count_tokens(chunk_content)
            if tokens <= max_budget:
                chunks.append(chunk_content)
                return

            words = chunk_content.split(" ")
            curr_words: List[str] = []
            for w in words:
                candidate = " ".join(curr_words + [w])
                if self.embeddings.count_tokens(candidate) > max_budget and curr_words:
                    chunks.append(" ".join(curr_words))
                    curr_words = [w]
                else:
                    curr_words.append(w)
            if curr_words:
                chunks.append(" ".join(curr_words))

        for unit in units:
            unit_tokens = self.embeddings.count_tokens(unit)

            if unit_tokens > self.target_chunk_tokens:
                if current_units:
                    _safe_add_chunk(header_prefix + "\n\n".join(current_units))
                    current_units = []

                sub_lines = unit.split("\n")
                sub_buffer: List[str] = []
                for sline in sub_lines:
                    test_str = header_prefix + "\n".join(sub_buffer + [sline])
                    if self.embeddings.count_tokens(test_str) > self.target_chunk_tokens and sub_buffer:
                        _safe_add_chunk(header_prefix + "\n".join(sub_buffer))
                        sub_buffer = []
                    sub_buffer.append(sline)
                if sub_buffer:
                    _safe_add_chunk(header_prefix + "\n".join(sub_buffer))
                continue

            test_units = current_units + [unit]
            test_candidate = header_prefix + "\n\n".join(test_units)
            if self.embeddings.count_tokens(test_candidate) <= self.target_chunk_tokens:
                current_units.append(unit)
            else:
                if current_units:
                    _safe_add_chunk(header_prefix + "\n\n".join(current_units))
                current_units = [unit]

        if current_units:
            _safe_add_chunk(header_prefix + "\n\n".join(current_units))

        return chunks

    def ingest_text(
        self,
        source: str,
        title: str,
        raw_content: str,
        document_id: Optional[str] = None,
    ) -> IngestionResult:
        """Ingest a text/markdown string into the vector store and database registry."""
        with self._lock:
            normalized = self.normalize_content(raw_content)
            if not normalized:
                raise ValueError(f"Document {source} contains no text after normalization.")

            content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            doc_id = document_id or f"doc_{hashlib.md5(source.encode('utf-8')).hexdigest()[:12]}"
            doc_title = self.extract_title(normalized, title)

            existing = self.database.get_document_by_source(source)
            if existing and existing.get("content_hash") == content_hash:
                logger.info("Source '%s' is unchanged (hash %s). Skipping.", source, content_hash[:8])
                return IngestionResult(
                    document_id=existing["document_id"],
                    source=source,
                    title=existing["title"],
                    chunk_count=existing["chunk_count"],
                    status="skipped",
                )

            sections = self._split_into_sections(normalized, doc_title)
            chunk_records: List[ChunkRecord] = []
            chunk_idx = 0

            for sec in sections:
                section_name = sec["section"]
                sec_text = sec["text"]
                sec_chunks = self._split_section_into_chunks(section_name, sec_text, doc_title)

                for chunk_str in sec_chunks:
                    c_hash = hashlib.sha256(chunk_str.encode("utf-8")).hexdigest()
                    cid = f"{doc_id}_c{chunk_idx:03d}_{c_hash[:8]}"
                    chunk_records.append(
                        ChunkRecord(
                            chunk_id=cid,
                            document_id=doc_id,
                            text=chunk_str,
                            source=source,
                            title=doc_title,
                            section=section_name,
                            chunk_index=chunk_idx,
                            content_hash=c_hash,
                        )
                    )
                    chunk_idx += 1

            if not chunk_records:
                raise ValueError(f"No chunks generated for {source}.")

            texts_to_embed = [c.text for c in chunk_records]
            embeddings = self.embeddings.embed_documents(texts_to_embed)

            metadatas = [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "source": c.source,
                    "title": c.title,
                    "section": c.section,
                    "chunk_index": c.chunk_index,
                    "content_hash": c.content_hash,
                }
                for c in chunk_records
            ]
            ids = [c.chunk_id for c in chunk_records]

            deleted_count = self.vector_store.delete_by_document_id(doc_id)
            if deleted_count > 0:
                logger.info("Removed %d obsolete chunks for document '%s'", deleted_count, doc_id)

            self.vector_store.add_chunks(
                chunk_ids=ids,
                documents=texts_to_embed,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            self.database.record_document(
                document_id=doc_id,
                source=source,
                title=doc_title,
                chunk_count=len(chunk_records),
                content_hash=content_hash,
            )

            status = "updated" if existing else "indexed"
            logger.info(
                "Successfully %s '%s' (%s): %d chunks indexed.",
                status, doc_title, doc_id, len(chunk_records)
            )
            return IngestionResult(
                document_id=doc_id,
                source=source,
                title=doc_title,
                chunk_count=len(chunk_records),
                status=status,
            )

    def ingest_file(self, file_path: Path, title: Optional[str] = None) -> IngestionResult:
        """Ingest a local .md or .txt file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        raw_text = file_path.read_text(encoding="utf-8")
        doc_title = title or file_path.stem.replace("_", " ").title()
        return self.ingest_text(
            source=str(file_path),
            title=doc_title,
            raw_content=raw_text,
            document_id=f"doc_{file_path.stem}",
        )

    def ingest_manifest(self, manifest_path: Path) -> List[IngestionResult]:
        """Ingest all documents declared in a corpus manifest file."""
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        results = []
        for doc in data.get("documents", []):
            local_path = Path(doc["local_path"])
            if not local_path.exists():
                logger.warning("Manifest path %s does not exist, skipping.", local_path)
                continue
            raw_text = local_path.read_text(encoding="utf-8")
            res = self.ingest_text(
                source=doc.get("source_url", str(local_path)),
                title=doc.get("title", local_path.stem),
                raw_content=raw_text,
                document_id=doc.get("document_id"),
            )
            results.append(res)
        return results
