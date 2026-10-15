"""Vector store service interface and ChromaDB implementation."""

from collections import Counter
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import chromadb
from chromadb.config import Settings as ChromaSettings

_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{1,}")
_STOP_WORDS = {
    "about", "also", "and", "are", "can", "does", "for", "from", "how",
    "into", "its", "one", "that", "the", "their", "this", "with", "work",
}


def _rank_keyword_documents(
    rows: List[Dict[str, Any]], query: str, top_k: int
) -> List[Dict[str, Any]]:
    """Rank a small corpus with an IDF-weighted term-frequency score."""
    terms = {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(query)
        if token.casefold() not in _STOP_WORDS
    }
    if not terms or not rows:
        return []

    tokenized = [
        [token.casefold() for token in _TOKEN_PATTERN.findall(row.get("text", ""))]
        for row in rows
    ]
    document_frequency = Counter(
        term for tokens in tokenized for term in terms if term in set(tokens)
    )
    avg_length = sum(map(len, tokenized)) / len(tokenized) or 1.0
    ranked = []
    for row, tokens in zip(rows, tokenized):
        frequencies = Counter(tokens)
        length_norm = 0.25 + 0.75 * len(tokens) / avg_length
        score = 0.0
        for term in terms:
            tf = frequencies[term]
            if not tf:
                continue
            idf = math.log(1 + (len(rows) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            score += idf * (tf * 2.2) / (tf + 1.2 * length_norm)
        if score > 0:
            ranked.append(({**row, "distance": 1.0 / (1.0 + score)}, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [row for row, _ in ranked[:max(0, top_k)]]


@runtime_checkable
class VectorStoreService(Protocol):
    """Protocol for vector database operations."""

    def add_chunks(
        self,
        chunk_ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """Add or update chunks in the vector collection."""
        ...

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search top-k nearest chunks using the query vector."""
        ...

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search top-k chunks using lexical overlap as a hybrid signal."""
        ...

    def delete_by_document_id(self, document_id: str) -> int:
        """Remove all chunks belonging to a document."""
        ...

    def count_chunks(self) -> int:
        """Return total number of indexed chunks."""
        ...

    def reset(self) -> None:
        """Clear all indexed data in the collection."""
        ...


class ChromaVectorStoreService:
    """Production vector store using ChromaDB persistent client."""

    def __init__(self, persist_dir: str = "data/chroma", collection_name: str = "fastapi_docs"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunk_ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        if not chunk_ids:
            return 0
        sanitized_metadatas = []
        for m in metadatas:
            clean = {
                k: v for k, v in m.items()
                if isinstance(v, (str, int, float, bool))
            }
            sanitized_metadatas.append(clean)

        self._collection.upsert(
            ids=chunk_ids,
            documents=documents,
            metadatas=sanitized_metadatas,
            embeddings=embeddings,
        )
        return len(chunk_ids)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if self._collection.count() == 0:
            return []
        actual_k = min(top_k, self._collection.count())
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        dists = results["distances"][0] if results.get("distances") else []

        for i, chunk_id in enumerate(ids):
            output.append({
                "chunk_id": chunk_id,
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else 0.0,
            })
        return output

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search document text lexically to complement vector similarity."""
        results = self._collection.get(include=["documents", "metadatas"])
        rows = []
        ids = results.get("ids") or []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        for index, chunk_id in enumerate(ids):
            rows.append({
                "chunk_id": chunk_id,
                "text": docs[index] if index < len(docs) else "",
                "metadata": metas[index] if index < len(metas) else {},
            })
        return _rank_keyword_documents(rows, query, top_k)

    def delete_by_document_id(self, document_id: str) -> int:
        results = self._collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        if results and results["ids"]:
            count = len(results["ids"])
            self._collection.delete(ids=results["ids"])
            return count
        return 0

    def count_chunks(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


class MockVectorStoreService:
    """In-memory mock vector store for unit tests without ChromaDB dependencies."""

    def __init__(self):
        self.chunks: Dict[str, Dict[str, Any]] = {}

    def add_chunks(
        self,
        chunk_ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        for i, cid in enumerate(chunk_ids):
            self.chunks[cid] = {
                "chunk_id": cid,
                "text": documents[i],
                "metadata": metadatas[i],
                "embedding": embeddings[i],
            }
        return len(chunk_ids)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        scored = []
        for cid, item in self.chunks.items():
            emb = item["embedding"]
            score = sum(a * b for a, b in zip(query_embedding, emb))
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, item in scored[:top_k]:
            results.append({
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "distance": 1.0 - score,
            })
        return results

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the mock corpus lexically for deterministic hybrid tests."""
        rows = [
            {"chunk_id": cid, "text": item["text"], "metadata": item["metadata"]}
            for cid, item in self.chunks.items()
        ]
        return _rank_keyword_documents(rows, query, top_k)

    def delete_by_document_id(self, document_id: str) -> int:
        to_del = [
            cid for cid, item in self.chunks.items()
            if item["metadata"].get("document_id") == document_id
        ]
        for cid in to_del:
            del self.chunks[cid]
        return len(to_del)

    def count_chunks(self) -> int:
        return len(self.chunks)

    def reset(self) -> None:
        self.chunks.clear()
