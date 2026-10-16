"""Vector store adapters backed by LangChain's vector store/retriever interfaces."""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnableLambda


@runtime_checkable
class VectorStoreService(Protocol):
    """Operations required by ingestion, query retrieval, and health reporting."""

    def add_chunks(
        self,
        chunk_ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int: ...
    def as_retriever(
        self, embeddings: Embeddings, top_k: int = 5
    ) -> Runnable[str, list[Document]]: ...
    def delete_by_document_id(self, document_id: str) -> int: ...
    def count_chunks(self) -> int: ...
    def reset(self) -> None: ...


class ChromaVectorStoreService:
    """Persistent Chroma store exposed through LangChain's Chroma integration."""

    def __init__(
        self,
        persist_dir: str = "data/chroma",
        collection_name: str = "fastapi_docs_gemini2_qa_v1",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._retriever_cache: dict[int, Runnable[str, list[Document]]] = {}

    def _langchain_store(self, embeddings: Embeddings) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            client=self._client,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunk_ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        if not chunk_ids:
            return 0
        clean_metadatas = [
            {
                key: value
                for key, value in metadata.items()
                if isinstance(value, (str, int, float, bool))
            }
            for metadata in metadatas
        ]
        self._collection.upsert(
            ids=chunk_ids, documents=documents, metadatas=clean_metadatas, embeddings=embeddings
        )
        return len(chunk_ids)

    def as_retriever(self, embeddings: Embeddings, top_k: int = 5) -> Runnable[str, list[Document]]:
        """Return LangChain's standard vector-store retriever (Runnable interface)."""
        key = id(embeddings) * 31 + top_k
        if key not in self._retriever_cache:
            self._retriever_cache[key] = self._langchain_store(embeddings).as_retriever(
                search_type="similarity", search_kwargs={"k": top_k}
            )
        return self._retriever_cache[key]

    def delete_by_document_id(self, document_id: str) -> int:
        results = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
        ids = results.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def count_chunks(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._retriever_cache.clear()


class MockVectorStoreService:
    """Deterministic in-memory test double; production retrieval uses LangChain Chroma."""

    def __init__(self):
        self.chunks: dict[str, dict[str, Any]] = {}

    def add_chunks(
        self,
        chunk_ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        for index, chunk_id in enumerate(chunk_ids):
            self.chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "text": documents[index],
                "metadata": metadatas[index],
                "embedding": embeddings[index],
            }
        return len(chunk_ids)

    def as_retriever(self, embeddings: Embeddings, top_k: int = 5) -> Runnable[str, list[Document]]:
        def retrieve(query: str) -> list[Document]:
            vector = embeddings.embed_query(query)
            ranked = sorted(
                self.chunks.values(),
                key=lambda item: sum(a * b for a, b in zip(vector, item["embedding"])),
                reverse=True,
            )[:top_k]
            return [
                Document(
                    page_content=item["text"],
                    metadata={**item["metadata"], "chunk_id": item["chunk_id"]},
                )
                for item in ranked
            ]

        return RunnableLambda(retrieve)

    def delete_by_document_id(self, document_id: str) -> int:
        ids = [
            cid
            for cid, item in self.chunks.items()
            if item["metadata"].get("document_id") == document_id
        ]
        for chunk_id in ids:
            del self.chunks[chunk_id]
        return len(ids)

    def count_chunks(self) -> int:
        return len(self.chunks)

    def reset(self) -> None:
        self.chunks.clear()
