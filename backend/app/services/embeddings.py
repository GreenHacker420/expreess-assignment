"""Embedding service interface and implementations."""

from typing import List, Protocol, runtime_checkable
import hashlib

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment, misc]


@runtime_checkable
class EmbeddingsService(Protocol):
    """Protocol for embedding and tokenization operations."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text documents into vector representations."""
        ...

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query into a vector representation."""
        ...

    def count_tokens(self, text: str) -> int:
        """Measure the token length of a given text."""
        ...


class SentenceTransformerEmbeddingsService:
    """Production embeddings service using local sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Please install it.")
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._tokenizer = self._model.tokenizer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return embedding.tolist()

    def count_tokens(self, text: str) -> int:
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)


class MockEmbeddingsService:
    """Deterministic, offline fake embeddings service for zero-network testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_to_vector(self, text: str) -> List[float]:
        # Generate deterministic floats based on MD5 hash of text
        h = hashlib.md5(text.encode("utf-8")).digest()
        vec = []
        for i in range(self.dimension):
            byte_val = h[i % len(h)]
            vec.append((byte_val / 255.0) * 2.0 - 1.0)
        # Normalize
        norm = (sum(x * x for x in vec)) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vector(text)

    def count_tokens(self, text: str) -> int:
        # Approximate 1 token ~ 4 characters or whitespace split for fast testing
        return len(text.split())
