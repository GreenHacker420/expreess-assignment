"""Embedding service interface and Google Gemini Embedding 2 implementation."""

import hashlib
import logging
import os
import re
from typing import Protocol, runtime_checkable

from langchain_core.embeddings import Embeddings

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = logging.getLogger("embeddings")


@runtime_checkable
class EmbeddingsService(Protocol):
    """Protocol for embedding and tokenization operations."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text documents into vector representations."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query into a vector representation."""
        ...

    def count_tokens(self, text: str) -> int:
        """Measure the token length of a given text."""
        ...


class GeminiEmbeddingsService:
    """Production cloud embeddings service using Google Gemini Embedding 2."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-embedding-2-preview",
        output_dimensionality: int = 768,
    ):
        if genai is None:
            raise ImportError(
                "google-genai package is required. Install with `uv add google-genai`."
            )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.output_dimensionality = output_dimensionality
        self._client = genai.Client(api_key=self.api_key) if self.api_key else None

    def _embed_single_with_retry(
        self,
        text: str,
        max_retries: int = 5,
        task_type: str | None = None,
    ) -> list[float]:
        import time

        delay = 2.0
        for attempt in range(max_retries):
            try:
                embed_config = {"output_dimensionality": self.output_dimensionality}
                if task_type:
                    embed_config["task_type"] = task_type
                res = self._client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                    config=types.EmbedContentConfig(**embed_config) if types else None,
                )
                return res.embeddings[0].values
            except Exception as e:
                err_str = str(e)
                if (
                    "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                ) and attempt < max_retries - 1:
                    logger.warning(
                        "Gemini embedding rate-limited (429). Retrying in %.1fs...", delay
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

    @staticmethod
    def _format_document_for_embedding(text: str) -> str:
        """Apply Gemini Embedding 2's documented title/text document format."""
        match = re.match(r"^\[(?P<title>.+?)\s*>\s*(?P<section>.+?)\]\s*\n\s*", text)
        if match:
            title = match.group("title").strip()
            section = match.group("section").strip()
            body = text[match.end() :].strip()
            text = f"section: {section}\n{body}"
        else:
            title = "none"
        return f"title: {title} | text: {text}"

    @staticmethod
    def _format_query_for_embedding(text: str) -> str:
        """Apply Gemini Embedding 2's documented asymmetric retrieval query format."""
        return f"task: question answering | query: {text}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import time

        if not texts:
            return []
        if not self._client:
            raise ValueError("GEMINI_API_KEY is required for GeminiEmbeddingsService.")

        results = []
        for text in texts:
            if "embedding-2" in self.model_name:
                emb = self._embed_single_with_retry(self._format_document_for_embedding(text))
            else:
                emb = self._embed_single_with_retry(text, task_type="RETRIEVAL_DOCUMENT")
            results.append(emb)
            time.sleep(0.3)  # Gentle spacing between requests to stay well within rate limits
        return results

    def embed_query(self, text: str) -> list[float]:
        if not self._client:
            raise ValueError("GEMINI_API_KEY is required for GeminiEmbeddingsService.")
        if "embedding-2" in self.model_name:
            return self._embed_single_with_retry(self._format_query_for_embedding(text))
        return self._embed_single_with_retry(text, task_type="RETRIEVAL_QUERY")

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class MockEmbeddingsService:
    """Deterministic, offline fake embeddings service for zero-network testing."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def _hash_to_vector(self, text: str) -> list[float]:
        h = hashlib.md5(text.encode("utf-8")).digest()
        vec = []
        for i in range(self.dimension):
            byte_val = h[i % len(h)]
            vec.append((byte_val / 255.0) * 2.0 - 1.0)
        norm = (sum(x * x for x in vec)) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class LangChainEmbeddingsAdapter(Embeddings):
    """Expose the configured embedding service through LangChain's standard interface."""

    def __init__(self, service: EmbeddingsService):
        self.service = service

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.service.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.service.embed_query(text)
