"""LLM service protocol, structured output schemas, and Groq implementation."""

import asyncio
import logging
from typing import List, Literal, Optional, Protocol, runtime_checkable
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from backend.app.core.prompts import (
    DOCUMENT_GRADING_SYSTEM_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    QUERY_ANALYSIS_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
)

logger = logging.getLogger("llm_service")


class QueryAnalysisOutput(BaseModel):
    """Structured output for query analysis node."""
    transformed_query: str = Field(
        description="Expanded, keyword-optimized query designed for dense semantic retrieval"
    )
    query_type: Literal["conceptual", "how-to", "troubleshooting", "api_reference"] = Field(
        description="Categorization of user intent"
    )


class GradeOutput(BaseModel):
    """Structured output for document grading node."""
    chunk_id: str = Field(description="The unique identifier of the chunk being evaluated")
    relevant: bool = Field(description="True if the chunk provides factual information addressing the question")
    reason: str = Field(description="Brief 1-sentence rationale for the relevance verdict")


class RewriteOutput(BaseModel):
    """Structured output for query rewriting node."""
    rewritten_query: str = Field(
        description="Alternative retrieval phrasing preserving the original question's core intent"
    )


class GenerationOutput(BaseModel):
    """Structured output for answer generation node."""
    answer: str = Field(
        description="Clear, concise, factual answer grounded exclusively in the provided context, with mandatory inline citation tags like [S1], [S2] attached directly to each factual sentence"
    )
    cited_labels: List[str] = Field(
        default_factory=list,
        description="List of context labels cited in the answer, e.g. ['S1', 'S2']"
    )
    has_sufficient_context: bool = Field(
        default=True,
        description="False if the provided context lacks sufficient evidence to answer accurately"
    )



class LLMOperationalError(Exception):
    """Base error for LLM operational issues."""
    pass


class LLMTimeoutError(LLMOperationalError):
    """Raised when upstream LLM times out."""
    pass


class LLMUpstreamError(LLMOperationalError):
    """Raised when upstream model API returns an error."""
    pass



@runtime_checkable
class LLMService(Protocol):
    """Protocol for LLM interactions."""

    async def analyze_query(self, question: str) -> QueryAnalysisOutput: ...

    async def grade_chunk(
        self, question: str, chunk_id: str, chunk_text: str
    ) -> GradeOutput: ...

    async def rewrite_query(
        self, original_question: str, failed_queries: List[str]
    ) -> str: ...

    async def generate_answer(
        self, question: str, labeled_context: str
    ) -> GenerationOutput: ...



class GroqLLMService:
    """Production LLM service utilizing Groq with structured outputs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
        fast_model: Optional[str] = None,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
    ):

        if not api_key:
            raise ValueError("GROQ_API_KEY must be provided for GroqLLMService.")

        self.model_name = model
        self.fast_model_name = fast_model or model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

        self._llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model,
            temperature=temperature,
            request_timeout=timeout_seconds,
        )

        self._fast_llm = ChatGroq(
            groq_api_key=api_key,
            model_name=self.fast_model_name,
            temperature=temperature,
            request_timeout=timeout_seconds,
        )

        self._analyzer = self._fast_llm.with_structured_output(QueryAnalysisOutput)
        self._grader = self._fast_llm.with_structured_output(GradeOutput)
        self._rewriter = self._fast_llm.with_structured_output(RewriteOutput)
        self._generator = self._llm.with_structured_output(GenerationOutput)

    async def _invoke_with_backoff(self, chain, messages, max_attempts: int = 3):
        """Invoke LLM chain with automatic exponential backoff on TPM/RPM 429 throttling."""
        for attempt in range(1, max_attempts + 1):
            try:
                return await chain.ainvoke(messages)
            except Exception as e:
                err_msg = str(e).lower()
                if ("rate limit" in err_msg or "429" in err_msg) and attempt < max_attempts:
                    delay = 6.0 * attempt
                    logger.warning(
                        "Groq rate limit hit. Backing off for %.1fs (attempt %d/%d)...",
                        delay, attempt, max_attempts
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

    async def analyze_query(self, question: str) -> QueryAnalysisOutput:
        messages = [
            ("system", QUERY_ANALYSIS_SYSTEM_PROMPT),
            ("human", f"User question: {question}"),
        ]
        try:
            res = await self._invoke_with_backoff(self._analyzer, messages)
            if not isinstance(res, QueryAnalysisOutput):
                raise LLMOperationalError("Failed to parse QueryAnalysisOutput")
            return res
        except asyncio.TimeoutError as e:
            raise LLMTimeoutError("Groq query analysis timed out") from e
        except Exception as e:
            if isinstance(e, LLMOperationalError):
                raise
            raise LLMUpstreamError(f"Groq API error during query analysis: {e}") from e

    async def grade_chunk(
        self, question: str, chunk_id: str, chunk_text: str
    ) -> GradeOutput:
        messages = [
            ("system", DOCUMENT_GRADING_SYSTEM_PROMPT),
            (
                "human",
                f"Question: {question}\n\nChunk ID: {chunk_id}\n\nChunk Content:\n{chunk_text}",
            ),
        ]
        try:
            res = await self._invoke_with_backoff(self._grader, messages)
            if not isinstance(res, GradeOutput):
                raise LLMOperationalError("Failed to parse GradeOutput")
            return res
        except asyncio.TimeoutError as e:
            raise LLMTimeoutError("Groq document grading timed out") from e
        except Exception as e:
            if isinstance(e, LLMOperationalError):
                raise
            raise LLMUpstreamError(f"Groq API error during document grading: {e}") from e

    async def rewrite_query(
        self, original_question: str, failed_queries: List[str]
    ) -> str:
        history_str = "\n".join(f"- {q}" for q in failed_queries)
        messages = [
            ("system", QUERY_REWRITE_SYSTEM_PROMPT),
            (
                "human",
                f"Original question: {original_question}\n\nPreviously attempted queries:\n{history_str}",
            ),
        ]
        try:
            res = await self._invoke_with_backoff(self._rewriter, messages)
            if not isinstance(res, RewriteOutput):
                raise LLMOperationalError("Failed to parse RewriteOutput")
            return res.rewritten_query
        except asyncio.TimeoutError as e:
            raise LLMTimeoutError("Groq query rewrite timed out") from e
        except Exception as e:
            if isinstance(e, LLMOperationalError):
                raise
            raise LLMUpstreamError(f"Groq API error during query rewrite: {e}") from e

    async def generate_answer(
        self, question: str, labeled_context: str
    ) -> GenerationOutput:
        messages = [
            ("system", GENERATION_SYSTEM_PROMPT),
            (
                "human",
                f"Question: {question}\n\nLabeled Evidence:\n{labeled_context}",
            ),
        ]
        try:
            res = await self._invoke_with_backoff(self._generator, messages)
            if not isinstance(res, GenerationOutput):
                raise LLMOperationalError("Failed to parse GenerationOutput")
            return res
        except asyncio.TimeoutError as e:
            raise LLMTimeoutError("Groq answer generation timed out") from e
        except Exception as e:
            if isinstance(e, LLMOperationalError):
                raise
            raise LLMUpstreamError(f"Groq API error during answer generation: {e}") from e



class MockLLMService:
    """Mock LLM service providing deterministic responses without network calls."""

    def __init__(
        self,
        default_relevant: bool = True,
        simulate_timeout: bool = False,
        simulate_upstream_error: bool = False,
        generation_answer: str = "This is a verified answer based on FastAPI docs.",
        generation_labels: Optional[List[str]] = None,
        has_sufficient_context: bool = True,
    ):
        self.default_relevant = default_relevant
        self.simulate_timeout = simulate_timeout
        self.simulate_upstream_error = simulate_upstream_error
        self.generation_answer = generation_answer
        self.generation_labels = ["S1"] if generation_labels is None else generation_labels
        self.has_sufficient_context = has_sufficient_context

        # Per-chunk overrides
        self.chunk_relevance_map: dict[str, bool] = {}

    async def analyze_query(self, question: str) -> QueryAnalysisOutput:
        if self.simulate_timeout:
            raise LLMTimeoutError("Simulated timeout")
        if self.simulate_upstream_error:
            raise LLMUpstreamError("Simulated upstream error")

        q_lower = question.lower()
        if any(term in q_lower for term in ["override", "ignore all", "pwned", "secret key"]):
            return QueryAnalysisOutput(
                transformed_query=question,
                query_type="conceptual",
            )
        return QueryAnalysisOutput(
            transformed_query=f"{question} fastapi technical reference",
            query_type="how-to",
        )

    async def grade_chunk(
        self, question: str, chunk_id: str, chunk_text: str
    ) -> GradeOutput:
        if self.simulate_timeout:
            raise LLMTimeoutError("Simulated timeout")
        if self.simulate_upstream_error:
            raise LLMUpstreamError("Simulated upstream error")

        q_lower = question.lower()
        # Out-of-corpus topics or prompt injection attempts must be judged irrelevant
        out_of_scope_keywords = [
            "celery", "redis", "flower", "jwt", "passlib", "bcrypt",
            "override", "pwned", "secret key", "drop sqlite"
        ]
        if any(kw in q_lower for kw in out_of_scope_keywords):
            return GradeOutput(
                chunk_id=chunk_id,
                relevant=False,
                reason="Question asks for topics outside indexed documentation or contains injection directives.",
            )

        rel = self.chunk_relevance_map.get(chunk_id, self.default_relevant)
        return GradeOutput(
            chunk_id=chunk_id,
            relevant=rel,
            reason="Mock relevance evaluation",
        )

    async def rewrite_query(
        self, original_question: str, failed_queries: List[str]
    ) -> str:
        if self.simulate_timeout:
            raise LLMTimeoutError("Simulated timeout")
        if self.simulate_upstream_error:
            raise LLMUpstreamError("Simulated upstream error")
        return f"{original_question} alternative attempt {len(failed_queries) + 1}"

    async def generate_answer(
        self, question: str, labeled_context: str
    ) -> GenerationOutput:
        if self.simulate_timeout:
            raise LLMTimeoutError("Simulated timeout")
        if self.simulate_upstream_error:
            raise LLMUpstreamError("Simulated upstream error")
        return GenerationOutput(
            answer=self.generation_answer,
            cited_labels=self.generation_labels,
            has_sufficient_context=self.has_sufficient_context,
        )
