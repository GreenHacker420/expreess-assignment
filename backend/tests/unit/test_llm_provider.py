"""Provider configuration and Gemini structured-output construction tests."""

import pytest

from backend.app.core.config import Settings
from backend.app.services.llm import GeminiLLMService, create_llm_service


def test_gemini_provider_builds_structured_chains_without_network_calls():
    settings = Settings(
        LLM_PROVIDER="gemini",
        GEMINI_API_KEY="unit-test-key",
        GEMINI_MODEL="gemini-3.8-flash",
        GEMINI_FAST_MODEL="gemini-3.8-flash",
    )

    service = create_llm_service(settings)

    assert isinstance(service, GeminiLLMService)
    assert service.model_name == "gemini-3.8-flash"
    assert service.fast_model_name == "gemini-3.8-flash"
    assert service._analyzer is not None
    assert service._grader is not None
    assert service._rewriter is not None
    assert service._generator is not None


def test_unknown_provider_is_rejected():
    settings = Settings(LLM_PROVIDER="unknown")

    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        create_llm_service(settings)
