"""Core settings and prompt templates."""

from backend.app.core.config import Settings, get_settings
from backend.app.core.prompts import (
    DOCUMENT_GRADING_SYSTEM_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    QUERY_ANALYSIS_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
)

__all__ = [
    "Settings",
    "get_settings",
    "DOCUMENT_GRADING_SYSTEM_PROMPT",
    "GENERATION_SYSTEM_PROMPT",
    "QUERY_ANALYSIS_SYSTEM_PROMPT",
    "QUERY_REWRITE_SYSTEM_PROMPT",
]
