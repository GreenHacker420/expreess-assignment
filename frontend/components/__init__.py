"""Frontend components package."""

from frontend.components.corpus_tab import render_corpus_tab
from frontend.components.header import render_header
from frontend.components.ingest_tab import render_ingest_tab
from frontend.components.qa_tab import render_qa_tab
from frontend.components.workflow_tab import render_workflow_tab

__all__ = [
    "render_header",
    "render_qa_tab",
    "render_corpus_tab",
    "render_ingest_tab",
    "render_workflow_tab",
]
