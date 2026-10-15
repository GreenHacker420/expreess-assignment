"""Formatting and text helpers for frontend UI."""

import re


def format_doc_title(raw_title: str) -> str:
    """Format markdown titles by stripping out bracket anchors like '{ #path-params }'."""
    clean = re.sub(r"\{.*?\}", "", raw_title).strip()
    return clean if clean else raw_title
