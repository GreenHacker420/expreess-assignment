"""FastAPI application entrypoint re-exporting backend.app.main."""

from backend.app.main import app, create_app

__all__ = ["app", "create_app"]
