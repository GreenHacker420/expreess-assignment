"""FastAPI entrypoint for the backend service."""

import uvicorn

from backend.app.core.config import get_settings
from backend.app.main import app, create_app

__all__ = ["app", "create_app"]

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
