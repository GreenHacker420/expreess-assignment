# Multi-stage production Dockerfile for FastAPI RAG Backend
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation for faster cold starts
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies first for optimal Docker layer caching
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy application source and indexed documentation corpus
COPY backend/ ./backend/
COPY app/ ./app/

# Final lightweight runtime image
FROM python:3.12-slim-bookworm AS runner

WORKDIR /app

# Create non-root system user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy virtual environment and application code from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/backend /app/backend
COPY --from=builder /app/app /app/app

# Set permissions for non-root user
RUN chown -R appuser:appuser /app

# Environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8000

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
