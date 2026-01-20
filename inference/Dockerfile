FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

ENV UV_HTTP_TIMEOUT=300

COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked

COPY src /app/src

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
