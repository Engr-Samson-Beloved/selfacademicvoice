# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- build stage
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Dependencies are copied and installed before the source so that editing code
# does not invalidate the (slow) dependency layer.
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# --------------------------------------------------------------- runtime stage
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

# curl is only for the container HEALTHCHECK below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 ghostwriter

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=ghostwriter:ghostwriter api/ ./api/
COPY --chown=ghostwriter:ghostwriter ghostwriter/ ./ghostwriter/
COPY --chown=ghostwriter:ghostwriter scripts/ ./scripts/
COPY --chown=ghostwriter:ghostwriter data/style_prompt.txt data/voice_profile.json ./data/
COPY --chown=ghostwriter:ghostwriter pyproject.toml README.md ./

# Rewritten output is written here at runtime; it must be writable and should be
# mounted as a volume if you need the files to survive a restart.
RUN mkdir -p /app/data/rewritten && chown -R ghostwriter:ghostwriter /app/data

USER ghostwriter
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# IMPORTANT: a single worker, deliberately.
#
# Job state lives in the in-process `_jobs` dict in api/main.py. With more than
# one worker, POST /rewrite is handled by one process and the client's
# GET /rewrite/{job_id} poll lands on another, which returns 404 for a job that
# is running perfectly well. Scaling out requires moving job state to Redis or a
# database first. Scale vertically (GHOSTWRITER_MAX_WORKERS) until then.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75"]
