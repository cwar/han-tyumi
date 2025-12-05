FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY . .
RUN uv sync --frozen --no-dev

# Make start script executable
RUN chmod +x /app/scripts/start.sh

# Build DB at startup (fetches fresh data from API)
CMD ["/app/scripts/start.sh"]
