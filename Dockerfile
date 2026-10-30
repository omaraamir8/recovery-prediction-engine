FROM python:3.12-slim

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
COPY src/ ./src/
# serving image: API + champion-model runtime (sklearn lives in the ml extra;
# torch stays out of the image — see README for the CPU-torch training setup)
RUN uv sync --frozen --no-dev --extra serve --extra ml

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "rpe.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
