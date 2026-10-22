FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY xrl_ops ./xrl_ops

RUN uv pip install --system --no-cache -e .

EXPOSE 8000

CMD ["uvicorn", "xrl_ops.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
