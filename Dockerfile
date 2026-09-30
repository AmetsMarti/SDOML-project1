# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7861 \
    GRADIO_ANALYTICS_ENABLED=False

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app


COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project


COPY sdoml_task1 ./sdoml_task1
COPY notebooks/03_gradio_interface.ipynb ./notebooks/
COPY data ./data
COPY models ./models
RUN uv sync --frozen


RUN .venv/bin/jupyter nbconvert --to script --output gradio_app --output-dir /app \
    notebooks/03_gradio_interface.ipynb

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD [".venv/bin/python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/', timeout=5).status == 200 else 1)"]

CMD [".venv/bin/python", "gradio_app.py"]
