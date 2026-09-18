# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# Dependencies first so code changes don't invalidate the layer.
COPY pyproject.toml README.md ./
RUN mkdir app && echo '__version__ = "0.0.0"' > app/__init__.py \
    && pip install . && pip uninstall -y nexusgate

COPY app ./app
COPY config ./config
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 nexus
USER nexus

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
