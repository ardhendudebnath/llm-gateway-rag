# Build with Podman:  podman build -t localhost/nexusgate-api:dev .
# Image names are fully qualified: Podman refuses ambiguous short names when it can't prompt.
FROM docker.io/library/python:3.11-slim

ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=Xenova/ms-marco-MiniLM-L-6-v2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Use LiteLLM's bundled price map instead of fetching it from GitHub on every start.
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    NEXUSGATE_MODEL_CACHE_DIR=/opt/models \
    NEXUSGATE_EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    NEXUSGATE_RAG_RERANKER_MODEL=${RERANKER_MODEL}

WORKDIR /srv

# Security fixes first. Debian's point updates land here without waiting for the base image to be
# rebuilt, and pip and setuptools go to current releases (pip only for the build: it is removed
# below). setuptools vendors its own copies of wheel and jaraco.context, which is where the base
# image's fixable HIGH findings were. CI scans the result and refuses to publish an image with a
# fixable HIGH or CRITICAL vulnerability.
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip setuptools

# Dependencies next so code changes don't invalidate the layer. pyproject.toml names README.md as
# the package readme; an empty placeholder keeps README edits from reinstalling every dependency.
COPY pyproject.toml ./
RUN touch README.md && mkdir app && echo '__version__ = "0.0.0"' > app/__init__.py \
    && pip install ".[embeddings]" && pip uninstall -y nexusgate

# Bake the embedding and reranking models into the image: pods never download weights at startup
# and can run without internet access. Before the app code, so code changes don't re-download.
RUN python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('${EMBEDDING_MODEL}', cache_dir='/opt/models'); \
TextCrossEncoder('${RERANKER_MODEL}', cache_dir='/opt/models')" \
    && chmod -R a+rX /opt/models
ENV HF_HUB_OFFLINE=1

COPY app ./app
COPY config ./config
# pip leaves with the last install. Nothing runs it at runtime, and it vendors its own copies of
# urllib3, msgpack and pkg_resources, which a vulnerability scan rightly counts against the image.
RUN pip install --no-deps . && pip uninstall -y pip

RUN useradd --create-home --uid 10001 nexus
USER 10001

EXPOSE 8000
# No HEALTHCHECK: it isn't part of the OCI image format Podman builds by default, and Kubernetes
# probes /healthz and /readyz instead (infra/k8s/base/api.yaml).
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
