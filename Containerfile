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

# Dependencies first so code changes don't invalidate the layer. pyproject.toml names README.md as
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
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 nexus
USER 10001

EXPOSE 8000
# No HEALTHCHECK: it isn't part of the OCI image format Podman builds by default, and Kubernetes
# probes /healthz and /readyz instead (infra/k8s/base/api.yaml).
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
