# Build with Podman:  podman build -t localhost/nexusgate-api:dev .
# Image names are fully qualified: Podman refuses ambiguous short names when it can't prompt.
FROM docker.io/library/python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Use LiteLLM's bundled price map instead of fetching it from GitHub on every start.
    LITELLM_LOCAL_MODEL_COST_MAP=True

WORKDIR /srv

# Dependencies first so code changes don't invalidate the layer.
COPY pyproject.toml README.md ./
RUN mkdir app && echo '__version__ = "0.0.0"' > app/__init__.py \
    && pip install . && pip uninstall -y nexusgate

COPY app ./app
COPY config ./config
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 nexus
USER 10001

EXPOSE 8000
# No HEALTHCHECK: it isn't part of the OCI image format Podman builds by default, and Kubernetes
# probes /healthz and /readyz instead (infra/k8s/api.yaml).
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
