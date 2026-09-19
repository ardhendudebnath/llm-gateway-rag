#!/usr/bin/env bash
# One command: build the API image with Podman, run a local kind cluster on Podman, deploy NexusGate.
#
#   scripts/cluster-up.sh              create or update (safe to re-run)
#   scripts/cluster-up.sh --recreate   delete and recreate the cluster first
#
# Every kubectl call is pinned to the kind-nexusgate context, so it never touches another cluster.
# Secrets come from .env; missing admin/JWT secrets are generated once and kept across runs.
# Linux: kind needs rootful Podman (run with sudo) or rootless Podman with cgroup v2 delegation,
# see https://kind.sigs.k8s.io/docs/user/rootless/. macOS/Windows use a rootful Podman machine.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER=nexusgate
CONTEXT="kind-$CLUSTER"
NS=nexusgate
IMAGE=localhost/nexusgate-api:dev
export KIND_EXPERIMENTAL_PROVIDER=podman
RECREATE=false
if [[ "${1:-}" == "--recreate" ]]; then RECREATE=true; fi

step() { printf '\n\033[36m==> %s\033[0m\n' "$*"; }
kc() { kubectl --context "$CONTEXT" "$@"; }

for tool in podman kind kubectl; do
  if ! command -v "$tool" >/dev/null; then
    echo "$tool not found on PATH (need podman, kind and kubectl)" >&2
    exit 1
  fi
done

if [[ "$(uname -s)" != "Linux" ]]; then
  step "Podman machine"
  if [[ -z "$(podman machine list --format '{{.Name}}')" ]]; then
    podman machine init --rootful --now
  else
    name=$(podman machine list --format '{{.Name}} {{.Default}}' | awk '$2 == "true" { print $1; exit }')
    name=${name:-$(podman machine list --format '{{.Name}}' | head -n 1)}
    name=${name%\*}
    state=$(podman machine inspect "$name" --format '{{.State}}')
    if [[ "$(podman machine inspect "$name" --format '{{.Rootful}}')" != "true" ]]; then
      if [[ "$state" == "running" ]]; then podman machine stop "$name"; fi
      podman machine set --rootful "$name"
      podman machine start "$name"
    elif [[ "$state" != "running" ]]; then
      podman machine start "$name"
    else
      echo "machine '$name' is running (rootful)"
    fi
  fi
fi

step "kind cluster '$CLUSTER'"
exists=false
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then exists=true; fi
if $exists && $RECREATE; then
  kind delete cluster --name "$CLUSTER"
  exists=false
fi
if $exists; then
  echo "reusing existing cluster"
else
  kind create cluster --config "$ROOT/infra/k8s/overlays/kind/cluster.yaml" --wait 180s
fi

step "Build $IMAGE with Podman"
podman build --tag "$IMAGE" --file "$ROOT/Containerfile" "$ROOT"

step "Load image into kind"
archive=$(mktemp -t nexusgate-api.XXXXXX)
secret_file=$(mktemp -t nexusgate-secrets.XXXXXX)
trap 'rm -f "$archive" "$secret_file"' EXIT
podman save --format docker-archive --output "$archive" "$IMAGE"
kind load image-archive "$archive" --name "$CLUSTER"

step "Secret nexusgate-secrets"
kc apply -f "$ROOT/infra/k8s/base/namespace.yaml"

dotenv_value() {
  [[ -f "$ROOT/.env" ]] || return 0
  sed -nE "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*([^#]*).*$/\1/p" "$ROOT/.env" | tail -n 1 |
    sed -E "s/[[:space:]]+$//; s/^[\"']//; s/[\"']$//"
}
existing_value() {
  kc -n "$NS" get secret nexusgate-secrets -o "jsonpath={.data.$1}" 2>/dev/null | base64 -d 2>/dev/null || true
}
new_secret() { head -c 32 /dev/urandom | base64 | tr '+/' '-_' | tr -d '=\n'; }

admin_token=""
providers=()
for key in NEXUSGATE_ADMIN_TOKEN NEXUSGATE_JWT_SECRET ANTHROPIC_API_KEY OPENAI_API_KEY; do
  value=$(dotenv_value "$key")
  if [[ -z "$value" || "$value" == change-me* ]]; then value=$(existing_value "$key"); fi
  if [[ -z "$value" && "$key" == NEXUSGATE_* ]]; then value=$(new_secret); fi  # prod refuses defaults
  if [[ -n "$value" ]]; then
    printf '%s=%s\n' "$key" "$value" >>"$secret_file"
    if [[ "$key" == *_API_KEY ]]; then providers+=("$key"); fi
  fi
  if [[ "$key" == NEXUSGATE_ADMIN_TOKEN ]]; then admin_token=$value; fi
done
# Through a file rather than --from-literal, so values don't appear in the process list.
kc -n "$NS" create secret generic nexusgate-secrets --from-env-file="$secret_file" \
  --dry-run=client -o yaml | kc apply -f -
echo "provider keys: ${providers[*]:-none (mock and chaos routes only)}"

step "Deploy (kustomize overlay infra/k8s/overlays/kind)"
kc apply -k "$ROOT/infra/k8s/overlays/kind"
# The tag is always :dev, so restart the API onto the image that was just loaded.
kc -n "$NS" rollout restart deployment/api
for workload in statefulset/redis deployment/api deployment/prometheus deployment/grafana; do
  kc -n "$NS" rollout status "$workload" --timeout=300s
done

step "NexusGate is up"
cat <<EOF
  API docs    http://localhost:8000/docs
  Prometheus  http://localhost:9090
  Grafana     http://localhost:3000
  Admin token $admin_token

  Smoke test  python scripts/smoke_test.py --admin-token <token> --prometheus-url http://localhost:9090
  Tear down   scripts/cluster-down.sh
EOF
