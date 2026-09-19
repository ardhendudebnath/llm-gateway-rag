#!/usr/bin/env bash
# Delete the local kind cluster. Add --stop-machine to also stop the Podman machine (macOS/Windows).
set -euo pipefail
export KIND_EXPERIMENTAL_PROVIDER=podman

kind delete cluster --name nexusgate
if [[ "${1:-}" == "--stop-machine" ]]; then podman machine stop; fi
