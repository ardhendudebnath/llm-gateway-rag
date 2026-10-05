"""Run an API image the way the Helm chart runs it, then smoke-test it.

    python scripts/check_image.py localhost/nexusgate-api:ci
    python scripts/check_image.py ghcr.io/ardhendudebnath/nexusgate-api:0.2.0

Renders infra/helm/nexusgate and starts what it describes in one Podman pod: the bundled Redis
Stack and Qdrant, the dependency wait, the API and a Celery worker. Each container gets the
environment, command, user and security context the chart gives it (non-root, read-only root
filesystem, no capabilities), and the chart's Service names resolve to the pod's shared localhost.
Nothing is copied out of the chart by hand, so this cannot drift from it. Then it runs
scripts/smoke_test.py against the API.

CI does this to every image before publishing it: a published image has already served traffic
with the chart's production settings. Needs podman and helm on PATH, and httpx and PyYAML.
Secrets are generated per run and never printed; the pod is removed afterwards unless --keep.
"""

import argparse
import base64
import os
import secrets
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "infra" / "helm" / "nexusgate"
RELEASE = "check"
POD = "nexusgate-check"


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    # Arguments can carry the generated secrets, so a failure reports the command, not its args.
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise SystemExit(f"`{args[0]} {args[1]}` failed:\n{result.stderr.strip()}")
    return result


def render(chart_secrets: dict[str, str]) -> list[dict]:
    sets = [a for key, value in chart_secrets.items() for a in ("--set", f"secrets.{key}={value}")]
    out = run("helm", "template", RELEASE, str(CHART), *sets).stdout
    return [doc for doc in yaml.safe_load_all(out) if doc]


def workloads(objects: list[dict]) -> dict[str, dict]:
    """Deployments and StatefulSets by component: api, worker, redis, qdrant."""
    prefix = f"{RELEASE}-nexusgate-"
    return {
        d["metadata"]["name"].removeprefix(prefix): d
        for d in objects
        if d["kind"] in {"Deployment", "StatefulSet"}
    }


def secret_data(secret: dict) -> dict[str, str]:
    if "stringData" in secret:
        return secret["stringData"]
    return {k: base64.b64decode(v).decode() for k, v in secret.get("data", {}).items()}


def environment(container: dict, objects: list[dict]) -> dict[str, str]:
    """What envFrom and env give a container, resolved against the rendered ConfigMap and Secret."""
    config = {d["metadata"]["name"]: d["data"] for d in objects if d["kind"] == "ConfigMap"}
    secret = {d["metadata"]["name"]: secret_data(d) for d in objects if d["kind"] == "Secret"}
    env: dict[str, str] = {}
    for source in container.get("envFrom", []):
        if "configMapRef" in source:
            env.update(config[source["configMapRef"]["name"]])
        else:
            env.update(secret[source["secretRef"]["name"]])
    for var in container.get("env", []):
        if "value" in var:
            env[var["name"]] = var["value"]
        else:
            ref = var["valueFrom"]["secretKeyRef"]
            env[var["name"]] = secret[ref["name"]][ref["key"]]
    return env


def podman_run(
    name: str, workload: dict, container: dict, objects: list[dict], *, image: str | None = None
) -> list[str]:
    """`podman run` arguments for one container of a Deployment or StatefulSet, as rendered."""
    pod_spec = workload["spec"]["template"]["spec"]
    context = {**pod_spec.get("securityContext", {}), **container.get("securityContext", {})}
    detach = name not in {c["name"] for c in pod_spec.get("initContainers", [])}
    args = ["podman", "run", "--pod", POD, "--name", f"{POD}-{name}"]
    args.append("--detach" if detach else "--rm")
    if "runAsUser" in context:
        user = str(context["runAsUser"])
        args += ["--user", f"{user}:{context['runAsGroup']}" if "runAsGroup" in context else user]
    if context.get("readOnlyRootFilesystem"):
        args.append("--read-only")
    if "ALL" in context.get("capabilities", {}).get("drop", []):
        args += ["--cap-drop", "all"]
    if context.get("allowPrivilegeEscalation") is False:
        args += ["--security-opt", "no-new-privileges"]
    # emptyDirs and volume claims alike become throwaway tmpfs mounts.
    for mount in container.get("volumeMounts", []):
        args += ["--tmpfs", f"{mount['mountPath']}:rw,mode=1777"]
    for key, value in environment(container, objects).items():
        args += ["--env", f"{key}={value}"]
    args.append(image or container["image"])
    return args + container.get("command", []) + container.get("args", [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="the API image to check, e.g. localhost/nexusgate-api:ci")
    parser.add_argument("--port", type=int, default=8000, help="host port for the API")
    parser.add_argument("--keep", action="store_true", help="leave the pod running afterwards")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with the smoke test's

    chart_secrets = {
        "adminToken": secrets.token_urlsafe(32),
        "jwtSecret": secrets.token_urlsafe(48),
        "qdrantApiKey": secrets.token_urlsafe(32),
    }
    objects = render(chart_secrets)
    workload = workloads(objects)
    api = workload["api"]
    (api_container,) = api["spec"]["template"]["spec"]["containers"]
    api_env = environment(api_container, objects)

    print(f"Running {args.image} as the Helm chart does")
    run("podman", "pod", "rm", "--force", "--ignore", POD)
    pod = ["podman", "pod", "create", "--name", POD]
    pod += ["--publish", f"{args.port}:{api_container['ports'][0]['containerPort']}"]
    for url in (api_env["NEXUSGATE_REDIS_URL"], api_env["NEXUSGATE_QDRANT_URL"]):
        pod += ["--add-host", f"{urlparse(url).hostname}:127.0.0.1"]
    run(*pod)

    try:
        for component in ("redis", "qdrant"):
            (container,) = workload[component]["spec"]["template"]["spec"]["containers"]
            run(*podman_run(component, workload[component], container, objects))
            print(f"  started {component} ({container['image']})")
        for init in api["spec"]["template"]["spec"].get("initContainers", []):
            run(*podman_run(init["name"], api, init, objects, image=args.image))
            print(f"  {init['name']}: done")
        for component in ("api", "worker"):
            (container,) = workload[component]["spec"]["template"]["spec"]["containers"]
            run(*podman_run(component, workload[component], container, objects, image=args.image))
            print(f"  started {component}")

        base_url = f"http://localhost:{args.port}"
        smoke = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "smoke_test.py"), "--base-url", base_url],
            env={**os.environ, "NEXUSGATE_ADMIN_TOKEN": chart_secrets["adminToken"]},
            check=False,
        )
        if smoke.returncode != 0:
            for component in ("api", "worker"):
                logs = run("podman", "logs", "--tail", "40", f"{POD}-{component}", check=False)
                print(f"\n--- {component} logs\n{logs.stdout}{logs.stderr}", file=sys.stderr)
        return smoke.returncode
    finally:
        if not args.keep:
            run("podman", "pod", "rm", "--force", POD, check=False)


if __name__ == "__main__":
    sys.exit(main())
