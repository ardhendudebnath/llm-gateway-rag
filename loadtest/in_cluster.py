"""Run a load or chaos test from inside the kind cluster, and bring the result back.

    python loadtest/in_cluster.py --label baseline                      # stepped load test
    python loadtest/in_cluster.py --label baseline -- --levels 50 100   # extra args for the script
    python loadtest/in_cluster.py --script chaos.py --label chaos       # chaos test

Why inside: driven from a Windows host, every request also crosses Podman's port forwarder,
which added ~2 s to about 5% of requests (server-side p95 96 ms, client-side 2,100 ms). Inside
the cluster, traffic goes pod to pod through the `api` Service, as a real client's would.

It packs loadtest/*.py and the eval corpus into ConfigMaps, runs the stock Locust image as a Job
in the nexusgate namespace (the admin token comes from the existing Secret, never the command
line), streams its output, and saves the `RESULT_JSON:` line to loadtest/results/<label>.json.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / "eval" / "corpus"
RESULTS = HERE / "results"
CONTEXT = "kind-nexusgate"
NAMESPACE = "nexusgate"
IMAGE = "docker.io/locustio/locust:2.46.6"


def kubectl(*args: str, stdin: str | None = None, check: bool = True) -> str:
    proc = subprocess.run(
        ["kubectl", "--context", CONTEXT, "-n", NAMESPACE, *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode:
        raise SystemExit(f"kubectl {' '.join(args)} failed:\n{proc.stderr}")
    return proc.stdout


def apply_configmap(name: str, files: list[Path]) -> None:
    manifest = kubectl(
        "create", "configmap", name, "--dry-run=client", "-o", "yaml",
        *[f"--from-file={p.name}={p}" for p in files],
    )  # fmt: skip
    kubectl("apply", "-f", "-", stdin=manifest)


def job(name: str, script: str, script_args: list[str]) -> dict:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "labels": {"app.kubernetes.io/name": "loadtest"}},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "metadata": {"labels": {"app.kubernetes.io/name": "loadtest"}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "loadtest",
                            "image": IMAGE,
                            "command": [
                                "python",
                                f"/loadtest/{script}",
                                "--base-url",
                                "http://api:8000",
                                "--corpus",
                                "/corpus",
                                "--results-dir",
                                "/tmp/results",
                                *script_args,
                            ],
                            "workingDir": "/tmp",
                            "env": [
                                {
                                    "name": "NEXUSGATE_ADMIN_TOKEN",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "nexusgate-secrets",
                                            "key": "NEXUSGATE_ADMIN_TOKEN",
                                        }
                                    },
                                },
                                {"name": "PYTHONUNBUFFERED", "value": "1"},
                            ],
                            "resources": {"requests": {"cpu": "1", "memory": "512Mi"}},
                            "volumeMounts": [
                                {"name": "scripts", "mountPath": "/loadtest"},
                                {"name": "corpus", "mountPath": "/corpus"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "scripts", "configMap": {"name": "loadtest-scripts"}},
                        {"name": "corpus", "configMap": {"name": "loadtest-corpus"}},
                    ],
                },
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--script", default="run_load.py", choices=["run_load.py", "chaos.py"])
    parser.add_argument("--timeout", type=int, default=1800, help="seconds to wait for the job")
    args, script_args = parser.parse_known_args()
    script_args = [a for a in script_args if a != "--"]

    apply_configmap("loadtest-scripts", sorted(HERE.glob("*.py")))
    apply_configmap("loadtest-corpus", sorted(CORPUS.glob("*.md")))

    name = f"loadtest-{args.label}".lower().replace("_", "-")[:52]
    kubectl("delete", "job", name, "--ignore-not-found", "--wait=true")
    kubectl(
        "apply", "-f", "-",
        stdin=json.dumps(job(name, args.script, ["--label", args.label, *script_args])),
    )  # fmt: skip
    print(f"job {name} started ({args.script}); streaming its output", flush=True)

    printed = 0
    deadline = time.monotonic() + args.timeout
    while True:
        logs = kubectl("logs", f"job/{name}", check=False).splitlines()
        for line in logs[printed:]:
            if not line.startswith("RESULT_JSON:"):
                print("  " + line, flush=True)
        printed = len(logs)
        status = kubectl("get", "job", name, "-o", "jsonpath={.status.succeeded},{.status.failed}")
        succeeded, _, failed = status.partition(",")
        if succeeded == "1" or failed == "1" or time.monotonic() > deadline:
            break
        time.sleep(5)

    result_lines = [line for line in logs if line.startswith("RESULT_JSON:")]
    if succeeded != "1" or not result_lines:
        print(f"job {name} did not succeed (succeeded={succeeded!r}, failed={failed!r})")
        return 1
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.label}.json"
    result = json.loads(result_lines[-1].removeprefix("RESULT_JSON:"))
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
