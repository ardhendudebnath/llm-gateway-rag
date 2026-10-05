"""The Helm chart: what it renders, what it refuses, and that it stays in step with the base stack.

Rendering needs the `helm` binary; those tests skip without it (CI's runners have it). The check
that the chart's copies of the alert rules and dashboard match the originals needs nothing, and
always runs — two copies of one file is exactly what drifts silently.
"""

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from app import __version__

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "infra" / "helm" / "nexusgate"
BASE_CONFIG = ROOT / "infra" / "k8s" / "base" / "config"
CI_VALUES = CHART / "ci" / "test-values.yaml"

needs_helm = pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")

SECRETS = [
    "--set",
    "secrets.adminToken=t",
    "--set",
    "secrets.jwtSecret=" + "j" * 40,
    "--set",
    "secrets.qdrantApiKey=q",
]


def render(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["helm", "template", "rel", str(CHART), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def objects(*args: str) -> list[dict]:
    result = render(*args)
    assert result.returncode == 0, result.stderr
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def by_kind(docs: list[dict], kind: str) -> list[dict]:
    return [d for d in docs if d["kind"] == kind]


def config_of(docs: list[dict]) -> dict:
    (config,) = [d for d in by_kind(docs, "ConfigMap") if d["metadata"]["name"].endswith("-config")]
    return config["data"]


@pytest.mark.parametrize(
    ("copy", "original"),
    [("alert-rules.yml", "alert-rules.yml"), ("grafana-dashboard.json", "grafana-dashboard.json")],
)
def test_the_charts_copies_match_the_base_stack(copy, original):
    """Helm cannot read outside the chart, so these are copies; this is what keeps them honest."""
    assert (CHART / "files" / copy).read_bytes() == (BASE_CONFIG / original).read_bytes(), (
        f"copy infra/k8s/base/config/{original} to infra/helm/nexusgate/files/{copy}"
    )


def test_the_chart_and_the_app_share_one_release_version():
    """A release tag publishes the chart and the image under one version, and CI refuses a tag
    that disagrees with pyproject.toml. The chart pulls the image tagged with its appVersion: if
    these drift, a published chart installs another build of the app, or one never published."""
    chart = yaml.safe_load((CHART / "Chart.yaml").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert chart["version"] == chart["appVersion"] == project["version"] == __version__


@needs_helm
def test_an_install_without_secrets_is_refused_with_a_reason():
    result = render()

    assert result.returncode != 0
    assert "secrets.adminToken is required" in result.stderr


@needs_helm
def test_a_short_jwt_secret_is_refused():
    result = render(
        "--set",
        "secrets.adminToken=t",
        "--set",
        "secrets.jwtSecret=short",
        "--set",
        "secrets.qdrantApiKey=q",
    )

    assert result.returncode != 0
    assert "at least 32 characters" in result.stderr


@needs_helm
def test_the_default_install_has_everything_it_needs_and_nothing_optional():
    docs = objects(*SECRETS)
    kinds = sorted(d["kind"] for d in docs)

    assert kinds.count("Deployment") == 2, "api and worker"
    assert kinds.count("StatefulSet") == 2, "bundled redis and qdrant"
    assert kinds.count("HorizontalPodAutoscaler") == 1, "the api's; the worker's needs an adapter"
    assert "PodDisruptionBudget" in kinds
    for optional in ("Ingress", "ServiceMonitor", "PrometheusRule"):
        assert optional not in kinds, f"{optional} must be opt-in"


@needs_helm
def test_a_default_install_pulls_the_published_image_of_its_own_release():
    pods = [d["spec"]["template"]["spec"] for d in by_kind(objects(*SECRETS), "Deployment")]
    images = {c["image"] for pod in pods for c in pod["containers"] + pod.get("initContainers", [])}

    assert images == {f"ghcr.io/ardhendudebnath/nexusgate-api:{__version__}"}


@needs_helm
def test_every_optional_piece_renders_when_switched_on():
    docs = objects("-f", str(CI_VALUES))
    kinds = {d["kind"] for d in docs}

    assert {"Ingress", "ServiceMonitor", "PrometheusRule"} <= kinds
    hpas = {d["metadata"]["name"] for d in by_kind(docs, "HorizontalPodAutoscaler")}
    assert hpas == {"rel-nexusgate-api", "rel-nexusgate-worker"}, "the worker's is opt-in"
    (worker_hpa,) = [d for d in by_kind(docs, "HorizontalPodAutoscaler") if "worker" in str(d)]
    assert worker_hpa["spec"]["metrics"][0]["external"]["metric"]["name"] == (
        "nexusgate_ingest_queue_depth"
    ), "scales on queue depth, never CPU"


@needs_helm
def test_the_shipped_retrieval_and_breaker_settings_reach_the_pods():
    data = config_of(objects(*SECRETS))

    assert data["NEXUSGATE_RETRIEVAL_HYBRID"] == "true"
    assert data["NEXUSGATE_RAG_RERANK_CANDIDATES"] == "12"
    assert data["NEXUSGATE_BREAKER_SHARED"] == "true"
    assert data["NEXUSGATE_REDIS_URL"] == "redis://rel-nexusgate-redis:6379/0"
    assert data["NEXUSGATE_QDRANT_URL"] == "http://rel-nexusgate-qdrant:6333"


@needs_helm
def test_external_datastores_replace_the_bundled_ones():
    docs = objects(
        *SECRETS,
        "--set",
        "redis.enabled=false",
        "--set",
        "redis.externalUrl=rediss://cache.example:6380/0",
        "--set",
        "qdrant.enabled=false",
        "--set",
        "qdrant.externalUrl=https://vectors.example:6333",
    )

    assert not by_kind(docs, "StatefulSet")
    data = config_of(docs)
    assert data["NEXUSGATE_REDIS_URL"] == "rediss://cache.example:6380/0"
    assert data["NEXUSGATE_QDRANT_URL"] == "https://vectors.example:6333"


@needs_helm
def test_disabling_a_datastore_without_saying_where_it_is_is_refused():
    result = render(*SECRETS, "--set", "redis.enabled=false")

    assert result.returncode != 0
    assert "redis.externalUrl is required" in result.stderr


@needs_helm
def test_an_existing_secret_is_referenced_and_none_is_created():
    docs = objects("--set", "secrets.existingSecret=managed-elsewhere")

    assert not by_kind(docs, "Secret")
    for deployment in by_kind(docs, "Deployment"):
        refs = deployment["spec"]["template"]["spec"]["containers"][0]["envFrom"]
        assert {"secretRef": {"name": "managed-elsewhere"}} in refs


@needs_helm
def test_the_autoscaler_owns_the_replica_count_when_it_is_on():
    docs = objects(*SECRETS)
    (api,) = [d for d in by_kind(docs, "Deployment") if d["metadata"]["name"].endswith("-api")]

    assert "replicas" not in api["spec"], "a fixed count would fight the HPA on every apply"

    docs = objects(
        *SECRETS, "--set", "api.autoscaling.enabled=false", "--set", "api.replicaCount=3"
    )
    (api,) = [d for d in by_kind(docs, "Deployment") if d["metadata"]["name"].endswith("-api")]
    assert api["spec"]["replicas"] == 3


@needs_helm
def test_a_single_replica_gets_no_disruption_budget():
    """minAvailable 1 on one replica would block every node drain forever."""
    docs = objects(
        *SECRETS, "--set", "api.autoscaling.enabled=false", "--set", "api.replicaCount=1"
    )

    assert not by_kind(docs, "PodDisruptionBudget")


@needs_helm
def test_changing_config_rolls_the_pods():
    def checksum(*extra):
        docs = objects(*SECRETS, *extra)
        (api,) = [d for d in by_kind(docs, "Deployment") if d["metadata"]["name"].endswith("-api")]
        return api["spec"]["template"]["metadata"]["annotations"]["checksum/config"]

    assert checksum() != checksum("--set", "config.NEXUSGATE_LOG_LEVEL=DEBUG")


@needs_helm
def test_every_container_runs_without_root_or_privilege_escalation():
    docs = objects(*SECRETS)
    for deployment in by_kind(docs, "Deployment"):
        spec = deployment["spec"]["template"]["spec"]
        assert spec["securityContext"]["runAsNonRoot"] is True
        for container in spec["containers"] + spec.get("initContainers", []):
            assert container["securityContext"]["allowPrivilegeEscalation"] is False
            assert container["securityContext"]["readOnlyRootFilesystem"] is True


@needs_helm
def test_scraped_targets_keep_the_job_name_the_alert_rules_select_on():
    docs = objects("-f", str(CI_VALUES))
    (monitor,) = by_kind(docs, "ServiceMonitor")
    relabel = monitor["spec"]["endpoints"][0]["relabelings"][0]

    assert relabel == {"targetLabel": "job", "replacement": "nexusgate-api"}
    (rule,) = by_kind(docs, "PrometheusRule")
    assert rule["spec"]["groups"], "the rules file's groups became the PrometheusRule spec"
