"""Generate a large distractor corpus, so retrieval is measured against real competition.

The hand-written corpus is 13 documents and about 70 chunks. At that size retrieval is not a hard
problem: any method puts the right passage in the top 5 of 70, and every config scores recall@5
1.000, which measures nothing. This module writes several hundred *more* documents in the same
shapes — service catalogues, error-code tables, configuration references, alert catalogues, runbooks
— so a question about `NG-1017` competes with thousands of rows that look almost exactly like it.

    python -m eval.generate_corpus --out /tmp/corpus-large --services 400 --codes 600

The filler is deterministic (one seed) and reproducible, and it is generated rather than committed
so the repository does not carry a megabyte of invented prose. Two rules keep the labels valid:

* every identifier used by a labelled question is reserved and never generated, so a question keeps
  exactly one correct answer;
* the hand-written documents are copied in unchanged, so the labels still point at real text.

`eval/retrieval_eval.py --corpus large` calls this, and the eval prints the corpus it used.
"""

import argparse
import random
import shutil
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"

TEAMS = ["Platform", "Identity", "Revenue", "Knowledge", "Growth", "Compliance", "Mobile", "Data"]
REGIONS = ["eu-central", "eu-west", "us-east", "us-west", "ap-south", "ap-northeast", "sa-east"]
NOUNS = [
    "ledger",
    "invoice",
    "receipt",
    "webhook",
    "digest",
    "roster",
    "catalogue",
    "indexer",
    "scheduler",
    "sweeper",
    "reconciler",
    "projector",
    "collector",
    "forwarder",
    "validator",
    "enricher",
    "deduplicator",
    "partitioner",
    "compactor",
    "replayer",
    "notifier",
    "exporter",
    "importer",
    "auditor",
    "throttler",
    "gatekeeper",
    "planner",
    "packer",
    "router",
    "mailer",
]
QUALIFIERS = [
    "batch",
    "stream",
    "edge",
    "core",
    "bulk",
    "delta",
    "legacy",
    "internal",
    "public",
    "async",
    "priority",
    "regional",
    "shared",
    "tenant",
    "billing",
    "search",
    "media",
    "report",
]
ACTIONS = [
    "Retry after the interval in the response",
    "Fix the payload and send it again",
    "Use a key issued for this tenant",
    "Reduce the request size and retry",
    "Wait for the next billing period",
    "Contact the owning team with the request id",
    "Nothing; the gateway recovers on its own",
    "Check the job record and re-upload",
]
SYMPTOMS = [
    "the request was rejected by the upstream service",
    "the record was already processed by another worker",
    "the tenant has no active subscription for this feature",
    "the payload referenced an object that has been deleted",
    "the field is present but outside the allowed range",
    "the operation would exceed the tenant's storage quota",
    "the downstream queue refused the message",
    "the credential has expired and must be rotated",
]


def reserved_identifiers(qa_file: Path) -> set[str]:
    """Identifiers a labelled question depends on. Generating one would create two right answers."""
    import json

    reserved: set[str] = set()
    for line in qa_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        question = json.loads(line)
        for text in [question["question"], *question["evidence"]]:
            for token in text.replace("|", " ").split():
                token = token.strip("`,.?()")
                if any(c.isdigit() for c in token) or "-" in token or "_" in token:
                    reserved.add(token.upper())
    return reserved


def service_rows(rng: random.Random, count: int, reserved: set[str]) -> list[tuple[str, ...]]:
    rows, seen = [], set()
    while len(rows) < count:
        name = f"{rng.choice(QUALIFIERS)}-{rng.choice(NOUNS)}"
        if name in seen or name.upper() in reserved:
            continue
        seen.add(name)
        tier = rng.choice([1, 2, 2, 3, 3])
        target = {1: "99.95%", 2: "99.5%", 3: "99.0%"}[tier]
        rows.append((name, rng.choice(TEAMS), str(tier), target))
    return rows


def write_service_catalogues(out: Path, rng: random.Random, rows: list[tuple[str, ...]]) -> int:
    per_doc, written = 25, 0
    for start in range(0, len(rows), per_doc):
        chunk = rows[start : start + per_doc]
        region = REGIONS[(start // per_doc) % len(REGIONS)]
        index = start // per_doc
        lines = [
            f"# Service catalogue — {region} shard {index + 1}",
            "",
            "Services deployed in this shard. Tier 1 pages at any hour, tier 2 during business",
            "hours, tier 3 becomes a ticket for the owning team.",
            "",
            "| Service | Owner | Tier | Availability target |",
            "|---|---|---|---|",
        ]
        lines += [f"| {name} | {team} | {tier} | {target} |" for name, team, tier, target in chunk]
        lines += [
            "",
            "## Ownership",
            "",
            f"A service in {region} without a named owning team cannot be deployed: the pipeline",
            "checks this catalogue and fails the build.",
            "",
        ]
        (out / f"service-catalogue-{region}-{index + 1}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        written += 1
    return written


def write_error_codes(out: Path, rng: random.Random, count: int, reserved: set[str]) -> int:
    # The labels use NG-1xxx and NG-2xxx, so the filler lives in other ranges entirely.
    codes = [f"NG-{prefix}{n:03d}" for prefix in (3, 4, 5, 6, 7) for n in range(1, 200)]
    codes = [c for c in codes if c.upper() not in reserved][:count]
    per_doc, written = 40, 0
    for start in range(0, len(codes), per_doc):
        chunk = codes[start : start + per_doc]
        lines = [
            f"# API error codes {chunk[0]} to {chunk[-1]}",
            "",
            "Stable codes returned in `error.code`. Branch on the code, never on the message.",
            "",
            "| Code | Meaning | What the caller should do |",
            "|---|---|---|",
        ]
        for code in chunk:
            lines.append(
                f"| {code} | {rng.choice(SYMPTOMS).capitalize()} | {rng.choice(ACTIONS)} |"
            )
        lines.append("")
        (out / f"error-codes-{chunk[0].lower()}.md").write_text("\n".join(lines), encoding="utf-8")
        written += 1
    return written


def write_config_references(out: Path, rng: random.Random, count: int, reserved: set[str]) -> int:
    prefixes = ["WORKER", "INDEXER", "EXPORTER", "MAILER", "SCHEDULER", "COLLECTOR", "GATEWAY"]
    suffixes = [
        "MAX_CONCURRENCY",
        "MAX_QUEUE",
        "BATCH_SIZE",
        "TIMEOUT_SECONDS",
        "RETRY_LIMIT",
        "BACKOFF_SECONDS",
        "POOL_SIZE",
        "BUFFER_BYTES",
        "FLUSH_INTERVAL_SECONDS",
        "SHARD_COUNT",
        "TTL_SECONDS",
        "MAX_ATTEMPTS",
        "WINDOW_SECONDS",
        "THRESHOLD_PERCENT",
    ]
    names = [
        f"{prefix}_{suffix}"
        for prefix in prefixes
        for suffix in suffixes
        if f"{prefix}_{suffix}" not in reserved
    ][:count]
    per_doc, written = 30, 0
    for start in range(0, len(names), per_doc):
        chunk = names[start : start + per_doc]
        lines = [
            f"# Configuration reference — {chunk[0].split('_')[0].lower()} components, part "
            f"{start // per_doc + 1}",
            "",
            "Read from the environment. Defaults are what the production cluster runs.",
            "",
            "| Setting | Default | Effect |",
            "|---|---|---|",
        ]
        for name in chunk:
            default = rng.choice([1, 2, 4, 8, 16, 30, 32, 60, 64, 128, 300, 500, 1000, 3600])
            lines.append(
                f"| {name} | {default} | {rng.choice(['Bounds', 'Caps', 'Limits', 'Sets'])} "
                f"{name.split('_', 1)[1].lower().replace('_', ' ')} for this component |"
            )
        lines.append("")
        (out / f"config-{chunk[0].lower()}.md").write_text("\n".join(lines), encoding="utf-8")
        written += 1
    return written


def write_alert_catalogues(out: Path, rng: random.Random, count: int, reserved: set[str]) -> int:
    subjects = ["Queue", "Cache", "Index", "Export", "Mail", "Shard", "Replica", "Snapshot"]
    conditions = ["Lag", "Depth", "Errors", "Latency", "Restarts", "Staleness", "Rejections"]
    runbooks = [f"RB-{n}" for n in range(200, 900) if f"RB-{n}" not in reserved]
    alerts = [f"{s}{c}High" for s in subjects for c in conditions][:count]
    per_doc, written = 20, 0
    for start in range(0, len(alerts), per_doc):
        chunk = alerts[start : start + per_doc]
        lines = [
            f"# Alert catalogue, part {start // per_doc + 1}",
            "",
            "Each alert names a runbook; an alert without one may not page anyone.",
            "",
            "| Alert | Fires when | For | Severity | Runbook |",
            "|---|---|---|---|---|",
        ]
        for i, alert in enumerate(chunk):
            lines.append(
                f"| {alert} | the measured value exceeds its budget by "
                f"{rng.choice([10, 20, 25, 50, 100])}% | {rng.choice([2, 5, 10, 15, 30])} minutes "
                f"| {rng.choice(['warning', 'critical'])} | {runbooks[start + i]} |"
            )
        lines.append("")
        (out / f"alerts-{start // per_doc + 1}.md").write_text("\n".join(lines), encoding="utf-8")
        written += 1
    return written


def generate(
    out: Path,
    services: int = 400,
    codes: int = 600,
    settings: int = 300,
    alerts: int = 200,
    seed: int = 20260926,
    qa_file: Path | None = None,
) -> dict:
    """Write the hand-written corpus plus deterministic filler into `out`."""
    out.mkdir(parents=True, exist_ok=True)
    for path in out.glob("*.md"):
        path.unlink()
    for path in sorted(CORPUS_DIR.glob("*.md")):
        shutil.copy2(path, out / path.name)

    reserved = reserved_identifiers(qa_file or EVAL_DIR / "labeled_qa.jsonl")
    rng = random.Random(seed)
    written = {
        "hand-written": len(list(CORPUS_DIR.glob("*.md"))),
        "service catalogues": write_service_catalogues(
            out, rng, service_rows(rng, services, reserved)
        ),
        "error code tables": write_error_codes(out, rng, codes, reserved),
        "config references": write_config_references(out, rng, settings, reserved),
        "alert catalogues": write_alert_catalogues(out, rng, alerts, reserved),
    }
    words = sum(len(p.read_text(encoding="utf-8").split()) for p in out.glob("*.md"))
    return {"documents": len(list(out.glob("*.md"))), "words": words, "written": written}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--services", type=int, default=400)
    parser.add_argument("--codes", type=int, default=600)
    parser.add_argument("--settings", type=int, default=300)
    parser.add_argument("--alerts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    summary = generate(args.out, args.services, args.codes, args.settings, args.alerts, args.seed)
    print(f"{summary['documents']} documents, {summary['words']} words -> {args.out}")
    for what, count in summary["written"].items():
        print(f"  {count:4} {what}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
