"""Render loadtest/results/*.json into loadtest/RESULTS.md.

    python loadtest/report.py --runs baseline after-fix after-wait-budget --chaos chaos

Each run is one `in_cluster.py` load test; they are shown side by side, in the order given, so
the effect of each change is visible.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def load(label: str) -> dict | None:
    path = RESULTS / f"{label}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def ceiling(run: dict) -> str:
    c = run.get("ceiling")
    return f"**{c['rps']} req/s** at {c['users']} users" if c else "**none**"


# Per-class objectives. The runs themselves use one 500 ms objective over the whole mix; RAG can't
# meet it on this hardware even unloaded (see README), so results are also judged per class.
CLASS_SLOS = {"chat": 500.0, "rag": 1000.0}

# How long a whole stream takes is a generation length, not a latency: holding it to a 500 ms
# objective would say a gateway that streams a long answer is failing. Time to first token is the
# figure that belongs to the chat class, and it is measured under its own name.
NOT_A_LATENCY = {"chat: stream complete"}


def class_ceiling(run: dict, prefix: str) -> str:
    """Highest total throughput at which every request kind of a class meets its objective."""
    slo = CLASS_SLOS[prefix]
    passing = [
        lv
        for lv in run["levels"]
        if all(
            k["p95_ms"] <= slo and k["fail_pct"] < 1
            for name, k in lv["by_kind"].items()
            if name.startswith(prefix) and name not in NOT_A_LATENCY
        )
    ]
    if not passing:
        return "none"
    best = max(passing, key=lambda lv: lv["rps"])
    return f"{best['rps']} req/s ({best['users']} users)"


def by_users(run: dict) -> dict[int, dict]:
    """The last level measured at each user count.

    A run may repeat a level (`--levels 100 100 200 200`); `repetitions_section` is what reports
    those properly. This keeps the side-by-side tables one row per load level.
    """
    return {lv["users"]: lv for lv in run["levels"]}


def repetitions_section(label: str, run: dict) -> list[str]:
    """Per-level spread, when a run measured the same load more than once.

    Beyond saturation this stack's throughput is not a single number: repeated 200-user levels on
    one laptop ranged 66.8 to 120.1 req/s with identical code, which is wide enough to swamp the
    effect of any change one might want to measure. A run that repeats its levels can say so.
    """
    groups: dict[int, list[dict]] = {}
    for lv in run["levels"]:
        groups.setdefault(lv["users"], []).append(lv)
    if all(len(g) < 2 for g in groups.values()):
        return []
    lines = [
        f"## Repeated levels (`{label}`)",
        "",
        f"The same load measured {max(len(g) for g in groups.values())} times, interleaved, at "
        f"{run['duration_s']} s per level. Spread is (max - min) / median: where it is large, a "
        "single sample is not a result.",
        "",
        "| Users | runs | req/s median | req/s range | spread | p95 median | p95 range | errors |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for users, levels in sorted(groups.items()):
        rps = sorted(lv["rps"] for lv in levels)
        p95 = sorted(lv["p95_ms"] for lv in levels)
        worst_errors = max(lv["fail_pct"] for lv in levels)
        median_rps = statistics.median(rps)
        spread = (rps[-1] - rps[0]) / median_rps if median_rps else 0
        lines.append(
            f"| {users} | {len(levels)} | {median_rps:.1f} | {rps[0]:.1f} to {rps[-1]:.1f} "
            f"| {spread:.0%} | {statistics.median(p95):.0f} ms "
            f"| {p95[0]:.0f} to {p95[-1]:.0f} ms | {worst_errors}% |"
        )
    return lines


def comparison(runs: dict[str, dict], cell) -> list[str]:
    labels = list(runs)
    users = sorted({u for run in runs.values() for u in by_users(run)})
    rows = [
        "| Users | " + " | ".join(labels) + " |",
        "|---:|" + "---:|" * len(labels),
    ]
    for u in users:
        cells = [cell(by_users(runs[label]).get(u)) for label in labels]
        rows.append(f"| {u} | " + " | ".join(cells) + " |")
    return rows


def throughput(level: dict | None) -> str:
    if level is None:
        return "-"
    errors = f", **{level['fail_pct']}% errors**" if level["fail_pct"] else ""
    return f"{level['rps']} req/s{errors}"


def p95(level: dict | None) -> str:
    return "-" if level is None else f"{level['p95_ms']:.0f} ms"


def kinds_table(level: dict) -> list[str]:
    rows = [
        "| Request kind | req/s | p50 | p95 | p99 | errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, k in sorted(level["by_kind"].items()):
        rows.append(
            f"| {name} | {k['rps']} | {k['p50_ms']:.0f} ms | {k['p95_ms']:.0f} ms "
            f"| {k['p99_ms']:.0f} ms | {k['fail_pct']}% |"
        )
    return rows


def by_kind_lines(per_1k: dict | None) -> list[str]:
    """Cost per 1,000 split by what answered the request. Absent from older runs' JSON."""
    if not per_1k:
        return []
    counts = per_1k.get("requests", {})
    kinds = [
        ("cache hit", "cache_hit", "no provider call"),
        ("provider call", "provider_call", "paid"),
        ("self-hosted", "self_hosted", "you pay for the hardware, not per token"),
    ]
    rows = [
        f"  - {label}: ${per_1k[key]:.6f} per 1,000 ({counts.get(key, 0):,} requests, {note})"
        for label, key, note in kinds
    ]
    return ["- Cost per 1,000 by kind:", *rows]


TTFT = "chat: stream first token"
WHOLE_STREAM = "chat: stream complete"


def streaming_section(label: str, run: dict) -> list[str]:
    """A run with streamed requests, reported on its own.

    It cannot share the comparison columns above: adding a fifth request kind changes the traffic
    mix, so its throughput is not the same measurement as a run without it.
    """
    slo = CLASS_SLOS["chat"]
    levels = [lv for lv in run["levels"] if TTFT in lv["by_kind"]]
    if not levels:
        return []
    lines = [
        f"## Streaming (`{label}`)",
        "",
        f"A separate run whose mix includes streamed requests (weight {run.get('stream_weight')} "
        "against 4 for each other chat kind), so its throughput is not comparable with the runs "
        "above. Time to first token is what a streaming client feels; the whole-stream figure is "
        "how long the generation took, which is not a latency to hold to an objective.",
        "",
        "| Users | streamed req/s | TTFT p50 | TTFT p95 | whole stream p50 | whole stream p95 "
        "| errors |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lv in levels:
        ttft, whole = lv["by_kind"][TTFT], lv["by_kind"].get(WHOLE_STREAM, {})
        lines.append(
            f"| {lv['users']} | {ttft['rps']} | {ttft['p50_ms']:.0f} ms | {ttft['p95_ms']:.0f} ms "
            f"| {whole.get('p50_ms', 0):.0f} ms | {whole.get('p95_ms', 0):.0f} ms "
            f"| {ttft['fail_pct']}% |"
        )
    within = [lv for lv in levels if lv["by_kind"][TTFT]["p95_ms"] <= slo]
    best = max(within, key=lambda lv: lv["rps"]) if within else None
    lines += [
        "",
        f"- Highest load where time to first token kept p95 under {slo:.0f} ms: "
        + (
            f"**{best['users']} users** ({best['rps']} req/s over the whole mix, "
            f"{best['by_kind'][TTFT]['rps']} of them streamed)"
            if best
            else "**none**"
        ),
    ]
    return lines


def chaos_section(chaos: dict) -> list[str]:
    t = chaos["timeline_s"]
    lines = [
        "## Chaos test",
        "",
        f"A steady {chaos['rate_rps']:.0f} req/s to the `mock` route with the cache off. At "
        f"{t['fault_at']:.0f}s the primary deployment was made to fail every call (through the "
        f"admin fault-injection API, on every replica); at {t['cleared_at']:.0f}s the fault was "
        "cleared.",
        "",
        "| Phase | Requests | Succeeded | Served by primary | p50 | p95 | Cost / 1k requests |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, p in chaos["phases"].items():
        lines.append(
            f"| {name} | {p['requests']} | {p['success_pct']}% | {p['served_by_primary_pct']}% "
            f"| {p['p50_ms']:.0f} ms | {p['p95_ms']:.0f} ms | ${p['cost_per_1k_requests_usd']} |"
        )
    cooldown = chaos["breaker_cooldown_s"]
    lines += [
        "",
        f"- **User-visible failures: {chaos['user_visible_failures']}.** Every request during the "
        "fault was answered, by the fallback.",
        f"- **Breakers opened {chaos['seconds_until_breakers_opened']} s after the fault**, after "
        f"{chaos['failures_before_breakers_opened']} failed primary calls, all absorbed by "
        "fallback: 3 per replica (the breaker threshold), plus one already in flight. The first "
        "failure lands up to ~1 s after injection because each replica caches the fault table for "
        "a second.",
        f"- {chaos['failed_half_open_probes']} failed half-open probes while the primary stayed "
        f"down: one per replica per {cooldown:.0f} s cooldown, checking whether it had recovered.",
        f"- **Primary serving again {chaos['seconds_until_primary_served_again']} s after the "
        f"fix**: the breakers re-check once per cooldown, so recovery takes up to {cooldown:.0f} s "
        "after a fix.",
        f"- Failed primary calls, seconds after the fault: "
        f"{chaos['primary_failure_times_after_fault_s']}",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", default=["baseline", "after-fix"])
    parser.add_argument("--chaos", default="chaos")
    parser.add_argument(
        "--streaming", help="label of a run whose mix includes streamed requests, reported apart"
    )
    parser.add_argument(
        "--repeated", help="label of a run that measured each level more than once, for its spread"
    )
    parser.add_argument("--detail-users", type=int, help="level for the per-kind table")
    args = parser.parse_args()

    runs = {label: run for label in args.runs if (run := load(label))}
    missing = [label for label in args.runs if label not in runs]
    if missing:
        raise SystemExit(f"missing results: {', '.join(missing)}")
    final = runs[args.runs[-1]]
    detail = by_users(final).get(args.detail_users) if args.detail_users else None
    if detail is None:  # default: the busiest level without errors
        clean = [lv for lv in final["levels"] if lv["fail_pct"] < 1] or final["levels"]
        detail = max(clean, key=lambda lv: lv["rps"])
    usage = final["usage"]
    chaos = load(args.chaos)

    lines = [
        "# Load test results",
        "",
        "Generated by `python loadtest/report.py` from `loadtest/results/`. Each level ran "
        f"{final['duration_s']} s with Locust *inside* the kind cluster (see "
        "[README](README.md) for why). SLO: p95 under "
        f"{final['slo_p95_ms']:.0f} ms with under 1% errors, over the whole traffic mix.",
        "",
        *[
            f"- Whole-mix ceiling (p95 < {final['slo_p95_ms']:.0f} ms), `{label}`: {ceiling(run)}"
            for label, run in runs.items()
        ],
        "",
        "## Ceiling per request class",
        "",
        "Total throughput at the highest load where every request of that class met its "
        "objective with under 1% errors.",
        "",
        "| Class (p95 objective) | " + " | ".join(f"`{label}`" for label in runs) + " |",
        "|---|" + "---:|" * len(runs),
        *[
            f"| {prefix} (< {slo:.0f} ms) | "
            + " | ".join(class_ceiling(run, prefix) for run in runs.values())
            + " |"
            for prefix, slo in CLASS_SLOS.items()
        ],
        "",
        "## Throughput and errors",
        "",
        *comparison(runs, throughput),
        "",
        "## p95 latency, whole traffic mix",
        "",
        *comparison(runs, p95),
        "",
        f"## By request kind: `{args.runs[-1]}` at {detail['users']} users",
        "",
        *kinds_table(detail),
        "",
        f"## Cache and cost (`{args.runs[-1]}`, all levels)",
        "",
        f"- Cache hit rate: {usage['cache_hit_rate']:.1%} of {usage['requests']} requests",
        f"- Cost per 1,000 requests: ${usage['cost_per_1k_requests_usd']}; saved by the cache per "
        f"1,000: ${usage['saved_per_1k_requests_usd']}",
        *by_kind_lines(usage.get("cost_per_1k_by_kind")),
        "- Costs use the mock route's pricing (priced like a small hosted model) and its "
        "word-count token estimates, so they compare configurations; they are not a bill.",
        "",
        "Autoscaling is measured separately, by `loadtest/autoscale.py`: see "
        "[AUTOSCALING.md](AUTOSCALING.md).",
    ]
    if args.repeated:
        repeated = load(args.repeated)
        if repeated is None:
            raise SystemExit(f"missing repeated result: {args.repeated}")
        lines += ["", *repetitions_section(args.repeated, repeated)]
    if repeated := repetitions_section(args.runs[-1], final):
        lines += ["", *repeated]
    if args.streaming:
        streaming = load(args.streaming)
        if streaming is None:
            raise SystemExit(f"missing streaming result: {args.streaming}")
        lines += ["", *streaming_section(args.streaming, streaming)]
    if chaos:
        lines += ["", *chaos_section(chaos)]
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {HERE / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
