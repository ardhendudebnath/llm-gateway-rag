"""Answer evaluation: are the gateway's RAG answers correct, grounded in their passages, and cited?

    python -m eval.answer_eval --admin-token <token>                # the `local` route
    python -m eval.answer_eval --model mock --label control         # must score ~0: see below
    python -m eval.answer_eval --endpoint agent --label agent-local # the research agent instead

It drives the running API end to end, as a client would: a fresh tenant gets the eval corpus
through the upload endpoint and the ingestion workers, then every question goes to
/v1/rag/answer (or /v1/agents/research). What is measured is what a client is served.

Scoring is deterministic, against the labelled gold answers and evidence spans. There is no LLM
judge: the only model to hand is the 1.5B one being judged. Per question:

* retrieved    a passage handed to the model contains an evidence span. If none does, the
               question tested retrieval, not generation, and "correct" can only come from
               outside knowledge
* correct      the answer carries the gold answer's facts: every number and identifier in it,
               and at least half of its other content words
* cited        the answer cites a passage inline, as the prompt asks: [n]
* cited_right  a passage it cites contains the evidence
* unsupported  numbers or identifiers in the answer found in neither the question nor any
               passage it was given: specifics it made up
* abstained    it says it doesn't know, which the prompt asks for when the passages fall short

Unanswerable questions (eval/unanswerable.jsonl, absent from the corpus by construction) score
the other way: abstaining is right, and anything else is a fabrication.

The `mock` route echoes the question back, so `--model mock` is a negative control: an eval that
gives an echo credit is measuring the question, not the answer.
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from eval.retrieval_eval import EVAL_DIR, Question, load_corpus, load_questions, normalise

UNANSWERABLE_FILE = EVAL_DIR / "unanswerable.jsonl"
RESULTS_DIR = EVAL_DIR / "answers"

# --- scoring: pure functions, unit-tested ------------------------------------------------------

_MARKER = re.compile(r"\[(\d+)\]")
_LIST_ITEM = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TOKEN = re.compile(r"[a-z0-9]+(?:[-.:/][a-z0-9]+)*")
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "ninety": "90",
    "hundred": "100",
}  # fmt: skip
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "do", "does", "for", "from",
    "has", "have", "if", "in", "into", "is", "it", "its", "may", "must", "of", "on", "or", "per",
    "should", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "whom", "why", "will",
    "with", "would", "you", "your",
})  # fmt: skip
_ABSTAIN = re.compile(
    r"\b(?:"
    r"(?:i )?(?:do not|don't|dont) know"
    r"|(?:can ?not|can't|couldn't|could not|unable to) (?:find|answer|determine|tell|say|provide)"
    r"|(?:do|does|did)(?: not|n't) (?:contain|mention|say|specify|provide|include|state|cover)"
    r"|no (?:information|mention|details?)"
    r"|not (?:mentioned|specified|stated|provided|covered|included|available)"
    r"|(?:not enough|insufficient) information"
    r"|(?:do not|don't) have enough information"
    r")\b",
    re.IGNORECASE,
)


def tokens(text: str) -> list[str]:
    """Lowercased words, numbers and identifiers (NG-1017, 10:00, 99.9), with citation markers,
    list numbering and thousands separators removed, and small number words as digits."""
    text = _THOUSANDS.sub("", _LIST_ITEM.sub(" ", _MARKER.sub(" ", text)))
    return [_NUMBER_WORDS.get(t, t) for t in _TOKEN.findall(text.lower().replace("|", " "))]


def specifics(text: str) -> set[str]:
    """Numbers and identifiers: the facts an answer can get exactly wrong. Hyphenated ones also
    contribute their numeric parts, so "a 15-minute window" carries the 15 that a gold answer of
    "15 minutes" needs."""
    out = set()
    for t in tokens(text):
        if any(c.isdigit() for c in t):
            out.add(t)
            out.update(p for p in t.split("-") if any(c.isdigit() for c in p))
    return out


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def content_words(text: str) -> set[str]:
    return {
        _stem(t)
        for t in tokens(text)
        if t not in _STOPWORDS and len(t) > 2 and not any(c.isdigit() for c in t)
    }


def is_correct(answer: str, gold: str, question: str = "") -> bool:
    """Every number and identifier of the gold answer, and at least half of the content words it
    adds to the question. Words the question already holds don't count: echoing "Which team owns
    billing-reconciler?" would otherwise earn half of "The Revenue team" for the word "team"."""
    if not specifics(gold) <= specifics(answer):
        return False
    words = (content_words(gold) - content_words(question)) or content_words(gold)
    return not words or len(words & content_words(answer)) / len(words) >= 0.5


def abstained(answer: str) -> bool:
    return bool(_ABSTAIN.search(answer))


def unsupported(answer: str, question: str, passages: list[str]) -> list[str]:
    """Specifics in the answer that the question and the passages never mention."""
    known = specifics(question).union(*(specifics(p) for p in passages))
    return sorted(specifics(answer) - known)


def holds_evidence(passage: str, evidence: list[str]) -> bool:
    text = normalise(passage)
    return any(normalise(span) in text for span in evidence)


# --- running against the API --------------------------------------------------------------------


@dataclass(frozen=True)
class Unanswerable:
    id: str
    question: str
    absent: list[str]
    near: str


def load_unanswerable(path: Path = UNANSWERABLE_FILE) -> list[Unanswerable]:
    with path.open(encoding="utf-8") as f:
        return [Unanswerable(**json.loads(line)) for line in f if line.strip()]


@dataclass
class Result:
    id: str
    kind: str  # semantic / lexical / terse, or "unanswerable"
    question: str
    gold: str | None
    answer: str = ""
    error: str | None = None
    retrieved: bool = False
    correct: bool = False
    cited: bool = False
    cited_right: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    abstained: bool = False
    latency_s: float = 0.0
    deployment: str | None = None
    llm_calls: int | None = None


def score(result: Result, passages: list[str], cited_ns: set[int], evidence: list[str]) -> None:
    answer = result.answer
    result.abstained = abstained(answer)
    result.unsupported = unsupported(answer, result.question, passages)
    numbers = {int(n) for n in _MARKER.findall(answer)}
    result.invalid_citations = sorted(n for n in numbers if not 1 <= n <= len(passages))
    result.cited = bool(cited_ns)
    if evidence:
        holders = {n for n, p in enumerate(passages, 1) if holds_evidence(p, evidence)}
        result.retrieved = bool(holders)
        result.cited_right = bool(holders & cited_ns)
        result.correct = bool(result.gold) and is_correct(answer, result.gold, result.question)


class Gateway:
    def __init__(self, base_url: str, admin_token: str):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=600)
        self.admin = {"X-Admin-Token": admin_token}
        self.auth: dict[str, str] = {}
        self.key_id = ""
        self.doc_ids: list[str] = []

    async def call(self, method: str, url: str, **kw) -> httpx.Response:
        """One request, waiting out the per-key rate limit rather than failing on it."""
        for _ in range(10):
            r = await self.client.request(method, url, **kw)
            if r.status_code != 429:
                return r
            await asyncio.sleep(float(r.headers.get("retry-after", 2)))
        return r

    async def setup(self) -> None:
        r = await self.call(
            "POST",
            "/v1/admin/keys",
            headers=self.admin,
            json={"tenant_id": f"answer-eval-{uuid.uuid4().hex[:8]}", "name": "answer-eval"},
        )
        r.raise_for_status()
        self.auth = {"Authorization": f"Bearer {r.json()['api_key']}"}
        self.key_id = r.json()["record"]["key_id"]
        jobs = []
        for name, data in load_corpus().items():
            r = await self.call(
                "POST",
                "/v1/rag/documents",
                headers=self.auth,
                files={"file": (name, data, "text/markdown")},
            )
            r.raise_for_status()
            jobs.append(r.json()["job_id"])
        for job_id in jobs:  # one at a time: polling counts against the rate limit too
            while True:
                job = (await self.call("GET", f"/v1/rag/jobs/{job_id}", headers=self.auth)).json()
                if job["status"] == "done":
                    self.doc_ids.append(job["doc_id"])
                    break
                if job["status"] == "failed":
                    raise RuntimeError(f"ingestion failed: {job.get('error')}")
                await asyncio.sleep(2)

    async def teardown(self) -> None:
        for doc_id in self.doc_ids:
            await self.call("DELETE", f"/v1/rag/documents/{doc_id}", headers=self.auth)
        await self.call("DELETE", f"/v1/admin/keys/{self.key_id}", headers=self.admin)
        await self.client.aclose()


async def ask(gw: Gateway, endpoint: str, model: str, result: Result, evidence: list[str]) -> None:
    started = time.perf_counter()
    if endpoint == "agent":
        path = "/v1/agents/research"
        body = {"question": result.question, "model": model, "max_revisions": 1}
    else:
        path = "/v1/rag/answer"
        body = {"question": result.question, "model": model, "cache": False, "temperature": 0}
    try:
        r = await gw.call("POST", path, headers=gw.auth, json=body)
        result.latency_s = round(time.perf_counter() - started, 2)
        if r.status_code != 200:
            result.error = f"{r.status_code}: {r.text[:200]}"
            return
        data = r.json()
    except httpx.HTTPError as e:
        result.error = f"{type(e).__name__}: {e}"
        return
    result.answer = data["answer"]
    result.deployment = (data.get("nexusgate") or {}).get("deployment")
    result.llm_calls = data.get("llm_calls")
    citations = data.get("citations") or []
    passages = [c["text"] for c in citations]
    cited_ns = {c["n"] for c in citations if c.get("cited")}
    score(result, passages, cited_ns, evidence)


def _share(results: list[Result], attr: str) -> float:
    return round(sum(getattr(r, attr) for r in results) / len(results), 3) if results else 0.0


def summarise(results: list[Result]) -> dict:
    answerable = [r for r in results if r.kind != "unanswerable" and not r.error]
    unanswerable = [r for r in results if r.kind == "unanswerable" and not r.error]
    retrieved = [r for r in answerable if r.retrieved]
    latencies = sorted(r.latency_s for r in results if not r.error)
    return {
        "questions": len(results),
        "errors": sum(1 for r in results if r.error),
        "answerable": {
            "n": len(answerable),
            "retrieved": _share(answerable, "retrieved"),
            "correct": _share(answerable, "correct"),
            "correct_when_retrieved": _share(retrieved, "correct"),
            "cited": _share(answerable, "cited"),
            "cited_right_when_retrieved": _share(retrieved, "cited_right"),
            "with_unsupported_specifics": round(
                sum(bool(r.unsupported) for r in answerable) / len(answerable), 3
            )
            if answerable
            else 0.0,
            "abstained": _share(answerable, "abstained"),
            "by_kind": {
                kind: {
                    "n": len(group),
                    "correct": _share(group, "correct"),
                    "cited": _share(group, "cited"),
                }
                for kind in ("semantic", "lexical", "terse")
                if (group := [r for r in answerable if r.kind == kind])
            },
        },
        "unanswerable": {
            "n": len(unanswerable),
            "abstained": _share(unanswerable, "abstained"),
            "with_unsupported_specifics": round(
                sum(bool(r.unsupported) for r in unanswerable) / len(unanswerable), 3
            )
            if unanswerable
            else 0.0,
        },
        "latency_s": {
            "p50": statistics.median(latencies) if latencies else None,
            "p95": latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))]
            if latencies
            else None,
        },
    }


async def run(args: argparse.Namespace) -> dict:
    questions: list[Question] = load_questions()[: args.limit or None]
    unanswerable = load_unanswerable()[: args.limit or None]
    jobs: list[tuple[Result, list[str]]] = [
        (Result(q.id, q.kind, q.question, q.answer), q.evidence) for q in questions
    ] + [(Result(u.id, "unanswerable", u.question, None), []) for u in unanswerable]

    gw = Gateway(args.base_url, args.admin_token)
    print(f"Ingesting the eval corpus into a fresh tenant at {args.base_url}")
    await gw.setup()
    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def one(result: Result, evidence: list[str]) -> None:
        nonlocal done
        async with sem:
            await ask(gw, args.endpoint, args.model, result, evidence)
        done += 1
        if done % 10 == 0 or done == len(jobs):
            print(f"  {done}/{len(jobs)} answered")

    started = time.perf_counter()
    try:
        await asyncio.gather(*(one(r, ev) for r, ev in jobs))
    finally:
        await gw.teardown()
    results = [r for r, _ in jobs]
    return {
        "label": args.label or f"{args.endpoint}-{args.model}",
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "endpoint": args.endpoint,
        "model": args.model,
        "wall_s": round(time.perf_counter() - started, 1),
        "summary": summarise(results),
        "results": [asdict(r) for r in results],
    }


def print_summary(report: dict) -> None:
    s = report["summary"]
    a, u = s["answerable"], s["unanswerable"]
    print(
        f"\n{report['label']}: {report['endpoint']} on `{report['model']}`, {s['questions']} "
        f"questions, {s['errors']} errors, {report['wall_s']} s"
    )
    print(f"  answerable ({a['n']}):")
    for name in (
        "retrieved",
        "correct",
        "correct_when_retrieved",
        "cited",
        "cited_right_when_retrieved",
        "with_unsupported_specifics",
        "abstained",
    ):
        print(f"    {name:28} {a[name]:.3f}")
    for kind, k in a["by_kind"].items():
        print(f"    {kind:28} correct {k['correct']:.3f}  cited {k['cited']:.3f}  (n={k['n']})")
    print(
        f"  unanswerable ({u['n']}): abstained {u['abstained']:.3f}, "
        f"with unsupported specifics {u['with_unsupported_specifics']:.3f}"
    )
    print(f"  latency per answer: p50 {s['latency_s']['p50']} s, p95 {s['latency_s']['p95']} s")


def resummarise(path: Path) -> dict:
    """Recompute a saved run's summary after the question sets change, without re-asking the
    model: results for questions no longer in either set are dropped. Two unanswerable questions
    turned out to be answerable after a run (the corpus names its paging tool, and its meal
    allowance applies to every trip); this is how that run stayed usable."""
    report = json.loads(path.read_text(encoding="utf-8"))
    live = {q.id for q in load_questions()} | {u.id for u in load_unanswerable()}
    kept = [Result(**r) for r in report["results"] if r["id"] in live]
    report["dropped"] = sorted(r["id"] for r in report["results"] if r["id"] not in live)
    report["results"] = [asdict(r) for r in kept]
    report["summary"] = summarise(kept)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--admin-token", default=os.environ.get("NEXUSGATE_ADMIN_TOKEN"))
    parser.add_argument("--model", default="local", help="route alias that generates the answers")
    parser.add_argument("--endpoint", choices=["answer", "agent"], default="answer")
    parser.add_argument("--label", help="results file name (default: <endpoint>-<model>)")
    parser.add_argument("--concurrency", type=int, default=2, help="the model server has 2 slots")
    parser.add_argument("--limit", type=int, help="only the first N questions of each set")
    parser.add_argument(
        "--resummarise",
        type=Path,
        metavar="RESULTS",
        help="recompute a saved run's summary against the current question sets, and exit",
    )
    args = parser.parse_args()
    if args.resummarise:
        report = resummarise(args.resummarise)
        print_summary(report)
        print(f"\ndropped {report['dropped'] or 'nothing'} -> {args.resummarise}")
        return 0
    if not args.admin_token:
        parser.error("--admin-token (or NEXUSGATE_ADMIN_TOKEN) is required")

    report = asyncio.run(run(args))
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{report['label']}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_summary(report)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
