"""Publish the one-container demo to a Hugging Face Space, which gives it a public URL.

    hf auth login                                        # once, in your own terminal
    python scripts/deploy_space.py <hf-user>/nexusgate   # create or update the Space
    python scripts/deploy_space.py <hf-user>/nexusgate --dry-run --stage-dir build/space

A Space is a git repo that Hugging Face builds and runs. It expects a ``Dockerfile`` and a README
with a YAML header at its root, so this stages exactly what Containerfile.demo needs under those
names and uploads it. Hugging Face builds the image on its side (the models download there), then
this waits for the Space to come up and checks /readyz. The login token stays in the Hugging Face
CLI's own store; this script never sees it. Re-running it deploys the current working tree.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/ardhendudebnath/llm-gateway-rag"

SPACE_README = """---
title: NexusGate
emoji: 🚦
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: LLM gateway + RAG demo on mock providers, free to try
---

# NexusGate: public demo

An LLM gateway and RAG backend: provider fallback with circuit breakers, a semantic cache, per-key
rate limiting and cost metering, and retrieval with reranking. This Space runs it on offline mock
providers with a pre-loaded handbook, so nothing costs anything. Open the app for a key and curl
examples.

Source, design notes, the retrieval eval and the load tests: {github}

Deployed from commit `{commit}`.
"""


def copy_contents(src: Path, dst: Path) -> None:
    """Copy file contents only. Not copytree: that copies attributes too, and OneDrive marks
    folders read-only, which leaves a staged folder that Windows then refuses to delete."""
    for path in src.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            out = dst / path.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out)


def stage(target: Path, commit: str) -> list[Path]:
    """Lay out the Space repo: Containerfile.demo as the Dockerfile, plus what it copies."""
    shutil.copyfile(ROOT / "Containerfile.demo", target / "Dockerfile")
    shutil.copyfile(ROOT / "pyproject.toml", target / "pyproject.toml")
    for part in ("app", "config", "eval/corpus"):
        copy_contents(ROOT / part, target / part)
    readme = SPACE_README.format(github=GITHUB, commit=commit)
    (target / "README.md").write_text(readme, encoding="utf-8")
    return sorted(p.relative_to(target) for p in target.rglob("*") if p.is_file())


def current_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        sha = git("rev-parse", "--short", "HEAD")
        return f"{sha}+local changes" if git("status", "--porcelain") else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def app_url(space: str) -> str:
    owner, name = space.split("/")
    host = f"{owner}-{name}".lower().replace("_", "-").replace(".", "-")
    return f"https://{host}.hf.space"


def wait_until_running(api, space: str, timeout_s: float) -> bool:
    # On a redeploy the old version reports RUNNING until the new build starts; don't mistake
    # it for the new one. (If nothing changed, no build starts, and after a minute we go on.)
    for _ in range(12):
        if api.get_space_runtime(space).stage != "RUNNING":
            break
        time.sleep(5)
    deadline, last = time.monotonic() + timeout_s, None
    while time.monotonic() < deadline:
        stage_ = api.get_space_runtime(space).stage
        if stage_ != last:
            print(f"  space: {stage_}")
            last = stage_
        if stage_ == "RUNNING":
            return True
        if stage_ in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}:
            return False
        time.sleep(10)
    print("  gave up waiting; the build may still finish, check the Space's logs")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("space", help="<hf-user or org>/<space-name>")
    parser.add_argument("--private", action="store_true", help="only you can open the Space")
    parser.add_argument("--dry-run", action="store_true", help="stage the files, upload nothing")
    parser.add_argument("--stage-dir", type=Path, help="where to stage (default: a temp dir)")
    parser.add_argument("--wait-minutes", type=float, default=20)
    args = parser.parse_args()
    if args.space.count("/") != 1:
        parser.error("space must look like <hf-user>/<space-name>")

    commit = current_commit()
    with tempfile.TemporaryDirectory() as tmp:
        target = args.stage_dir or Path(tmp)
        if args.stage_dir and target.exists():
            shutil.rmtree(target)  # loudly: a half-cleared stage would deploy a broken app
        target.mkdir(parents=True, exist_ok=True)
        files = stage(target, commit)
        print(f"staged {len(files)} files from {commit} in {target}")
        if args.dry_run:
            for f in files:
                print(f"  {f.as_posix()}")
            return 0

        from huggingface_hub import HfApi  # installed with the `embeddings` extra

        api = HfApi()
        try:
            user = api.whoami()["name"]
        except Exception:
            sys.exit("not logged in to Hugging Face: run `hf auth login` in your terminal first")
        print(f"logged in as {user}; deploying to {args.space}")
        api.create_repo(
            args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True
        )
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=target,
            commit_message=f"Deploy {commit}",
            delete_patterns=["app/**", "config/**", "eval/**"],  # drop files removed locally
        )

    print(f"uploaded; Hugging Face is building it: https://huggingface.co/spaces/{args.space}")
    if not wait_until_running(api, args.space, args.wait_minutes * 60):
        return 1
    url = app_url(args.space)
    if args.private:
        print(f"running; it is private, so open it while logged in to Hugging Face: {url}")
        return 0
    for _ in range(30):  # the container is up before the handbook has finished loading
        try:
            if httpx.get(f"{url}/readyz", timeout=10).status_code == 200:
                print(f"live: {url}")
                return 0
        except httpx.HTTPError:
            pass
        time.sleep(5)
    print(f"the Space is running but {url}/readyz is not answering yet")
    return 1


if __name__ == "__main__":
    sys.exit(main())
