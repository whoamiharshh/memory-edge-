"""Secret scan of the repository (docs/THREATS.md, "Insecure logs / secrets"): no credentials in tracked files, and
private folders (runtime state with live demo tokens, raw data, model files) are git-ignored."""
import json
import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATTERNS = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "OpenAI/Anthropic-style key": re.compile(r"\bsk-(ant-)?[A-Za-z0-9_-]{20,}\b"),
    "assigned api key": re.compile(r"(?i)\b(api[_-]?key|secret|password)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),
}
pytestmark = pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(),
                                reason="needs git and a git repository")


def tracked_files() -> list[pathlib.Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ROOT / p for p in out.splitlines() if p]


def test_no_credential_patterns_in_tracked_files():
    hits = []
    for f in tracked_files():
        if f.suffix in (".png", ".jpg", ".zip", ".npz", ".gguf") or not f.is_file():
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for name, rx in PATTERNS.items():
            for m in rx.finditer(text):
                hits.append(f"{f.relative_to(ROOT)}: {name}: {m.group(0)[:12]}...")
    assert hits == [], hits


def test_live_demo_tokens_are_not_committed():
    boot = ROOT / "runtime" / "cloud" / "bootstrap.json"
    if not boot.exists():
        pytest.skip("demo has not been bootstrapped on this machine")
    b = json.loads(boot.read_text())
    tokens = [b["admin"]] + [d["token"] for d in b["devices"].values()]
    leaked = [f.relative_to(ROOT) for f in tracked_files() if f.is_file() and f.suffix not in (".png", ".zip")
              and any(t in f.read_text(encoding="utf-8", errors="ignore") for t in tokens)]
    assert leaked == []


@pytest.mark.parametrize("path", ["runtime/cloud/bootstrap.json", "runtime/cloud/tokens.json", "data/raw/x.mat",
                                  "models_cache/x.gguf", ".env", ".venv-laya/x", "qdrant_server/qdrant.exe"])
def test_private_paths_are_git_ignored(path):
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert r.returncode == 0, f"{path} is not git-ignored"
