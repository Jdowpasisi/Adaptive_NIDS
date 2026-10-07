"""Guards against source silently missing from git (src/xnids/models was ignored by `models/` until 7 Oct 2026)."""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def test_no_source_or_config_file_is_gitignored():
    if _git("rev-parse", "--git-dir").returncode:
        return                                   # not a git checkout (e.g. a source tarball)
    files = [str(p.relative_to(REPO)) for top in ("src", "configs", "scripts", "tests")
             for p in (REPO / top).rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and not any(part.endswith(".egg-info") for part in p.parts)]   # build artefacts: ignored on purpose
    ignored = _git("check-ignore", *files).stdout.split()
    assert ignored == [], f"gitignored but part of the code: {ignored}"
