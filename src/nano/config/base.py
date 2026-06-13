"""Project paths and git/baseline metadata helpers."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))

CONFIGS_DIR = PROJECT_ROOT / "configs"
FEATURE_SETS_DIR = CONFIGS_DIR / "feature_sets"
GENERATED_DIR = PROJECT_ROOT / "generated"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "runs"
BASELINE_DIR = PROJECT_ROOT / "baseline"


def baseline_sha() -> str:
    """Return the vendored baseline source sha (best-effort)."""
    src = BASELINE_DIR / "SOURCE_SHA.txt"
    try:
        return src.read_text().strip()
    except OSError:
        return "unknown"


def repo_git_info() -> dict:
    """Return ``{repo, sha, dirty}`` for the remodded-nanogpt repo (best-effort)."""
    info = {"repo": "remodded-nanogpt", "sha": "unknown", "dirty": False}
    try:
        import git  # GitPython

        repo = git.Repo(PROJECT_ROOT, search_parent_directories=True)
        info["sha"] = repo.head.commit.hexsha
        info["dirty"] = repo.is_dirty(untracked_files=False)
        try:
            url = repo.remotes.origin.url
            info["repo"] = url.rsplit("/", 2)[-2] + "/" + url.rsplit("/", 1)[-1].removesuffix(".git")
        except Exception:
            info["repo"] = Path(repo.working_dir).name
    except Exception:
        pass
    return info
