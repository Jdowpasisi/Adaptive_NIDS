"""Thin MLflow wrapper so every run carries its config, config hash, git commit and seed."""

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")   # must precede `import mlflow`

import subprocess  # noqa: E402
import tempfile  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from xnids.utils import paths  # noqa: E402
from xnids.utils.config import cfg_hash, flatten  # noqa: E402


def git_state() -> tuple[str, bool]:
    """(commit sha, dirty?) of the repo; ('unknown', True) outside git."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=paths.REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=paths.REPO, text=True
            ).strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


def setup(experiment: str) -> None:
    mlflow.set_tracking_uri(paths.TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        paths.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(experiment, artifact_location=(paths.ARTIFACT_ROOT / experiment).as_uri())
    mlflow.set_experiment(experiment)


@contextmanager
def start_run(cfg: dict, seed: int | None = None, experiment: str = "driftguard", run_name: str | None = None,
              tags: dict | None = None) -> Iterator[mlflow.ActiveRun]:
    """Open an MLflow run with the standard tags. Yields the active run."""
    setup(experiment)
    h = cfg_hash(cfg)
    sha, dirty = git_state()
    name = run_name or cfg.get("run", {}).get("name", "run")
    all_tags = {"cfg_hash": h, "git_commit": sha, "git_dirty": str(dirty), **(tags or {})}
    if seed is not None:
        all_tags["seed"] = str(seed)
    with mlflow.start_run(run_name=f"{name}-s{seed}" if seed is not None else name, tags=all_tags) as run:
        params = {k: str(v)[:6000] for k, v in flatten(cfg).items()}
        mlflow.log_params(params)
        mlflow.log_dict(cfg, "config.json")
        if seed is not None:
            mlflow.log_param("seed", seed)
        yield run


def log_df_artifact(df: pd.DataFrame, name: str) -> None:
    """Save a DataFrame as Parquet inside the active run (e.g. scores_{target}.parquet)."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / name
        df.to_parquet(p, index=False)
        mlflow.log_artifact(str(p))


def find_runs(experiment: str = "driftguard", **tags: str) -> pd.DataFrame:
    """All runs whose tags match, e.g. find_runs(cfg_hash='ab12cd34ef')."""
    mlflow.set_tracking_uri(paths.TRACKING_URI)
    filt = " and ".join(f"tags.{k} = '{v}'" for k, v in tags.items())
    return mlflow.search_runs(experiment_names=[experiment], filter_string=filt)
