"""Single source of truth for on-disk locations.

Everything resolves relative to the repo root unless overridden by an env var:
  DG_DATA    -> data root        (default <repo>/data)
  DG_MODELS  -> model bundles    (default <repo>/models)
  DG_REPORTS -> tables / figures (default <repo>/reports)
  MLFLOW_TRACKING_URI            (default sqlite:///<repo>/mlflow.db)
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _env_path(var: str, default: Path) -> Path:
    return Path(os.environ[var]).expanduser().resolve() if os.environ.get(var) else default


DATA = _env_path("DG_DATA", REPO / "data")
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
LOGS = DATA / "logs"
MANIFEST = DATA / "MANIFEST.csv"

MODELS = _env_path("DG_MODELS", REPO / "models")
REPORTS = _env_path("DG_REPORTS", REPO / "reports")
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
CONFIGS = REPO / "configs"

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO / 'mlflow.db'}")
ARTIFACT_ROOT = REPO / "mlruns"


def ensure_dirs() -> None:
    for p in (RAW, INTERIM, PROCESSED, SPLITS, LOGS, MODELS, TABLES, FIGURES, ARTIFACT_ROOT):
        p.mkdir(parents=True, exist_ok=True)
