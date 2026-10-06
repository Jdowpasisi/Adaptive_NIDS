"""DriftGuard: cross-dataset NIDS under drift."""

import os

# MLflow >= 3.16 logs an "agent hint" on import; the package is imported before mlflow everywhere.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
