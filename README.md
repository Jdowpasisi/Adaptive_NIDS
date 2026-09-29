# DriftGuard: cross-dataset NIDS under drift

Measures how ML intrusion detectors degrade across networks, explains why, detects drift live,
and adapts with a human approval gate. The build plan is [Build_Guide.md](Build_Guide.md). Scope is set by the
Complete Project Guide in this folder.

## Setup (≈10 minutes)

Requirements: Linux or macOS, `git`, `make`, and [uv](https://docs.astral.sh/uv/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh`). uv downloads Python 3.11 itself.

```bash
git clone https://github.com/Jdowpasisi/Adaptive_NIDS.git && cd Adaptive_NIDS
make setup        # venv at ~/.venvs/driftguard (override with VENV=...), pinned deps, package in editable mode
make test         # unit tests
make smoke        # logs one dummy run to MLflow (mlflow.db in the repo root)
make mlflow       # UI at http://127.0.0.1:5000
```

Activate the venv for interactive work: `source ~/.venvs/driftguard/bin/activate`.

**Colab / Kaggle:** `pip install -r requirements.txt && pip install -e . --no-deps`. Point runs back at the
shared store by exporting `MLFLOW_TRACKING_URI`, or copy the run's `mlruns/` folder into the repo afterwards.

## Layout

| Path | What |
|---|---|
| `configs/` | every run starts from a YAML here; its hash is logged with the run |
| `src/xnids/` | the package (data, features, models, eval, analysis, drift, adapt, select, live, attack) |
| `scripts/` | one CLI per pipeline step; every table and figure comes from here |
| `data/` | raw, interim, processed, splits (gitignored; files identified by `data/MANIFEST.csv`) |
| `reports/` | generated tables and figures |

Locations can be overridden with `DG_DATA`, `DG_MODELS`, `DG_REPORTS` and `MLFLOW_TRACKING_URI`.

## Rules (Build Guide §2, short form)

Configs, not code edits · config hash + git commit on every run · seeds 0, 1, 2 · thresholds from
source validation only · each test split used once · report FPR at the frozen threshold together with the DR it
achieves, plus PR-AUC and MCC · no results from notebooks · data never committed · identifiers never features.
