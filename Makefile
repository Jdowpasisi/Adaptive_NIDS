# DriftGuard. Run `make setup` once; everything else runs inside the venv.
VENV   ?= $(HOME)/.venvs/driftguard
PY     := $(VENV)/bin/python
UV     ?= $(shell command -v uv || echo $(HOME)/.local/bin/uv)
export PYTHONPATH := $(CURDIR)/src
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: setup lock test lint fmt mlflow smoke ingest tracks bridge split check-splits advval train matrix backup api dashboard reproduce

setup:            ## create the venv and install pinned deps + the package
	$(UV) venv $(VENV) --python 3.11 --allow-existing
	VIRTUAL_ENV=$(VENV) $(UV) pip install -r requirements.txt
	VIRTUAL_ENV=$(VENV) $(UV) pip install -e . --no-deps

lock:             ## re-pin requirements.txt from requirements.in
	$(UV) pip compile requirements.in --python-version 3.11 -o requirements.txt

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src scripts tests dashboard

fmt:
	$(PY) -m ruff check --fix src scripts tests dashboard

mlflow:           ## MLflow UI on http://127.0.0.1:5000
	$(VENV)/bin/mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

smoke:
	$(PY) scripts/smoke_run.py --config configs/smoke.yaml

# Pipeline steps (implemented in C1+). DATASET selects one entry from configs/data.yaml.
DATASET ?= all
ingest:
	$(PY) scripts/ingest.py --dataset $(DATASET)
tracks:
	$(PY) scripts/build_tracks.py --dataset $(DATASET)
bridge:           ## H8: validate the core map on LycoS18 vs NF-CSE-CIC-IDS2018-v2
	$(PY) scripts/bridge_check.py
split:
	$(PY) scripts/split.py --dataset $(DATASET)
check-splits:     ## verify frozen split files against configs/split.yaml and the committed lock
	$(PY) scripts/split.py --dataset all --check
advval:           ## C4: adversarial validation for every pair (resumable; ~4 h for all tracks)
	$(PY) scripts/advval.py
CONFIG ?= configs/train/cic77/lda.yaml
train:            ## C5: make train CONFIG=configs/train/<track>/<model>.yaml [ARGS='--source lycos17 --seeds 0']
	$(PY) scripts/train.py --config $(CONFIG) $(ARGS)
backup:           ## copy mlflow.db (consistent snapshot), mlruns, models, splits to ~/driftguard_backup
	bash scripts/backup.sh
TRACK ?= cic77
matrix:
	$(PY) scripts/matrix.py --track $(TRACK)
api:
	$(VENV)/bin/uvicorn xnids.live.api:app --host 0.0.0.0 --port 8000
dashboard:
	$(VENV)/bin/streamlit run dashboard/app.py
reproduce: ingest bridge tracks split
	@echo "Remaining reproduce steps are added as components land (see Build_Guide.md)."
