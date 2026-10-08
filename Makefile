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
demo-data:        ## C13: fetch CSE18 PCAPs, NFStream flows, track/split, demo MLP, demo.pcap + labels, check
	$(PY) scripts/fetch_cse18_pcaps.py
	$(PY) scripts/extract_flows.py --dataset nfs17
	$(PY) scripts/extract_flows.py --dataset nfs18
	$(PY) scripts/build_tracks.py --dataset nfs17 --track nfs
	$(PY) scripts/build_tracks.py --dataset nfs18 --track nfs
	$(PY) scripts/split.py --dataset nfs17
	$(PY) scripts/split.py --dataset nfs18
	$(PY) scripts/train.py --config configs/train/nfs/mlp.yaml
	$(PY) scripts/build_demo_pcap.py
	$(PY) scripts/demo_check.py
backup:           ## copy mlflow.db (consistent snapshot), mlruns, models, splits to ~/driftguard_backup
	bash scripts/backup.sh
TRACK ?= cic77
matrix:
	$(PY) scripts/matrix.py --track $(TRACK)
api:              ## C14: detector API on :8000 (config: configs/live.yaml or $$DG_LIVE_CONFIG)
	$(VENV)/bin/uvicorn xnids.live.api:app --host 0.0.0.0 --port 8000
bench-api:        ## C14: p50/p99 latency for batches of 1 / 100 / 1000 -> reports/tables/c14_latency.csv
	$(PY) scripts/bench_api.py
calibrate-monitor: ## C15: calibrate the live drift monitor on time-ordered raw source traffic
	$(PY) scripts/calibrate_monitor.py
replay:           ## C15: demo.pcap -> NFStream -> fresh API (+ monitor); ARGS='--rate 2000 --auto-adapt --auto-approve'
	$(PY) scripts/replay.py --mode file $(ARGS)
load-test:        ## C15: sustained flows/s with p99 < 100 ms -> reports/tables/c15_load.csv
	$(PY) scripts/load_test.py
replay-live:      ## C15: live mode (sudo): veth pair + tcpreplay + NFStream on veth1, slice at x1
	sudo bash scripts/replay_live.sh
walkthrough:      ## C14: demo replay through the API (monitor -> recommend -> adapt -> gates -> promote -> rollback)
	$(PY) scripts/c14_walkthrough.py --approve --adapt-from-window 37
dashboard:        ## C16: dashboard only, against $$DG_API (default http://127.0.0.1:8000)
	PYTHONPATH=src $(VENV)/bin/streamlit run dashboard/app.py
demo:             ## C16: fresh API + dashboard (http://localhost:8501); press "Start replay" in the sidebar
	$(PY) scripts/run_demo.py
rehearse:         ## C16: run the 5-minute demo script 3x through the real dashboard (AppTest) -> c16_rehearsals.csv
	$(PY) scripts/rehearse_demo.py
reproduce: ingest bridge tracks split
	@echo "Remaining reproduce steps are added as components land (see Build_Guide.md)."
