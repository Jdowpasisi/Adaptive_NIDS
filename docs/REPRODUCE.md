# Reproducing DriftGuard

Two levels: **re-derive every reported number from the logged runs** (minutes), or **rerun the whole pipeline
from raw data** (about 30–45 hours on the reference laptop).

Reference machine: Arch Linux, 14.9 GB RAM + 8 GB swap, RTX 3050 (4 GB), Python 3.11 (uv). See
[RESOURCES.md](RESOURCES.md) for memory peaks and crash safety.

## 0. Setup (10 minutes)

```bash
git clone https://github.com/Jdowpasisi/Adaptive_NIDS.git && cd Adaptive_NIDS
make setup && make test
```

## 1. Re-derive every table and figure from the logged runs (~3 minutes)

Needs `mlflow.db` + `mlruns/` (the run store), `data/` (processed flows, splits, replay) and `models/`, copied from
a machine that ran the pipeline, or restored from `~/driftguard_backup` (`scripts/backup.sh` explains how).

```bash
make tables          # = python scripts/make_tables.py --derived
make model-cards
```

- **What it runs.** `make tables` rebuilds every MLflow-backed table and figure from the finished runs' logged
  metrics and per-run tables, with no training. It also recomputes the few derived ones (bridge check, demo
  check, rehearsal figure).
- **Expected output:** `152 files from 14 steps: 152 byte-identical after the rebuild, 0 changed`.
- **Provenance.** `reports/PROVENANCE.csv` lists, for every file in `reports/tables` and `reports/figures`:
  - its sha256 and producing step and command;
  - its MLflow experiment and the number of runs behind it;
  - `reports/provenance/<step>_runs.csv`, which names those run IDs with their config hash, seed, model and
    dataset tags.
- **Inspect a run:** `make mlflow`, then open http://127.0.0.1:5000.
- **Not rebuilt by `make tables`:** the live-system results (C13–C16 benchmarks, replays, rehearsals) and the
  data-preparation tables (ingest profiles, splits). They come from running those steps, listed below.

## 2. Rerun everything from raw data

### Data you must fetch yourself

| Dataset | Where | Put in |
|---|---|---|
| LycoS-IDS2017 | downloaded by `scripts/ingest.py` | (automatic) |
| LycoS-Unicas-IDS2018 | downloaded by `scripts/ingest.py` (Google Drive) | (automatic) |
| NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2, NF-ToN-IoT-v2 | UQ RDM portal (form, browser only) | `data/raw/<name>/*.zip` |
| CIC-IDS2017 PCAPs (Mon, Wed, Fri) | https://www.unb.ca/cic/datasets/ids-2017.html (request form) | `data/raw/cic17_pcap/` |
| CSE-CIC-IDS2018 Fri-16-02-2018 (15 members) | downloaded by `scripts/fetch_cse18_pcaps.py` (HTTP ranges, ~9 GB) | (automatic) |

Checksums of every file used are in `data/MANIFEST.csv`. Splits are frozen: `scripts/split.py` refuses to write a
split that differs from `reports/tables/splits_lock.csv`, so a rerun gets the same rows.

### Run

```bash
make reproduce                       # every stage, in dependency order
make reproduce STAGES="c6 c7"        # a subset
bash scripts/reproduce.sh --list     # stages and time estimates
```

| Stage | What | Time | On rerun |
|---|---|---|---|
| data | C1–C3 ingest, bridge check, tracks, frozen splits | ~1 h | splits verified against the lock |
| c4 | adversarial validation | ~3.5 h | finished runs skipped |
| c6 | 7 models × 4 tracks × 3 seeds, results matrix | ~5 h | finished runs skipped |
| c7 | ablation / normalisation (H3), time drift (H4) | ~6 h | finished runs skipped |
| c8 | drift monitor evaluation (H5) | ~25 min | finished runs reused |
| c9 | adapters | ~5 h | finished runs skipped |
| c12 | Reptile (its meta-models are C10 actions, so it runs before C10) | ~4 h | finished runs skipped |
| c10 | adaptation logs + selector | ~8 h | finished pairs skipped; selector refitted (~1 min) |
| c13 | demo data: PCAP download, NFStream, demo models, demo.pcap | ~1 h | downloads and extractions reused |
| c15 | live system: monitor calibration, API benchmark, walkthroughs, replay, load test, 3× rehearsal | ~1 h | recomputed |
| c17 | poisoning test (H7) | ~45 min | recomputed |
| reports | `make tables` + model cards | ~5 min | — |

- **Logs and backups:** each stage logs to `logs/reproduce/<stage>.log`, and `scripts/backup.sh` snapshots the run
  store after every stage.
- **If a stage fails:** fix the cause and run the same command again.
- **Long runs:** start them as a `systemd-run --user` unit (RESOURCES.md) so they survive a closed terminal.

### Determinism

- Seeds 0, 1, 2 are set everywhere (`xnids.utils.seed`).
- The C6 numbers are means ± std over the seeds. LDA and XGBoost train deterministically, so their seeds
  coincide.
- GPU training (MLP, TabNet, XGBoost) can differ in the last digits between machines.
- Rebuilding from the run store is exact (152 of 152 files byte-identical).

## 3. Live demo

`make demo`, then follow [DEMO.md](DEMO.md). `make rehearse` runs the 5-minute script three times through the real
dashboard without a browser.
