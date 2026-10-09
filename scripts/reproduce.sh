#!/usr/bin/env bash
# C18: reproduce DriftGuard end to end, in dependency order. Every stage is resumable (finished MLflow runs and
# existing files are skipped), so after a crash just run the same command again. See docs/REPRODUCE.md.
#
#   bash scripts/reproduce.sh                 # every stage (~30-45 h on the 14 GB / RTX 3050 laptop)
#   bash scripts/reproduce.sh c6 c7           # selected stages
#   bash scripts/reproduce.sh --list
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/driftguard/bin/python}
export PYTHONPATH=src MLFLOW_DISABLE_AGENT_HINT=1
mkdir -p logs/reproduce

declare -A DESC=(
  [data]="C1-C3  ingest, bridge check, tracks, frozen splits (~1 h; NF-v2 zips must be in data/raw first)"
  [c4]="C4     adversarial validation, 84 curves (~3.5 h)"
  [c6]="C5-C6  train 7 models x 4 tracks x 3 seeds, then the matrix (~5 h)"
  [c7]="C7     ablation + normalisation (H3) and time drift (H4) (~6 h)"
  [c8]="C8     drift monitor evaluation (H5) (~25 min)"
  [c9]="C9     adapters before / after (~5 h)"
  [c12]="C12    Reptile meta-learning, leave one target out (~4 h; before C10: its meta-models are C10 actions)"
  [c10]="C10-11 adaptation logs, then the selector (~8 h)"
  [c13]="C13    demo data: CSE-CIC-IDS2018 download, NFStream, demo models, demo.pcap (~1 h + 9 GB download)"
  [c15]="C14-16 live system: monitor calibration, API benchmark, walkthroughs, replay, load test, 3x rehearsal (~1 h)"
  [c17]="C17    poisoning test (H7) (~45 min)"
  [reports]="C18    rebuild every table / figure from MLflow, model cards, provenance (~5 min)"
)
ORDER=(data c4 c6 c7 c8 c9 c12 c10 c13 c15 c17 reports)

stage() {
  case "$1" in
    data)
      $PY scripts/ingest.py --dataset all
      $PY scripts/bridge_check.py
      $PY scripts/build_tracks.py --dataset all
      $PY scripts/split.py --dataset all          # refuses to change a frozen split; --check verifies the lock
      $PY scripts/split.py --dataset all --check ;;
    c4)  $PY scripts/advval.py ;;
    c6)  $PY scripts/sweep.py && $PY scripts/matrix.py ;;
    c7)  bash scripts/run_c7.sh && $PY scripts/timedrift.py ;;
    c8)  $PY scripts/drift_eval.py ;;
    c9)  $PY scripts/adapt_eval.py ;;
    c12) $PY scripts/reptile_eval.py ;;
    c10) $PY scripts/build_logs.py && $PY scripts/selector_eval.py ;;
    c13)
      $PY scripts/fetch_cse18_pcaps.py
      $PY scripts/extract_flows.py --dataset nfs17 && $PY scripts/extract_flows.py --dataset nfs18
      $PY scripts/build_tracks.py --dataset nfs17 --track nfs && $PY scripts/build_tracks.py --dataset nfs18 --track nfs
      $PY scripts/split.py --dataset nfs17 && $PY scripts/split.py --dataset nfs18
      $PY scripts/train.py --config configs/train/nfs/mlp.yaml --resume
      $PY scripts/train.py --config configs/train/nfs/xgb.yaml --resume
      $PY scripts/build_demo_pcap.py && $PY scripts/demo_check.py ;;
    c15)
      $PY scripts/calibrate_monitor.py
      $PY scripts/bench_api.py
      $PY scripts/c14_walkthrough.py --approve --adapt-from-window 37
      bash scripts/c14_actions.sh
      $PY scripts/replay.py --mode file --run-id file-rehearsal --auto-adapt --auto-approve
      $PY scripts/plot_replay.py --run file-rehearsal
      $PY scripts/load_test.py
      $PY scripts/rehearse_demo.py ;;
    c17) $PY scripts/poison_eval.py ;;
    reports)
      $PY scripts/make_tables.py --derived
      $PY scripts/make_model_cards.py ;;
    *) echo "unknown stage $1"; return 2 ;;
  esac
}

if [ "${1:-}" = "--list" ]; then
  for s in "${ORDER[@]}"; do printf "%-8s %s\n" "$s" "${DESC[$s]}"; done
  exit 0
fi
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=("${ORDER[@]}")
for s in "${STAGES[@]}"; do
  [ -n "${DESC[$s]:-}" ] || { echo "unknown stage '$s' (see --list)"; exit 2; }
  echo "=== $(date +%T) stage $s: ${DESC[$s]}"
  if ! stage "$s" > "logs/reproduce/$s.log" 2>&1; then
    echo "=== stage $s FAILED: see logs/reproduce/$s.log (fix, then rerun: finished work is skipped)"
    exit 1
  fi
  echo "=== $(date +%T) stage $s done"
  bash scripts/backup.sh > /dev/null 2>&1 || true            # crash safety: snapshot after every stage
done
echo "=== all requested stages done; provenance in reports/PROVENANCE.csv"
