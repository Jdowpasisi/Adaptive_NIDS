#!/usr/bin/env bash
# C7 driver: one process per ablation pair, then the normalisation variant, then the summary (all resumable).
cd "$(dirname "$0")/.."
PY=~/.venvs/driftguard/bin/python
export MLFLOW_DISABLE_AGENT_HINT=1
for i in 0 1 2 3 4 5 6 7; do
  echo "=== ablation pair $i start $(date +%T)"
  $PY scripts/ablation.py --part ablation --pairs "$i" > /dev/null || echo "=== pair $i FAILED"
  echo "=== ablation pair $i done $(date +%T)"
done
echo "=== norm start $(date +%T)"
$PY scripts/ablation.py --part norm > /dev/null || echo "=== norm FAILED"
echo "=== summary $(date +%T)"
$PY scripts/ablation.py --summary-only
echo "=== C7 DONE $(date +%T)"
