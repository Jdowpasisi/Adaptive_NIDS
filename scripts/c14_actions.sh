#!/usr/bin/env bash
# C14: every enabled /adapt action built at window 37 of the demo replay (pool = the first two all-benign
# segment-B1 windows), promoted only when all gates pass -> reports/tables/c14_actions_in_B.csv
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-~/.venvs/driftguard/bin/python}
out=reports/tables/c14_actions_in_B.csv
echo "action,gates_passed,failed_gates,canary_dr,canary_fpr,fpr_B1,dr_B2" > "$out"
for a in adabn tent scaling "coral(lam=1.0)" "fewshot(budget=200,rule=random)" \
         "xgb:fewshot(budget=200,rule=random)" "xgb:fewshot(budget=1000,rule=random)"; do
  PYTHONPATH=src $PY -u scripts/c14_walkthrough.py --approve --adapt-from-window 37 --action "$a" \
      --live-dir data/live/actions > logs/c14_action.log 2>&1
  PYTHONPATH=src $PY - "$a" >> "$out" <<'PY'
import json, sys, polars as pl
from xnids.utils import paths
a = sys.argv[1]
import sqlite3
db = sqlite3.connect(paths.REPO / "data/live/actions/live.db")
g = json.loads(db.execute("SELECT detail FROM actions WHERE kind='gate_check'").fetchone()[0])
failed = "+".join(k for k, v in g.items() if isinstance(v, dict) and v.get("passed") is False)
d = pl.read_csv(paths.TABLES / "c14_walkthrough.csv").filter(pl.col("window") > 37)
b1 = d.filter(pl.col("segment") == "B1")["fpr_active"].mean()
b2 = d.filter(pl.col("segment") == "B2")["dr_active"].mean()
print(f'"{a}",{g["all_passed"]},{failed},{g["canary_dr"]["value"]:.4f},{g["canary_fpr"]["value"]:.4f},{b1:.4f},{b2:.4f}')
PY
done
cat "$out"
