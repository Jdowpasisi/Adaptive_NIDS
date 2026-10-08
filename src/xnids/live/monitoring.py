"""C15: build the calibrated live drift monitor from models/monitor/<track>-<source>.json
(written by scripts/calibrate_monitor.py): the same raw reference flows, novelty cutoff and window thresholds."""

import json

import polars as pl

from xnids.data import schema
from xnids.drift.monitor import DriftMonitor
from xnids.models.bundle import bundle_for
from xnids.utils import paths


def calibration_path(cfg: dict):
    m = cfg["model"]
    return paths.MODELS / "monitor" / f"{m['track']}-{m['source']}.json"


def build_monitor(cfg: dict) -> DriftMonitor:
    m, mc = cfg["model"], cfg["monitor"]
    cal = json.loads(calibration_path(cfg).read_text())
    b = bundle_for(m["name"], m["track"], m["source"], m["seed"])
    if b.meta["version"] != cal["model"]:
        raise RuntimeError(f"monitor calibrated for {cal['model']}, not {b.meta['version']}: re-run calibrate_monitor")
    ref = (pl.scan_parquet(paths.PROCESSED / m["source"] / f"{m['track']}.parquet")
           .filter(pl.col("row_id").is_in(cal["reference_row_ids"])).collect())
    c = mc | {"mode": "calibrated", "novelty_tau": cal["novelty_tau"],
              "thresholds": {"novelty_share": cal["thresholds"]["novelty_share"]},
              "reference": {"n": ref.height, "seed": mc["reference"]["seed"]}}
    return DriftMonitor(b, ref.select(schema.features(m["track"])), ref["y"].to_numpy(), c)
