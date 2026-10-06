"""C5: train one model on a track's source(s), threshold on source val, score every target.

    python scripts/train.py --config configs/train/cic77/rf.yaml
    python scripts/train.py --config configs/train/cic77/mlp.yaml --source lycos17 --seeds 0

--source / --seeds only select which of the config's runs to execute; the logged config is always the fully
resolved one (single source, explicit target list), so every number traces to one run ID and config hash.
"""

import argparse
import logging

import pandas as pd

from xnids.eval import harness
from xnids.utils import config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", default=None, help="restrict to one source dataset")
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 0 or 0,1,2 (default: from config)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = config.load(args.config)
    if args.source:
        cfg["data"]["source"] = args.source
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    table = harness.run_config(cfg, seeds)
    cols = ["model", "source", "target", "kind", "seed", "fpr_at_thr", "dr_at_thr", "oracle_fpr_at_dr", "pr_auc",
            "mcc_at_thr"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
