"""C6: run every config listed in configs/matrix.yaml (resumable: finished (config hash, seed) runs are skipped).

    python scripts/sweep.py                      # everything, in matrix.yaml order
    python scripts/sweep.py --track cic77 --models lda,rf
"""

import argparse
import logging
import time

from xnids.eval import harness
from xnids.utils import config, paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/matrix.yaml")
    ap.add_argument("--track", default="all")
    ap.add_argument("--models", default="all")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    plan = config.load(args.config)["sweep"]
    for track, models in plan.items():
        if args.track not in ("all", track):
            continue
        for m in models:
            if args.models != "all" and m not in args.models.split(","):
                continue
            t0 = time.time()
            logging.info("=== %s/%s start", track, m)
            harness.run_config(config.load(paths.CONFIGS / "train" / track / f"{m}.yaml"), resume=True)
            logging.info("=== %s/%s done in %.0fs", track, m, time.time() - t0)
    logging.info("=== SWEEP DONE")


if __name__ == "__main__":
    main()
