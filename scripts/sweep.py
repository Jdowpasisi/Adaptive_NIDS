"""C6: run every config listed in configs/matrix.yaml (resumable: finished (config hash, seed) runs are skipped).

Each config runs in its own `scripts/train.py --resume` subprocess, so all memory is returned to the OS between
configs (a single long-lived process was OOM-killed on the 14 GB laptop).

    python scripts/sweep.py                      # everything, in matrix.yaml order
    python scripts/sweep.py --track cic77 --models lda,rf
"""

import argparse
import logging
import subprocess
import sys
import time

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
            cfg_path = paths.CONFIGS / "train" / track / f"{m}.yaml"
            rc = subprocess.run([sys.executable, str(paths.REPO / "scripts" / "train.py"), "--config", str(cfg_path),
                                 "--resume"], cwd=paths.REPO).returncode
            if rc != 0:
                logging.error("=== %s/%s FAILED with exit code %d (continuing; re-run the sweep to retry)", track, m, rc)
                continue
            logging.info("=== %s/%s done in %.0fs", track, m, time.time() - t0)
    logging.info("=== SWEEP DONE")


if __name__ == "__main__":
    main()
