"""Log one dummy run to MLflow to prove the tracking setup works (C0 'done when')."""

import argparse

import mlflow
import numpy as np

from xnids.utils import config, log, seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    args = ap.parse_args()
    cfg = config.load(args.config)
    for s in cfg["run"]["seeds"]:
        seed.set_seed(s)
        with log.start_run(cfg, seed=s, experiment="smoke") as run:
            mlflow.log_metric("dummy", float(np.random.rand()))
            print(f"run_id={run.info.run_id} cfg_hash={config.cfg_hash(cfg)}")


if __name__ == "__main__":
    main()
