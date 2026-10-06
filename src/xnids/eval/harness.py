"""C5 train/eval harness: one config -> train on a source, threshold on source val, score every target.

For each source dataset and each seed (one MLflow run per (source, seed)):
  1. load the source train / val splits (dev-subsampled if data.max_rows_per_split is set);
  2. fit the Preprocessor (median imputer + constant-column drop) on source TRAIN only;
  3. fit every point of model.grid; keep the one with the lowest FPR at target DR on source VAL (ties: PR-AUC);
  4. freeze the threshold t = threshold_at_dr(source val);
  5. score the TEST split of every target in the track (the source itself = the within-dataset cell) at t;
  6. log metrics `{target}/{metric}`, a tidy metrics.parquet and scores_{target}.parquet to MLflow, and save the
     model bundle to models/{version}/.
Nothing about any target is seen before step 5.
"""

import logging
import time
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
import polars as pl

from xnids.data import schema
from xnids.data.split import dev_subsample
from xnids.eval import metrics
from xnids.features import tracks
from xnids.models import zoo
from xnids.models.bundle import Bundle
from xnids.models.preprocess import Preprocessor
from xnids.utils import config, log, paths, seed

LOG = logging.getLogger(__name__)
METRIC_KEYS = ("fpr_at_thr", "dr_at_thr", "pr_auc", "mcc_at_thr", "roc_auc", "oracle_fpr_at_dr")


@dataclass
class Split:
    X: pl.DataFrame
    y: np.ndarray
    row_id: np.ndarray
    family: np.ndarray


def load_split(dataset: str, track: str, split: str, max_rows: int | None = None, min_per_family: int = 5000,
               dev_seed: int = 0) -> Split:
    feats = schema.features(track)
    rows = pl.scan_parquet(paths.SPLITS / f"{dataset}.parquet").filter(pl.col("split") == split).select("row_id")
    if max_rows:
        fam = pl.scan_parquet(tracks.processed_path(dataset, track)).select("row_id", "family")
        rows = dev_subsample(rows.join(fam, on="row_id"), max_rows, min_per_family, dev_seed).select("row_id")
    df = (pl.scan_parquet(tracks.processed_path(dataset, track)).join(rows, on="row_id")
          .select("row_id", "y", "family", *feats).sort("row_id").collect())
    return Split(df.select(feats), df["y"].to_numpy(), df["row_id"].to_numpy(), df["family"].to_numpy())


def expand(cfg: dict) -> list[dict]:
    """One fully resolved config per source (data.source may be a name, a list, or 'all')."""
    d = cfg["data"]
    ds = schema.track_cfg(d["track"])["datasets"]
    src = d.get("source", d.get("sources", "all"))
    sources = ds if src == "all" else [src] if isinstance(src, str) else list(src)
    out = []
    for s in sources:
        if s not in ds:
            raise ValueError(f"source {s} is not in track {d['track']} ({ds})")
        tg = d.get("targets", "all")
        targets = ds if tg == "all" else list(dict.fromkeys([s, *tg]))       # always include the within cell
        c = config.deep_merge(cfg, {"data": {"source": s, "targets": targets}})
        c["data"].pop("sources", None)
        out.append(c)
    return out


def select_on_val(name: str, base: dict, grid: list[dict], seed_: int, device: str, Xtr, ytr, Xval, yval,
                  dr: float) -> tuple[zoo.BaseModel, pd.DataFrame, str]:
    rows, best, best_key, best_g = [], None, None, ""
    for g in grid or [{}]:
        t0 = time.time()
        m = zoo.build(name, base | g, seed_, device).fit(Xtr, ytr, Xval, yval)
        s = m.score(Xval)
        thr = metrics.threshold_at_dr(yval, s, dr)
        fpr, _ = metrics.rates(yval, s, thr)
        ap = metrics.evaluate(yval, s, thr, dr)["pr_auc"]
        rows.append({"grid": str(g), "val_fpr_at_thr": fpr, "val_pr_auc": ap, "fit_s": time.time() - t0, **m.info})
        key = (fpr, -ap)
        if best is None or key < best_key:
            best, best_key, best_g = m, key, str(g)
    return best, pd.DataFrame(rows), best_g


def run_one(cfg: dict, seed_: int) -> dict:
    d, mc = cfg["data"], cfg["model"]
    dr = cfg["eval"]["target_dr"]
    track, source = d["track"], d["source"]
    mr, mpf, dseed = d.get("max_rows_per_split"), d.get("min_per_family", 5000), d.get("dev_seed", 0)
    h = config.cfg_hash(cfg)
    seed.set_seed(seed_)
    with log.start_run(cfg, seed=seed_, experiment=cfg.get("experiment", "train"),
                       run_name=f"{mc['name']}-{track}-{source}",
                       tags={"model": mc["name"], "track": track, "source": source}) as run:
        t0 = time.time()
        tr, va = (load_split(source, track, s, mr, mpf, dseed) for s in ("train", "val"))
        pre = Preprocessor().fit(tr.X)
        Xtr, Xva = pre.transform(tr.X), pre.transform(va.X)
        mlflow.log_params({"n_train": len(Xtr), "n_val": len(Xva), "n_features": Xtr.shape[1],
                           "dropped_constant": ",".join(pre.dropped_constant)[:500]})
        model, grid_tab, chosen = select_on_val(mc["name"], mc.get("params", {}), mc.get("grid", []), seed_,
                                        mc.get("device", "auto"), Xtr, tr.y, Xva, va.y, dr)
        s_val = model.score(Xva)
        thr = metrics.threshold_at_dr(va.y, s_val, dr)
        mlflow.log_metric("threshold", thr)
        mlflow.log_metric("train_s", time.time() - t0)
        log.log_df_artifact(grid_tab, "grid_selection.parquet")
        mlflow.log_params({"selected": chosen})
        for k, v in metrics.evaluate(va.y, s_val, thr, dr).items():
            if k in METRIC_KEYS:
                mlflow.log_metric(f"val/{k}", v)

        rows = []
        for tgt in d["targets"]:
            te = load_split(tgt, track, "test", mr, mpf, dseed)
            s = model.score(pre.transform(te.X))
            ev = metrics.evaluate(te.y, s, thr, dr)
            mlflow.log_metrics({f"{tgt}/{k}": v for k, v in ev.items() if k in METRIC_KEYS})
            rows.append({"cfg_hash": h, "run_id": run.info.run_id, "model": mc["name"], "track": track,
                         "source": source, "target": tgt, "kind": "within" if tgt == source else "cross",
                         "seed": seed_, "threshold": thr, **ev})
            log.log_df_artifact(pd.DataFrame({"row_id": te.row_id, "score": s.astype(np.float32), "y": te.y}),
                                f"scores_{tgt}.parquet")
            LOG.info("%s %s->%s seed %d: FPR %.4f at DR %.4f (oracle FPR %.4f) PR-AUC %.4f MCC %.4f", mc["name"],
                     source, tgt, seed_, ev["fpr_at_thr"], ev["dr_at_thr"], ev["oracle_fpr_at_dr"], ev["pr_auc"],
                     ev["mcc_at_thr"])
        table = pd.DataFrame(rows)
        log.log_df_artifact(table, "metrics.parquet")

        version = f"{mc['name']}-{track}-{source}-s{seed_}-{h}"
        sha, _ = log.git_state()
        bundle = Bundle(model, pre, thr, pre.features_in, {
            "version": version, "cfg_hash": h, "run_id": run.info.run_id, "parent_version": None,
            "model": mc["name"], "track": track, "target_dr": dr, "git_commit": sha, "seed": seed_,
            "training_data": {"dataset": source, "split_hash": pl.scan_parquet(paths.SPLITS / f"{source}.parquet")
                              .select(pl.col("split_hash").first()).collect().item(),
                              "n_train": len(Xtr), "n_val": len(Xva), "max_rows_per_split": mr}})
        bundle.save(paths.MODELS / version)
        mlflow.set_tag("bundle", str(paths.MODELS / version))
        return {"run_id": run.info.run_id, "version": version, "metrics": table}


def run_config(cfg: dict, seeds: list[int] | None = None) -> pd.DataFrame:
    out = []
    for c in expand(cfg):
        for s in seeds if seeds is not None else c["run"]["seeds"]:
            out.append(run_one(c, s)["metrics"])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
