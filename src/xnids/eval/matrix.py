"""C6: build the train x test results matrix from MLflow.

Only runs whose config hash equals the hash of the CURRENT config files (configs/train/<track>/<model>.yaml,
expanded per source) are used, latest FINISHED run per (hash, seed). So every number in a table traces to one run
ID, and a stale or hand-run experiment cannot leak in. Cells with fewer than the planned seeds are kept but flagged
by `n_seeds`.
"""

import logging

import numpy as np
import pandas as pd

from xnids.eval import harness
from xnids.utils import config, paths

LOG = logging.getLogger(__name__)
HEADLINE = ["fpr_at_thr", "dr_at_thr", "pr_auc", "mcc_at_thr", "oracle_fpr_at_dr", "roc_auc"]


def collect(plan: dict, tracks: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(long table: one row per model/source/target/seed, missing: planned (config, seed) without a run)."""
    rows, missing = [], []
    for track, models in plan.items():
        if tracks and track not in tracks:
            continue
        for m in models:
            cfg = config.load(paths.CONFIGS / "train" / track / f"{m}.yaml")
            for c in harness.expand(cfg):
                for s in c["run"]["seeds"]:
                    rid = harness.finished_run(c, s)
                    if rid is None:
                        missing.append({"track": track, "model": m, "source": c["data"]["source"], "seed": s,
                                        "cfg_hash": config.cfg_hash(c)})
                        continue
                    t = harness.logged_metrics(rid)
                    t = t[t["target"].isin(c["data"]["targets"])].copy()
                    t["track"] = track      # the plan group: keeps e.g. the H2 control apart from the cic77 cells
                    rows.append(t)
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return long, pd.DataFrame(missing)


def aggregate(long: pd.DataFrame) -> pd.DataFrame:
    """mean and std (over seeds, ddof=1) of every headline metric per (track, model, source, target)."""
    keys = ["track", "model", "source", "target", "kind"]
    g = long.groupby(keys)
    out = g[HEADLINE].mean().add_suffix("_mean").join(g[HEADLINE].std().add_suffix("_std"))
    out["n_seeds"] = g["seed"].nunique()
    out["run_ids"] = g["run_id"].agg(lambda r: ";".join(sorted(set(r))))
    return out.reset_index()


def fmt(mean: float, std: float, pct: bool = False) -> str:
    if np.isnan(mean):
        return "–"
    k = 100 if pct else 1
    sd = "" if np.isnan(std) else f" ± {k * std:.2f}" if pct else f" ± {std:.3f}"
    return f"{k * mean:.2f}{sd}" if pct else f"{mean:.3f}{sd}"


def wide(agg: pd.DataFrame, track: str, metric: str, pct: bool = False) -> pd.DataFrame:
    """model x source (rows) by target (columns) with 'mean ± std' strings."""
    a = agg[agg.track == track].copy()
    a["cell"] = [fmt(m, s, pct) for m, s in zip(a[f"{metric}_mean"], a[f"{metric}_std"], strict=True)]
    return a.pivot_table(index=["model", "source"], columns="target", values="cell", aggfunc="first")


def h1_checks(agg: pd.DataFrame) -> pd.DataFrame:
    """For every cross cell, compare with the within cell of the same model and source."""
    w = agg[agg.kind == "within"].set_index(["track", "model", "source"])
    c = agg[agg.kind == "cross"].copy()
    for m in ("mcc_at_thr", "oracle_fpr_at_dr", "dr_at_thr", "fpr_at_thr", "pr_auc"):
        c[f"within_{m}"] = [w.loc[(t, mo, s), f"{m}_mean"] for t, mo, s in zip(c.track, c.model, c.source, strict=True)]
    c["mcc_drop"] = c["within_mcc_at_thr"] - c["mcc_at_thr_mean"]
    c["oracle_fpr_rise"] = c["oracle_fpr_at_dr_mean"] - c["within_oracle_fpr_at_dr"]
    c["h1_mcc_worse"] = c["mcc_drop"] > 0
    c["h1_oracle_fpr_higher"] = c["oracle_fpr_rise"] > 0
    c["h1_frozen_fpr_higher"] = c["fpr_at_thr_mean"] > c["within_fpr_at_thr"]
    return c[["track", "model", "source", "target", "within_mcc_at_thr", "mcc_at_thr_mean", "mcc_drop",
              "within_oracle_fpr_at_dr", "oracle_fpr_at_dr_mean", "oracle_fpr_rise", "within_dr_at_thr",
              "dr_at_thr_mean", "within_fpr_at_thr", "fpr_at_thr_mean", "h1_mcc_worse", "h1_oracle_fpr_higher",
              "h1_frozen_fpr_higher", "n_seeds"]]


def cantone_comparison(agg: pd.DataFrame, ref: dict) -> pd.DataFrame:
    a = agg[(agg.track == "cic77") & agg.model.isin(ref["models"])]
    rows = []

    def add(what, ours, theirs, note=""):
        rows.append({"comparison": what, "ours_mcc": ours, "cantone_mcc": theirs, "note": note})

    add("within-dataset average (LDA/DT/RF/XGB)", a[a.kind == "within"]["mcc_at_thr_mean"].mean(), ref["within_avg_mcc"],
        "theirs averages 4 datasets incl. original CIC releases; ours is LycoS17 + LycoS18 only")
    add("cross-dataset average (LDA/DT/RF/XGB)", a[a.kind == "cross"]["mcc_at_thr_mean"].mean(), ref["cross_avg_mcc"],
        "theirs averages 12 cross pairs; ours is the 2 LycoS pairs")
    p = a[(a.source == "lycos17") & (a.target == "lycos18")]
    add("LycoS17 -> LycoS18 (average of the 4 models)", p["mcc_at_thr_mean"].mean(), ref["lycos17_to_lycos18_mcc"],
        "their worst cross pair")
    p = a[(a.source == "lycos18") & (a.target == "lycos17") & (a.model == "lda")]
    add("LDA, LycoS18 -> LycoS17", p["mcc_at_thr_mean"].mean() if len(p) else np.nan, ref["lda_lycos18_to_lycos17_mcc"],
        "their single best cross result")
    return pd.DataFrame(rows)


def h2_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Within-dataset performance: original CIC-IDS2017 (random split) vs corrected LycoS-IDS2017 on a RANDOM split
    (the H2 control, like for like) and on its time-block split (the main matrix)."""
    w = agg[agg.kind == "within"]
    o = w[w.track == "cic_orig"].set_index("model")
    r_ = w[w.track == "h2_control"].set_index("model")
    t_ = w[(w.track == "cic77") & (w.source == "lycos17")].set_index("model")
    rows = []
    for m in [m for m in o.index if m in t_.index]:
        r = {"model": m}
        for k in ("mcc_at_thr", "fpr_at_thr", "dr_at_thr", "pr_auc"):
            r[f"original_{k}"] = o.loc[m, f"{k}_mean"]
            r[f"corrected_random_{k}"] = r_.loc[m, f"{k}_mean"] if m in r_.index else np.nan
            r[f"corrected_timeblock_{k}"] = t_.loc[m, f"{k}_mean"]
        r["mcc_original_minus_corrected_random"] = r["original_mcc_at_thr"] - r["corrected_random_mcc_at_thr"]
        r["pr_auc_original_minus_corrected_random"] = r["original_pr_auc"] - r["corrected_random_pr_auc"]
        rows.append(r)
    return pd.DataFrame(rows)


H2_NOTE = ("H2 compares like with like via the control: original CIC-IDS2017 and LycoS-IDS2017 both on random "
           "stratified splits (columns original_* vs corrected_random_*). Remaining confound: the original release "
           "uses CICFlowMeter features and LycoS uses LycoSTand, so the difference is 'original vs corrected release' "
           "(extractor + labels), not label errors alone. corrected_timeblock_* shows how much the time-block split "
           "alone changes LycoS17.")
