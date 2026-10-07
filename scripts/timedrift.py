"""C7 / H4: train on LycoS17's early days, watch benign FPR hour by hour over the later days, and compare with the
sudden switch to LycoS18.

    python scripts/timedrift.py                # run (resumable per model x seed) and summarise
    python scripts/timedrift.py --summary-only

Outputs: reports/tables/h4_timedrift.csv (per model, seed, hour block), h4_summary.csv,
reports/figures/h4_benign_fpr_over_time.png; one MLflow run per (model, seed) in experiment c7.
"""

import argparse
import logging

import mlflow
import numpy as np
import pandas as pd
import polars as pl

from xnids.data import schema
from xnids.drift import ks
from xnids.eval import harness, metrics
from xnids.features import tracks
from xnids.models.preprocess import Preprocessor
from xnids.utils import config, log, paths, seed

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SLOTS = {"ae": "#2a78d6", "mlp": "#eb6834", "xgb": "#1baf7a", "rf": "#eda100"}


def lycos17_frame(cfg: dict) -> pl.DataFrame:
    feats = schema.features(cfg["track"])
    lf = (pl.scan_parquet(tracks.processed_path(cfg["source"], cfg["track"]))
          .join(pl.scan_parquet(paths.SPLITS / f"{cfg['source']}.parquet").select("row_id", "split"), on="row_id")
          .with_columns(local=pl.from_epoch("ts", time_unit="us").dt.offset_by(f"{cfg['utc_offset_hours']}h"))
          .with_columns(day=pl.col("local").dt.strftime("%a"), hour=pl.col("local").dt.hour()))
    return lf.select("row_id", "split", "day", "hour", "y", "family", *feats).collect()


def run_one(cfg: dict, df: pl.DataFrame, m: str, s: int) -> pd.DataFrame:
    base = config.load(paths.CONFIGS / "train" / cfg["track"] / f"{m}.yaml")
    rcfg = config.deep_merge({k: v for k, v in cfg.items() if k not in ("models", "seeds")},
                             {"model": base["model"], "eval": base["eval"], "run": {"name": f"h4_{m}"}})
    feats = schema.features(cfg["track"])
    dr = base["eval"]["target_dr"]
    early = df.filter(pl.col("day").is_in(cfg["early_days"]))
    tr, va = early.filter(pl.col("split") == "train"), early.filter(pl.col("split") == "val")
    seed.set_seed(s)
    with log.start_run(rcfg, seed=s, experiment=cfg["experiment"], run_name=f"h4-{m}",
                       tags={"study": cfg["study"], "model": m}) as run:
        pre = Preprocessor().fit(tr.select(feats))
        Xtr, Xva = pre.transform(tr.select(feats)), pre.transform(va.select(feats))
        model, _, _ = harness.select_on_val(m, base["model"].get("params", {}), base["model"].get("grid", []), s,
                                            base["model"].get("device", "auto"), Xtr, tr["y"].to_numpy(), Xva,
                                            va["y"].to_numpy(), dr)
        thr = metrics.threshold_at_dr(va["y"].to_numpy(), model.score(Xva), dr)
        ref_b = Xva[va["y"].to_numpy() == 0]
        ref_b = ref_b[np.random.default_rng(0).choice(len(ref_b), min(20000, len(ref_b)), replace=False)]
        rows = []

        def block(frame: pl.DataFrame, label: dict) -> None:
            if frame.height == 0:
                return
            y = frame["y"].to_numpy()
            X = pre.transform(frame.select(feats))
            sc = model.score(X)
            ben, att = y == 0, y == 1
            kb = ks.ks_features(ref_b, X[ben], pre.features_out) if ben.sum() >= 50 else {}
            rows.append({**label, "n_benign": int(ben.sum()), "n_attack": int(att.sum()),
                         "fpr_benign": float((sc[ben] >= thr).mean()) if ben.any() else np.nan,
                         "dr": float((sc[att] >= thr).mean()) if att.any() else np.nan,
                         "benign_score_q99": float(np.quantile(sc[ben], 0.99)) if ben.any() else np.nan,
                         "ks_max_benign": max((v[0] for v in kb.values()), default=np.nan),
                         "attack_families": ";".join(sorted(set(frame.filter(pl.col("y") == 1)["family"])))})

        block(early.filter(pl.col("split") == "test"), {"period": "early_test", "day": "Mon-Tue", "hour": -1})
        for (day, hour), g in df.filter(pl.col("day").is_in(cfg["later_days"])).group_by(["day", "hour"]):
            block(g, {"period": "later", "day": day, "hour": int(hour)})
        tgt = harness.load_split(cfg["sudden_target"], cfg["track"], "test", 2_000_000, 5000, 0)
        block(pl.DataFrame({"y": tgt.y, "family": tgt.family}).hstack(tgt.X), {"period": "sudden", "day": "LycoS18",
                                                                               "hour": -1})
        out = pd.DataFrame(rows).assign(model=m, seed=s, threshold=thr, run_id=run.info.run_id)
        log.log_df_artifact(out, "h4_blocks.parquet")
        for _, r in out[out.period != "later"].iterrows():
            mlflow.log_metric(f"{r.period}/fpr_benign", r.fpr_benign)
        mlflow.log_metric("later/fpr_benign_mean", out[out.period == "later"].fpr_benign.mean())
    return out


def plot(t: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = {"Wed": 0, "Thu": 1, "Fri": 2}
    later = t[t.period == "later"].copy()
    later["x"] = later.day.map(order) * 10 + (later.hour - 8)
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    for m, col in SLOTS.items():
        g = later[later.model == m].groupby("x").fpr_benign.mean()
        ax.plot(g.index, 100 * g.values, color=col, lw=2, marker="o", ms=3.5, markeredgecolor=SURFACE,
                label=f"{m.upper()}: later days, hourly")
        early = t[(t.model == m) & (t.period == "early_test")].fpr_benign.mean()
        sudden = t[(t.model == m) & (t.period == "sudden")].fpr_benign.mean()
        ax.axhline(100 * sudden, color=col, lw=1.2, ls=(0, (4, 3)), label=f"{m.upper()}: sudden switch (LycoS18)")
        ax.axhline(100 * early, color=col, lw=1, ls=(0, (1, 2)), alpha=0.8)
    for d, i in order.items():
        ax.axvline(i * 10 - 1, color=AXIS, lw=1)
        ax.text(i * 10 - 0.6, 0.98, d, transform=ax.get_xaxis_transform(), fontsize=8, color=INK2, va="top")
    ax.set_xticks([i * 10 + h for i in order.values() for h in (0, 4, 8)],
                  [f"{8 + h}:00" for _ in order for h in (0, 4, 8)])
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylabel("benign FPR (%) at the\nMon-Tue threshold", fontsize=8, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=7.5,
               labelcolor=INK2)
    ax.set_title("H4: gradual drift (LycoS17, later days) vs sudden switch (LycoS18); dotted = Mon-Tue test FPR",
                 fontsize=9.5, color=INK, loc="left")
    fig.savefig(paths.FIGURES / "h4_benign_fpr_over_time.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/timedrift.yaml")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = config.load(args.config)
    out = paths.TABLES / "h4_timedrift.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["model", "seed"])
    if not args.summary_only:
        df = lycos17_frame(cfg)
        for m in cfg["models"]:
            for s in cfg["seeds"]:
                if ((done.model == m) & (done.seed == s)).any():
                    continue
                done = pd.concat([done, run_one(cfg, df, m, s)], ignore_index=True)
                done.to_csv(out, index=False)
                logging.info("H4 %s seed %d done", m, s)
    t = done
    summ = t.groupby(["model", "period"]).agg(fpr_benign=("fpr_benign", "mean"), fpr_benign_max=("fpr_benign", "max"),
                                             benign_q99=("benign_score_q99", "mean"), ks_max=("ks_max_benign", "mean"),
                                             dr=("dr", "mean"), threshold=("threshold", "mean")).reset_index()
    later_day = t[t.period == "later"].groupby(["model", "day"]).agg(fpr_benign=("fpr_benign", "mean"),
                                                                     dr=("dr", "mean")).reset_index()
    summ.to_csv(paths.TABLES / "h4_summary.csv", index=False)
    later_day.to_csv(paths.TABLES / "h4_by_day.csv", index=False)
    plot(t)
    with pd.option_context("display.width", 200):
        print(summ.pivot_table(index="model", columns="period",
                               values=["fpr_benign", "benign_q99", "ks_max", "dr", "threshold"]).round(4).to_string())
        print(later_day.pivot_table(index="model", columns="day", values=["fpr_benign", "dr"]).round(4))


if __name__ == "__main__":
    main()
