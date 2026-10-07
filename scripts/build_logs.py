"""C10: full-information adaptation logs. Every action on every drift window, with the label-free drift features
the selector will see at decision time and the outcome on the held-out half of the window.

    python scripts/build_logs.py                 # all pairs in configs/adapt/c10.yaml (resumable per pair)
    python scripts/build_logs.py --pairs 0 --windows 3      # quick check
    python scripts/build_logs.py --summary-only

Per pair: data/logs/c10/{track}__{source}__{target}__{hash}.parquet (+ an MLflow run in experiment c10).
All pairs: data/logs/adapt_log.parquet, reports/tables/c10_summary.csv, reports/figures/c10_sanity.png.
Row schema (Build Guide C10): scenario_id, pair, track, window_id, seed | drift features | adapter, params, model |
labels_used, compute_s | fpr/dr/mcc/pr_auc before and after | delta_fpr = fpr_before - fpr_after (+ delta_mcc, delta_dr).
"""

import argparse
import json
import logging
import time

import mlflow
import numpy as np
import pandas as pd
import polars as pl

from xnids import adapt
from xnids.adapt.base import AdaptContext
from xnids.drift.adwin import ConfidenceADWIN
from xnids.drift.monitor import DriftMonitor
from xnids.eval import harness, metrics
from xnids.features import tracks
from xnids.models.bundle import bundle_for
from xnids.select.logs import Pool, make_windows
from xnids.utils import config, log, paths, seed

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
OUT = paths.LOGS / "c10"
SHORT = {"fpr_at_thr": "fpr", "dr_at_thr": "dr", "mcc_at_thr": "mcc", "pr_auc": "pr_auc"}


def label(a: dict) -> str:
    rest = {k: v for k, v in a.items() if k != "name"}
    return a["name"] + ("(" + ",".join(f"{k}={v}" for k, v in rest.items()) + ")" if rest else "")


def outcome(b, X: pl.DataFrame, y: np.ndarray) -> dict:
    ev = metrics.evaluate(y, b.score(X), b.threshold)
    return {k: ev[k] for k in ("fpr_at_thr", "dr_at_thr", "mcc_at_thr", "pr_auc")}


def load_pool(ds: str, track: str, split: str, d: dict, with_ts: bool) -> Pool:
    s = harness.load_split(ds, track, split, d["max_rows_per_split"], d["min_per_family"], d["dev_seed"])
    ts = None
    if with_ts:
        tsdf = pl.scan_parquet(tracks.processed_path(ds, track))
        if "ts" in tsdf.collect_schema():
            ts = (pl.DataFrame({"row_id": s.row_id}).join(tsdf.select("row_id", "ts").collect(), on="row_id",
                                                           how="left")["ts"].to_numpy())
    return Pool(s.X, s.y, s.family, ts)


def run_pair(cfg: dict, track: str, src: str, tgt: str, n_windows: int | None) -> pd.DataFrame:
    d = cfg["data"]
    rcfg = {k: v for k, v in cfg.items() if k != "pairs"} | {"pair": [track, src, tgt]}
    if n_windows:
        rcfg["windows"] = {**rcfg["windows"], "per_pair": n_windows}
    h = config.cfg_hash(rcfg)
    out = OUT / f"{track}__{src}__{tgt}__{h}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    t0 = time.time()
    src_tr = load_pool(src, track, "train", d, False)
    src_va = load_pool(src, track, "val", d, False)
    tgt_tr = load_pool(tgt, track, "train", d, True)
    rng = np.random.default_rng(0)
    si = np.sort(rng.choice(src_tr.X.height, min(cfg["src_rows"], src_tr.X.height), replace=False))
    xi = si[: cfg["xgb_src_rows"]]
    pi = np.sort(rng.choice(tgt_tr.X.height, min(cfg["pool_rows"], tgt_tr.X.height), replace=False))
    seeds = cfg["seeds"]
    mlp = {s: bundle_for("mlp", track, src, s) for s in seeds}
    xgb = {s: bundle_for("xgb", track, src, s) for s in seeds}
    mon = {s: DriftMonitor(mlp[s], src_va.X, src_va.y, cfg["monitor"]) for s in seeds}
    pair_ctx = {s: AdaptContext((src_tr.X[si], src_tr.y[si]), (src_va.X, src_va.y), tgt_tr.X[pi], seed=s)
                for s in seeds}
    per_pair = {}
    for s in seeds:                                    # CORAL / DANN: trained once per (pair, seed), Build Guide
        for a in cfg["actions"]["mlp_per_pair"]:
            t1 = time.time()
            nb = adapt.get(a["name"], **{k: v for k, v in a.items() if k != "name"}).adapt(mlp[s], pair_ctx[s])
            per_pair[(s, label(a))] = (nb, time.time() - t1)
    windows = make_windows(tgt_tr, Pool(src_va.X, src_va.y, src_va.family), cfg["windows"] |
                           ({"per_pair": n_windows} if n_windows else {}), cfg["windows"]["seed"])
    rows = []
    for w, win in enumerate(windows):
        s = seeds[w % len(seeds)]
        seed.set_seed(s)
        m = mon[s]
        m.adwin = ConfidenceADWIN(cfg["monitor"]["adwin"]["delta"])      # each window is judged on its own
        rep = m.process(win.X_adapt)
        feats = rep.features()
        base = {"scenario_id": f"{track}:{src}->{tgt}:{w}", "pair": f"{src}->{tgt}", "track": track,
                "source": src, "target": tgt, "window_id": w, "seed": s, "window_kind": win.kind,
                "window_size": win.size, "target_share": win.target_share,
                "eval_attack_share": float(win.y_eval.mean()), **feats,
                "drift_combined": bool(rep.detectors["combined"]), "drift_recommend": bool(rep.recommend)}
        ctx = AdaptContext((src_tr.X[si], src_tr.y[si]), (src_va.X, src_va.y), win.X_adapt, seed=s,
                           params={"pool_labels": win.y_adapt})
        xctx = AdaptContext((src_tr.X[xi], src_tr.y[xi]), (src_va.X, src_va.y), win.X_adapt, seed=s,
                            params={"pool_labels": win.y_adapt})

        def add(model, b0_out, adapter, params, b, labels, secs, win=win, base=base):
            o = outcome(b, win.X_eval, win.y_eval) if b is not None else b0_out
            rows.append({**base, "model": model, "adapter": adapter, "params": json.dumps(params), "labels_used": labels,
                         "compute_s": secs, **{f"{SHORT[k]}_before": v for k, v in b0_out.items()},
                         **{f"{SHORT[k]}_after": v for k, v in o.items()}})

        for model, bundle, acts, c in (("mlp", mlp[s], cfg["actions"]["mlp"], ctx),
                                       ("xgb", xgb[s], cfg["actions"]["xgb"], xctx)):
            b0 = outcome(bundle, win.X_eval, win.y_eval)
            add(model, b0, "wait", {}, None, 0, 0.0)
            for a in acts:
                t1 = time.time()
                nb = adapt.get(a["name"], **{k: v for k, v in a.items() if k != "name"}).adapt(bundle, c)
                add(model, b0, label(a), a, nb, int(nb.meta.get("labels_used", 0)), time.time() - t1)
            if model == "mlp":
                for a in cfg["actions"]["mlp_per_pair"]:
                    nb, secs = per_pair[(s, label(a))]
                    add(model, b0, label(a), a, nb, 0, secs)
        logging.info("%s->%s window %d/%d (%s, %d flows, seed %d) done", src, tgt, w + 1, len(windows), win.kind,
                     win.size, s)
    df = pd.DataFrame(rows)
    df["delta_fpr"] = df.fpr_before - df.fpr_after          # Build Guide: positive = improvement
    df["delta_mcc"] = df.mcc_after - df.mcc_before
    df["delta_dr"] = df.dr_after - df.dr_before
    OUT.mkdir(parents=True, exist_ok=True)
    with log.start_run(rcfg | {"run": {"name": "c10"}}, experiment=cfg["experiment"], run_name=f"c10-{src}-{tgt}",
                       tags={"track": track, "source": src, "target": tgt}) as run:
        mlflow.log_metric("rows", len(df))
        mlflow.log_metric("minutes", (time.time() - t0) / 60)
        df["run_id"] = run.info.run_id
        log.log_df_artifact(df, "adapt_log_part.parquet")
    df.to_parquet(out, index=False)
    return df


def summarise(log_df: pd.DataFrame) -> pd.DataFrame:
    return log_df.groupby(["track", "model", "adapter"]).agg(
        rows=("delta_fpr", "size"), labels=("labels_used", "mean"), compute_s=("compute_s", "mean"),
        delta_fpr=("delta_fpr", "mean"), delta_mcc=("delta_mcc", "mean"), delta_dr=("delta_dr", "mean"),
        best_within_model_share=("is_best", "mean")).reset_index()   # best among THIS model's actions only


def plot(summ: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tracks_ = list(dict.fromkeys(summ.track))
    fig, axes = plt.subplots(1, len(tracks_), figsize=(5.4 * len(tracks_), 4.8), squeeze=False, sharey=True,
                             facecolor=SURFACE, layout="constrained")
    s2 = summ.assign(name=summ.model + ": " + summ.adapter)
    order = sorted(set(s2.name), key=lambda n: (n.split(":")[0], n))[::-1]
    for ax, t in zip(axes[0], tracks_, strict=True):
        g = s2[s2.track == t].set_index("name").reindex(order)
        y = np.arange(len(order))
        ax.set_facecolor(SURFACE)
        ax.barh(y, g.delta_mcc, color=np.where(g.delta_mcc >= 0, "#2a78d6", "#e34948"), height=0.7,
                edgecolor=SURFACE, linewidth=1, zorder=2)
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_yticks(y, order, fontsize=7.5, color=INK2)
        ax.set_title(f"track {t}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("mean change in MCC on the held-out half (after - before)", fontsize=8, color=INK2)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
    fig.suptitle("C10 sanity check: average effect of each action over all drift windows", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.savefig(paths.FIGURES / "c10_sanity.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/adapt/c10.yaml")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--windows", type=int, default=None, help="override windows per pair (quick checks)")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = config.load(args.config)
    idx = [int(i) for i in args.pairs.split(",")] if args.pairs else range(len(cfg["pairs"]))
    parts = []
    for i in idx:
        track, src, tgt = cfg["pairs"][i]
        if args.summary_only:
            rcfg = {k: v for k, v in cfg.items() if k != "pairs"} | {"pair": [track, src, tgt]}
            p = OUT / f"{track}__{src}__{tgt}__{config.cfg_hash(rcfg)}.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p))
            continue
        parts.append(run_pair(cfg, track, src, tgt, args.windows))
    if not parts:
        print("no C10 logs yet")
        return
    df = pd.concat(parts, ignore_index=True)
    best = df.groupby(["scenario_id", "model"]).delta_mcc.transform("max")
    df["is_best"] = df.delta_mcc == best
    if args.windows is None and args.pairs is None:
        df.to_parquet(paths.LOGS / "adapt_log.parquet", index=False)
    summ = summarise(df)
    summ.to_csv(paths.TABLES / "c10_summary.csv", index=False)
    plot(summ)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summ.round(3).to_string(index=False))
        print(f"\nrows: {len(df)}  windows: {df.scenario_id.nunique()}  pairs: {df.pair.nunique()}  "
              f"tracks: {sorted(df.track.unique())}  adapters: {df.adapter.nunique()}")


if __name__ == "__main__":
    main()
