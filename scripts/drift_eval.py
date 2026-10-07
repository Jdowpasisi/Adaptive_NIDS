"""C8 / H5: evaluate the drift monitor on synthetic streams built from real test splits.

    python scripts/drift_eval.py                       # every pair x seed in configs/drift.yaml
    python scripts/drift_eval.py --pairs 0 --seeds 0   # Mid-Sem demo: LycoS17 -> LycoS18 only

Per (pair, seed), with a fresh monitor (ADWIN state reset) for each stream:
  nodrift  held-out SOURCE validation flows (not in the reference sample): exchangeable with the reference, so
           every alarm is a false alarm                  -> false alarms per 100 windows, per detector
  later    SOURCE test flows: the same network, other time blocks for LycoS17 (real temporal drift, cf. H4);
           for the randomly split datasets this is a second null
  switch   nodrift-style windows, then target windows   -> detection delay in flows after the switch
  ramp     target share rising 0 -> 100% across windows -> delay and target share at first alarm

Outputs: reports/tables/drift_eval_long.csv, drift_h5.csv, drift_vs_retraining.csv, drift_explanations.csv,
reports/figures/drift_switch_{track}_{source}_{target}.png, and one MLflow run per (pair, seed).
"""

import argparse
import collections
import json
import logging

import mlflow
import numpy as np
import pandas as pd
import polars as pl

from xnids.drift.monitor import DriftMonitor
from xnids.eval.harness import load_split
from xnids.models.bundle import bundle_for
from xnids.utils import config, log, paths, seed

DETECTORS = ["ks", "ks_effect", "mmd", "adwin", "combined", "recommend"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SLOT1, SLOT2, SLOT3 = "#2a78d6", "#eb6834", "#1baf7a"


def windows(X: pl.DataFrame, n_windows: int, size: int, rng: np.random.Generator) -> list[pl.DataFrame]:
    n_windows = min(n_windows, X.height // size)
    idx = rng.permutation(X.height)[: n_windows * size]
    # random order WITHIN each window too: file order groups flows by class (e.g. NF-ToN), which ADWIN, the only
    # order-sensitive detector, would otherwise flag as a change inside a drift-free window
    return [X[idx[i * size:(i + 1) * size]] for i in range(n_windows)]


def mixed(Xs: pl.DataFrame, Xt: pl.DataFrame, n_windows: int, size: int, rng) -> tuple[list, list[float]]:
    out, shares = [], []
    for w in range(n_windows):
        share = w / max(n_windows - 1, 1)
        nt = int(round(share * size))
        a = Xs[np.sort(rng.choice(Xs.height, size - nt, replace=False))]
        b = Xt[np.sort(rng.choice(Xt.height, nt, replace=False))]
        ab = pl.concat([a, b])
        out.append(ab[rng.permutation(ab.height)])
        shares.append(share)
    return out, shares


def flags(rep) -> dict[str, bool]:
    return {**rep.detectors, "recommend": rep.recommend}


def eval_pair(cfg: dict, track: str, src: str, tgt: str, s: int) -> tuple[list[dict], dict, list]:
    ec, mc, st = cfg["eval"], cfg["monitor"], cfg["eval"]["streams"]
    size = mc["window"]
    d = ec["data"]
    bundle = bundle_for(ec["model"], track, src, s)
    val = load_split(src, track, "val", d["max_rows_per_split"], d["min_per_family"], d["dev_seed"])
    src_te = load_split(src, track, "test", d["max_rows_per_split"], d["min_per_family"], d["dev_seed"]).X
    tgt_te = load_split(tgt, track, "test", d["max_rows_per_split"], d["min_per_family"], d["dev_seed"]).X
    rng = np.random.default_rng(1000 + s)
    mk = lambda: DriftMonitor(bundle, val.X, val.y, mc)  # noqa: E731
    held = val.X.filter(~pl.Series(np.isin(np.arange(val.X.height), mk().ref_idx)))   # val minus reference
    rows = []

    for scen, pool in (("nodrift", held), ("later", src_te)):
        reps = mk().run(windows(pool, st["nodrift_windows"], size, rng))
        for det in DETECTORS:
            rows.append({"scenario": scen, "detector": det, "windows": len(reps),
                         "fa_per_100": 100 * np.mean([flags(r)[det] for r in reps])})

    pre = windows(held, st["switch_pre"], size, rng)
    post = windows(tgt_te, st["switch_post"], size, rng)
    sw = mk().run(pre + post)
    for det in DETECTORS:
        f = [flags(r)[det] for r in sw]
        hit = next((i for i, v in enumerate(f[len(pre):]) if v), None)
        rows.append({"scenario": "switch", "detector": det, "pre_switch_alarms": int(sum(f[:len(pre)])),
                     "detected": hit is not None, "delay_flows": None if hit is None else (hit + 1) * size,
                     "post_alarm_share": float(np.mean(f[len(pre):]))})

    ramp_w, shares = mixed(held, tgt_te, st["ramp_windows"], size, rng)
    rp = mk().run(ramp_w)
    for det in DETECTORS:
        f = [flags(r)[det] for r in rp]
        first_pos = next((i for i, v in enumerate(f) if v and shares[i] > 0), None)
        rows.append({"scenario": "ramp", "detector": det, "detected": first_pos is not None,
                     "delay_flows": None if first_pos is None else (first_pos + 1) * size,
                     "share_at_detection": None if first_pos is None else shares[first_pos],
                     "alarm_at_share0": bool(f[0])})
    for r in rows:
        r.update({"track": track, "source": src, "target": tgt, "seed": s})

    tops = collections.Counter(f for r in sw[len(pre):] for f in r.top_features)
    expl = {"track": track, "source": src, "target": tgt, "seed": s,
            "top5_post_switch": [f for f, _ in tops.most_common(5)],
            "example_message": sw[len(pre)].message,
            "est_error_rise_pre": float(np.mean([r.est_fpr_rise for r in sw[:len(pre)]])),
            "est_error_rise_post": float(np.mean([r.est_fpr_rise for r in sw[len(pre):]])),
            "attack_share_pred_pre": float(np.mean([r.attack_share_pred for r in sw[:len(pre)]])),
            "attack_share_pred_post": float(np.mean([r.attack_share_pred for r in sw[len(pre):]]))}
    return rows, expl, sw


def c4_overlap(expl: pd.DataFrame) -> pd.DataFrame:
    lab = json.loads((paths.TABLES / "advval_lab_features.json").read_text())
    out = []
    for _, e in expl.iterrows():
        pairs = lab.get(e.track, {})
        key = f"{e.source}|{e.target}" if f"{e.source}|{e.target}" in pairs else f"{e.target}|{e.source}"
        c4 = pairs.get(key, {}).get("all", [])[:10]
        top = e.top5_post_switch
        out.append({**e.to_dict(), "c4_top10": c4, "overlap_at5": len(set(top) & set(c4)),
                    "overlap_share": len(set(top) & set(c4)) / max(len(top), 1)})
    return pd.DataFrame(out)


def vs_retraining(h5: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Cost per day and mean exposure (flows processed while drifted, before action) for one switch per day."""
    c, size = cfg["monitor"]["cost"], cfg["monitor"]["window"]
    per_day = c["flows_per_hour"] * 24
    windows_per_day = per_day / size
    rc = cfg["retrain"]["cost_per_retrain"]
    rows = []
    for det in ("ks", "ks_effect", "mmd", "adwin", "combined", "recommend"):
        nd = h5[(h5.scenario == "nodrift") & (h5.detector == det)]["fa_per_100"].mean()
        swd = h5[(h5.scenario == "switch") & (h5.detector == det)]
        det_rate = swd["detected"].mean()
        delay = swd["delay_flows"].dropna().mean()
        missed_exposure = per_day / 2           # a missed switch stays unhandled for ~half a day on average
        exposure = det_rate * delay + (1 - det_rate) * missed_exposure if det_rate > 0 else missed_exposure
        actions = nd / 100 * windows_per_day + det_rate
        rows.append({"strategy": f"monitor:{det}", "false_alarms_per_day": nd / 100 * windows_per_day,
                     "switch_detect_rate": det_rate, "mean_exposure_flows": exposure,
                     "actions_per_day": actions, "cost_per_day": actions * rc})
    for n in cfg["retrain"]["every_flows"]:
        rows.append({"strategy": f"scheduled:every {n:,} flows", "false_alarms_per_day": np.nan,
                     "switch_detect_rate": 1.0, "mean_exposure_flows": n / 2, "actions_per_day": per_day / n,
                     "cost_per_day": per_day / n * rc})
    return pd.DataFrame(rows)


def plot_switch(sw, n_pre: int, track: str, src: str, tgt: str, alpha_mmd: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    x = np.arange(len(sw))
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.2), sharex=True, facecolor=SURFACE, layout="constrained")
    series = [
        ("Max KS statistic over features", [r.features()["ks_max"] for r in sw], SLOT1, None),
        ("MMD permutation p-value (floor = 1 / (permutations + 1))", [r.mmd[1] for r in sw], SLOT2, alpha_mmd),
        ("Mean model confidence", [r.mean_conf for r in sw], SLOT3, None),
    ]
    for ax, (title, y, col, ref) in zip(axes, series, strict=True):
        ax.set_facecolor(SURFACE)
        ax.plot(x, y, color=col, lw=2, marker="o", ms=4, markeredgecolor=SURFACE, zorder=3)
        ax.axvline(n_pre - 0.5, color=INK2, lw=1, ls=(0, (4, 3)))
        if ref is not None:
            ax.axhline(ref, color=MUTED, lw=1, ls=(0, (1, 2)))
            ax.text(len(sw) - 0.5, ref * 1.15, f"alpha {ref}", ha="right", fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
    adwin = [i for i, r in enumerate(sw) if r.adwin_change]
    for i in adwin:
        axes[2].axvline(i, color=SLOT3, lw=0.8, alpha=0.5)
    axes[1].set_yscale("log")
    axes[0].text(n_pre - 0.6, 0.92, "switch to target", fontsize=7.5, color=INK2, ha="right",
                 transform=axes[0].get_xaxis_transform())
    if adwin:
        axes[2].text(adwin[0] + 0.2, 0.92, "ADWIN change", fontsize=7, color=INK2,
                     transform=axes[2].get_xaxis_transform())
    axes[2].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[2].set_xlabel(f"window ({sw[0].n:,} flows each)", fontsize=8, color=INK2)
    fig.suptitle(f"Drift monitor on a sudden switch: {src} -> {tgt} (track {track}, model trained on {src})",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.savefig(paths.FIGURES / f"drift_switch_{track}_{src}_{tgt}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/drift.yaml")
    ap.add_argument("--pairs", default=None, help="comma list of pair indices (default: all)")
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = config.load(args.config)
    pairs = cfg["eval"]["pairs"]
    sel = [pairs[int(i)] for i in args.pairs.split(",")] if args.pairs else pairs
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else cfg["run"]["seeds"]
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    all_rows, expls = [], []
    for track, src, tgt in sel:
        for s in seeds:
            seed.set_seed(s)
            run_cfg = {k: v for k, v in cfg.items() if k != "eval"} | {
                "eval": {**cfg["eval"], "pairs": [[track, src, tgt]]}}
            with log.start_run(run_cfg, seed=s, experiment=cfg["experiment"], run_name=f"drift-{src}-{tgt}",
                               tags={"track": track, "source": src, "target": tgt}) as run:
                rows, expl, sw = eval_pair(cfg, track, src, tgt, s)
                for r in rows:
                    r["run_id"] = run.info.run_id
                    key = f"{r['scenario']}.{r['detector']}"
                    if "fa_per_100" in r:
                        mlflow.log_metric(f"{key}.fa_per_100", r["fa_per_100"])
                    if r.get("delay_flows") is not None:
                        mlflow.log_metric(f"{key}.delay_flows", r["delay_flows"])
                log.log_df_artifact(pd.DataFrame(rows), "drift_eval.parquet")
                mlflow.log_text("\n".join(rep.to_json() for rep in sw), "switch_reports.jsonl")
                expl["run_id"] = run.info.run_id
            all_rows += rows
            expls.append(expl)
            if s == seeds[0]:
                plot_switch(sw, cfg["eval"]["streams"]["switch_pre"], track, src, tgt, cfg["monitor"]["mmd"]["alpha"])
            logging.info("%s->%s seed %d: %s", src, tgt, s, {r["detector"]: r.get("delay_flows") for r in rows
                                                              if r["scenario"] == "switch"})
    long = pd.DataFrame(all_rows)
    long.to_csv(paths.TABLES / "drift_eval_long.csv", index=False)
    nd = long[long.scenario == "nodrift"].groupby("detector")["fa_per_100"].agg(["mean", "std"]).add_prefix("fa_per_100_")
    nd = nd.join(long[long.scenario == "later"].groupby("detector")["fa_per_100"].mean().rename("alarms_per_100_later"))
    sw = long[long.scenario == "switch"].groupby("detector").agg(
        switch_detect_rate=("detected", "mean"), switch_delay_flows=("delay_flows", "mean"),
        pre_switch_alarms=("pre_switch_alarms", "mean"))
    rp = long[long.scenario == "ramp"].groupby("detector").agg(
        ramp_detect_rate=("detected", "mean"), ramp_delay_flows=("delay_flows", "mean"),
        ramp_share_at_detection=("share_at_detection", "mean"))
    h5 = nd.join(sw).join(rp).reindex(DETECTORS)
    h5.to_csv(paths.TABLES / "drift_h5.csv")
    by_track = long.assign(group=long.track).groupby(["group", "scenario", "detector"]).agg(
        fa=("fa_per_100", "mean"), det=("detected", "mean"), delay=("delay_flows", "mean")).reset_index()
    by_track.to_csv(paths.TABLES / "drift_h5_by_track.csv", index=False)
    vr = vs_retraining(long, cfg)
    vr.to_csv(paths.TABLES / "drift_vs_retraining.csv", index=False)
    ex = c4_overlap(pd.DataFrame(expls))
    ex.to_csv(paths.TABLES / "drift_explanations.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 70):
        print("\n=== H5: detectors (mean over pairs and seeds)\n" + h5.round(3).to_string())
        print("\n=== by track\n" + by_track.pivot_table(index=["group", "detector"], columns="scenario",
                                                         values=["fa", "det"]).round(2).to_string())
        print("\n=== vs scheduled retraining (cost assumptions in configs/drift.yaml)\n" + vr.round(2).to_string(index=False))
        print("\n=== explanations vs C4 lab-telling features\n" + ex[["source", "target", "seed", "top5_post_switch",
                                                                       "overlap_at5"]].to_string(index=False))
        print("\nexample:", ex.example_message.iloc[0])


if __name__ == "__main__":
    main()
