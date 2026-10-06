"""C4: adversarial validation for every dataset pair of every track.

    python scripts/advval.py                         # all tracks in configs/advval.yaml (resumable)
    python scripts/advval.py --track cic77 --seeds 0

Per (track, pair, subset, seed) one MLflow run (experiment 'advval') is logged with its curve. Outputs:
  reports/tables/advval/{track}__{a}__{b}__{subset}.csv   AUC after each removal step, all seeds
  reports/tables/advval_summary.csv                       one row per (track, pair, subset)
  reports/tables/advval_lab_features.json                 consensus lab-telling ranking, for C7 and C8
  reports/figures/advval_{track}.png                      AUC vs features removed (mean ± std over seeds)
"""

import argparse
import itertools
import json
import logging

import mlflow
import numpy as np
import pandas as pd

from xnids.analysis import advval
from xnids.data import schema
from xnids.utils import config, log, paths, seed

OUT = paths.TABLES / "advval"


def pairs_for(track: str, spec) -> list[tuple[str, str]]:
    ds = schema.track_cfg(track)["datasets"]
    return list(itertools.combinations(ds, 2)) if spec == "all_pairs" else [tuple(p) for p in spec]


def curve_path(track: str, a: str, b: str, subset: str):
    return OUT / f"{track}__{a}__{b}__{subset}.csv"


def run_one(cfg: dict, track: str, a: str, b: str, subset: str, s: int) -> pd.DataFrame:
    run_cfg = {k: v for k, v in cfg.items() if k not in ("tracks", "subsets")} | {
        "track": track, "pair": [a, b], "subset": subset}
    h = config.cfg_hash(run_cfg)
    path = curve_path(track, a, b, subset)
    if path.exists():
        prev = pd.read_csv(path)
        done = prev[(prev["seed"] == s) & (prev["cfg_hash"] == h)]
        if len(done):
            return done
    seed.set_seed(s)
    n = cfg["sample_per_domain"]
    Xs, Xt = (advval.load_domain(d, track, subset, n, s) for d in (a, b))
    with log.start_run(run_cfg, seed=s, experiment="advval", run_name=f"{track}-{a}-{b}-{subset}",
                       tags={"track": track, "pair": f"{a}|{b}", "subset": subset}) as run:
        curve = advval.iterative_removal(Xs, Xt, s, cfg["model"], cfg["cv_folds"], cfg["auc_stop"],
                                         cfg["max_removed"])
        curve.insert(0, "seed", s)
        curve["cfg_hash"], curve["run_id"] = h, run.info.run_id
        curve["n_source"], curve["n_target"] = len(Xs), len(Xt)
        mlflow.log_metric("auc_all_features", curve["auc"].iloc[0])
        mlflow.log_metric("auc_final", curve["auc"].iloc[-1])
        mlflow.log_metric("n_removed", len(curve) - 1)
        log.log_df_artifact(curve, "advval_curve.parquet")
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[old["seed"] != s]
    pd.concat([old, curve], ignore_index=True).sort_values(["seed", "step"]).to_csv(path, index=False)
    return curve


def summarise(cfg: dict) -> pd.DataFrame:
    rows, lab = [], {}
    for track, spec in cfg["tracks"].items():
        for a, b in pairs_for(track, spec):
            for subset in cfg["subsets"]:
                p = curve_path(track, a, b, subset)
                if not p.exists():
                    continue
                c = pd.read_csv(p)
                first = c[c["step"] == 0]["auc"]
                last = c.sort_values("step").groupby("seed").tail(1)
                rank = advval.consensus_ranking(c, cfg["max_removed"])
                lab.setdefault(track, {}).setdefault(f"{a}|{b}", {})[subset] = rank
                rows.append({"track": track, "source": a, "target": b, "subset": subset,
                             "seeds": c["seed"].nunique(),
                             "auc_all_mean": first.mean(), "auc_all_std": first.std(ddof=0),
                             "auc_final_mean": last["auc"].mean(), "removed_mean": last["step"].mean(),
                             "reached_stop": bool((last["auc"] < cfg["auc_stop"]).all()),
                             "top5_consensus": ";".join(rank[:5]), "run_ids": ";".join(c["run_id"].unique())})
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    (paths.TABLES / "advval_lab_features.json").write_text(json.dumps(lab, indent=1))
    out = pd.DataFrame(rows)
    out.to_csv(paths.TABLES / "advval_summary.csv", index=False)
    return out


# Reference palette (dataviz skill, light mode): slot 1 blue, slot 2 orange; chrome tokens.
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = {"benign": "#2a78d6", "all": "#eb6834"}
MARKER = {"benign": "o", "all": "s"}          # secondary encoding, so identity is never colour alone
DRAW_ORDER = ["all", "benign"]                # benign on top: it is the headline subset


def plot(cfg: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    for track, spec in cfg["tracks"].items():
        prs = [p for p in pairs_for(track, spec)
               if any(curve_path(track, *p, s).exists() for s in cfg["subsets"])]
        if not prs:
            continue
        ncol = min(4, len(prs))
        nrow = int(np.ceil(len(prs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(max(7.5, 3.3 * ncol), 2.7 * nrow + 0.9), squeeze=False,
                                 sharex=True, sharey=True, facecolor=SURFACE, layout="constrained")
        for ax, (a, b) in zip(axes.flat, prs, strict=False):
            ax.set_facecolor(SURFACE)
            ax.axhline(cfg["auc_stop"], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.axhline(0.5, color=AXIS, lw=1, zorder=1)
            means = {}
            for subset in sorted(cfg["subsets"], key=DRAW_ORDER.index):
                p = curve_path(track, a, b, subset)
                if not p.exists():
                    continue
                g = pd.read_csv(p).groupby("step")["auc"].agg(["mean", "std", "count"]).reset_index()
                g["std"] = g["std"].fillna(0)
                means[subset] = g.set_index("step")["mean"]
                col = SERIES[subset]
                ax.fill_between(g["step"], g["mean"] - g["std"], g["mean"] + g["std"], color=col, alpha=0.15, lw=0)
                ax.plot(g["step"], g["mean"], color=col, lw=2, marker=MARKER[subset], ms=5, zorder=3,
                        markeredgecolor=SURFACE, markeredgewidth=1, label=f"{subset} traffic")
            if len(means) == 2:
                diff = (means["benign"] - means["all"]).abs().dropna()
                if len(diff) and diff.max() < 0.003:
                    ax.text(0.98, 0.06, "benign and all overlap", transform=ax.transAxes, ha="right",
                            fontsize=7.5, color=INK2)
            ax.set_title(f"{a} vs {b}", fontsize=9, color=INK, loc="left")
            ax.set_ylim(0.45, 1.02)
            ax.set_xlim(-0.3, cfg["max_removed"] + 0.3)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.tick_params(colors=MUTED, labelsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(AXIS)
        for ax in axes.flat[len(prs):]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("features removed", fontsize=8, color=INK2)
        for ax in axes[:, 0]:
            ax.set_ylabel("domain AUC", fontsize=8, color=INK2)
        h, lab = axes.flat[0].get_legend_handles_labels()
        order = sorted(range(len(lab)), key=lambda i: lab[i] != "benign traffic")
        h, lab = [h[i] for i in order], [lab[i] for i in order]
        fig.legend(h + [plt.Line2D([], [], color=MUTED, ls=(0, (4, 3)))], lab + [f"stop at AUC {cfg['auc_stop']}"],
                   loc="outside lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK2)
        fig.suptitle(f"Adversarial validation, track {track}: how easily a classifier tells the two datasets apart",
                     x=0.01, ha="left", fontsize=10, color=INK)
        fig.savefig(paths.FIGURES / f"advval_{track}.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/advval.yaml")
    ap.add_argument("--track", default="all")
    ap.add_argument("--subset", default="all_subsets")
    ap.add_argument("--seeds", default=None, help="comma list; default from config")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    cfg = config.load(args.config)
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else cfg["run"]["seeds"]
    subsets = cfg["subsets"] if args.subset == "all_subsets" else args.subset.split(",")
    if not args.summary_only:
        for track, spec in cfg["tracks"].items():
            if args.track not in ("all", track):
                continue
            for a, b in pairs_for(track, spec):
                for subset in subsets:
                    for s in seeds:
                        c = run_one(cfg, track, a, b, subset, s)
                        logging.info("%s %s|%s %s seed %d: AUC %.4f -> %.4f after %d removed (%s)", track, a, b,
                                     subset, s, c["auc"].iloc[0], c["auc"].iloc[-1], len(c) - 1,
                                     ", ".join(c["removed"].dropna()))
    summary = summarise(cfg)
    plot(cfg)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60):
        print(summary.drop(columns=["run_ids"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
