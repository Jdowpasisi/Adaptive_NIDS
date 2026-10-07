"""C7 / H3: ablation of lab-telling features and the per-domain normalisation variant (resumable).

    python scripts/ablation.py                    # run everything missing, then summarise
    python scripts/ablation.py --summary-only
    python scripts/ablation.py --part ablation --pairs 0,1

Each run starts from the C6 model config (configs/train/<track>/<model>.yaml) plus, from configs/ablation.yaml,
either data.drop_features = the pair's top-k C4 lab-telling features (k in ks) or data.domain_norm = rank.
The resolved config is what is hashed and logged, so finished (config hash, seed) runs are skipped on re-run.
Bundles are not saved (run.save_bundle = false); per-flow scores are still logged to MLflow.

Outputs (reports/tables/): c7_long.csv, c7_shares.csv, c7_norm_shares.csv; figure reports/figures/c7_share_mcc.png
"""

import argparse
import json
import logging

import pandas as pd

from xnids.analysis.ablation import shares
from xnids.eval import harness, matrix
from xnids.utils import config, paths

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def lab_features(track: str, src: str, tgt: str, subset: str) -> list[str]:
    lab = json.loads((paths.TABLES / "advval_lab_features.json").read_text())[track]
    key = f"{src}|{tgt}" if f"{src}|{tgt}" in lab else f"{tgt}|{src}"
    return lab[key][subset]


def plan(acfg: dict, part: str, pair_idx: list[int] | None) -> list[tuple[dict, dict]]:
    """[(resolved config, labels for the long table)]"""
    out = []
    pairs = [p for i, p in enumerate(acfg["pairs"]) if pair_idx is None or i in pair_idx]
    if part in ("all", "ablation"):
        for track, src, tgt in pairs:
            ranking = lab_features(track, src, tgt, acfg["ranking_subset"])
            for m in acfg["models"]:
                base = config.load(paths.CONFIGS / "train" / track / f"{m}.yaml")
                for k in acfg["ks"]:
                    c = config.deep_merge(base, {
                        "experiment": acfg["experiment"], "study": acfg["study"],
                        "run": {"name": f"{m}_{track}_abl_k{k}", "seeds": acfg["seeds"], "save_bundle": False},
                        "data": {"source": src, "targets": [tgt], "drop_features": ranking[:k]}})
                    out.append((harness.expand(c)[0], {"part": "ablation", "k": k, "removed": ";".join(ranking[:k]),
                                                       "pair_target": tgt}))
    if part in ("all", "norm"):
        nc = acfg["normalisation"]
        for track in nc["tracks"]:
            for m in nc["models"]:
                base = config.load(paths.CONFIGS / "train" / track / f"{m}.yaml")
                c = config.deep_merge(base, {
                    "experiment": acfg["experiment"], "study": acfg["study"],
                    "run": {"name": f"{m}_{track}_rank", "seeds": acfg["seeds"], "save_bundle": False},
                    "data": {"domain_norm": "rank"}})
                out += [(e, {"part": "norm", "k": -1, "removed": "", "pair_target": "all"}) for e in harness.expand(c)]
    return out


def run_and_collect(items: list[tuple[dict, dict]], run: bool) -> pd.DataFrame:
    rows = []
    for c, lab in items:
        for s in c["run"]["seeds"]:
            rid = harness.finished_run(c, s)
            if rid is None and run:
                rid = harness.run_one(c, s)["run_id"]
            if rid is None:
                continue
            t = harness.logged_metrics(rid)
            rows.append(t.assign(**lab))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def reference() -> pd.DataFrame:
    long = pd.read_csv(paths.TABLES / "matrix_long.csv")
    return matrix.aggregate(long[long.track.isin(["cic77", "nf43"])])


def plot(sh: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tracks = [t for t in ("cic77", "nf43") if (sh.track == t).any()]
    models = [m for m in ("rf", "xgb", "mlp") if (sh.model == m).any()]
    fig, axes = plt.subplots(len(tracks), len(models), figsize=(3.4 * len(models), 2.9 * len(tracks) + 0.9),
                             squeeze=False, sharex=True, facecolor=SURFACE, layout="constrained")
    pairs = list(dict.fromkeys(zip(sh.source, sh.target, strict=True)))
    color = {p: SLOTS[i % len(SLOTS)] for i, p in enumerate(pairs)}
    for i, t in enumerate(tracks):
        for j, m in enumerate(models):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            ax.axhline(0, color=AXIS, lw=1)
            ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
            g = sh[(sh.track == t) & (sh.model == m)]
            for (s_, t_), gg in g.groupby(["source", "target"]):
                gg = gg.sort_values("k")
                ax.plot(gg.k, gg.share_mcc_at_thr, color=color[(s_, t_)], lw=2, marker="o", ms=4,
                        markeredgecolor=SURFACE, label=f"{s_} → {t_}")
            ax.set_title(f"{m.upper()}, track {t}", fontsize=8.5, color=INK, loc="left")
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(AXIS)
            if j == 0:
                ax.set_ylabel("share of MCC collapse\nexplained", fontsize=8, color=INK2)
            if i == len(tracks) - 1:
                ax.set_xlabel("lab-telling features removed (k)", fontsize=8, color=INK2)
                ax.set_xticks([1, 3, 5, 10])
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles), loc="outside lower center", ncol=4, frameon=False, fontsize=7.5,
               labelcolor=INK2)
    fig.suptitle("Share of the cross-dataset MCC collapse recovered by removing the top-k lab-telling features "
                 "(1 = fully explained; dashed)", x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.savefig(paths.FIGURES / "c7_share_mcc.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ablation.yaml")
    ap.add_argument("--part", default="all", choices=["all", "ablation", "norm"])
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    acfg = config.load(args.config)
    idx = [int(i) for i in args.pairs.split(",")] if args.pairs else None
    long = run_and_collect(plan(acfg, args.part, idx), run=not args.summary_only)
    if long.empty:
        print("no finished C7 runs yet")
        return
    long.to_csv(paths.TABLES / "c7_long.csv", index=False)
    ref = reference()
    keys = ["track", "model", "source", "target", "kind", "k", "pair_target"]   # pair_target keeps each pair's
    # within row (same source, different features dropped) with that pair's cross row
    g = long.groupby(keys)
    agg = g[matrix.HEADLINE].mean().add_suffix("_mean").join(g["seed"].nunique().rename("n_seeds")).reset_index()
    tracks_of = agg[["track", "source"]].drop_duplicates()

    def with_track(df: pd.DataFrame) -> pd.DataFrame:      # shares() returns an empty frame when a part has no runs
        return df.merge(tracks_of, on="source") if len(df) else df

    abl = with_track(shares(agg[agg.k.apply(lambda v: isinstance(v, int | float) and v >= 0)], ref))
    nrm = with_track(shares(agg[agg.k == -1].assign(k="rank"), ref))
    abl.to_csv(paths.TABLES / "c7_shares.csv", index=False)
    nrm.to_csv(paths.TABLES / "c7_norm_shares.csv", index=False)
    if len(abl):
        plot(abl)
    cols = ["track", "model", "source", "target", "k", "mcc_at_thr_cross0", "mcc_at_thr_cross_k", "share_mcc_at_thr",
            "share_oracle_fpr_at_dr", "share_fpr_thr", "mcc_at_thr_within_k"]
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 200):
        if len(abl):
            print("\n=== ablation: share explained\n" + abl[cols].round(3).to_string(index=False))
            print("\n=== median share of MCC collapse by track, model, k\n" + abl.pivot_table(
                index=["track", "model"], columns="k", values="share_mcc_at_thr", aggfunc="median").round(3).to_string())
        if len(nrm):
            print("\n=== normalisation variant\n" + nrm[cols].round(3).to_string(index=False))
    print(f"\nruns collected: {long.run_id.nunique()}; mean seeds per cell: {agg.n_seeds.mean():.2f}")


if __name__ == "__main__":
    main()
