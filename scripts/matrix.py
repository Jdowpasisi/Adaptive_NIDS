"""C6: results matrix, H1/H2 checks and the comparison with Cantone et al., built from MLflow.

    python scripts/matrix.py                 # every track in configs/matrix.yaml
    python scripts/matrix.py --track cic77

Writes (reports/tables/):
  matrix_long.csv                one row per (model, source, target, seed) with its run_id
  matrix_{track}.csv             mean / std per cell, n_seeds, run_ids
  matrix_{track}_{metric}.csv/.tex   model x source rows by target columns, 'mean ± std'
  h1_checks.csv                  every cross cell vs the within cell of the same model and source
  cantone_comparison.csv         our MCC vs Cantone et al. (2024)
  h2_original_vs_corrected.csv   within-dataset, original CIC-IDS2017 vs LycoS-IDS2017 (+ confound note)
  matrix_missing.csv             planned (config, seed) runs that have no finished run yet
and figures (reports/figures/): matrix_{track}_{metric}.png heatmaps, within_vs_cross_{track}.png.
"""

import argparse
import logging

import numpy as np
import pandas as pd

from xnids.eval import matrix
from xnids.utils import config, paths

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SLOT1, SLOT2 = "#2a78d6", "#eb6834"
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"
HEAT = {  # metric: (title, percent?, kind)
    "mcc_at_thr": ("MCC at the frozen source threshold", False, "diverging"),
    "dr_at_thr": ("Detection rate at the frozen source threshold", True, "seq"),
    "fpr_at_thr": ("False-positive rate at the frozen source threshold", True, "seq_sqrt"),
    "oracle_fpr_at_dr": ("Oracle FPR @ 95% DR (diagnostic: threshold re-picked on the target)", True, "seq"),
}
SHORT = {"lycos17": "LycoS17", "lycos18": "LycoS18", "nf_unsw_v2": "NF-UNSW", "nf_cse_cic18_v2": "NF-CSE18",
         "nf_ton_v2": "NF-ToN", "cic17_orig": "CIC17-orig"}
MODEL_NAMES = {"lda": "LDA", "dt": "DT", "rf": "RF", "xgb": "XGB", "mlp": "MLP", "ae": "AE", "tabnet": "TabNet"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _ink_for(rgb) -> str:
    r, g, b = rgb[:3]
    return "#ffffff" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5 else INK


def heatmaps(agg: pd.DataFrame, track: str, models: list[str]) -> None:
    from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm, TwoSlopeNorm

    plt = _plt()
    a = agg[agg.track == track]
    ds = list(dict.fromkeys(list(a.source) + list(a.target)))
    order = [d for d in SHORT if d in ds]
    models = [m for m in MODEL_NAMES if m in set(a.model)]
    if len(order) < 2 or not models:
        return
    for metric, (title, pct, kind) in HEAT.items():
        if kind == "diverging":
            cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
            norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        else:
            cmap = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
            norm = PowerNorm(0.5, 0, 1) if kind == "seq_sqrt" else Normalize(0, 1)
        ncol = min(4, len(models))
        nrow = int(np.ceil(len(models) / ncol))
        cell = 0.62 if len(order) > 2 else 0.9
        w, h = max(7.5, ncol * (len(order) * cell + 1.3)), nrow * (len(order) * cell + 1.0) + 1.0
        fig, axes = plt.subplots(nrow, ncol, figsize=(w, h), squeeze=False, facecolor=SURFACE, layout="constrained")
        for ax, m in zip(axes.flat, models, strict=False):
            sub = a[a.model == m]
            M = np.full((len(order), len(order)), np.nan)
            for _, r in sub.iterrows():
                M[order.index(r.source), order.index(r.target)] = r[f"{metric}_mean"]
            ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="equal")
            for i in range(len(order)):
                for j in range(len(order)):
                    if np.isnan(M[i, j]):
                        continue
                    txt = f"{100 * M[i, j]:.1f}" if pct else f"{M[i, j]:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                            color=_ink_for(cmap(norm(M[i, j]))), fontweight="bold" if i == j else "normal")
            ax.set_xticks(range(len(order)), [SHORT[d] for d in order], rotation=35, ha="right", fontsize=7.5,
                          color=INK2)
            ax.set_yticks(range(len(order)), [SHORT[d] for d in order], fontsize=7.5, color=INK2)
            ax.set_title(MODEL_NAMES.get(m, m), fontsize=9, color=INK, loc="left")
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            # 2px surface gaps between cells
            ax.set_xticks(np.arange(-0.5, len(order)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
            ax.grid(which="minor", color=SURFACE, linewidth=2)
            ax.tick_params(which="minor", length=0)
        for ax in axes.flat[len(models):]:
            ax.set_visible(False)
        unit = " (%)" if pct else ""
        fig.suptitle(f"{title}{unit}, track {track}. Rows: trained on; columns: tested on; diagonal (bold) = within",
                     x=0.01, ha="left", fontsize=9.5, color=INK)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axes, location="bottom", shrink=0.35, aspect=30, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
        if pct:
            cb.ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}")
        fig.savefig(paths.FIGURES / f"matrix_{track}_{metric}.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)


def within_vs_cross(long: pd.DataFrame, track: str, models: list[str], ref: dict | None) -> None:
    plt = _plt()
    a = long[long.track == track]
    if a.kind.nunique() < 2:
        return
    per_seed = a.groupby(["model", "kind", "seed"])["mcc_at_thr"].mean().reset_index()
    stats = per_seed.groupby(["model", "kind"])["mcc_at_thr"].agg(["mean", "std"]).reset_index()
    models = [m for m in MODEL_NAMES if m in set(stats.model)]      # one fixed model order in every figure
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6.5, 1.1 * len(models) + 2), 3.6), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    bw = 0.36
    for k, (kind, col, label) in enumerate([("within", SLOT1, "within-dataset"), ("cross", SLOT2, "cross-dataset")]):
        s = stats[stats.kind == kind].set_index("model").reindex(models)
        xs = x + (k - 0.5) * (bw + 0.03)
        ax.bar(xs, s["mean"], width=bw, color=col, label=label, zorder=2, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, s["mean"], yerr=s["std"].fillna(0), fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
        for xi, v, sd in zip(xs, s["mean"], s["std"].fillna(0), strict=True):
            lab = f"{0.0 if abs(v) < 0.005 else v:.2f}"
            if v > 0.2:   # at the base of tall bars: clear of error bars and the reference lines above
                ax.text(xi, 0.025, lab, ha="center", va="bottom", fontsize=7, color="#ffffff")
            else:         # short bars: above the error bar
                ax.text(xi, max(v, 0) + sd + 0.02, lab, ha="center", va="bottom", fontsize=7, color=INK2)
    if ref and track == "cic77":
        for val, lab in ((ref["within_avg_mcc"], "Cantone et al. within avg"), (ref["cross_avg_mcc"], "Cantone et al. cross avg")):
            ax.axhline(val, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.text(-0.62, val + 0.012, f"{lab} {val:.2f}", ha="left", va="bottom", fontsize=7, color=MUTED, zorder=4,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(x, [MODEL_NAMES.get(m, m) for m in models], fontsize=8.5, color=INK2)
    ax.set_xlim(-0.65, len(models) - 0.35)
    ax.set_ylabel("MCC at frozen threshold (mean over cells)", fontsize=8, color=INK2)
    ax.set_ylim(min(-0.25, stats["mean"].min() - 0.1), 1.08)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK2, ncol=2, bbox_to_anchor=(1, 1.12))
    ax.set_title(f"Within vs cross-dataset MCC, track {track} (error bars: std over 3 seeds)", fontsize=9.5,
                 color=INK, loc="left", pad=18)
    fig.savefig(paths.FIGURES / f"within_vs_cross_{track}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/matrix.yaml")
    ap.add_argument("--track", default="all")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    mcfg = config.load(args.config)
    plan = mcfg["sweep"]
    tracks = list(plan) if args.track == "all" else args.track.split(",")
    long, missing = matrix.collect(plan, tracks)
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    missing.to_csv(paths.TABLES / "matrix_missing.csv", index=False)
    if long.empty:
        print("no finished runs yet")
        return
    long.to_csv(paths.TABLES / "matrix_long.csv", index=False)
    agg = matrix.aggregate(long)
    for t in tracks:
        a = agg[agg.track == t]
        if a.empty:
            continue
        a.to_csv(paths.TABLES / f"matrix_{t}.csv", index=False)
        for metric, pct in (("fpr_at_thr", True), ("dr_at_thr", True), ("mcc_at_thr", False), ("pr_auc", False),
                            ("oracle_fpr_at_dr", True)):
            w = matrix.wide(agg, t, metric, pct)
            w.to_csv(paths.TABLES / f"matrix_{t}_{metric}.csv")
            unit = r" (\%)" if pct else ""
            w.to_latex(paths.TABLES / f"matrix_{t}_{metric}.tex",
                       caption=f"{metric}{unit}, track {t}: " + r"mean $\pm$ std over seeds. "
                               "Rows: model and training dataset; columns: test dataset.",
                       label=f"tab:matrix_{t}_{metric}")
        heatmaps(agg, t, plan[t])
        within_vs_cross(long, t, plan[t], mcfg.get("cantone"))
    h1 = matrix.h1_checks(agg)
    h1.to_csv(paths.TABLES / "h1_checks.csv", index=False)
    cant = matrix.cantone_comparison(agg, mcfg["cantone"])
    cant.to_csv(paths.TABLES / "cantone_comparison.csv", index=False)
    h2 = matrix.h2_table(agg)
    with open(paths.TABLES / "h2_original_vs_corrected.csv", "w") as f:
        f.write(f"# {matrix.H2_NOTE}\n")
        h2.to_csv(f, index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 40):
        for t in tracks:
            if (agg.track == t).any():
                print(f"\n=== {t}: FPR % at frozen threshold | DR % at frozen threshold")
                print(matrix.wide(agg, t, "fpr_at_thr", True).to_string())
                print(matrix.wide(agg, t, "dr_at_thr", True).to_string())
        print("\n=== H1 (cross vs within, same model and source)")
        print(h1.groupby("track")[["h1_mcc_worse", "h1_oracle_fpr_higher", "h1_frozen_fpr_higher"]]
              .agg(["sum", "count"]).to_string())
        print("\n=== vs Cantone et al.\n" + cant.round(4).to_string(index=False))
        if len(h2):
            print("\n=== H2 original vs corrected (within)\n" + h2.round(4).to_string(index=False))
        print(f"\nmissing runs: {len(missing)}")


if __name__ == "__main__":
    main()
