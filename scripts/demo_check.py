"""C13 done-when check: score every flow of data/replay/demo.pcap with the demo model (seed-0 MLP bundle of
configs/train/nfs/mlp.yaml) at its frozen source-val threshold and report, per replay segment, the benign FPR and
the per-family detection rate.

    python scripts/demo_check.py

Writes reports/tables/c13_demo_segments.csv and reports/figures/c13_demo_fpr.png (FPR per 5,000-flow window, the
C15 monitor's window size, in replay order).
"""

import argparse

import numpy as np
import polars as pl

from xnids.data import schema
from xnids.live import replay_labels
from xnids.models.bundle import bundle_for
from xnids.utils import paths

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def plot(w: pl.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    cols = {"A": "#2a78d6", "B1": "#eb6834", "B2": "#eb6834"}
    top = float(w["fpr"].max()) * 100 * 1.25
    for s in w["segment"].unique(maintain_order=True):
        g = w.filter(pl.col("segment") == s)
        ax.plot(g["window"], g["fpr"] * 100, color=cols.get(s, MUTED), lw=1.8, marker="o", ms=3, zorder=3)
        ax.text(g["window"].min(), top * 0.97, f"segment {s}", color=INK2, fontsize=8, va="top")
    for x in w.group_by("segment", maintain_order=True).agg(pl.col("window").min())["window"][1:]:
        ax.axvline(x - 0.5, color=AXIS, lw=1, ls="--", zorder=1)
    ax.set_ylim(0, top)
    ax.set_xlabel("5,000-flow window of demo.pcap (replay order)", fontsize=8, color=INK2)
    ax.set_ylabel("false-positive rate on benign flows (%)", fontsize=8, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.set_title("C13: the demo model's FPR at its frozen threshold, CIC-2017 (A) then CSE-CIC-IDS2018 (B)",
                 fontsize=9.5, color=INK, loc="left")
    fig.savefig(paths.FIGURES / "c13_demo_fpr.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=5000)
    args = ap.parse_args()
    b = bundle_for("mlp", "nfs", "nfs17", args.seed)
    flows = pl.read_parquet(paths.REPLAY / "demo_flows.parquet")
    df = replay_labels.attach(flows)
    assert df["segment"].null_count() == 0
    df = df.sort("timestamp", maintain_order=True)
    X = df.select([pl.col(c).cast(pl.Float32) for c in schema.features("nfs")])
    s = b.score(X)
    df = df.with_columns(score=pl.Series(s), alert=pl.Series(s >= b.threshold))
    rows = []
    for seg, g in df.group_by("segment", maintain_order=True):
        ben = g.filter(pl.col("y") == 0)
        row = {"segment": seg[0], "flows": g.height, "benign": ben.height, "attacks": g.height - ben.height,
               "fpr": ben["alert"].mean() if ben.height else np.nan}
        att = g.filter(pl.col("y") == 1)
        for lbl in sorted(att["label"].unique()):                 # sorted: a deterministic column order
            a = att.filter(pl.col("label") == lbl)
            row[f"dr_{lbl}"] = a["alert"].mean()
            row[f"n_{lbl}"] = a.height
        rows.append(row)
    seg = pl.DataFrame(rows, strict=False)
    seg.write_csv(paths.TABLES / "c13_demo_segments.csv")
    w = (df.with_row_index("i").with_columns(window=(pl.col("i") // args.window))
         .group_by("window", maintain_order=True)
         .agg(pl.col("segment").mode().first(), pl.len().alias("flows"), (pl.col("y") == 0).sum().alias("benign"),
              pl.col("alert").filter(pl.col("y") == 0).mean().alias("fpr"),
              pl.col("alert").filter(pl.col("y") == 1).mean().alias("dr")))
    plot(w.filter(pl.col("benign") >= 50))     # FPR of windows with < 50 benign flows is noise
    with pl.Config(tbl_cols=30, tbl_rows=60, tbl_width_chars=200, float_precision=4):
        print(f"demo model {b.meta.get('version')}: threshold {b.threshold:.4f}")
        print(seg)
        print(w)


if __name__ == "__main__":
    main()
