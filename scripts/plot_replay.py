"""C15: figure of one replay run (windows.csv): novelty share vs its calibrated threshold, and the active model's
benign FPR (evaluation overlay), with the first alert and any promotion marked.

    python scripts/plot_replay.py --run file-rehearsal      # -> reports/figures/c15_rehearsal.png
"""

import argparse
import json

import polars as pl

from xnids.live import replay_labels
from xnids.utils import paths

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="file-rehearsal")
    ap.add_argument("--out", default="c15_rehearsal.png")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = paths.REPO / "data/live/runs" / args.run
    w = pl.read_csv(run / "windows.csv").sort("window")
    m = replay_labels.label_map().with_columns(replay_labels.flow_key_expr()).select("flow_key", "segment") \
        .unique("flow_key")
    seg = (pl.read_parquet(run / "flows.parquet", columns=["order", "flow_key"]).join(m, on="flow_key", how="left")
           .with_columns(window=pl.col("order") // 5000).group_by("window").agg(pl.col("segment").mode().first()))
    w = w.join(seg, on="window", how="left").sort("window")
    thr = json.loads((paths.MODELS / "monitor/nfs-nfs17.json").read_text())["thresholds"]["novelty_share"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, facecolor=SURFACE, layout="constrained")
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
        for x in w.group_by("segment", maintain_order=True).agg(pl.col("window").min())["window"][1:]:
            ax.axvline(x - 0.5, color=AXIS, lw=1, ls="--", zorder=1)
    a1.bar(w["window"], w["novelty"] * 100, color=["#eb6834" if r else "#2a78d6" for r in w["recommend"]],
           width=0.8, zorder=2)
    a1.axhline(thr * 100, color=INK2, lw=1, zorder=3)
    a1.text(0, thr * 100 * 1.15, f"calibrated threshold {thr * 100:.1f}% (q99 of time-ordered 2017 windows)",
            fontsize=7.5, color=INK2, va="bottom")
    a1.set_ylabel("novel flows (%)", fontsize=8, color=INK2)
    a1.set_title("C15 rehearsal: demo.pcap through NFStream -> API, calibrated monitor (orange = alert: 2 windows "
                 "over threshold)", fontsize=9.5, color=INK, loc="left")
    for s in w["segment"].unique(maintain_order=True):
        a1.text(w.filter(pl.col("segment") == s)["window"].min() + 1.5, a1.get_ylim()[1] * 0.92, f"segment {s}",
                fontsize=8, color=INK2)
    ok = w.filter(pl.col("benign") >= 50)
    a2.plot(ok["window"], ok["fpr_active"] * 100, color="#2a78d6", marker="o", ms=3, lw=1.6, zorder=3)
    a2.set_ylabel("benign FPR of the\nactive model (%)", fontsize=8, color=INK2)
    a2.set_xlabel("5,000-flow window (emission order); FPR from the replay labels (evaluation overlay)",
                  fontsize=8, color=INK2)
    for r in w.filter(pl.col("event").fill_null("") != "").iter_rows(named=True):
        for ax in (a1, a2):
            ax.axvline(r["window"], color="#1baf7a", lw=1.6, zorder=4)
        a2.text(r["window"] + 0.3, a2.get_ylim()[1] * 0.85, r["event"].replace(", ", "\n"), fontsize=7,
                color="#1baf7a", va="top")
    fig.savefig(paths.FIGURES / args.out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
