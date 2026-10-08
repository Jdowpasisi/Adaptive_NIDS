"""C15 load test: the sustained flows/s at which /score p99 < 100 ms on this laptop.

    python scripts/load_test.py [--rates 1000 2000 5000 10000 20000 0] [--seconds 20]

Each rate is one replay run of pre-extracted demo flows (source 'parquet', so NFStream extraction does not cap it),
through the same runner (200 ms micro-batches, max 1,000 flows per request, calibrated monitor ON in its thread),
against a fresh API (scores stored). Rate 0 = top speed. The end-to-end file-mode ceiling (NFStream, one meter, plus
the API) comes from a file-mode run of the whole demo.pcap. Writes reports/tables/c15_load.csv and
reports/figures/c15_load.png.
"""

import argparse
import json

import polars as pl

from xnids.live.monitoring import build_monitor
from xnids.live.replay import FlowSource, ReplayRunner
from xnids.live.server import ApiServer
from xnids.utils import config, paths

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def plot(t: pl.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    ax.plot(t["achieved_flows_per_s"], t["p99_ms"], color="#2a78d6", marker="o", lw=1.8, zorder=3, label="p99")
    ax.plot(t["achieved_flows_per_s"], t["p50_ms"], color="#1baf7a", marker="o", lw=1.8, zorder=3, label="p50")
    ax.axhline(100, color="#eb6834", lw=1, ls="--", zorder=2)
    ax.text(t["achieved_flows_per_s"].min(), 103, "100 ms target", color="#eb6834", fontsize=8, va="bottom")
    for r in t.iter_rows(named=True):
        lab = "top speed" if not r["target_rate"] else f"target {r['target_rate']:,}"
        ax.annotate(lab, (r["achieved_flows_per_s"], r["p99_ms"]), textcoords="offset points", xytext=(4, 6),
                    fontsize=7, color=INK2)
    ax.set_xlabel("achieved flows / s", fontsize=8, color=INK2)
    ax.set_ylabel("/score/columns latency (ms)", fontsize=8, color=INK2)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    ax.set_title("C15: replay load test (200 ms micro-batches, monitor on, one API worker)", fontsize=9.5,
                 color=INK, loc="left")
    fig.savefig(paths.FIGURES / "c15_load.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/replay.yaml")
    ap.add_argument("--rates", type=float, nargs="+", default=[1000, 2000, 5000, 10000, 20000, 0])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--port", type=int, default=8020)
    args = ap.parse_args()
    cfg = config.load(args.config)
    rows = []
    for rate in args.rates:
        limit = int(rate * args.seconds) if rate else 150_000
        run = paths.REPO / "data/live/runs" / f"load-{int(rate)}"
        with ApiServer(run / "api", args.port) as srv:
            r = ReplayRunner(cfg, srv.url, run, rate=rate or None, monitor=build_monitor(cfg), labels=False)
            s = r.run(FlowSource("parquet", paths.REPLAY / "demo_flows.parquet", limit))
        row = {"target_rate": int(rate), "flows": s["flows"], "achieved_flows_per_s": round(s["flows_per_s"]),
               "p50_ms": round(s["p50_ms"], 1), "p99_ms": round(s["p99_ms"], 1), "requests": s["requests"],
               "kept_up": bool(not rate or s["flows_per_s"] >= 0.95 * rate), "p99_ok": s["p99_ms"] < 100}
        rows.append(row)
        print(json.dumps(row), flush=True)
    t = pl.DataFrame(rows)
    fm = paths.REPO / "data/live/runs/file-rehearsal/summary.json"
    if fm.exists():
        f = json.loads(fm.read_text())
        print(f"file mode end to end (NFStream one meter + API + monitor): {f['flows_per_s']:.0f} flows/s, "
              f"p99 {f['p99_ms']:.0f} ms")
    t.write_csv(paths.TABLES / "c15_load.csv")
    plot(t)
    ok = t.filter(pl.col("kept_up") & pl.col("p99_ok"))
    print("sustained with p99 < 100 ms:", ok["achieved_flows_per_s"].max() if ok.height else "none")


if __name__ == "__main__":
    main()
