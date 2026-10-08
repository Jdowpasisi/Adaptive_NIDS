"""C15: do file mode and live mode produce the same flows? (Build Guide: "the results should match live mode")

    python scripts/replay_parity.py --make-slice                 # data/replay/demo_slice.pcap (10 min, A -> B1)
    python scripts/replay_parity.py --file-run                   # file mode on the slice -> data/live/runs/parity-file
    sudo bash scripts/replay_live.sh                             # live mode on the slice at x1 (prints the run dir)
    python scripts/replay_parity.py --live data/live/runs/live-<...>

Flows are matched on the 5-tuple, in start order within a 5-tuple (live timestamps are wall-clock). Reports the flow
counts, the match rate, per-feature KS between matched file and live flows, and the alert rate of each run.
Writes reports/tables/c15_parity.csv.
"""

import argparse
import json
import subprocess

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

from xnids.data import schema
from xnids.utils import paths

SLICE = paths.REPLAY / "demo_slice.pcap"
TUP = ["src_addr", "src_port", "dst_addr", "dst_port", "ip_prot"]


def make_slice(minutes: float = 5) -> None:
    segs = json.loads((paths.REPLAY / "demo.json").read_text())["segments"]
    cut = segs[1]["replay_first"]                                   # the A -> B1 switch
    lo, hi = int(cut - minutes * 60), int(cut + minutes * 60)
    subprocess.run(["editcap", "-F", "pcap", "-A", str(lo), "-B", str(hi), str(paths.REPLAY / "demo.pcap"),
                    str(SLICE)], check=True)
    print(f"{SLICE}: {lo} .. {hi} (switch at {cut:.0f})")


def flows(run: str) -> pl.DataFrame:
    f = pl.read_parquet(paths.REPO / run / "flows.parquet")
    parts = pl.col("flow_key").str.split("|")
    f = f.with_columns(tup=parts.list.slice(0, 5).list.join("|"), start=parts.list.get(5).cast(pl.Int64))
    return f.sort("tup", "start").with_columns(k=pl.int_range(pl.len()).over("tup"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-slice", action="store_true")
    ap.add_argument("--file-run", action="store_true")
    ap.add_argument("--file", default="data/live/runs/parity-file")
    ap.add_argument("--live", default=None)
    args = ap.parse_args()
    if args.make_slice:
        make_slice()
    if args.file_run:
        subprocess.run(["python", "scripts/replay.py", "--mode", "file", "--source", str(SLICE), "--run-id",
                        "parity-file", "--port", "8031"], check=True)
    if not args.live:
        return
    F, L = flows(args.file), flows(args.live)
    m = F.join(L, on=["tup", "k"], how="inner", suffix="_live")
    feats = schema.features("nfs")
    ks = {f: ks_2samp(m[f].to_numpy(), m[f + "_live"].to_numpy()).statistic for f in feats}
    exact = {f: float(np.mean(np.isclose(m[f].to_numpy(), m[f + "_live"].to_numpy(), rtol=1e-3, atol=1)))
             for f in feats}
    al = {}
    for name, run in (("file", args.file), ("live", args.live)):
        b = pl.read_csv(paths.REPO / run / "batches.csv")
        al[name] = float(b["alerts_active"].sum() / b["n"].sum())
    worst = sorted(ks, key=ks.get, reverse=True)[:5]
    row = {"file_flows": F.height, "live_flows": L.height, "matched": m.height,
           "match_rate_file": m.height / F.height, "match_rate_live": m.height / L.height,
           "ks_max": max(ks.values()), "ks_worst": ";".join(f"{f}={ks[f]:.3f}" for f in worst),
           "share_equal_min": min(exact.values()), "alert_rate_file": al["file"], "alert_rate_live": al["live"]}
    pl.DataFrame([row]).write_csv(paths.TABLES / "c15_parity.csv")
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
