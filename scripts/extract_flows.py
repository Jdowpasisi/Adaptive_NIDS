"""C13: NFStream flows from the demo PCAPs -> data/interim/<dataset>/<dataset>.parquet (C1 interim format).

    python scripts/extract_flows.py --dataset nfs17        # CIC-IDS2017 Mon/Wed/Fri PCAPs, LycoS17 labelling rules
    python scripts/extract_flows.py --dataset nfs18    # CSE-CIC-IDS2018 Fri-16-02 hosts, official schedule

Per-PCAP flow files go to data/interim/<dataset>/parts/ and are reused when their .meta.json carries the current
nfstream hash. nfs17 labels: the verified C1 port of the LycoS labelling script (epoch-microsecond windows on the
exact attacker/victim pair; the PCAP clock is UTC, as is LycoS's). nfs18 labels: the official schedule in local
time (configs/demo.yaml), shifted by the measured utc_offset (override: --utc-offset), on the listed attacker/victim addresses; a flow captured on
several hosts (same 5-tuple and start time) is kept once.
"""

import argparse
import json
import logging

import polars as pl

from xnids.data.labellers import lycos17, schedule_labels
from xnids.live.extract import nfs_hash, to_parquet
from xnids.utils import config, paths

log = logging.getLogger("extract_flows")


def parts(dataset: str, pcaps: list) -> list:
    out = []
    for p in pcaps:
        dst = paths.INTERIM / dataset / "parts" / (p.name.removesuffix(".pcap") + ".parquet")
        meta = dst.with_suffix(".meta.json")
        if dst.exists() and meta.exists() and json.loads(meta.read_text())["nfstream_hash"] == nfs_hash():
            log.info("have %s", dst.name)
        else:
            log.info("extracting %s (%.1f GB)", p.name, p.stat().st_size / 2**30)
            n = to_parquet(p, dst, {"source_file": p.name})
            log.info("  %d flows", n)
        out.append(dst)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nfs17", "nfs18"], required=True)
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--utc-offset", type=float, default=None, help="nfs18: local schedule time = UTC + offset")
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()
    c = config.load(args.config)
    out = paths.INTERIM / args.dataset / f"{args.dataset}.parquet"
    if args.dataset == "nfs17":
        files = parts("nfs17", [paths.RAW / "cic17_pcap" / p for p in c["cic17"]["pcaps"]])
        if args.extract_only:
            return
        lf = pl.concat([lycos17(pl.scan_parquet(f), f.stem) for f in files], how="vertical_relaxed")
    else:
        pc = [paths.RAW / "cse18_pcap" / (m.rsplit("/", 1)[1].removesuffix(".pcap") + ".pcap")
              for m in c["cse18"]["members"]]
        files = parts("nfs18", pc)
        if args.extract_only:
            return
        if args.utc_offset is None:
            args.utc_offset = c["cse18"]["utc_offset"]          # measured in C13 (configs/demo.yaml)
        lf = pl.concat([pl.scan_parquet(f) for f in files], how="vertical_relaxed")
        key = ["src_addr", "src_port", "dst_addr", "dst_port", "ip_prot", "timestamp"]
        lf = schedule_labels(lf.unique(subset=key, keep="first", maintain_order=True), c["cse18"]["attacks"],
                         c["cse18"]["day"], args.utc_offset)
    # one global flow id over the concatenation (row position), as the C1 interim files have
    lf = lf.drop("flow_id").with_row_index("flow_id").sort("timestamp", maintain_order=True)
    lf.sink_parquet(out, compression="zstd", row_group_size=500_000)
    counts = pl.scan_parquet(out).group_by("source_file", "label").len().sort("source_file", "label").collect()
    with pl.Config(tbl_rows=100):
        print(counts)
    out.with_suffix(".meta.json").write_text(json.dumps(
        {"nfstream_hash": nfs_hash(), "utc_offset": args.utc_offset, "rows": int(counts["len"].sum())}, indent=1))


if __name__ == "__main__":
    main()
