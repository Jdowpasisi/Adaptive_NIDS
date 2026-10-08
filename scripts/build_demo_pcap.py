"""C13: build data/replay/demo.pcap (segment A = held-out CIC-2017 minutes, segment B = CSE-CIC-IDS2018 traffic) and
data/replay/demo_labels.parquet (one row per NFStream flow of demo.pcap, keyed by 5-tuple + start time).

    python scripts/build_demo_pcap.py

Slices are cut with editcap -A/-B (UTC), shifted with editcap -t so each starts `gap_s` after the previous one ends,
and concatenated with mergecap -a. CSE-2018 slices merge the host captures in time order (mergecap) and drop
packets captured twice (editcap -D). Labels: the demo flows are extracted with the frozen extractor (the same flows
C15 will stream), shifted back to their original clock per segment, and labelled with the LycoS17 rules (A) or the
2018 schedule (B, local = UTC + utc_offset).
"""

import datetime as dt
import json
import logging
import subprocess
from pathlib import Path

import polars as pl

from xnids.data import labels
from xnids.data.labellers import lycos17, schedule_labels
from xnids.live.extract import nfs_hash, to_parquet
from xnids.utils import config, paths

log = logging.getLogger("build_demo_pcap")
KEY = ["src_addr", "src_port", "dst_addr", "dst_port", "ip_prot", "timestamp"]


def utc_s(t: str) -> float:
    return dt.datetime.fromisoformat(t).replace(tzinfo=dt.UTC).timestamp()


def run(*cmd) -> None:
    subprocess.run([str(c) for c in cmd], check=True)


def first_last(pcap: Path) -> tuple[float, float]:
    out = subprocess.run(["capinfos", "-aeS", "-T", "-r", str(pcap)], check=True, capture_output=True, text=True)
    vals = out.stdout.strip().split("\t")
    return float(vals[-2]), float(vals[-1])


def cut(src: list[Path], window: list[str], out: Path, dedup: int) -> None:
    lo, hi = (int(utc_s(t)) for t in window)
    pieces = []
    for i, p in enumerate(src):
        piece = out.with_suffix(f".{i}.pcap")
        run("editcap", "-F", "pcap", "-A", lo, "-B", hi, p, piece)
        pieces.append(piece)
    if len(pieces) == 1:
        pieces[0].replace(out)
        return
    merged = out.with_suffix(".merged.pcap")
    run("mergecap", "-F", "pcap", "-w", merged, *pieces)
    run("editcap", "-F", "pcap", "-D", dedup, merged, out)
    for p in (*pieces, merged):
        p.unlink()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    c = config.load("configs/demo.yaml")
    r = c["replay"]
    out_dir = paths.REPLAY
    out_dir.mkdir(parents=True, exist_ok=True)
    cse = [paths.RAW / "cse18_pcap" / (m.rsplit("/", 1)[1].removesuffix(".pcap") + ".pcap")
           for m in c["cse18"]["members"]]
    shifted, segs, t_next = [], [], None
    for s in r["segments"]:
        src = [paths.RAW / "cic17_pcap" / s["pcap"]] if s["source"] == "cic17" else cse
        raw = out_dir / f"seg{s['name']}.raw.pcap"
        log.info("segment %s: cutting %s", s["name"], s["window"])
        cut(src, s["window"], raw, r["dedup_window"])
        first, last = first_last(raw)
        shift = 0.0 if t_next is None else round(t_next - first, 6)
        seg = out_dir / f"seg{s['name']}.pcap"
        if shift:
            run("editcap", "-F", "pcap", "-t", f"{shift:.6f}", raw, seg)
            raw.unlink()
        else:
            raw.replace(seg)
        t_next = last + shift + r["gap_s"]
        segs.append({**s, "shift_s": shift, "orig_first": first, "orig_last": last,
                     "replay_first": first + shift, "replay_last": last + shift})
        shifted.append(seg)
        log.info("  %s: %.0f s, shift %+.0f s", s["name"], last - first, shift)
    demo = out_dir / "demo.pcap"
    run("mergecap", "-F", "pcap", "-a", "-w", demo, *shifted)
    for p in shifted:
        p.unlink()

    log.info("extracting the demo flows with the frozen NFStream settings")
    flows_path = out_dir / "demo_flows.parquet"
    to_parquet(demo, flows_path, {"source_file": "demo.pcap"})
    flows = pl.read_parquet(flows_path)
    parts = []
    # segment boundaries at the middle of each gap (flow start times are truncated to whole milliseconds)
    cuts = [-(2**62)] + [int((a["replay_last"] + b["replay_first"]) / 2 * 1e6) for a, b in zip(segs, segs[1:], strict=False)] \
        + [2**62]
    for s, lo, hi in zip(segs, cuts, cuts[1:], strict=False):
        f = flows.filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") < hi))
        orig = f.with_columns((pl.col("timestamp") - int(round(s["shift_s"] * 1e6))).alias("timestamp"))
        if s["source"] == "cic17":
            lab = lycos17(orig.lazy(), s["pcap"]).collect()
        else:
            lab = schedule_labels(orig.lazy(), c["cse18"]["attacks"], c["cse18"]["day"],
                                  c["cse18"]["utc_offset"]).collect()
        parts.append(f.select(KEY + ["end_timestamp"]).with_columns(
            label=lab["label"], segment=pl.lit(s["name"]), orig_timestamp=lab["timestamp"]))
    lab = pl.concat(parts)
    assert lab.height == flows.height, f"{flows.height - lab.height} demo flows fall outside every segment"
    fam = labels.to_family(pl.col("label"))
    lab = lab.with_columns(family=fam, y=labels.to_binary(fam))
    lab.write_parquet(out_dir / "demo_labels.parquet")
    counts = lab.group_by("segment", "family", "label").len().sort("segment", "family")
    with pl.Config(tbl_rows=50):
        print(counts)
    (out_dir / "demo.json").write_text(json.dumps({
        "nfstream_hash": nfs_hash(), "segments": segs, "flows": lab.height,
        "pcap_bytes": demo.stat().st_size, "counts": counts.rows(named=True)}, indent=1, default=str))


if __name__ == "__main__":
    main()
