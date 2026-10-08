"""C13 / C15: the ONE flow extractor. NFStream over a PCAP file or an interface with the frozen configs/nfstream.yaml.

`extract(source)` yields pandas chunks with the columns in COLUMNS (identifiers renamed to the LycoS17 names the
C1 labeller expects: src_addr, dst_addr, src_port, dst_port, ip_prot, timestamp in epoch MICROseconds) plus the
NFStream statistical features. `to_parquet(source, out)` streams those chunks into one Parquet file.
"""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from nfstream import NFPlugin

from xnids.utils import config, paths

CFG_PATH = paths.CONFIGS / "nfstream.yaml"

ID_RENAME = {"src_ip": "src_addr", "dst_ip": "dst_addr", "protocol": "ip_prot"}
IDENTIFIERS = ["flow_id", "src_addr", "src_port", "dst_addr", "dst_port", "ip_prot", "timestamp", "end_timestamp"]
_DROP_PREFIX = ("udps.", "src_mac", "dst_mac", "src_oui", "dst_oui", "application_", "requested_server_name",
                "client_fingerprint", "server_fingerprint", "user_agent", "content_type", "splt_")
_DROP = {"id", "expiration_id", "ip_version", "vlan_id", "tunnel_id"}


class TCPTermination(NFPlugin):
    """End a TCP flow on RST, or on the first non-FIN packet after BOTH sides sent a FIN (the closing ACK), so one
    TCP connection = one flow as in CICFlowMeter / LycoSTand. NFStream alone only expires on idle/active timeouts,
    which merges successive connections that reuse a source port (C13: ~12 DoS-Hulk connections per flow) and
    delays a flow's report by the idle timeout."""

    def on_init(self, packet, flow):
        flow.udps.fin_dirs = (1 << packet.direction) if packet.fin else 0

    def on_update(self, packet, flow):
        if packet.rst or (flow.udps.fin_dirs == 3 and not packet.fin):
            flow.expiration_id = -1           # custom expiration: this packet is the flow's last
        elif packet.fin:
            flow.udps.fin_dirs |= 1 << packet.direction


def nfs_cfg() -> dict:
    return config.load(CFG_PATH)


def nfs_hash() -> str:
    """sha1 of the canonical JSON of configs/nfstream.yaml (same scheme as utils.config.cfg_hash)."""
    return config.cfg_hash(nfs_cfg())


def check_frozen() -> str:
    lock = CFG_PATH.with_suffix(".lock")
    h = nfs_hash()
    if lock.exists() and lock.read_text().strip() != h:
        raise RuntimeError(f"configs/nfstream.yaml changed (hash {h} != locked {lock.read_text().strip()}): "
                           "re-extract every flow file and retrain the demo model, then update the lock")
    return h


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in df.columns if c in _DROP or c.startswith(_DROP_PREFIX)])
    df = df.rename(columns=ID_RENAME)
    df.insert(0, "timestamp", df.pop("bidirectional_first_seen_ms").astype("int64") * 1000)
    df.insert(1, "end_timestamp", df.pop("bidirectional_last_seen_ms").astype("int64") * 1000)
    # per-direction first/last seen are absolute times too: keep only as durations (already present)
    df = df.drop(columns=[c for c in df.columns if c.endswith(("_first_seen_ms", "_last_seen_ms"))])
    return df


def extract(source: str | Path, chunk_rows: int | None = None) -> Iterator[pd.DataFrame]:
    """Flows of `source` (pcap path or interface name) in chunks, columns tidied (see module doc)."""
    from nfstream import NFStreamer

    check_frozen()
    c = nfs_cfg()
    chunk_rows = chunk_rows or c["chunk_rows"]
    udps = [TCPTermination()] if c.get("tcp_termination") else None
    streamer = NFStreamer(source=str(source), udps=udps, **c["nfstreamer"])
    cols, buf, n = None, [], 0
    for f in streamer:
        if cols is None:
            cols = f.keys()
        buf.append(f.values())
        if len(buf) >= chunk_rows:
            yield _tidy(pd.DataFrame(buf, columns=cols))
            n += len(buf)
            buf = []
    if buf:
        yield _tidy(pd.DataFrame(buf, columns=cols))


def _sha1_head(p: Path, n: int = 1 << 20) -> str:
    """sha1 of the first MiB only (a cheap fingerprint of a multi-GB capture)."""
    with p.open("rb") as fh:
        return hashlib.sha1(fh.read(n)).hexdigest()


def stream_flows(source: str | Path, ordered: bool = True) -> Iterator[tuple[list[str], list]]:
    """One (column names, values) per flow, as NFStream emits it (C15: the replay runner micro-batches these).
    Same frozen settings as extract(); turn a list of values into the tidy frame with tidy_rows().

    ordered=True runs ONE meter. With several, each meter handles its share of the packets at its own pace and their
    flows interleave from far-apart points of the capture (C15: segment-B flows filled 16-50% of segment-A windows).
    The meter count changes parallelism only, not the content of a flow, so the frozen config hash still applies."""
    from nfstream import NFStreamer

    check_frozen()
    c = nfs_cfg()
    udps = [TCPTermination()] if c.get("tcp_termination") else None
    kw = c["nfstreamer"] | ({"n_meters": 1} if ordered else {})
    cols = None
    for f in NFStreamer(source=str(source), udps=udps, **kw):
        if cols is None:
            cols = f.keys()
        yield cols, f.values()


def tidy_rows(cols: list[str], rows: list) -> pd.DataFrame:
    return _tidy(pd.DataFrame(rows, columns=cols))


def to_parquet(source: str | Path, out: Path, extra: dict | None = None) -> int:
    """Stream extract(source) into `out` (zstd Parquet); `extra` constant columns are added (e.g. source_file).
    Writes <out>.meta.json with the nfstream hash and the row count. Returns the number of flows."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    writer, n = None, 0
    for df in extract(source):
        for k, v in (extra or {}).items():
            df[k] = v
        df.insert(0, "flow_id", range(n, n + len(df)))
        t = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
        writer.write_table(t.cast(writer.schema))
        n += len(df)
    if writer:
        writer.close()
    tmp.replace(out)
    meta = {"source": str(source), "flows": n, "nfstream_hash": nfs_hash(),
            "source_sha1_head": _sha1_head(Path(source)) if Path(source).is_file() else None}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    return n
