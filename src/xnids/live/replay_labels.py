"""C13 / C15: attach ground-truth labels to flows of the demo replay.

Labels are keyed by the 5-tuple + start time (epoch us, whole ms). A key is not unique (a trailing packet after a
TCP close can open a second flow of the same 5-tuple in the same millisecond: 97 of 272,768 demo flows), but every
key carries ONE label, which attach() asserts, so the lookup is exact."""

import polars as pl

from xnids.utils import paths

KEY = ["src_addr", "src_port", "dst_addr", "dst_port", "ip_prot", "timestamp"]
COLS = ["segment", "family", "label", "y"]


def label_map(labels: pl.DataFrame | None = None) -> pl.DataFrame:
    labels = pl.read_parquet(paths.REPLAY / "demo_labels.parquet") if labels is None else labels
    m = labels.group_by(KEY).agg(*[pl.col(c).first() for c in COLS], pl.col("label").n_unique().alias("_n"))
    if (bad := m.filter(pl.col("_n") > 1)).height:
        raise ValueError(f"{bad.height} flow keys carry more than one label")
    return m.drop("_n")


def attach(flows: pl.DataFrame, labels: pl.DataFrame | None = None) -> pl.DataFrame:
    """flows + segment/family/label/y (null where a flow is not in the label table)."""
    return flows.join(label_map(labels), on=KEY, how="left", validate="m:1")
