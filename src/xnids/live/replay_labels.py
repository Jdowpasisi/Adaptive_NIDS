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


def flow_key_expr() -> pl.Expr:
    """The API's flow_key string for a flow table with the KEY columns: 'src|sport|dst|dport|proto|start_us'."""
    return pl.concat_str([pl.col(c).cast(pl.String) for c in KEY], separator="|").alias("flow_key")


class ReplayOracle:
    """Labels for flow_keys from the demo replay's ground truth: stands in for an analyst labelling the flows that
    few-shot adaptation selects (C14). Raises on a key it does not know."""

    def __init__(self, labels: pl.DataFrame | None = None) -> None:
        m = label_map(labels)
        self.y = dict(zip(m.select(flow_key_expr())["flow_key"].to_list(), m["y"].to_list(), strict=True))
        # live mode: capture timestamps are wall-clock, so fall back to the 5-tuple where it carries ONE label
        # (96.5% of the demo's flows)
        t = (m.with_columns(tup=pl.concat_str([pl.col(c).cast(pl.String) for c in KEY[:-1]], separator="|"))
             .group_by("tup").agg(pl.col("y").first(), pl.col("label").n_unique().alias("_n"))
             .filter(pl.col("_n") == 1))
        self.y_tuple = dict(zip(t["tup"].to_list(), t["y"].to_list(), strict=True))

    def get(self, key: str, default=None):
        """Label of a flow_key: exact key, else its 5-tuple when unambiguous, else default."""
        y = self.y.get(key)
        return y if y is not None else self.y_tuple.get(key.rsplit("|", 1)[0], default)

    def __call__(self, keys: list[str]):
        import numpy as np

        missing = [k for k in keys if k not in self.y]
        if missing:
            raise KeyError(f"{len(missing)} flows have no replay label, e.g. {missing[0]}")
        return np.array([self.y[k] for k in keys], dtype=int)
