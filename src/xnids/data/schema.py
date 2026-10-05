"""The processed-flow contract (Build Guide §4) and its validator.

A processed file `data/processed/{dataset}/{track}.parquet` holds, in this order:
  row_id (UInt32, stable per dataset across tracks) | [ts (Int64 epoch µs) if the dataset has one]
  | y (Int8 0/1) | family (String, one of 8) | label (raw string, for per-attack analysis)
  | the track's feature columns (Float32, in the order the track config lists them)
"""

import re
from pathlib import Path

import polars as pl

from xnids.data.labels import FAMILIES
from xnids.utils import config, paths

META = ["row_id", "ts", "y", "family", "label"]

# Rule 9: identifiers are never features. Destination port is the single, flagged exception.
IDENTIFIER_RE = re.compile(
    r"(^|_)(flow_id|src_addr|dst_addr|source_ip|destination_ip|ipv4_(src|dst)_addr|src_port|source_port|"
    r"l4_src_port|timestamp|ts|dns_query_id)$"
)
FLAGGED_OK = {"dst_port", "destination_port", "l4_dst_port"}


class SchemaError(ValueError):
    pass


def track_cfg(track: str) -> dict:
    return config.load(paths.CONFIGS / "tracks" / f"{track}.yaml")


def features(track: str) -> list[str]:
    cfg = track_cfg(track)
    if "features" in cfg:
        return list(cfg["features"])
    return [f["name"] for f in config.load(paths.CONFIGS / "tracks" / cfg["core_map"])["features"]
            if not f.get("drop")]


def identifier_features(cols: list[str]) -> list[str]:
    return [c for c in cols if IDENTIFIER_RE.search(c) and c not in FLAGGED_OK]


def validate(df: pl.DataFrame | pl.LazyFrame | Path | str, track: str) -> None:
    """Raise SchemaError unless the frame matches the contract for `track`."""
    if isinstance(df, Path | str):
        df = pl.scan_parquet(df)
    lf = df.lazy()
    schema = lf.collect_schema()
    feats = features(track)
    meta = [c for c in META if c in schema]
    errors = []
    if leaks := identifier_features(feats):
        errors.append(f"identifier columns listed as features: {leaks}")
    for c in ("row_id", "y", "family", "label"):
        if c not in schema:
            errors.append(f"missing meta column {c}")
    if list(schema) != meta + feats:
        missing, extra = set(feats) - set(schema), set(schema) - set(feats) - set(META)
        errors.append(f"columns differ from track {track}: missing={sorted(missing)} extra={sorted(extra)}"
                      if missing or extra else "feature columns out of order")
    errors += [f"{c} is {schema[c]}, expected Float32" for c in feats if c in schema and schema[c] != pl.Float32]
    expected = {"row_id": pl.UInt32, "ts": pl.Int64, "y": pl.Int8, "family": pl.String, "label": pl.String}
    errors += [f"{c} is {schema[c]}, expected {t}" for c, t in expected.items() if c in schema and schema[c] != t]
    if not errors:
        bad = lf.select(
            (~pl.col("y").is_in([0, 1])).sum().alias("y"),
            (~pl.col("family").is_in(FAMILIES)).sum().alias("family"),
            ((pl.col("family") == "Benign").cast(pl.Int8) == pl.col("y")).sum().alias("y_vs_family"),
            (pl.col("row_id").n_unique() - pl.len()).alias("row_id_dupes"),
        ).collect(engine="streaming").row(0, named=True)
        errors += [f"{k}: {v} bad rows" for k, v in bad.items() if v]
    if errors:
        raise SchemaError(f"{track}: " + "; ".join(errors))
