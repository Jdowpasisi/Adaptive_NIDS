"""Track C ('core'): a small set of semantically equivalent features across the CIC/LycoS and NetFlow families.

configs/tracks/core_map.yaml lists each core feature once with one polars-SQL expression per family, written
against the cleaned interim column names, e.g.

    - name: duration_s
      cic: flow_duration / 1000000.0
      nf:  flow_duration_milliseconds / 1000.0
      unit: s
      caveat: ...

A feature with `drop: <reason>` stays documented but is excluded from the track (see the bridge check).
"""

from pathlib import Path

import polars as pl

from xnids.utils import config, paths

CORE_MAP = paths.CONFIGS / "tracks" / "core_map.yaml"


def load(path: Path = CORE_MAP) -> dict:
    return config.load(path)


def family_of(dataset: str, cmap: dict | None = None) -> str:
    cmap = cmap or load()
    try:
        return cmap["schema_family"][dataset]
    except KeyError as e:
        raise KeyError(f"core_map.yaml has no schema_family for dataset {dataset!r}") from e


def expressions(dataset: str, include_dropped: bool = False, cmap: dict | None = None) -> list[pl.Expr]:
    """Float32 polars expressions producing the core features for `dataset`, in map order."""
    cmap = cmap or load()
    fam = family_of(dataset, cmap)
    return [
        pl.sql_expr(str(f[fam])).cast(pl.Float32).alias(f["name"])
        for f in cmap["features"]
        if include_dropped or not f.get("drop")
    ]
