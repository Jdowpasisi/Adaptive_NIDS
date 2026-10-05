"""C2: build data/processed/{dataset}/{track}.parquet from the C1 interim Parquet.

Native tracks (cic77, nf43, cic_orig) select and cast the listed columns. The core track evaluates the
expressions in core_map.yaml. Every track gets the same row_id for a given dataset (the row position in the
interim file), so one split file per dataset serves all of its tracks.
"""

import logging
from pathlib import Path

import polars as pl

from xnids.data import labels, schema
from xnids.data.ingest import clean_name
from xnids.features import core_map
from xnids.utils import config, paths

log = logging.getLogger(__name__)


def dataset_spec(dataset: str) -> dict:
    return config.load(paths.CONFIGS / "data.yaml")["datasets"][dataset]


def interim_path(dataset: str) -> Path:
    return paths.INTERIM / dataset / f"{dataset}.parquet"


def processed_path(dataset: str, track: str) -> Path:
    return paths.PROCESSED / dataset / f"{track}.parquet"


def track_frame(dataset: str, track: str) -> pl.LazyFrame:
    tcfg = schema.track_cfg(track)
    if dataset not in tcfg["datasets"]:
        raise ValueError(f"track {track} is not defined for dataset {dataset} (datasets: {tcfg['datasets']})")
    spec = dataset_spec(dataset)
    label_col = clean_name(spec["label_col"])
    lf = pl.scan_parquet(interim_path(dataset)).with_row_index("row_id")

    raw_labels = lf.select(pl.col(label_col).unique()).collect()[label_col].cast(pl.String).to_list()
    labels.check_mapped(raw_labels)
    family = labels.to_family(pl.col(label_col))

    meta = [pl.col("row_id")]
    if spec.get("ts_col"):
        meta.append(pl.col(spec["ts_col"]).cast(pl.Int64).alias("ts"))
    meta += [labels.to_binary(family).alias("y"), family.alias("family"),
             pl.col(label_col).cast(pl.String).alias("label")]

    if "core_map" in tcfg:
        feats = core_map.expressions(dataset)
    else:
        feats = [pl.col(c).cast(pl.Float32) for c in tcfg["features"]]
    return lf.select(*meta, *feats)


def build_track(dataset: str, track: str) -> Path:
    out = processed_path(dataset, track)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("%s/%s -> %s", dataset, track, out)
    track_frame(dataset, track).sink_parquet(out, compression="zstd", row_group_size=500_000)
    schema.validate(out, track)
    return out


def tracks_for(dataset: str) -> list[str]:
    return sorted(p.stem for p in (paths.CONFIGS / "tracks").glob("*.yaml")
                  if p.stem != "core_map" and dataset in schema.track_cfg(p.stem)["datasets"])


def family_counts(dataset: str, track: str) -> pl.DataFrame:
    return (pl.scan_parquet(processed_path(dataset, track)).group_by("family").len()
            .sort("len", descending=True).collect())
