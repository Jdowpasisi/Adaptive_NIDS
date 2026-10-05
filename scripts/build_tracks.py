"""C2: write data/processed/{dataset}/{track}.parquet for every track a dataset belongs to, validate it,
and save family counts to reports/tables/family_counts.csv.

    python scripts/build_tracks.py --dataset all
    python scripts/build_tracks.py --dataset lycos17 --track cic77
"""

import argparse
import logging

import polars as pl

from xnids.features import tracks
from xnids.utils import config, paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--track", default="all")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")

    names = (list(config.load(paths.CONFIGS / "data.yaml")["datasets"]) if args.dataset == "all"
             else args.dataset.split(","))
    counts = []
    for ds in names:
        for tr in tracks.tracks_for(ds) if args.track == "all" else args.track.split(","):
            out = tracks.build_track(ds, tr)
            fc = tracks.family_counts(ds, tr).with_columns(dataset=pl.lit(ds), track=pl.lit(tr))
            counts.append(fc)
            print(f"{ds}/{tr}: {pl.scan_parquet(out).select(pl.len()).collect().item():,} rows, valid")

    table = paths.TABLES / "family_counts.csv"
    new = pl.concat(counts).select("dataset", "track", "family", "len")
    if table.exists():  # keep rows for datasets/tracks not rebuilt this time
        old = pl.read_csv(table).join(new.select("dataset", "track").unique(), on=["dataset", "track"], how="anti")
        new = pl.concat([old, new])
    new.sort("dataset", "track", "len", descending=[False, False, True]).write_csv(table)


if __name__ == "__main__":
    main()
