"""C3: dedup + frozen train/val/test split per dataset.

    python scripts/split.py --dataset all
    python scripts/split.py --dataset lycos17 --check      # verify existing files against the lock, write nothing

Writes data/splits/{dataset}.parquet, reports/tables/clean_{dataset}.csv (rows removed at each step) and
reports/tables/split_family_counts.csv, and records every split file's hash + sha256 in
reports/tables/splits_lock.csv (committed). An existing split file is never overwritten when its split_hash
differs from the config's: bump `version` in configs/split.yaml to change a frozen split.
"""

import argparse
import logging
import sys

import polars as pl

from xnids.data import split
from xnids.data.ingest import sha256
from xnids.utils import config, paths

LOCK = paths.TABLES / "splits_lock.csv"


def lock_table() -> pl.DataFrame:
    return pl.read_csv(LOCK) if LOCK.exists() else pl.DataFrame(
        schema={"dataset": pl.String, "version": pl.Int64, "split_hash": pl.String, "sha256": pl.String,
                "n_train": pl.Int64, "n_val": pl.Int64, "n_test": pl.Int64})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/split.yaml")
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--check", action="store_true", help="only verify existing split files against the lock")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = config.load(args.config)
    names = list(cfg["datasets"]) if args.dataset == "all" else args.dataset.split(",")
    paths.SPLITS.mkdir(parents=True, exist_ok=True)
    lock = lock_table()
    status = 0
    for ds in names:
        out_path = paths.SPLITS / f"{ds}.parquet"
        want = split.split_hash(ds, cfg)
        locked = lock.filter(pl.col("dataset") == ds)
        if out_path.exists():
            have = pl.scan_parquet(out_path).select(pl.col("split_hash").first()).collect().item()
            sha_ok = locked.height == 1 and locked["sha256"][0] == sha256(out_path)
            if have == want and sha_ok:
                print(f"{ds}: frozen, matches config and lock ({want})")
                continue
            bumped = locked.height == 1 and cfg["version"] > locked["version"][0]
            if args.check or (have != want and not bumped):
                print(f"{ds}: MISMATCH file hash={have} config hash={want} lock sha256 ok={sha_ok}. "
                      f"Bump `version` in {args.config} to re-split on purpose.")
                status = 1
                continue
            logging.warning("%s: version bumped to %s, re-splitting; re-run everything downstream", ds, cfg["version"])
        elif args.check:
            print(f"{ds}: no split file")
            status = 1
            continue

        df, report = split.build_split(ds, cfg)
        df.write_parquet(out_path, compression="zstd")
        pl.DataFrame([report]).write_csv(paths.TABLES / f"clean_{ds}.csv")
        lock = pl.concat([lock.filter(pl.col("dataset") != ds), pl.DataFrame([{
            "dataset": ds, "version": cfg["version"], "split_hash": want, "sha256": sha256(out_path),
            **{f"n_{s}": report[f"n_{s}"] for s in split.SPLITS}}], schema=lock.schema)])
        lock.sort("dataset").write_csv(LOCK)
        print(f"{ds}: {report}")

    fams = []
    for ds in cfg["datasets"]:
        p = paths.SPLITS / f"{ds}.parquet"
        if p.exists():
            fams.append(pl.read_parquet(p).group_by("split", "family").len().with_columns(dataset=pl.lit(ds)))
    if fams:
        (pl.concat(fams).pivot(on="split", index=["dataset", "family"], values="len").fill_null(0)
         .select("dataset", "family", *split.SPLITS).sort("dataset", "train", descending=[False, True])
         .write_csv(paths.TABLES / "split_family_counts.csv"))
    return status


if __name__ == "__main__":
    sys.exit(main())
