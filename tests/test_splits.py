"""C3 tests. Unit tests run on toy data; the 'data' tests check the real frozen split files when present."""

import numpy as np
import polars as pl
import pytest

from xnids.data import clean, schema, split
from xnids.utils import config, paths

FR = {"train": 0.6, "val": 0.2, "test": 0.2}


# ------------------------------------------------------------------ unit tests

def test_splitmix_deterministic_and_uniform():
    ids = np.arange(200_000)
    u1, u2 = split.uniform01(ids, 0), split.uniform01(ids, 0)
    assert np.array_equal(u1, u2) and not np.array_equal(u1, split.uniform01(ids, 1))
    assert abs(u1.mean() - 0.5) < 0.005 and u1.min() >= 0 and u1.max() < 1


def test_dedup_exact_and_conflict():
    lf = pl.LazyFrame({
        "row_id": [0, 1, 2, 3, 4, 5],
        "a": [1.0, 1.0, 2.0, 2.0, 3.0, float("nan")],
        "b": [5.0, 5.0, 6.0, 6.0, 7.0, 8.0],
        "label": ["benign", "benign", "benign", "dos", "dos", "dos"],
    })
    st = dict(clean.dedup_status(lf, ["a", "b"]).collect().sort("row_id").iter_rows())
    assert st == {0: "keep", 1: "duplicate", 2: "conflict", 3: "conflict", 4: "keep", 5: "keep"}


def test_stratified_fractions_and_coverage():
    n = 100_000
    df = pl.DataFrame({"row_id": np.arange(n), "family": np.where(np.arange(n) % 1000 == 0, "Bot/Backdoor", "Benign")})
    out = split.stratified_split(df, FR, seed=0)
    share = out.group_by("split").len().with_columns(pl.col("len") / n)
    assert dict(share.iter_rows())["train"] == pytest.approx(0.6, abs=0.01)
    assert set(out.filter(pl.col("family") == "Bot/Backdoor")["split"]) == set(split.SPLITS)


def test_coverage_fix_moves_rows_into_empty_split():
    df = pl.DataFrame({"row_id": np.arange(40), "family": ["DoS"] * 40, "split": ["train"] * 40})
    out = split.ensure_family_coverage(df, seed=0)
    assert set(out["split"]) == set(split.SPLITS) and out.height == 40


def test_time_block_keeps_blocks_whole_and_covers_families():
    rng = np.random.default_rng(0)
    block_us = 300_000_000
    rows = []
    for b in range(60):                     # 60 blocks of benign, attacks in blocks 10-19 and 40-44
        fam = "DoS" if 10 <= b < 20 else "DDoS" if 40 <= b < 45 else "Benign"
        for _ in range(50):
            rows.append((b * block_us + int(rng.integers(block_us)), fam if rng.random() < 0.5 else "Benign"))
    df = pl.DataFrame(rows, schema=["ts", "family"], orient="row").with_row_index("row_id")
    out, _ = split.time_block_split(df, FR, seed=0, block_us=block_us)
    merged = out.join(df.select("row_id", "ts"), on="row_id").with_columns(block=pl.col("ts") // block_us)
    assert merged.group_by("block").agg(pl.col("split").n_unique()).get_column("split").max() == 1
    for fam in ("DoS", "DDoS", "Benign"):
        assert set(merged.filter(pl.col("family") == fam)["split"]) == set(split.SPLITS), fam


def test_dev_subsample_floor_keeps_rare():
    n = 50_000
    df = pl.DataFrame({"row_id": np.arange(n), "family": np.where(np.arange(n) < 300, "WebAttack", "Benign")})
    sub = split.dev_subsample(df.lazy(), max_rows=5_000, min_per_family=1_000, seed=0).collect()
    assert sub.filter(pl.col("family") == "WebAttack").height == 300
    assert 4_000 < sub.height < 6_500


def test_constant_cols_and_imputer():
    X = pl.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, float("inf"), 3.0], "c": [float("nan"), 2.0, 4.0]})
    assert clean.constant_cols(X, ["a", "b", "c"]) == ["a"]
    imp = clean.MedianImputer().fit(X)
    out = imp.transform(X)
    assert out["b"].to_list() == [1.0, 2.0, 3.0] and out["c"].to_list() == [3.0, 2.0, 4.0]
    assert clean.MedianImputer.from_dict(imp.to_dict()).transform(X).equals(out)


# ------------------------------------------------------------------ real split files (Build Guide C3 tests)

CFG = config.load(paths.CONFIGS / "split.yaml")
DATASETS = [d for d in CFG["datasets"] if (paths.SPLITS / f"{d}.parquet").exists()]
needs_data = pytest.mark.skipif(not DATASETS, reason="no split files yet (run scripts/split.py)")


@needs_data
@pytest.mark.parametrize("ds", DATASETS)
def test_no_row_in_two_splits(ds):
    s = pl.read_parquet(paths.SPLITS / f"{ds}.parquet")
    assert s["row_id"].n_unique() == s.height
    assert set(s["split"].unique()) == set(split.SPLITS)


@needs_data
@pytest.mark.parametrize("ds", DATASETS)
def test_split_hash_matches_config(ds):
    s = pl.scan_parquet(paths.SPLITS / f"{ds}.parquet").select(pl.col("split_hash").unique()).collect()
    assert s.to_series().to_list() == [split.split_hash(ds, CFG)]


@needs_data
@pytest.mark.parametrize("ds", DATASETS)
def test_every_family_with_30_rows_in_every_split(ds):
    s = pl.read_parquet(paths.SPLITS / f"{ds}.parquet")
    tot = s.group_by("family").len().filter(pl.col("len") >= 30)["family"]
    for fam in tot:
        assert set(s.filter(pl.col("family") == fam)["split"]) == set(split.SPLITS), f"{ds}: {fam}"


@pytest.mark.parametrize("track", ["cic77", "nf43", "core", "cic_orig"])
def test_no_identifier_features(track):
    assert schema.identifier_features(schema.features(track)) == []
