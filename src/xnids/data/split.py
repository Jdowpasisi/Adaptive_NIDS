"""C3 splits: data/splits/{dataset}.parquet = (row_id, split, split_hash), split in {train, val, test}.

Rows removed by dedup are not in the file; their counts go to reports/tables/clean_{dataset}.csv.

Schemes:
  stratified  - per family, each row goes to a split by a seeded hash of its row_id (NF-v2, LycoS18 and
                cic17_orig, which have no usable timestamp).
  time_block  - LycoS17: cut time into fixed blocks and assign whole blocks, so neighbouring flows of the same
                burst never straddle train and test. Blocks are allocated greedily, rarest family first, to
                the split furthest below its target share of that family; then by rows.

All randomness comes from splitmix64 on row_id / block id with the config seed, implemented in numpy, so the
assignment does not depend on the polars version or the machine.
"""

import numpy as np
import polars as pl

from xnids.data import clean, schema
from xnids.features import tracks
from xnids.utils.config import cfg_hash

SPLITS = ("train", "val", "test")
_U64 = np.uint64


def splitmix64(x: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic 64-bit mix of integer ids (wrapping uint64 arithmetic)."""
    with np.errstate(over="ignore"):
        z = x.astype(_U64) + _U64(seed) * _U64(0x9E3779B97F4A7C15) + _U64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> _U64(27))) * _U64(0x94D049BB133111EB)
    return z ^ (z >> _U64(31))


def uniform01(ids: np.ndarray, seed: int) -> np.ndarray:
    return (splitmix64(ids, seed) >> _U64(11)).astype(np.float64) / float(1 << 53)


def _assign(u: np.ndarray, fractions: dict[str, float]) -> np.ndarray:
    edges = np.cumsum([fractions[s] for s in SPLITS])
    return np.array(SPLITS)[np.searchsorted(edges / edges[-1], u, side="right").clip(0, 2)]


def stratified_split(df: pl.DataFrame, fractions: dict[str, float], seed: int) -> pl.DataFrame:
    """df: row_id, family. Hash-uniform per row gives ~exact fractions within every family (stratified by
    construction, since the draw is independent of family). Families too small to fill all splits are
    topped up below by `ensure_family_coverage`."""
    u = uniform01(df["row_id"].to_numpy(), seed)
    out = df.select("row_id", "family").with_columns(split=pl.Series(_assign(u, fractions)))
    return ensure_family_coverage(out, seed)


def ensure_family_coverage(df: pl.DataFrame, seed: int, min_rows: int = 30) -> pl.DataFrame:
    """For a family with >= min_rows rows that is missing from a split, move its lowest-hash rows there from
    the split holding the most of that family (only possible in the row-wise scheme)."""
    fam_tot = df.group_by("family").len()
    for fam in fam_tot.filter(pl.col("len") >= min_rows)["family"]:
        for s in SPLITS:
            have = dict(df.filter(pl.col("family") == fam).group_by("split").len().iter_rows())
            if have.get(s, 0) > 0:
                continue
            donor = max(have, key=have.get)
            rows = df.filter((pl.col("family") == fam) & (pl.col("split") == donor))["row_id"].to_numpy()
            take = rows[np.argsort(splitmix64(rows, seed + 1))[: max(1, len(rows) // 10)]]
            df = df.with_columns(split=pl.when(pl.col("row_id").is_in(take)).then(pl.lit(s)).otherwise("split"))
    return df


def time_block_split(df: pl.DataFrame, fractions: dict[str, float], seed: int,
                     block_us: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """df: row_id, ts (epoch µs), family. Returns (row_id, family, split) and the block table."""
    df = df.with_columns(block=(pl.col("ts") // block_us))
    blocks = df.group_by("block", "family").len().pivot(on="family", index="block", values="len").fill_null(0)
    fams = [c for c in blocks.columns if c != "block"]
    fam_tot = blocks.select(fams).sum().row(0, named=True)
    order = sorted(fams, key=lambda f: fam_tot[f])                      # rarest family first
    target = np.array([fractions[s] for s in SPLITS]) / sum(fractions.values())
    have = {f: np.zeros(3) for f in fams}
    rows_have = np.zeros(3)
    assigned: dict[int, int] = {}
    block_ids = blocks["block"].to_numpy()
    tiebreak = dict(zip(block_ids, uniform01(block_ids, seed), strict=True))
    bt = {b: r for b, r in zip(block_ids, blocks.select(fams).iter_rows(named=True), strict=True)}

    def place(b: int, fam: str | None) -> None:
        vec = np.array([bt[b][f] for f in fams])
        if fam is not None:
            deficit = target * fam_tot[fam] - have[fam]
        else:
            deficit = target * sum(fam_tot.values()) - rows_have
        s = int(np.argmax(deficit + 1e-9 * np.arange(3)[::-1]))       # ties -> train, then val
        assigned[b] = s
        for f, v in zip(fams, vec, strict=True):
            have[f][s] += v
        rows_have[s] += vec.sum()

    for fam in order:
        if fam == "Benign":
            continue
        cand = sorted((b for b in block_ids if bt[b][fam] > 0 and b not in assigned),
                      key=lambda b: (-bt[b][fam], tiebreak[b]))
        for b in cand:
            place(b, fam)
    for b in sorted((b for b in block_ids if b not in assigned), key=lambda b: tiebreak[b]):
        place(b, None)

    amap = pl.DataFrame({"block": list(assigned), "split": [SPLITS[s] for s in assigned.values()]},
                        schema={"block": df.schema["block"], "split": pl.String})
    out = df.join(amap, on="block").select("row_id", "family", "split")
    return out, blocks.join(amap, on="block")


def dev_subsample(lf: pl.LazyFrame, max_rows: int, min_per_family: int, seed: int) -> pl.LazyFrame:
    """Stratified subsample of ONE split (lf has row_id, family): keeps ~max_rows rows in family proportion,
    but never fewer than min(min_per_family, family size) rows of any family, so rare attacks survive."""
    counts = dict(lf.group_by("family").len().collect().iter_rows())
    total = sum(counts.values())
    if total <= max_rows:
        return lf
    base = max_rows / total
    probs = {f: min(1.0, max(base, min_per_family / n)) for f, n in counts.items()}
    df = lf.collect()
    u = uniform01(df["row_id"].to_numpy(), seed + 7)
    p = df["family"].replace_strict(probs, return_dtype=pl.Float64).to_numpy()
    return df.filter(pl.Series(u < p)).lazy()


def build_split(dataset: str, cfg: dict) -> tuple[pl.DataFrame, dict]:
    """Dedup on the dataset's native track, then split the kept rows. Returns (split table, report)."""
    dcfg = cfg["datasets"][dataset]
    track = dcfg["dedup_track"]
    lf = pl.scan_parquet(tracks.processed_path(dataset, track))
    status = clean.dedup_status(lf, schema.features(track), keep_hash=True).collect(engine="streaming")
    counts = dict(status.group_by("status").len().iter_rows())
    conflict_vectors = status.filter(pl.col("status") == "conflict")["_h"].n_unique()
    status = status.drop("_h")
    cols = ["row_id", "family"] + (["ts"] if dcfg["scheme"] == "time_block" else [])
    kept = (lf.select(cols).join(status.lazy().filter(pl.col("status") == "keep"), on="row_id")
            .select(cols).collect(engine="streaming"))
    if dcfg["scheme"] == "time_block":
        out, _ = time_block_split(kept, cfg["fractions"], cfg["seed"], cfg["block_minutes"] * 60_000_000)
    elif dcfg["scheme"] == "stratified":
        out = stratified_split(kept, cfg["fractions"], cfg["seed"])
    else:
        raise ValueError(f"unknown split scheme {dcfg['scheme']}")
    h = split_hash(dataset, cfg)
    out = out.sort("row_id").with_columns(split_hash=pl.lit(h))
    report = {"dataset": dataset, "dedup_track": track, "scheme": dcfg["scheme"], "split_hash": h,
              "rows_in": status.height, "duplicates_dropped": counts.get("duplicate", 0),
              "conflict_rows_dropped": counts.get("conflict", 0),
              # one representative per conflicting vector would have survived exact dedup; this is the real cost
              "conflict_vectors_dropped": conflict_vectors, "rows_kept": out.height,
              **{f"n_{s}": out.filter(pl.col("split") == s).height for s in SPLITS}}
    return out, report


def split_hash(dataset: str, cfg: dict) -> str:
    return cfg_hash({"version": cfg["version"], "seed": cfg["seed"], "fractions": cfg["fractions"],
                     "block_minutes": cfg["block_minutes"], "dataset": dataset, **cfg["datasets"][dataset]})
