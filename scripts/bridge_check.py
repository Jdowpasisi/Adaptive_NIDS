"""C2 / H8: validate the core map on the bridge pair.

LycoS18 and NF-CSE-CIC-IDS2018-v2 were extracted from the SAME CSE-CIC-IDS2018 PCAPs by different tools.
For each core feature we compare the two distributions (quantiles, share of zeros, two-sample KS) on benign
traffic, on all traffic, and per attack family present in both. On identical traffic a well-mapped feature
should agree; a large KS points at a unit error or an extractor artefact.

Flows are not row-aligned between extractors (different flow timeouts), so this compares distributions only.

    python scripts/bridge_check.py [--sample 400000] [--seed 0]
"""

import argparse

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

from xnids.data import labels
from xnids.data.ingest import clean_name
from xnids.features import core_map, tracks
from xnids.utils import paths

A, B = "lycos18", "nf_cse_cic18_v2"
QS = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def sample(dataset: str, n: int, seed: int) -> pl.DataFrame:
    spec = tracks.dataset_spec(dataset)
    lf = pl.scan_parquet(tracks.interim_path(dataset)).with_row_index("i")
    family = labels.to_family(pl.col(clean_name(spec["label_col"])))
    total = lf.select(pl.len()).collect().item()
    keep = min(1.0, 3 * n / total)  # oversample, then draw exactly per group below
    lf = lf.filter((pl.col("i").hash(seed) % 1_000_000) < int(keep * 1_000_000))
    return lf.select(family.alias("family"), *core_map.expressions(dataset, include_dropped=True)).collect()


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    ks = ks_2samp(a, b)
    row = {"n_a": len(a), "n_b": len(b), "ks": ks.statistic, "zero_a": float((a == 0).mean()),
           "zero_b": float((b == 0).mean())}
    for q, qa, qb in zip(QS, np.quantile(a, QS), np.quantile(b, QS), strict=True):
        row[f"q{int(q * 100):02d}_a"], row[f"q{int(q * 100):02d}_b"] = float(qa), float(qb)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    da, db = sample(A, args.sample, args.seed), sample(B, args.sample, args.seed)
    feats = [f["name"] for f in core_map.load()["features"]]
    shared = sorted(set(da["family"]) & set(db["family"]))
    groups = {"benign": "Benign", "all": None} | {f"fam:{f}": f for f in shared if f != "Benign"}
    rows = []
    for gname, fam in groups.items():
        ga = da if fam is None else da.filter(pl.col("family") == fam)
        gb = db if fam is None else db.filter(pl.col("family") == fam)
        ga, gb = ga.head(args.sample), gb.head(args.sample)
        if min(ga.height, gb.height) < 200:
            continue
        for f in feats:
            rows.append({"group": gname, "feature": f,
                         **compare(ga[f].to_numpy().astype(float), gb[f].to_numpy().astype(float))})
    out = pl.DataFrame(rows)
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    out.write_csv(paths.TABLES / "bridge_check.csv")
    summary = (out.filter(pl.col("group").is_in(["benign", "all"]))
               .pivot(on="group", index="feature", values="ks")
               .join(out.filter(pl.col("group").str.starts_with("fam:")).group_by("feature")
                     .agg(pl.col("ks").median().alias("ks_attack_median")), on="feature")
               .join(out.filter(pl.col("group") == "benign")
                     .select("feature", "q50_a", "q50_b", "zero_a", "zero_b"), on="feature"))
    summary = pl.DataFrame({"feature": feats}).join(summary, on="feature", how="left")
    summary.write_csv(paths.TABLES / "bridge_check_summary.csv")
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=200, float_precision=3):
        print(f"groups compared: {list(groups)}")
        print(summary)


if __name__ == "__main__":
    main()
