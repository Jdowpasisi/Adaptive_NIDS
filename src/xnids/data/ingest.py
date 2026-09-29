"""C1 data ingest: fetch -> checksum -> manifest -> CSV-to-Parquet (streaming) -> profile.

Output per dataset:
  data/raw/<name>/...                      downloaded files (checksummed in data/MANIFEST.csv)
  data/interim/<name>/<name>.parquet       all rows, cleaned column names, floats as float32
  reports/tables/rename_map_<name>.json    original header -> cleaned name, per source CSV
  reports/tables/profile_<name>.csv        per column dtype / null / NaN / inf counts
  reports/tables/labels_<name>.csv         raw label counts

Polars streams the CSV, so the 13.7M-row LycoS18 file never has to fit in RAM.
"""

import csv
import datetime as dt
import glob
import hashlib
import json
import logging
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

from xnids.data.labellers import LABELLERS
from xnids.utils import paths

log = logging.getLogger(__name__)

MANIFEST_COLS = ["dataset", "file", "sha256", "bytes", "rows", "date"]


class ManualDownloadRequired(RuntimeError):
    pass


# ---------------------------------------------------------------- column names

def clean_name(name: str) -> str:
    """' Fwd Header Length' -> 'fwd_header_length'; 'Flow Bytes/s' -> 'flow_bytes_s'."""
    s = name.strip().lower()
    s = re.sub(r"[^0-9a-z]+", "_", s)
    return s.strip("_") or "unnamed"


def clean_names(header: list[str]) -> list[str]:
    """Clean every name and de-duplicate explicitly: the 2nd 'fwd_header_length' becomes 'fwd_header_length_2'."""
    out, seen = [], {}
    for h in header:
        c = clean_name(h)
        seen[c] = seen.get(c, 0) + 1
        out.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    return out


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="latin-1") as f:
        return next(csv.reader(f))


# ---------------------------------------------------------------- fetch & checksum

def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _download_url(url: str, dest: Path) -> None:
    """Resumable HTTP download (Range request onto a .part file)."""
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={have}-"})
    with urllib.request.urlopen(req) as r:
        mode = "ab" if r.status == 206 else "wb"
        with part.open(mode) as f:
            shutil.copyfileobj(r, f, length=1 << 22)
    part.rename(dest)


def fetch(name: str, spec: dict) -> list[Path]:
    """Make sure the dataset's source files are in data/raw/<name>/ and return them."""
    raw = paths.RAW / name
    raw.mkdir(parents=True, exist_ok=True)
    f = spec["fetch"]
    method = f["method"]
    if method == "from":
        return []
    if method == "manual":
        found = sorted(p for p in raw.glob(f["file"]) if p.is_file())
        if not found:
            raise ManualDownloadRequired(
                f"{name}: download it from {f['url']} (fill in the UQ form in a browser) and put the "
                f"file into {raw}/ then re-run."
            )
        return found
    dest = raw / f["file"]
    if dest.exists():
        return [dest]
    log.info("%s: downloading -> %s", name, dest)
    if method == "url":
        _download_url(f["url"], dest)
    elif method == "gdown":
        import gdown

        gdown.download(id=f["id"], output=str(dest), quiet=False, resume=True)
    else:
        raise ValueError(f"unknown fetch method {method}")
    return [dest]


def extract(name: str, spec: dict) -> None:
    """Unzip the archives named by `extract` patterns, in order (so nested zips work).

    Only members missing on disk are written, so re-running is cheap and also restores CSVs that were
    deleted with --delete-raw-csv.
    """
    base = paths.RAW / (spec["fetch"].get("from") or name)
    for pattern in spec.get("extract", []):
        for z in sorted(glob.glob(str(base / pattern), recursive=True)):
            z = Path(z)
            with zipfile.ZipFile(z) as zf:
                missing = [m for m in zf.infolist() if not m.is_dir() and not (z.parent / m.filename).exists()]
                if missing:
                    log.info("%s: extracting %d file(s) from %s", name, len(missing), z.name)
                    for m in missing:
                        zf.extract(m, z.parent)


def update_manifest(name: str, files: list[Path], rows: int | None) -> None:
    paths.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if paths.MANIFEST.exists():
        with paths.MANIFEST.open(newline="") as fh:
            existing = [r for r in csv.DictReader(fh) if r["dataset"] != name]
    today = dt.date.today().isoformat()
    new = [
        {"dataset": name, "file": str(p.relative_to(paths.RAW)), "sha256": sha256(p),
         "bytes": p.stat().st_size, "rows": rows if rows is not None else "", "date": today}
        for p in files
    ]
    with paths.MANIFEST.open("w", newline="") as fh:
        w = csv.DictWriter(fh, MANIFEST_COLS)
        w.writeheader()
        w.writerows(existing + new)


# ---------------------------------------------------------------- CSV -> Parquet

def source_csvs(name: str, spec: dict) -> list[Path]:
    base = paths.RAW / (spec["fetch"].get("from") or name)
    files = sorted(Path(p) for p in glob.glob(str(base / spec["csv_glob"]), recursive=True))
    if not files:
        raise FileNotFoundError(f"{name}: no CSV matches {base / spec['csv_glob']}")
    return files


def _scan(path: Path, spec: dict, defaults: dict) -> tuple[pl.LazyFrame, dict[str, str]]:
    header = read_header(path)
    names = clean_names(header)
    id_cols = set(spec.get("id_cols", []))
    # Every non-id, non-label column in these datasets is a numeric flow feature, so declare it Float64
    # rather than trusting inference: CIC files contain 'Infinity', and integer-looking columns can turn
    # float late in a file. A truly non-numeric feature then fails loudly instead of becoming a string.
    # Ids and the label keep their inferred type (timestamps stay exact integers).
    sample = pl.read_csv(path, n_rows=defaults["infer_schema_rows"], new_columns=names,
                         encoding=defaults["csv_encoding"], infer_schema_length=None,
                         null_values=["", "NaN", "nan"])
    keep = id_cols | {clean_name(spec["label_col"])}
    schema = {c: ((t if t != pl.Null else pl.String) if c in keep else pl.Float64) for c, t in sample.schema.items()}
    lf = pl.scan_csv(path, has_header=True, new_columns=names, schema=schema,
                     encoding=defaults["csv_encoding"], null_values=["", "NaN", "nan"])
    return lf, dict(zip(header, names, strict=True))


def to_parquet(name: str, spec: dict, defaults: dict) -> Path:
    out_dir = paths.INTERIM / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.parquet"
    label_col = clean_name(spec["label_col"])
    frames, rename_maps = [], {}
    for p in source_csvs(name, spec):
        lf, rmap = _scan(p, spec, defaults)
        rename_maps[p.name] = rmap
        # rows where every field is empty (the original CIC-IDS2017 CSVs carry ~289k of these)
        lf = lf.filter(~pl.all_horizontal(pl.all().is_null()))
        if spec.get("labeller"):
            lf = LABELLERS[spec["labeller"]](lf, p.name, label_col)
        frames.append(lf.with_columns(pl.lit(p.name).alias("source_file")))
    lf = pl.concat(frames, how="diagonal_relaxed")
    id_cols = set(spec.get("id_cols", []))
    float_t = pl.Float32 if defaults["float_dtype"] == "float32" else pl.Float64
    lf = lf.with_columns(
        pl.col(c).cast(float_t) for c, t in lf.collect_schema().items() if t == pl.Float64 and c not in id_cols
    )
    log.info("%s: writing %s", name, out)
    lf.sink_parquet(out, compression="zstd", row_group_size=500_000)
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    (paths.TABLES / f"rename_map_{name}.json").write_text(json.dumps(rename_maps, indent=1))
    return out


# ---------------------------------------------------------------- profile

def profile(name: str, parquet: Path, label_col: str) -> dict:
    lf = pl.scan_parquet(parquet)
    schema = lf.collect_schema()
    floats = [c for c, t in schema.items() if t.is_float()]
    stats = lf.select(
        pl.len().alias("__rows"),
        *[pl.col(c).null_count().alias(f"null|{c}") for c in schema],
        *[pl.col(c).is_nan().sum().alias(f"nan|{c}") for c in floats],
        *[pl.col(c).is_infinite().sum().alias(f"inf|{c}") for c in floats],
    ).collect(engine="streaming").row(0, named=True)
    rows = stats["__rows"]
    prof = pl.DataFrame({
        "column": list(schema),
        "dtype": [str(t) for t in schema.values()],
        "nulls": [stats[f"null|{c}"] for c in schema],
        "nans": [stats.get(f"nan|{c}", 0) for c in schema],
        "infs": [stats.get(f"inf|{c}", 0) for c in schema],
    })
    prof.write_csv(paths.TABLES / f"profile_{name}.csv")
    labels = (lf.group_by(label_col).len().sort("len", descending=True)
              .with_columns((pl.col("len") / rows).alias("share")).collect(engine="streaming"))
    labels.write_csv(paths.TABLES / f"labels_{name}.csv")
    benign = labels.filter(pl.col(label_col).cast(pl.String).str.to_lowercase().is_in(["benign", "normal"]))
    return {"rows": rows, "n_labels": labels.height,
            "benign_share": float(benign["share"].sum()) if benign.height else 0.0}


# ---------------------------------------------------------------- driver

def ingest(name: str, spec: dict, defaults: dict, delete_raw_csv: bool = False) -> dict:
    fetched = fetch(name, spec)
    extract(name, spec)
    parquet = to_parquet(name, spec, defaults)
    result = profile(name, parquet, clean_name(spec["label_col"]))
    expected = spec.get("expected_rows")
    result["expected_rows"] = expected
    result["rows_match"] = expected is None or expected == result["rows"]
    # checksum the downloaded artefacts; for archives also the CSVs we actually read
    csvs = source_csvs(name, spec)
    files = fetched + [c for c in csvs if c not in fetched]
    update_manifest(name, files, result["rows"])
    if delete_raw_csv:
        # CSVs unpacked from an archive (the archive stays) and gdown files (re-downloadable by id) can go;
        # their sha256 stays in the manifest. Manually downloaded files are never deleted.
        for c in csvs:
            if c not in fetched or spec["fetch"]["method"] == "gdown":
                c.unlink()
    return result
