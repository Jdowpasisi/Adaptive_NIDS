"""C13: demo-data pieces (NFStream column tidying, schedule labeller, split holdout window, nfs track)."""

import datetime as dt

import numpy as np
import pandas as pd
import polars as pl

from xnids.data import labellers, schema, split
from xnids.live import extract


def test_tidy_renames_ids_and_drops_absolute_times():
    raw = pd.DataFrame({"id": [0], "expiration_id": [0], "src_ip": ["1.1.1.1"], "src_mac": ["x"], "src_port": [5],
                        "dst_ip": ["2.2.2.2"], "dst_port": [80], "protocol": [6], "ip_version": [4],
                        "bidirectional_first_seen_ms": [1000], "bidirectional_last_seen_ms": [3000],
                        "src2dst_first_seen_ms": [1000], "dst2src_last_seen_ms": [3000],
                        "bidirectional_duration_ms": [2000], "application_name": ["HTTP"], "src2dst_bytes": [9]})
    t = extract._tidy(raw)
    assert list(t.columns[:2]) == ["timestamp", "end_timestamp"] and t.timestamp[0] == 1_000_000
    assert {"src_addr", "dst_addr", "ip_prot"} <= set(t.columns)
    assert not [c for c in t.columns if c.endswith("_seen_ms") or c.startswith(("application_", "src_mac"))]


def test_nfstream_config_matches_lock():
    assert extract.check_frozen() == extract.nfs_hash()


def test_schedule_labels_respect_offset_pair_and_window():
    att = [{"label": "DoS-Hulk", "start": "13:45", "end": "14:19", "attacker": ["9.9.9.9"], "victim": ["1.1.1.1"]}]
    utc = lambda h, m: int(dt.datetime(2018, 2, 16, h, m, tzinfo=dt.UTC).timestamp() * 1e6)  # noqa: E731
    df = pl.DataFrame({"src_addr": ["9.9.9.9", "9.9.9.9", "8.8.8.8", "9.9.9.9"],
                       "dst_addr": ["1.1.1.1", "1.1.1.1", "1.1.1.1", "1.1.1.1"],
                       # 13:50 local = 18:50 UTC at offset -5; 13:50 UTC is outside; wrong attacker; 19:30 UTC outside
                       "timestamp": [utc(18, 50), utc(13, 50), utc(18, 50), utc(19, 30)]})
    out = labellers.schedule_labels(df.lazy(), att, "Friday-16-02-2018", -5).collect()["label"].to_list()
    assert out == ["dos_hulk", "benign", "benign", "benign"]


def test_holdout_window_rows_get_no_split(tmp_path, monkeypatch):
    feats = schema.features("nfs")
    n = 4000
    rng = np.random.default_rng(0)
    t0 = int(dt.datetime(2017, 7, 5, 12, 0, tzinfo=dt.UTC).timestamp() * 1e6)
    ts = t0 + np.sort(rng.integers(0, 3 * 3600 * 10**6, n))
    fam = np.where(rng.random(n) < 0.3, "dos", "benign")
    df = pl.DataFrame({"row_id": np.arange(n, dtype=np.uint32), "ts": ts, "y": (fam == "dos").astype(np.int8),
                       "family": fam, "label": fam,
                       **{f: rng.normal(size=n).astype(np.float32) for f in feats}})
    p = tmp_path / "nfs.parquet"
    df.write_parquet(p)
    monkeypatch.setattr(split.tracks, "processed_path", lambda ds, tr: p)
    cfg = {"version": 1, "seed": 0, "fractions": {"train": 0.6, "val": 0.2, "test": 0.2}, "block_minutes": 5,
           "datasets": {"toy": {"scheme": "time_block", "dedup_track": "nfs",
                                "holdout": ["2017-07-05T13:00:00", "2017-07-05T13:30:00"]}}}
    out, rep = split.build_split("toy", cfg)
    lo, hi = t0 + 3600 * 10**6, t0 + 5400 * 10**6
    inside = set(df.filter(pl.col("ts").is_between(lo, hi, closed="left"))["row_id"])
    assert rep["rows_held_out"] == len(inside) > 0
    assert not inside & set(out["row_id"]) and out.height == n - len(inside)
    assert set(out["split"]) == set(split.SPLITS)


def test_nfs_track_has_no_identifiers():
    assert schema.identifier_features(schema.features("nfs")) == []


def test_replay_label_lookup_tolerates_duplicate_keys_but_not_conflicts():
    import pytest

    from xnids.live import replay_labels

    k = {"src_addr": ["a", "a", "b"], "src_port": [1, 1, 2], "dst_addr": ["c", "c", "c"], "dst_port": [80, 80, 80],
         "ip_prot": [6, 6, 6], "timestamp": [5, 5, 7]}
    lab = pl.DataFrame({**k, "segment": ["A"] * 3, "family": ["DoS", "DoS", "Benign"],
                        "label": ["dos_hulk", "dos_hulk", "benign"], "y": [1, 1, 0]})
    out = replay_labels.attach(pl.DataFrame(k), lab)
    assert out.height == 3 and out["y"].to_list() == [1, 1, 0]
    with pytest.raises(ValueError):
        replay_labels.attach(pl.DataFrame(k), lab.with_columns(label=pl.Series(["dos_hulk", "benign", "benign"])))
