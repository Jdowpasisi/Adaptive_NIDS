import csv
import zipfile

import polars as pl
import pytest

from xnids.data import ingest, labellers


def test_clean_names_dedupes():
    hdr = [" Destination Port", "Flow Bytes/s", " Fwd Header Length", "Fwd Header Length", " Label"]
    assert ingest.clean_names(hdr) == [
        "destination_port", "flow_bytes_s", "fwd_header_length", "fwd_header_length_2", "label"]


@pytest.fixture
def raw_env(tmp_path, monkeypatch):
    for attr, sub in [("RAW", "raw"), ("INTERIM", "interim"), ("TABLES", "tables"), ("MANIFEST", "MANIFEST.csv")]:
        monkeypatch.setattr(ingest.paths, attr, tmp_path / sub)
    return tmp_path


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


DEFAULTS = {"csv_encoding": "utf8-lossy", "infer_schema_rows": 2, "float_dtype": "float32"}


def test_csv_to_parquet_roundtrip(raw_env):
    # duplicate header, Infinity, an int column that turns float after the inference sample, a NaN
    _write_csv(raw_env / "raw/toy/toy.csv",
               [" Flow ID", " Dst Port", "Fwd Header Length", " Fwd Header Length", "Flow Bytes/s", " Label"],
               [["a", 80, 1, 1, 10, "BENIGN"], ["b", 443, 2, 2, "Infinity", "DoS"],
                ["c", 22, 3, 3.5, "NaN", "BENIGN"]])
    spec = {"fetch": {"method": "manual", "url": "x", "file": "toy.csv"}, "csv_glob": "toy.csv",
            "label_col": " Label", "expected_rows": 3, "id_cols": ["flow_id", "dst_port"]}
    res = ingest.ingest("toy", spec, DEFAULTS)
    assert res["rows"] == 3 and res["rows_match"]
    assert res["benign_share"] == pytest.approx(2 / 3)
    df = pl.read_parquet(raw_env / "interim/toy/toy.parquet")
    assert df.columns[:5] == ["flow_id", "dst_port", "fwd_header_length", "fwd_header_length_2", "flow_bytes_s"]
    assert df.schema["fwd_header_length_2"] == pl.Float32
    assert df.schema["dst_port"] == pl.Int64          # id column not downcast
    assert df["fwd_header_length_2"].to_list() == [1.0, 2.0, 3.5]
    prof = pl.read_csv(raw_env / "tables/profile_toy.csv")
    fb = prof.filter(pl.col("column") == "flow_bytes_s").row(0, named=True)
    assert fb["infs"] == 1 and fb["nulls"] == 1
    man = pl.read_csv(raw_env / "MANIFEST.csv")
    assert man.height == 1 and man["rows"][0] == 3 and len(man["sha256"][0]) == 64
    # re-ingest replaces, not appends, the dataset's manifest rows
    ingest.ingest("toy", spec, DEFAULTS)
    assert pl.read_csv(raw_env / "MANIFEST.csv").height == 1


def test_manual_missing_raises(raw_env):
    spec = {"fetch": {"method": "manual", "url": "https://x", "file": "*.zip"}}
    with pytest.raises(ingest.ManualDownloadRequired):
        ingest.fetch("nf", spec)


def test_extract_restores_deleted_member(raw_env):
    d = raw_env / "raw/z"
    d.mkdir(parents=True)
    with zipfile.ZipFile(d / "a.zip", "w") as zf:
        zf.writestr("inner/x.csv", "a,b\n1,2\n")
    spec = {"fetch": {"method": "manual", "file": "a.zip"}, "extract": ["a.zip"]}
    ingest.extract("z", spec)
    assert (d / "inner/x.csv").exists()
    (d / "inner/x.csv").unlink()
    ingest.extract("z", spec)
    assert (d / "inner/x.csv").exists()


def _lycos_frame(rows):
    cols = ["timestamp", "src_addr", "dst_addr", "src_port", "dst_port", "ip_prot"]
    return pl.LazyFrame(rows, schema=cols, orient="row")


def test_lycos17_tuesday():
    lf = _lycos_frame([
        (1499170700000000, "172.16.0.1", "192.168.10.50", 5000, 21, 6),   # ftp window
        (1499188200000000, "172.16.0.1", "192.168.10.50", 5000, 22, 6),   # ssh window
        (1499170700000000, "10.0.0.1", "192.168.10.50", 5000, 21, 6),     # wrong attacker
    ])
    out = labellers.lycos17(lf, "Tuesday-WorkingHours.pcap_lycos.csv").collect()
    assert out["label"].to_list() == ["ftp_patator", "ssh_patator", "benign"]


def test_lycos17_friday_drops_and_order():
    lf = _lycos_frame([
        (1499436200000000, "192.168.10.5", "205.174.165.73", 1, 8080, 6),  # dropped (after bot window)
        (1499431000000000, "192.168.10.5", "205.174.165.73", 1, 8080, 6),  # bot
        (1499431000000000, "192.168.10.5", "52.6.13.28", 1, 80, 6),        # dropped
        (1499453800000000, "172.16.0.1", "192.168.10.50", 1, 80, 6),       # ddos
        (1499443600000000, "172.16.0.1", "192.168.10.50", 1, 80, 17),      # UDP -> benign
    ])
    out = labellers.lycos17(lf, "Friday-WorkingHours.pcap_lycos.csv").collect()
    assert out["label"].to_list() == ["bot", "ddos", "benign"]


def test_lycos17_thursday_afternoon_dropped():
    lf = _lycos_frame([(1499353300000000, "a", "b", 1, 1, 6), (1499343400000000, "172.16.0.1", "192.168.10.50", 1, 80, 6)])
    out = labellers.lycos17(lf, "Thursday-WorkingHours.pcap_lycos.csv").collect()
    assert out["label"].to_list() == ["webattack_bruteforce"]


def test_all_empty_rows_dropped(raw_env):
    _write_csv(raw_env / "raw/e/e.csv", ["a", "Label"], [[1, "BENIGN"], ["", ""], [2, "DoS"], ["", ""]])
    spec = {"fetch": {"method": "manual", "url": "x", "file": "e.csv"}, "csv_glob": "e.csv",
            "label_col": "Label", "expected_rows": 2, "id_cols": []}
    assert ingest.ingest("e", spec, DEFAULTS)["rows"] == 2
