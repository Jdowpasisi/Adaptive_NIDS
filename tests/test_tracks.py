import glob

import polars as pl
import pytest

from xnids.data import labels, schema
from xnids.features import core_map
from xnids.utils import paths


def test_norm_handles_encodings():
    assert labels.norm("Web Attack � XSS") == labels.norm("Web Attack – XSS") == "web attack xss"
    assert labels.norm("webattack_sql_injection") == "webattack sql injection"


def test_every_label_seen_in_c1_is_mapped():
    tables = glob.glob(str(paths.TABLES / "labels_*.csv"))
    if not tables:
        pytest.skip("no C1 label tables")
    raw = [str(v) for t in tables for v in pl.read_csv(t, infer_schema=False).to_series(0).to_list()]
    labels.check_mapped(raw)


def test_unmapped_label_raises():
    with pytest.raises(labels.UnmappedLabelError, match="Totally New Attack"):
        labels.check_mapped(["Benign", "Totally New Attack"])


def test_family_and_binary():
    df = pl.DataFrame({"l": ["BENIGN", "DoS Hulk", "Brute Force -Web", "SSH-Bruteforce", "scanning"]})
    out = df.select(f=labels.to_family(pl.col("l"))).with_columns(y=labels.to_binary(pl.col("f")))
    assert out["f"].to_list() == ["Benign", "DoS", "WebAttack", "BruteForce", "Recon"]
    assert out["y"].to_list() == [0, 1, 1, 1, 1]


def test_families_are_closed_set():
    assert set(labels.load_map().values()) <= set(labels.FAMILIES)


def test_identifier_detection():
    cols = ["dst_port", "l4_dst_port", "flow_bytes_s", "src_addr", "l4_src_port", "timestamp",
            "dns_query_id", "ipv4_dst_addr", "source_port"]
    assert schema.identifier_features(cols) == [
        "src_addr", "l4_src_port", "timestamp", "dns_query_id", "ipv4_dst_addr", "source_port"]


@pytest.mark.parametrize("track", ["cic77", "nf43", "core", "cic_orig"])
def test_track_configs_have_no_identifiers(track):
    assert schema.identifier_features(schema.features(track)) == []


def _frame(track, **override):
    feats = schema.features(track)
    df = pl.DataFrame({"row_id": pl.Series([0, 1], dtype=pl.UInt32), "y": pl.Series([0, 1], dtype=pl.Int8),
                       "family": ["Benign", "DoS"], "label": ["benign", "dos_hulk"],
                       **{f: pl.Series([1.0, 2.0], dtype=pl.Float32) for f in feats}})
    return df.with_columns(**override) if override else df


def test_validate_accepts_contract():
    schema.validate(_frame("core"), "core")


@pytest.mark.parametrize("bad, msg", [
    ({"y": pl.Series([1, 1], dtype=pl.Int8)}, "y_vs_family"),
    ({"family": pl.lit("Nonsense")}, "family"),
    ({"pkts_fwd": pl.Series([1.0, 2.0], dtype=pl.Float64)}, "expected Float32"),
    ({"row_id": pl.Series([3, 3], dtype=pl.UInt32)}, "row_id_dupes"),
])
def test_validate_rejects(bad, msg):
    with pytest.raises(schema.SchemaError, match=msg):
        schema.validate(_frame("core", **bad), "core")


def test_validate_rejects_missing_feature():
    with pytest.raises(schema.SchemaError, match="missing"):
        schema.validate(_frame("core").drop("dst_port"), "core")


def test_core_map_nf_flags_and_bytes():
    nf = pl.DataFrame({c: [0.0] for c in [
        "flow_duration_milliseconds", "in_pkts", "out_pkts", "in_bytes", "out_bytes", "protocol",
        "longest_flow_pkt", "shortest_flow_pkt", "tcp_win_max_in", "tcp_win_max_out", "l4_dst_port"]}
    ).with_columns(tcp_flags=pl.lit(27.0), in_pkts=pl.lit(4.0), in_bytes=pl.lit(400.0))  # 27 = FIN|SYN|PSH|ACK
    out = nf.select(core_map.expressions("nf_unsw_v2")).row(0, named=True)
    assert (out["flag_fin"], out["flag_syn"], out["flag_rst"], out["flag_psh"], out["flag_ack"]) == (1, 1, 0, 1, 1)
    assert out["bytes_per_pkt_fwd"] == 100.0 and out["bytes_per_pkt_bwd"] == 0.0
    assert "duration_s" not in out  # dropped by the bridge check


def test_core_map_cic_ip_byte_rebuild():
    cols = ["flow_duration", "fwd_pkt_cnt", "bwd_pkt_cnt", "fwd_pkt_len_tot", "bwd_pkt_len_tot",
            "fwd_pkt_hdr_len_tot", "bwd_pkt_hdr_len_tot", "ip_prot", "pkt_len_max", "pkt_len_min",
            "flag_syn", "flag_fin", "flag_rst", "flag_psh", "flag_ack", "fwd_tcp_init_win_bytes",
            "bwd_tcp_init_win_bytes", "dst_port"]
    row = dict.fromkeys(cols, 0.0) | {"fwd_pkt_cnt": 3.0, "fwd_pkt_len_tot": 200.0, "fwd_pkt_hdr_len_tot": 72.0,
                                      "ip_prot": 6.0, "pkt_len_max": 200.0, "flag_syn": 2.0}
    out = pl.DataFrame([row]).select(core_map.expressions("lycos18")).row(0, named=True)
    assert out["bytes_fwd"] == 200 + 72 + 60          # payload + L4 headers + 3 * 20-byte IPv4 header
    assert out["pkt_len_max"] == 240                  # 200 payload + 20 IP + 20 TCP
    assert out["flag_syn"] == 1.0 and out["flag_fin"] == 0.0
