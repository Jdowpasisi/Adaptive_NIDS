"""C15 replay runner against the real API in-process (TestClient), on synthetic pre-extracted flows."""

import json

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from test_api import NAMES, FakeSelector, _cfg, _domain, parts  # noqa: F401  (fixture re-used)

from xnids.drift.monitor import DriftMonitor
from xnids.live.api import create_app
from xnids.live.replay import FlowSource, ReplayRunner
from xnids.live.service import DetectorService


def _flows(n, shift, seed, t0=0):
    X, y = _domain(n, shift, seed)
    rng = np.random.default_rng(seed)
    return X.with_columns(src_addr=pl.lit("10.0.0.1"), src_port=pl.Series(rng.integers(1024, 65535, n)),
                          dst_addr=pl.lit("10.0.0.2"), dst_port=pl.lit(80), ip_prot=pl.lit(6),
                          timestamp=pl.Series(np.arange(n) * 1000 + t0), end_timestamp=pl.Series(
                              np.arange(n) * 1000 + t0 + 500)), y


@pytest.fixture
def setup(parts, tmp_path):  # noqa: F811
    bundle_dir, src, (Xv, yv) = parts
    svc = DetectorService(_cfg(tmp_path), initial_bundle=bundle_dir, source=src, selector=FakeSelector())
    a, _ = _flows(3000, 0.0, 10)
    b, _ = _flows(3000, 1.0, 11, t0=10**9)
    p = tmp_path / "flows.parquet"
    pl.concat([a, b]).write_parquet(p)
    cfg = {"model": {"track": "toy"}, "monitor": {"window": 1000}, "micro_batch_ms": 50, "max_batch": 250}
    return svc, p, cfg, (Xv, yv)


def test_runner_streams_everything_in_order_and_records_batches(setup, tmp_path):
    svc, p, cfg, _ = setup
    c = TestClient(create_app(svc))
    r = ReplayRunner(cfg, "http://test", tmp_path / "run", rate=None, labels=False, client=c, features=NAMES)
    s = r.run(FlowSource("parquet", p))
    assert s["flows"] == 6000 and s["requests"] >= 24                      # max_batch 250
    f = pl.read_parquet(tmp_path / "run" / "flows.parquet")
    src = pl.read_parquet(p)
    assert f.height == 6000 and f["end_timestamp"].to_list() == src["end_timestamp"].to_list()   # emission order
    assert svc.store.query("SELECT COUNT(*) AS n FROM scores")[0]["n"] == 6000
    b = pl.read_csv(tmp_path / "run" / "batches.csv")
    assert b["n"].sum() == 6000 and (b["latency_s"] > 0).all()


def test_runner_rate_limit_is_respected(setup, tmp_path):
    svc, p, cfg, _ = setup
    c = TestClient(create_app(svc))
    r = ReplayRunner(cfg, "http://test", tmp_path / "run", rate=4000, labels=False, client=c, features=NAMES)
    s = r.run(FlowSource("parquet", p, limit=2000))
    assert s["flows"] == 2000 and 0.3 < s["seconds"] < 2.5                  # ~0.5 s at 4,000 flows/s


def test_runner_posts_one_drift_report_per_window_and_can_auto_adapt(setup, tmp_path):
    svc, p, cfg, (Xv, yv) = setup
    svc.cfg["min_adapt_rows"] = 500
    c = TestClient(create_app(svc))
    mon = DriftMonitor(svc.registry.get("active"), Xv, yv, {"reference": {"n": 2000},
                                                             "mmd": {"n_perm": 20, "n_ref": 300, "n_win": 300}})
    mon.process = _forced(mon.process)                       # recommend from the 5th window on
    r = ReplayRunner(cfg, "http://test", tmp_path / "run", labels=False, client=c, features=NAMES, monitor=mon,
                     auto_adapt=True, auto_approve=False, adapt_action="adabn")
    r.run(FlowSource("parquet", p))
    w = pl.read_csv(tmp_path / "run" / "windows.csv")
    assert w["window"].to_list() == list(range(6))
    assert svc.store.query("SELECT COUNT(*) AS n FROM drift")[0]["n"] == 6
    assert w.filter(pl.col("window") == 4)["event"][0].startswith("candidate adabn")
    assert svc.registry.version("candidate") is not None and svc.registry.version("active") == "toy-v1"


def _forced(process):
    def f(X):
        rep = process(X)
        rep.recommend = rep.window_id >= 4
        return rep
    return f


def test_flow_key_label_fallback_to_tuple():
    from xnids.live.replay_labels import ReplayOracle

    lab = pl.DataFrame({"src_addr": ["a", "b", "b"], "src_port": [1, 2, 2], "dst_addr": ["c"] * 3,
                        "dst_port": [80] * 3, "ip_prot": [6] * 3, "timestamp": [5, 7, 9], "segment": ["A"] * 3,
                        "family": ["DoS", "Benign", "DoS"], "label": ["dos_hulk", "benign", "dos_hulk"],
                        "y": [1, 0, 1]})
    o = ReplayOracle(lab)
    assert o.get("a|1|c|80|6|5") == 1                         # exact
    assert o.get("a|1|c|80|6|999") == 1                       # live: wall-clock start, unambiguous 5-tuple
    assert o.get("b|2|c|80|6|999") is None                    # ambiguous 5-tuple: no label
    assert json.dumps(o.get("x|1|c|80|6|1", -1)) == "-1"
