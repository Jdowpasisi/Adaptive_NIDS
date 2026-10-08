"""C16 dashboard: the real dashboard/app.py under Streamlit AppTest, against the toy API in-process."""

import types

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest
from test_api import FakeSelector, _cfg, _domain, _post_rows, parts  # noqa: F401  (fixture re-used)

from xnids.live import dashboard_data
from xnids.live.api import create_app
from xnids.live.service import DetectorService
from xnids.utils import paths


class Oracle:
    def __init__(self, y: dict) -> None:
        self.y = y

    def get(self, k, default=None):
        return self.y.get(k, default)


@pytest.fixture
def env(parts, tmp_path, monkeypatch):  # noqa: F811
    bundle_dir, src, _ = parts
    Xt, yt = _domain(3000, 1.0, 2)
    keys = [f"k{i}" for i in range(Xt.height)]
    truth = dict(zip(keys, yt.tolist(), strict=True))
    svc = DetectorService(_cfg(tmp_path), initial_bundle=bundle_dir, source=src, selector=FakeSelector(),
                          oracle=lambda ks: np.array([truth[k] for k in ks]))
    client = TestClient(create_app(svc))
    # DashboardData talks httpx to $DG_API: route it to the in-process app instead
    monkeypatch.setattr(dashboard_data, "httpx", types.SimpleNamespace(
        Client=lambda **kw: client, TransportError=Exception))
    monkeypatch.setattr(dashboard_data.DashboardData, "oracle", property(lambda self: Oracle(truth)))
    monkeypatch.setenv("DG_API", "http://toy")
    import streamlit as st

    st.cache_resource.clear()               # app.py caches one DashboardData per API URL across reruns
    return svc, client, Xt, keys


def _app() -> AppTest:
    at = AppTest.from_file(str(paths.REPO / "dashboard/app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_dashboard_renders_with_no_traffic(env):
    at = _app()
    assert [t.label for t in at.tabs] == ["📈 Live", "🌊 Drift", "🛠️ Adapt", "📜 Audit"]
    assert any("Waiting for flows" in i.value for i in at.info)
    assert any("No drift report yet" in i.value for i in at.info)


def test_overlay_series_follow_versions_across_promotion(env):
    svc, client, X, keys = env
    _post_rows(client, X.head(1500), keys[:1500])
    client.post("/adapt", json={"action": "adabn"})
    _post_rows(client, X.slice(1500, 500), keys[1500:2000])         # scored by active AND candidate
    client.post("/models/promote", json={"approved_by": "ana", "reason": "x"})
    _post_rows(client, X.slice(2000, 500), keys[2000:2500])         # active = adapted, shadow = original
    D = dashboard_data.DashboardData("http://toy")
    ov = D.overlay(client.get("/models").json()["original"], bucket_flows=500)
    n = ov.group_by("series").agg(pl.col("n").sum()).sort("series")
    assert n.rows() == [("adapted", 1000), ("original", 2500)]     # original scored throughout
    assert ov["fpr"].drop_nulls().is_between(0, 1).all()


def test_approve_from_the_dashboard_needs_name_and_reason(env):
    svc, client, X, keys = env
    _post_rows(client, X.head(1500), keys[:1500])
    client.post("/adapt", json={"action": "adabn"})
    at = _app()
    approve = next(b for b in at.button if b.label.startswith("✅ Approve"))
    assert not approve.disabled
    approve.click()
    at.run()                                                        # no name / reason -> API refuses (422)
    assert svc.registry.version("active") == "toy-v1"
    at.text_input(key="approver").input("ana")
    at.text_input(key="reason").input("gates pass")
    next(b for b in at.button if b.label.startswith("✅ Approve")).click()
    at.run()
    assert svc.registry.version("active") != "toy-v1"
    assert any("Promoted" in s.value for s in at.success)
    at.text_input(key="rb_name").input("ana")
    at.text_input(key="rb_reason").input("demo")
    next(b for b in at.button if b.label.startswith("↩")).click()
    at.run()
    assert svc.registry.version("active") == "toy-v1"


def test_rebuild_offered_only_when_gates_fail(env):
    svc, client, X, keys = env
    _post_rows(client, X.head(1500), keys[:1500])
    client.post("/adapt", json={"action": "adabn"})
    at = _app()
    assert svc.registry.state["candidate"]["gates"]["all_passed"]
    assert not [b for b in at.button if b.key == "rebuild"]          # gates pass: no rebuild offered
    svc.registry.state["candidate"]["gates"]["all_passed"] = False   # make it a blocked candidate
    at = _app()
    assert next(b for b in at.button if b.label.startswith("✅ Approve")).disabled
    at.button(key="rebuild").click()
    at.run()
    c = svc.registry.state["candidate"]
    assert c and c["gates"]["all_passed"] in (True, False)            # a NEW candidate was built
    kinds = [a["kind"] for a in svc.store.actions()]
    assert "reject" in kinds and kinds.count("adapt") == 2
