"""C14 detector API (Build Guide done-when: 422 on a bad schema, promote needs approval, rollback restores the
version) plus the registry, gate checks, drift -> recommendation, and /adapt on a small synthetic MLP (CPU)."""

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from xnids.drift.monitor import DriftMonitor
from xnids.eval import metrics
from xnids.live import registry as reg
from xnids.live.api import create_app
from xnids.live.service import DetectorService, SourceData, parse_action
from xnids.models import zoo
from xnids.models.bundle import Bundle
from xnids.models.preprocess import Preprocessor

NAMES = ["a", "b", "c", "d"]


def _domain(n, shift, seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    X[y == 1, 0] += 3.0
    X[:, 1:] = X[:, 1:] * (1 + shift) + 2 * shift
    return pl.DataFrame(np.abs(X) * 10, schema=NAMES), y


class FakeSelector:
    actions_ = ["adabn", "tent", "wait"]
    margin = 0.0

    def choose(self, feats, actions=None):
        return ("adabn", 0.12, 0.03) if feats["ks_max"] > 0.2 else ("wait", 0.0, 0.0)


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    Xs, ys = _domain(6000, 0.0, 0)
    Xv, yv = _domain(3000, 0.0, 1)
    Xc, yc = _domain(2000, 0.0, 3)
    pre = Preprocessor().fit(Xs)
    m = zoo.build("mlp", {"max_epochs": 5, "batch_size": 256}, seed=0, device="cpu").fit(
        pre.transform(Xs), ys, pre.transform(Xv), yv)
    thr = metrics.threshold_at_dr(yv, m.score(pre.transform(Xv)), 0.95)
    d = tmp_path_factory.mktemp("bundle") / "toy-v1"
    Bundle(m, pre, thr, NAMES, {"version": "toy-v1", "track": "toy"}).save(d)
    return d, SourceData((Xs, ys), (Xv, yv), (Xc, yc)), (Xv, yv)


def _cfg(tmp_path):
    return {"live_dir": str(tmp_path / "live"), "candidates_dir": str(tmp_path / "cand"), "buffer_rows": 5000,
            "min_adapt_rows": 500, "store_scores": True, "actions": ["adabn", "tent", "fewshot(budget=200,rule=random)"],
            "label_oracle": None, "metrics_window_s": 60,
            "gates": {"canary_dr_ratio": 0.9, "canary_fpr_factor": 2.0, "canary_fpr_abs": 0.02,
                      "max_param_change": 0.5, "attack_rate": [0.001, 0.95],
                      "max_rate_ratio": 3.0}}


@pytest.fixture
def client(parts, tmp_path):
    bundle_dir, src, _ = parts
    Xt, yt = _domain(3000, 1.0, 2)
    keys = [f"k{i}" for i in range(Xt.height)]
    truth = dict(zip(keys, yt.tolist(), strict=True))
    svc = DetectorService(_cfg(tmp_path), initial_bundle=bundle_dir, source=src,
                          oracle=lambda ks: np.array([truth[k] for k in ks]), selector=FakeSelector())
    return TestClient(create_app(svc)), svc, Xt, keys


def _post_rows(c, X, keys):
    return c.post("/score", json=[{"flow_key": k, "features": r} for k, r in zip(keys, X.to_dicts(), strict=True)])


# ---------------------------------------------------------------- the Build Guide's three required behaviours

def test_score_rejects_bad_schema_with_422(client):
    c, _, X, keys = client
    assert _post_rows(c, X.drop("d").head(3), keys[:3]).status_code == 422                 # missing feature
    assert _post_rows(c, X.with_columns(e=pl.lit(1.0)).head(3), keys[:3]).status_code == 422  # extra feature
    assert c.post("/score", json=[]).status_code == 422
    assert c.post("/score/columns", json={"flow_keys": ["x"], "columns": {"a": [1.0]}}).status_code == 422
    assert _post_rows(c, X.head(3), keys[:3]).status_code == 200


def test_promote_needs_approval_and_passing_gates(client):
    c, svc, X, keys = client
    assert c.post("/models/promote", json={"approved_by": "ana", "reason": "x"}).status_code == 409   # no candidate
    _post_rows(c, X.head(1000), keys[:1000])
    assert c.post("/adapt", json={"action": "adabn"}).status_code == 200
    assert c.post("/models/promote", json={"reason": "looks good"}).status_code == 422             # no approver
    assert c.post("/models/promote", json={"approved_by": "", "reason": "x"}).status_code == 422
    assert c.post("/models/promote", json={"approved_by": "ana", "reason": ""}).status_code == 422
    assert c.get("/models").json()["active"]["version"] == "toy-v1"                               # still original
    # a candidate whose gates failed cannot be promoted, and the refusal is audited
    svc.registry.state["candidate"]["gates"] = {"canary_dr": {"passed": False}, "all_passed": False}
    r = c.post("/models/promote", json={"approved_by": "ana", "reason": "x"})
    assert r.status_code == 409 and "canary_dr" in r.json()["detail"]
    assert any(a["kind"] == "promote_refused" for a in c.get("/audit").json())


def test_promote_then_rollback_restores_the_version(client):
    c, _, X, keys = client
    _post_rows(c, X.head(1000), keys[:1000])
    cand = c.post("/adapt", json={"action": "adabn", "requested_by": "ana"}).json()["candidate"]
    assert c.get("/models").json()["candidate"]["version"] == cand
    r = c.post("/models/promote", json={"approved_by": "ana", "reason": "FPR down on segment B"})
    assert r.status_code == 200 and r.json() == {"active": cand, "previous": "toy-v1"}
    m = c.get("/models").json()
    assert m["active"]["version"] == cand and m["candidate"] is None and m["history"] == ["toy-v1"]
    r = c.post("/models/rollback", json={"requested_by": "ana", "reason": "demo"})
    assert r.status_code == 200 and r.json()["active"] == "toy-v1"
    assert c.get("/models").json()["active"]["version"] == "toy-v1"
    assert c.post("/models/rollback", json={"requested_by": "ana", "reason": "again"}).status_code == 409
    kinds = [a["kind"] for a in c.get("/audit").json()]
    assert {"adapt", "gate_check", "promote", "rollback"} <= set(kinds)


# ---------------------------------------------------------------- scoring, drift, adapt, registry

def test_score_returns_active_and_candidate_side_by_side(client):
    c, svc, X, keys = client
    r = _post_rows(c, X.head(5), keys[:5]).json()
    assert set(r) >= {"active"} and "candidate" not in r and len(r["active"]["scores"]) == 5
    _post_rows(c, X.head(1000), keys[:1000])
    c.post("/adapt", json={"action": "adabn"})
    cols = {k: v.to_list() for k, v in X.head(7).to_dict().items()}
    r = c.post("/score/columns", json={"flow_keys": keys[:7], "columns": cols}).json()
    assert len(r["active"]["alerts"]) == len(r["candidate"]["alerts"]) == 7
    assert r["active"]["version"] != r["candidate"]["version"]
    n = svc.store.query("SELECT role, COUNT(*) AS n FROM scores GROUP BY role")
    assert {x["role"]: x["n"] for x in n}["candidate"] == 7
    m = c.get("/metrics").json()
    assert m["flows"] >= 1012 and m["latency_ms"]["p50"] is not None


def test_drift_report_is_stored_with_a_recommendation(client, parts):
    c, svc, X, _ = client
    _, _, (Xv, yv) = parts
    mon = DriftMonitor(svc.registry.get("active"), Xv, yv, {"reference": {"n": 2000}, "mmd": {"n_perm": 50}})
    rep = mon.process(X.head(2000))
    r = c.post("/drift/report", json=__import__("json").loads(rep.to_json()))
    assert r.status_code == 200
    rec = r.json()["recommendation"]
    assert rec["selector_pick"] == "adabn"                          # ks_max on this shift is large
    assert rec["action"] == ("adabn" if rep.recommend else "wait")
    latest = c.get("/drift/latest").json()
    assert latest["window_id"] == 0 and latest["report"]["top_features"] == rep.top_features
    assert c.post("/drift/report", json={"window_id": 1}).status_code == 422


def test_adapt_rules(client):
    c, _, X, keys = client
    assert c.post("/adapt", json={"action": "adabn"}).status_code == 409                # empty buffer
    _post_rows(c, X.head(1000), keys[:1000])
    assert c.post("/adapt", json={"action": "dann"}).status_code == 409                 # not enabled
    assert c.post("/adapt", json={}).status_code == 409                                 # no recommendation yet
    r = c.post("/adapt", json={"action": "fewshot(budget=200,rule=random)"})
    assert r.status_code == 200 and r.json()["labels_used"] == 200                      # labels via the oracle
    assert set(r.json()["gates"]) >= {"canary_dr", "canary_fpr", "param_change", "attack_rate", "all_passed"}
    assert c.post("/models/reject", json={"requested_by": "ana", "reason": "no"}).json()["rejected"]
    assert c.get("/models").json()["candidate"] is None


def test_gate_checks_catch_a_silent_candidate(parts):
    bundle_dir, src, _ = parts
    a = Bundle.load(bundle_dir)
    silent = Bundle.load(bundle_dir)
    silent.threshold = 1.01                                    # never alerts
    X, _ = _domain(1000, 1.0, 5)
    g = reg.gate_checks(a, silent, src.canary, X, _cfg_gates())
    assert g["canary_dr"]["passed"] is False and g["attack_rate"]["passed"] is False and not g["all_passed"]
    assert g["param_change"]["value"] == 0.0 and g["param_change"]["passed"] is True
    assert g["canary_fpr"]["passed"] is True                     # silent = no false positives
    noisy = Bundle.load(bundle_dir)
    noisy.threshold = -0.01                                     # alerts on everything
    g = reg.gate_checks(a, noisy, src.canary, X, _cfg_gates())
    assert g["canary_fpr"]["passed"] is False and g["canary_dr"]["passed"] is True and not g["all_passed"]
    assert reg.gate_checks(a, Bundle.load(bundle_dir), src.canary, X, _cfg_gates())["all_passed"]


def _cfg_gates():
    return _cfg(__import__("pathlib").Path("/nonexistent"))["gates"]


def test_registry_state_survives_restart(parts, tmp_path):
    bundle_dir, src, _ = parts
    r1 = reg.Registry(tmp_path / "live", tmp_path / "cand", bundle_dir)
    v = r1.set_candidate(Bundle.load(bundle_dir), {"all_passed": True}, "ana", "adabn")
    r1.promote("ana", "ok")
    r2 = reg.Registry(tmp_path / "live", tmp_path / "cand", bundle_dir)
    assert r2.version("active") == v and r2.summary()["history"] == ["toy-v1"]
    assert r2.get("active").threshold == pytest.approx(r1.get("active").threshold)


def test_model_prefixed_action_starts_from_that_model(parts, tmp_path):
    bundle_dir, src, _ = parts
    Xs, ys = src.train
    pre = Preprocessor().fit(Xs)
    xm = zoo.build("xgb", {"n_estimators": 30, "max_depth": 3}, seed=0, device="cpu").fit(
        pre.transform(Xs), ys, pre.transform(src.val[0]), src.val[1])
    xb = Bundle(xm, pre, 0.5, NAMES, {"version": "toy-xgb"})
    Xt, yt = _domain(3000, 1.0, 2)
    keys = [f"k{i}" for i in range(Xt.height)]
    truth = dict(zip(keys, yt.tolist(), strict=True))
    cfg = _cfg(tmp_path) | {"actions": ["xgb:fewshot(budget=200,rule=random)"]}
    svc = DetectorService(cfg, initial_bundle=bundle_dir, source=src, bases={"xgb": xb},
                          oracle=lambda ks: np.array([truth[k] for k in ks]), selector=FakeSelector())
    c = TestClient(create_app(svc))
    _post_rows(c, Xt.head(1500), keys[:1500])
    r = c.post("/adapt", json={"action": "xgb:fewshot(budget=200,rule=random)"}).json()
    assert r["candidate"].startswith("toy-xgb+fewshot_random_200@")
    assert r["gates"]["param_change"]["passed"] is None                 # MLP vs XGBoost: not comparable
    assert svc.registry.get("candidate").model.name == "xgb"


def test_parse_action():
    assert parse_action("adabn") == ("adabn", {})
    assert parse_action("fewshot(budget=200,rule=random)") == ("fewshot", {"budget": 200, "rule": "random"})
    assert parse_action("coral(lam=1.0)") == ("coral", {"lam": 1.0})
