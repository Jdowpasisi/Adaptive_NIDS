"""C14 detector service (the logic behind live/api.py; usable without HTTP, e.g. in tests and C17).

    svc = DetectorService(config.load("configs/live.yaml"))
    svc.score(flow_keys, X)            # active + candidate scores / alerts; stored; flows buffered
    svc.drift_report(report_dict)      # stored; selector recommendation
    svc.adapt(action, requested_by)    # adapter on the buffered flows -> candidate + gate checks (never promoted)
    svc.promote(approved_by, reason) / svc.reject(by, reason) / svc.rollback(by, reason)
"""

import collections
import itertools
import pickle
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from xnids import adapt as adapters
from xnids.adapt.base import AdaptContext
from xnids.drift.monitor import DriftReport
from xnids.live.registry import Registry, RegistryError, gate_checks
from xnids.live.store import Store
from xnids.utils import paths

ROLES = ("active", "candidate")


class ServiceError(RuntimeError):
    """A request that cannot be served in the current state (-> HTTP 409)."""


@dataclass
class SourceData:
    train: tuple[pl.DataFrame, np.ndarray]
    val: tuple[pl.DataFrame, np.ndarray]
    canary: tuple[pl.DataFrame, np.ndarray]


def load_source(cfg: dict) -> SourceData:
    """Source train / val samples for adapters, and the fixed canary set from source TEST (never adapted on)."""
    from xnids.eval.harness import load_split

    m, s = cfg["model"], cfg["source_sample"]
    tr = load_split(m["source"], m["track"], "train", s["rows"], 1000, s["seed"])
    va = load_split(m["source"], m["track"], "val", s["rows"], 1000, s["seed"])
    te = load_split(m["source"], m["track"], "test")
    c = cfg["canary"]
    rng = np.random.default_rng(c["seed"])
    att, ben = np.flatnonzero(te.y == 1), np.flatnonzero(te.y == 0)
    idx = np.sort(np.concatenate([rng.choice(att, min(c["attacks"], len(att)), replace=False),
                                  rng.choice(ben, min(c["benign"], len(ben)), replace=False)]))
    return SourceData((tr.X, tr.y), (va.X, va.y), (te.X[idx], te.y[idx]))


def parse_action(action: str) -> tuple[str, dict]:
    """'fewshot(budget=200,rule=random)' -> ('fewshot', {'budget': 200, 'rule': 'random'})."""
    m = re.fullmatch(r"([a-z_]+)(?:\((.*)\))?", action.strip())
    if not m:
        raise ValueError(f"bad action {action!r}")
    params = {}
    for kv in filter(None, (m.group(2) or "").split(",")):
        k, v = kv.split("=", 1)
        try:
            params[k.strip()] = int(v) if re.fullmatch(r"-?\d+", v.strip()) else float(v)
        except ValueError:
            params[k.strip()] = v.strip()
    return m.group(1), params


class DetectorService:
    def __init__(self, cfg: dict, initial_bundle: Path | None = None, source: SourceData | None = None,
                 oracle=None, selector=None, bases: dict | None = None) -> None:
        self.cfg = cfg
        live_dir = paths.REPO / cfg["live_dir"] if not Path(cfg["live_dir"]).is_absolute() else Path(cfg["live_dir"])
        cand_dir = Path(cfg["candidates_dir"])
        cand_dir = cand_dir if cand_dir.is_absolute() else paths.REPO / cand_dir
        if initial_bundle is None:
            from xnids.models.bundle import bundle_for

            m = cfg["model"]
            initial_bundle = bundle_for(m["name"], m["track"], m["source"], m["seed"]).path
        self.registry = Registry(live_dir, cand_dir, initial_bundle)
        self.store = Store(live_dir / "live.db")
        self.features = list(self.registry.get("active").features)
        self._source = source
        self._oracle = oracle
        self._bases: dict = dict(bases or {})               # model family -> source bundle ("xgb:..." actions)
        self.selector = selector if selector is not None else self._load_selector()
        self.lock = threading.RLock()
        self.buffer: collections.deque = collections.deque()      # (flow_keys, DataFrame) chunks, newest last
        self.buffer_rows = 0
        self.batch_ids = itertools.count(int(time.time() * 1000))
        self.events: collections.deque = collections.deque(maxlen=200_000)   # (t, n, alerts_active, alerts_cand)
        self.latency: collections.deque = collections.deque(maxlen=20_000)  # (t, n, seconds)
        self.store.log("service_start", version_to=self.registry.version("active"))

    # ------------------------------------------------------------------ lazy heavy pieces
    @property
    def source(self) -> SourceData:
        if self._source is None:
            self._source = load_source(self.cfg)
        return self._source

    @property
    def oracle(self):
        if self._oracle is None and self.cfg.get("label_oracle") == "replay":
            from xnids.live.replay_labels import ReplayOracle

            self._oracle = ReplayOracle()
        return self._oracle

    @staticmethod
    def _load_selector():
        p = paths.MODELS / "selector" / "selector.pkl"
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------ scoring
    def check_schema(self, columns: list[str]) -> None:
        missing, extra = set(self.features) - set(columns), set(columns) - set(self.features)
        if missing or extra:
            raise ValueError(f"schema mismatch: missing {sorted(missing)[:10]}, unexpected {sorted(extra)[:10]}")

    def score(self, flow_keys: list[str], X: pl.DataFrame) -> dict:
        self.check_schema(X.columns)
        X = X.select([pl.col(c).cast(pl.Float32) for c in self.features])
        now, batch = time.time(), next(self.batch_ids)
        out, rows, alerts = {}, [], {}
        with self.registry.lock:
            for role in ROLES:
                b = self.registry.get(role)
                if b is None:
                    continue
                s = b.score(X)
                a = s >= b.threshold
                v = self.registry.version(role)
                out[role] = {"version": v, "scores": s.round(6).tolist(), "alerts": a.astype(int).tolist(),
                             "threshold": b.threshold}
                alerts[role] = int(a.sum())
                if self.cfg.get("store_scores", True):
                    rows += list(zip(itertools.repeat(now), itertools.repeat(batch), flow_keys,
                                     itertools.repeat(role), itertools.repeat(v), s.tolist(),
                                     a.astype(int).tolist(), strict=False))
        if rows:
            self.store.add_scores(rows)
        with self.lock:
            self.buffer.append((list(flow_keys), X))
            self.buffer_rows += X.height
            while self.buffer_rows - self.buffer[0][1].height >= self.cfg["buffer_rows"]:
                self.buffer_rows -= self.buffer.popleft()[1].height
            self.events.append((now, X.height, alerts.get("active", 0), alerts.get("candidate", 0)))
        return {"batch_id": batch, "n": X.height, **out}

    def recent(self) -> tuple[list[str], pl.DataFrame]:
        with self.lock:
            if not self.buffer:
                return [], pl.DataFrame(schema={c: pl.Float32 for c in self.features})
            keys = list(itertools.chain.from_iterable(k for k, _ in self.buffer))
            X = pl.concat([x for _, x in self.buffer])
        n = self.cfg["buffer_rows"]
        return keys[-n:], X.tail(n)

    def record_latency(self, n: int, seconds: float) -> None:
        self.latency.append((time.time(), n, seconds))

    # ------------------------------------------------------------------ drift + recommendation
    def drift_report(self, report: dict) -> dict:
        rep = DriftReport(**report)
        feats = rep.features()
        actions = list(self.cfg["actions"])
        if self.selector is None:
            rec = {"action": "wait", "gain": 0.0, "std": 0.0, "note": "no selector model (models/selector/)"}
        else:
            known = [a for a in actions if a in self.selector.actions_]
            a, g, sd = self.selector.choose(feats, known)
            rec = {"action": a, "gain": g, "std": sd, "selector_pick": a, "candidates": known}
            if not rep.recommend:
                rec.update(action="wait", gain=0.0, std=0.0,
                           note="the monitor did not recommend acting (no combined drift + cost trigger)")
        rid = self.store.add_drift(report, rec, self.registry.version("active"))
        if rep.recommend:
            self.store.log("drift_alert", "monitor", self.registry.version("active"), reason=rep.message,
                           window_id=rep.window_id, top_features=rep.top_features, drift_id=rid)
            self.store.log("recommendation", "selector", self.registry.version("active"), reason=rec["action"],
                           gain=rec["gain"], std=rec["std"], drift_id=rid)
        return {"drift_id": rid, "recommend": rep.recommend, "message": rep.message, "features": feats,
                "recommendation": rec}

    def latest_drift(self) -> dict | None:
        return self.store.latest_drift()

    # ------------------------------------------------------------------ adapt / promote / reject / rollback
    def adapt(self, action: str | None = None, requested_by: str = "analyst", seed: int = 0) -> dict:
        if action is None:
            d = self.latest_drift()
            action = d["rec_action"] if d else None
            if not action or action == "wait":
                raise ServiceError("no action given and the latest recommendation is 'wait' (or there is none)")
        if action not in self.cfg["actions"]:
            raise ServiceError(f"action {action!r} is not enabled; enabled: {self.cfg['actions']}")
        keys, recent = self.recent()
        n = self.cfg.get("adapt_rows", self.cfg["buffer_rows"])
        keys, pool = keys[-n:], recent.tail(n)
        if pool.height < self.cfg["min_adapt_rows"]:
            raise ServiceError(f"only {pool.height} buffered flows; /adapt needs >= {self.cfg['min_adapt_rows']}")
        base_model, _, act = action.rpartition(":")         # "xgb:fewshot(...)" = start from the XGBoost bundle
        name, params = parse_action(act)
        adapter = adapters.get(name, **params)
        ctx_params = {}
        if adapter.needs_labels:
            if self.oracle is None:
                raise ServiceError(f"{action} needs labels and no label oracle is configured")
            oracle = self.oracle
            ctx_params["pool_labels"] = lambda idx: oracle([keys[i] for i in idx])
        src = self.source
        ctx = AdaptContext(src.train, src.val, pool, seed=seed, params=ctx_params)
        active = self.registry.get("active")
        start = self.base_bundle(base_model) if base_model else active
        t0 = time.time()
        try:
            cand = adapter.adapt(start, ctx)
        except Exception as e:                                           # logged, then surfaced as 409
            self.store.log("adapt_failed", requested_by, self.registry.version("active"), reason=action,
                           error=repr(e))
            raise ServiceError(f"{action} failed: {e!r}") from e
        gates = gate_checks(active, cand, src.canary, recent, self.cfg["gates"])
        cand.meta.update({"adapted_on_rows": pool.height, "action": action, "adapt_seconds": time.time() - t0})
        v = self.registry.set_candidate(cand, gates, requested_by, action)
        self.store.log("adapt", requested_by, self.registry.version("active"), v, reason=action,
                       rows=pool.height, seconds=time.time() - t0, labels_used=cand.meta.get("labels_used", 0))
        self.store.log("gate_check", "system", self.registry.version("active"), v,
                       reason="passed" if gates["all_passed"] else "FAILED", **gates)
        return {"candidate": v, "action": action, "rows": pool.height, "gates": gates,
                "labels_used": cand.meta.get("labels_used", 0)}

    def promote(self, approved_by: str, reason: str) -> dict:
        try:
            old, new = self.registry.promote(approved_by, reason)
        except RegistryError as e:
            self.store.log("promote_refused", approved_by or "?", self.registry.version("active"),
                           self.registry.version("candidate"), reason=str(e))
            raise ServiceError(str(e)) from e
        self.store.log("promote", approved_by, old, new, reason=reason)
        return {"active": new, "previous": old}

    def reject(self, rejected_by: str, reason: str) -> dict:
        try:
            v = self.registry.reject()
        except RegistryError as e:
            raise ServiceError(str(e)) from e
        self.store.log("reject", rejected_by, self.registry.version("active"), v, reason=reason)
        return {"rejected": v}

    def rollback(self, requested_by: str, reason: str) -> dict:
        try:
            cur, prev = self.registry.rollback()
        except RegistryError as e:
            raise ServiceError(str(e)) from e
        self.store.log("rollback", requested_by, cur, prev, reason=reason)
        return {"active": prev, "rolled_back": cur}

    def base_bundle(self, model: str):
        """The source-trained bundle of another model family on the same track (C10/C11 'model:action' actions)."""
        if model not in self._bases:
            from xnids.models.bundle import bundle_for

            m = self.cfg["model"]
            self._bases[model] = bundle_for(model, m["track"], m["source"], m["seed"])
        return self._bases[model]

    # ------------------------------------------------------------------ monitoring
    def metrics(self) -> dict:
        w = self.cfg["metrics_window_s"]
        now = time.time()
        ev = [e for e in list(self.events) if e[0] >= now - w]
        lat = [s for t, _, s in list(self.latency) if t >= now - w]
        span = max(now - ev[0][0], 1.0) if ev else w
        return {"window_s": w, "flows": sum(e[1] for e in ev), "flows_per_s": sum(e[1] for e in ev) / span,
                "alerts_per_min": {"active": sum(e[2] for e in ev) / span * 60,
                                   "candidate": sum(e[3] for e in ev) / span * 60},
                "latency_ms": {"p50": float(np.percentile(lat, 50) * 1e3) if lat else None,
                               "p99": float(np.percentile(lat, 99) * 1e3) if lat else None, "requests": len(lat)},
                "buffer_rows": self.buffer_rows,
                "versions": {r: self.registry.version(r) for r in ROLES}}

    def health(self) -> dict:
        return {"status": "ok", "versions": {r: self.registry.version(r) for r in ROLES},
                "n_features": len(self.features), "selector": self.selector is not None,
                "actions": self.cfg["actions"]}
