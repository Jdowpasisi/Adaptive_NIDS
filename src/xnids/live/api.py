"""C14 detector API (FastAPI). Thin HTTP layer over xnids.live.service.DetectorService.

    make api            # uvicorn xnids.live.api:app on :8000 (config: $DG_LIVE_CONFIG, default configs/live.yaml)

Endpoints (Build Guide C14): POST /score, POST /score/columns (same, columnar: faster for C15 micro-batches),
GET /health, GET /models, POST /drift/report, GET /drift/latest, POST /adapt, POST /models/promote,
POST /models/reject, POST /models/rollback, GET /metrics, GET /audit.
A candidate is never promoted automatically: /models/promote needs approved_by + reason and passing gate checks.
"""

import os
import time

import polars as pl
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from xnids.live.service import DetectorService, ServiceError
from xnids.utils import config, paths


class Flow(BaseModel):
    flow_key: str
    features: dict[str, float]


class ColumnBatch(BaseModel):
    flow_keys: list[str]
    columns: dict[str, list[float | None]]


class AdaptRequest(BaseModel):
    action: str | None = None                  # default: the latest recommendation
    requested_by: str = "analyst"
    seed: int = 0


class Decision(BaseModel):
    approved_by: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ActionRequest(BaseModel):
    requested_by: str = Field(min_length=1)
    reason: str = Field(min_length=1)


def create_app(service: DetectorService | None = None, cfg_path: str | None = None) -> FastAPI:
    app = FastAPI(title="DriftGuard detector", version="1.0")
    state = {"svc": service}

    def svc() -> DetectorService:
        if state["svc"] is None:
            state["svc"] = DetectorService(config.load(cfg_path or os.environ.get(
                "DG_LIVE_CONFIG", str(paths.CONFIGS / "live.yaml"))))
        return state["svc"]

    def conflict(e: Exception):
        raise HTTPException(status_code=409, detail=str(e)) from e

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        if request.url.path.startswith("/score") and resp.status_code == 200:
            n = int(resp.headers.get("x-flows", "0"))
            svc().record_latency(n, time.perf_counter() - t0)
        return resp

    def _score(keys: list[str], X: pl.DataFrame):
        from fastapi.responses import JSONResponse

        try:
            svc().check_schema(X.columns)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        out = svc().score(keys, X)
        return JSONResponse(out, headers={"x-flows": str(len(keys))})

    @app.post("/score")
    def score(batch: list[Flow]):
        if not batch:
            raise HTTPException(status_code=422, detail="empty batch")
        cols = set(batch[0].features)
        if any(set(f.features) != cols for f in batch):
            raise HTTPException(status_code=422, detail="flows in one batch must carry the same feature names")
        return _score([f.flow_key for f in batch], pl.DataFrame([f.features for f in batch]))

    @app.post("/score/columns")
    def score_columns(batch: ColumnBatch):
        n = len(batch.flow_keys)
        if not n or any(len(v) != n for v in batch.columns.values()):
            raise HTTPException(status_code=422, detail="every column must have one value per flow_key")
        return _score(batch.flow_keys, pl.DataFrame(batch.columns))

    @app.get("/health")
    def health():
        return svc().health()

    @app.get("/models")
    def models():
        return svc().registry.summary()

    @app.post("/drift/report")
    def drift_report(report: dict):
        try:
            return svc().drift_report(report)
        except TypeError as e:
            raise HTTPException(status_code=422, detail=f"not a DriftReport: {e}") from e

    @app.get("/drift/latest")
    def drift_latest():
        return svc().latest_drift() or {}

    @app.post("/adapt")
    def adapt_(req: AdaptRequest):
        try:
            return svc().adapt(req.action, req.requested_by, req.seed)
        except ServiceError as e:
            conflict(e)

    @app.post("/models/promote")
    def promote(d: Decision):
        try:
            return svc().promote(d.approved_by, d.reason)
        except ServiceError as e:
            conflict(e)

    @app.post("/models/reject")
    def reject(d: ActionRequest):
        try:
            return svc().reject(d.requested_by, d.reason)
        except ServiceError as e:
            conflict(e)

    @app.post("/models/rollback")
    def rollback(d: ActionRequest):
        try:
            return svc().rollback(d.requested_by, d.reason)
        except ServiceError as e:
            conflict(e)

    @app.get("/metrics")
    def metrics():
        return svc().metrics()

    @app.get("/audit")
    def audit(limit: int = 100):
        return svc().store.actions(limit)

    return app


app = create_app()
