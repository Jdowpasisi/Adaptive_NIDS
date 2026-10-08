"""C14 latency benchmark: p50 / p99 of POST /score and /score/columns for batches of 1, 100 and 1,000 demo flows,
over real HTTP to a uvicorn server (started here, one worker, scores stored in SQLite as in the demo).

    python scripts/bench_api.py [--port 8765] [--requests 300 200 50]

Client-side wall time per request (serialise + HTTP + validate + score active model + store + respond). Writes
reports/tables/c14_latency.csv.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import httpx
import numpy as np
import polars as pl

from xnids.data import schema
from xnids.utils import paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--requests", type=int, nargs=3, default=[300, 200, 50], help="per batch size 1 / 100 / 1000")
    args = ap.parse_args()
    live = paths.REPO / "data/live/bench"
    shutil.rmtree(live, ignore_errors=True)
    cfg = live / "live.yaml"
    live.mkdir(parents=True)
    base = (paths.CONFIGS / "live.yaml").read_text()
    cfg.write_text(base.replace("live_dir: data/live ", f"live_dir: {live} ")
                   .replace("candidates_dir: models/live ", f"candidates_dir: {live}/candidates "))
    env = os.environ | {"DG_LIVE_CONFIG": str(cfg), "PYTHONPATH": str(paths.REPO / "src")}
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "xnids.live.api:app", "--port", str(args.port),
                            "--log-level", "warning"], env=env)
    url = f"http://127.0.0.1:{args.port}"
    try:
        with httpx.Client(base_url=url, timeout=60) as c:
            for _ in range(120):
                try:
                    if c.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    time.sleep(0.5)
            feats = schema.features("nfs")
            flows = pl.read_parquet(paths.REPLAY / "demo_flows.parquet").head(60_000)
            keys = [f"bench-{i}" for i in range(flows.height)]
            rows = flows.select(feats).cast(pl.Float64).to_dicts()
            out = []
            for n, reps in zip((1, 100, 1000), args.requests, strict=True):
                for endpoint in ("/score", "/score/columns"):
                    lat = []
                    for r in range(reps + 5):                         # 5 warm-up requests, not counted
                        i = (r * n) % (flows.height - n)
                        if endpoint == "/score":
                            body = [{"flow_key": keys[j], "features": rows[j]} for j in range(i, i + n)]
                        else:
                            sl = flows.slice(i, n)
                            body = {"flow_keys": keys[i:i + n],
                                    "columns": {f: sl[f].cast(pl.Float64).to_list() for f in feats}}
                        t0 = time.perf_counter()
                        resp = c.post(endpoint, json=body)
                        dt = time.perf_counter() - t0
                        resp.raise_for_status()
                        if r >= 5:
                            lat.append(dt)
                    lat = np.array(lat) * 1e3
                    row = {"endpoint": endpoint, "batch": n, "requests": reps, "p50_ms": np.percentile(lat, 50),
                           "p99_ms": np.percentile(lat, 99), "mean_ms": lat.mean(),
                           "flows_per_s": n / (lat.mean() / 1e3)}
                    out.append(row)
                    print({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
            m = c.get("/metrics").json()
            print("server-side /metrics (last 60 s):", m["latency_ms"])
    finally:
        srv.terminate()
        srv.wait(10)
    pl.DataFrame(out).write_csv(paths.TABLES / "c14_latency.csv")


if __name__ == "__main__":
    main()
