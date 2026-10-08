"""C15 replay runner: flows -> detector API at a controlled rate, windows -> calibrated drift monitor -> API.

Sources (all yield flows in emission order):
  file     NFStream on a PCAP file (the frozen extractor), as fast as it extracts
  live     NFStream on a network interface (e.g. veth1 fed by tcpreplay; needs CAP_NET_RAW)
  parquet  pre-extracted flows (data/replay/demo_flows.parquet): no extraction cost, for the load test

Every `micro_batch_ms` the runner sends what is available (up to the rate budget) to POST /score/columns in requests
of at most `max_batch` flows. Every `window` flows go to a monitor thread (the C15 calibrated monitor), whose
DriftReport is posted to /drift/report, so drift statistics never block scoring. Optional rehearsal policy
(--auto-adapt / --auto-approve) calls /adapt on the first persisted alert and /models/promote when the gates pass;
the real demo leaves both to the analyst (C16).

Per run, <run_dir>/ gets batches.csv (latency, alerts, emission lag), windows.csv (drift + evaluation overlay from the
replay labels where flows have them) and flows.parquet (flow keys in emission order, for file/live parity checks).
"""

import collections
import json
import queue
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from xnids.data import schema
from xnids.live import extract, replay_labels


class FlowSource(threading.Thread):
    """Producer: flows into a thread-safe buffer of tidy polars frames."""

    def __init__(self, kind: str, source: str | Path, limit: int | None = None, chunk: int = 256) -> None:
        super().__init__(daemon=True)
        self.kind, self.source, self.limit, self.chunk = kind, source, limit, chunk
        self.buf: collections.deque = collections.deque()
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.error: BaseException | None = None
        self.produced = 0

    def _put(self, df: pl.DataFrame) -> None:
        with self.lock:
            self.buf.append(df)
        self.produced += df.height

    def run(self) -> None:
        try:
            if self.kind == "parquet":
                df = pl.read_parquet(self.source)
                df = df.head(self.limit) if self.limit else df
                for i in range(0, df.height, 10_000):
                    self._put(df.slice(i, 10_000))
            else:
                rows, cols, last = [], None, time.monotonic()
                for cols, vals in extract.stream_flows(self.source):
                    rows.append(vals)
                    # flush in small chunks, and at least every 100 ms when flows trickle in (live mode)
                    if len(rows) >= self.chunk or time.monotonic() - last > 0.1:
                        self._put(pl.from_pandas(extract.tidy_rows(cols, rows)))
                        rows, last = [], time.monotonic()
                    if self.limit and self.produced + len(rows) >= self.limit:
                        break
                if rows:
                    self._put(pl.from_pandas(extract.tidy_rows(cols, rows)))
        except BaseException as e:  # noqa: BLE001  (re-raised by the runner)
            self.error = e
        finally:
            self.done.set()

    def take(self, n: int) -> pl.DataFrame | None:
        out, got = [], 0
        with self.lock:
            while self.buf and got < n:
                df = self.buf.popleft()
                if got + df.height > n:
                    self.buf.appendleft(df.slice(n - got))
                    df = df.slice(0, n - got)
                out.append(df)
                got += df.height
        return pl.concat(out, how="vertical_relaxed") if out else None

    def exhausted(self) -> bool:
        with self.lock:
            return self.done.is_set() and not self.buf


class ReplayRunner:
    def __init__(self, cfg: dict, api: str, run_dir: Path, rate: float | None = None, monitor=None,
                 auto_adapt: bool = False, auto_approve: bool = False, adapt_action: str | None = None,
                 labels: bool = True, idle_stop: float | None = None, client: httpx.Client | None = None,
                 features: list[str] | None = None) -> None:
        self.cfg, self.api, self.rate = cfg, api.rstrip("/"), rate
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.auto_adapt, self.auto_approve, self.adapt_action = auto_adapt, auto_approve, adapt_action
        self.features = features or schema.features(cfg["model"]["track"])
        self.client = client                            # tests inject FastAPI's TestClient (an httpx.Client)
        self.oracle = replay_labels.ReplayOracle() if labels else {}
        self.batches, self.windows, self.keys, self.flow_frames = [], [], [], []
        self.win_q: queue.Queue = queue.Queue()
        self.adapted = False
        self.max_end = 0
        self.idle_stop = idle_stop                     # live mode: stop after this many seconds without a flow

    # ---------------------------------------------------------------- monitor thread
    def _monitor_loop(self, client: httpx.Client) -> None:
        while True:
            item = self.win_q.get()
            if item is None:
                return
            wid, X, overlay = item
            t0 = time.perf_counter()
            rep = self.monitor.process(X)
            d = client.post("/drift/report", content=rep.to_json(),
                            headers={"content-type": "application/json"}).json()
            row = {"window": wid, "flows": X.height, "novelty": rep.novelty_share, "streak": rep.streak,
                   "recommend": rep.recommend, "recommendation": d["recommendation"]["action"],
                   "rec_gain": d["recommendation"]["gain"], "message": rep.message,
                   "monitor_s": time.perf_counter() - t0, "event": "", **overlay}
            if rep.recommend and self.auto_adapt and not self.adapted:
                action = self.adapt_action or (d["recommendation"]["action"]
                                               if d["recommendation"]["action"] != "wait" else None)
                if action:
                    self.adapted = True
                    r = client.post("/adapt", json={"action": action, "requested_by": "replay-runner"}, timeout=600)
                    res = r.json()
                    if r.status_code != 200:
                        row["event"] = f"adapt refused: {res.get('detail')}"
                    else:
                        ok = res["gates"]["all_passed"]
                        row["event"] = f"candidate {action}, gates {'pass' if ok else 'FAIL'}"
                        if ok and self.auto_approve:
                            p = client.post("/models/promote", json={"approved_by": "replay-runner (rehearsal)",
                                                                     "reason": "auto-approve rehearsal"})
                            row["event"] += "; promoted" if p.status_code == 200 else f"; promote {p.status_code}"
            self.windows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                              if k not in ("message",)}), flush=True)

    # ---------------------------------------------------------------- main loop
    def _send(self, client: httpx.Client, df: pl.DataFrame, win_buf: list) -> None:
        df = df.with_columns(replay_labels.flow_key_expr())
        keys = df["flow_key"].to_list()
        body = {"flow_keys": keys, "columns": {f: df[f].cast(pl.Float64).fill_nan(None).to_list()
                                                for f in self.features}}
        t0 = time.perf_counter()
        r = client.post("/score/columns", json=body)
        lat = time.perf_counter() - t0
        r.raise_for_status()
        out = r.json()
        end = df["end_timestamp"].to_numpy()
        self.max_end = max(self.max_end, int(end.max()))
        lag = (self.max_end - end) / 1e6                       # how long after its last packet a flow was emitted
        alerts = {role: np.array(out[role]["alerts"]) for role in ("active", "candidate") if role in out}
        self.batches.append({"t": time.time(), "n": df.height, "latency_s": lat,
                             **{f"alerts_{k}": int(v.sum()) for k, v in alerts.items()},
                             "lag_p50_s": float(np.median(lag)), "lag_p95_s": float(np.quantile(lag, 0.95)),
                             "active": out["active"]["version"]})
        self.keys += keys
        self.flow_frames.append(df.select("flow_key", "end_timestamp", *self.features))
        y = np.array([self.oracle.get(k, -1) for k in keys])
        win_buf.append((df.select(self.features), y, alerts))

    def _flush_window(self, win_buf: list, wid: int, force: bool = False) -> tuple[list, int]:
        n = sum(x[0].height for x in win_buf)
        W = self.cfg["monitor"]["window"]
        while n >= W or (force and n):
            X = pl.concat([x[0] for x in win_buf])
            y = np.concatenate([x[1] for x in win_buf])
            al = {r: np.concatenate([x[2].get(r, np.full(x[0].height, -1)) for x in win_buf])
                  for r in ("active", "candidate")}
            take = min(W, n)
            overlay = {"labelled": int((y[:take] >= 0).sum()), "benign": int((y[:take] == 0).sum())}
            for r, a in al.items():
                a, yy = a[:take], y[:take]
                if (a >= 0).any():
                    overlay[f"fpr_{r}"] = float(a[yy == 0].mean()) if (yy == 0).any() else None
                    overlay[f"dr_{r}"] = float(a[yy == 1].mean()) if (yy == 1).any() else None
            if self.monitor is not None:
                self.win_q.put((wid, X.slice(0, take), overlay))
            wid += 1
            rest = X.slice(take)
            win_buf = [(rest, y[take:], {r: a[take:] for r, a in al.items()})] if rest.height else []
            n -= take
        return win_buf, wid

    def run(self, source: FlowSource) -> dict:
        tick = self.cfg["micro_batch_ms"] / 1000
        max_batch = self.cfg["max_batch"]
        with self.client or httpx.Client(base_url=self.api, timeout=120) as client:
            mon_t = threading.Thread(target=self._monitor_loop, args=(client,), daemon=True)
            if self.monitor is not None:
                mon_t.start()
            source.start()
            t_start, allowance, win_buf, wid = time.monotonic(), 0.0, [], 0
            next_tick = t_start
            last_flow = time.monotonic()
            while not source.exhausted():
                if source.error:
                    raise source.error
                if self.idle_stop and source.produced and time.monotonic() - last_flow > self.idle_stop:
                    break
                if self.rate:
                    allowance = min(allowance + self.rate * tick, self.rate * 2)   # at most 2 s of backlog
                    budget = int(allowance)
                else:
                    budget = 10**9
                sent = 0
                while sent < budget:
                    df = source.take(min(max_batch, budget - sent))
                    if df is None:
                        break
                    self._send(client, df, win_buf)
                    sent += df.height
                    win_buf, wid = self._flush_window(win_buf, wid)
                allowance -= sent
                if sent:
                    last_flow = time.monotonic()
                if not self.rate:                       # top speed: only wait when nothing was available
                    if not sent:
                        time.sleep(0.005)
                    continue
                next_tick += tick
                wait = next_tick - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                elif wait < -1.0:                       # fell more than 1 s behind: do not burst to catch up
                    next_tick = time.monotonic()
            win_buf, wid = self._flush_window(win_buf, wid, force=True)
            elapsed = time.monotonic() - t_start
            if self.monitor is not None:
                self.win_q.put(None)
                mon_t.join()
        b = pl.DataFrame(self.batches)
        b.write_csv(self.run_dir / "batches.csv")
        if self.windows:
            pl.DataFrame(sorted(self.windows, key=lambda r: r["window"]), strict=False).write_csv(
                self.run_dir / "windows.csv")
        pl.concat(self.flow_frames, how="vertical_relaxed").with_row_index("order").write_parquet(
            self.run_dir / "flows.parquet")                  # emitted flows + features (file / live parity)
        lat = b["latency_s"].to_numpy() * 1e3
        summary = {"flows": int(b["n"].sum()), "seconds": elapsed, "flows_per_s": float(b["n"].sum() / elapsed),
                   "requests": b.height, "p50_ms": float(np.percentile(lat, 50)), "p99_ms": float(np.percentile(lat, 99)),
                   "lag_p50_s": float(b["lag_p50_s"].median()), "lag_p95_s": float(b["lag_p95_s"].quantile(0.95)),
                   "windows": len(self.windows), "target_rate": self.rate}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        return summary
