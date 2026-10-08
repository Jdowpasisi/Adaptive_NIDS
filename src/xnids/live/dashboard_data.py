"""C16 dashboard data: the detector API, its live.db (scores, drift reports) and the replay runner's status file.

Kept out of dashboard/app.py so it can be tested without Streamlit. Series are keyed by model VERSION, not role:
"original" = the first deployed model (scored as active, then as the shadow after a promotion), "adapted" = every
other version (scored as candidate, then as active), so each line is continuous across promote / rollback.
The evaluation overlay (FPR / DR) uses the replay's ground truth: known only because this is a replay.
"""

import json
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from xnids.utils import paths


class DashboardData:
    def __init__(self, api: str | None = None) -> None:
        self.api = (api or os.environ.get("DG_API", "http://127.0.0.1:8000")).rstrip("/")
        self.http = httpx.Client(base_url=self.api, timeout=10)
        self._live_dir: Path | None = None
        self._oracle = None
        self._last_rowid = 0
        self._batches = pl.DataFrame(schema={"batch_id": pl.Int64, "ts": pl.Float64, "version": pl.String,
                                             "n": pl.Int64, "alerts": pl.Int64, "benign": pl.Int64,
                                             "fp": pl.Int64, "attacks": pl.Int64, "tp": pl.Int64})

    # ---------------------------------------------------------------- API
    def get(self, path: str, **params):
        try:
            r = self.http.get(path, params=params)
            return r.json() if r.status_code == 200 else None
        except httpx.TransportError:
            return None

    def post(self, path: str, body: dict, timeout: float = 600) -> tuple[bool, dict]:
        try:
            r = self.http.post(path, json=body, timeout=timeout)
        except httpx.TransportError as e:
            return False, {"detail": f"API unreachable: {e}"}
        try:
            out = r.json()
        except ValueError:
            out = {"detail": r.text}
        return r.status_code == 200, out

    def health(self):
        h = self.get("/health")
        if h and not self._live_dir:
            self._live_dir = Path(h["live_dir"])
        return h

    @property
    def live_dir(self) -> Path | None:
        if self._live_dir is None:
            self.health()
        return self._live_dir

    # ---------------------------------------------------------------- live.db
    def _db(self) -> sqlite3.Connection | None:
        if not self.live_dir or not (self.live_dir / "live.db").exists():
            return None
        return sqlite3.connect(f"file:{self.live_dir / 'live.db'}?mode=ro", uri=True, timeout=5)

    @property
    def oracle(self):
        if self._oracle is None:
            from xnids.live.replay_labels import ReplayOracle

            self._oracle = ReplayOracle()
        return self._oracle

    def refresh_scores(self) -> pl.DataFrame:
        """Fold the scores rows added since the last call into per-(batch, version) counts."""
        db = self._db()
        if db is None:
            return self._batches
        try:
            rows = db.execute("SELECT rowid, ts, batch_id, flow_key, version, alert FROM scores WHERE rowid > ? "
                              "ORDER BY rowid LIMIT 400000", (self._last_rowid,)).fetchall()
        finally:
            db.close()
        if not rows:
            return self._batches
        self._last_rowid = rows[-1][0]
        df = pl.DataFrame(rows, schema=["rowid", "ts", "batch_id", "flow_key", "version", "alert"], orient="row")
        y = np.array([self.oracle.get(k, -1) for k in df["flow_key"].to_list()])
        df = df.with_columns(y=pl.Series(y))
        agg = df.group_by("batch_id", "version").agg(
            pl.col("ts").first(), pl.len().cast(pl.Int64).alias("n"), pl.col("alert").sum().cast(pl.Int64).alias("alerts"),
            (pl.col("y") == 0).sum().cast(pl.Int64).alias("benign"),
            ((pl.col("y") == 0) & (pl.col("alert") == 1)).sum().cast(pl.Int64).alias("fp"),
            (pl.col("y") == 1).sum().cast(pl.Int64).alias("attacks"),
            ((pl.col("y") == 1) & (pl.col("alert") == 1)).sum().cast(pl.Int64).alias("tp"))
        self._batches = (pl.concat([self._batches, agg.select(self._batches.columns)])
                         .group_by("batch_id", "version").agg(pl.col("ts").first(), pl.exclude("ts").sum()))
        return self._batches

    def overlay(self, original: str | None, bucket_flows: int = 5000) -> pl.DataFrame:
        """FPR / DR / alerts per `bucket_flows` streamed flows, per series ('original', 'adapted')."""
        b = self.refresh_scores()
        if b.is_empty():
            return b
        flows = (b.group_by("batch_id").agg(pl.col("n").max(), pl.col("ts").min()).sort("batch_id")
                 .with_columns(streamed=pl.col("n").cum_sum()))
        b = b.join(flows.select("batch_id", "streamed"), on="batch_id").with_columns(
            series=pl.when(pl.col("version") == original).then(pl.lit("original")).otherwise(pl.lit("adapted")),
            bucket=(pl.col("streamed") - 1) // bucket_flows)
        return (b.group_by("series", "bucket").agg(pl.col("streamed").max(), pl.col("ts").max(),
                                                   pl.col("n", "alerts", "benign", "fp", "attacks", "tp").sum())
                .with_columns(fpr=pl.when(pl.col("benign") >= 50).then(pl.col("fp") / pl.col("benign")),
                              dr=pl.when(pl.col("attacks") >= 50).then(pl.col("tp") / pl.col("attacks")))
                .sort("series", "bucket"))

    def drift(self) -> pl.DataFrame:
        db = self._db()
        if db is None:
            return pl.DataFrame()
        try:
            rows = db.execute("SELECT id, ts, window_id, recommend, message, rec_action, rec_gain, rec_std, "
                              "active_version, report FROM drift ORDER BY id").fetchall()
        finally:
            db.close()
        out = []
        for r in rows:
            rep = json.loads(r[9])
            out.append({"id": r[0], "ts": r[1], "window": r[2], "alert": bool(r[3]), "message": r[4],
                        "action": r[5], "gain": r[6], "std": r[7], "active": r[8],
                        "novelty": rep.get("novelty_share"), "streak": rep.get("streak", 0),
                        "mmd_p": rep["mmd"][1], "adwin": bool(rep.get("adwin_change")),
                        "top": rep.get("top_features", []), "ks": rep.get("ks", {}),
                        "explanations": rep.get("explanations", [])})
        return pl.DataFrame(out, strict=False) if out else pl.DataFrame()

    @staticmethod
    def novelty_threshold(track: str = "nfs", source: str = "nfs17") -> float | None:
        p = paths.MODELS / "monitor" / f"{track}-{source}.json"
        return json.loads(p.read_text())["thresholds"]["novelty_share"] if p.exists() else None

    # ---------------------------------------------------------------- replay control (demo convenience)
    def replay_dir(self) -> Path | None:
        return self.live_dir / "replay" if self.live_dir else None

    def replay_status(self) -> dict | None:
        d = self.replay_dir()
        if d is None or not (d / "status.json").exists():
            return None
        try:
            st = json.loads((d / "status.json").read_text())
        except (json.JSONDecodeError, OSError):
            return None
        st["running"] = self.replay_running()
        return st

    def replay_running(self) -> bool:
        d = self.replay_dir()
        pid_f = d / "pid" if d else None
        if not pid_f or not pid_f.exists():
            return False
        try:
            os.kill(int(pid_f.read_text()), 0)
            return True
        except (ProcessLookupError, ValueError):
            return False

    def start_replay(self, rate_a: float, rate_b: float, source: str = "file") -> int:
        """Replay demo.pcap (file mode) against this API: rate_a flows/s through segment A, then rate_b."""
        d = self.replay_dir()
        d.mkdir(parents=True, exist_ok=True)
        counts = json.loads((paths.REPLAY / "demo.json").read_text())["counts"]
        n_a = sum(c["len"] for c in counts if c["segment"] == "A")
        cmd = [sys.executable, str(paths.REPO / "scripts/replay.py"), "--mode", source, "--api", self.api,
               "--run-dir", str(d.relative_to(paths.REPO) if d.is_relative_to(paths.REPO) else d),
               "--rate-schedule", f"0:{rate_a},{n_a}:{rate_b}"]
        log = open(d / "replay.log", "w")  # noqa: SIM115
        p = subprocess.Popen(cmd, cwd=paths.REPO, stdout=log, stderr=log,
                             env=os.environ | {"PYTHONPATH": str(paths.REPO / "src")})
        (d / "pid").write_text(str(p.pid))
        return p.pid

    def stop_replay(self) -> None:
        d = self.replay_dir()
        if d and (d / "pid").exists():
            try:
                os.kill(int((d / "pid").read_text()), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
