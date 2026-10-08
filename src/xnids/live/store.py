"""C14 store: SQLite live.db, read by the dashboard (C16).

    scores   one row per (flow, model role) scored by /score
    drift    one row per DriftReport received, with the selector's recommendation
    actions  the audit log: drift alerts, recommendations, adapt / gate / promote / reject / rollback, who and why
"""

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
    ts REAL, batch_id INTEGER, flow_key TEXT, role TEXT, version TEXT, score REAL, alert INTEGER);
CREATE INDEX IF NOT EXISTS scores_ts ON scores(ts);
CREATE TABLE IF NOT EXISTS drift (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, window_id INTEGER, n INTEGER, recommend INTEGER,
    message TEXT, rec_action TEXT, rec_gain REAL, rec_std REAL, active_version TEXT, report TEXT);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, actor TEXT, version_from TEXT, version_to TEXT,
    reason TEXT, detail TEXT);
"""


class Store:
    def __init__(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")          # the dashboard reads while the API writes
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def add_scores(self, rows: list[tuple]) -> None:
        with self.lock:
            self.db.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?,?)", rows)

    def add_drift(self, report: dict, rec: dict, active_version: str) -> int:
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO drift (ts, window_id, n, recommend, message, rec_action, rec_gain, rec_std, "
                "active_version, report) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (time.time(), report.get("window_id"), report.get("n"), int(bool(report.get("recommend"))),
                 report.get("message"), rec.get("action"), rec.get("gain"), rec.get("std"), active_version,
                 json.dumps(report, default=float)))
            return int(cur.lastrowid)

    def log(self, kind: str, actor: str = "system", version_from: str | None = None, version_to: str | None = None,
            reason: str = "", **detail) -> int:
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO actions (ts, kind, actor, version_from, version_to, reason, detail) VALUES (?,?,?,?,?,?,?)",
                (time.time(), kind, actor, version_from, version_to, reason, json.dumps(detail, default=float)))
            return int(cur.lastrowid)

    def query(self, sql: str, args: tuple = ()) -> list[dict]:
        with self.lock:
            cur = self.db.execute(sql, args)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

    def latest_drift(self) -> dict | None:
        r = self.query("SELECT * FROM drift ORDER BY id DESC LIMIT 1")
        if not r:
            return None
        r[0]["report"] = json.loads(r[0]["report"])
        return r[0]

    def actions(self, limit: int = 100) -> list[dict]:
        out = self.query("SELECT * FROM actions ORDER BY id DESC LIMIT ?", (limit,))
        for r in out:
            r["detail"] = json.loads(r["detail"]) if r["detail"] else {}
        return out
