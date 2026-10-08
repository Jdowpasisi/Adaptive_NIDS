"""C14 end-to-end walkthrough on the demo replay, through the real API (in-process TestClient, no network):

    python scripts/c14_walkthrough.py [--live-dir data/live/walkthrough] [--approve]

demo flows (data/replay, replay order) -> POST /score/columns in 1,000-flow micro-batches -> every 5,000 flows the C8
monitor (configs/drift.yaml `monitor`, reference = source validation) -> POST /drift/report. On the first window
whose recommendation is not 'wait': POST /adapt (the recommended action), print the gate checks and, with --approve,
POST /models/promote. The run ends with a rollback. FPR / DR per window for the active and the candidate model are
computed from the replay labels (an evaluation overlay; the service never sees them except through the few-shot
label oracle). Writes reports/tables/c14_walkthrough.csv.
"""

import argparse
import json
import shutil

import numpy as np
import polars as pl
from fastapi.testclient import TestClient

from xnids.drift.monitor import DriftMonitor
from xnids.live import replay_labels
from xnids.live.api import create_app
from xnids.live.service import DetectorService
from xnids.utils import config, paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/live.yaml")
    ap.add_argument("--live-dir", default="data/live/walkthrough")
    ap.add_argument("--approve", action="store_true", help="promote the candidate if its gates pass")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--no-adapt", action="store_true", help="only score + monitor (the original model throughout)")
    ap.add_argument("--adapt-from-window", type=int, default=0, help="ignore recommendations before this window")
    ap.add_argument("--action", default=None, help="adapt with this action instead of the recommendation")
    args = ap.parse_args()
    cfg = config.load(args.config)
    live = paths.REPO / args.live_dir
    shutil.rmtree(live, ignore_errors=True)                       # a fresh registry / live.db for every run
    cfg |= {"live_dir": str(live), "candidates_dir": str(live / "candidates")}
    svc = DetectorService(cfg)
    c = TestClient(create_app(svc))
    mcfg = config.load("configs/drift.yaml")["monitor"]
    src = svc.source
    mon = DriftMonitor(svc.registry.get("active"), *src.val, mcfg)
    flows = replay_labels.attach(pl.read_parquet(paths.REPLAY / "demo_flows.parquet"))
    flows = flows.sort("timestamp", maintain_order=True).with_columns(replay_labels.flow_key_expr())
    feats, W = svc.features, mcfg["window"]
    rows, event = [], ""
    for w0 in range(0, flows.height, W):
        win = flows.slice(w0, W)
        sc = {"active": [], "candidate": []}
        for b0 in range(0, win.height, args.batch):
            b = win.slice(b0, args.batch)
            r = c.post("/score/columns", json={"flow_keys": b["flow_key"].to_list(),
                                                "columns": {f: b[f].cast(pl.Float64).to_list() for f in feats}})
            r.raise_for_status()
            out = r.json()
            for role in sc:
                sc[role] += out[role]["alerts"] if role in out else [None] * b.height
        rep = mon.process(win.select(feats))
        d = c.post("/drift/report", json=json.loads(rep.to_json())).json()
        y = win["y"].to_numpy()
        det = rep.detectors
        row = {"window": w0 // W, "segment": win["segment"].mode()[0], "flows": win.height,
               "benign": int((y == 0).sum()), "drift": d["recommend"], "ks_max": d["features"]["ks_max"],
               "votes": "+".join(k for k in ("ks_effect", "mmd", "adwin") if det[k]), "est_rise": rep.est_fpr_rise,
               "recommendation": d["recommendation"]["action"], "rec_gain": d["recommendation"]["gain"],
               "active": svc.registry.version("active"), "event": event}
        event = ""
        for role, a in sc.items():
            if a[0] is None:
                continue
            a = np.array(a, dtype=bool)
            row[f"fpr_{role}"] = float(a[y == 0].mean()) if (y == 0).any() else np.nan
            row[f"dr_{role}"] = float(a[y == 1].mean()) if (y == 1).any() else np.nan
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "active"})
        if not args.no_adapt and svc.registry.version("candidate") is None and not any(r.get("adapted") for r in rows) \
                and row["window"] >= args.adapt_from_window \
                and (args.action or d["recommendation"]["action"] != "wait"):
            r = c.post("/adapt", json={"requested_by": "walkthrough", "action": args.action})
            res = r.json()
            rows[-1]["adapted"] = True
            if r.status_code != 200:
                print("ADAPT FAILED:", res)
                event = "adapt failed"
                continue
            g = res["gates"]
            print(f"\n== /adapt {res['action']} on {res['rows']} buffered flows -> {res['candidate']}")
            for k, v in g.items():
                if isinstance(v, dict):
                    print(f"   gate {k}: " + ", ".join(f"{a}={b}" for a, b in v.items() if a != "note"))
            event = f"candidate {res['action']} (gates {'pass' if g['all_passed'] else 'FAIL'})"
            if args.approve and g["all_passed"]:
                p = c.post("/models/promote", json={"approved_by": "walkthrough", "reason": "C14 walkthrough"})
                print("== promote:", p.status_code, p.json(), "\n")
                if p.status_code == 200:
                    event += "; promoted"
                    mon = DriftMonitor(svc.registry.get("active"), *src.val, mcfg)   # monitor follows the active
    rb = c.post("/models/rollback", json={"requested_by": "walkthrough", "reason": "end of walkthrough"})
    print("== rollback:", rb.status_code, rb.json())
    out = pl.DataFrame(rows, strict=False)
    out.write_csv(paths.TABLES / "c14_walkthrough.csv")
    by = out.group_by("segment", maintain_order=True).agg(pl.col("^fpr_.*$").mean(), pl.col("^dr_.*$").mean(),
                                                         pl.col("drift").sum())
    with pl.Config(tbl_cols=20, float_precision=4):
        print(by)
    print("\naudit:")
    for a in reversed(c.get("/audit", params={"limit": 50}).json()):
        print(f"  {a['kind']:<16} {a['actor']:<12} {a['reason'][:70]}")


if __name__ == "__main__":
    main()
