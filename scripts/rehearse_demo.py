"""C16 done-when: run the 5-minute demo script start to finish N times in a row, through the REAL dashboard
(Streamlit AppTest: the same app.py, its buttons and forms) and a fresh detector API per run.

    python scripts/rehearse_demo.py [--runs 3] [--rate-a 2000] [--rate-b 300]

Per run, as an analyst would: Start replay -> wait for the drift alert -> Build candidate (the recommended action)
-> check the gates -> type name + reason -> Approve -> watch original vs adapted for --watch seconds -> Roll back
-> Stop. Each step must happen as scripted, otherwise the run fails. Writes reports/tables/c16_rehearsals.csv.
"""

import argparse
import json
import os
import time

import polars as pl
from streamlit.testing.v1 import AppTest

from xnids.live.dashboard_data import DashboardData
from xnids.live.server import ApiServer
from xnids.utils import paths


class Rehearsal:
    def __init__(self, url: str) -> None:
        os.environ["DG_API"] = url
        self.at = AppTest.from_file(str(paths.REPO / "dashboard/app.py"), default_timeout=600)
        self.D = DashboardData(url)
        self.t0 = time.monotonic()
        self.log: dict = {}

    def run(self) -> None:
        self.at.run()
        if self.at.exception:
            raise RuntimeError(f"dashboard exception: {self.at.exception[0].value}")

    def btn(self, label_or_key: str):
        for b in self.at.button:
            if b.key == label_or_key or b.label == label_or_key:
                return b
        raise KeyError(f"no button {label_or_key!r}; have {[b.label for b in self.at.button]}")

    def mark(self, name: str) -> None:
        self.log[name] = round(time.monotonic() - self.t0, 1)
        print(f"  {self.log[name]:7.1f} s  {name}", flush=True)

    def wait(self, cond, what: str, timeout: float) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.run()
            if cond():
                return
            time.sleep(2)                                # the dashboard's own polling period
        raise TimeoutError(f"timed out after {timeout:.0f} s waiting for: {what}")


def rehearse(i: int, args) -> dict:
    with ApiServer(paths.REPO / f"data/live/rehearsal-{i}", 8040 + i) as srv:
        R = Rehearsal(srv.url)
        R.run()
        R.at.number_input(key="rate_a").set_value(args.rate_a)
        R.at.number_input(key="rate_b").set_value(args.rate_b)
        R.btn("start_replay").click()
        R.run()
        R.mark("replay started")
        n_a = sum(c["len"] for c in json.loads((paths.REPLAY / "demo.json").read_text())["counts"]
                  if c["segment"] == "A")
        R.wait(lambda: (R.D.replay_status() or {}).get("flows_sent", 0) >= n_a, "segment B", 600)
        R.mark("switch to segment B")
        drift_a = R.D.drift()
        alerts_in_a = 0 if drift_a.is_empty() else int(drift_a.filter(pl.col("alert"))["window"].len())
        R.wait(lambda: (R.D.get("/drift/latest") or {}).get("recommend") == 1, "drift alert", 300)
        R.mark("drift alert")
        latest = R.D.get("/drift/latest")
        rec = latest["rec_action"]
        assert rec and rec != "wait", f"recommendation is {rec}"
        assert R.at.selectbox(key="adapt_action").value == rec, "action box should default to the recommendation"
        R.btn("build_candidate").click()
        R.run()                                          # blocks while /adapt runs
        cand = (R.D.get("/models") or {}).get("candidate")
        assert cand, "no candidate after Build candidate"
        R.mark(f"candidate built ({rec})")
        rebuilds = 0
        while not cand["gates"]["all_passed"]:                   # the gate blocked a bad candidate: the analyst's
            failed = [k for k, v in cand["gates"].items() if isinstance(v, dict) and v.get("passed") is False]
            R.mark(f"gates FAILED ({', '.join(failed)}): reject & rebuild")
            assert rebuilds < 3, "three rebuilds failed the gates"
            R.btn("rebuild").click()                             # scripted next step is Reject & rebuild
            R.run()
            rebuilds += 1
            cand = R.D.get("/models")["candidate"]
            R.mark(f"candidate rebuilt (seed {rebuilds})")
        R.at.text_input(key="approver").input("demo analyst")
        R.at.text_input(key="reason").input("FPR jump on the new network; gates pass")
        R.btn("✅ Approve & promote").click()
        R.run()
        active = R.D.get("/models")["active"]["version"]
        assert active == cand["version"], "promotion did not happen"
        R.mark("approved and promoted")
        n_prom = (R.D.replay_status() or {})["flows_sent"]
        time.sleep(args.watch)
        R.run()
        ov = R.D.overlay(R.D.get("/models")["original"])
        after = ov.filter(pl.col("streamed") > n_prom + 5000)          # buckets fully after the promotion
        fpr = {s: after.filter(pl.col("series") == s).select(pl.col("fp").sum() / pl.col("benign").sum()).item()
               for s in ("original", "adapted")}
        R.mark("side by side after approval: FPR original {:.1%} vs adapted {:.1%}".format(
            fpr["original"], fpr["adapted"]))
        assert fpr["adapted"] < fpr["original"], "adapted FPR should be below the original's"
        R.at.text_input(key="rb_name").input("demo analyst")
        R.at.text_input(key="rb_reason").input("demonstrate rollback")
        R.btn("↩ Roll back").click()
        R.run()
        m = R.D.get("/models")
        assert m["active"]["version"] == m["original"], "rollback did not restore the original"
        R.mark("rolled back")
        R.btn("stop_replay").click()
        R.run()
        kinds = [a["kind"] for a in reversed(R.D.get("/audit", limit=1000))]
        order = [k for k in kinds if k in ("drift_alert", "recommendation", "adapt", "gate_check", "promote",
                                           "rollback")]
        need = ["drift_alert", "recommendation", "adapt", "gate_check", "promote", "rollback"]
        it = iter(order)
        assert all(k in it for k in need), f"audit order {order[:12]}"
        R.mark("audit log complete")
        return {"run": i, "alerts_in_segment_a": alerts_in_a, "action": rec, "rebuilds": rebuilds,
                **{k: v for k, v in R.log.items()},
                "fpr_original_after": fpr["original"], "fpr_adapted_after": fpr["adapted"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rate-a", type=float, default=2000)
    ap.add_argument("--rate-b", type=float, default=300)
    ap.add_argument("--watch", type=float, default=40)
    args = ap.parse_args()
    rows = []
    for i in range(1, args.runs + 1):
        print(f"== rehearsal {i}", flush=True)
        rows.append(rehearse(i, args))
    t = pl.DataFrame(rows, strict=False)
    t.write_csv(paths.TABLES / "c16_rehearsals.csv")
    with pl.Config(tbl_cols=20, tbl_width_chars=250, float_precision=3):
        print(t)
    print(f"{len(rows)} of {args.runs} rehearsals ran start to finish without intervention "
          f"({sum(r['rebuilds'] for r in rows)} gate-blocked candidates rebuilt from the dashboard)")


if __name__ == "__main__":
    main()
