"""C18: regenerate every table (and the figures drawn from them) from MLflow run IDs, then write PROVENANCE.csv.

    python scripts/make_tables.py               # the 'mlflow' steps: rebuilt from finished runs, no training (~3 min)
    python scripts/make_tables.py --derived     # + the steps recomputed from local data (bridge check, demo check, ...)
    python scripts/make_tables.py --list        # what produces each table / figure, without running anything

Each step's command and the MLflow runs behind it are in reports/PROVENANCE.csv and reports/provenance/<step>_runs.csv.
The 'unchanged_by_rebuild' column says whether a file is byte-identical after the rebuild (determinism check).
Steps of kind 'pipeline' (ingest, splits, live-system runs) are listed but rerun only by `make reproduce`.
"""

import argparse

from xnids import reporting as R


def main(figures_only: bool = False) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--steps", default=None, help="comma list of step names")
    args = ap.parse_args()
    if args.list:
        for s in R.STEPS:
            print(f"{s.component:4} {s.kind:8} {s.name:18} {s.command}   -> {', '.join(s.outputs) or '(figures)'}")
        return
    kinds = {"mlflow"} | ({"derived"} if args.derived else set())
    steps = [s for s in R.STEPS if s.kind in kinds and (not args.steps or s.name in args.steps.split(","))]
    if figures_only:
        steps = [s for s in steps if any(g.endswith((".png", "*")) for g in s.outputs) or not s.outputs]
    before = R.snapshot()
    failed = []
    for s in steps:
        ok, sec, tail = R.run(s)
        print(f"{'ok  ' if ok else 'FAIL'} {s.name:18} {sec:6.1f} s   {s.command}", flush=True)
        if not ok:
            failed.append(s.name)
            print("     " + tail.replace("\n", "\n     "))
    p = R.provenance(before)
    rebuilt = p[p.step.isin([s.name for s in steps])]
    same = rebuilt.unchanged_by_rebuild.fillna(False).astype(bool)
    print(f"\n{len(rebuilt)} files from {len(steps)} steps: {int(same.sum())} byte-identical after the rebuild, "
          f"{int((~same).sum())} changed")
    if (~same).any():
        print("changed: " + ", ".join(rebuilt[~same].file.tolist()[:20]))
    unreg = p[p.step == "UNREGISTERED"]
    if len(unreg):
        print("UNREGISTERED outputs (no producer in xnids.reporting.STEPS): " + ", ".join(unreg.file))
    print("provenance: reports/PROVENANCE.csv")
    if failed:
        raise SystemExit(f"failed steps: {failed}")


if __name__ == "__main__":
    main()
