"""C18: where every table and figure in reports/ comes from, and how to regenerate it.

Three kinds of step:
  mlflow   rebuilt from finished MLflow runs (their logged metrics / per-run tables): no training, minutes in total.
           scripts/make_tables.py runs these by default.
  derived  recomputed from local data files (data/, models/, live run directories): minutes each.
           make_tables.py --derived also runs these.
  pipeline produced while building data or running the live system (ingest, splits, demo PCAP, API benchmarks,
           replays, rehearsals). Listed with the command that produced them; `make reproduce` reruns them.

provenance() writes reports/PROVENANCE.csv: one row per file with its sha256, step, kind, producing command, MLflow
experiment(s), the number of finished runs behind it and reports/provenance/<step>_runs.csv listing those run IDs
with their key tags, so every number traces back to a run ID.
"""

import datetime as dt
import fnmatch
import hashlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from xnids.utils import log, paths


@dataclass
class Step:
    name: str
    component: str
    cmd: list[str]
    kind: str                                   # mlflow | derived | pipeline
    outputs: list[str]                          # globs relative to reports/tables or reports/figures
    experiments: list[str] = field(default_factory=list)
    run_filter: dict = field(default_factory=dict)    # tags selecting this step's runs inside the experiment
    runs_table: str | None = None               # a long table whose run_id column lists exactly the runs used

    @property
    def command(self) -> str:
        return " ".join(self.cmd if self.cmd[0] == "bash" else ["python"] + self.cmd)


STEPS = [
    # ---------------------------------------------------------------- rebuilt from MLflow
    Step("c4_advval", "C4", ["scripts/advval.py", "--summary-only"], "mlflow",
         ["advval_*", "advval/*"], ["advval"]),
    Step("c6_matrix", "C6", ["scripts/matrix.py"], "mlflow",
         ["matrix_*", "h1_checks.csv", "cantone_comparison.csv", "h2_original_vs_corrected.csv", "within_vs_cross_*"],
         ["train"], runs_table="matrix_long.csv"),
    Step("c7_ablation", "C7", ["scripts/ablation.py", "--summary-only"], "mlflow", ["c7_*"], ["c7"],
         {"study": "c7"}, runs_table="c7_long.csv"),
    Step("c7_h4", "C7", ["scripts/timedrift.py", "--summary-only"], "mlflow", ["h4_*"], ["c7"],
         {"study": "c7_timedrift"}, runs_table="h4_timedrift.csv"),
    Step("c8_drift", "C8", ["scripts/drift_eval.py", "--summary-only"], "mlflow", ["drift_*"], ["drift"], runs_table="drift_eval_long.csv"),
    Step("c9_adapt", "C9", ["scripts/adapt_eval.py", "--summary-only"], "mlflow", ["c9_*"], ["c9"], runs_table="c9_long.csv"),
    Step("c10_logs", "C10", ["scripts/build_logs.py", "--summary-only"], "mlflow", ["c10_*"], ["c10"]),
    Step("c11_selector", "C11", ["scripts/selector_eval.py", "--tables-only"], "mlflow", ["c11_*"], ["c10", "c11"]),
    Step("c12_reptile", "C12", ["scripts/reptile_eval.py", "--summary-only"], "mlflow", ["c12_*"], ["c12"], runs_table="c12_long.csv"),
    Step("c17_poison", "C17", ["scripts/poison_eval.py", "--summary-only"], "mlflow", ["c17_*"], ["c17"]),
    # ---------------------------------------------------------------- recomputed from local data
    Step("c2_bridge", "C2", ["scripts/bridge_check.py"], "derived", ["bridge_check*"]),
    Step("c13_demo", "C13", ["scripts/demo_check.py"], "derived", ["c13_demo_*"], ["train"], {"study": "demo"}),
    Step("c15_rehearsal_fig", "C15", ["scripts/plot_replay.py", "--run", "file-rehearsal"], "derived",
         ["c15_rehearsal.png"]),
    Step("c17_figures", "C17", ["scripts/poison_eval.py", "--plot-only"], "derived", []),
    # ---------------------------------------------------------------- produced by the pipeline / live system
    Step("c1_ingest", "C1", ["scripts/ingest.py", "--dataset", "all"], "pipeline",
         ["profile_*", "rename_map_*", "labels_*"]),
    Step("c2_tracks", "C2", ["scripts/build_tracks.py"], "pipeline", ["family_counts.csv"]),
    Step("c3_split", "C3", ["scripts/split.py", "--dataset", "all"], "pipeline",
         ["clean_*", "splits_lock.csv", "split_family_counts.csv"]),
    Step("c14_walkthrough", "C14", ["scripts/c14_walkthrough.py", "--approve", "--adapt-from-window", "37"],
         "pipeline", ["c14_walkthrough.csv"]),
    Step("c14_actions", "C14", ["bash", "scripts/c14_actions.sh"], "pipeline", ["c14_actions_in_B.csv"]),
    Step("c14_latency", "C14", ["scripts/bench_api.py"], "pipeline", ["c14_latency.csv"]),
    Step("c15_monitor", "C15", ["scripts/calibrate_monitor.py"], "pipeline", ["c15_monitor_*"]),
    Step("c15_load", "C15", ["scripts/load_test.py"], "pipeline", ["c15_load*"]),
    Step("c15_parity", "C15", ["scripts/replay_parity.py", "--live", "<run>"], "pipeline", ["c15_parity.csv"]),
    Step("c16_rehearsal", "C16", ["scripts/rehearse_demo.py"], "pipeline", ["c16_rehearsals.csv"]),
]


def outputs() -> list[Path]:
    return sorted(p for d in (paths.TABLES, paths.FIGURES) for p in d.rglob("*") if p.is_file())


def step_for(p: Path) -> Step | None:
    rel = str(p.relative_to(paths.TABLES if p.is_relative_to(paths.TABLES) else paths.FIGURES))
    for s in STEPS:
        if any(fnmatch.fnmatch(rel, g) for g in s.outputs):
            return s
    return None


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(step: Step) -> tuple[bool, float, str]:
    t0 = dt.datetime.now()
    r = subprocess.run([sys.executable, *step.cmd] if step.cmd[0] != "bash" else step.cmd, cwd=paths.REPO,
                       capture_output=True, text=True, env={**__import__("os").environ,
                                                            "PYTHONPATH": str(paths.REPO / "src"),
                                                            "MLFLOW_DISABLE_AGENT_HINT": "1"})
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    return r.returncode == 0, (dt.datetime.now() - t0).total_seconds(), "\n".join(tail)


def step_runs(step: Step) -> pd.DataFrame:
    """The runs behind a step: exactly those listed in its long table when it has one, else the experiment's
    finished runs matching the step's tags."""
    used = None
    if step.runs_table and (paths.TABLES / step.runs_table).exists():
        used = set(pd.read_csv(paths.TABLES / step.runs_table, usecols=["run_id"]).run_id.dropna())
    frames = []
    for e in step.experiments:
        try:
            r = log.find_runs(experiment=e, finished_only=True, **step.run_filter)
        except Exception:  # noqa: BLE001  (an experiment that does not exist yet)
            continue
        if not r.empty:
            tags = ("cfg_hash", "seed", "model", "track", "source", "target", "study", "part", "git_commit")
            keep = ["run_id", "start_time"] + [f"tags.{t}" for t in tags if f"tags.{t}" in r.columns]
            frames.append(r[keep].rename(columns=lambda c: c.removeprefix("tags.")).assign(experiment=e))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out[out.run_id.isin(used)] if used is not None and len(out) else out


def provenance(before: dict[str, str] | None = None) -> pd.DataFrame:
    """Write reports/PROVENANCE.csv and reports/provenance/<step>_runs.csv; return the table."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.REPO, capture_output=True,
                            text=True).stdout.strip()
    out_dir = paths.REPORTS / "provenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_cache: dict[str, int] = {}
    rows = []
    for p in outputs():
        s = step_for(p)
        rel = str(p.relative_to(paths.REPORTS))
        h = sha256(p)
        if s and s.experiments and s.name not in runs_cache:
            r = step_runs(s)
            runs_cache[s.name] = len(r)
            if len(r):
                r.to_csv(out_dir / f"{s.name}_runs.csv", index=False)
        rows.append({"file": rel, "sha256": h[:16], "component": s.component if s else "",
                     "step": s.name if s else "UNREGISTERED", "kind": s.kind if s else "",
                     "command": s.command if s else "", "experiments": ";".join(s.experiments) if s else "",
                     "mlflow_runs": runs_cache.get(s.name, 0) if s else 0,
                     "runs_file": f"provenance/{s.name}_runs.csv" if s and runs_cache.get(s.name) else "",
                     "unchanged_by_rebuild": (before.get(rel) == h) if before is not None and rel in before else None,
                     "git_commit": commit})
    df = pd.DataFrame(rows)
    df.to_csv(paths.REPORTS / "PROVENANCE.csv", index=False)
    return df


def snapshot() -> dict[str, str]:
    return {str(p.relative_to(paths.REPORTS)): sha256(p) for p in outputs()}
