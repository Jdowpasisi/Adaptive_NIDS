"""C14 model registry: one ACTIVE bundle, at most one CANDIDATE, and the history of previous actives.

A candidate is only ever registered (by /adapt); it becomes active through promote(), which requires a named
approver, a reason and passing gate checks. rollback() restores the previous active. The state survives restarts
in <live_dir>/registry.json; every bundle lives in its own directory (models/<version>/ or models/live/<slug>/).
"""

import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from xnids.models.bundle import Bundle


class RegistryError(RuntimeError):
    pass


def slug(version: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", version)


def param_change(active: Bundle, cand: Bundle) -> float | None:
    """||theta_c - theta_a|| / ||theta_a|| over trainable parameters (BN running statistics excluded: AdaBN changes
    only those, by design). None when the two models are not the same torch architecture."""
    na, nc = getattr(active.model, "net", None), getattr(cand.model, "net", None)
    if not isinstance(na, torch.nn.Module) or not isinstance(nc, torch.nn.Module):
        return None
    pa, pc = dict(na.named_parameters()), dict(nc.named_parameters())
    if pa.keys() != pc.keys() or any(pa[k].shape != pc[k].shape for k in pa):
        return None
    with torch.no_grad():
        num = sum(float(((pc[k].cpu() - pa[k].cpu()) ** 2).sum()) for k in pa)
        den = sum(float((pa[k].cpu() ** 2).sum()) for k in pa)
    return (num / den) ** 0.5 if den else None


def gate_checks(active: Bundle, cand: Bundle, canary: tuple[pl.DataFrame, np.ndarray], recent: pl.DataFrame | None,
                cfg: dict) -> dict:
    """The promotion checks: the Build Guide's three (canary DR, parameter change, predicted attack rate) plus canary
    FPR when configured. Each: value, limit, passed (None = not applicable)."""
    Xc, yc = canary
    att = yc == 1

    def dr(b: Bundle) -> float:
        return float(np.mean(b.predict(Xc[np.flatnonzero(att)]))) if att.any() else float("nan")

    def fpr(b: Bundle) -> float:
        return float(np.mean(b.predict(Xc[np.flatnonzero(~att)]))) if (~att).any() else float("nan")

    dr_a, dr_c = dr(active), dr(cand)
    out = {"canary_dr": {"value": dr_c, "active": dr_a, "limit": cfg["canary_dr_ratio"] * dr_a,
                         "passed": bool(dr_c >= cfg["canary_dr_ratio"] * dr_a),
                         "note": f"candidate DR on {int(att.sum())} fixed source-test attacks >= "
                                 f"{cfg['canary_dr_ratio']} x active"}}
    if "canary_fpr_factor" in cfg:
        # added in C14 (not in the Build Guide list): the mirror image of canary_dr. canary_dr catches a candidate
        # that goes silent; this catches one that floods analysts. The C14 walkthrough's label-free candidates
        # built on segment B passed every Build Guide gate with canary FPR 14-41% (active 2.2%).
        fa, fc = fpr(active), fpr(cand)
        lim = max(cfg["canary_fpr_factor"] * fa, fa + cfg["canary_fpr_abs"])
        out["canary_fpr"] = {"value": fc, "active": fa, "limit": lim, "passed": bool(fc <= lim),
                             "note": f"candidate FPR on {int((~att).sum())} fixed source-test benign flows <= "
                                     f"max({cfg['canary_fpr_factor']} x active, active + {cfg['canary_fpr_abs']})"}
    pc = param_change(active, cand)
    out["param_change"] = {"value": pc, "limit": cfg["max_param_change"],
                           "passed": None if pc is None else bool(pc <= cfg["max_param_change"]),
                           "note": "relative L2 change of the trainable parameters" if pc is not None
                           else "not applicable (different model family or architecture)"}
    if recent is not None and recent.height:
        ra, rc = float(np.mean(active.predict(recent))), float(np.mean(cand.predict(recent)))
        lo, hi = cfg["attack_rate"]
        k = cfg["max_rate_ratio"]
        ok = lo <= rc <= hi and (ra == 0 or (ra / k <= rc <= ra * k))
        out["attack_rate"] = {"value": rc, "active": ra, "limit": [lo, hi], "ratio_limit": k, "passed": bool(ok),
                              "note": f"predicted attack share on the {recent.height} buffered flows"}
    else:
        out["attack_rate"] = {"value": None, "passed": None, "note": "no buffered flows"}
    out["all_passed"] = all(v["passed"] is not False for v in out.values() if isinstance(v, dict))
    return out


class Registry:
    def __init__(self, live_dir: Path, candidates_dir: Path, initial: Path) -> None:
        self.state_path = Path(live_dir) / "registry.json"
        self.candidates_dir = Path(candidates_dir)
        self.lock = threading.RLock()
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {"active": {"version": Bundle.load(initial).meta.get("version", Path(initial).name),
                                     "path": str(initial), "since": time.time()},
                          "candidate": None, "history": []}
            self._save()
        self.bundles: dict[str, Bundle] = {}
        self.get("active")
        if self.state["candidate"]:
            self.get("candidate")

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=float))
        tmp.replace(self.state_path)

    def get(self, role: str) -> Bundle | None:
        with self.lock:
            entry = self.state.get(role)
            if not entry:
                return None
            if entry["path"] not in self.bundles:
                self.bundles[entry["path"]] = Bundle.load(Path(entry["path"]))
            return self.bundles[entry["path"]]

    def version(self, role: str) -> str | None:
        e = self.state.get(role)
        return e["version"] if e else None

    def set_candidate(self, b: Bundle, gates: dict, created_by: str, action: str) -> str:
        with self.lock:
            version = f"{b.meta['version']}@{time.strftime('%Y%m%dT%H%M%S')}"
            b.meta["version"] = version
            d = b.save(self.candidates_dir / slug(version))
            self.bundles[str(d)] = b
            self.state["candidate"] = {"version": version, "path": str(d), "since": time.time(), "action": action,
                                       "created_by": created_by, "gates": gates}
            self._save()
            return version

    def promote(self, approved_by: str, reason: str) -> tuple[str, str]:
        with self.lock:
            cand = self.state["candidate"]
            if not cand:
                raise RegistryError("no candidate to promote")
            if not approved_by.strip() or not reason.strip():
                raise RegistryError("promotion needs approved_by and a reason")
            if not cand["gates"].get("all_passed"):
                failed = [k for k, v in cand["gates"].items() if isinstance(v, dict) and v.get("passed") is False]
                raise RegistryError(f"gate checks failed: {failed}")
            old = self.state["active"]
            self.state["history"].append(old)
            self.state["active"] = {**cand, "since": time.time(), "approved_by": approved_by, "reason": reason}
            self.state["candidate"] = None
            self._save()
            return old["version"], cand["version"]

    def reject(self) -> str:
        with self.lock:
            cand = self.state["candidate"]
            if not cand:
                raise RegistryError("no candidate to reject")
            self.state["candidate"] = None
            self._save()
            return cand["version"]

    def rollback(self) -> tuple[str, str]:
        with self.lock:
            if not self.state["history"]:
                raise RegistryError("no previous active version to roll back to")
            cur, prev = self.state["active"], self.state["history"].pop()
            self.state["active"] = {**prev, "since": time.time()}
            self._save()
            return cur["version"], prev["version"]

    def summary(self) -> dict:
        s = json.loads(json.dumps(self.state, default=float))
        return {"active": s["active"], "candidate": s["candidate"],
                "history": [h["version"] for h in s["history"]]}
