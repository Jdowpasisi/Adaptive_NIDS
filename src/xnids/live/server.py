"""Start a detector API (uvicorn subprocess) with its own fresh live directory: C14 benchmark, C15 runner / load test."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

from xnids.utils import paths


class ApiServer:
    def __init__(self, live_dir: Path, port: int = 8000, fresh: bool = True, overrides: dict | None = None) -> None:
        self.live_dir, self.port = Path(live_dir), port
        if fresh:
            shutil.rmtree(self.live_dir, ignore_errors=True)
        self.live_dir.mkdir(parents=True, exist_ok=True)
        cfg = (paths.CONFIGS / "live.yaml").read_text()
        lines = [ln for ln in cfg.splitlines() if not ln.startswith(("live_dir:", "candidates_dir:"))]
        lines += [f"live_dir: {self.live_dir}", f"candidates_dir: {self.live_dir / 'candidates'}"]
        lines += [f"{k}: {v}" for k, v in (overrides or {}).items()]
        self.cfg_path = self.live_dir / "live.yaml"
        self.cfg_path.write_text("\n".join(lines) + "\n")
        self.url = f"http://127.0.0.1:{port}"
        self.proc = None

    def __enter__(self) -> "ApiServer":
        env = os.environ | {"DG_LIVE_CONFIG": str(self.cfg_path), "PYTHONPATH": str(paths.REPO / "src")}
        log = open(self.live_dir / "api.log", "w")  # noqa: SIM115
        self.proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "xnids.live.api:app", "--port", str(self.port),
                                      "--log-level", "warning"], env=env, stdout=log, stderr=log)
        for _ in range(240):
            try:
                if httpx.get(self.url + "/health", timeout=2).status_code == 200:
                    return self
            except httpx.TransportError:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(f"API exited; see {self.live_dir / 'api.log'}")
            time.sleep(0.5)
        raise RuntimeError("API did not come up in 120 s")

    def __exit__(self, *exc) -> None:
        if self.proc:
            self.proc.terminate()
            self.proc.wait(15)
