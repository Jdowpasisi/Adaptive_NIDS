"""C16 demo launcher: a FRESH detector API (data/live/demo, port 8000) + the Streamlit dashboard (port 8501).

    make demo            # then open http://localhost:8501 and press "Start replay" in the sidebar

Ctrl-C stops both. Every launch starts from the original model with an empty audit log.
"""

import argparse
import os
import subprocess
import sys

from xnids.live.server import ApiServer
from xnids.utils import paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ui-port", type=int, default=8501)
    ap.add_argument("--address", default="127.0.0.1",
                    help="dashboard bind address (its buttons promote models: keep it local unless needed)")
    args = ap.parse_args()
    with ApiServer(paths.REPO / "data/live/demo", args.port) as srv:
        print(f"API {srv.url} (state in data/live/demo); dashboard http://localhost:{args.ui_port}", flush=True)
        env = os.environ | {"DG_API": srv.url, "PYTHONPATH": str(paths.REPO / "src")}
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(paths.REPO / "dashboard/app.py"),
                            "--server.port", str(args.ui_port), "--server.address", args.address,
                            "--server.headless", "true",
                            "--browser.gatherUsageStats", "false"], env=env, check=False)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
