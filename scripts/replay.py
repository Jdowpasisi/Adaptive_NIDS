"""C15 replay runner CLI.

    python scripts/replay.py --mode file                      # demo.pcap -> NFStream -> API (started here, fresh)
    python scripts/replay.py --mode file --rate 2000 --auto-adapt --auto-approve      # unattended rehearsal
    python scripts/replay.py --mode live --source veth1 --api http://127.0.0.1:8000   # see scripts/replay_live.sh
    python scripts/replay.py --mode parquet --rate 5000 --no-monitor                  # load test building block

--start-api (default unless --api is given) runs a fresh API in data/live/runs/<run-id>/api. Outputs go to
data/live/runs/<run-id>/ (batches.csv, windows.csv, flows.parquet, summary.json).
"""

import argparse
import json
import time

from xnids.live.monitoring import build_monitor
from xnids.live.replay import FlowSource, ReplayRunner
from xnids.live.server import ApiServer
from xnids.utils import config, paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/replay.yaml")
    ap.add_argument("--mode", choices=["file", "live", "parquet"], default="file")
    ap.add_argument("--source", default=None, help="pcap (file), interface (live) or parquet path")
    ap.add_argument("--rate", type=float, default=0, help="target flows/s (0 = as fast as possible)")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many flows")
    ap.add_argument("--api", default=None, help="use a running API instead of starting a fresh one")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-monitor", action="store_true")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--auto-adapt", action="store_true", help="rehearsal: /adapt on the first persisted alert")
    ap.add_argument("--auto-approve", action="store_true", help="rehearsal: promote when the gates pass")
    ap.add_argument("--action", default=None, help="adapt with this action instead of the recommendation")
    ap.add_argument("--idle-stop", type=float, default=None, help="live: stop after N s without a flow")
    args = ap.parse_args()
    cfg = config.load(args.config)
    run_id = args.run_id or f"{args.mode}-{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir = paths.REPO / "data/live/runs" / run_id
    source = args.source or {"file": cfg["pcap"], "parquet": str(paths.REPLAY / "demo_flows.parquet"),
                             "live": "veth1"}[args.mode]
    monitor = None if args.no_monitor else build_monitor(cfg)

    def go(api: str) -> dict:
        r = ReplayRunner(cfg, api, run_dir, rate=args.rate or None, monitor=monitor, auto_adapt=args.auto_adapt,
                         auto_approve=args.auto_approve, adapt_action=args.action, labels=not args.no_labels,
                         idle_stop=args.idle_stop)
        return r.run(FlowSource(args.mode, source, args.limit))

    if args.api:
        summary = go(args.api)
    else:
        with ApiServer(run_dir / "api", args.port) as srv:
            summary = go(srv.url)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
