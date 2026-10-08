"""C1: download, checksum and convert datasets to Parquet.

    python scripts/ingest.py --dataset lycos18
    python scripts/ingest.py --dataset all            # everything available; manual ones are reported, not fatal
    python scripts/ingest.py --dataset lycos18 --delete-raw-csv
"""

import argparse
import json
import logging
import sys

from xnids.data.ingest import ManualDownloadRequired, ingest
from xnids.utils import config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--delete-raw-csv", action="store_true",
                    help="after a verified conversion, delete CSVs that can be recreated (unzipped or re-downloadable)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")

    cfg = config.load(args.config)
    names = list(cfg["datasets"]) if args.dataset == "all" else args.dataset.split(",")
    status = 0
    for name in names:
        spec = cfg["datasets"][name]
        if spec["fetch"]["method"] == "pcap":
            logging.info("%s: built from PCAPs by scripts/extract_flows.py (C13), not by ingest", name)
            continue
        try:
            res = ingest(name, spec, cfg["defaults"], delete_raw_csv=args.delete_raw_csv)
        except ManualDownloadRequired as e:
            logging.warning(str(e))
            status = status or 2
            continue
        flag = "" if res["rows_match"] else "   <-- ROW COUNT MISMATCH, explain in DATASET_CARD.md"
        print(f"{name}: {json.dumps(res)}{flag}")
        if not res["rows_match"]:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
