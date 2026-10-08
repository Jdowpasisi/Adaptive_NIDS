"""C13: download selected host PCAPs of one CSE-CIC-IDS2018 day without the 36 GB pcap.zip.

    python scripts/fetch_cse18_pcaps.py            # members listed in configs/demo.yaml (cse18)

The archive's central directory is read with HTTP range requests (xnids.live.remotezip); each wanted member's
compressed bytes are then fetched as ONE range (resumed by appending after a drop) and inflated locally, with the zip CRC-32
checked. Output: data/raw/cse18_pcap/<member basename>.pcap (+ sha256 in data/MANIFEST.csv via the ingest helper).
"""

import argparse
import struct
import subprocess
import zipfile
import zlib
from pathlib import Path

from xnids.data.ingest import update_manifest
from xnids.live.remotezip import HTTPRangeFile
from xnids.utils import config, paths


def fetch_member(url: str, info: zipfile.ZipInfo, raw: Path, out: Path) -> None:
    f = HTTPRangeFile(url)
    f.seek(info.header_offset)
    hdr = f.read(30)
    sig, *_rest = struct.unpack("<IHHHHHIIIHH", hdr)
    assert sig == 0x04034B50, "bad local header"
    name_len, extra_len = struct.unpack("<HH", hdr[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    end = start + info.compress_size - 1
    part = raw / (out.name + ".deflate")
    for _attempt in range(20):                    # resume by hand: curl refuses -C together with -r
        have = part.stat().st_size if part.exists() else 0
        if have >= info.compress_size:
            break
        with part.open("ab") as fh:
            subprocess.run(["curl", "-s", "-S", "-f", "--retry", "5", "-r", f"{start + have}-{end}", url],
                           stdout=fh, check=False)
    if part.stat().st_size != info.compress_size:
        raise RuntimeError(f"{out.name}: got {part.stat().st_size} bytes, expected {info.compress_size}")
    crc = 0
    dec = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    with part.open("rb") as src, out.open("wb") as dst:
        while chunk := src.read(1 << 22):
            data = dec.decompress(chunk) if dec else chunk
            crc = zlib.crc32(data, crc)
            dst.write(data)
        if dec:
            tail = dec.flush()
            crc = zlib.crc32(tail, crc)
            dst.write(tail)
    if crc != info.CRC or out.stat().st_size != info.file_size:
        raise RuntimeError(f"{out.name}: CRC / size mismatch")
    part.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    args = ap.parse_args()
    c = config.load(args.config)["cse18"]
    raw = paths.RAW / "cse18_pcap"
    raw.mkdir(parents=True, exist_ok=True)
    url = c["url"]
    z = zipfile.ZipFile(HTTPRangeFile(url))
    members = {i.filename: i for i in z.infolist()}
    done = []
    for name in c["members"]:
        info = members[name]
        out = raw / (Path(name).name.removesuffix(".pcap") + ".pcap")
        if out.exists() and out.stat().st_size == info.file_size:
            print(f"have {out.name}")
        else:
            print(f"fetching {name} ({info.compress_size / 2**20:.0f} MB zipped, {info.file_size / 2**20:.0f} MB)")
            fetch_member(url, info, raw, out)
        done.append(out)
    update_manifest("cse18_pcap", done, None)
    print("ok:", [p.name for p in done])


if __name__ == "__main__":
    main()
