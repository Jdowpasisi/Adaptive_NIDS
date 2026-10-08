"""Read members of a large remote zip over HTTP range requests, without downloading the whole archive.

zipfile only needs a seekable file object: the central directory sits at the end of the archive, so listing the
members costs a few small range requests, and extracting one member downloads only that member's bytes.
Used for CSE-CIC-IDS2018's per-day pcap.zip (36 GB), of which the demo needs a few hosts.
"""

import io
import urllib.request


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 1 << 20) -> None:
        self.url, self.pos, self.block = url, 0, block
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
        self._cache: dict[int, bytes] = {}

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _range(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req) as r:
            return r.read()

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n >= self.block:                       # large reads (member data) go straight through
            data = self._range(self.pos, self.pos + n)
        else:                                     # small reads (headers, central directory) are block-cached
            b0 = self.pos // self.block
            if b0 not in self._cache:
                self._cache[b0] = self._range(b0 * self.block, min(self.size, (b0 + 1) * self.block))
            blk = self._cache[b0]
            off = self.pos - b0 * self.block
            data = blk[off:off + n]
            if len(data) < n:                     # spans a block boundary
                self.pos += len(data)
                return data + self.read(n - len(data))
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)
