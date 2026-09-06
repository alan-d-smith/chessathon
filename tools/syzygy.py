"""Fetch Syzygy endgame tablebases.

Nothing here is a third party engine: a tablebase is solved ground truth, and AGENTS.md allows
shipping one. What it is useful for is two different things. Locally it gives exact labels for
the positions Stockfish can only estimate, which is the part of the pool the network is weakest
on. In the zip it lets the agent stop searching a position it can simply look up -- but only as
much of it as the fifty megabyte limit allows, which is nothing like all of it.

    uv run python -m tools.syzygy --out data/syzygy --max-men 5
"""

import argparse
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "http://tablebase.sesse.net/syzygy/3-4-5/"
LINK = re.compile(r'href="([^"]+\.rtb[wz])"')
# The mirror throttles hard: eight workers with no user agent failed 144 of 145 files while
# the same requests one at a time all succeeded. A couple of workers and a retry is the
# difference between a working download and an empty directory.
HEADERS = {"User-Agent": "chessathon-tablebase-fetch/1.0"}


def men(name: str) -> int:
    """Pieces in the ending this file covers, counted from its name: KRPvKR is six."""
    return sum(character.isupper() for character in name.split(".")[0])


def listing(timeout: float) -> list[str]:
    html = urllib.request.urlopen(BASE, timeout=timeout).read().decode("utf-8", "replace")
    return sorted(set(LINK.findall(html)))


def fetch(task: tuple[str, Path, float, int]) -> tuple[str, int, str]:
    """Download one table, retrying on the throttling the mirror does under any concurrency.

    Tablebase files never change, so anything already on disk with bytes in it is kept.
    """
    name, out, timeout, attempts = task
    destination = out / name
    if destination.is_file() and destination.stat().st_size > 0:
        return name, destination.stat().st_size, "have"
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(BASE + name, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as body:
                blob = body.read()
        except Exception as failure:
            if attempt + 1 == attempts:
                return name, 0, f"failed: {type(failure).__name__}"
            time.sleep(2**attempt)
            continue
        destination.write_bytes(blob)
        return name, len(blob), "got"
    return name, 0, "failed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Syzygy tablebases.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-men", type=int, default=5)
    parser.add_argument("--dtz", action="store_true", help="also fetch the distance-to-zero files")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--only", type=Path, help="a file of table names to fetch, one per line")
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)
    if arguments.only:
        wanted = [
            line.strip()
            for line in arguments.only.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        wanted = [
            name
            for name in listing(arguments.timeout)
            if men(name) <= arguments.max_men
            and (arguments.dtz or name.endswith(".rtbw"))
        ]

    print(f"{len(wanted)} tables to {arguments.out}")
    tasks = [
        (name, arguments.out, arguments.timeout, arguments.attempts) for name in wanted
    ]
    total = 0
    failures: list[str] = []
    with ThreadPoolExecutor(arguments.workers) as pool:
        for done, (name, size, how) in enumerate(pool.map(fetch, tasks), start=1):
            total += size
            if how.startswith("failed"):
                failures.append(f"{name}: {how}")
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}, {total / 1e6:.1f} MB")
    for failure in failures:
        print(f"  FAIL {failure}")
    print(f"{total / 1e6:.1f} MB in {arguments.out}")


if __name__ == "__main__":
    main()
