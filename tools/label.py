"""Label positions with Stockfish evaluations, one engine per core.

    uv run python -m tools.label --in data/positions.txt --out data/labelled.jsonl --nodes 200000

Each line of the output is {"fen", "cp", "mate", "best", "nodes"}. `cp` is centipawns from
the side to move, which is the perspective a search wants back from its evaluation, so the
labels can be trained against directly without a sign flip.

The run is resumable: positions already present in --out are skipped, so a long labelling job
survives being interrupted. Killing it mid-write can leave one truncated final line, which the
loader below drops.
"""

import argparse
import json
import multiprocessing
import sys
from pathlib import Path

import chess
import chess.engine

from tools.engine import spawn

ENGINE: chess.engine.SimpleEngine | None = None
LIMIT: chess.engine.Limit | None = None


def start(nodes: int, depth: int | None, hash_mb: int) -> None:
    """Runs once per worker process. Each holds its own engine for the life of the pool."""
    global ENGINE, LIMIT
    ENGINE = spawn(hash_mb)
    LIMIT = chess.engine.Limit(depth=depth) if depth else chess.engine.Limit(nodes=nodes)


def evaluate(fen: str) -> str | None:
    if ENGINE is None or LIMIT is None:
        raise RuntimeError("worker used before its engine was started")
    board = chess.Board(fen)
    info = ENGINE.analyse(board, LIMIT)
    score = info["score"].relative
    principal = info.get("pv")
    record = {
        "fen": fen,
        "cp": score.score(),
        "mate": score.mate(),
        "best": principal[0].uci() if principal else None,
        "nodes": info.get("nodes"),
    }
    return json.dumps(record)


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(json.loads(line)["fen"])
            except (json.JSONDecodeError, KeyError):
                continue  # a truncated final line from an interrupted run
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Label positions with Stockfish.")
    parser.add_argument("--in", dest="source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=200_000)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count() - 1)
    arguments = parser.parse_args()

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(arguments.out)
    fens = [line.strip() for line in arguments.source.read_text(encoding="utf-8").splitlines()]
    todo = [fen for fen in fens if fen and fen not in done]
    print(f"{len(fens)} positions, {len(done)} already labelled, {len(todo)} to do")
    if not todo:
        return

    # Every worker starts its own engine, and each of those loads the net before it is any
    # use. On a many-core box that startup dwarfs a small job, so never open more engines
    # than there are positions to hand them.
    workers = max(1, min(arguments.workers, len(todo)))
    settings = (arguments.nodes, arguments.depth, arguments.hash_mb)
    print(f"labelling with {workers} workers")
    with (
        arguments.out.open("a", encoding="utf-8") as sink,
        multiprocessing.Pool(workers, initializer=start, initargs=settings) as pool,
    ):
        for finished, line in enumerate(pool.imap_unordered(evaluate, todo), start=1):
            if line is None:
                continue
            sink.write(line + "\n")
            if finished % 100 == 0:
                sink.flush()
                print(f"{finished}/{len(todo)}", file=sys.stderr)

    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
