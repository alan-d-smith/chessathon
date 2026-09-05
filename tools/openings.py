"""Build a suite of near-level opening positions to play matches from, one engine per core.

Rated games start from curated positions that are close to level and the set is not published,
so the honest way to test is a suite of our own with the same shape. Two deterministic agents
started from the standard position play one game and then replay it, which reads as a score but
carries no information; a suite of distinct openings is what makes a match mean anything, and
for outcome-labelled data it is what makes the games distinct at all.

    uv run python -m tools.openings --count 300 --out data/openings.txt

A position is kept when Stockfish puts it inside --max-cp of level, so neither side is handed
the game before either agent has moved. Most candidates are rejected, so this is mostly engine
time and worth spreading across the cores.
"""

import argparse
import multiprocessing
import random
import sys
from pathlib import Path

import chess
import chess.engine

from tools.engine import spawn

ENGINE: chess.engine.SimpleEngine | None = None


def key(board: chess.Board) -> str:
    return " ".join(board.fen().split(" ")[:4])


def start() -> None:
    """Runs once per worker. Each keeps its own engine for the life of the pool."""
    global ENGINE
    ENGINE = spawn(hash_mb=32)


def walk(rng: random.Random, plies: int) -> chess.Board | None:
    """Random legal moves, which is what makes the suite diverse rather than merely long."""
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            return None
        board.push(rng.choice(moves))
    return None if board.is_game_over() else board


def batch(task: tuple[int, int, int, int, int, int]) -> tuple[list[str], int]:
    """One worker's share of the accepted positions, and how many it had to try."""
    if ENGINE is None:
        raise RuntimeError("worker used before its engine was started")
    seed, count, min_plies, max_plies, max_cp, nodes = task
    rng = random.Random(seed)
    limit = chess.engine.Limit(nodes=nodes)
    kept: list[str] = []
    tried = 0
    while len(kept) < count:
        tried += 1
        board = walk(rng, rng.randint(min_plies, max_plies))
        if board is None:
            continue
        score = ENGINE.analyse(board, limit)["score"].relative
        if score.is_mate() or abs(score.score() or 0) > max_cp:
            continue
        kept.append(board.fen())
    return kept, tried


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced opening suite.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--min-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=12)
    parser.add_argument("--max-cp", type=int, default=60)
    parser.add_argument("--nodes", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    arguments = parser.parse_args()

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(arguments.workers, arguments.count))
    share, spare = divmod(arguments.count, workers)
    tasks = [
        (
            arguments.seed + index,
            share + (1 if index < spare else 0),
            arguments.min_plies,
            arguments.max_plies,
            arguments.max_cp,
            arguments.nodes,
        )
        for index in range(workers)
    ]

    print(f"{arguments.count} openings across {workers} engines", file=sys.stderr)
    seen: set[str] = set()
    written = 0
    tried = 0

    # Most candidates are rejected, so this is expensive engine time. Finished work is flushed
    # as it arrives and an existing file is resumed, rather than held to the end where an
    # interruption would take all of it.
    if arguments.out.is_file():
        with arguments.out.open(encoding="utf-8") as existing:
            for line in existing:
                if line.strip():
                    seen.add(" ".join(line.strip().split(" ")[:4]))
                    written += 1
        print(f"resuming with {written} openings already on disk", file=sys.stderr)

    with (
        arguments.out.open("a", encoding="utf-8") as sink,
        multiprocessing.Pool(workers, initializer=start) as pool,
    ):
        for done, (found, attempts) in enumerate(pool.imap_unordered(batch, tasks), start=1):
            tried += attempts
            for fen in found:
                # Workers draw independently, so the same position can surface twice.
                identity = " ".join(fen.split(" ")[:4])
                if identity not in seen:
                    seen.add(identity)
                    sink.write(fen + "\n")
                    written += 1
            sink.flush()
            print(f"  worker {done}/{workers}, {written} unique", file=sys.stderr)

    print(f"{written} balanced openings in {arguments.out} ({tried} positions tried)")


if __name__ == "__main__":
    main()
