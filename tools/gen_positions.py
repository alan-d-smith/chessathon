"""Build a pool of positions to label, one engine per core.

Random plies alone give positions no real game reaches, so each walk takes a few random moves
for variety and then lets Stockfish play on at a low node count, sampling as it goes. That
lands in the same middlegames the rated games will, which is the point: an evaluation fitted on
positions the search never sees is fitted on the wrong thing.

    uv run python -m tools.gen_positions --games 4000 --out data/positions.txt
    uv run python -m tools.gen_positions --from-pgn games.pgn --out data/positions.txt

Positions are deduplicated on the board alone, ignoring move counters, so transpositions
collapse. Output is one FEN per line, ready for tools/label.py.
"""

import argparse
import multiprocessing
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from tools.engine import spawn

ENGINE: chess.engine.SimpleEngine | None = None


def key(board: chess.Board) -> str:
    """Board, side, castling and en passant, without the move counters."""
    return " ".join(board.fen().split(" ")[:4])


def start() -> None:
    """Runs once per worker. Each keeps its own engine for the life of the pool."""
    global ENGINE
    ENGINE = spawn(hash_mb=32)


def walk(
    engine: chess.engine.SimpleEngine,
    rng: random.Random,
    opening_plies: int,
    playout_nodes: int,
    max_plies: int,
    every: int,
) -> Iterator[chess.Board]:
    """One randomised game, yielding every `every`th position after the random opening."""
    board = chess.Board()
    for ply in range(max_plies):
        # Deliberately not claim_draw: that replays the move stack hunting a threefold, at
        # every ply of every game, and it dominated the run. Checkmate, stalemate and the
        # material draws are enough to know a playout is over.
        if board.is_game_over():
            return
        if ply < opening_plies:
            board.push(rng.choice(list(board.legal_moves)))
            continue
        if (ply - opening_plies) % every == 0:
            yield board.copy(stack=False)
        played = engine.play(board, chess.engine.Limit(nodes=playout_nodes)).move
        if played is None:
            return
        board.push(played)


def batch(task: tuple[int, int, int, int, int, int]) -> list[str]:
    """One worker's share of the games, as FENs."""
    if ENGINE is None:
        raise RuntimeError("worker used before its engine was started")
    seed, games, opening_plies, playout_nodes, max_plies, every = task
    rng = random.Random(seed)
    found: list[str] = []
    for _ in range(games):
        for board in walk(ENGINE, rng, opening_plies, playout_nodes, max_plies, every):
            found.append(board.fen())
    return found


def from_pgn(path: Path, every: int) -> Iterator[chess.Board]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                if ply % every == 0:
                    yield board.copy(stack=False)
                board.push(move)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate positions for labelling.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--from-pgn", type=Path)
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument("--playout-nodes", type=int, default=8_000)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--every", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    arguments = parser.parse_args()

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0

    # Resume rather than restart. Generation is the slowest stage in the loop, and holding
    # everything in memory until the end once cost a twenty minute run to a single interrupt.
    if arguments.out.is_file():
        with arguments.out.open(encoding="utf-8") as existing:
            for line in existing:
                fen = line.strip()
                if fen:
                    seen.add(" ".join(fen.split(" ")[:4]))
                    written += 1
        print(f"resuming with {written:,} positions already on disk", file=sys.stderr)

    sink = arguments.out.open("a", encoding="utf-8")

    def keep(fen: str) -> None:
        nonlocal written
        identity = " ".join(fen.split(" ")[:4])
        if identity not in seen:
            seen.add(identity)
            sink.write(fen + "\n")
            written += 1

    if arguments.from_pgn:
        for board in from_pgn(arguments.from_pgn, arguments.every):
            if not board.is_game_over():
                keep(board.fen())
    else:
        workers = max(1, min(arguments.workers, arguments.games))
        share, spare = divmod(arguments.games, workers)
        tasks = [
            (
                arguments.seed + index,
                share + (1 if index < spare else 0),
                arguments.opening_plies,
                arguments.playout_nodes,
                arguments.max_plies,
                arguments.every,
            )
            for index in range(workers)
        ]
        print(f"{arguments.games} games across {workers} engines", file=sys.stderr)
        with multiprocessing.Pool(workers, initializer=start) as pool:
            for done, found in enumerate(pool.imap_unordered(batch, tasks), start=1):
                for fen in found:
                    keep(fen)
                # Flushed per worker, so whatever has finished survives an interruption.
                sink.flush()
                print(f"  worker {done}/{workers}, {written:,} unique", file=sys.stderr)

    sink.close()
    print(f"{written:,} unique positions in {arguments.out}")


if __name__ == "__main__":
    main()
