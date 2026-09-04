"""Build a pool of positions to label.

Random plies alone give positions no real game reaches, so the default walks a few random
moves for variety and then lets Stockfish play on at a low node count, sampling as it goes.
That lands in the same middlegames the rated games will, which is the point: an evaluation is
only worth what it scores on the positions it actually sees.

    uv run python -m tools.gen_positions --games 200 --out data/positions.txt
    uv run python -m tools.gen_positions --from-pgn games.pgn --out data/positions.txt

Positions are deduplicated on the board alone, ignoring move counters, so transpositions
collapse. Output is one FEN per line, ready for tools/label.py.
"""

import argparse
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from tools.engine import spawn


def key(board: chess.Board) -> str:
    """Board, side, castling and en passant, without the move counters."""
    return " ".join(board.fen().split(" ")[:4])


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
        if board.is_game_over(claim_draw=True):
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
    parser.add_argument("--playout-nodes", type=int, default=20_000)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--every", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0

    with arguments.out.open("w", encoding="utf-8") as sink:
        def keep(board: chess.Board) -> None:
            nonlocal written
            if board.is_game_over(claim_draw=True) or key(board) in seen:
                return
            seen.add(key(board))
            sink.write(board.fen() + "\n")
            written += 1

        if arguments.from_pgn:
            for board in from_pgn(arguments.from_pgn, arguments.every):
                keep(board)
        else:
            rng = random.Random(arguments.seed)
            engine = spawn()
            try:
                for game in range(arguments.games):
                    for board in walk(
                        engine,
                        rng,
                        arguments.opening_plies,
                        arguments.playout_nodes,
                        arguments.max_plies,
                        arguments.every,
                    ):
                        keep(board)
                    progress = f"game {game + 1}/{arguments.games}, {written} positions"
                    print(progress, file=sys.stderr)
            finally:
                engine.close()

    print(f"{written} unique positions written to {arguments.out}")


if __name__ == "__main__":
    main()
