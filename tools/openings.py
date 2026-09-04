"""Build a suite of near-level opening positions to play matches from.

Rated games start from curated positions that are close to level and the set is not published,
so the honest way to test is a suite of our own with the same shape. Two deterministic agents
started from the standard position play one game and then replay it, which reads as a score but
carries no information; a suite of distinct openings is what makes a match mean anything.

    uv run python -m tools.openings --count 60 --out data/openings.txt

A position is kept when Stockfish puts it inside --max-cp of level, so neither side is handed
the game before either agent has moved.
"""

import argparse
import random
import sys
from pathlib import Path

import chess
import chess.engine

from tools.engine import spawn


def key(board: chess.Board) -> str:
    return " ".join(board.fen().split(" ")[:4])


def walk(rng: random.Random, plies: int) -> chess.Board | None:
    """Random legal moves, which is what makes the suite diverse rather than merely long."""
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            return None
        board.push(rng.choice(moves))
    return None if board.is_game_over(claim_draw=True) else board


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced opening suite.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--min-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=12)
    parser.add_argument("--max-cp", type=int, default=60)
    parser.add_argument("--nodes", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(arguments.seed)
    engine = spawn()
    limit = chess.engine.Limit(nodes=arguments.nodes)

    kept: list[str] = []
    seen: set[str] = set()
    tried = 0
    try:
        while len(kept) < arguments.count:
            tried += 1
            board = walk(rng, rng.randint(arguments.min_plies, arguments.max_plies))
            if board is None or key(board) in seen:
                continue
            score = engine.analyse(board, limit)["score"].relative
            if score.is_mate() or abs(score.score() or 0) > arguments.max_cp:
                continue
            seen.add(key(board))
            kept.append(board.fen())
            print(f"{len(kept)}/{arguments.count} kept from {tried} tried", file=sys.stderr)
    finally:
        engine.close()

    arguments.out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"{len(kept)} balanced openings written to {arguments.out} ({tried} positions tried)")


if __name__ == "__main__":
    main()
