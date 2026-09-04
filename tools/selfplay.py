"""Play games and record every position labelled with how that game finished.

Fitting the evaluation to Stockfish's evaluation teaches it to agree with Stockfish, which is
not the same thing as teaching it to win. It inherits the teacher's opinions, including the ones
that only make sense to a search far deeper than ours, and a gain measured against Stockfish can
then be partly the model recognising its own teacher rather than playing better chess.

Game results carry no such opinion. A position is labelled 1, 0.5 or 0 by what actually happened
from it, so the fit optimises the only thing that scores points. The signal per position is much
noisier than a centipawn score, which is why this wants tens of thousands of games rather than
tens of thousands of positions.

    uv run python -m tools.selfplay --games 400 --out data/outcomes.jsonl

Positions from the opening are skipped, since they are shared by many games and their result
says more about what followed than about them.
"""

import argparse
import io
import json
import multiprocessing
from pathlib import Path

import chess
import chess.pgn

from harness.referee import play_match
from harness.sandbox import local

# A void game is one where both agents broke, so it says nothing about the positions in it.
RESULTS: dict[str, float] = {"white": 1.0, "black": 0.0, "draw": 0.5}


def one(task: tuple[Path, Path, str, int, int, int]) -> list[tuple[str, float]]:
    """Play one game and return every sampled position with the result, from White's view."""
    white, black, fen, base_ms, increment_ms, skip = task
    outcome = play_match(local(white), local(black), base_ms, increment_ms, start_fen=fen)
    score = RESULTS.get(outcome.result)
    if score is None:
        return []

    game = chess.pgn.read_game(io.StringIO(outcome.pgn))
    if game is None:
        return []
    board = game.board()
    rows: list[tuple[str, float]] = []
    for ply, move in enumerate(game.mainline_moves()):
        board.push(move)
        if ply >= skip and not board.is_check():
            rows.append((board.fen(), score))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate outcome-labelled positions.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--white", type=Path, default=Path("."))
    parser.add_argument("--black", type=Path, default=Path("."))
    parser.add_argument("--openings", type=Path, default=Path("data/openings.txt"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--base-ms", type=int, default=4_000)
    parser.add_argument("--increment-ms", type=int, default=40)
    parser.add_argument("--skip-plies", type=int, default=8)
    parser.add_argument(
        "--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2 - 2)
    )
    arguments = parser.parse_args()

    openings = [
        line.strip()
        for line in arguments.openings.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    white = arguments.white.resolve()
    black = arguments.black.resolve()
    tasks = [
        (
            white if index % 2 == 0 else black,
            black if index % 2 == 0 else white,
            openings[index % len(openings)],
            arguments.base_ms,
            arguments.increment_ms,
            arguments.skip_plies,
        )
        for index in range(arguments.games)
    ]
    workers = max(1, min(arguments.workers, len(tasks)))
    arguments.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    print(f"{len(tasks)} games, {workers} at a time")
    with (
        arguments.out.open("w", encoding="utf-8") as sink,
        multiprocessing.Pool(workers) as pool,
    ):
        for done, rows in enumerate(pool.imap_unordered(one, tasks), start=1):
            for fen, score in rows:
                sink.write(json.dumps({"fen": fen, "result": score}) + "\n")
                written += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} games, {written:,} positions")

    print(f"{written:,} outcome-labelled positions written to {arguments.out}")


if __name__ == "__main__":
    main()
