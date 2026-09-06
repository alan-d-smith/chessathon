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
from collections.abc import Iterator
from pathlib import Path

import chess
import chess.pgn

from harness.referee import play_match
from harness.sandbox import local

# A void game is one where both agents broke, so it says nothing about the positions in it.
RESULTS: dict[str, float] = {"white": 1.0, "black": 0.0, "draw": 0.5}
LINE_END = chr(10)
# A sentinel row for a game the harness could not finish, counted and then dropped.
FAILED = ""


def row(fen: str, score: float, game_id: int) -> dict[str, object]:
    """One record. The game id is what lets a holdout split by game rather than by
    position, which matters because every position in a game carries the same label."""
    return {"fen": fen, "result": score, "game": game_id}


def one(task: tuple[Path, Path, str, int, int, int, int]) -> list[tuple[str, float, int]]:
    """Play one game and return every sampled position with the result, from White's view.

    Every position carries the id of the game it came from, because they all share that one
    game's label. Without it a holdout split by position lands near-duplicates on both sides
    and scores the fit against labels it has effectively already been shown.
    """
    white, black, fen, base_ms, increment_ms, skip, game_id = task
    try:
        outcome = play_match(local(white), local(black), base_ms, increment_ms, start_fen=fen)
    # A game can fail for reasons that have nothing to do with the position: a subprocess that
    # will not start, a pipe that closes under load. Raised out of a pool worker that ends the
    # whole run, which for a self-play process meant to last for days is the difference between
    # losing one game and losing the night. One game is worth nothing; the run is worth a lot.
    except Exception:
        return [(FAILED, 0.0, game_id)]
    score = RESULTS.get(outcome.result)
    if score is None:
        return []

    game = chess.pgn.read_game(io.StringIO(outcome.pgn))
    if game is None:
        return []
    board = game.board()
    rows: list[tuple[str, float, int]] = []
    for ply, move in enumerate(game.mainline_moves()):
        board.push(move)
        if ply >= skip and not board.is_check():
            rows.append((board.fen(), score, game_id))
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
        "--fens",
        type=Path,
        help="also write the bare positions here, for the labeller to score",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="add to what is already there instead of starting again",
    )
    # Only one side of a game thinks at a time; the other is blocked waiting for a move. A
    # concurrent game therefore costs about one core, not two, which is why half the cores
    # left the machine half idle. Four fifths keeps it busy with headroom to spare.
    parser.add_argument(
        "--workers", type=int, default=max(1, int(multiprocessing.cpu_count() * 0.8))
    )
    arguments = parser.parse_args()

    openings = [
        line.strip()
        for line in arguments.openings.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    white = arguments.white.resolve()
    black = arguments.black.resolve()
    # Streamed, not built. A run meant to last until it is stopped asks for millions of games,
    # and materialising that many tuples of paths and positions costs gigabytes in this process
    # alone -- enough that the agent subprocesses started failing, which arrives as a write to
    # a dead pipe rather than as anything mentioning memory.
    def stream() -> Iterator[tuple[Path, Path, str, int, int, int, int]]:
        for index in range(arguments.games):
            yield (
                white if index % 2 == 0 else black,
                black if index % 2 == 0 else white,
                openings[index % len(openings)],
                arguments.base_ms,
                arguments.increment_ms,
                arguments.skip_plies,
                index,
            )

    workers = max(1, min(arguments.workers, arguments.games))
    arguments.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    # Game ids have to stay unique across rounds, or a holdout split by game would put two
    # different games' positions in the same group.
    offset = 0
    mode = "a" if arguments.append else "w"
    if arguments.append and arguments.out.is_file():
        with arguments.out.open(encoding="utf-8") as existing:
            for line in existing:
                try:
                    offset = max(offset, json.loads(line).get("game", 0) + 1)
                except json.JSONDecodeError:
                    continue
        print(f"appending after {offset} games already recorded")

    print(f"{arguments.games} games, {workers} at a time")
    fens = arguments.fens.open(mode, encoding="utf-8") if arguments.fens else None
    with (
        arguments.out.open(mode, encoding="utf-8") as sink,
        multiprocessing.Pool(workers) as pool,
    ):
        broken = 0
        for done, rows in enumerate(pool.imap_unordered(one, stream(), chunksize=1), start=1):
            if rows and rows[0][0] == FAILED:
                broken += 1
                rows = []
            for fen, score, game_id in rows:
                sink.write(json.dumps(row(fen, score, game_id + offset)) + LINE_END)
                if fens is not None:
                    fens.write(fen + LINE_END)
                written += 1
            sink.flush()
            if fens is not None:
                fens.flush()
            if done % 50 == 0:
                print(f"  {done}/{arguments.games} games, {written:,} positions, "
                      f"{broken} lost to the harness ({broken / done * 100:.1f}%)")

    if fens is not None:
        fens.close()
    print(f"{written:,} outcome-labelled positions written to {arguments.out}")


if __name__ == "__main__":
    main()
