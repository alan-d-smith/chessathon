"""Stockfish as a local sparring partner. This is never shipped.

Third party engines are banned in a submission and the check is retroactive, so this
directory exists purely to give `harness/arena.py` an opponent worth measuring against.
It cannot leak into an upload: `harness/package.py` packages `*.py` at the repo root plus
`weights/`, and nothing under `baselines/`.

Strength comes from the environment, so one directory is a whole ladder of opponents:

    SF_ELO=1320 uv run python -m harness.arena --opponent baselines/stockfish --games 20

    SF_BIN     engine path; else `stockfish` on PATH, else the usual install spots
    SF_ELO     UCI_Elo, 1320-3190. Default 1320, the floor
    SF_SKILL   Skill Level 0-20. Overrides SF_ELO. 20 is full strength, unlimited
    SF_NODES   fixed nodes per move. The reproducible limit: ignores the clock entirely
    SF_DEPTH   fixed depth per move
    SF_DIVISOR otherwise spend time_left_ms/SF_DIVISOR on each move. Default 40
"""

import atexit
import os
import shutil
from pathlib import Path

import chess
import chess.engine

CANDIDATES = (
    Path.home() / "PycharmProjects/stockfish/bin/stockfish/stockfish-windows-x86-64-avx2.exe",
    Path("/usr/local/bin/stockfish"),
    Path("/usr/games/stockfish"),
)
HASH_MB = 16
MIN_MOVE_S = 0.01


def binary() -> str:
    """Locate the engine, preferring an explicit SF_BIN over anything discovered."""
    override = os.environ.get("SF_BIN")
    if override:
        return override
    on_path = shutil.which("stockfish")
    if on_path:
        return on_path
    for candidate in CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("no stockfish binary found; set SF_BIN to its path")


def options() -> dict[str, object]:
    """One core and a small hash, so a game against it measures search, not hardware."""
    chosen: dict[str, object] = {"Threads": 1, "Hash": HASH_MB}
    skill = os.environ.get("SF_SKILL")
    if skill is not None:
        chosen["Skill Level"] = int(skill)
    else:
        chosen["UCI_LimitStrength"] = True
        chosen["UCI_Elo"] = int(os.environ.get("SF_ELO", "1320"))
    return chosen


def limit(time_left_ms: int) -> chess.engine.Limit:
    nodes = os.environ.get("SF_NODES")
    if nodes:
        return chess.engine.Limit(nodes=int(nodes))
    depth = os.environ.get("SF_DEPTH")
    if depth:
        return chess.engine.Limit(depth=int(depth))
    divisor = float(os.environ.get("SF_DIVISOR", "40"))
    return chess.engine.Limit(time=max(MIN_MOVE_S, time_left_ms / divisor / 1000.0))


# Start once, at import, inside the init budget, and keep it for the whole game. Exactly the
# lifecycle a real agent gets, so the clock it consumes is the clock it would really consume.
engine = chess.engine.SimpleEngine.popen_uci(binary())
engine.configure(options())
atexit.register(engine.close)


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    played = engine.play(board, limit(time_left_ms)).move
    if played is None:
        raise RuntimeError("stockfish resigned or returned no move")
    return played.uci()
