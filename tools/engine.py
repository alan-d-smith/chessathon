"""Locating and starting the local Stockfish, shared by the dataset tools.

Nothing here is shipped. Labelling positions with an existing engine is explicitly allowed;
what the ban covers is the contents of the zip.
"""

import os
import shutil
from pathlib import Path

import chess.engine

CANDIDATES = (
    Path.home() / "PycharmProjects/stockfish/bin/stockfish/stockfish-windows-x86-64-avx2.exe",
    Path("/usr/local/bin/stockfish"),
    Path("/usr/games/stockfish"),
)


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


def spawn(hash_mb: int = 64, timeout: float = 120.0) -> chess.engine.SimpleEngine:
    """One engine, one thread. Parallelism here is one process per core, not one engine.

    The startup timeout is generous because these tools run beside a loop that keeps every
    core busy, and the default ten seconds is not enough for a process to finish handshaking
    on a machine under that kind of load.
    """
    engine = chess.engine.SimpleEngine.popen_uci(binary(), timeout=timeout)
    engine.configure({"Threads": 1, "Hash": hash_mb})
    return engine
