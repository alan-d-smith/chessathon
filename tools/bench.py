"""Search throughput on a fixed set of positions.

Strength work is guesswork without a repeatable number. This fixes the position and the time
per move, then reports the depth reached and the nodes per second, so a change that claims to
be faster has to show it before it costs a few hundred arena games to find out otherwise.

    uv run python -m tools.bench
    uv run python -m tools.bench --ms 3000 --profile

The positions are a spread of opening, middlegame and endgame rather than the start position
alone, because a search that only ever sees a full board tunes to the wrong thing.
"""

import argparse
import cProfile
import importlib
import pstats
import time
from types import ModuleType

import chess

POSITIONS: tuple[tuple[str, str], ...] = (
    ("start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("open", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"),
    ("middle", "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"),
    ("tactical", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
)


def load(name: str) -> ModuleType:
    module = importlib.import_module(name)
    return importlib.reload(module)


def run(agent: ModuleType, budget_ms: int) -> None:
    total_nodes = 0
    total_seconds = 0.0
    for label, fen in POSITIONS:
        # The agent budgets a share of the clock over the moves it expects to still play,
        # so the clock that yields a given budget depends on how far in the position is.
        clock = budget_ms * agent.remaining_moves(chess.Board(fen).fullmove_number)
        agent.reset_transposition()
        agent.seen.clear()
        # The agent infers the increment from how the clock moves between its turns. A bench
        # hands it the same clock every time, which reads as a huge increment unless reset.
        agent.last_clock_ms = None
        agent.increment_ms = float(agent.ASSUMED_INCREMENT_MS)
        started = time.monotonic()
        move = agent.get_move(fen, clock)
        elapsed = time.monotonic() - started
        total_nodes += agent.nodes
        total_seconds += elapsed
        rate = agent.nodes / elapsed if elapsed else 0.0
        print(
            f"{label:9} {move}  depth {agent.reached:>2}  {agent.nodes:>9,} nodes  "
            f"{elapsed:5.2f}s  {rate:>10,.0f} n/s"
        )

    rate = total_nodes / total_seconds if total_seconds else 0.0
    print(f"{'total':9} {'':4}  {total_nodes:>9,} nodes  {total_seconds:5.2f}s  {rate:>10,.0f} n/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure search throughput.")
    parser.add_argument("--agent", default="agent", help="module to import, e.g. agent")
    parser.add_argument("--ms", type=int, default=2000, help="budget per position")
    parser.add_argument("--profile", action="store_true", help="print the hottest functions")
    arguments = parser.parse_args()

    agent = load(arguments.agent)
    if not arguments.profile:
        run(agent, arguments.ms)
        return

    profiler = cProfile.Profile()
    profiler.enable()
    run(agent, arguments.ms)
    profiler.disable()
    print()
    pstats.Stats(profiler).sort_stats("tottime").print_stats(18)


if __name__ == "__main__":
    main()
