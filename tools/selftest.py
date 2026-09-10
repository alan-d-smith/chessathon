"""Adversarial positions and clocks, checked for legality and for overrunning the budget.

A crash, an illegal move or a flag costs the whole game, and none of them show up in an Elo
number until they have already cost points. These are the shapes that break agents: a single
legal reply, a position one move from stalemate, promotion and en passant, and a clock small
enough that the search has to give up before it has finished anything.

    uv run python -m tools.selftest
"""

import sys
import time
from types import ModuleType

import chess

CASES: tuple[tuple[str, str], ...] = (
    ("mate in one", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
    ("only legal move", "7k/8/8/8/8/8/5KQ1/7r b - - 0 1"),
    ("stalemate next door", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
    ("promotion available", "8/P6k/8/8/8/8/6K1/8 w - - 0 1"),
    ("en passant available", "8/8/8/3pP3/8/8/6K1/6k1 w - d6 0 2"),
    ("king and pawn endgame", "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"),
    ("in check, must respond", "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"),
    (
        "wide open middlegame",
        "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9",
    ),
    ("bare kings", "8/8/4k3/8/8/4K3/8/8 w - - 0 1"),
)

# The referee measures wall time and a flag is a lost game, so the ceiling is what the agent
# claims it will spend plus room for the round trip, not merely "about right".
CLOCKS_MS: tuple[int, ...] = (120_000, 10_000, 1_000, 300, 120, 40)
OVERRUN_ALLOWANCE_MS = 250.0


def reset(agent: ModuleType) -> None:
    agent.reset_transposition()
    agent.pawn_cache.clear()
    agent.seen.clear()
    # History is a flat table rather than a dict, and may be a numpy array when the compiled
    # move generator is in use, so it is emptied by writing zeros over it rather than cleared.
    agent.history[:] = [0] * len(agent.history)
    agent.last_clock_ms = None
    agent.increment_ms = float(agent.ASSUMED_INCREMENT_MS)


def main() -> None:
    import agent

    failures: list[str] = []
    slowest = 0.0
    for label, fen in CASES:
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        for clock in CLOCKS_MS:
            reset(agent)
            if not legal:
                continue  # a finished game is never handed to an agent
            started = time.monotonic()
            try:
                uci = agent.get_move(fen, clock)
            # Anything escaping here is a lost game on the platform, so catch the lot.
            except Exception as failure:
                failures.append(f"{label} @ {clock}ms: raised {type(failure).__name__}: {failure}")
                continue
            spent_ms = (time.monotonic() - started) * 1000.0

            if chess.Move.from_uci(uci) not in legal:
                failures.append(f"{label} @ {clock}ms: illegal move {uci}")
            # The ceiling, not the ordinary budget: an unsettled search is allowed to
            # spend up to it, so that is the number an overrun has to be measured against.
            budget_ms = agent.budget_s(clock, board.fullmove_number)[1] * 1000.0
            overrun = spent_ms - budget_ms
            slowest = max(slowest, overrun)
            if overrun > OVERRUN_ALLOWANCE_MS:
                failures.append(
                    f"{label} @ {clock}ms: spent {spent_ms:.0f}ms on a {budget_ms:.0f}ms "
                    f"budget, {overrun:.0f}ms over"
                )

    checks = len(CASES) * len(CLOCKS_MS)
    print(f"{checks} position/clock pairs, worst overrun {slowest:+.0f}ms")
    if failures:
        for failure in failures:
            print(f"  FAIL {failure}")
        sys.exit(1)
    print("every reply legal and inside its budget")


if __name__ == "__main__":
    main()
