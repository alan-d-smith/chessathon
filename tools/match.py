"""Play one agent against another over an opening suite, in parallel, and report the gap.

harness/arena.py starts every game from the same position, which is fine against a baseline
that picks randomly among equal moves and useless between two deterministic agents: the same
game is replayed and the score is one result wearing a sample size. This plays each opening
once with each colour, so both agents get both sides of every position, and runs the games
across cores because measuring a 20 Elo change needs hundreds of games rather than twenty.

    uv run python -m tools.match --agent . --opponent baselines/v1 --openings data/openings.txt

The Elo estimate carries a 95% interval. If that interval spans zero the change is unproven,
which is the number that should decide whether it stays.
"""

import argparse
import math
import multiprocessing
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.sandbox import local

FAST_BASE_MS = 10_000
FAST_INCREMENT_MS = 100


def one(task: tuple[Path, Path, str, bool, int, int]) -> tuple[float, str, str]:
    """Score from the agent's point of view, how the game ended, and who broke if anyone.

    Blame matters more than the score does. A flag or a crash is a whole point, and a report
    that says only that one happened cannot tell an agent that is about to lose games on the
    platform from an opponent that is merely losing them here.
    """
    agent, opponent, fen, agent_is_white, base_ms, increment_ms = task
    white, black = (agent, opponent) if agent_is_white else (opponent, agent)
    outcome = play_match(local(white), local(black), base_ms, increment_ms, start_fen=fen)

    if outcome.result == "void":
        return 0.5, outcome.termination, "both"
    if outcome.result == "draw":
        return 0.5, outcome.termination, ""
    won = (outcome.result == "white") == agent_is_white
    broke = outcome.termination in FAILED_TERMINATIONS
    blame = ("opponent" if won else "agent") if broke else ""
    return (1.0 if won else 0.0), outcome.termination, blame


def elo(score: float, games: int) -> tuple[float, float]:
    """Elo difference and the half-width of a 95% interval on it."""
    if score <= 0.0:
        return -800.0, 0.0
    if score >= 1.0:
        return 800.0, 0.0
    difference = -400.0 * math.log10(1.0 / score - 1.0)
    # The spread of a single game's score, carried through the logistic at the measured point.
    deviation = math.sqrt(max(score * (1.0 - score), 1e-9) / games)
    slope = 400.0 / (math.log(10.0) * score * (1.0 - score))
    return difference, 1.96 * deviation * slope


def main() -> None:
    parser = argparse.ArgumentParser(description="Match two agents over an opening suite.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/v1"))
    parser.add_argument("--openings", type=Path, default=Path("data/openings.txt"))
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 3))
    arguments = parser.parse_args()

    openings = [
        line.strip() for line in arguments.openings.read_text(encoding="utf-8").splitlines()
    ]
    openings = [fen for fen in openings if fen]
    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()

    # Each opening twice, so a position that happens to favour White cannot favour one agent.
    tasks = [
        (agent, opponent, fen, as_white, arguments.base_ms, arguments.increment_ms)
        for fen in openings
        for as_white in (True, False)
    ]
    # Two agent processes per game, so a worker per core would oversubscribe by two.
    workers = max(1, min(arguments.workers, len(tasks)))
    print(f"{len(tasks)} games from {len(openings)} openings, {workers} at a time")

    points = 0.0
    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    faults: dict[str, int] = {}
    with multiprocessing.Pool(workers) as pool:
        results = pool.imap_unordered(one, tasks)
        for done, (score, termination, blame) in enumerate(results, start=1):
            points += score
            if score == 1.0:
                wins += 1
            elif score == 0.5:
                draws += 1
            else:
                losses += 1
            terminations[termination] = terminations.get(termination, 0) + 1
            if blame:
                faults[f"{blame}:{termination}"] = faults.get(f"{blame}:{termination}", 0) + 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}  +{wins} ={draws} -{losses}")

    games = len(tasks)
    score = points / games
    difference, interval = elo(score, games)
    print(f"\n{arguments.agent} vs {arguments.opponent}")
    print(f"+{wins} ={draws} -{losses} over {games} games, score {score:.1%}")
    print(f"elo {difference:+.0f} +/- {interval:.0f} (95%)")
    verdict = "unproven, the interval spans zero" if abs(difference) < interval else "significant"
    print(f"verdict: {verdict}")
    print("terminations: " + ", ".join(f"{n} {c}" for n, c in sorted(terminations.items())))
    if faults:
        print("faults: " + ", ".join(f"{n} {c}" for n, c in sorted(faults.items())))
        if any(name.startswith(("agent:", "both:")) for name in faults):
            print("OUR AGENT FAILED A GAME. Nothing else matters until that is fixed.")
    else:
        print("faults: none, every game finished on the board")


if __name__ == "__main__":
    main()
