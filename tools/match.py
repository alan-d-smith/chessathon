"""Play one agent against another over an opening suite, in parallel, and report the gap.

harness/arena.py starts every game from the same position, which is fine against a baseline
that picks randomly among equal moves and useless between two deterministic agents: the same
game is replayed and the score is one result wearing a sample size. This plays each opening
once with each colour, so both agents get both sides of every position, and runs the games
across cores because measuring a 20 Elo change needs hundreds of games rather than twenty.

    uv run python -m tools.match --agent . --opponent baselines/v1 --openings data/openings.txt

By default the match runs as an SPRT against "no better" versus "worth at least 15 Elo", and
stops the moment either is established, so a clear result costs a hundred games and only a
marginal one costs the full suite. Pass --no-sprt to play every game regardless, and --elo0 /
--elo1 to move the bounds. The Elo estimate carries a 95% interval alongside it; an interval
that spans zero means the change is unproven whatever the score looks like.
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


def expected(difference: float) -> float:
    """The score an agent this many Elo ahead is expected to take."""
    return 1.0 / (1.0 + 10.0 ** (-difference / 400.0))


def llr(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    """Log-likelihood ratio between "no better than elo0" and "at least elo1".

    This is what lets a match stop as soon as the answer is known rather than at a round number
    of games picked in advance. A change that is clearly good crosses the upper bound in a
    hundred games; one that is clearly bad is rejected just as fast; only the genuinely
    marginal ones cost a full run, which is exactly where the games are worth spending.
    """
    games = wins + draws + losses
    if games == 0 or (wins == 0 and draws == 0) or (losses == 0 and draws == 0):
        return 0.0
    win, draw = wins / games, draws / games
    score = win + draw / 2.0
    variance = (win + draw / 4.0) - score * score
    if variance <= 0.0:
        return 0.0
    score0, score1 = expected(elo0), expected(elo1)
    return games * (score1 - score0) * (2.0 * score - score0 - score1) / (2.0 * variance)


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
    # Only one side of a game thinks at a time; the other is blocked waiting for a move. A
    # concurrent game therefore costs about one core, not two, which is why half the cores
    # left the machine half idle. Four fifths keeps it busy with headroom to spare.
    parser.add_argument(
        "--workers", type=int, default=max(1, int(multiprocessing.cpu_count() * 0.8))
    )
    parser.add_argument("--elo0", type=float, default=0.0, help="SPRT null: no improvement")
    parser.add_argument("--elo1", type=float, default=15.0, help="SPRT alternative: worth taking")
    parser.add_argument("--no-sprt", action="store_true", help="play every game regardless")
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
    workers = max(1, min(arguments.workers, len(tasks)))
    print(f"{len(tasks)} games from {len(openings)} openings, {workers} at a time")

    # Wald's bounds for a 5% chance each of accepting a bad change or rejecting a good one.
    upper = math.log(0.95 / 0.05)
    lower = math.log(0.05 / 0.95)

    points = 0.0
    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    faults: dict[str, int] = {}
    stopped = ""
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

            ratio = llr(wins, draws, losses, arguments.elo0, arguments.elo1)
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}  +{wins} ={draws} -{losses}  llr {ratio:+.2f}")
            if not arguments.no_sprt and done >= 40:
                if ratio >= upper:
                    stopped = f"accepted after {done} games, llr {ratio:+.2f} >= {upper:.2f}"
                    break
                if ratio <= lower:
                    stopped = f"rejected after {done} games, llr {ratio:+.2f} <= {lower:.2f}"
                    break
        if stopped:
            pool.terminate()

    games = wins + draws + losses
    score = points / games
    difference, interval = elo(score, games)
    print(f"\n{arguments.agent} vs {arguments.opponent}")
    print(f"+{wins} ={draws} -{losses} over {games} games, score {score:.1%}")
    print(f"elo {difference:+.0f} +/- {interval:.0f} (95%)")
    if stopped:
        print(f"sprt: {stopped}")
    else:
        ratio = llr(wins, draws, losses, arguments.elo0, arguments.elo1)
        print(f"sprt: inconclusive over the whole suite, llr {ratio:+.2f} in [{lower:.2f}, "
              f"{upper:.2f}] for elo0={arguments.elo0:.0f} elo1={arguments.elo1:.0f}")
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
