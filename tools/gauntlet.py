"""Play the agent against a ladder of external engines and place it on an absolute scale.

Every other measurement here is against our own previous version, which answers "is this better
than what I had" and not "how strong is this". Two versions that share a blind spot both fall
into it, so a self-play gain can be real and still not transfer to a different opponent. This
plays graded outside opposition instead, and reports the rating implied by each rung.

    uv run python -m tools.gauntlet --base-ms 30000 --increment-ms 300 --games 60

Stockfish's UCI_Elo is a calibration, not gospel, and it is tuned for longer time controls than
these. Treat a rung's implied rating as an anchor with a real error bar, not a certificate. What
it is good for is the thing self-play cannot do: telling us whether a change that won its match
against our last version also moved us against opposition that never shared our mistakes.
"""

import argparse
import math
import multiprocessing
import os
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.sandbox import local

# Each rung is a nominal strength and the environment that makes baselines/stockfish play there.
# A rating of None means the rung is a transfer check rather than an anchor: the score is
# reported and no rating is claimed from it.
LADDER: tuple[tuple[str, int | None, dict[str, str]], ...] = (
    ("sf-1320", 1320, {"SF_ELO": "1320"}),
    ("sf-1600", 1600, {"SF_ELO": "1600"}),
    ("sf-1900", 1900, {"SF_ELO": "1900"}),
    ("sf-2200", 2200, {"SF_ELO": "2200"}),
    ("sf-2500", 2500, {"SF_ELO": "2500"}),
)
OPPONENT = Path("baselines/stockfish")

# Engines that share no code with Stockfish, node-limited to somewhere near our own strength.
# Measuring only against one engine family risks fitting to that family's particular weaknesses:
# a change that beats Stockfish and nothing else has not necessarily made us stronger. These are
# a transfer check, so they claim no rating; only whether a gain shows up here too.
# Limited by depth, not nodes. `go nodes` is optional in the UCI spec and Weiss simply never
# answers it, which showed up as the opponent losing every game on time rather than as an
# error. Depth and movetime are the limits an arbitrary engine can be relied on to honour.
# Depth 6 is where Weiss scores about 50% against us, so the rung sits at our own level
# where a change actually shows, rather than at a whitewash in either direction.
DIVERSE: tuple[tuple[str, str, str], ...] = (("weiss", "WEISS_BIN", "6"),)


def diverse_rungs(depth: str) -> list[tuple[str, int | None, dict[str, str]]]:
    """Rungs for whichever outside engines are actually installed on this machine."""
    rungs: list[tuple[str, int | None, dict[str, str]]] = []
    for name, variable, default in DIVERSE:
        binary = os.environ.get(variable)
        if binary and Path(binary).is_file():
            rungs.append((name, None, {"SF_BIN": binary, "SF_DEPTH": depth or default}))
    return rungs


def one(task: tuple[Path, Path, str, bool, int, int, dict[str, str]]) -> tuple[float, str, str]:
    agent, opponent, fen, agent_is_white, base_ms, increment_ms, environment = task
    # The sandbox spawns with the worker's environment, so setting it here is what selects the
    # rung. One game per worker at a time, so this cannot race with another rung's settings.
    os.environ.update(environment)
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


def implied(rating: int, score: float, games: int) -> tuple[float, float]:
    """The rating this score implies against an opponent of the given strength, with an interval."""
    bounded = min(max(score, 0.5 / games), 1.0 - 0.5 / games)
    difference = -400.0 * math.log10(1.0 / bounded - 1.0)
    deviation = math.sqrt(max(bounded * (1.0 - bounded), 1e-9) / games)
    slope = 400.0 / (math.log(10.0) * bounded * (1.0 - bounded))
    return rating + difference, 1.96 * deviation * slope


def main() -> None:
    parser = argparse.ArgumentParser(description="Place the agent against outside engines.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--openings", type=Path, default=Path("data/openings.txt"))
    parser.add_argument("--games", type=int, default=60, help="per rung, split over both colours")
    parser.add_argument("--base-ms", type=int, default=30_000)
    parser.add_argument("--increment-ms", type=int, default=300)
    parser.add_argument("--rungs", type=str, default="", help="comma separated names to restrict")
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="also play the outside engines named by WEISS_BIN",
    )
    parser.add_argument("--diverse-depth", type=str, default="", help="depth cap for those")
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
    wanted = {name for name in arguments.rungs.split(",") if name}
    available = list(LADDER)
    if arguments.diverse:
        available += diverse_rungs(arguments.diverse_depth)
    rungs = [rung for rung in available if not wanted or rung[0] in wanted]
    agent = arguments.agent.resolve()
    opponent = OPPONENT.resolve()

    print(f"{arguments.agent} over {arguments.games} games a rung, "
          f"{arguments.base_ms / 1000:.0f}s + {arguments.increment_ms / 1000:.1f}s")
    estimates: list[tuple[float, float]] = []
    for name, rating, environment in rungs:
        tasks = [
            (
                agent,
                opponent,
                openings[index % len(openings)],
                index % 2 == 0,
                arguments.base_ms,
                arguments.increment_ms,
                environment,
            )
            for index in range(arguments.games)
        ]
        workers = max(1, min(arguments.workers, len(tasks)))
        points = 0.0
        wins = draws = losses = 0
        faults: dict[str, int] = {}
        with multiprocessing.Pool(workers) as pool:
            for score, termination, blame in pool.imap_unordered(one, tasks):
                points += score
                if score == 1.0:
                    wins += 1
                elif score == 0.5:
                    draws += 1
                else:
                    losses += 1
                if blame:
                    faults[f"{blame}:{termination}"] = faults.get(f"{blame}:{termination}", 0) + 1

        share = points / len(tasks)
        detail = ", ".join(f"{kind} {count}" for kind, count in sorted(faults.items()))
        note = f"  faults: {detail}" if faults else ""
        if rating is None:
            # A transfer check: the opponent's own rating is unknown, so a score is all this
            # says, and inventing a rating from it would be the sort of number that misleads.
            print(f"  {name:8} +{wins} ={draws} -{losses}  score {share:5.1%}  (no anchor){note}")
        else:
            rating_estimate, interval = implied(rating, share, len(tasks))
            estimates.append((rating_estimate, interval))
            print(
                f"  {name:8} +{wins} ={draws} -{losses}  score {share:5.1%}  "
                f"implies {rating_estimate:4.0f} +/- {interval:3.0f}{note}"
            )

    if estimates:
        # Inverse-variance weighting: the rungs we scored near 50% against say the most, and a
        # whitewash either way says almost nothing beyond "somewhere past here".
        weights = [1.0 / max(interval, 1.0) ** 2 for _, interval in estimates]
        combined = sum(w * r for w, (r, _) in zip(weights, estimates, strict=True)) / sum(weights)
        spread = math.sqrt(1.0 / sum(weights))
        print(f"\n  combined estimate {combined:.0f} +/- {spread:.0f} (95%)")


if __name__ == "__main__":
    main()
