"""Improve the agent unattended: play, label, retrain, test, promote, repeat.

One round is the whole cycle. The champion plays a few thousand games against itself; every
position in them is kept and tagged with how that game finished; Stockfish scores the same
positions; and the network is retrained on both signals at once, warm started from the
champion's own weights. The candidate then has to beat the champion over real games before it
replaces it, judged by the same SPRT that judges everything else here.

    uv run python -m tools.loop --rounds 1000 --games 3000

Two signals rather than one, because each is weak where the other is strong. A Stockfish score
is dense and precise but is an opinion formed by a search far deeper than ours, and fitting it
teaches the network to agree with Stockfish rather than to win. A game result is ground truth
with no opinion in it at all, but it is one number per game, so on its own it needs tens of
thousands of games before it says anything. Trained on 800 games alone it produced weights that
scored bishop pair at minus seventy and lost by 359 Elo.

Nothing is promoted on a training loss. A network that fits the labels better and loses the
match is worse, and that has already happened twice: a fit that improved its holdout by 17%
lost by 359 Elo. The match is the only thing with a vote.

The champion lives in baselines/champion and the loop only ever writes there. Shipping stays a
deliberate act: copy the champion over agent.py when you are satisfied with it.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

CHAMPION = Path("baselines/champion")
CANDIDATE = Path("data/candidate")
POOL = Path("data/loop_positions.txt")
OUTCOMES = Path("data/loop_outcomes.jsonl")
LABELS = Path("data/loop_labelled.jsonl")
NET = Path("data/nets/loop_net.npz")
# The best candidate so far, by match result rather than by training loss. Warm starting from
# the previous round regardless of whether it was any good is a random walk; starting from the
# best one measured makes it a climb.
BEST = Path("data/nets/loop_best.npz")
BEST_SCORE = Path("data/nets/loop_best.txt")
LOG = Path("data/loop_log.txt")


def run(
    command: list[str], quiet: bool = False, python: str = "", limit: float = 1800.0
) -> tuple[int, str]:
    """Run one stage under a deadline. A stuck stage must not take the night with it.

    `python` lets the training stage run under a different interpreter. The agent must match
    the competition environment exactly, which is CPU-only torch; fitting the network has no
    such constraint and is much faster on a GPU, so it gets its own interpreter when one exists.

    The deadline is the point of this function. A pool of workers driving hundreds of agent
    subprocesses can deadlock, and without one the loop simply stops for the rest of the night
    with every process idle: that happened, and cost an hour before it was noticed. Killing the
    stage is not enough either, because its workers and their agents are grandchildren and
    outlive it, so the whole tree goes.
    """
    process = subprocess.Popen(
        [python or sys.executable, "-m", *command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = process.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        kill_tree(process.pid)
        output, _ = process.communicate()
        print(f"      stage passed {limit / 60:.0f} min and was killed with its children")
        return 1, output or ""

    if not quiet:
        for line in (output or "").strip().splitlines()[-3:]:
            print(f"      {line}")
    return process.returncode, output or ""


def kill_tree(pid: int) -> None:
    """Kill a stage and everything it spawned. Killing only the parent orphans the workers."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False
        )
    else:
        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(pid), signal.SIGKILL)


def note(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def elo_of(verdict: list[str]) -> float | None:
    """The Elo line from a match report, as a number."""
    for line in verdict:
        if line.startswith("elo "):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def read_best() -> float:
    """How good the best candidate so far was. Nothing yet means anything is an improvement."""
    if not BEST_SCORE.is_file():
        return float("-inf")
    try:
        return float(BEST_SCORE.read_text(encoding="utf-8").strip())
    except ValueError:
        return float("-inf")


def prepare_champion(hidden: int) -> None:
    """Seed the champion from whatever is currently shipped, if it does not exist yet."""
    if (CHAMPION / "agent.py").is_file():
        return
    CHAMPION.mkdir(parents=True, exist_ok=True)
    shutil.copy("agent.py", CHAMPION / "agent.py")
    shutil.copy("weights.py", CHAMPION / "weights.py")
    note(f"champion seeded from the shipped agent ({hidden} hidden units to be trained)")


def build_candidate(source: Path, net: Path) -> None:
    """A candidate is the network-using agent plus the freshly trained weights."""
    if CANDIDATE.exists():
        shutil.rmtree(CANDIDATE)
    (CANDIDATE / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy(source / "agent.py", CANDIDATE / "agent.py")
    shutil.copy(source / "weights.py", CANDIDATE / "weights.py")
    shutil.copy(net, CANDIDATE / "weights" / "net.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the improvement loop until stopped.")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--games", type=int, default=3000, help="self-play games a round")
    parser.add_argument("--nodes", type=int, default=50_000, help="labelling depth")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--priority", type=float, default=0.6)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument(
        "--stage-limit",
        type=float,
        default=1800.0,
        help="seconds any one stage may take before it is killed and the round abandoned",
    )
    parser.add_argument("--openings", type=Path, default=Path("data/bigopenings.txt"))
    parser.add_argument(
        "--train-python",
        type=str,
        default="",
        help="interpreter for the training stage, e.g. a venv with CUDA torch",
    )
    parser.add_argument(
        "--outcome-weight",
        type=float,
        default=0.7,
        help="share of the training target taken from the engine score rather than the result",
    )
    arguments = parser.parse_args()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    NET.parent.mkdir(parents=True, exist_ok=True)

    # Resolved to an absolute path, because Windows will not find a relative one with forward
    # slashes and fails the spawn outright. A missing interpreter falls back to this one rather
    # than taking the night down: training slower is better than not training.
    trainer = ""
    if arguments.train_python:
        candidate = Path(arguments.train_python).resolve()
        if candidate.is_file():
            trainer = str(candidate)
        else:
            print(f"no interpreter at {candidate}, training on this one instead")
    arguments.train_python = trainer
    prepare_champion(arguments.hidden)
    note(f"loop starting: {arguments.rounds} rounds, {arguments.games} games a round")

    for round_number in range(1, arguments.rounds + 1):
        started = time.monotonic()
        # The champion plays its own games and every position in them is kept, tagged with how
        # that game finished. This is the part that makes it self-play rather than distillation:
        # the training signal comes from what actually won, not only from what Stockfish thinks.
        note(f"round {round_number}: the champion plays {arguments.games} games")
        code, _ = run(
            [
                "tools.selfplay",
                "--white", str(CHAMPION),
                "--black", str(CHAMPION),
                "--games", str(arguments.games),
                "--workers", str(arguments.workers),
                "--openings", str(arguments.openings),
                "--out", str(OUTCOMES),
                "--fens", str(POOL),
                "--append",
                "--base-ms", "3000",
                "--increment-ms", "30",
            ],
            limit=arguments.stage_limit,
        )
        if code != 0 or not POOL.is_file():
            note(f"round {round_number}: self-play failed, skipping round")
            continue
        played = sum(1 for _ in OUTCOMES.open(encoding="utf-8"))
        note(f"round {round_number}: {played:,} positions from games played so far")

        note(f"round {round_number}: labelling")
        # Appends and skips what it already has, so the pool grows across rounds.
        code, _ = run(
            [
                "tools.label",
                "--in", str(POOL),
                "--out", str(LABELS),
                "--nodes", str(arguments.nodes),
                "--workers", str(arguments.workers),
                "--hash-mb", "32",
            ],
            limit=arguments.stage_limit,
        )
        if code != 0:
            note(f"round {round_number}: labelling failed, skipping round")
            continue
        total = sum(1 for _ in LABELS.open(encoding="utf-8"))
        note(f"round {round_number}: {total:,} labelled positions in the pool")

        note(f"round {round_number}: training")
        code, output = run(
            [
                "tools.nnue",
                "--data", str(LABELS),
                "--out", str(NET),
                "--hidden", str(arguments.hidden),
                "--epochs", str(arguments.epochs),
                "--rate", "3e-3",
                "--priority", str(arguments.priority),
                # Both signals: the engine score for precision, the game result for truth.
                "--outcomes", str(OUTCOMES),
                "--outcome-weight", str(arguments.outcome_weight),
                *(["--warm", str(BEST)] if BEST.is_file() else []),
            ],
            quiet=True,
            python=arguments.train_python,
            limit=arguments.stage_limit,
        )
        if code != 0 or not NET.is_file():
            note(f"round {round_number}: training failed, skipping round")
            continue
        holdout = [line for line in output.splitlines() if "best holdout" in line]
        note(f"round {round_number}: {holdout[-1].strip() if holdout else 'trained'}")

        note(f"round {round_number}: playing the champion")
        build_candidate(CHAMPION, NET)
        code, output = run(
            [
                "tools.match",
                "--agent", str(CANDIDATE),
                "--opponent", str(CHAMPION),
                "--openings", "data/openings.txt",
            ],
            quiet=True,
            limit=arguments.stage_limit,
        )
        verdict = [line for line in output.splitlines() if line.startswith(("elo ", "sprt:"))]
        for line in verdict:
            note(f"round {round_number}: {line.strip()}")

        # Keep the best candidate measured, not the most recent one trained, and start the next
        # round from it. Round 2 fitted the labels better than round 1 and played 38 Elo worse,
        # which is exactly the case where following the training loss walks downhill.
        measured = elo_of(verdict)
        if measured is not None:
            previous = read_best()
            if measured > previous:
                shutil.copy(NET, BEST)
                BEST_SCORE.write_text(f"{measured:.1f}" + chr(10), encoding="utf-8")
                note(f"round {round_number}: best candidate so far at {measured:+.0f} elo")
            else:
                note(f"round {round_number}: keeping the {previous:+.0f} elo net to build on")

        if any("accepted" in line for line in verdict):
            shutil.copy(NET, Path("data/nets") / f"champion_r{round_number}.npz")
            (CHAMPION / "weights").mkdir(parents=True, exist_ok=True)
            shutil.copy(NET, CHAMPION / "weights" / "net.npz")
            shutil.copy(CANDIDATE / "agent.py", CHAMPION / "agent.py")
            note(f"round {round_number}: PROMOTED, the champion now uses the network")
        else:
            note(f"round {round_number}: rejected, champion unchanged")

        note(f"round {round_number}: done in {(time.monotonic() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
