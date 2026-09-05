"""Improve the agent unattended: generate, label, retrain, play, promote, repeat.

One round is the whole cycle. Positions come from randomised Stockfish playouts, get labelled,
join everything gathered so far, and the network is retrained from the current champion's
weights rather than from noise. The candidate then has to beat the champion over real games
before it replaces it, judged by the same SPRT that judges everything else here.

    uv run python -m tools.loop --rounds 20 --games 4000

Nothing is promoted on a training loss. A network that fits the labels better and loses the
match is worse, and that has already happened twice: a fit that improved its holdout by 17%
lost by 359 Elo. The match is the only thing with a vote.

The champion lives in baselines/champion and the loop only ever writes there. Shipping stays a
deliberate act: copy the champion over agent.py when you are satisfied with it.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

CHAMPION = Path("baselines/champion")
CANDIDATE = Path("data/candidate")
POOL = Path("data/loop_positions.txt")
LABELS = Path("data/loop_labelled.jsonl")
NET = Path("data/nets/loop_net.npz")
LOG = Path("data/loop_log.txt")


def run(command: list[str], quiet: bool = False) -> tuple[int, str]:
    """Run one stage. A stage that fails should not take the loop down with it."""
    finished = subprocess.run(
        [sys.executable, "-m", *command],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (finished.stdout or "") + (finished.stderr or "")
    if not quiet:
        for line in output.strip().splitlines()[-3:]:
            print(f"      {line}")
    return finished.returncode, output


def note(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


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
    parser.add_argument("--games", type=int, default=4000, help="playout games a round")
    parser.add_argument("--nodes", type=int, default=50_000, help="labelling depth")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--priority", type=float, default=0.6)
    parser.add_argument("--workers", type=int, default=48)
    arguments = parser.parse_args()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    NET.parent.mkdir(parents=True, exist_ok=True)
    prepare_champion(arguments.hidden)
    note(f"loop starting: {arguments.rounds} rounds, {arguments.games} games a round")

    for round_number in range(1, arguments.rounds + 1):
        started = time.monotonic()
        note(f"round {round_number}: generating positions")
        code, _ = run(
            [
                "tools.gen_positions",
                "--games", str(arguments.games),
                "--workers", str(arguments.workers),
                "--out", str(POOL),
                "--playout-nodes", "6000",
                "--every", "4",
                "--seed", str(round_number * 7919),
            ]
        )
        if code != 0 or not POOL.is_file():
            note(f"round {round_number}: generation failed, skipping round")
            continue

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
            ]
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
                *(["--warm", str(NET)] if NET.is_file() else []),
            ],
            quiet=True,
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
        )
        verdict = [line for line in output.splitlines() if line.startswith(("elo ", "sprt:"))]
        for line in verdict:
            note(f"round {round_number}: {line.strip()}")

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
