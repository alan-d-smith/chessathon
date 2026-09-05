"""Train the small network the agent evaluates positions with, and ship it as float32 weights.

The hand-written evaluation is a dot product over features somebody chose. A network chooses
its own, which is the whole point: it can notice that a knight on f5 matters more when the enemy
king is castled short, and no amount of tuning a piece-square table will ever express that.

The shape is deliberately modest. Evaluation is called at every leaf, so the budget is a couple
of microseconds; 768 inputs into 32 hidden units into one output measures about 2.4us per call
through numba, which is roughly six percent of what a node already costs. A deeper net would
evaluate better and search less, and at this depth searching less is the more expensive half.

    uv run python -m tools.nnue --data data/labelled.jsonl --out weights/net.npz

Features are relative to the side to move, with the board mirrored when Black is to move, so one
set of weights serves both colours and the evaluation is symmetric by construction rather than
by luck. Training against engine labels is explicitly permitted; the network that ships is ours.
"""

import argparse
import json
import math
from pathlib import Path

import chess
import numpy as np
import torch

SQUARES = 64
PIECES = 6
# Ours and theirs, six piece types, sixty-four squares.
INPUTS = 2 * PIECES * SQUARES
SCALE = math.log(10.0) / 400.0
CLAMP_CP = 1500
MATE_CP = 2500
MIRROR = [chess.square_mirror(square) for square in range(SQUARES)]


def indices(board: chess.Board) -> list[int]:
    """Feature indices for this position, from the side to move's point of view."""
    mover = board.turn
    found: list[int] = []
    for colour in (chess.WHITE, chess.BLACK):
        mine = board.occupied_co[colour]
        owner = 0 if colour == mover else 1
        for offset, piece in enumerate(
            (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
        ):
            squares = board.pieces_mask(piece, colour) & mine
            while squares:
                lowest = squares & -squares
                squares ^= lowest
                square = lowest.bit_length() - 1
                # Mirror for Black so "our first rank" is always the bottom of the board.
                view = square if mover == chess.WHITE else MIRROR[square]
                found.append(owner * (PIECES * SQUARES) + offset * SQUARES + view)
    return found


def load(path: Path, limit: int | None) -> tuple[list[list[int]], list[float]]:
    """Positions as feature indices, and the label in centipawns from the side to move."""
    rows: list[list[int]] = []
    targets: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            mate, centipawns = record.get("mate"), record.get("cp")
            if mate is not None:
                score = MATE_CP if mate > 0 else -MATE_CP
            elif centipawns is not None:
                score = max(-CLAMP_CP, min(CLAMP_CP, int(centipawns)))
            else:
                continue
            board = chess.Board(record["fen"])
            # Labels are already relative to the side to move, and so are the features.
            rows.append(indices(board))
            targets.append(float(score))
            if limit and len(rows) >= limit:
                break
    return rows, targets


def densify(rows: list[list[int]]) -> torch.Tensor:
    matrix = torch.zeros((len(rows), INPUTS), dtype=torch.float32)
    for row, found in enumerate(rows):
        matrix[row, torch.tensor(found, dtype=torch.long)] = 1.0
    return matrix


class Network(torch.nn.Module):
    """768 -> hidden -> 1, output in centipawns so the search can mix it with mate scores."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(INPUTS, hidden)
        self.output = torch.nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.hidden(features))).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the evaluation network.")
    parser.add_argument("--data", type=Path, default=Path("data/labelled.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("weights/net.npz"))
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4_096)
    parser.add_argument("--rate", type=float, default=1e-3)
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warm", type=Path, help="start from an existing net instead of noise")
    parser.add_argument(
        "--priority",
        type=float,
        default=0.0,
        help="prioritised sampling exponent; 0 is uniform, 0.6 is a typical value",
    )
    parser.add_argument(
        "--correction",
        type=float,
        default=0.4,
        help="importance-sampling exponent that undoes the bias prioritising introduces",
    )
    arguments = parser.parse_args()

    rows, targets = load(arguments.data, arguments.limit)
    print(f"{len(rows):,} positions, {arguments.hidden} hidden units")
    if len(rows) < 5_000:
        raise SystemExit("not enough positions to train anything trustworthy")

    features = densify(rows)
    wanted = torch.sigmoid(torch.tensor(targets, dtype=torch.float32) * SCALE)
    count = len(rows)
    order = torch.randperm(count)
    split = int(count * (1.0 - arguments.holdout))
    train, test = order[:split], order[split:]

    net = Network(arguments.hidden)
    if arguments.warm and arguments.warm.is_file():
        with np.load(arguments.warm) as data:
            if data["hidden_bias"].shape[0] == arguments.hidden:
                with torch.no_grad():
                    net.hidden.weight.copy_(torch.tensor(data["hidden_weight"].T))
                    net.hidden.bias.copy_(torch.tensor(data["hidden_bias"]))
                    net.output.weight.copy_(torch.tensor(data["output_weight"]).reshape(1, -1))
                    net.output.bias.copy_(torch.tensor(data["output_bias"]))
                print(f"  warm started from {arguments.warm}")
    optimiser = torch.optim.Adam(net.parameters(), lr=arguments.rate)

    best = float("inf")
    kept = {name: tensor.detach().clone() for name, tensor in net.state_dict().items()}
    # Uniform to begin with; prioritised sampling replaces this once there are errors to rank by.
    priority = torch.ones(len(train), dtype=torch.float64)
    for epoch in range(arguments.epochs):
        net.train()
        if arguments.priority > 0.0 and epoch > 0:
            chance = priority / priority.sum()
            picked = torch.multinomial(chance, len(train), replacement=True)
            shuffled = train[picked]
            # Sampling hard positions more often skews the gradient towards them, and the
            # hardest positions are often the ones the labeller got wrong. The importance
            # weight is what buys the focus without inheriting the bias that comes with it.
            weight = (1.0 / (len(train) * chance[picked])) ** arguments.correction
            batch_weight = (weight / weight.max()).float()
        else:
            shuffled = train[torch.randperm(len(train))]
            batch_weight = torch.ones(len(train), dtype=torch.float32)

        total = 0.0
        for start in range(0, len(shuffled), arguments.batch):
            span = slice(start, start + arguments.batch)
            rows_batch = shuffled[span]
            optimiser.zero_grad()
            predicted = torch.sigmoid(net(features[rows_batch]) * SCALE)
            errors = (predicted - wanted[rows_batch]) ** 2
            loss = (errors * batch_weight[span]).mean()
            loss.backward()
            optimiser.step()
            total += float(loss) * len(rows_batch)

        if arguments.priority > 0.0:
            # Re-rank on what the net now gets wrong, so the next pass chases current errors
            # rather than the ones it has already learned away.
            with torch.no_grad():
                gap = (
                    torch.sigmoid(net(features[train]) * SCALE) - wanted[train]
                ).abs().double()
            priority = (gap + 1e-4) ** arguments.priority
        net.eval()
        with torch.no_grad():
            held = float(
                torch.nn.functional.mse_loss(
                    torch.sigmoid(net(features[test]) * SCALE), wanted[test]
                )
            )
        if held < best:
            best = held
            kept = {name: tensor.detach().clone() for name, tensor in net.state_dict().items()}
        print(
            f"  epoch {epoch + 1:3}/{arguments.epochs}  train {total / len(shuffled):.6f}  "
            f"holdout {held:.6f}{'  *' if held == best else ''}"
        )

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.out,
        hidden_weight=kept["hidden.weight"].numpy().T.astype(np.float32),
        hidden_bias=kept["hidden.bias"].numpy().astype(np.float32),
        output_weight=kept["output.weight"].numpy().reshape(-1).astype(np.float32),
        output_bias=kept["output.bias"].numpy().astype(np.float32),
    )
    size = arguments.out.stat().st_size
    print(f"best holdout {best:.6f}; wrote {arguments.out} ({size:,} bytes)")


if __name__ == "__main__":
    main()
