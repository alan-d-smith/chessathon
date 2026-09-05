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


def load(
    path: Path, limit: int | None, skip: int = 0
) -> tuple[list[list[int]], list[float], list[float], list[str]]:
    """Feature indices, the label in centipawns, and what the tuned tables score.

    All three come from one pass. Deriving any of them from a second pass over the file means
    two filters that have to agree forever, and they stopped agreeing the moment the feature
    cache let the first pass skip work the second still did.
    """
    import agent

    rows: list[list[int]] = []
    targets: list[float] = []
    tables: list[float] = []
    fens: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for _ in range(skip):
            if handle.readline() == "":
                break
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
            tables.append(float(agent.evaluate_tables(board)))
            fens.append(record["fen"])
            if limit and len(rows) >= limit:
                break
    return rows, targets, tables, fens


# A position has at most 32 pieces, so every row of features fits in 32 slots padded with an
# index whose embedding is pinned to zero. Storing it this way is 83MB rather than 2GB, and the
# first layer sums 32 rows instead of multiplying through 768 mostly-zero columns.
MAX_PIECES = 32
PAD = INPUTS


def blend_targets(
    fens: list[str], scores: list[float], outcomes: Path, weight: float
) -> tuple[list[float], list[int]]:
    """Mix the engine's score with what actually happened, and say which game each came from.

    This is how a network wants to be taught. The engine score is dense and precise but is only
    an opinion, and one formed by a search far deeper than ours. The game result is the ground
    truth and carries no opinion at all, but one bit of it per game is a thin signal. Blending
    keeps the precision of the first and anchors it to the second.

    Positions with no game attached keep the engine score alone and sit in their own group.
    """
    from_game: dict[str, tuple[float, int]] = {}
    if outcomes.is_file():
        with outcomes.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                from_game[record["fen"]] = (float(record["result"]), int(record.get("game", -1)))

    blended: list[float] = []
    groups: list[int] = []
    matched = 0
    loner = 10_000_000
    for fen, score in zip(fens, scores, strict=True):
        engine = 1.0 / (1.0 + math.exp(-score * SCALE))
        found = from_game.get(fen)
        if found is None:
            blended.append(engine)
            groups.append(loner)
            loner += 1
            continue
        result, game = found
        blended.append(weight * engine + (1.0 - weight) * result)
        groups.append(game)
        matched += 1
    print(f"  {matched:,} positions carry a game result as well as an engine score")
    return blended, groups


def cached(
    source: Path, limit: int | None
) -> tuple[list[list[int]], list[float], list[float], list[str]]:
    """Extract features, reusing anything already extracted from an earlier run.

    The labelled pool only ever grows, so the features of the first N positions never change.
    The improvement loop retrains every round, and without this it would re-parse every FEN it
    has ever seen, every round, for an answer it already had.
    """
    store = source.with_suffix(".features.npz")
    names = source.with_suffix(".fens.txt")
    rows: list[list[int]] = []
    targets: list[float] = []
    tables: list[float] = []
    fens: list[str] = []
    done = 0
    if store.is_file():
        try:
            with np.load(store) as data:
                packed, lengths = data["packed"], data["lengths"]
                scores, scored = data["targets"], data["tables"]
            rows = [packed[i, : lengths[i]].tolist() for i in range(len(lengths))]
            targets = scores.tolist()
            tables = scored.tolist()
            fens = names.read_text(encoding="utf-8").splitlines() if names.is_file() else []
            if len(fens) != len(rows):
                raise ValueError("cached positions and their names disagree")
            done = int(data_lines_consumed(store))
            print(f"  reused features for {len(rows):,} positions")
        except (OSError, KeyError, ValueError):
            rows, targets, tables, fens, done = [], [], [], [], 0

    fresh_rows, fresh_targets, fresh_tables, fresh_fens = load(source, limit, skip=done)
    rows.extend(fresh_rows)
    targets.extend(fresh_targets)
    tables.extend(fresh_tables)
    fens.extend(fresh_fens)
    if fresh_rows and limit:
        # A limited run is for debugging. Writing its partial result to the cache while
        # recording the whole file as consumed poisons every later run, which is exactly what
        # happened: a --limit 80000 smoke test left the next full run training on 80k rows.
        print(f"  {len(rows):,} positions (limited run, cache left alone)")
        return rows, targets, tables, fens

    if fresh_rows:
        print(f"  extracted features for {len(fresh_rows):,} new positions")
        widest = max((len(found) for found in rows), default=1)
        packed = np.zeros((len(rows), widest), dtype=np.int32)
        lengths = np.zeros(len(rows), dtype=np.int32)
        for index, found in enumerate(rows):
            packed[index, : len(found)] = found
            lengths[index] = len(found)
        np.savez(
            store,
            packed=packed,
            lengths=lengths,
            targets=np.asarray(targets, dtype=np.float32),
            tables=np.asarray(tables, dtype=np.float32),
            consumed=np.asarray([count_lines(source)], dtype=np.int64),
        )
        names.write_text(chr(10).join(fens) + chr(10), encoding="utf-8")
    return rows, targets, tables, fens


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def data_lines_consumed(store: Path) -> int:
    with np.load(store) as data:
        return int(data["consumed"][0]) if "consumed" in data else 0


def pack(rows: list[list[int]]) -> torch.Tensor:
    """Feature indices as a padded (positions, 32) table."""
    packed = torch.full((len(rows), MAX_PIECES), PAD, dtype=torch.long)
    for row, found in enumerate(rows):
        packed[row, : len(found)] = torch.tensor(found[:MAX_PIECES], dtype=torch.long)
    return packed


class Network(torch.nn.Module):
    """768 -> hidden -> 1, output in centipawns so the search can mix it with mate scores.

    The first layer is an EmbeddingBag rather than a Linear because the input is one-hot: with
    at most 32 features set, summing 32 weight rows is the same arithmetic as a 768-wide matrix
    multiply and about twenty times less of it. The exported weights are identical either way.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.embed = torch.nn.EmbeddingBag(INPUTS + 1, hidden, mode="sum", padding_idx=PAD)
        self.hidden_bias = torch.nn.Parameter(torch.zeros(hidden))
        self.output = torch.nn.Linear(hidden, 1)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        summed = self.embed(packed) + self.hidden_bias
        return self.output(torch.relu(summed)).squeeze(1)


def save(state: dict[str, torch.Tensor], path: Path, residual: bool = False) -> None:
    """Write the net in the shape agent.py reads: 768 by hidden, then the output layer.

    The residual flag travels with the weights. A net trained to correct the tables scores
    something entirely different from one trained to replace them, and loading either in the
    wrong mode would produce a plausible-looking evaluation that is quietly nonsense.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        residual=np.asarray([1 if residual else 0], dtype=np.int8),
        hidden_weight=state["embed.weight"][:INPUTS].numpy().astype(np.float32),
        hidden_bias=state["hidden_bias"].numpy().astype(np.float32),
        output_weight=state["output.weight"].numpy().reshape(-1).astype(np.float32),
        output_bias=state["output.bias"].numpy().astype(np.float32),
    )


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
        "--residual",
        action="store_true",
        help="learn what the tuned tables get wrong, rather than the whole evaluation",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        help="game results to blend into the target, from tools/selfplay.py",
    )
    parser.add_argument(
        "--outcome-weight",
        type=float,
        default=0.7,
        help="how much of the target is the engine score; the rest is the game result",
    )
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

    # Training is offline, so it is free to use hardware the agent never will: the platform
    # gives one CPU core and no GPU, but nothing stops the fitting from running on one here.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"  training on {torch.cuda.get_device_name(0)}")

    rows, targets, tables, fens = cached(arguments.data, arguments.limit)
    print(f"{len(rows):,} positions, {arguments.hidden} hidden units")
    if len(rows) < 5_000:
        raise SystemExit("not enough positions to train anything trustworthy")

    features = pack(rows).to(device)
    count = len(rows)

    if arguments.residual:
        # Subtract what the tables already score, so the network is only asked for the part
        # they get wrong. It then starts level with them rather than having to catch up.
        targets = [
            target - scored for target, scored in zip(targets, tables, strict=True)
        ]
        spread = sum(abs(target) for target in targets) / max(len(targets), 1)
        print(f"  residual targets: {spread:.0f}cp away from the tables on average")
    groups: list[int] = []
    if arguments.outcomes and arguments.outcomes.is_file():
        mixed, groups = blend_targets(fens, targets, arguments.outcomes, arguments.outcome_weight)
        wanted = torch.tensor(mixed, dtype=torch.float32).to(device)
    else:
        wanted = torch.sigmoid(torch.tensor(targets, dtype=torch.float32) * SCALE).to(device)

    if groups:
        # Split by game: positions from one game share a result, so splitting by position puts
        # near-copies of the same answer on both sides and the holdout flatters the fit.
        unique = sorted(set(groups))
        shuffled = [unique[i] for i in torch.randperm(len(unique)).tolist()]
        held = set(shuffled[: max(1, int(len(unique) * arguments.holdout))])
        train = torch.tensor(
            [i for i, g in enumerate(groups) if g not in held], dtype=torch.long
        ).to(device)
        test = torch.tensor(
            [i for i, g in enumerate(groups) if g in held], dtype=torch.long
        ).to(device)
    else:
        order = torch.randperm(count, device=device)
        split = int(count * (1.0 - arguments.holdout))
        train, test = order[:split], order[split:]

    net = Network(arguments.hidden).to(device)
    if arguments.warm and arguments.warm.is_file():
        with np.load(arguments.warm) as data:
            stored = int(data["hidden_bias"].shape[0])
            if stored == arguments.hidden:
                with torch.no_grad():
                    net.embed.weight[:INPUTS].copy_(torch.tensor(data["hidden_weight"]))
                    net.hidden_bias.copy_(torch.tensor(data["hidden_bias"]))
                    net.output.weight.copy_(torch.tensor(data["output_weight"]).reshape(1, -1))
                    net.output.bias.copy_(torch.tensor(data["output_bias"]))
                print(f"  warm started from {arguments.warm}")
            else:
                # Weights cannot be loaded into a different shape, so a mismatch means training
                # from noise while looking exactly like a normal round. That happened once and
                # produced a net of the wrong width that then won and became the champion.
                print(
                    f"  WARNING: {arguments.warm} holds {stored} units but training at "
                    f"{arguments.hidden}. Starting from scratch, not from the best net."
                )
    optimiser = torch.optim.Adam(net.parameters(), lr=arguments.rate)

    best = float("inf")
    kept = {name: tensor.detach().clone() for name, tensor in net.state_dict().items()}
    # Uniform to begin with; prioritised sampling replaces this once there are errors to rank by.
    priority = torch.ones(len(train), dtype=torch.float64, device=device)
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
            shuffled = train[torch.randperm(len(train), device=device)]
            batch_weight = torch.ones(len(train), dtype=torch.float32, device=device)

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
            # Report the plain error, not the importance-weighted one the gradient used, so
            # the training number stays comparable with the holdout beside it.
            total += float(errors.detach().mean()) * len(rows_batch)

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
            kept = {
                name: tensor.detach().cpu().clone()
                for name, tensor in net.state_dict().items()
            }
            # Written the moment it improves: training runs long enough
            # that an interruption should not cost a result already reached.
            save(kept, arguments.out, arguments.residual)
        print(
            f"  epoch {epoch + 1:3}/{arguments.epochs}  train {total / len(shuffled):.6f}  "
            f"holdout {held:.6f}{'  *' if held == best else ''}"
        )

    save(kept, arguments.out, arguments.residual)
    size = arguments.out.stat().st_size
    print(f"best holdout {best:.6f}; wrote {arguments.out} ({size:,} bytes)")


if __name__ == "__main__":
    main()
