"""Fit the evaluation's weights to labelled positions, instead of typing numbers and hoping.

Texel tuning. Every weight in the evaluation is one coefficient of a dot product, so the whole
evaluation is a linear model over the features in tools/features.py and can be fitted directly.
The target is what Stockfish thinks of the position, squashed through the same logistic a chess
score belongs on, so the fit spends its effort where the difference between +0.3 and +0.6
matters and ignores positions that are already decided.

    uv run python -m tools.tune --data data/labelled.jsonl --out weights.py

Labelling positions with an existing engine is explicitly permitted, and the weights that come
out are ours: the engine is a teacher here, not a passenger in the zip.

Fitting starts from the current hand-set values rather than from noise, so a term the data has
little to say about keeps its sensible default instead of drifting somewhere strange.
"""

import argparse
import json
import math
from pathlib import Path

import chess
import torch

from tools import features

# The logistic a centipawn score belongs on: +400cp is about a 90% score.
SCALE = math.log(10.0) / 400.0
CLAMP_CP = 1500
MATE_CP = 2500


def is_quiet(board: chess.Board, best: str | None) -> bool:
    """Whether a static evaluation can be expected to mean anything here.

    The agent only ever evaluates positions its quiescence search has already settled, so a
    position with the king in check, or whose best move is a capture, is one where the label
    reflects a tactic the linear model has no way to represent. Fitting on those teaches it to
    predict tactics it structurally cannot see, at the cost of the positional terms it can.
    """
    if board.is_check():
        return False
    if best is None:
        return True
    try:
        move = chess.Move.from_uci(best)
    except ValueError:
        return True
    return not board.is_capture(move) and move.promotion is None


def load_outcomes(path: Path, limit: int | None) -> tuple[list[str], list[float], list[int]]:
    """Positions labelled by how their game finished, already on the 0..1 scale a score lives on.

    No sigmoid conversion: a result is a probability rather than an opinion in centipawns, which
    is the whole point of using it. Far noisier per position than an engine label, and far
    harder to fool, because nothing here has a view about chess except what won.
    """
    fens: list[str] = []
    targets: list[float] = []
    games: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            fens.append(row["fen"])
            targets.append(float(row["result"]))
            games.append(int(row.get("game", len(games))))
            if limit and len(fens) >= limit:
                break
    return fens, targets, games


def load(path: Path, limit: int | None, quiet_only: bool = False) -> tuple[list[str], list[float]]:
    """Positions and their labels, as centipawns from White's point of view."""
    fens: list[str] = []
    targets: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a truncated final line from an interrupted labelling run
            mate, centipawns = row.get("mate"), row.get("cp")
            if mate is not None:
                score = MATE_CP if mate > 0 else -MATE_CP
            elif centipawns is not None:
                score = max(-CLAMP_CP, min(CLAMP_CP, int(centipawns)))
            else:
                continue
            # Labels are relative to the side to move; features are always White's view.
            board = chess.Board(row["fen"])
            if quiet_only and not is_quiet(board, row.get("best")):
                continue
            fens.append(row["fen"])
            targets.append(float(score if board.turn == chess.WHITE else -score))
            if limit and len(fens) >= limit:
                break
    return fens, targets


def extract(fens: list[str]) -> torch.Tensor:
    """The feature matrix, one row per position.

    Dense on purpose. Each row is only about eighty non-zeros out of eight hundred, but a
    few hundred thousand rows still fit in memory several times over, and a dense matrix makes
    the model an ordinary matrix multiply rather than a scatter-add with hand-managed offsets.
    """
    matrix = torch.zeros((len(fens), features.FEATURES), dtype=torch.float32)
    for row, fen in enumerate(fens):
        indices, values = features.vector(chess.Board(fen))
        matrix[row, torch.tensor(indices, dtype=torch.long)] = torch.tensor(
            values, dtype=torch.float32
        )
    return matrix


def initial() -> torch.Tensor:
    """Start from the weights already in play, so a run refines rather than reinvents.

    Reading weights.py rather than hard-coded defaults makes tuning iterative: fit, play the
    result, and if it holds up, fit again from there.
    """
    import weights as current

    fitted = torch.zeros(features.FEATURES, dtype=torch.float32)
    tables = (
        (current.PAWN_MG, current.PAWN_EG),
        (current.KNIGHT_MG, current.KNIGHT_EG),
        (current.BISHOP_MG, current.BISHOP_EG),
        (current.ROOK_MG, current.ROOK_EG),
        (current.QUEEN_MG, current.QUEEN_EG),
        (current.KING_MG, current.KING_EG),
    )
    for offset, (midgame, endgame) in enumerate(tables):
        for square in range(64):
            # weights.py is written rank 8 first; the weight vector is indexed by square.
            row = (7 - square // 8) * 8 + square % 8
            term = offset * features.SQUARES + square
            fitted[term * 2] = midgame[row]
            fitted[term * 2 + 1] = endgame[row]

    for name in features.STRUCTURAL:
        term = features.STRUCTURAL_INDEX[name]
        # A term added since the last fit starts at zero, so the data decides it from scratch
        # rather than the tuner failing on a key that weights.py has never heard of.
        fitted[term * 2] = current.STRUCTURAL_MG.get(name, 0)
        fitted[term * 2 + 1] = current.STRUCTURAL_EG.get(name, 0)
    return fitted


def emit(weights: torch.Tensor, path: Path, positions: int) -> None:
    rounded = [round(float(value)) for value in weights]

    def table(offset: int, phase: int) -> str:
        rows = []
        for rank in range(7, -1, -1):
            row = [
                rounded[(offset * features.SQUARES + rank * 8 + file) * 2 + phase]
                for file in range(8)
            ]
            rows.append("    " + " ".join(f"{value:5d}," for value in row))
        return "\n".join(rows)

    names = ("PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING")
    lines = [
        '"""Evaluation weights, fitted to Stockfish-labelled positions by tools/tune.py.',
        "",
        f"Generated from {positions:,} positions. Every value is centipawns. Each table is",
        "written rank 8 first, the way a board is drawn, and there are two of everything: a",
        "midgame table and an endgame one, blended by how much material is still on.",
        "",
        "Do not hand-edit. Re-run the tuner instead, and let the games decide whether to keep it.",
        '"""',
        "",
        "# fmt: off",
    ]
    for offset, name in enumerate(names):
        for phase, suffix in ((0, "MG"), (1, "EG")):
            lines.append(f"{name}_{suffix} = [")
            lines.append(table(offset, phase))
            lines.append("]")
    lines.append("")
    lines.append("STRUCTURAL_MG = {")
    for name in features.STRUCTURAL:
        term = features.STRUCTURAL_INDEX[name]
        lines.append(f'    "{name}": {rounded[term * 2]},')
    lines.append("}")
    lines.append("STRUCTURAL_EG = {")
    for name in features.STRUCTURAL:
        term = features.STRUCTURAL_INDEX[name]
        lines.append(f'    "{name}": {rounded[term * 2 + 1]},')
    lines.append("}")
    lines.append("# fmt: on")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit evaluation weights to labelled positions.")
    parser.add_argument("--data", type=Path, default=Path("data/labelled.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("weights.py"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4_096)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument(
        "--quiet-only",
        action="store_true",
        help="skip checks and positions whose best move is a capture",
    )
    parser.add_argument(
        "--outcomes",
        action="store_true",
        help="the data is {fen, result} from played games rather than engine evaluations",
    )
    arguments = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    groups: list[int] = []
    if arguments.outcomes:
        fens, targets, groups = load_outcomes(arguments.data, arguments.limit)
        distinct = len(set(groups))
        print(f"{len(fens):,} positions from {distinct:,} games")
        # One label per game, so the games are the sample size, not the positions. Fitting
        # ~800 weights to a few hundred games produces confident nonsense, and the holdout
        # will not say so unless it is split by game.
        if distinct < features.FEATURES:
            print(
                f"  WARNING: {distinct:,} games for {features.FEATURES:,} weights. "
                "Expect overfitting; gather more games before believing this."
            )
    else:
        fens, targets = load(arguments.data, arguments.limit, arguments.quiet_only)
        print(f"{len(fens):,} labelled positions")
    if len(fens) < 1000:
        raise SystemExit("not enough labelled positions to fit anything trustworthy")

    matrix = extract(fens)
    scores = torch.tensor(targets, dtype=torch.float32)
    # Game results are already the thing the model predicts; centipawns need squashing onto it.
    wanted = scores if arguments.outcomes else torch.sigmoid(scores * SCALE)
    weights = initial().clone().requires_grad_(True)

    # A holdout the fit never sees, so an improving training loss with a worsening holdout
    # shows up as what it is rather than as progress.
    count = len(fens)
    if groups:
        # Split by game. Positions from one game share a label, so splitting by position puts
        # near-copies of the same answer on both sides and the holdout flatters the fit.
        unique = sorted(set(groups))
        shuffled = [unique[i] for i in torch.randperm(len(unique)).tolist()]
        held = set(shuffled[: max(1, int(len(unique) * arguments.holdout))])
        train = torch.tensor([i for i, g in enumerate(groups) if g not in held], dtype=torch.long)
        test = torch.tensor([i for i, g in enumerate(groups) if g in held], dtype=torch.long)
    else:
        order = torch.randperm(count)
        split = int(count * (1.0 - arguments.holdout))
        train, test = order[:split], order[split:]
    optimiser = torch.optim.Adam([weights], lr=arguments.rate)

    # What the weights already in play score on the holdout. Every later number is only
    # meaningful against this one: a loss that falls is not the same as a loss that beats it.
    with torch.no_grad():
        baseline = float(
            torch.nn.functional.mse_loss(
                torch.sigmoid(matrix[test] @ weights * SCALE), wanted[test]
            )
        )
    print(f"  holdout before tuning {baseline:.6f}")

    best = float("inf")
    kept = weights.detach().clone()
    for epoch in range(arguments.epochs):
        shuffled = train[torch.randperm(len(train))]
        total = 0.0
        for start in range(0, len(shuffled), arguments.batch):
            rows = shuffled[start : start + arguments.batch]
            optimiser.zero_grad()
            predicted = torch.sigmoid(matrix[rows] @ weights * SCALE)
            loss = torch.nn.functional.mse_loss(predicted, wanted[rows])
            loss.backward()
            optimiser.step()
            total += float(loss) * len(rows)
        with torch.no_grad():
            held = float(
                torch.nn.functional.mse_loss(
                    torch.sigmoid(matrix[test] @ weights * SCALE), wanted[test]
                )
            )
        if held < best:
            best, kept = held, weights.detach().clone()
        print(
            f"  epoch {epoch + 1:3}/{arguments.epochs}  train {total / len(shuffled):.6f}  "
            f"holdout {held:.6f}{'  *' if held == best else ''}"
        )

    gain = (baseline - best) / baseline if baseline else 0.0
    print(f"best holdout {best:.6f} against {baseline:.6f} untuned, {gain:+.1%}")
    emit(kept, arguments.out, len(fens))
    print(f"weights written to {arguments.out}")


if __name__ == "__main__":
    main()
