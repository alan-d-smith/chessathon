"""The submission entrypoint. The platform imports this file and calls get_move.

A negamax search with alpha-beta, iterative deepening, a transposition table that survives
across moves, quiescence at the leaves, and an evaluation of material plus piece-square tables
tapered towards the endgame. Pure python-chess: nothing here needs a warm-up, so import costs
almost nothing of the 90 second budget.

The ordering of priorities is deliberate. Alpha-beta only pays for itself when good moves come
first, so most of the code below is about move ordering; and a flag is a whole point, so the
search abandons a depth rather than finish it, and get_move always has a legal move in hand.
"""

import time
from collections.abc import Hashable, Iterator
from pathlib import Path
from typing import Any, Final

import chess

import weights

INFINITY: Final = 1 << 20
MATE: Final = 1 << 16
# Anything past this is a mate score rather than an evaluation, and has to be handled as a
# distance rather than a number.
MATE_BOUND: Final = (1 << 16) - 64
MAX_DEPTH: Final = 64
# The search can only stop on a clock check, so this interval is the worst case by which it
# overruns its budget. 512 nodes costs a few microseconds a second and bounds the overrun to
# milliseconds; at 2048 a slow search can sail a quarter of a second past the deadline.
CHECK_INTERVAL: Final = 512

# Wall time is what the referee measures, so the budget leaves room for the round trip and the
# search checks the clock mid-flight rather than only between depths.
MOVE_OVERHEAD_MS: Final = 150
EXPECTED_MOVES: Final = 30
MAX_CLOCK_FRACTION: Final = 0.35
INCREMENT_SHARE: Final = 0.75
ASSUMED_INCREMENT_MS: Final = 500
# Twice the published increment. The inferred value feeds a budget whose spend feeds the next
# inference, so it needs a ceiling that a feedback loop cannot climb past.
INCREMENT_CEILING_MS: Final = 1000.0

VALUE: Final = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
# Phase runs from all the heavy pieces on to none of them, and tapers the king between the
# table that wants it castled and the table that wants it marching. Minors count 1, rooks 2,
# queens 4, so a full board is 24.
TOTAL_PHASE: Final = 24

TT_LIMIT: Final = 400_000
REPETITION_PENALTY: Final = 40

# Search shaping. Alpha-beta only pays off when it can cut, so these decide what never gets
# searched at all: a null move to prove a position is already winning, reductions for quiet
# moves the ordering put last, and a margin below which a capture cannot rescue the line.
NULL_MIN_DEPTH: Final = 3
NULL_REDUCTION: Final = 2
LMR_MIN_DEPTH: Final = 3
LMR_MIN_MOVE: Final = 4
LMR_DEEP_MOVE: Final = 8
DELTA_MARGIN: Final = 200
# Shallow positions far enough the wrong side of the window are decided already. Both margins
# scale with depth, because the deeper the remaining search the more a position can still move.
RFP_MAX_DEPTH: Final = 3
RFP_MARGIN: Final = 120
FUTILITY_MAX_DEPTH: Final = 2
FUTILITY_MARGIN: Final = 150
PRUNE_MAX_DEPTH: Final = 3


# Mirroring once at import saves a square_mirror call in the hottest loop there is.
MIRROR: Final = [chess.square_mirror(square) for square in range(64)]

# Structural terms, in the order the fitted weights arrive in. Material and placement alone
# cannot tell a passed pawn from a blocked one, and at this depth the search will not discover
# the difference on its own.
DOUBLED: Final = 0
ISOLATED: Final = 1
PASSED_2: Final = 2
BISHOP_PAIR: Final = 8
ROOK_OPEN: Final = 9
ROOK_SEMI: Final = 10
SHIELD_MISSING: Final = 11
TEMPO: Final = 12
STRUCTURAL_NAMES: Final = (
    "doubled",
    "isolated",
    "passed_2",
    "passed_3",
    "passed_4",
    "passed_5",
    "passed_6",
    "passed_7",
    "bishop_pair",
    "rook_open",
    "rook_semi",
    "shield_missing",
    "tempo",
)
SHIELD_WANTED: Final = 3


def from_rows(values: list[int]) -> list[int]:
    """weights.py writes tables rank 8 first, as a board is drawn. python-chess counts a1 as 0."""
    return [values[(7 - rank) * 8 + file] for rank in range(8) for file in range(8)]


def blend(midgame: list[int], endgame: list[int], phase: int) -> list[int]:
    return [
        (midgame[square] * phase + endgame[square] * (TOTAL_PHASE - phase)) // TOTAL_PHASE
        for square in range(64)
    ]


PAIRS: Final = (
    (from_rows(weights.PAWN_MG), from_rows(weights.PAWN_EG)),
    (from_rows(weights.KNIGHT_MG), from_rows(weights.KNIGHT_EG)),
    (from_rows(weights.BISHOP_MG), from_rows(weights.BISHOP_EG)),
    (from_rows(weights.ROOK_MG), from_rows(weights.ROOK_EG)),
    (from_rows(weights.QUEEN_MG), from_rows(weights.QUEEN_EG)),
    (from_rows(weights.KING_MG), from_rows(weights.KING_EG)),
)

# Every term is tapered between a midgame and an endgame value, but phase only takes 25 values,
# so all 25 blends are built once at import. Tapering everything then costs the search nothing:
# evaluation still does one table lookup per piece, exactly as it did untapered.
TABLES_W: Final = [
    [blend(midgame, endgame, phase) for midgame, endgame in PAIRS]
    for phase in range(TOTAL_PHASE + 1)
]
TABLES_B: Final = [
    [[table[MIRROR[square]] for square in range(64)] for table in tables] for tables in TABLES_W
]
STRUCTURAL: Final = [
    tuple(
        (weights.STRUCTURAL_MG[name] * phase + weights.STRUCTURAL_EG[name] * (TOTAL_PHASE - phase))
        // TOTAL_PHASE
        for name in STRUCTURAL_NAMES
    )
    for phase in range(TOTAL_PHASE + 1)
]

FILE_OF: Final = [chess.square_file(square) for square in range(64)]
RANK_OF: Final = [chess.square_rank(square) for square in range(64)]
NEIGHBOUR_FILES: Final = [
    (chess.BB_FILES[file - 1] if file > 0 else 0)
    | (chess.BB_FILES[file + 1] if file < 7 else 0)
    for file in range(8)
]


def ahead_mask(square: int, colour: chess.Color) -> int:
    """Own and adjacent files, on every rank in front of this square, from that colour's view."""
    file = FILE_OF[square]
    files = chess.BB_FILES[file] | NEIGHBOUR_FILES[file]
    ranks = range(RANK_OF[square] + 1, 8) if colour else range(0, RANK_OF[square])
    ahead = 0
    for rank in ranks:
        ahead |= chess.BB_RANKS[rank]
    return files & ahead


def shield_mask(square: int, colour: chess.Color) -> int:
    """The two ranks directly in front of a king, across its own and adjacent files."""
    file = FILE_OF[square]
    files = chess.BB_FILES[file] | NEIGHBOUR_FILES[file]
    rank = RANK_OF[square]
    steps = (rank + 1, rank + 2) if colour else (rank - 1, rank - 2)
    ahead = 0
    for step in steps:
        if 0 <= step <= 7:
            ahead |= chess.BB_RANKS[step]
    return files & ahead


PASSED: Final = (
    [ahead_mask(square, chess.BLACK) for square in range(64)],
    [ahead_mask(square, chess.WHITE) for square in range(64)],
)
SHIELD: Final = (
    [shield_mask(square, chess.BLACK) for square in range(64)],
    [shield_mask(square, chess.WHITE) for square in range(64)],
)

# Pawn structure changes on maybe one move in six, so the same skeleton is counted over and
# over. The cache holds counts rather than a score, so it stays valid at every phase.
pawn_cache: dict[tuple[int, int], tuple[int, ...]] = {}
PAWN_CACHE_LIMIT: Final = 200_000

EXACT: Final = 0
LOWER: Final = 1
UPPER: Final = 2


class Timeout(Exception):
    """Raised mid-search when the budget is spent. The caller keeps the last finished depth."""


# Module state survives between the moves of one game and never into the next, which is exactly
# the lifetime a transposition table wants.
transposition: dict[Hashable, tuple[int, int, int, chess.Move | None]] = {}
killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_DEPTH + 2)]
history: dict[tuple[int, int], int] = {}
seen: set[Hashable] = set()

nodes = 0
deadline = 0.0
reached = 0  # deepest iteration completed on the last move, for diagnostics

# The increment is inferred from how the clock moves between our own turns, starting from the
# published 0.5s and correcting itself after one move at whatever the real time control is.
increment_ms = float(ASSUMED_INCREMENT_MS)
last_clock_ms: float | None = None
last_spent_ms = 0.0


def tick() -> None:
    """Give up on the clock inside the search, not only between depths."""
    global nodes
    nodes += 1
    if nodes % CHECK_INTERVAL == 0 and time.monotonic() > deadline:
        raise Timeout


def scan(mask: int, table: list[int]) -> int:
    """Sum a table over the set bits of a bitboard, in plain integer arithmetic.

    board.pieces() would read better, but it allocates a SquareSet per call and evaluation is
    the hottest function in the search: this is where the node rate comes from.
    """
    total = 0
    while mask:
        lowest = mask & -mask
        total += table[lowest.bit_length() - 1]
        mask ^= lowest
    return total


def pawn_counts(white_pawns: int, black_pawns: int) -> tuple[int, ...]:
    """Net doubled, isolated and passed-by-rank counts, White positive.

    Counts, not a score, because the weights they are multiplied by depend on the phase and the
    pawns do not. One cache then serves every phase the same skeleton turns up in.
    """
    cached = pawn_cache.get((white_pawns, black_pawns))
    if cached is not None:
        return cached

    tally = [0] * len(STRUCTURAL_NAMES)
    for colour, mine, theirs, sign in (
        (chess.WHITE, white_pawns, black_pawns, 1),
        (chess.BLACK, black_pawns, white_pawns, -1),
    ):
        for file in range(8):
            count = (mine & chess.BB_FILES[file]).bit_count()
            if count > 1:
                tally[DOUBLED] += sign * (count - 1)

        remaining = mine
        passed_masks = PASSED[colour]
        while remaining:
            lowest = remaining & -remaining
            square = lowest.bit_length() - 1
            remaining ^= lowest
            if not mine & NEIGHBOUR_FILES[FILE_OF[square]]:
                tally[ISOLATED] += sign
            if not theirs & passed_masks[square]:
                # Rank counted from the pawn's own side, so both colours share one set of weights.
                advance = RANK_OF[square] if colour == chess.WHITE else 7 - RANK_OF[square]
                if 2 <= advance <= 7:
                    tally[PASSED_2 + advance - 2] += sign

    counts = tuple(tally)
    if len(pawn_cache) < PAWN_CACHE_LIMIT:
        pawn_cache[(white_pawns, black_pawns)] = counts
    return counts


def evaluate(board: chess.Board) -> int:
    """Material and placement, from the side to move. Positive means the mover is better."""
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    pawns, knights = board.pawns, board.knights
    bishops, rooks, queens, kings = board.bishops, board.rooks, board.queens, board.kings

    # Popcounts beat counting squares one at a time, and the phase is only ever a weighted count.
    phase = (knights | bishops).bit_count() + rooks.bit_count() * 2 + queens.bit_count() * 4
    phase = min(phase, TOTAL_PHASE)
    ours, theirs = TABLES_W[phase], TABLES_B[phase]

    score = (
        scan(pawns & white, ours[0])
        - scan(pawns & black, theirs[0])
        + scan(knights & white, ours[1])
        - scan(knights & black, theirs[1])
        + scan(bishops & white, ours[2])
        - scan(bishops & black, theirs[2])
        + scan(rooks & white, ours[3])
        - scan(rooks & black, theirs[3])
        + scan(queens & white, ours[4])
        - scan(queens & black, theirs[4])
        + scan(kings & white, ours[5])
        - scan(kings & black, theirs[5])
    )

    weight = STRUCTURAL[phase]
    for term, count in enumerate(pawn_counts(pawns & white, pawns & black)):
        if count:
            score += count * weight[term]

    # Two bishops cover both colour complexes, which is worth more than the pieces separately.
    if (bishops & white).bit_count() >= 2:
        score += weight[BISHOP_PAIR]
    if (bishops & black).bit_count() >= 2:
        score -= weight[BISHOP_PAIR]

    for colour, mine, sign in ((chess.WHITE, white, 1), (chess.BLACK, black, -1)):
        # A rook is worth having where it can actually see down the board.
        remaining = rooks & mine
        while remaining:
            lowest = remaining & -remaining
            remaining ^= lowest
            file_mask = chess.BB_FILES[FILE_OF[lowest.bit_length() - 1]]
            if not pawns & file_mask:
                score += sign * weight[ROOK_OPEN]
            elif not pawns & mine & file_mask:
                score += sign * weight[ROOK_SEMI]

        king = kings & mine
        if king:
            square = king.bit_length() - 1
            present = (pawns & mine & SHIELD[colour][square]).bit_count()
            missing = SHIELD_WANTED - min(present, SHIELD_WANTED)
            score += sign * missing * weight[SHIELD_MISSING]

    # Having the move is worth something, and it is the one thing the piece placement cannot
    # say. Added from White's view so the flip below hands it to whoever is actually to move.
    score += weight[TEMPO] if board.turn == chess.WHITE else -weight[TEMPO]

    return score if board.turn == chess.WHITE else -score


# An optional network, used only when weights/net.npz ships alongside this file. numpy and
# numba are imported inside the loader rather than at the top, so an agent without a network
# pays nothing for the possibility of one: no import cost, no compile, no new way to fail.
# Typed loosely on purpose: numpy and numba are not imported unless a network ships, so
# nothing here can be given a real type at module scope without importing them.
net_forward: Any = None
net_state: dict[str, Any] = {}


def load_net() -> bool:
    """Load and compile the network if one shipped. Returning False keeps the tables."""
    path = Path(__file__).resolve().parent / "weights" / "net.npz"
    if not path.is_file():
        return False
    try:
        import numpy as np
        from numba import njit
    except ImportError:
        return False

    try:
        with np.load(path) as data:
            hidden_w = np.ascontiguousarray(data["hidden_weight"], dtype=np.float32)
            hidden_b = np.ascontiguousarray(data["hidden_bias"], dtype=np.float32)
            out_w = np.ascontiguousarray(data["output_weight"], dtype=np.float32)
            out_b = float(data["output_bias"][0])
    except (OSError, KeyError, ValueError, IndexError):
        return False
    if hidden_w.shape != (768, hidden_b.shape[0]) or out_w.shape != hidden_b.shape:
        return False

    # numba has no bit_length for a uint64, so squares come out of a bitboard by de Bruijn.
    # Built with Python integers and masked by hand: the wrap it relies on is what numpy would
    # call an overflow, and it would warn on every import for something entirely intended.
    table = np.zeros(64, dtype=np.int64)
    for square in range(64):
        table[(((1 << square) * 0x03F79D71B4CB0A89) & 0xFFFFFFFFFFFFFFFF) >> 58] = square

    @njit(cache=False)
    def forward(  # type: ignore[no-untyped-def]  # numba infers these from the call site
        pawns, knights, bishops, rooks, queens, kings, ours, theirs, flip,
        weight_in, bias_in, weight_out, bias_out, lookup, magic,
    ):
        """Centipawns from the side to move. One call, taking the raw bitboards.

        Pulling the piece squares out in Python first would cost more than the network does.
        A full refresh each call rather than an accumulator kept in step with make and unmake:
        at this width refreshing is fast enough, and it cannot fall out of step with the board.
        """
        hidden = bias_in.copy()
        for piece in range(6):
            if piece == 0:
                board = pawns
            elif piece == 1:
                board = knights
            elif piece == 2:
                board = bishops
            elif piece == 3:
                board = rooks
            elif piece == 4:
                board = queens
            else:
                board = kings
            for owner in range(2):
                squares = board & (ours if owner == 0 else theirs)
                base = owner * 384 + piece * 64
                while squares:
                    lowest = squares & (~squares + np.uint64(1))
                    square = lookup[(lowest * magic) >> np.uint64(58)]
                    if flip:
                        # Mirror so our own first rank is always the bottom of the board, which
                        # is what lets one set of weights serve both colours.
                        square = square ^ 56
                    index = base + square
                    for unit in range(hidden.shape[0]):
                        hidden[unit] += weight_in[index, unit]
                    squares ^= lowest

        total = bias_out
        for unit in range(hidden.shape[0]):
            value = hidden[unit]
            if value > 0.0:
                total += value * weight_out[unit]
        return total

    global net_forward
    net_forward = forward
    net_state.update(
        numpy=np,
        hidden_w=hidden_w,
        hidden_b=hidden_b,
        out_w=out_w,
        out_b=out_b,
        table=table,
        magic=np.uint64(0x03F79D71B4CB0A89),
    )
    return True


def evaluate_net(board: chess.Board) -> int:
    """The network's view, in centipawns from the side to move."""
    np = net_state["numpy"]
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    mover_is_white = board.turn == chess.WHITE
    ours, theirs = (white, black) if mover_is_white else (black, white)
    return int(
        net_forward(
            np.uint64(board.pawns),
            np.uint64(board.knights),
            np.uint64(board.bishops),
            np.uint64(board.rooks),
            np.uint64(board.queens),
            np.uint64(board.kings),
            np.uint64(ours),
            np.uint64(theirs),
            not mover_is_white,
            net_state["hidden_w"],
            net_state["hidden_b"],
            net_state["out_w"],
            net_state["out_b"],
            net_state["table"],
            net_state["magic"],
        )
    )


USING_NET: Final = load_net()

if USING_NET:
    # numba compiles on the first call, and that call costs far more than the move it would be
    # part of. Spending it here puts it inside the 90 second import budget rather than on the
    # clock, warmed with the argument types the real calls will use.
    evaluate_net(chess.Board())
    evaluate = evaluate_net


def capture_value(board: chess.Board, move: chess.Move) -> int:
    """MVV-LVA: take the biggest thing with the smallest thing, and try that order first."""
    victim = board.piece_type_at(move.to_square)
    gain = VALUE[victim] if victim is not None else VALUE[chess.PAWN]  # None means en passant
    if move.promotion is not None:
        gain += VALUE[move.promotion]
    attacker = board.piece_type_at(move.from_square)
    return gain * 16 - (VALUE[attacker] if attacker is not None else 0)


def candidates(board: chess.Board, best: chess.Move | None, ply: int) -> Iterator[chess.Move]:
    """The transposition move first, then captures, then the quiet moves that have been cutting.

    A generator rather than a list so the move list is never built at all when the first move
    already cuts. Generating moves is the most expensive thing in the search, and the whole
    point of trying the transposition move first is that it usually is the one that cuts.
    """
    if best is not None and board.is_legal(best):
        yield best

    slot = killers[ply]

    def rank(move: chess.Move) -> int:
        if board.is_capture(move) or move.promotion is not None:
            return (1 << 20) + capture_value(board, move)
        if move in slot:
            return 1 << 19
        return history.get((move.from_square, move.to_square), 0)

    rest = [move for move in board.legal_moves if move != best]
    rest.sort(key=rank, reverse=True)
    yield from rest


def quiesce(board: chess.Board, alpha: int, beta: int) -> int:
    """Resolve the captures. Evaluating mid-exchange is how a good evaluation reads as noise."""
    tick()
    standing = evaluate(board)
    if standing >= beta:
        return beta
    alpha = max(alpha, standing)

    captures = sorted(
        board.generate_legal_captures(), key=lambda move: capture_value(board, move), reverse=True
    )
    for move in captures:
        # Delta pruning: if winning the piece outright still falls short of alpha, the whole
        # line is irrelevant and searching it is time spent proving something already known.
        if move.promotion is None:
            victim = board.piece_type_at(move.to_square)
            gain = VALUE[victim] if victim is not None else VALUE[chess.PAWN]
            if standing + gain + DELTA_MARGIN < alpha:
                continue
            # Taking a defended piece with a more valuable one loses material unless something
            # deeper justifies it, and quiescence is where most of the nodes are. Skipping
            # these shrinks the tree rather than making each node faster, which is the only
            # kind of gain left: even a free move generator would only be worth 1.3x nodes.
            attacker = board.piece_type_at(move.from_square)
            if (
                attacker is not None
                and VALUE[attacker] > gain
                and board.is_attacked_by(not board.turn, move.to_square)
            ):
                continue
        board.push(move)
        score = -quiesce(board, -beta, -alpha)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def to_store(score: int, ply: int) -> int:
    """Rebase a mate score from "distance from the root" to "distance from this node".

    A mate is scored -MATE + ply, which is only meaningful at the ply that found it. The same
    position reached at another depth would read that entry as a mate a different number of
    moves away, so the table stores node-relative distances and converts back on the way out.
    """
    if score > MATE_BOUND:
        return score + ply
    if score < -MATE_BOUND:
        return score - ply
    return score


def from_store(score: int, ply: int) -> int:
    """Undo to_store, putting a mate distance back into the root's frame of reference."""
    if score > MATE_BOUND:
        return score - ply
    if score < -MATE_BOUND:
        return score + ply
    return score


def has_pieces(board: chess.Board, colour: chess.Color) -> bool:
    """Whether a side still holds a piece beyond pawns, which is what makes zugzwang unlikely."""
    mine = board.occupied_co[colour]
    return bool((board.knights | board.bishops | board.rooks | board.queens) & mine)


def negamax(board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    tick()
    if board.is_insufficient_material() or board.halfmove_clock >= 100:
        return 0

    # Never hand a position back to the evaluation with the king under fire: the reply is
    # forced and the score is meaningless. Bounded by ply, so perpetual check cannot recurse
    # forever on an extension that keeps renewing itself.
    in_check = board.is_check()
    if in_check and ply < MAX_DEPTH:
        depth += 1

    original = alpha
    # Private, but it is the key the board already keeps for its own repetition checks, and it
    # is the one docs/IDEAS.md points at.
    key = board._transposition_key()
    stored = transposition.get(key)
    if stored is not None and stored[0] >= depth:
        score = from_store(stored[1], ply)
        flag = stored[2]
        if flag == EXACT:
            return score
        if flag == LOWER:
            alpha = max(alpha, score)
        elif flag == UPPER:
            beta = min(beta, score)
        if alpha >= beta:
            return score

    if depth <= 0:
        return quiesce(board, alpha, beta)

    # One static evaluation, shared by both shallow prunings below. Meaningless in check, and
    # not worth computing at depths where the search will settle the position anyway.
    static: int | None = None
    if not in_check and depth <= PRUNE_MAX_DEPTH:
        static = evaluate(board)
        # Standing this far above beta with this little depth left, the opponent has no moves
        # in hand that pull it back down. Return the evaluation rather than prove it.
        if (
            depth <= RFP_MAX_DEPTH
            and beta < MATE - MAX_DEPTH
            and static - RFP_MARGIN * depth >= beta
        ):
            return static

    # Null move: hand the opponent a free move, and if the position still beats beta then it
    # was never worth searching properly. Skipped in check, and skipped without a piece on the
    # board, because those are the positions where passing would genuinely have been best.
    if depth >= NULL_MIN_DEPTH and not in_check and has_pieces(board, board.turn):
        board.push(chess.Move.null())
        score = -negamax(board, depth - 1 - NULL_REDUCTION, -beta, -beta + 1, ply + 1)
        board.pop()
        if score >= beta:
            return beta

    best_move: chess.Move | None = None
    best_score = -INFINITY
    index = -1
    for index, move in enumerate(candidates(board, stored[3] if stored else None, ply)):
        quiet = not board.is_capture(move) and move.promotion is None
        board.push(move)

        # Futility: a quiet move that neither captures nor checks, from a position already this
        # far below alpha, is not going to climb back over it in a couple of plies. Never the
        # first move, so the node always searches something and always has a score to return.
        if (
            static is not None
            and quiet
            and index > 0
            and depth <= FUTILITY_MAX_DEPTH
            and best_score > -MATE + MAX_DEPTH
            and static + FUTILITY_MARGIN * depth <= alpha
            and not board.is_check()
        ):
            board.pop()
            continue

        # Late quiet moves are searched shallow and on a null window, on the bet that the
        # ordering was right and they will not beat alpha. A move that beats it anyway is
        # re-searched at full depth, so the bet costs nothing when it is wrong.
        reduction = 0
        if depth >= LMR_MIN_DEPTH and index >= LMR_MIN_MOVE and quiet and not board.is_check():
            reduction = 1 if index < LMR_DEEP_MOVE else 2

        if index == 0:
            score = -negamax(board, depth - 1, -beta, -alpha, ply + 1)
        else:
            score = -negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
            if reduction and score > alpha:
                score = -negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
            if alpha < score < beta:
                score = -negamax(board, depth - 1, -beta, -alpha, ply + 1)

        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, score)
        if alpha >= beta:
            if quiet:
                slot = killers[ply]
                if move != slot[0]:
                    slot[1] = slot[0]
                    slot[0] = move
                edge = (move.from_square, move.to_square)
                history[edge] = history.get(edge, 0) + depth * depth
            break

    # Nothing was yielded, so there was nothing legal to play: mate if the king is attacked,
    # stalemate otherwise. Checked here because the generator never built a list to count.
    if index < 0:
        return -MATE + ply if in_check else 0

    if len(transposition) < TT_LIMIT:
        flag = EXACT if original < best_score < beta else (LOWER if best_score >= beta else UPPER)
        transposition[key] = (depth, to_store(best_score, ply), flag, best_move)
    return best_score


def budget_s(time_left_ms: int) -> float:
    """Spend a share of the clock, never a fixed amount, and never most of what is left.

    The increment is worth spending because it comes back every move, but the contract never
    states it, so it is measured rather than assumed: guessing 0.5s at a time control that pays
    0.1s spends five times the increment every move and walks the clock down to a flag.
    """
    usable = max(0.0, time_left_ms - MOVE_OVERHEAD_MS)
    share = usable / EXPECTED_MOVES + increment_ms * INCREMENT_SHARE
    return min(share, usable * MAX_CLOCK_FRACTION) / 1000.0


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    """
    global deadline, nodes, reached, increment_ms, last_clock_ms, last_spent_ms

    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError(f"no legal move in {fen}")

    # Whatever the clock gained back since our last turn, beyond what we spent, is the
    # increment. Measured from inside, so it under-reads by the round trip and errs slow.
    # Bounded hard: the inference feeds the budget that the same spend then feeds back into,
    # so an unmoving clock would otherwise ratchet it upwards a move at a time.
    if last_clock_ms is not None:
        measured = time_left_ms - (last_clock_ms - last_spent_ms)
        increment_ms = min(max(measured, 0.0), INCREMENT_CEILING_MS)

    started = time.monotonic()
    seen.add(board._transposition_key())
    deadline = started + budget_s(time_left_ms)
    nodes = 0
    reached = 0

    # Whatever happens below, this is already legal, so a timeout or a bug in the search costs
    # a weaker move rather than the game.
    choice = legal[0]

    for depth in range(1, MAX_DEPTH + 1):
        best_move: chess.Move | None = None
        best_score = -INFINITY
        alpha = -INFINITY
        finished = False
        try:
            for move in candidates(board, choice, 0):
                board.push(move)
                score = -negamax(board, depth - 1, -INFINITY, -alpha, 1)
                # Shuffling back into a position we have already been asked about hands over a
                # draw the referee claims for us, so it is only worth it when we are worse.
                repeated = board._transposition_key() in seen
                board.pop()
                if repeated and score > 0:
                    score -= REPETITION_PENALTY
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            finished = True
        except Timeout:
            pass

        # A finished depth is trustworthy. An abandoned one is only worth taking if it had
        # already improved on the move the previous depth settled on.
        if best_move is not None and (finished or best_score > -INFINITY):
            choice = best_move
        if not finished:
            break
        reached = depth
        if abs(best_score) > MATE - MAX_DEPTH:
            break  # a mate score will not get better with depth

    last_clock_ms = float(time_left_ms)
    last_spent_ms = (time.monotonic() - started) * 1000.0
    return choice.uci()
