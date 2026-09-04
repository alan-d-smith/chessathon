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
from collections.abc import Hashable
from typing import Final

import chess

INFINITY: Final = 1 << 20
MATE: Final = 1 << 16
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


def read(rows: str) -> list[int]:
    """Tables are written rank 8 first, the way a board is drawn. python-chess counts a1 as 0."""
    values = [int(value) for value in rows.split()]
    return [values[(7 - rank) * 8 + file] for rank in range(8) for file in range(8)]


PAWN_TABLE: Final = read("""
     0   0   0   0   0   0   0   0
    50  50  50  50  50  50  50  50
    10  10  20  30  30  20  10  10
     5   5  10  25  25  10   5   5
     0   0   0  20  20   0   0   0
     5  -5 -10   0   0 -10  -5   5
     5  10  10 -20 -20  10  10   5
     0   0   0   0   0   0   0   0
""")
KNIGHT_TABLE: Final = read("""
   -50 -40 -30 -30 -30 -30 -40 -50
   -40 -20   0   0   0   0 -20 -40
   -30   0  10  15  15  10   0 -30
   -30   5  15  20  20  15   5 -30
   -30   0  15  20  20  15   0 -30
   -30   5  10  15  15  10   5 -30
   -40 -20   0   5   5   0 -20 -40
   -50 -40 -30 -30 -30 -30 -40 -50
""")
BISHOP_TABLE: Final = read("""
   -20 -10 -10 -10 -10 -10 -10 -20
   -10   0   0   0   0   0   0 -10
   -10   0   5  10  10   5   0 -10
   -10   5   5  10  10   5   5 -10
   -10   0  10  10  10  10   0 -10
   -10  10  10  10  10  10  10 -10
   -10   5   0   0   0   0   5 -10
   -20 -10 -10 -10 -10 -10 -10 -20
""")
ROOK_TABLE: Final = read("""
     0   0   0   0   0   0   0   0
     5  10  10  10  10  10  10   5
    -5   0   0   0   0   0   0  -5
    -5   0   0   0   0   0   0  -5
    -5   0   0   0   0   0   0  -5
    -5   0   0   0   0   0   0  -5
    -5   0   0   0   0   0   0  -5
     0   0   0   5   5   0   0   0
""")
QUEEN_TABLE: Final = read("""
   -20 -10 -10  -5  -5 -10 -10 -20
   -10   0   0   0   0   0   0 -10
   -10   0   5   5   5   5   0 -10
    -5   0   5   5   5   5   0  -5
     0   0   5   5   5   5   0  -5
   -10   5   5   5   5   5   0 -10
   -10   0   5   0   0   0   0 -10
   -20 -10 -10  -5  -5 -10 -10 -20
""")
KING_MIDGAME: Final = read("""
   -30 -40 -40 -50 -50 -40 -40 -30
   -30 -40 -40 -50 -50 -40 -40 -30
   -30 -40 -40 -50 -50 -40 -40 -30
   -30 -40 -40 -50 -50 -40 -40 -30
   -20 -30 -30 -40 -40 -30 -30 -20
   -10 -20 -20 -20 -20 -20 -20 -10
    20  20   0   0   0   0  20  20
    20  30  10   0   0  10  30  20
""")
KING_ENDGAME: Final = read("""
   -50 -40 -30 -20 -20 -30 -40 -50
   -30 -20 -10   0   0 -10 -20 -30
   -30 -10  20  30  30  20 -10 -30
   -30 -10  30  40  40  30 -10 -30
   -30 -10  30  40  40  30 -10 -30
   -30 -10  20  30  30  20 -10 -30
   -30 -30   0   0   0   0 -30 -30
   -50 -30 -30 -30 -30 -30 -30 -50
""")

# Mirroring once at import saves a square_mirror call in the hottest loop there is.
MIRROR: Final = [chess.square_mirror(square) for square in range(64)]


def fold(table: list[int], value: int) -> tuple[list[int], list[int]]:
    """Fold material into the square table, once per colour, so evaluation is one lookup."""
    return (
        [value + table[square] for square in range(64)],
        [value + table[MIRROR[square]] for square in range(64)],
    )


PAWN_W, PAWN_B = fold(PAWN_TABLE, VALUE[chess.PAWN])
KNIGHT_W, KNIGHT_B = fold(KNIGHT_TABLE, VALUE[chess.KNIGHT])
BISHOP_W, BISHOP_B = fold(BISHOP_TABLE, VALUE[chess.BISHOP])
ROOK_W, ROOK_B = fold(ROOK_TABLE, VALUE[chess.ROOK])
QUEEN_W, QUEEN_B = fold(QUEEN_TABLE, VALUE[chess.QUEEN])
KING_MG_B: Final = [KING_MIDGAME[MIRROR[square]] for square in range(64)]
KING_EG_B: Final = [KING_ENDGAME[MIRROR[square]] for square in range(64)]

# Structural terms. Material and placement alone cannot tell a passed pawn from a blocked one,
# and at this depth the search will not discover the difference on its own.
BISHOP_PAIR: Final = 30
DOUBLED_PENALTY: Final = 12
ISOLATED_PENALTY: Final = 14
ROOK_OPEN_FILE: Final = 18
ROOK_SEMI_OPEN_FILE: Final = 9
PASSED_BONUS: Final = (0, 5, 12, 22, 40, 70, 115, 0)

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


PASSED_W: Final = [ahead_mask(square, chess.WHITE) for square in range(64)]
PASSED_B: Final = [ahead_mask(square, chess.BLACK) for square in range(64)]

# Pawn structure changes on maybe one move in six, so the same skeleton is evaluated over and
# over. Keying a cache on the two pawn bitboards turns most of that work into a dict lookup.
pawn_cache: dict[tuple[int, int], int] = {}
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


def pawn_score(white_pawns: int, black_pawns: int) -> int:
    """Doubled, isolated and passed pawns, from White's point of view."""
    cached = pawn_cache.get((white_pawns, black_pawns))
    if cached is not None:
        return cached

    score = 0
    for pawns, enemy, passed_masks, sign in (
        (white_pawns, black_pawns, PASSED_W, 1),
        (black_pawns, white_pawns, PASSED_B, -1),
    ):
        for file in range(8):
            count = (pawns & chess.BB_FILES[file]).bit_count()
            if count > 1:
                score -= sign * DOUBLED_PENALTY * (count - 1)

        remaining = pawns
        while remaining:
            lowest = remaining & -remaining
            square = lowest.bit_length() - 1
            remaining ^= lowest
            if not pawns & NEIGHBOUR_FILES[FILE_OF[square]]:
                score -= sign * ISOLATED_PENALTY
            if not enemy & passed_masks[square]:
                # Rank counted from the pawn's own side, so both colours read the same table.
                advance = RANK_OF[square] if sign > 0 else 7 - RANK_OF[square]
                score += sign * PASSED_BONUS[advance]

    if len(pawn_cache) < PAWN_CACHE_LIMIT:
        pawn_cache[(white_pawns, black_pawns)] = score
    return score


def evaluate(board: chess.Board) -> int:
    """Material and placement, from the side to move. Positive means the mover is better."""
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    pawns, knights = board.pawns, board.knights
    bishops, rooks, queens = board.bishops, board.rooks, board.queens

    score = (
        scan(pawns & white, PAWN_W)
        - scan(pawns & black, PAWN_B)
        + scan(knights & white, KNIGHT_W)
        - scan(knights & black, KNIGHT_B)
        + scan(bishops & white, BISHOP_W)
        - scan(bishops & black, BISHOP_B)
        + scan(rooks & white, ROOK_W)
        - scan(rooks & black, ROOK_B)
        + scan(queens & white, QUEEN_W)
        - scan(queens & black, QUEEN_B)
    )

    score += pawn_score(pawns & white, pawns & black)

    # Two bishops cover both colour complexes, which is worth more than the pieces separately.
    if (bishops & white).bit_count() >= 2:
        score += BISHOP_PAIR
    if (bishops & black).bit_count() >= 2:
        score -= BISHOP_PAIR

    # A rook is worth having where it can actually see down the board.
    for rook_set, sign in ((rooks & white, 1), (rooks & black, -1)):
        remaining = rook_set
        while remaining:
            lowest = remaining & -remaining
            remaining ^= lowest
            file_mask = chess.BB_FILES[FILE_OF[lowest.bit_length() - 1]]
            if not pawns & file_mask:
                score += sign * ROOK_OPEN_FILE
            elif not (pawns & (white if sign > 0 else black)) & file_mask:
                score += sign * ROOK_SEMI_OPEN_FILE

    # Popcounts beat counting squares one at a time, and the phase is only ever a weighted count.
    phase = (knights | bishops).bit_count() + rooks.bit_count() * 2 + queens.bit_count() * 4
    phase = min(phase, TOTAL_PHASE)
    endgame = TOTAL_PHASE - phase

    kings = board.kings
    white_king = kings & white
    if white_king:
        square = white_king.bit_length() - 1
        score += (KING_MIDGAME[square] * phase + KING_ENDGAME[square] * endgame) // TOTAL_PHASE
    black_king = kings & black
    if black_king:
        square = black_king.bit_length() - 1
        score -= (KING_MG_B[square] * phase + KING_EG_B[square] * endgame) // TOTAL_PHASE

    return score if board.turn == chess.WHITE else -score


def capture_value(board: chess.Board, move: chess.Move) -> int:
    """MVV-LVA: take the biggest thing with the smallest thing, and try that order first."""
    victim = board.piece_type_at(move.to_square)
    gain = VALUE[victim] if victim is not None else VALUE[chess.PAWN]  # None means en passant
    if move.promotion is not None:
        gain += VALUE[move.promotion]
    attacker = board.piece_type_at(move.from_square)
    return gain * 16 - (VALUE[attacker] if attacker is not None else 0)


def ordered(board: chess.Board, best: chess.Move | None, ply: int) -> list[chess.Move]:
    """The transposition move, then captures, then the quiet moves that have been cutting."""
    slot = killers[ply]

    def rank(move: chess.Move) -> int:
        if move == best:
            return 1 << 30
        if board.is_capture(move) or move.promotion is not None:
            return (1 << 20) + capture_value(board, move)
        if move in slot:
            return 1 << 19
        return history.get((move.from_square, move.to_square), 0)

    return sorted(board.legal_moves, key=rank, reverse=True)


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
        board.push(move)
        score = -quiesce(board, -beta, -alpha)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


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
        _, score, flag, _ = stored
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

    # Null move: hand the opponent a free move, and if the position still beats beta then it
    # was never worth searching properly. Skipped in check, and skipped without a piece on the
    # board, because those are the positions where passing would genuinely have been best.
    if depth >= NULL_MIN_DEPTH and not in_check and has_pieces(board, board.turn):
        board.push(chess.Move.null())
        score = -negamax(board, depth - 1 - NULL_REDUCTION, -beta, -beta + 1, ply + 1)
        board.pop()
        if score >= beta:
            return beta

    moves = ordered(board, stored[3] if stored is not None else None, ply)
    if not moves:
        return -MATE + ply if in_check else 0

    best_move: chess.Move | None = None
    best_score = -INFINITY
    for index, move in enumerate(moves):
        quiet = not board.is_capture(move) and move.promotion is None
        board.push(move)

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

    if len(transposition) < TT_LIMIT:
        flag = EXACT if original < best_score < beta else (LOWER if best_score >= beta else UPPER)
        transposition[key] = (depth, best_score, flag, best_move)
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
            for move in ordered(board, choice, 0):
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
