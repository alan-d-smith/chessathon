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
CHECK_INTERVAL: Final = 2048

# Wall time is what the referee measures, so the budget leaves room for the round trip and the
# search checks the clock mid-flight rather than only between depths.
MOVE_OVERHEAD_MS: Final = 150
EXPECTED_MOVES: Final = 30
MAX_CLOCK_FRACTION: Final = 0.35
INCREMENT_SHARE: Final = 0.75
ASSUMED_INCREMENT_MS: Final = 500

VALUE: Final = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
# Phase runs from all the heavy pieces on to none of them, and tapers the king between the
# table that wants it castled and the table that wants it marching.
PHASE_WEIGHT: Final = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
TOTAL_PHASE: Final = 24

TT_LIMIT: Final = 400_000
REPETITION_PENALTY: Final = 40


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

TABLES: Final = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
}
# Mirroring once at import saves a square_mirror call in the hottest loop there is.
MIRROR: Final = [chess.square_mirror(square) for square in range(64)]

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


def tick() -> None:
    """Give up on the clock inside the search, not only between depths."""
    global nodes
    nodes += 1
    if nodes % CHECK_INTERVAL == 0 and time.monotonic() > deadline:
        raise Timeout


def phase_of(board: chess.Board) -> int:
    remaining = sum(
        weight * len(board.pieces(piece, chess.WHITE) | board.pieces(piece, chess.BLACK))
        for piece, weight in PHASE_WEIGHT.items()
    )
    return min(remaining, TOTAL_PHASE)


def evaluate(board: chess.Board) -> int:
    """Material and placement, from the side to move. Positive means the mover is better."""
    score = 0
    for piece, table in TABLES.items():
        value = VALUE[piece]
        for square in board.pieces(piece, chess.WHITE):
            score += value + table[square]
        for square in board.pieces(piece, chess.BLACK):
            score -= value + table[MIRROR[square]]

    phase = phase_of(board)
    for colour, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        king = board.king(colour)
        if king is None:
            continue
        square = king if colour == chess.WHITE else MIRROR[king]
        tapered = KING_MIDGAME[square] * phase + KING_ENDGAME[square] * (TOTAL_PHASE - phase)
        score += sign * tapered // TOTAL_PHASE

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
        board.push(move)
        score = -quiesce(board, -beta, -alpha)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def negamax(board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    tick()
    if board.is_insufficient_material() or board.halfmove_clock >= 100:
        return 0

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

    moves = ordered(board, stored[3] if stored is not None else None, ply)
    if not moves:
        return -MATE + ply if board.is_check() else 0

    best_move: chess.Move | None = None
    best_score = -INFINITY
    for move in moves:
        quiet = not board.is_capture(move)
        board.push(move)
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
    """Spend a share of the clock, never a fixed amount, and never most of what is left."""
    usable = max(0.0, time_left_ms - MOVE_OVERHEAD_MS)
    share = usable / EXPECTED_MOVES + ASSUMED_INCREMENT_MS * INCREMENT_SHARE
    return min(share, usable * MAX_CLOCK_FRACTION) / 1000.0


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    """
    global deadline, nodes

    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError(f"no legal move in {fen}")

    seen.add(board._transposition_key())
    deadline = time.monotonic() + budget_s(time_left_ms)
    nodes = 0

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
        if abs(best_score) > MATE - MAX_DEPTH:
            break  # a mate score will not get better with depth

    return choice.uci()
