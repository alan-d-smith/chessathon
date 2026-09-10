"""The submission entrypoint. The platform imports this file and calls get_move.

A negamax search with alpha-beta, iterative deepening, a transposition table that survives
across moves, quiescence at the leaves, and an evaluation of material plus piece-square tables
tapered towards the endgame. Pure python-chess: nothing here needs a warm-up, so import costs
almost nothing of the 90 second budget.

The ordering of priorities is deliberate. Alpha-beta only pays for itself when good moves come
first, so most of the code below is about move ordering; and a flag is a whole point, so the
search abandons a depth rather than finish it, and get_move always has a legal move in hand.
"""

import hashlib
import os
import sys
import time
from collections import Counter
from collections.abc import Hashable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any, Final

import chess

import weights

# Bumped when a build is frozen for upload. The digest below is what actually identifies a
# build; this is only here so a log is readable without looking anything up.
VERSION: Final = "v12"

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
# Search a fixed number of nodes instead of a slice of the clock. Only ever set by the tools
# that compare versions, where a time bound would decide the answer before the change did.
NODE_BUDGET: Final = int(os.environ.get("AGENT_NODE_BUDGET", "0"))

# Wall time is what the referee measures, so the budget leaves room for the round trip and the
# search checks the clock mid-flight rather than only between depths.
MOVE_OVERHEAD_MS: Final = 150
# A fixed thirtieth of what is left never reaches zero: it spends four seconds on move ten,
# where the position is nearly book, and under one on move eighty, where the game is being
# decided. Three rated losses ended in mate with a quarter of the clock still unspent. The FEN
# carries the move number, so the divisor can be what is plausibly left to play instead.
EXPECTED_TOTAL_MOVES: Final = 56
MIN_REMAINING_MOVES: Final = 18
MAX_CLOCK_FRACTION: Final = 0.35
INCREMENT_SHARE: Final = 0.75
# A search that has just discovered the position is worse than it thought is worth paying
# more for, because that is where another iteration can still find something. A root move that
# merely changed is not the same signal: in round 40 that fired on 31% of moves, spent 78
# seconds on the first 25 and left 68 for the next 47, and four of six blunders came in the
# starved tail. So the trigger is the score falling, not the move moving.
EXTENSION_FACTOR: Final = 2.5
INSTABILITY_CP: Final = 40
# A score that falls is not the only way a search says it is unsure. Round 72 threw a won
# game on a move whose score never moved 25 points while the root move changed on every one
# of five iterations, and one more iteration played the move that held the win. A search
# that has never once agreed with itself has not found its move yet. Counted over 272 rated
# moves this fires on 8.1%, where extending on any root change fires on 26.5% and spends a
# clock that has nothing spare. The floor is because agreeing at depth two means nothing.
CHURN_MIN_ITERATIONS: Final = 4
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
# The order the compiled move ordering indexes piece values by: index is the piece type.
PIECE_ORDER: Final = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

# One entry per slot, indexed by the key's hash. A power of two so the index is a mask.
TT_BITS: Final = 20
TT_SIZE: Final = 1 << TT_BITS
TT_MASK: Final = TT_SIZE - 1
# Walking back to a position seen once is worth discouraging, not forbidding: sometimes
# it is the only move. Walking into the third occurrence is not a matter of degree, and
# is scored below as the draw it actually is.
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
# Exercises doubled, isolated and passed pawns, open and half open files and a broken
# shield, so a compiled evaluation that agrees here is not merely agreeing about zeroes.
SANITY_FEN: Final = "r3k2r/1pp2ppp/p1n5/3Pp3/1P6/P1N2N2/5PPP/R3K2R w KQkq - 0 1"


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
# slot -> (signature, depth, score, flag, move, generation), or None while never written.
transposition: list[tuple[int, int, int, int, chess.Move | None, int] | None] = (
    [None] * TT_SIZE
)
# Which search an entry belongs to. Entries from an earlier move are the first to go when
# a slot is contested, because the position they describe is usually behind us.
generation = 0
# Killers and history are keyed by move_key rather than by Move: the objects are dataclasses
# and comparing them is one of the more expensive things the ordering used to do.
killers: list[list[int]] = [[-1, -1] for _ in range(MAX_DEPTH + 2)]
# Indexed by move_key of a quiet move, which never exceeds from | to << 6, so the whole
# range fits in one flat table and a miss is a zero already sitting there.
HISTORY_SLOTS: Final = 1 << 12
history: Any = [0] * HISTORY_SLOTS
# Counted, not just remembered. The referee claims the draw on the third occurrence, so
# the difference between having been somewhere once and twice is the difference between
# a nudge away from a line and the line being worth exactly nothing.
seen: Counter[Hashable] = Counter()

nodes = 0
deadline = 0.0
reached = 0  # deepest iteration completed on the last move, for diagnostics
extensions = 0  # how many iterations asked for more time, counted across the game

# The increment is inferred from how the clock moves between our own turns, starting from the
# published 0.5s and correcting itself after one move at whatever the real time control is.
increment_ms = float(ASSUMED_INCREMENT_MS)
last_clock_ms: float | None = None
last_spent_ms = 0.0


def tick() -> None:
    """Give up on the clock inside the search, not only between depths.

    A node budget replaces the clock entirely when one is set, which makes the search
    reproducible: the same position gives the same move every time, on any machine. Nothing
    sets it in a real game, so the shipped behaviour is the clock exactly as before. It exists
    because a time bound makes every comparison noisy -- the same position at the same clock
    was found to reach depth 5 once and depth 6 the next time, and play a different move.
    """
    global nodes
    nodes += 1
    if NODE_BUDGET:
        if nodes >= NODE_BUDGET:
            raise Timeout
        return
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


def evaluate_tables(board: chess.Board) -> int:
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


def load_fast_tables(cached: bool = True) -> bool:
    """Compile evaluate_tables, and adopt it only if it agrees with the python it replaces.

    The tables are the largest thing left in an evaluation and every line of them is integer
    bitboard work, which is exactly what numba is for. Nothing here changes what is computed:
    the same score comes out, the same search follows from it, and if numba is missing or the
    compile fails the python below runs unchanged.
    """
    global evaluate_tables, evaluate_tables_python

    try:
        import os
        import tempfile

        os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())
        import numpy as np
        from numba import njit
    except ImportError:
        return False

    # Grouped so the call passes a handful of arrays rather than a dozen. Passed, never closed
    # over: numba freezes a closed-over array into the cached artefact, and weights.py changes.
    placement = np.array([TABLES_W, TABLES_B], dtype=np.int64)
    structural = np.array(STRUCTURAL, dtype=np.int64)
    files = np.array([[chess.BB_FILES[f] for f in range(8)], NEIGHBOUR_FILES], dtype=np.uint64)
    zones = np.array([PASSED, SHIELD], dtype=np.uint64)
    scan_table = [0] * 64
    for square in range(64):
        scan_table[(((1 << square) * 0x03F79D71B4CB0A89) & 0xFFFFFFFFFFFFFFFF) >> 58] = square
    geometry = np.array([FILE_OF, RANK_OF, scan_table], dtype=np.int64)
    magic = np.uint64(0x03F79D71B4CB0A89)

    @njit(cache=cached)
    def count_bits(mask):  # type: ignore[no-untyped-def]
        """Population count. Every constant is uint64: mixing widths here silently gives floats."""
        mask = mask - ((mask >> np.uint64(1)) & np.uint64(0x5555555555555555))
        mask = (mask & np.uint64(0x3333333333333333)) + (
            (mask >> np.uint64(2)) & np.uint64(0x3333333333333333)
        )
        mask = (mask + (mask >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return np.int64((mask * np.uint64(0x0101010101010101)) >> np.uint64(56))

    # Spelled out so the dispatcher has nothing to work out per call, and so a python int
    # is unboxed straight to a uint64 rather than wrapped by hand first. Arrays are declared
    # C contiguous, which is what np.array gives and what lets numba index them directly.
    signature = (
        "int64(uint64, uint64, uint64, uint64, uint64, uint64, uint64, uint64, boolean,"
        " int64[:, :, :, ::1], int64[:, ::1], int64[:, ::1], uint64[:, ::1],"
        " uint64[:, :, ::1], uint64)"
    )

    @njit(signature, cache=cached)
    def scored(  # type: ignore[no-untyped-def]
        pawns, knights, bishops, rooks, queens, kings, white, black, white_to_move,
        placement, structural, geometry, files, zones, magic,
    ):
        """evaluate_tables, in integer arithmetic throughout. Positive is good for White."""
        one = np.uint64(1)
        shift = np.uint64(58)
        empty = np.uint64(0)
        phase = count_bits(knights | bishops) + count_bits(rooks) * 2 + count_bits(queens) * 4
        if phase > 24:
            phase = 24

        score = np.int64(0)
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
            for side in range(2):
                squares = board & (white if side == 0 else black)
                while squares:
                    lowest = squares & (~squares + one)
                    square = geometry[2, (lowest * magic) >> shift]
                    if side == 0:
                        score += placement[0, phase, piece, square]
                    else:
                        score -= placement[1, phase, piece, square]
                    squares ^= lowest

        # Doubled, isolated and passed pawns. The python tallies counts and multiplies once at
        # the end; adding each pawn's own weighted contribution comes to the same integer.
        for side in range(2):
            if side == 0:
                mine = pawns & white
                theirs = pawns & black
                sign = np.int64(1)
                view = 1
            else:
                mine = pawns & black
                theirs = pawns & white
                sign = np.int64(-1)
                view = 0
            for file in range(8):
                doubled = count_bits(mine & files[0, file])
                if doubled > 1:
                    score += sign * (doubled - 1) * structural[phase, 0]
            remaining = mine
            while remaining:
                lowest = remaining & (~remaining + one)
                square = geometry[2, (lowest * magic) >> shift]
                remaining ^= lowest
                if (mine & files[1, geometry[0, square]]) == empty:
                    score += sign * structural[phase, 1]
                if (theirs & zones[0, view, square]) == empty:
                    advance = geometry[1, square] if view == 1 else 7 - geometry[1, square]
                    if 2 <= advance <= 7:
                        score += sign * structural[phase, advance]

        if count_bits(bishops & white) >= 2:
            score += structural[phase, 8]
        if count_bits(bishops & black) >= 2:
            score -= structural[phase, 8]

        for side in range(2):
            if side == 0:
                mine = white
                sign = np.int64(1)
                view = 1
            else:
                mine = black
                sign = np.int64(-1)
                view = 0
            remaining = rooks & mine
            while remaining:
                lowest = remaining & (~remaining + one)
                square = geometry[2, (lowest * magic) >> shift]
                remaining ^= lowest
                file_mask = files[0, geometry[0, square]]
                if (pawns & file_mask) == empty:
                    score += sign * structural[phase, 9]
                elif (pawns & mine & file_mask) == empty:
                    score += sign * structural[phase, 10]

            king = kings & mine
            if king:
                square = geometry[2, (king * magic) >> shift]
                present = count_bits(pawns & mine & zones[1, view, square])
                if present > 3:
                    present = 3
                score += sign * (3 - present) * structural[phase, 11]

        score += structural[phase, 12] if white_to_move else -structural[phase, 12]
        return score if white_to_move else -score

    def fast(board: chess.Board) -> int:
        """The jitted tables, handed the board's own integers with no conversion in python."""
        score: int = scored(
            board.pawns,
            board.knights,
            board.bishops,
            board.rooks,
            board.queens,
            board.kings,
            board.occupied_co[chess.WHITE],
            board.occupied_co[chess.BLACK],
            board.turn == chess.WHITE,
            placement,
            structural,
            geometry,
            files,
            zones,
            magic,
        )
        return score

    # Compile, and check it against the python before anything is rebound. A disagreement here
    # is a compile that went wrong, and the answer to that is to keep the python.
    try:
        for probe in (chess.Board(), chess.Board(SANITY_FEN)):
            if fast(probe) != evaluate_tables(probe):
                return False
    except Exception:
        return False
    evaluate_tables_python = evaluate_tables
    evaluate_tables = fast
    return True


# Kept so the jitted tables can always be checked against the python they replaced.
evaluate_tables_python = evaluate_tables
USING_FAST_TABLES: Final = load_fast_tables() or load_fast_tables(cached=False)

# What the search calls. Rebound below if a network ships, either to replace this or to
# correct it; the search itself never needs to know which.
evaluate = evaluate_tables

# An optional network, used only when weights/net.npz ships alongside this file. numpy and
# numba are imported inside the loader rather than at the top, so an agent without a network
# pays nothing for the possibility of one: no import cost, no compile, no new way to fail.
# Typed loosely on purpose: numpy and numba are not imported unless a network ships, so
# nothing here can be given a real type at module scope without importing them.
net_forward: Any = None
net_state: dict[str, Any] = {}

# The accumulator: the network's hidden layer carried along with the board rather than rebuilt
# from the pieces at every evaluation. Two of them, one per point of view, because the features
# are relative to the side to move and every ply swaps which side that is. Indexed by how many
# moves the board has had pushed onto it, so a level is always written from its parent and can
# never drift out of step with the board -- and unmaking costs nothing at all, because the
# parent level is still sitting there untouched.
ACC_LEVELS: Final = 256
net_refresh: Any = None
net_advance: Any = None
net_readout: Any = None
# True only inside a search, where every push goes through push_move. Anywhere else an
# evaluation falls back to the full refresh rather than trust a stack it did not build.
accumulating = False


def load_net() -> bool:
    """Load and compile the network if one shipped. Returning False keeps the tables."""
    path = Path(__file__).resolve().parent / "weights" / "net.npz"
    if not path.is_file():
        return False
    try:
        # numba compiles at import and caches the result next to the source, which is read only
        # on the platform. Pointing it at the writable scratch directory first means the compile
        # is paid once per machine rather than once per process: about 2.4 seconds either way
        # here, but that is 2.4 seconds of the init budget on every game, and hours across a
        # self-play run that starts two processes per game.
        import os
        import tempfile

        os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())
        import numpy as np
        from numba import njit
    except ImportError:
        return False

    try:
        with np.load(path) as data:
            residual = bool(data["residual"][0]) if "residual" in data else False
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

    @njit(cache=True)
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

    @njit(cache=True)
    def refresh(  # type: ignore[no-untyped-def]
        acc, level, pawns, knights, bishops, rooks, queens, kings, white, black,
        weight_in, bias_in, lookup, magic,
    ):
        """Build one level from the board itself. The only place a level is built from scratch."""
        for unit in range(acc.shape[2]):
            acc[level, 0, unit] = bias_in[unit]
            acc[level, 1, unit] = bias_in[unit]
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
            for colour in range(2):
                squares = board & (white if colour == 0 else black)
                while squares:
                    lowest = squares & (~squares + np.uint64(1))
                    square = lookup[(lowest * magic) >> np.uint64(58)]
                    # The same feature seen from both ends: the other side owns it, and the
                    # board is mirrored so each view has its own first rank at the bottom.
                    white_feature = colour * 384 + piece * 64 + square
                    black_feature = (1 - colour) * 384 + piece * 64 + (square ^ 56)
                    for unit in range(acc.shape[2]):
                        acc[level, 0, unit] += weight_in[white_feature, unit]
                        acc[level, 1, unit] += weight_in[black_feature, unit]
                    squares ^= lowest

    @njit(cache=True)
    def advance(acc, level, weight_in, packed, count):  # type: ignore[no-untyped-def]
        """Write the level below from this one, applying only what the move changed."""
        for unit in range(acc.shape[2]):
            acc[level + 1, 0, unit] = acc[level, 0, unit]
            acc[level + 1, 1, unit] = acc[level, 1, unit]
        for slot in range(count):
            code = (packed >> (11 * slot)) & 0x7FF
            sign = 1.0 if (code & 1) == 1 else -1.0
            square = (code >> 1) & 63
            piece = (code >> 7) & 7
            colour = (code >> 10) & 1
            white_feature = colour * 384 + piece * 64 + square
            black_feature = (1 - colour) * 384 + piece * 64 + (square ^ 56)
            for unit in range(acc.shape[2]):
                acc[level + 1, 0, unit] += sign * weight_in[white_feature, unit]
                acc[level + 1, 1, unit] += sign * weight_in[black_feature, unit]

    @njit(cache=True)
    def readout(acc, level, view, weight_out, bias_out):  # type: ignore[no-untyped-def]
        """The output layer, over a hidden layer that has already been kept up to date."""
        total = bias_out
        for unit in range(acc.shape[2]):
            value = acc[level, view, unit]
            if value > 0.0:
                total += value * weight_out[unit]
        return total

    global net_forward, net_refresh, net_advance, net_readout
    net_forward = forward
    net_refresh = refresh
    net_advance = advance
    net_readout = readout
    net_state.update(
        # A residual net scores the difference from the tables rather than the position, so the
        # two must be added. The flag ships with the weights: reading it from the file means a
        # net cannot be loaded in the wrong mode by mistake.
        residual=residual,
        numpy=np,
        hidden_w=hidden_w,
        hidden_b=hidden_b,
        out_w=out_w,
        out_b=out_b,
        table=table,
        magic=np.uint64(0x03F79D71B4CB0A89),
        acc=np.zeros((ACC_LEVELS, 2, hidden_b.shape[0]), dtype=np.float32),
    )

    # numba compiles on the first call, not at decoration, so the warm-up below is also where
    # a broken cache first shows itself. It can break for reasons that have nothing to do with
    # us: a cache directory that is not writable, a half written entry, or an entry pickled
    # against a module name that no longer exists. Every one of those raises, and an exception
    # here would be raised at import, which loses every game of the round rather than one move.
    # So the cache is an optimisation that is allowed to fail: compile fresh without it, and
    # failing that, play on the tables alone.
    def warm() -> None:
        """Every jitted entry point, once, with the argument types the search will use."""
        probe = chess.Board()
        evaluate_net(probe)
        refresh_accumulator(probe)
        net_advance(net_state["acc"], 0, net_state["hidden_w"], 0, 0)
        net_readout(net_state["acc"], 0, 0, net_state["out_w"], net_state["out_b"])

    try:
        warm()
        return True
    # Whatever the cache did, it must not reach the referee.
    except Exception:
        pass
    try:
        # The same function compiled in memory. Costs the compile on every process instead of
        # once per machine, which is seconds of the init budget rather than the whole game.
        net_forward = njit(cache=False)(forward.py_func)
        net_refresh = njit(cache=False)(refresh.py_func)
        net_advance = njit(cache=False)(advance.py_func)
        net_readout = njit(cache=False)(readout.py_func)
        warm()
        return True
    # The tables are always there.
    except Exception:
        return False


# Endgame tablebases, if any shipped. AGENTS.md allows them: a tablebase is solved ground truth
# rather than an engine's opinion. Only the three and four man tables are here, which is 4MB of
# the 50MB the zip may hold; five man is 279MB and does not fit. That band is small but it is
# exactly where the search was failing, because a piece square evaluation has nothing to say
# about driving a bare king to the edge: king and rook against king was being shuffled into a
# threefold repetition, and so were two bishops.
TB_MEN: Final = 4
# Typed loosely on purpose: naming chess.syzygy here would import it at module scope, and it
# is only ever needed by a game that reaches four men.
tablebase: Any = None


def load_tablebase() -> Any:
    """Open the shipped tables, or return None and play on the search alone."""
    path = Path(__file__).resolve().parent / "weights" / "syzygy"
    if not path.is_dir():
        return None
    try:
        import chess.syzygy

        return chess.syzygy.open_tablebase(str(path))
    except Exception:
        return None


def tablebase_move(board: chess.Board) -> chess.Move | None:
    """The move the tables say is best, or None when they cannot answer.

    Winning is not enough on its own: every move that keeps a won position looks equally won to
    a win/draw/loss probe, which is how a won ending gets shuffled. Distance to zero is what
    orders them, so the move chosen is the one that actually makes progress towards the pawn
    move or capture that resets the fifty move count, and from there to mate.
    """
    best: chess.Move | None = None
    best_key: tuple[int, int, int, int] | None = None
    for move in board.legal_moves:
        zeroing = board.is_capture(move) or board.piece_type_at(move.from_square) == chess.PAWN
        board.push(move)
        try:
            if board.is_checkmate():
                board.pop()
                return move
            # Both probes are from the side to move, which after our move is the opponent, so
            # both are negated to read as ours.
            outcome = -tablebase.probe_wdl(board)
            distance = -tablebase.probe_dtz(board)
        except Exception:
            board.pop()
            return None
        repeat = board._transposition_key() in seen
        board.pop()

        # Win first. Then never repeat a won position: distance to zero counts plies to the
        # next pawn move or capture, not to mate, so every king move in a won pawn ending can
        # share one distance and the choice between them falls to whatever came first. That is
        # how king and two pawns against a bare king was drawn by repetition. Then the shortest
        # distance when winning and the longest when losing, and finally a zeroing move, which
        # is the one that actually resets the fifty move count and moves the game forward.
        walking = 0 if (repeat and outcome > 0) else 1
        progress = -abs(distance) if outcome > 0 else abs(distance)
        forward = (1 if zeroing else 0) if outcome > 0 else (0 if zeroing else 1)
        # When winning, a zeroing move outranks the distance. Distance to zero counts plies to
        # the next pawn move or capture by either side, so a defender with a pawn can keep
        # resetting it and the number stops describing our progress at all: rook against king
        # and pawn sat at a distance of one while the rook toured the eighth rank. Every move
        # considered here already holds the win, so taking the one that resets the fifty move
        # count cannot throw it away, and it is the only thing that reliably ends the game.
        key = (outcome, walking, forward, progress) if outcome > 0 else (
            outcome, walking, progress, forward
        )
        if best_key is None or key > best_key:
            best, best_key = move, key
    return best


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


def evaluate_residual(board: chess.Board) -> int:
    """The tuned tables, corrected by the network.

    Learning the whole evaluation from raw piece placement means rediscovering passed pawns,
    king safety and phase tapering, all of which the tables are simply handed. Learning only
    what the tables get wrong starts level with them instead of a hundred Elo behind, and the
    network can only add to what already works.
    """
    return evaluate_tables(board) + evaluate_net(board)


def refresh_accumulator(board: chess.Board, level: int = 0) -> None:
    """Rebuild one level of the accumulator from the board."""
    np = net_state["numpy"]
    net_refresh(
        net_state["acc"],
        level,
        np.uint64(board.pawns),
        np.uint64(board.knights),
        np.uint64(board.bishops),
        np.uint64(board.rooks),
        np.uint64(board.queens),
        np.uint64(board.kings),
        np.uint64(board.occupied_co[chess.WHITE]),
        np.uint64(board.occupied_co[chess.BLACK]),
        net_state["hidden_w"],
        net_state["hidden_b"],
        net_state["table"],
        net_state["magic"],
    )


def delta(board: chess.Board, move: chess.Move) -> tuple[int, int]:
    """What a move changes, packed eleven bits per feature: colour, piece, square, and sign.

    One integer rather than an array, so the jitted update takes scalars and allocates nothing
    per ply. A move changes at most four features -- castling moves two pieces -- which is
    forty four bits. Read off the move rather than by diffing the board before and after:
    diffing needs no special cases but costs more than the refresh it is meant to replace.
    """
    if not move:
        return 0, 0  # a null move moves no piece, and both views are kept, so nothing changes
    from_bb = 1 << move.from_square
    if board.pawns & from_bb:
        piece = 0
    elif board.knights & from_bb:
        piece = 1
    elif board.bishops & from_bb:
        piece = 2
    elif board.rooks & from_bb:
        piece = 3
    elif board.queens & from_bb:
        piece = 4
    elif board.kings & from_bb:
        piece = 5
    else:
        return 0, 0  # not a move in this position; the push below will say so
    mover = 0 if board.turn == chess.WHITE else 1
    landed = move.promotion - 1 if move.promotion else piece
    packed = (mover << 10) | (piece << 7) | (move.from_square << 1)
    packed |= ((mover << 10) | (landed << 7) | (move.to_square << 1) | 1) << 11
    count = 2

    # Only a king castles and only a pawn takes en passant, so both questions are settled by an
    # integer compare before anything more expensive is asked. Everything else is a capture
    # exactly when the square being moved to is occupied, which is one test against occupied.
    if piece == chess.KING - 1 and board.is_castling(move):
        # The king is the pair above; the rook is the other half of the same move.
        home = 0 if board.turn == chess.WHITE else 56
        kingside = move.to_square > move.from_square
        rook_from, rook_to = (home + 7, home + 5) if kingside else (home, home + 3)
        rook = chess.ROOK - 1
        packed |= ((mover << 10) | (rook << 7) | (rook_from << 1)) << 22
        packed |= ((mover << 10) | (rook << 7) | (rook_to << 1) | 1) << 33
        return packed, 4

    to_bb = 1 << move.to_square
    if board.occupied & to_bb:
        if board.pawns & to_bb:
            victim = chess.PAWN
        elif board.knights & to_bb:
            victim = chess.KNIGHT
        elif board.bishops & to_bb:
            victim = chess.BISHOP
        elif board.rooks & to_bb:
            victim = chess.ROOK
        elif board.queens & to_bb:
            victim = chess.QUEEN
        else:
            victim = chess.KING
        square = move.to_square
    elif piece == chess.PAWN - 1 and move.to_square == board.ep_square:
        # A pawn reaching an empty en passant square can only have got there by taking, and the
        # pawn it took is not on the square it moved to. It cannot have pushed there: the square
        # in front of a pawn that has just moved two is the one that pawn came through.
        victim = chess.PAWN
        square = move.to_square + (-8 if board.turn == chess.WHITE else 8)
    else:
        return packed, count
    packed |= (((1 - mover) << 10) | ((victim - 1) << 7) | (square << 1)) << 22
    return packed, 3


def push_move(board: chess.Board, move: chess.Move) -> None:
    """Play a move and carry the accumulator down with it.

    Every push inside a search goes through here. Unmaking needs no counterpart at all: this
    writes the level below and leaves the current one alone, so board.pop() on its own puts the
    evaluation back exactly where it was. That is what makes drift impossible rather than
    merely unlikely -- there is no inverse update to get wrong, and no state to resynchronise.
    """
    level = len(board.move_stack)
    if accumulating and level + 1 < ACC_LEVELS:
        packed, count = delta(board, move)
        board.push(move)
        net_advance(net_state["acc"], level, net_state["hidden_w"], packed, count)
    else:
        board.push(move)


def evaluate_accumulated(board: chess.Board) -> int:
    """The network's view, read from the hidden layer the search has been carrying."""
    level = len(board.move_stack)
    if not accumulating or level >= ACC_LEVELS:
        return evaluate_net(board)
    return int(
        net_readout(
            net_state["acc"],
            level,
            0 if board.turn == chess.WHITE else 1,
            net_state["out_w"],
            net_state["out_b"],
        )
    )


def evaluate_residual_accumulated(board: chess.Board) -> int:
    """evaluate_residual, over the accumulator rather than a refresh."""
    return evaluate_tables(board) + evaluate_accumulated(board)


tablebase = load_tablebase()
USING_NET: Final = load_net()

if USING_NET:
    # load_net has already made the first call, which is what compiles the network and what
    # pays for it: inside the 90 second import budget rather than on the clock, and warmed with
    # the argument types the real calls will use. Reaching here means that call succeeded.
    evaluate = (
        evaluate_residual_accumulated if net_state.get("residual") else evaluate_accumulated
    )


def move_key(move: chess.Move) -> int:
    """From, to and promotion in one integer. Two moves are the same move iff these match."""
    promotion = move.promotion
    return move.from_square | move.to_square << 6 | (promotion << 12 if promotion else 0)


def capture_value(board: chess.Board, move: chess.Move) -> int:
    """MVV-LVA: take the biggest thing with the smallest thing, and try that order first."""
    victim = board.piece_type_at(move.to_square)
    gain = VALUE[victim] if victim is not None else VALUE[chess.PAWN]  # None means en passant
    if move.promotion is not None:
        gain += VALUE[move.promotion]
    attacker = board.piece_type_at(move.from_square)
    return gain * 16 - (VALUE[attacker] if attacker is not None else 0)


# Positions the generator must reproduce before it is allowed anywhere near the search: castling
# from both sides, a pinned piece, promotions with and without a capture, and a position in check
# so the hand-back path is exercised too.
MOVEGEN_PROBES: Final = (
    chess.STARTING_FEN,
    "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
    "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R b KQkq - 0 1",
    "8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
    "8/PPPk4/8/8/8/8/4Kppp/8 b - - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
)


def load_fast_moves(cached: bool = True) -> bool:
    """Compile a move generator that reproduces python-chess move for move, or keep python-chess.

    Move generation is the largest thing left in the search by a distance, and python-chess does
    it in python. This does it in one compiled call. What it must not do is reorder anything:
    the search sorts with a stable sort, so moves that tie on rank come out in generation order,
    and a generator that produced the same moves in a different sequence would search a different
    move first and play a different game. So it follows python-chess phase for phase.

    Two shapes are handed straight back rather than reimplemented: being in check, which has its
    own evasion generator with its own order, and a position where an en passant capture is
    available, whose legality needs a skewer test for one rare move. Together they are about one
    node in twenty, and python-chess answers those exactly as before.
    """
    global history, fast_generate, fast_order, movegen_state

    try:
        import os
        import tempfile

        os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())
        import numpy as np
        from numba import njit
    except ImportError:
        return False

    magic = np.uint64(0x03F79D71B4CB0A89)
    one = np.uint64(1)
    shift = np.uint64(58)

    def ray_table(steps: tuple[tuple[int, bool], ...]) -> Any:
        """Every square beyond a given one along each direction, for the classical scan below."""
        table = np.zeros((2, 2, 64), dtype=np.uint64)
        for index, (step, positive) in enumerate(steps):
            for square in range(64):
                mask = 0
                current = square
                while True:
                    nxt = current + step
                    # Off the board, or wrapped round its edge, and the ray has ended.
                    if not 0 <= nxt < 64 or abs((nxt % 8) - (current % 8)) > 1:
                        break
                    mask |= 1 << nxt
                    current = nxt
                table[0 if positive else 1, index % 2, square] = mask
        return table

    scan = np.zeros(64, dtype=np.int64)
    for square in range(64):
        scan[(((1 << square) * 0x03F79D71B4CB0A89) & 0xFFFFFFFFFFFFFFFF) >> 58] = square
    knight_t = np.array(chess.BB_KNIGHT_ATTACKS, dtype=np.uint64)
    king_t = np.array(chess.BB_KING_ATTACKS, dtype=np.uint64)
    pawn_t = np.array([chess.BB_PAWN_ATTACKS[0], chess.BB_PAWN_ATTACKS[1]], dtype=np.uint64)
    rays = np.array([[chess.ray(a, b) for b in range(64)] for a in range(64)], dtype=np.uint64)
    between = np.array(
        [[chess.between(a, b) for b in range(64)] for a in range(64)], dtype=np.uint64
    )
    rank_empty = np.array([chess.BB_RANK_ATTACKS[s][0] for s in range(64)], dtype=np.uint64)
    file_empty = np.array([chess.BB_FILE_ATTACKS[s][0] for s in range(64)], dtype=np.uint64)
    diag_empty = np.array([chess.BB_DIAG_ATTACKS[s][0] for s in range(64)], dtype=np.uint64)
    rook_rays = ray_table(((8, True), (1, True), (-8, False), (-1, False)))
    bishop_rays = ray_table(((9, True), (7, True), (-9, False), (-7, False)))
    values = np.array(
        [0]
        + [VALUE[piece] for piece in PIECE_ORDER],
        dtype=np.int64,
    )

    @njit(cache=cached, inline="always")
    def lowest_bit(mask, scan):  # type: ignore[no-untyped-def]
        return scan[((mask & (~mask + one)) * magic) >> shift]

    @njit(cache=cached, inline="always")
    def highest_bit(mask, scan):  # type: ignore[no-untyped-def]
        """Smear every bit below the top one down, then isolate it."""
        mask |= mask >> np.uint64(1)
        mask |= mask >> np.uint64(2)
        mask |= mask >> np.uint64(4)
        mask |= mask >> np.uint64(8)
        mask |= mask >> np.uint64(16)
        mask |= mask >> np.uint64(32)
        return scan[((mask ^ (mask >> np.uint64(1))) * magic) >> shift]

    @njit(cache=cached)
    def slide(square, occupied, dirs, scan):  # type: ignore[no-untyped-def]
        """Classical ray attacks: run each direction out to its first blocker, inclusive."""
        attacks = np.uint64(0)
        for direction in range(2):
            ray = dirs[0, direction, square]
            blockers = ray & occupied
            if blockers:
                ray &= ~dirs[0, direction, lowest_bit(blockers, scan)]
            attacks |= ray
        for direction in range(2):
            ray = dirs[1, direction, square]
            blockers = ray & occupied
            if blockers:
                ray &= ~dirs[1, direction, highest_bit(blockers, scan)]
            attacks |= ray
        return attacks

    @njit(cache=cached)
    def attackers(  # type: ignore[no-untyped-def]
        by_white, square, occupied, pawns, knights, bishops, rooks, queens, kings, white, black,
        knight_t, king_t, pawn_t, rook_rays, bishop_rays, scan,
    ):
        """Every piece of one colour bearing on a square, for a given occupancy."""
        straight = slide(square, occupied, rook_rays, scan)
        diagonal = slide(square, occupied, bishop_rays, scan)
        found = (
            (king_t[square] & kings)
            | (knight_t[square] & knights)
            | (straight & (queens | rooks))
            | (diagonal & (queens | bishops))
            | (pawn_t[0 if by_white else 1, square] & pawns)
        )
        return found & (white if by_white else black)

    @njit(cache=cached)
    def piece_attacks(  # type: ignore[no-untyped-def]
        square, bb, occupied, pawns, knights, bishops, rooks, queens, kings, white,
        knight_t, king_t, pawn_t, rook_rays, bishop_rays, scan,
    ):
        if bb & pawns:
            return pawn_t[1 if bb & white else 0, square]
        if bb & knights:
            return knight_t[square]
        if bb & kings:
            return king_t[square]
        attacks = np.uint64(0)
        if bb & bishops or bb & queens:
            attacks = slide(square, occupied, bishop_rays, scan)
        if bb & rooks or bb & queens:
            attacks |= slide(square, occupied, rook_rays, scan)
        return attacks

    generate_types = (
        "int64(uint64, uint64, uint64, uint64, uint64, uint64, uint64, uint64, uint64,"
        " boolean, uint64, uint64, int64[::1], uint64[::1], uint64[::1], uint64[:, ::1],"
        " uint64[:, ::1], uint64[:, ::1], uint64[::1], uint64[::1], uint64[::1],"
        " uint64[:, :, ::1], uint64[:, :, ::1], int64[::1])"
    )

    @njit(generate_types, cache=cached)
    def generate(  # type: ignore[no-untyped-def]
        pawns, knights, bishops, rooks, queens, kings, white, black, occupied,
        white_to_move, clean_castling, to_mask, out,
        knight_t, king_t, pawn_t, rays, between, rank_empty, file_empty, diag_empty,
        rook_rays, bishop_rays, scan,
    ):
        """Legal moves, packed as from | to << 6 | promotion << 12, in python-chess's order.

        Returns the count, or -1 for a position that has to go back to python-chess.
        """
        ours = white if white_to_move else black
        theirs = black if white_to_move else white
        king_bb = kings & ours
        if king_bb == np.uint64(0):
            return -1
        king = highest_bit(king_bb, scan)

        if attackers(not white_to_move, king, occupied, pawns, knights, bishops, rooks, queens,
                     kings, white, black, knight_t, king_t, pawn_t, rook_rays, bishop_rays,
                     scan) != np.uint64(0):
            return -1

        # Our own pieces standing alone between our king and an enemy slider: the pinned ones.
        snipers = (
            ((rank_empty[king] | file_empty[king]) & (rooks | queens))
            | (diag_empty[king] & (bishops | queens))
        ) & theirs
        blockers = np.uint64(0)
        while snipers:
            sniper = highest_bit(snipers, scan)
            snipers ^= np.uint64(1) << np.uint64(sniper)
            occupied_between = between[king, sniper] & occupied
            if occupied_between and (occupied_between & (occupied_between - one)) == np.uint64(0):
                blockers |= occupied_between
        blockers &= ours

        count = 0

        # Everything that is not a pawn, highest square first, targets highest first.
        remaining = ours & ~pawns
        while remaining:
            from_square = highest_bit(remaining, scan)
            from_bb = np.uint64(1) << np.uint64(from_square)
            remaining ^= from_bb
            targets = piece_attacks(
                from_square, from_bb, occupied, pawns, knights, bishops, rooks, queens, kings,
                white, knight_t, king_t, pawn_t, rook_rays, bishop_rays, scan,
            ) & ~ours & to_mask
            while targets:
                to_square = highest_bit(targets, scan)
                targets ^= np.uint64(1) << np.uint64(to_square)
                if from_square == king:
                    if attackers(not white_to_move, to_square, occupied, pawns, knights, bishops,
                                 rooks, queens, kings, white, black, knight_t, king_t, pawn_t,
                                 rook_rays, bishop_rays, scan) != np.uint64(0):
                        continue
                elif blockers & from_bb and not rays[from_square, to_square] & king_bb:
                    continue
                out[count] = from_square | to_square << 6
                count += 1

        # Castling, which python-chess yields after the piece moves and before the pawns.
        backrank = np.uint64(0xFF) if white_to_move else np.uint64(0xFF00000000000000)
        home = king_bb & backrank
        if home:
            rights = clean_castling & backrank & to_mask
            while rights:
                rook_square = highest_bit(rights, scan)
                rook_bb = np.uint64(1) << np.uint64(rook_square)
                rights ^= rook_bb
                if rook_bb < home:
                    king_to_bb = np.uint64(0x04) if white_to_move else np.uint64(0x0400000000000000)
                    rook_to_bb = np.uint64(0x08) if white_to_move else np.uint64(0x0800000000000000)
                else:
                    king_to_bb = np.uint64(0x40) if white_to_move else np.uint64(0x4000000000000000)
                    rook_to_bb = np.uint64(0x20) if white_to_move else np.uint64(0x2000000000000000)
                king_to = highest_bit(king_to_bb, scan)
                king_path = between[king, king_to]
                rook_path = between[rook_square, highest_bit(rook_to_bb, scan)]
                if (occupied ^ home ^ rook_bb) & (king_path | rook_path | king_to_bb | rook_to_bb):
                    continue
                # The king may not start in, pass through, or land on an attacked square, and
                # each leg is tested with the pieces that have already moved taken off.
                walk = king_path | home
                trimmed = occupied ^ home
                blocked = False
                while walk:
                    square = highest_bit(walk, scan)
                    walk ^= np.uint64(1) << np.uint64(square)
                    if attackers(not white_to_move, square, trimmed, pawns, knights, bishops,
                                 rooks, queens, kings, white, black, knight_t, king_t, pawn_t,
                                 rook_rays, bishop_rays, scan):
                        blocked = True
                        break
                if blocked:
                    continue
                trimmed = occupied ^ home ^ rook_bb ^ rook_to_bb
                if attackers(not white_to_move, king_to, trimmed, pawns, knights, bishops, rooks,
                             queens, kings, white, black, knight_t, king_t, pawn_t, rook_rays,
                             bishop_rays, scan):
                    continue
                out[count] = king | king_to << 6
                count += 1

        our_pawns = pawns & ours
        if our_pawns == np.uint64(0):
            return count

        # Pawn captures, promoting queen, rook, bishop, knight, in that order.
        remaining = our_pawns
        while remaining:
            from_square = highest_bit(remaining, scan)
            from_bb = np.uint64(1) << np.uint64(from_square)
            remaining ^= from_bb
            targets = pawn_t[1 if white_to_move else 0, from_square] & theirs & to_mask
            while targets:
                to_square = highest_bit(targets, scan)
                targets ^= np.uint64(1) << np.uint64(to_square)
                if blockers & from_bb and (rays[from_square, to_square] & king_bb) == np.uint64(0):
                    continue
                packed = from_square | to_square << 6
                if to_square >= 56 or to_square < 8:
                    out[count] = packed | 5 << 12
                    out[count + 1] = packed | 4 << 12
                    out[count + 2] = packed | 3 << 12
                    out[count + 3] = packed | 2 << 12
                    count += 4
                else:
                    out[count] = packed
                    count += 1

        # Single then double advances. The double set comes off the single set before to_mask
        # narrows it, which is how python-chess computes it and matters when to_mask is a filter.
        if white_to_move:
            singles = (our_pawns << np.uint64(8)) & ~occupied
            doubles = (singles << np.uint64(8)) & ~occupied & np.uint64(0x00000000FFFF0000)
        else:
            singles = (our_pawns >> np.uint64(8)) & ~occupied
            doubles = (singles >> np.uint64(8)) & ~occupied & np.uint64(0x0000FFFF00000000)
        singles &= to_mask
        doubles &= to_mask

        while singles:
            to_square = highest_bit(singles, scan)
            singles ^= np.uint64(1) << np.uint64(to_square)
            from_square = to_square - 8 if white_to_move else to_square + 8
            from_bb = np.uint64(1) << np.uint64(from_square)
            if blockers & from_bb and (rays[from_square, to_square] & king_bb) == np.uint64(0):
                continue
            packed = from_square | to_square << 6
            if to_square >= 56 or to_square < 8:
                out[count] = packed | 5 << 12
                out[count + 1] = packed | 4 << 12
                out[count + 2] = packed | 3 << 12
                out[count + 3] = packed | 2 << 12
                count += 4
            else:
                out[count] = packed
                count += 1

        while doubles:
            to_square = highest_bit(doubles, scan)
            doubles ^= np.uint64(1) << np.uint64(to_square)
            from_square = to_square - 16 if white_to_move else to_square + 16
            from_bb = np.uint64(1) << np.uint64(from_square)
            if blockers & from_bb and (rays[from_square, to_square] & king_bb) == np.uint64(0):
                continue
            out[count] = from_square | to_square << 6
            count += 1

        return count

    order_types = (
        "int64(int64[::1], int64[::1], int64, int64, int64, int64, int64, int64[::1],"
        " int64[::1], uint64, uint64, uint64, uint64, uint64, uint64)"
    )

    @njit(order_types, cache=cached)
    def order(  # type: ignore[no-untyped-def]
        out, ranks, count, mode, first, killer0, killer1, history, values,
        pawns, knights, bishops, rooks, queens, theirs,
    ):
        """Rank and sort in place, exactly as the python ordering did.

        A packed move is its own move_key, so killers and history need nothing built first. The
        sort is insertion sort: quick at this size and, which is the point, stable. It moves an
        entry left only past a strictly smaller rank, so ties keep generation order, and that is
        what decides which of two equally ranked moves the search tries first.
        """
        kept = 0
        for index in range(count):
            packed = out[index]
            if packed == first:
                continue
            from_bb = np.uint64(1) << np.uint64(packed & 63)
            to_bb = np.uint64(1) << np.uint64((packed >> 6) & 63)
            promotion = packed >> 12

            if to_bb & pawns:
                victim = 1
            elif to_bb & knights:
                victim = 2
            elif to_bb & bishops:
                victim = 3
            elif to_bb & rooks:
                victim = 4
            elif to_bb & queens:
                victim = 5
            else:
                victim = 0

            if mode == 1 or (to_bb & theirs) or promotion:
                gain = values[victim] if victim else values[1]
                if promotion:
                    gain += values[promotion]
                if from_bb & pawns:
                    attacker = 1
                elif from_bb & knights:
                    attacker = 2
                elif from_bb & bishops:
                    attacker = 3
                elif from_bb & rooks:
                    attacker = 4
                elif from_bb & queens:
                    attacker = 5
                else:
                    attacker = 6
                rank = gain * 16 - values[attacker]
                if mode == 0:
                    rank += 1 << 20
            elif packed in (killer0, killer1):
                rank = 1 << 19
            else:
                rank = history[packed]

            out[kept] = packed
            ranks[kept] = rank
            kept += 1

        for index in range(1, kept):
            move = out[index]
            rank = ranks[index]
            slot = index - 1
            while slot >= 0 and ranks[slot] < rank:
                out[slot + 1] = out[slot]
                ranks[slot + 1] = ranks[slot]
                slot -= 1
            out[slot + 1] = move
            ranks[slot + 1] = rank
        return kept

    state: dict[str, Any] = {
        "out": np.zeros(256, dtype=np.int64),
        "ranks": np.zeros(256, dtype=np.int64),
        "values": values,
        "tables": (
            knight_t, king_t, pawn_t, rays, between, rank_empty, file_empty, diag_empty,
            rook_rays, bishop_rays, scan,
        ),
    }
    table = np.array(history, dtype=np.int64)

    previous = fast_generate, fast_order, movegen_state, history
    fast_generate, fast_order, movegen_state, history = generate, order, state, table
    def unpack(value: int) -> chess.Move:
        return chess.Move(int(value) & 63, (int(value) >> 6) & 63, (int(value) >> 12) or None)

    def unranked(board: chess.Board, move: chess.Move) -> int:
        capture = board.is_capture(move) or move.promotion is not None
        return (1 << 20) + capture_value(board, move) if capture else 0

    try:
        for fen in MOVEGEN_PROBES:
            board = chess.Board(fen)
            legal = list(board.legal_moves)
            # Generation order first, because that is what the stable sort falls back on when
            # moves tie: a generator that agreed only on the set would still play a different
            # game. Ranking is checked separately, below, against the python ordering.
            count = generate(
                board.pawns, board.knights, board.bishops, board.rooks, board.queens,
                board.kings, board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
                board.occupied, board.turn == chess.WHITE, board.clean_castling_rights(),
                chess.BB_ALL, state["out"], *state["tables"],
            )
            if count < 0:
                continue  # handed back, and python-chess answers it exactly as it always did
            if [unpack(value) for value in state["out"][:count]] != legal:
                raise ValueError(fen)
            packed = ordered_moves(board, chess.BB_ALL, 0, -1, -1, -1)
            if packed is None:
                continue
            expected = sorted(legal, key=lambda move: unranked(board, move), reverse=True)
            if [unpack(value) for value in packed] != expected:
                raise ValueError(fen)
    # A compile that disagrees with python-chess is a compile that does not get used.
    except Exception:
        fast_generate, fast_order, movegen_state, history = previous
        return False
    return True


def ordered_moves(
    board: chess.Board, to_mask: int, mode: int, first: int, killer0: int, killer1: int
) -> list[int] | None:
    """The ordered move list as packed integers, or None where python-chess has to do it.

    The packed integers are copied out of the shared buffer before returning, because the search
    holds this list open across recursive calls that will use the buffer again.
    """
    square = board.ep_square
    if square is not None:
        mine = board.pawns & board.occupied_co[board.turn]
        if chess.BB_PAWN_ATTACKS[not board.turn][square] & mine:
            return None
    state: Any = movegen_state
    out = state["out"]
    count = fast_generate(
        board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
        board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK], board.occupied,
        board.turn == chess.WHITE, board.clean_castling_rights(), to_mask, out,
        *state["tables"],
    )
    if count < 0:
        return None
    kept = fast_order(
        out, state["ranks"], count, mode, first, killer0, killer1, history, state["values"],
        board.pawns, board.knights, board.bishops, board.rooks, board.queens,
        board.occupied_co[not board.turn],
    )
    packed: list[int] = out[:kept].tolist()
    return packed


fast_generate: Any = None
fast_order: Any = None
movegen_state: Any = None
USING_FAST_MOVES: Final = load_fast_moves() or load_fast_moves(cached=False)
def candidates(board: chess.Board, best: chess.Move | None, ply: int) -> Iterator[chess.Move]:
    """The transposition move first, then captures, then the quiet moves that have been cutting.

    A generator rather than a list so the move list is never built at all when the first move
    already cuts. Generating moves is the most expensive thing in the search, and the whole
    point of trying the transposition move first is that it usually is the one that cuts.
    """
    if best is not None and board.is_legal(best):
        yield best

    first = move_key(best) if best is not None else -1
    killer, spare = killers[ply]

    if USING_FAST_MOVES:
        packed = ordered_moves(board, chess.BB_ALL, 0, first, killer, spare)
        if packed is not None:
            # Built one at a time as they are asked for: most of this list is never reached,
            # because the ordering exists so that the first move or two causes the cut.
            for value in packed:
                yield chess.Move(value & 63, (value >> 6) & 63, (value >> 12) or None)
            return

    def rank(move: chess.Move) -> int:
        if board.is_capture(move) or move.promotion is not None:
            return (1 << 20) + capture_value(board, move)
        key = move_key(move)
        if key in (killer, spare):
            return 1 << 19
        scored_by_history: int = history[key]
        return scored_by_history

    rest = [move for move in board.legal_moves if move_key(move) != first]
    rest.sort(key=rank, reverse=True)
    yield from rest


def quiesce(board: chess.Board, alpha: int, beta: int) -> int:
    """Resolve the captures. Evaluating mid-exchange is how a good evaluation reads as noise."""
    tick()
    standing = evaluate(board)
    if standing >= beta:
        return beta
    alpha = max(alpha, standing)

    captures: Iterator[chess.Move] | list[chess.Move] | None = None
    if USING_FAST_MOVES:
        packed = ordered_moves(board, board.occupied_co[not board.turn], 1, -1, -1, -1)
        if packed is not None:
            captures = (
                chess.Move(value & 63, (value >> 6) & 63, (value >> 12) or None)
                for value in packed
            )
    if captures is None:
        captures = sorted(
            board.generate_legal_captures(),
            key=lambda move: capture_value(board, move),
            reverse=True,
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
        push_move(board, move)
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


def negamax(
    board: chess.Board, depth: int, alpha: int, beta: int, ply: int, checked: bool | None = None
) -> int:
    tick()
    # Neither side can be short of material while a pawn, rook or queen is still on: that is
    # the first thing python-chess checks, so testing it here skips the call outright.
    if board.halfmove_clock >= 100 or (
        not (board.pawns | board.rooks | board.queens) and board.is_insufficient_material()
    ):
        return 0

    # Never hand a position back to the evaluation with the king under fire: the reply is
    # forced and the score is meaningless. Bounded by ply, so perpetual check cannot recurse
    # forever on an extension that keeps renewing itself.
    # The caller already had to know this to decide how to search the move, and asking the
    # board again cannot give a different answer for the same position.
    in_check = board.is_check() if checked is None else checked
    if in_check and ply < MAX_DEPTH:
        depth += 1

    original = alpha
    # Private, but it is the key the board already keeps for its own repetition checks, and it
    # is the one docs/IDEAS.md points at.
    key = board._transposition_key()
    signature = hash(key)
    # Not "slot": the killer update below already uses that name, and the store at the
    # end of this function needs this index intact.
    bucket = signature & TT_MASK
    stored = transposition[bucket]
    # A slot holds whatever position last claimed it, so the signature has to be checked
    # before the entry means anything at all.
    if stored is not None and stored[0] != signature:
        stored = None
    if stored is not None and stored[1] >= depth:
        score = from_store(stored[2], ply)
        flag = stored[3]
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
        push_move(board, chess.Move.null())
        score = -negamax(board, depth - 1 - NULL_REDUCTION, -beta, -beta + 1, ply + 1)
        board.pop()
        if score >= beta:
            return beta

    best_move: chess.Move | None = None
    best_score = -INFINITY
    index = -1
    for index, move in enumerate(candidates(board, stored[4] if stored else None, ply)):
        quiet = not board.is_capture(move) and move.promotion is None
        push_move(board, move)
        # Asked once here and then handed down: futility wants it, so does late move
        # reduction, and so does every re-search of the same child at a different window.
        gives_check = board.is_check()

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
            and not gives_check
        ):
            board.pop()
            continue

        # Late quiet moves are searched shallow and on a null window, on the bet that the
        # ordering was right and they will not beat alpha. A move that beats it anyway is
        # re-searched at full depth, so the bet costs nothing when it is wrong.
        reduction = 0
        if depth >= LMR_MIN_DEPTH and index >= LMR_MIN_MOVE and quiet and not gives_check:
            reduction = 1 if index < LMR_DEEP_MOVE else 2

        if index == 0:
            score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, gives_check)
        else:
            score = -negamax(
                board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, gives_check
            )
            if reduction and score > alpha:
                score = -negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, gives_check)
            if alpha < score < beta:
                score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, gives_check)

        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, score)
        if alpha >= beta:
            if quiet:
                slot = killers[ply]
                # Not "key": that name already holds this node's transposition key, and the
                # store below needs it intact.
                edge = move_key(move)
                if edge != slot[0]:
                    slot[1] = slot[0]
                    slot[0] = edge
                history[edge] += depth * depth
            break

    # Nothing was yielded, so there was nothing legal to play: mate if the king is attacked,
    # stalemate otherwise. Checked here because the generator never built a list to count.
    if index < 0:
        return -MATE + ply if in_check else 0

    # Take the slot unless it holds something worth more: a deeper result from this same
    # search. Anything from an earlier move goes without argument, because the position it
    # describes is usually one we have already played through. The old table never replaced
    # at all -- it filled by about move 48 and then refused every store for the rest of the
    # game, which cost 1.5 ply from that point and six in an endgame.
    flag = EXACT if original < best_score < beta else (LOWER if best_score >= beta else UPPER)
    existing = transposition[bucket]
    if (
        existing is None
        or existing[5] != generation
        or existing[0] == signature
        or depth >= existing[1]
    ):
        transposition[bucket] = (
            signature, depth, to_store(best_score, ply), flag, best_move, generation
        )
    return best_score


def remaining_moves(move_number: int) -> int:
    """How many more moves to plan for, given how far into the game we already are.

    A game already forty moves old will not last another thirty. The floor matters more than
    the slope: it is what stops a long endgame being played at a tenth of a second a move, and
    because the spend stays a fraction of what is left, the clock asymptotes rather than runs
    out. At this floor it settles around two seconds remaining, spending the increment.
    """
    return max(MIN_REMAINING_MOVES, EXPECTED_TOTAL_MOVES - move_number)


def reset_transposition() -> None:
    """Empty the table, which is only ever right between games rather than during one."""
    transposition[:] = [None] * TT_SIZE


def budget_s(time_left_ms: int, move_number: int) -> tuple[float, float]:
    """What to spend on this move, and the most an unsettled search may extend to.

    The increment is worth spending because it comes back every move, but the contract never
    states it, so it is measured rather than assumed: guessing 0.5s at a time control that pays
    0.1s spends five times the increment every move and walks the clock down to a flag.
    """
    usable = max(0.0, time_left_ms - MOVE_OVERHEAD_MS)
    share = usable / remaining_moves(move_number) + increment_ms * INCREMENT_SHARE
    ceiling = usable * MAX_CLOCK_FRACTION
    return min(share, ceiling) / 1000.0, min(share * EXTENSION_FACTOR, ceiling) / 1000.0


def identify() -> None:
    """Announce the build, before a single move is played.

    A rated game is read days later from its moves alone, and the first question is always which
    upload played it. Answering that by replaying candidates and counting agreements is slow and
    inconclusive; saying it at the time is neither. The digest covers this file and the network
    beside it, so it describes the bytes that are running rather than what a constant claims.

    The loader flags ride along because the other thing a log should never leave open is whether
    the compiled evaluation and move generator came up on that machine, or whether it spent the
    game on the python fallback.
    """
    with suppress(Exception):
        digest = hashlib.sha256()
        here = Path(__file__).resolve()
        digest.update(here.read_bytes())
        net = here.parent / "weights" / "net.npz"
        if net.is_file():
            digest.update(net.read_bytes())
        units = net_state["hidden_b"].shape[0] if USING_NET else 0
        print(
            f"checkers {VERSION} build={digest.hexdigest()[:12]} units={units} "
            f"net={USING_NET} tables={USING_FAST_TABLES} moves={USING_FAST_MOVES} "
            f"syzygy={tablebase is not None}",
            file=sys.stderr,
            flush=True,
        )


def trace(move: chess.Move, depth: int, searched: int, spent_ms: float) -> None:
    """Record what the search did, on stderr, where the referee keeps it.

    A rated game is played on another machine against a clock, so the number of nodes behind a
    move is not recoverable from the record afterwards: the moves and the clocks are not enough,
    because a search that spends a different number of nodes leaves a different transposition
    table and the move after it reads that table. Saying so at the time is the only way a game
    can be replayed exactly rather than approximately.

    stdout carries the move and nothing else may go there. Failing to write must never cost a
    game, so it cannot raise.
    """
    # A log that cannot be written is not worth a forfeit.
    with suppress(Exception):
        print(
            f"{move.uci()} d{depth} n{searched} t{spent_ms:.0f}", file=sys.stderr, flush=True
        )


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    """
    global deadline, nodes, reached, increment_ms, last_clock_ms, last_spent_ms, extensions
    global generation
    global accumulating

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
    # Positions we have already stood in, and the ones our own moves have already produced.
    # The search tests the position after a candidate move, so it is the second set that makes
    # the test able to match at all: a key carries the side to move, so a position with us to
    # move can never equal one with them to move. Without the move played below, this check
    # was dead code that had never once fired.
    seen[board._transposition_key()] += 1
    # A new search, so everything already in the table belongs to an older one.
    generation = (generation + 1) & 0xFFFF
    soft, hard = budget_s(time_left_ms, board.fullmove_number)
    deadline = started + soft
    nodes = 0
    reached = 0

    # Whatever happens below, this is already legal, so a timeout or a bug in the search costs
    # a weaker move rather than the game.
    choice = legal[0]
    settled: chess.Move | None = None
    previous_score = -INFINITY
    changes = 0  # iterations whose best move differed from the one before it
    # Nothing stood in twice means no threefold is reachable, and the scan below can be skipped
    # entirely, which is every move of a normal game.
    repeatable = any(count >= 2 for count in seen.values())

    # A position the tables cover is already solved, and a search can only be wrong about it.
    # Costs one popcount on every other move, which is nothing.
    if tablebase is not None and chess.popcount(board.occupied) <= TB_MEN:
        answer = tablebase_move(board)
        if answer is not None:
            board.push(answer)
            seen[board._transposition_key()] += 1
            board.pop()
            last_clock_ms = float(time_left_ms)
            last_spent_ms = (time.monotonic() - started) * 1000.0
            trace(answer, 0, 0, last_spent_ms)
            return answer.uci()

    # From here every push goes through push_move, so the accumulator tracks the board for the
    # whole search. The timeout below rebuilds the board at the root, which is level zero, so
    # an abandoned iteration lands back on an accumulator that is already correct for it.
    if USING_NET:
        refresh_accumulator(board)
        accumulating = True

    for depth in range(1, MAX_DEPTH + 1):
        best_move: chess.Move | None = None
        best_score = -INFINITY
        alpha = -INFINITY
        finished = False
        try:
            for move in candidates(board, choice, 0):
                push_move(board, move)
                score = -negamax(board, depth - 1, -INFINITY, -alpha, 1)
                # Returning to a position our own play has already produced walks towards
                # the threefold the referee claims automatically, so it is only worth it when
                # we are worse. A won game can otherwise be drawn without ever being told, and
                # round 42 was: two rooks up at +1500, the search took a check that repeated
                # for the third time, and a forty point nudge was never going to outweigh it.
                # A third occurrence is not worth its evaluation minus a penalty. It is worth
                # a draw, because that is what the referee will score it.
                before = seen[board._transposition_key()]
                # The referee claims the threefold on whatever position occurs three times,
                # and in round 42 that was the one after the opponent's reply, not the one
                # after our move: we checked, they had a single legal answer, and the draw
                # was theirs to take. So a winning side has to look one further ply. Guarded
                # by repeatable, because with nothing seen twice there is nothing to find,
                # and by the score, because a draw is only worth refusing when we are winning.
                drawn = before >= 2
                if not drawn and repeatable and score > 0:
                    for reply in board.legal_moves:
                        push_move(board, reply)
                        drawn = seen[board._transposition_key()] >= 2
                        board.pop()
                        if drawn:
                            break
                board.pop()
                if drawn:
                    score = 0
                elif before and score > 0:
                    score -= REPETITION_PENALTY
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            finished = True
        except Timeout:
            # The search unwinds through every frame without popping, so the board is left
            # part way down whatever line it was in. Nothing below used it until the repeat
            # check needed it, and a stale board there is an illegal push, not a wrong score.
            board = chess.Board(fen)

        # A finished depth is trustworthy. An abandoned one is only worth taking if it had
        # already improved on the move the previous depth settled on.
        if best_move is not None and (finished or best_score > -INFINITY):
            choice = best_move
        if not finished:
            break
        reached = depth
        if abs(best_score) > MATE - MAX_DEPTH:
            break  # a mate score will not get better with depth

        # A score that just fell means the last iteration was wrong about this position and
        # the next one is worth paying for. Moving the deadline is the whole mechanism: a
        # settled search finds it already behind and the next iteration stops on its first
        # clock check, costing microseconds.
        unstable = settled is not None and best_score < previous_score - INSTABILITY_CP
        if settled is not None and best_move != settled:
            changes += 1
        # Never the same move twice running, from a depth where that means something.
        if depth >= CHURN_MIN_ITERATIONS and changes == depth - 1:
            unstable = True
        settled, previous_score = best_move, best_score
        if unstable:
            extensions += 1
        deadline = started + (hard if unstable else soft)

    accumulating = False

    # Remember where this move leaves the board, so a later move returning here is counted.
    board.push(choice)
    seen[board._transposition_key()] += 1
    board.pop()

    last_clock_ms = float(time_left_ms)
    last_spent_ms = (time.monotonic() - started) * 1000.0
    trace(choice, reached, nodes, last_spent_ms)
    return choice.uci()


identify()
