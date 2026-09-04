"""The evaluation's feature vector, which is the same thing the agent computes at every leaf.

The evaluation is a dot product: every term is some count of a thing on the board multiplied
by a weight, and the weights are currently numbers somebody typed. Writing the same evaluation
out as an explicit feature vector makes that dot product visible, and once it is visible the
weights can be fitted to labelled positions instead of guessed.

This module is the single definition of that layout. `tools/tune.py` fits weights against it
and `weights.py` holds the result, so agent.py and the tuner can never drift apart on what
feature number 412 is supposed to mean.

Every term is tapered: each weight is a midgame value and an endgame value, blended by phase.
So a feature contributes `phase/24` to its midgame slot and `1 - phase/24` to its endgame slot,
which keeps the whole thing linear and therefore fittable in closed form.
"""

import chess

PIECES: tuple[int, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
TOTAL_PHASE = 24
SQUARES = 64

# 6 piece types on 64 squares, then the structural terms, each with a midgame and an endgame
# slot. Index arithmetic lives here so nothing else has to know it.
PSQT_TERMS = len(PIECES) * SQUARES
STRUCTURAL: tuple[str, ...] = (
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
)
TERMS = PSQT_TERMS + len(STRUCTURAL)
FEATURES = TERMS * 2  # a midgame and an endgame slot for every term

MIRROR = [chess.square_mirror(square) for square in range(SQUARES)]
FILE_OF = [chess.square_file(square) for square in range(SQUARES)]
RANK_OF = [chess.square_rank(square) for square in range(SQUARES)]
NEIGHBOUR_FILES = [
    (chess.BB_FILES[file - 1] if file > 0 else 0)
    | (chess.BB_FILES[file + 1] if file < 7 else 0)
    for file in range(8)
]


def _ahead(square: int, colour: chess.Color) -> int:
    file = FILE_OF[square]
    files = chess.BB_FILES[file] | NEIGHBOUR_FILES[file]
    ranks = range(RANK_OF[square] + 1, 8) if colour else range(0, RANK_OF[square])
    mask = 0
    for rank in ranks:
        mask |= chess.BB_RANKS[rank]
    return files & mask


def _shield(square: int, colour: chess.Color) -> int:
    file = FILE_OF[square]
    files = chess.BB_FILES[file] | NEIGHBOUR_FILES[file]
    steps = (RANK_OF[square] + 1, RANK_OF[square] + 2) if colour else (
        RANK_OF[square] - 1,
        RANK_OF[square] - 2,
    )
    mask = 0
    for step in steps:
        if 0 <= step <= 7:
            mask |= chess.BB_RANKS[step]
    return files & mask


PASSED = (
    [_ahead(square, chess.BLACK) for square in range(SQUARES)],
    [_ahead(square, chess.WHITE) for square in range(SQUARES)],
)
SHIELD = (
    [_shield(square, chess.BLACK) for square in range(SQUARES)],
    [_shield(square, chess.WHITE) for square in range(SQUARES)],
)
SHIELD_WANTED = 3
STRUCTURAL_INDEX = {name: PSQT_TERMS + offset for offset, name in enumerate(STRUCTURAL)}


def phase_of(board: chess.Board) -> int:
    minors = board.knights | board.bishops
    phase = minors.bit_count() + board.rooks.bit_count() * 2 + board.queens.bit_count() * 4
    return min(phase, TOTAL_PHASE)


def counts(board: chess.Board) -> dict[int, int]:
    """Term index to net count, from White's point of view. Black's terms simply count down."""
    net: dict[int, int] = {}

    def add(term: int, amount: int) -> None:
        if amount:
            net[term] = net.get(term, 0) + amount

    for colour, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        mine = board.occupied_co[colour]
        for offset, piece in enumerate(PIECES):
            squares = board.pieces_mask(piece, colour) & mine
            while squares:
                lowest = squares & -squares
                squares ^= lowest
                square = lowest.bit_length() - 1
                # Black's tables are White's, mirrored, so both colours share one set of weights.
                view = square if colour == chess.WHITE else MIRROR[square]
                add(offset * SQUARES + view, sign)

    pawns = board.pawns
    for colour, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        mine = pawns & board.occupied_co[colour]
        theirs = pawns & board.occupied_co[not colour]
        for file in range(8):
            count = (mine & chess.BB_FILES[file]).bit_count()
            if count > 1:
                add(STRUCTURAL_INDEX["doubled"], sign * (count - 1))
        squares = mine
        while squares:
            lowest = squares & -squares
            squares ^= lowest
            square = lowest.bit_length() - 1
            if not mine & NEIGHBOUR_FILES[FILE_OF[square]]:
                add(STRUCTURAL_INDEX["isolated"], sign)
            if not theirs & PASSED[colour][square]:
                advance = RANK_OF[square] if colour == chess.WHITE else 7 - RANK_OF[square]
                if 2 <= advance <= 7:
                    add(STRUCTURAL_INDEX[f"passed_{advance}"], sign)

        if (board.bishops & board.occupied_co[colour]).bit_count() >= 2:
            add(STRUCTURAL_INDEX["bishop_pair"], sign)

        rooks = board.rooks & board.occupied_co[colour]
        while rooks:
            lowest = rooks & -rooks
            rooks ^= lowest
            file_mask = chess.BB_FILES[FILE_OF[lowest.bit_length() - 1]]
            if not pawns & file_mask:
                add(STRUCTURAL_INDEX["rook_open"], sign)
            elif not mine & file_mask:
                add(STRUCTURAL_INDEX["rook_semi"], sign)

        king = board.kings & board.occupied_co[colour]
        if king:
            square = king.bit_length() - 1
            present = (mine & SHIELD[colour][square]).bit_count()
            add(STRUCTURAL_INDEX["shield_missing"], sign * (SHIELD_WANTED - min(present, 3)))

    return net


def vector(board: chess.Board) -> tuple[list[int], list[float]]:
    """Sparse features: the indices that are non-zero, and what they weigh after tapering."""
    phase = phase_of(board)
    opening = phase / TOTAL_PHASE
    indices: list[int] = []
    values: list[float] = []
    for term, count in counts(board).items():
        indices.append(term * 2)
        values.append(count * opening)
        indices.append(term * 2 + 1)
        values.append(count * (1.0 - opening))
    return indices, values
