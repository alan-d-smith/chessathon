# AI Chessathon starter

Fork this to build an agent for [AI Chessathon](https://aichessathon.com). It gives you a working
submission, baselines to beat, and a local harness that speaks the same protocol and enforces the
same clock as the platform, so you can see whether a change actually helped before you upload it.

```
git clone https://github.com/advitrocks9/aichessathon-starter
cd aichessathon-starter
make setup
make play
```

That plays your agent against a baseline over a full 120 s + 0.5 s game and prints the result.
When you like it, `make zip` and drop `submission.zip` on your dashboard.

## Writing an agent

`agent.py` is the whole submission. One function:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    return "e2e4"
```

The fork ships a legal random-mover, so the loop works before you write anything. Replace the body.

```
make play                                          # one game, real time control
make arena                                         # 20 fast games, prints a score
make play FEN="<fen>"                              # start from a given position
uv run python -m harness.play --black baselines/minimax --pgn game.pgn
uv run python -m harness.arena --opponent ../my-old-version --games 200
```

Anything your agent prints shows up under the result, so `print` debugging works. The platform
keeps it too. Every rated game leaves a log on your dashboard next to the PGN, holding your
output plus your init time, your time on each move, and the clock you had left. Only your team
can read it.

## The ladder

Measured with `harness/arena.py`. Beating greedy is a search. Beating minimax is a search plus an
evaluation worth searching with.

| Matchup | Games | Time control | Score |
|---|---|---|---|
| random vs greedy | 20 | 10 s + 0.1 s | 10.0% (+1 =2 -17) |
| greedy vs minimax | 6 | 120 s + 0.5 s | 0.0% (+0 =0 -6) |
| numba vs minimax | 6 | 10 s + 0.5 s | 66.7% (+2 =4 -0) |

- `baselines/random` plays a uniformly random legal move. It is what `agent.py` starts as.
- `baselines/greedy` searches one ply on material.
- `baselines/minimax` searches two plies on material and mobility, with no time management.
- `baselines/numba` is `minimax` with the evaluation jitted. It is barely stronger, which is
  the point: jitting a shallow search buys headroom, not depth. Read it for the warm-up call
  at the bottom, which is how you keep compilation off your clock.

## Sparring and training data

`baselines/stockfish` is a local opponent only. Shipping a third party engine, a wrapper around
one, or a network derived from one is disqualifying and the check is retroactive, so it lives
under `baselines/` where `harness/package.py` cannot reach it: the packager takes `*.py` at the
repo root plus `weights/`, and nothing else. Labelling positions with an existing engine is
explicitly allowed; the ban is on what the zip contains.

It needs a Stockfish binary. Point `SF_BIN` at one, or put it on `PATH`. Strength is set by
environment variable, so the one directory is a ladder of graded opponents:

```
SF_NODES=1000 uv run python -m harness.arena --opponent baselines/stockfish --games 20
SF_ELO=1600   uv run python -m harness.arena --opponent baselines/stockfish --games 20
SF_SKILL=20   uv run python -m harness.arena --opponent baselines/stockfish --games 20
```

`SF_NODES` is the one to measure with, because a fixed node count is reproducible and ignores
the clock; `SF_ELO` (1320-3190) and `SF_SKILL` (0-20) add deliberate randomness. `SF_NODES=1000`
is roughly `baselines/minimax` strength, which makes it the first rung.

## Measuring a change

`harness/arena.py` starts every game from the standard position. Against a baseline that picks
randomly among equal moves that is fine, but between two deterministic agents it replays one
game and reports it as a sample: twenty games, two distinct results. Every A/B comparison here
goes through `tools/match.py` instead, which plays an opening suite from both sides, in
parallel, and stops as soon as the result is statistically settled.

```
uv run python -m tools.openings --count 300 --out data/openings.txt
uv run python -m tools.match --agent . --opponent baselines/v4
uv run python -m tools.bench --ms 2000 --profile
```

The match runs an SPRT against "no better" versus "worth at least 15 Elo", so a clear change is
decided in around a hundred games and only a marginal one costs the full suite. It reports Elo
with a 95% interval, and names which side was responsible for any flag or crash, because an
opponent losing on time and our agent losing on time read identically in a score line and mean
opposite things. `data/openings.txt` is committed so comparisons stay on one yardstick.

`baselines/v1` through `v4` are frozen previous versions. "Better than my last one" is the only
comparison that matters, so each accepted change becomes the next opponent.

Building a dataset is two steps. Positions first, then labels:

```
uv run python -m tools.gen_positions --games 500 --out data/positions.txt
uv run python -m tools.label --in data/positions.txt --out data/labelled.jsonl --nodes 200000
```

`gen_positions` walks a few random plies for variety and lets the engine play on from there, so
the positions resemble ones a real game reaches rather than ones only a random mover does.
`label` writes `{fen, cp, mate, best, nodes}` per line with `cp` from the side to move, runs one
engine per core, and resumes where it left off. `data/` is gitignored.

## What's here

```
agent.py             your submission
baselines/           random, greedy, minimax, numba; each is a directory with an agent.py
baselines/stockfish  a local sparring partner, never shipped
tools/               position generation and Stockfish labelling for training data
harness/runner.py    the process the platform runs your agent in
harness/referee.py   the clock, legality, draw and adjudication rules
harness/rules.py     the event constants the harness enforces
harness/sandbox.py   the one process, spoken to as the platform speaks to a container
harness/play.py      one game between two agent directories
harness/arena.py     many games, with a score
harness/package.py   builds submission.zip with agent.py at the root
docs/IDEAS.md        where the strength actually comes from
```

Local games start from the normal position unless you pass `--fen`. Rated games start from
curated neutral positions.

The harness is here so your games are honest, not so you can pre-validate an upload. Acceptance
happens on the platform, and the validation log on your dashboard is the authority on it.

## The rules

[aichessathon.com/docs](https://aichessathon.com/docs) is canonical and changes. Read it before
you upload.
