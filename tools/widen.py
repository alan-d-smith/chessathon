"""Widen a trained network into a larger one that starts out computing the same thing.

Training a wider net from noise throws away everything the current one knows and asks it to
find all of it again, which is a poor use of a night when the existing net is already the best
thing we have. Duplicating the hidden layer and dividing the output weights by the number of
copies is exact: every hidden unit appears n times and contributes 1/n as much, so the sum is
unchanged and the wider net starts at precisely the quality of the narrower one.

The catch is that identical units receive identical gradients and stay identical forever, so a
256 unit net built this way would train as a 64 unit net wearing a larger coat. A little noise
on the hidden weights breaks that tie without moving the function far, and training pulls the
copies apart from there.

    uv run python -m tools.widen --net weights/net.npz --out data/nets/wide256.npz --hidden 256

The result is meant for tools/nnue.py --warm, which requires the stored width to match the width
being trained. It is not something to ship: it is the same evaluation as the net it came from,
only more expensive to compute, until it has actually been trained.
"""

import argparse
from pathlib import Path

import numpy as np


def widen(source: Path, destination: Path, hidden: int, noise: float, seed: int) -> None:
    with np.load(source) as data:
        fields = {name: data[name] for name in data.files}

    base = int(fields["hidden_bias"].shape[0])
    if hidden % base:
        raise SystemExit(f"{hidden} is not a whole number of {base} unit copies")
    copies = hidden // base
    if copies < 2:
        raise SystemExit(f"{source} is already {base} units wide")

    generator = np.random.default_rng(seed)
    weight = np.tile(fields["hidden_weight"], (1, copies))
    # Scaled to the weights themselves, so the same setting means the same thing whatever the
    # net was fitted to. Only the hidden layer is disturbed: the output layer stays exact, so
    # the whole perturbation is one small step away from the function we started with.
    spread = noise * float(fields["hidden_weight"].std())
    weight = weight + generator.normal(0.0, spread, size=weight.shape)

    fields["hidden_weight"] = np.ascontiguousarray(weight, dtype=np.float32)
    fields["hidden_bias"] = np.ascontiguousarray(
        np.tile(fields["hidden_bias"], copies), dtype=np.float32
    )
    fields["output_weight"] = np.ascontiguousarray(
        np.tile(fields["output_weight"], copies) / copies, dtype=np.float32
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **fields)
    residual = bool(fields["residual"][0]) if "residual" in fields else False
    print(
        f"{base} -> {hidden} units ({copies} copies), noise {noise:g} of a weight's spread, "
        f"residual={residual}"
    )
    print(f"written to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--net", type=Path, default=Path("weights/net.npz"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument(
        "--noise",
        type=float,
        default=0.02,
        help="symmetry breaking, as a fraction of the hidden weights' own spread",
    )
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    widen(arguments.net, arguments.out, arguments.hidden, arguments.noise, arguments.seed)


if __name__ == "__main__":
    main()
