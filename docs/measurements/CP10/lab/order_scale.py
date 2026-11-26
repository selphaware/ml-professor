"""CP10, section C: the order reading under each form, on a column of repeated values,
as the table grows.

    python -B order_scale.py <package folder> [--seeds 30]

The reading alone, on arrays (no run of the tool): the training rows are all the rows
made. Columns, from numpy.random.default_rng([11200, case, rows, distinct, seed]):

    noise      whole numbers 1..d drawn at random, a target of noise
    weak       whole numbers 1..d on which the target's mean rises steadily, explaining
               5% or 20% of its variance (years of service against pay)
    copy       the target itself put into d bands of equal width (a column of d values)
    copy 2%    the same, with the values of 2% of the rows rotated among them first

For each: on how many seeds of 30 the share against the order is at most 0.10 (the
column would be warned about), and the range of the share.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser()
parser.add_argument("package")
parser.add_argument("--seeds", type=int, default=30)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.package).resolve().parent))
from mlprofessor import core                            # noqa: E402

FORMS = ("old", "mean", "median", "random", "pairs")


def make(case, n, d, seed):
    rng = np.random.default_rng([11200, ["noise", "weak05", "weak20", "copy", "copy2"].index(case),
                                 n, d, seed])
    if case == "noise":
        return rng.integers(1, d + 1, n).astype(float), rng.normal(size=n)
    if case in ("weak05", "weak20"):
        r2 = 0.05 if case == "weak05" else 0.20
        x = rng.integers(1, d + 1, n).astype(float)
        signal = (x - x.mean()) / x.std()
        return x, signal + rng.normal(size=n) * np.sqrt((1 - r2) / r2)
    y = np.cumsum(rng.normal(size=n)) if n <= 5000 else rng.normal(size=n)
    values = 1.07 * y + 3
    if case == "copy2":
        picks = rng.choice(n, round(0.02 * n), replace=False)
        values[picks] = np.roll(values[picks], 1)
    low, high = values.min(), values.max()
    width = (high - low) / d
    return low + (np.minimum(d - 1, ((values - low) / width).astype(int)) + 0.5) * width, y


print("| column | rows | distinct values | " + " | ".join(f"{f}: warned, share" for f in FORMS) + " |")
print("|---|---|---|" + "---|" * len(FORMS))
seconds = {f: 0.0 for f in FORMS}
for case in ("noise", "weak05", "weak20", "copy", "copy2"):
    for n in (1125, 20000, 100000):
        for d in (25, 40, 100, 400):
            cells = []
            for form in FORMS:
                shares = []
                for seed in range(args.seeds):
                    x, y = make(case, n, d, seed)
                    started = time.perf_counter()
                    shares.append(core._order_share(x, y, form, seed))
                    seconds[form] += time.perf_counter() - started
                have = [s for s in shares if s is not None]
                fired = sum(s <= 0.10 for s in have)
                cells.append(f"{fired}, {min(have):.3f} to {max(have):.3f}" if have else "no value")
            print(f"| {case} | {n} | {d} | " + " | ".join(cells) + " |", flush=True)
print("\nseconds spent in each form over all the columns above:",
      {f: round(s, 1) for f, s in seconds.items()})
