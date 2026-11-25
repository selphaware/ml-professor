"""CP9 A3(a): the largest row's share of the squared error on the holdout, on tables
with no extreme cell: how large does it get by chance, at each size and for each
shape of error?

    python -B a3_share.py <package folder> [--seeds 30]

A regression table of n rows: y = 2 x1 - x2 + e, three more columns of noise; e is
normal, Student t with 3 degrees of freedom (heavy tails), or the target is
exp(1.2 z) (log-normal: a few rows are far larger than the rest). Random scheme,
fast profile, tool seed 0, data from numpy.random.default_rng([9300, n, seed]).
The holdout is a quarter of the rows.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
SIZES = (100, 200, 400, 1000, 4000)
SHAPES = ("normal", "t3", "lognormal")


def _init(package):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(Path(package).resolve().parent))
    import mlprofessor      # noqa: F401


def _job(job):
    import numpy as np
    import pandas as pd
    import mlprofessor
    n, shape, seed = job
    rng = np.random.default_rng([9300, n, seed])
    x = rng.normal(size=(n, 5))
    if shape == "normal":
        y = 2 * x[:, 0] - x[:, 1] + rng.normal(size=n)
    elif shape == "t3":
        y = 2 * x[:, 0] - x[:, 1] + rng.standard_t(3, size=n)
    else:
        y = np.exp(1.2 * (0.6 * x[:, 0] + 0.8 * rng.normal(size=n)))
    frame = pd.DataFrame(x, columns=[f"x{i + 1}" for i in range(5)])
    frame["y"] = y
    h = mlprofessor.profile(frame, "y", time_col=False, group_col=False, profile="fast",
                            n_jobs=1, seed=0).to_dict()["headline"]
    return n, shape, seed, h["largest_row_share"], h["baseline_largest_row_share"], h["skill"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--per-seed", action="store_true")
    args = parser.parse_args()
    jobs = [(n, shape, seed) for n in SIZES for shape in SHAPES for seed in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=12, initializer=_init, initargs=(args.package,)) as pool:
        found = list(pool.map(_job, jobs))
    print("| rows | holdout rows | errors | model: largest row's share | baseline: largest row's "
          "share | either above 0.5 | either above 0.25 |")
    print("|---|---|---|---|---|---|---|")
    for n in SIZES:
        for shape in SHAPES:
            rows = [f for f in found if f[0] == n and f[1] == shape]
            m, b = [f[3] for f in rows], [f[4] for f in rows]
            print(f"| {n:,} | {-(-n // 4)} | {shape} | {min(m):.3f} to {max(m):.3f} "
                  f"(mean {sum(m) / len(m):.3f}) | {min(b):.3f} to {max(b):.3f} "
                  f"(mean {sum(b) / len(b):.3f}) | {sum(max(x, y) > 0.5 for x, y in zip(m, b))} of "
                  f"{len(rows)} | {sum(max(x, y) > 0.25 for x, y in zip(m, b))} of {len(rows)} |")
    if args.per_seed:
        for n in SIZES:
            for shape in SHAPES:
                rows = sorted(f for f in found if f[0] == n and f[1] == shape)
                print(f"{n} {shape}: " + " ".join(f"{f[3]:.3f}/{f[4]:.3f}" for f in rows))
