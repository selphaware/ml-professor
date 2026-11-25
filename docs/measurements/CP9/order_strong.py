"""CP9, D76 (1): does the order reading warn about a legitimately strong feature?

    python -B order_strong.py <package folder> [--seeds 30]

Tables under time order with one strong feature and no copy of the target, at 50,
100 and 200 rows and a true R-squared of 0.90, 0.95 and 0.99, from
numpy.random.default_rng([9900, rows, round(1000 * R2), seed]). Two shapes:

    independent   x is drawn afresh in every row; y = x + noise
    persistent    x is a random walk (a slow driver known before the outcome);
                  y = x + noise

with two more columns of noise, a time column `t` (0, 1, 2, ...) and
`time_col="t", group_col=False`. The noise is scaled so that x explains the stated
share of the variance of y in the table.

The order reading (D76): with the training rows put in the order of a column, the
share of neighbouring pairs (whose values of the column differ) in which the target
moves the other way is at most 0.10, and the same share for the previous value is
above 0.10. Printed for each cell, over the seeds:

    share          the range of that share for x over the training rows
    order reading  on how many seeds its two conditions hold for x
    run            what the run does with x: excluded as a leak (the first reading at
                   0.98), warned by the first reading (0.90), warned by the order
                   reading alone, or neither

A table of 50 rows is below the tool's floor of 100 usable rows, so the run stops
with `too_few_rows`; for those the share is computed on the earliest 37 rows, the
training rows a quarter holdout would leave.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
SIZES = (50, 100, 200)
STRENGTHS = (0.90, 0.95, 0.99)


def table(shape, n, r2, seed):
    rng = np.random.default_rng([9900, n, round(1000 * r2), seed, 0 if shape == "independent" else 1])
    x = rng.normal(size=n) if shape == "independent" else np.cumsum(rng.normal(size=n))
    noise = rng.normal(size=n)
    noise *= np.std(x) * np.sqrt((1 - r2) / r2) / np.std(noise)
    return pd.DataFrame({"t": np.arange(n), "x": x, "n1": rng.normal(size=n),
                         "n2": rng.normal(size=n), "y": x + noise})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--per-seed", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.package).resolve().parent))
    import mlprofessor
    from mlprofessor import core, errors, _proto as P
    cut = P.ORDER_CUT
    print(f"cut-off {cut:g}\n")
    print("| shape | rows | true R-squared | share for x (range) | order reading holds, of seeds "
          "| run: excluded / warned by skill / warned by order alone / neither |")
    print("|---|---|---|---|---|---|")
    lines = []
    for shape in ("independent", "persistent"):
        for n in SIZES:
            for r2 in STRENGTHS:
                shares, holds, outcome = [], 0, [0, 0, 0, 0]
                stopped = None
                for seed in range(args.seeds):
                    frame = table(shape, n, r2, seed)
                    train = frame.iloc[: n - (-(-n // 4))]
                    y, x = train["y"].to_numpy(), train["x"].to_numpy()
                    prev = np.r_[np.nan, y[:-1]]
                    share = core._against_order(x, y)
                    before = core._against_order(prev, y)
                    shares.append(share)
                    holds += bool(share is not None and share <= cut
                                  and (before is None or before > cut))
                    try:
                        d = mlprofessor.profile(frame, "y", time_col="t", group_col=False,
                                                profile="fast", n_jobs=1, seed=0).to_dict()
                    except errors.ProfileError as exc:
                        stopped = exc.code
                        continue
                    role = next(c for c in d["columns"] if c["name"] == "x")
                    told = [w["message"] for w in d["warnings"]
                            if w["code"] == "strong_single_column" and w["column"] == "x"]
                    if role["reason"] == "suspected_leak":
                        outcome[0] += 1
                    elif any(not m.startswith("put in the order") for m in told):
                        outcome[1] += 1
                    elif told:
                        outcome[2] += 1
                    else:
                        outcome[3] += 1
                have = [s for s in shares if s is not None]
                span = f"{min(have):.3f} to {max(have):.3f}" if have else "too few pairs"
                run = f"stops: {stopped}" if stopped else " / ".join(map(str, outcome))
                print(f"| {shape} | {n} | {r2:.2f} | {span} | {holds} of {args.seeds} | {run} |")
                lines.append((shape, n, r2, shares))
    if args.per_seed:
        print("\nper seed, the share for x (seeds 0 to 29):")
        for shape, n, r2, shares in lines:
            print(f"{shape} {n} {r2:.2f}: " + " ".join("none" if s is None else f"{s:.3f}" for s in shares))
