"""CP9 A5: the id-like rule for whole-number columns, old and proposed, on D19's
controls and on the forms that defeat the old rule.

    python -B a5.py <package folder> [--seeds 30]

Each column is built at 200, 2,000, 20,000 and 100,000 rows from
numpy.random.default_rng([9500, size, seed]) and handed to the reference's own
rule, `core._whole_number_id`, as its present values in file order. Printed for
each: on how many of the seeds the column is called id-like, by

    old        D19: every value distinct, and consecutive or strictly ordered in file order
    dense50    more than 95% of the values distinct, filling at least 50% of their range
    dense25    the same with 25%
    dense10    the same with 10%
    d50+ord    dense50, or every value distinct and strictly ordered in file order

D19's own scripts and seeds were not kept (D35), so the controls are rebuilt here.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
SIZES = (200, 2000, 20000, 100000)
RULES = [("old", "old", 0.5), ("dense50", "dense", 0.5), ("dense25", "dense", 0.25),
         ("dense10", "dense", 0.10), ("d50+ord", "dense_or_ordered", 0.5)]


def measures(n, rng):
    """Whole-number measures: none is an identifier."""
    yield "salary, rounded to the pound", np.round(rng.lognormal(10.5, 0.4, n))
    yield "price in pence", np.round(100 * rng.lognormal(3.0, 0.8, n))
    yield "traded volume", np.round(rng.lognormal(13.0, 1.5, n))
    yield "age in years", rng.integers(18, 91, n).astype("float64")
    yield "a count (Poisson, mean 40)", rng.poisson(40, n).astype("float64")
    yield "a score from 0 to n", rng.integers(0, n + 1, n).astype("float64")
    yield "a distinct measure, 10% of its range", rng.choice(10 * n, n, replace=False).astype("float64") + 1000
    yield "a distinct measure, 33% of its range", rng.choice(3 * n, n, replace=False).astype("float64") + 1000


def identifiers(n, rng):
    """Whole-number identifiers, in the forms a table arrives in."""
    ids = np.arange(100001, 100001 + n, dtype="float64")
    yield "consecutive, in file order", ids
    blank = np.sort(rng.choice(n, max(1, n // 100), replace=False))
    yield "consecutive, in order, 1% blank", np.delete(ids, blank)
    shuffled = rng.permutation(ids)
    yield "consecutive, shuffled", shuffled
    yield "consecutive, shuffled, 1% blank", np.delete(shuffled, blank)
    kept = np.sort(rng.choice(n, n - max(1, round(0.03 * n)), replace=False))
    yield "3% of the rows deleted, in order", ids[kept]
    yield "3% of the rows deleted, shuffled", rng.permutation(ids[kept])
    for share, label in ((0, "one row"), (0.01, "1% of the rows"), (0.05, "5% of the rows"),
                         (0.10, "10% of the rows")):
        extra = rng.choice(ids, max(1, round(share * n)), replace=False)
        both = np.concatenate([ids, extra])
        yield f"consecutive, {label} repeated, in order", np.sort(both)
        yield f"consecutive, {label} repeated, shuffled", rng.permutation(both)
    yield "every row present twice, shuffled", rng.permutation(np.concatenate([ids, ids]))
    for share in (0.30, 0.60):
        kept = np.sort(rng.choice(n, n - round(share * n), replace=False))
        yield f"{round(100 * share)}% of the records gone, in order", ids[kept]
        yield f"{round(100 * share)}% of the records gone, shuffled", rng.permutation(ids[kept])
    for label, steps in (("gaps of 1 to 3", rng.integers(1, 4, n)), ("gaps of 1 to 10", rng.integers(1, 11, n)),
                         ("steps of 7", np.full(n, 7)), ("steps of 1,000", np.full(n, 1000))):
        sparse = 100001 + np.cumsum(steps).astype("float64")
        yield f"increasing, {label}, in order", sparse
        yield f"increasing, {label}, shuffled", rng.permutation(sparse)
    yield "random, unordered (nine digits)", rng.choice(10 ** 9, n, replace=False).astype("float64")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.package).resolve().parent))
    from mlprofessor import core, _proto as P

    def called(x):
        out = []
        for _, rule, dense in RULES:
            P.A5, P.DENSE = rule, dense
            out.append(bool(core._whole_number_id(x)))
        return out

    for title, maker, sorted_too in (("Measures: none is an identifier", measures, True),
                                     ("Identifiers", identifiers, False)):
        print(f"\n## {title}\n")
        print("| column | rows | " + " | ".join(name for name, _, _ in RULES) + " |")
        print("|---|---|" + "---|" * len(RULES))
        for size in SIZES:
            counts = {}
            for seed in range(args.seeds):
                rng = np.random.default_rng([9500, size, seed])
                for label, x in maker(size, rng):
                    forms = [(label, x)]
                    if sorted_too:
                        forms.append((label + ", the table sorted by it", np.sort(x)))
                    for name, values in forms:
                        tally = counts.setdefault(name, [0] * len(RULES))
                        for i, yes in enumerate(called(np.asarray(values, dtype="float64"))):
                            tally[i] += yes
            for name, tally in counts.items():
                print(f"| {name} | {size:,} | " + " | ".join(f"{c} of {args.seeds}" for c in tally) + " |")
