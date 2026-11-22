"""CP8, section B (D70, D72): how often the fast profile's cap leaves out a large
entity, by the order the entities are drawn in.

    python -B cap_order.py <package folder> [--per-seed]

<package folder> is the package `mlprofessor` to measure, imported from its
parent folder; no bytecode is written.

One entity of `big` rows and 300 of five rows (noise, classification; the table
from `numpy.random.default_rng(0)`), fast profile, `group_col`, `time_col=False`,
tool seeds 0 to 29, a constant model through the seam so that only the cap and
the split are at work. A run "leaves the entity out" when `dropped_row_cap` is at
least its size. D70 as first worded stops only an entity of more than 2,000 rows;
D72 keeps every entity of a quarter or more of the usable rows.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True


class Flat:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (len(X), 1))


FAMILY = [{"name": "flat", "grid": [{}], "make": lambda task, params, seed, n_jobs: Flat()}]


def table(big, small=300, per=5, seed=0):
    rng = np.random.default_rng(seed)
    names = ["BIG"] * big + [f"e{k:04d}" for k in range(small) for _ in range(per)]
    n = len(names)
    frame = pd.DataFrame({"who": names, "m1": rng.normal(size=n), "y": rng.integers(0, 2, size=n)})
    return frame.iloc[rng.permutation(n)].reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--per-seed", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.package).resolve().parent))
    import mlprofessor
    from mlprofessor import errors, extensions

    print(f"package: {Path(mlprofessor.__file__).resolve().parent}")
    print("| rows of the large entity | of the usable rows | left out by the cap, of 30 seeds | "
          "stopped | rows profiled when left out |")
    print("|---|---|---|---|---|")
    seen = {}
    for big in (500, 1000, 1500, 1900, 2000, 2001, 2500):
        frame = table(big)
        left, stopped, kept_rows, marks = 0, {}, set(), []
        for seed in range(30):
            try:
                with extensions.use(families=FAMILY):
                    result = mlprofessor.profile(frame, "y", time_col=False, group_col="who",
                                                 profile="fast", n_jobs=1, seed=seed)
            except errors.ProfileError as exc:
                stopped[exc.code] = stopped.get(exc.code, 0) + 1
                marks.append("stop")
                continue
            rows = {r["item"]: r["count"] for r in result.to_dict()["rows"]}
            out = rows["dropped_row_cap"] >= big
            left += out
            marks.append("out" if out else "kept")
            if out:
                kept_rows.add(rows["training"] + rows["holdout"])
        seen[big] = marks
        print(f"| {big} | {big / len(frame):.0%} | {left} | {stopped or 'none'} | "
              f"{sorted(kept_rows) or '-'} |")
    if args.per_seed:
        print("\nat each tool seed, 0 to 29: kept, out (left out by the cap) or stop")
        for big, marks in seen.items():
            print(f"{big}: " + " ".join(marks))
