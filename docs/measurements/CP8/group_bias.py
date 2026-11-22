"""CP8, section A3 (D69 point 3): skill under duplicates="group" as the number of
combinations of feature values falls, on noise and on a signal.

    python -B group_bias.py <package folder> [--seeds 30] [--per-seed]

<package folder> is the package `mlprofessor` to measure, imported from its
parent folder; no bytecode is written.

2,000 rows, fast profile, one core, tool seed 0, both statements `False`; the
tables come from `numpy.random.default_rng(100 + k)`, k = 0 to seeds - 1. Two
shapes for each number of combinations:

  crossed  two text columns that cross (4 x 2, 4 x 4, 6 x 4, 8 x 6, 12 x 10)
  nested   a region of four levels and a branch within it, as the gate's
           `few_valued` (8, 16 and 24 branches; one-hot keeps 30 levels, so no more)

First table: a label drawn at random, so the truth is skill 0. Second: the
label is the parity of the two crossed columns with one label in ten flipped,
the worst case of the sixth review's finding 3. For each: skill under "group"
and, on the same table, under "keep"; the units in the holdout; the runs with
an interval; the runs that stop (fewer than six units: `too_few_groups`).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

SAID = dict(time_col=False, group_col=False, profile="fast", n_jobs=1, seed=0)
CROSSED = {8: (4, 2), 16: (4, 4), 24: (6, 4), 48: (8, 6), 120: (12, 10)}


def crossed(levels, n, rng, parity=False):
    a = rng.integers(0, levels[0], size=n)
    b = rng.integers(0, levels[1], size=n)
    y = rng.integers(0, 2, size=n)
    if parity:                          # the review's table: a signal a model can learn
        flip = rng.random(n) < 0.1
        y = ((a + b) % 2) ^ flip
    return pd.DataFrame({"c1": [f"a{v:02d}" for v in a], "c2": [f"b{v:02d}" for v in b], "y": y})


def nested(branches, n, rng):
    b = rng.integers(0, branches, size=n)
    return pd.DataFrame({"region": np.array(["north", "south", "east", "west"])[b % 4],
                         "branch": [f"b{v + 1:03d}" for v in b],
                         "y": rng.integers(0, 2, size=n)})


def run(tool, frame, duplicates):
    mlprofessor, errors = tool
    try:
        d = mlprofessor.profile(frame, "y", duplicates=duplicates, **SAID).to_dict()
    except errors.ProfileError as exc:
        return exc.code, None, None
    return d["headline"]["skill"], d["validation"]["n_groups_holdout"], \
        d["headline"]["interval_method"]


def row(tool, label, make, seeds, per_seed):
    group, keep, units, interval, stops = [], [], [], 0, {}
    for k in range(seeds):
        frame = make(np.random.default_rng(100 + k))
        skill, held, method = run(tool, frame, "group")
        if held is None:
            stops[skill] = stops.get(skill, 0) + 1
            group.append(None), keep.append(None)
            continue
        group.append(skill), units.append(held)
        interval += method is not None
        keep.append(run(tool, frame, "keep")[0])
    g = np.array([v for v in group if v is not None])
    kp = np.array([v for v in keep if v is not None])
    print(f"| {label} | {g.min():+.3f} to {g.max():+.3f} | {g.mean():+.3f} | {g.std(ddof=1):.3f} | "
          f"{(g < -0.2).sum()} | {kp.min():+.3f} to {kp.max():+.3f} | {kp.mean():+.3f} | "
          f"{min(units)} to {max(units)} | {interval} of {len(g)} | {stops or 'none'} |")
    if per_seed:
        def shown(values):
            return " ".join("stop" if v is None else f"{v:+.3f}" for v in values)
        print(f"    group, seeds 0 to {seeds - 1}: {shown(group)}")
        print(f"    keep,  seeds 0 to {seeds - 1}: {shown(keep)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--per-seed", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.package).resolve().parent))
    import mlprofessor
    from mlprofessor import errors

    tool = (mlprofessor, errors)
    print(f"package: {Path(mlprofessor.__file__).resolve().parent}")
    head = ("| Combinations | `group`: skill, range | mean | sd | runs below -0.2 | `keep`: skill, "
            "range | mean | holdout units | runs with an interval | stopped |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    print("\nNoise: a label drawn at random, 2,000 rows (the truth is 0)\n\n" + head)
    for combos, levels in CROSSED.items():
        row(tool, f"{combos}, crossed {levels[0]} x {levels[1]}",
            lambda rng, levels=levels: crossed(levels, 2000, rng), args.seeds, args.per_seed)
    for branches in (8, 16, 24):
        row(tool, f"{branches}, a branch within a region",
            lambda rng, branches=branches: nested(branches, 2000, rng), args.seeds, args.per_seed)
    print("\nSignal: the label is the parity of the two columns, one label in ten flipped "
          "(the sixth review, finding 3)\n\n" + head)
    for combos, levels in CROSSED.items():
        row(tool, f"{combos}, crossed {levels[0]} x {levels[1]}",
            lambda rng, levels=levels: crossed(levels, 2000, rng, parity=True), args.seeds,
            args.per_seed)
