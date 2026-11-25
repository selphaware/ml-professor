"""CP9, D74 (3): the warning for a value far outside all the other values of its column.

    python -B extreme.py <package folder> [--seeds 30]

The rule is a function of one column's values (`core._far_outside`): its distinct
values in order; from each end the three outermost are looked at; a value is far
outside when the gap to the next value inward is more than K times the spread of
the ten distinct values after that one. Printed here, at K = 20, 25, 30, 40 and 50:

1. Planted cells. Every numeric feature that reaches a model and every regression
   target of every table of the gate (the roles are those of a clean run of the
   prototype at the gate seed), at data seeds 1115 + 100 k, with one cell of the
   column, in a row that has a target, set to -999 and to 999999: on how many
   seeds the value is named, and the smallest ratio of gap to spread it reaches.
2. Clean columns. The same columns with nothing planted: on how many seeds anything
   is named, and the largest ratio.
3. examples/customers.csv.
4. Long tails. Columns drawn at 200, 2,000, 20,000 and 100,000 rows from
   numpy.random.default_rng([9700, size, seed]): log-normal with sigma 1, 1.5 and 2;
   a traded volume (log-normal(13, 1.5), rounded); Pareto with shape 2.5 and 1.5;
   Student t with 3 degrees of freedom; exponential; a count; and normal.
"""
import argparse
import io
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab9                                             # noqa: E402

SIZES = (200, 2000, 20000, 100000)
CUTS = (20.0, 25.0, 30.0, 40.0, 50.0)
PLANTED = (-999.0, 999999.0)


def tails(n, rng):
    yield "normal", rng.normal(size=n)
    yield "log-normal, sigma 1", rng.lognormal(0, 1.0, n)
    yield "log-normal, sigma 1.5", rng.lognormal(0, 1.5, n)
    yield "log-normal, sigma 2", rng.lognormal(0, 2.0, n)
    yield "traded volume (log-normal(13, 1.5), rounded)", np.round(rng.lognormal(13.0, 1.5, n))
    yield "Pareto, shape 2.5", rng.pareto(2.5, n) + 1
    yield "Pareto, shape 1.5", rng.pareto(1.5, n) + 1
    yield "Student t, 3 degrees of freedom", rng.standard_t(3, n)
    yield "exponential", rng.exponential(1.0, n)
    yield "a count (Poisson, mean 3)", rng.poisson(3, n).astype("float64")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.package).resolve().parent))
    import mlprofessor
    from mlprofessor import core, _proto as P
    g = lab9.gate()

    def named_at(values, value=None):
        """For each cut-off: whether `value` is named (any value, when None)."""
        out = []
        for cut in CUTS:
            P.EXTREME_K = cut
            found = core._far_outside(values)
            out.append(bool(found) if value is None else any(v == value for v, _, _ in found))
        return out

    print(f"the {P.EXTREME_J} outermost distinct values at each end; the spread of the "
          f"{P.EXTREME_M} values after; cut-offs K = " + ", ".join(f"{c:g}" for c in CUTS))

    # the numeric features and regression targets of each table, from a clean run
    columns = {}
    for run, (dataset, extra, profile, _) in g.RUNS.items():
        if dataset in columns or extra == "api" or run in lab9.ADDED:
            continue
        text, target, call, _ = lab9.table_of(dict(run=run, k=0))
        try:
            d = mlprofessor.profile(pd.read_csv(io.StringIO(text), low_memory=False), target,
                                    n_jobs=1, seed=0, **dict(call, profile="fast")).to_dict()
        except Exception as exc:                # a run that must stop: no roles to read
            columns[dataset] = (run, target, None, getattr(exc, "code", type(exc).__name__))
            continue
        numeric = [c["name"] for c in d["columns"]
                   if c["kind"] == "numeric" and c["role"] == "feature"]
        columns[dataset] = (run, target, numeric + ([target] if d["task"]["type"] == "regression"
                                                    else []), None)

    print("\n## 1. One cell planted\n")
    print("| table | numeric features and regression target | cell | named at K = 20 / 25 / 30 / 40 / 50, "
          "of seeds x columns | smallest ratio |")
    print("|---|---|---|---|---|")
    clean_rows, partial, weakest = [], [], None
    for dataset, (run, target, names, why) in columns.items():
        if names is None:
            print(f"| {dataset} | (the run stops: {why}) | | | |")
            continue
        if not names:
            print(f"| {dataset} | none | | | |")
            continue
        hits = {v: [0] * len(CUTS) for v in PLANTED}
        lows = {v: None for v in PLANTED}
        per_column = {}
        clean_fired, clean_top, clean_n = [0] * len(CUTS), 0.0, 0
        n = 0
        for k in range(args.seeds):
            text, _, _, _ = lab9.table_of(dict(run=run, k=k))
            frame = pd.read_csv(io.StringIO(text), low_memory=False)
            usable = frame[target].notna().to_numpy()
            rng = random.Random(f"extreme.{dataset}.{k}")
            for name in names:
                values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype="float64")
                values = values[usable]
                present = np.flatnonzero(~np.isnan(values))
                clean_n += 1
                for i, yes in enumerate(named_at(values)):
                    clean_fired[i] += yes
                clean_top = max(clean_top, core._outermost_ratio(values) or 0.0)
                n += 1
                for v in PLANTED:
                    planted = values.copy()
                    planted[rng.choice(list(present))] = v
                    ratio = core._outermost_ratio(planted)
                    entry = per_column.setdefault((name, v), [[0] * len(CUTS), ratio])
                    entry[1] = min(entry[1], ratio)
                    for i, yes in enumerate(named_at(planted, v)):
                        hits[v][i] += yes
                        entry[0][i] += yes
                    lows[v] = ratio if lows[v] is None else min(lows[v], ratio)
        shown = ", ".join(names) if len(names) <= 8 else f"{names[0]} ... {names[-1]} ({len(names)})"
        for v in PLANTED:
            print(f"| {dataset} | {shown} | {v:g} | " + " / ".join(map(str, hits[v]))
                  + f" of {n} | {lows[v]:.3g} |")
        for (name, v), (count, low) in per_column.items():
            if min(count) < args.seeds:
                partial.append((dataset, name, v, count, low))
            elif weakest is None or low < weakest[0]:
                weakest = (low, dataset, name, v)
        clean_rows.append((dataset, clean_fired, clean_n, clean_top))
    print("\nEvery column and cell not named on all seeds at some cut-off (named of "
          f"{args.seeds} seeds at K = 20 / 25 / 30 / 40 / 50; smallest ratio):")
    for dataset, name, v, count, low in partial:
        print(f"  {dataset}.{name} with {v:g}: " + " / ".join(map(str, count)) + f"; {low:.3g}")
    print(f"\nThe weakest planted cell that is named on every seed at every cut-off: "
          f"{weakest[1]}.{weakest[2]} with {weakest[3]:g}, smallest ratio {weakest[0]:.3g}")
    print("\n## 2. Nothing planted\n")
    print("| table | columns x seeds where a value is named at K = 20 / 25 / 30 / 40 / 50 | largest ratio |")
    print("|---|---|---|")
    for dataset, fired, n, top in clean_rows:
        print(f"| {dataset} | " + " / ".join(map(str, fired)) + f" of {n} | {top:.3g} |")

    print("\n## 3. examples/customers.csv\n")
    frame = pd.read_csv(lab9.REPO / "examples" / "customers.csv", low_memory=False)
    for name in frame.columns:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype="float64")
        if np.isnan(values).mean() < 0.5:
            print(f"{name}: named at K = 20 / 25 / 30 / 40 / 50: {named_at(values)}; largest ratio "
                  f"{core._outermost_ratio(values):.3g}")

    print("\n## 4. Long tails\n")
    print("| column | rows | a value named at K = 20 / 25 / 30 / 40 / 50, of seeds | largest ratio |")
    print("|---|---|---|---|")
    for size in SIZES:
        tally = {}
        for seed in range(args.seeds):
            rng = np.random.default_rng([9700, size, seed])
            for label, values in tails(size, rng):
                values = np.asarray(values, dtype="float64")
                entry = tally.setdefault(label, [[0] * len(CUTS), 0.0])
                for i, yes in enumerate(named_at(values)):
                    entry[0][i] += yes
                entry[1] = max(entry[1], core._outermost_ratio(values) or 0.0)
        for label, (fired, top) in tally.items():
            print(f"| {label} | {size:,} | " + " / ".join(map(str, fired))
                  + f" of {args.seeds} | {top:.3g} |")
