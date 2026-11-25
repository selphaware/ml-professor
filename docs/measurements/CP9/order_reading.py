"""CP9, D74 (2): the three cases that must at least be warned about under time order,
without the order reading of D76 (the prototype with MLP_ORDER=0) and with it.

    python -B order_reading.py <package folder> [--seeds 30]

Runs `random_walk`, `int_stamps` and `panel` (time order; `panel` with its entity
column) at data seeds 1115 + 100 k with a column added that is: a rescaled copy of
the target with the values of 2% of the rows rotated among them; a rescaled copy
with noise of 5% of the target's spread; the square of the target plus 100; the
exponential of a twentieth of the target. For each: on how many seeds the column is
excluded, warned about or kept in silence

    as ruled     the repaired screen of D74 alone (the order reading switched off)
    with order   the same, and a warning when, put in the order of the column, the
                 target moves the other way between at most 10% of neighbouring
                 training rows, and does not follow the previous value as closely

and, as controls, how many clean runs of every gate table gain a warning with the
order reading (`gate` below), and a series that only rises with its own lag.
"""
import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab9                                             # noqa: E402

CASES = ["copy_2pct_cells_differ", "copy_noise_5pct", "copy_squared", "copy_exp"]
RUNS = ["random_walk", "int_stamps", "panel"]


def tally(found, case_rows):
    out = {}
    for r in found:
        if "error" in r:
            key = (r["run"], r["change"])
            out.setdefault(key, [0, 0, 0, 0])[3] += 1
            continue
        role = r["roles"].get("target_copy")
        warned = any(c == "strong_single_column" and col == "target_copy" for c, col in r["warnings"])
        entry = out.setdefault((r["run"], r["change"]), [0, 0, 0, 0])
        entry[0 if role and role[1] == "suspected_leak" else 1 if warned else 2] += 1
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    jobs = [dict(run=run, k=k, change=case) for run in RUNS for case in CASES
            for k in range(args.seeds)]
    ruled = tally(lab9.execute(args.package, jobs, {"MLP_ORDER": "0"}, workers=20), jobs)
    order = tally(lab9.execute(args.package, jobs, {"MLP_ORDER": "1"}, workers=20), jobs)
    print("| run | column added | as ruled: excluded / warned / silent | with the order "
          "reading: excluded / warned / silent |")
    print("|---|---|---|---|")
    for run in RUNS:
        for case in CASES:
            a, b = ruled[(run, case)], order[(run, case)]
            print(f"| {run} | {case} | {a[0]} / {a[1]} / {a[2]} | {b[0]} / {b[1]} / {b[2]} |")

    # controls: every clean run of the gate, with and without the order reading
    g = lab9.gate()
    clean = [dict(run=run, k=k) for run in g.RUNS if run != "api_signal_clf" and run not in lab9.ADDED
             and not run.startswith("full_") for k in range(args.seeds)]
    plain = lab9.execute(args.package, clean, {"MLP_ORDER": "0"}, workers=20)
    with_order = lab9.execute(args.package, clean, {"MLP_ORDER": "1"}, workers=20)
    gained = []
    for a, b in zip(plain, with_order):
        if "error" in a or "error" in b:
            continue
        new = sorted(set(map(tuple, b["warnings"])) - set(map(tuple, a["warnings"])))
        if new or a["roles"] != b["roles"] or a["skill"] != b["skill"]:
            gained.append((a["run"], a["k"], new))
    print(f"\ngate: {len(plain)} clean fast runs; runs that gain a warning or move with the "
          f"order reading: {len(gained)} {gained[:10]}")

    # a series that only rises (a running total), with its own lag, under time order
    os.environ["MLP_ORDER"] = "1"
    sys.path.insert(0, str(Path(args.package).resolve().parent))
    import importlib
    import mlprofessor
    from mlprofessor import _proto
    _proto.ORDER = True
    warned = 0
    for seed in range(args.seeds):
        rng = np.random.default_rng([9800, seed])
        level = np.cumsum(rng.exponential(1.0, 1500))
        frame = pd.DataFrame({"t": np.arange(1500), "level_prev": np.r_[0.0, level[:-1]],
                              "level": level})
        d = mlprofessor.profile(frame, "level", time_col="t", group_col=False, profile="fast",
                                n_jobs=1, seed=0).to_dict()
        warned += any(w["code"] == "strong_single_column" for w in d["warnings"])
    print(f"a running total with its own lag, under time order: the lag warned about on "
          f"{warned} of {args.seeds} seeds")
