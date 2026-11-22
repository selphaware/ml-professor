"""CP7, section A (the fifth review, finding 2): how often does the warning
`autocorrelated_panel_target` fire on a short panel?

    python -B short_panels.py [--seeds 30]

A panel of random walks, each at a level of its own, on the same timestamps, run
under a time column with `group_col=False`. The warning is raised when the lag-1
autocorrelation of the per-timestamp mean of the target over the training
timestamps is above 0.5 (spec step 11). It needs no model: the rule reads the
target and the split alone, so this script computes it with the reference's own
functions and fits nothing. Every such table is a persistent panel, so a rule
that never missed would warn on every one. No file is written.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "verify" / "reference"))
from mlprofessor import core  # noqa: E402

CUT = 0.5


def lag1(n_series, n_times, seed):
    """The autocorrelation the rule reads, or None when the means do not vary."""
    rng = np.random.default_rng(seed)
    keys = np.tile(np.arange(n_times, dtype="float64"), n_series)
    level = (rng.normal(0, 20, size=(n_series, 1))
             + np.cumsum(rng.normal(size=(n_series, n_times)), axis=1)).ravel()
    order = np.argsort(keys, kind="stable")
    keys, level = keys[order], level[order]
    train = core._split_time(keys, 0)["train"]
    return core._lag1_of_stamp_means(keys[train], level[train])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()
    print(f"random walks on the same timestamps, {args.seeds} seeds; the warning fires when the "
          f"autocorrelation is above {CUT}")
    print("| series | timestamps | training timestamps | warned | autocorrelation, min | median | max |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for n_series in (12, 40, 150):
        for n_times in (8, 12, 16, 24, 36, 60, 125):
            values = [lag1(n_series, n_times, seed) for seed in range(args.seeds)]
            held = [v for v in values if v is not None]
            warned = sum(v > CUT for v in held)
            keys = np.sort(np.tile(np.arange(n_times, dtype="float64"), n_series))
            n_train = len(np.unique(keys[core._split_time(keys, 0)["train"]]))
            print(f"| {n_series} | {n_times} | {n_train} | {warned} of {args.seeds} | "
                  f"{min(held):+.2f} | {float(np.median(held)):+.2f} | {max(held):+.2f} |")
            rows.append((n_series, n_times, values))
    if args.table:
        print("\nper seed: series | timestamps | the autocorrelation at seeds 0, 1, ...")
        for n_series, n_times, values in rows:
            print(f"{n_series} | {n_times} | " + " ".join(
                "none" if v is None else f"{v:+.2f}" for v in values))


if __name__ == "__main__":
    main()
