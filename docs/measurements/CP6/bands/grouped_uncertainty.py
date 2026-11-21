"""CP6: what a row bootstrap and a row permutation say under the grouped scheme.

    python -B grouped_uncertainty.py <gate script> <reference parent> <data seeds>

On `entities` nothing predicts the label of an entity the model has not seen,
so the true skill is zero. For each data seed the reference profiles the table
with its entity column declared (the grouped scheme), and from the holdout
predictions this script works out, itself:

  - a 95% interval from 1,000 resamples of rows, and one from 1,000 resamples of
    whole entities, and whether each holds zero;
  - a permutation p-value from 1,000 shuffles of the holdout labels row by row,
    and one from 1,000 shuffles of the entities' labels entity by entity (each
    entity here holds one label, so its rows move together).

A sound interval holds the true value in about 95% of runs; a sound p-value is
below 0.05 in about 5%. One line is printed for each data seed, then the
totals. Nothing is written inside the repo.
"""
import importlib.util
import sys
import tempfile
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(r"E:\CODE\ml-professor")


def auc(y, score):
    import numpy as np
    order = np.argsort(score, kind="mergesort")
    ordered = score[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    counts = np.diff(np.r_[starts, len(score)])
    ranks = np.empty(len(score))
    ranks[order] = np.repeat(starts + 1 + (counts - 1) / 2, counts)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main(gate_path, parent, n_seeds):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(Path(parent).resolve()))
    sys.path.insert(0, str(REPO / "verify"))
    import mlprofessor
    spec = importlib.util.spec_from_file_location("mlprof_gate", gate_path)
    gate = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, [gate_path]
    try:
        spec.loader.exec_module(gate)
    finally:
        sys.argv = saved
    folder = Path(tempfile.mkdtemp(prefix="cp6-grouped-"))
    warnings.simplefilter("ignore")
    rows_hold, groups_hold, tool_hold, p_rows, p_groups, skills = [], [], [], [], [], []
    print("| k | base seed | holdout rows | holdout entities | skill | interval, rows | "
          "interval, entities | the tool's interval | p, rows | p, entities |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in range(n_seeds):
        base = gate.SEED + 100 * k
        path = folder / "entities.csv"
        path.write_text(gate.dataset_text("entities", base), encoding="utf-8", newline="")
        source = pd.read_csv(path)
        result = mlprofessor.profile(str(path), "outcome", group_col="customer", profile="fast",
                                     seed=0, n_jobs=1)
        pred = result.predictions
        y = (pred["y_true"] == "yes").to_numpy().astype(int)
        score = pred["y_pred"].to_numpy(dtype=float)
        entity = source["customer"].to_numpy()[pred["row"].to_numpy()]
        _, inverse = np.unique(entity, return_inverse=True)
        members = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
        observed = 2 * auc(y, score) - 1
        skills.append(observed)
        rng = np.random.default_rng([base, 1])
        by_rows, by_groups = [], []
        for _ in range(1000):
            draw = rng.integers(0, len(y), len(y))
            a = auc(y[draw], score[draw])
            if a is not None:
                by_rows.append(2 * a - 1)
            draw = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
            a = auc(y[draw], score[draw])
            if a is not None:
                by_groups.append(2 * a - 1)
        row_lo, row_hi = np.percentile(by_rows, [2.5, 97.5])
        rows_hold.append(row_lo <= 0 <= row_hi)
        lo, hi = np.percentile(by_groups, [2.5, 97.5])
        groups_hold.append(lo <= 0 <= hi)
        h = result.headline
        tool_hold.append(h["skill_low"] <= 0 <= h["skill_high"])
        label_of = np.array([y[m[0]] for m in members])
        hits_rows = hits_groups = 0
        for _ in range(1000):
            hits_rows += 2 * auc(rng.permutation(y), score) - 1 >= observed
            shuffled = rng.permutation(label_of)[inverse]
            hits_groups += 2 * auc(shuffled, score) - 1 >= observed
        p_rows.append((1 + hits_rows) / 1001)
        p_groups.append((1 + hits_groups) / 1001)
        print(f"| {k} | {base} | {len(y)} | {len(members)} | {observed:+.4f} | "
              f"{row_lo:+.3f} to {row_hi:+.3f} | {lo:+.3f} to {hi:+.3f} | "
              f"{h['skill_low']:+.3f} to {h['skill_high']:+.3f} | {p_rows[-1]:.4f} | "
              f"{p_groups[-1]:.4f} |", flush=True)
    print()
    n = len(skills)
    print(f"{n} runs on `entities` under the grouped scheme; skill {min(skills):+.3f} to "
          f"{max(skills):+.3f}; the true skill is 0")
    print(f"  interval from resampled rows holds 0:          {sum(rows_hold)} of {n}")
    print(f"  interval from resampled entities holds 0:      {sum(groups_hold)} of {n}")
    print(f"  the tool's own interval (bootstrap_groups) holds 0: {sum(tool_hold)} of {n}")
    print(f"  p-value from shuffled rows below 0.05:         {sum(p < 0.05 for p in p_rows)} of {n}"
          f"   (smallest {min(p_rows):.4f})")
    print(f"  p-value from shuffled entities below 0.05:     {sum(p < 0.05 for p in p_groups)} of {n}"
          f"   (smallest {min(p_groups):.4f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
