"""CP10, section A: how many shuffles? From runs made with 25 shuffles for each column.

    python -B a_shuffles.py <a_two_k25.json>

For each table kind: the spread of one shuffle's fall for `pair_b` (the smaller of the
two columns that carry the target) and, for means of 1, 3, 5 and 10 shuffles drawn
from the 25 (200 draws a table): on how many draws the fall of `pair_b` comes out
below 0.05 (it would not be named), and on how many another column's comes out at or
above 0.05 (it would be).
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab10                                            # noqa: E402

recs = [r for r in lab10.load(sys.argv[1]) if "error" not in r and r["skill"] is not None]
rng = np.random.default_rng(10)
KS, DRAWS = (1, 3, 5, 10), 200


def falls(r, name):
    scores = np.array(r["log"]["rests"][name])
    without = 2 * scores - 1 if "chance" in r["baselines"] else 1 - scores / r["baseline_score"]
    return r["skill"] - without


by = defaultdict(list)
for r in recs:
    if r["skill"] >= 0.98 and "pair_b" in r["log"].get("rests", {}):
        by[r["run"]].append(r)
print("| run | tables at or above 0.98 | holdout rows | fall of pair_b: mean | sd of one shuffle | "
      + " | ".join(f"pair_b below 0.05 with {k}" for k in KS)
      + " | " + " | ".join(f"another column at or above 0.05 with {k}" for k in KS) + " | draws |")
print("|---|---|---|---|---|" + "---|" * (2 * len(KS) + 1))
for run, group in by.items():
    means, sds = [], []
    miss = {k: 0 for k in KS}
    false = {k: 0 for k in KS}
    for r in group:
        b = falls(r, "pair_b")
        means.append(b.mean())
        sds.append(b.std(ddof=1))
        others = np.array([falls(r, n) for n in r["log"]["rests"] if n not in ("pair_a", "pair_b")])
        for k in KS:
            picks = np.array([rng.choice(25, k, replace=False) for _ in range(DRAWS)])
            miss[k] += int((b[picks].mean(axis=1) < 0.05).sum())
            if len(others):
                false[k] += int((others[:, picks].mean(axis=2) >= 0.05).any(axis=0).sum())
    print(f"| {run} | {len(group)} | {group[0]['rows']['holdout']} | {np.mean(means):.3f} | "
          f"{min(sds):.3f} to {max(sds):.3f} | " + " | ".join(str(miss[k]) for k in KS) + " | "
          + " | ".join(str(false[k]) for k in KS) + f" | {DRAWS * len(group)} |")
