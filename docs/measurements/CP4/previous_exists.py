"""CP4: is there ever a validation or holdout row with no previous value?

    python -B previous_exists.py app|ref

The second review's finding 11 names a quiet fallback: the leak screen drops
the previous-value baseline when a previous value is missing. The spec says
none ever is (step 8): under time order with unique timestamps, the first
validation block starts at least `gap + 1` timestamps after the first training
timestamp, or the split has already stopped with `gap_too_large`.

This script looks for a counter-example through the documented step functions
alone. For every number of unique timestamps from 100 to 400 and every gap
from 0 up to the first one the split refuses, it takes
`stages.split_time(keys, gap)` and `stages.previous_value(y, keys, gap)`, with
the keys in shuffled order, and counts the rows of every validation block and
of the holdout whose previous value is NaN.

`app` is the application at the repo root, `ref` the reference under
verify/reference. Nothing is written.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if len(sys.argv) != 2 or sys.argv[1] not in ("app", "ref"):
    raise SystemExit(__doc__)
sys.path.insert(0, str(REPO if sys.argv[1] == "app" else REPO / "verify" / "reference"))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from mlprofessor import errors, stages  # noqa: E402

rng = np.random.default_rng(2026)
splits = rows_checked = without = refused = 0
largest_gap = {}
for n in range(100, 401):
    keys = rng.permutation(n).astype(float)
    y = rng.normal(size=n)
    gap = 0
    while True:
        try:
            split = stages.split_time(keys, gap)
        except errors.GapError:
            refused += 1
            break
        previous = stages.previous_value(y, keys, gap)
        asked = [fold["validation"] for fold in split["folds"]] + [split["holdout"]]
        for rows in asked:
            rows_checked += len(rows)
            without += int(np.isnan(previous[rows]).sum())
        # the first training timestamps, which no fold validates on, are the only rows without
        first = np.argsort(keys)[:gap + 1]
        assert np.isnan(previous[first]).all()
        splits += 1
        largest_gap[n] = gap
        gap += 1

print(f"{sys.argv[1]}: {splits} splits (timestamps 100 to 400, every gap the split accepts; "
      f"{refused} refused with gap_too_large)")
print(f"largest accepted gap: {min(largest_gap.values())} at the fewest timestamps, "
      f"{max(largest_gap.values())} at the most")
print(f"{rows_checked} validation and holdout rows checked; {without} without a previous value")
sys.exit(1 if without else 0)
