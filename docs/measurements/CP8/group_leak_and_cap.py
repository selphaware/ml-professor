"""CP8, sections A2 and B: what one package does with `duplicates="group"` beside a
column the leak screen removes (D69), and with an entity near the fast profile's
cap (D70, D72).

    python -B group_leak_and_cap.py <package folder>

<package folder> is the package `mlprofessor` to measure, imported from its
parent folder; no bytecode is written. Fast profile, one core, tool seed 0. The
tables come from `numpy.random.default_rng(0)` and `(1)`. Each line is one run:
the named error with the start of its message, or the scheme, the skill and the
row counts of a finished run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent))
import mlprofessor                                    # noqa: E402
from mlprofessor import errors                        # noqa: E402

print(f"package: {Path(mlprofessor.__file__).resolve().parent}")
FAST = dict(profile="fast", n_jobs=1, seed=0)
SAID = dict(time_col=False, group_col=False)


def outcome(frame, target="y", **kw):
    try:
        result = mlprofessor.profile(frame, target, **{**FAST, **kw})
    except errors.ProfileError as exc:
        return f"{exc.code}: {str(exc)[:230]}"
    d = result.to_dict()
    rows = {r["item"]: r["count"] for r in d["rows"]}
    return (f"ran: scheme {d['validation']['scheme']}, skill {d['headline']['skill']:+.3f}, "
            f"training {rows['training']}, holdout {rows['holdout']}, left out by the cap "
            f"{rows['dropped_row_cap']}, copies {d['validation']['n_holdout_copies']}, warnings "
            f"{sorted({w['code'] for w in d['warnings']})}")


def few_valued(n=2000, seed=0, status=None):
    """A region of four levels and a branch of sixteen, and a label drawn at random.
    `status`: a column that copies the label, two-valued or distinct in every row."""
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 16, size=n)
    y = rng.integers(0, 2, size=n)
    frame = pd.DataFrame({"region": np.array(["north", "south", "east", "west"])[b % 4],
                          "branch": [f"b{v + 1:02d}" for v in b]})
    if status == "two-valued":
        frame["status"] = np.where(y == 1, "closed", "open")
    elif status == "distinct in every row":
        frame["status"] = y + 0.001 * np.arange(n) / n
    frame["y"] = y
    return frame


print("\n-- D69: duplicates='group' and a column the leak screen removes (noise: the truth is 0)")
for status in (None, "two-valued", "distinct in every row"):
    for dup in ("group", "keep", "error"):
        print(f"copy of the label: {status}; duplicates={dup}: "
              + outcome(few_valued(status=status), duplicates=dup, **SAID))

print("\n-- D70, D72: one entity and 300 of five rows, the fast profile")
rng = np.random.default_rng(1)


def entities(big, small=300, per=5, classification=True):
    names = ["BIG"] * big + [f"e{k:04d}" for k in range(small) for _ in range(per)]
    n = len(names)
    frame = pd.DataFrame({"who": names, "m1": rng.normal(size=n), "m2": rng.normal(size=n)})
    frame["y"] = rng.integers(0, 2, size=n) if classification else rng.normal(size=n)
    return frame.iloc[rng.permutation(n)].reset_index(drop=True)


for big in (1200, 1500, 2000, 2001, 2500, 6000):
    frame = entities(big)
    print(f"an entity of {big} rows among {len(frame)}: "
          + outcome(frame, time_col=False, group_col="who"))
print("the 2,500, full profile: "
      + outcome(entities(2500), time_col=False, group_col="who", profile="full")[:200])
print("2,500 and three small entities (the full profile cannot split either): "
      + outcome(entities(2500, small=3), time_col=False, group_col="who"))
two = pd.concat([entities(1100, small=0).assign(who="A"), entities(1100, small=0).assign(who="B"),
                 entities(0, small=360)], ignore_index=True)
print(f"two entities of 1,100 rows among {len(two)} (each a quarter or more; 2,200 together): "
      + outcome(two, time_col=False, group_col="who"))
print("a few-valued table of 40,000 rows (sets of 2,500), duplicates='group', no entity column: "
      + outcome(few_valued(n=40000), duplicates="group", **SAID))
print("an entity of 2,500 rows under a time column (the rule does not apply): "
      + outcome(entities(2500).assign(t=np.arange(4000)), time_col="t", group_col="who"))
