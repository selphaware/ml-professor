"""CP8, section A1: what a sentinel in a column of dates costs, and what the rule
of D72 gives back.

    python -B sentinel_skill.py <package folder> [--seeds 30] [--per-seed]

<package folder> is the package `mlprofessor` to measure: the application
(E:\\CODE\\ml-professor\\mlprofessor) or a copy of the reference. It is imported
from its parent folder; no bytecode is written.

The gate's `day_numbers` (the label is random per block of five days; the day is
a whole number YYYYMMDD) at the base seeds 1115, 1215, ..., with `time_col=False,
group_col=False`, fast profile, one core, tool seed 0: clean, and with values
that are no dates planted from the stream `numpy.random.default_rng(8000 + k)`.
For each: on how many seeds `day` is excluded as a date and named by the
warning, and the range of skill. The truth is skill about zero.
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab                                              # noqa: E402


def variants(frame, rng):
    n = len(frame)
    picks = rng.permutation(n)

    def planted(share=None, values=(99991231,), cells=1):
        day = frame["day"].to_numpy().copy()
        if share is None:
            for j, value in enumerate(values):
                day[picks[j * cells:(j + 1) * cells]] = value
        else:
            chosen = picks[:round(share * n)]
            for j, value in enumerate(values):
                day[chosen[j::len(values)]] = value
        return frame.assign(day=day)

    yield "clean", frame
    yield "one cell of 99991231", planted()
    yield "one cell of 0", planted(values=(0,))
    yield "one cell of 20240230", planted(values=(20240230,))
    yield "5% of the rows 99991231", planted(share=0.05)
    yield "30% of the rows 99991231", planted(share=0.30)
    yield "60% of the rows 99991231", planted(share=0.60)
    yield "two sentinels, 0 and 99991231, one cell each", planted(values=(0, 99991231))
    yield "three sentinels, one cell each", planted(values=(0, 99991231, 99999999))
    yield "70 distinct values that are no dates, one cell each (a stated limit)", planted(
        values=tuple(99991231 - k for k in range(70)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--per-seed", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.package).resolve().parent))
    import mlprofessor
    from mlprofessor import errors

    gate = lab.gate_module()
    print(f"package: {Path(mlprofessor.__file__).resolve().parent}")
    rows = {}
    for k in range(args.seeds):
        text = gate.dataset_text("day_numbers", gate.SEED + 100 * k)
        frame = pd.read_csv(io.StringIO(text), low_memory=False)
        for name, table in variants(frame, np.random.default_rng(8000 + k)):
            try:
                d = mlprofessor.profile(table, "regime", time_col=False, group_col=False,
                                        profile="fast", n_jobs=1, seed=0).to_dict()
            except errors.ProfileError as exc:
                rows.setdefault(name, []).append((exc.code, None, None))
                continue
            day = next(c for c in d["columns"] if c["name"] == "day")
            warned = any(w["code"] == "date_column_undeclared" and w["column"] == "day"
                         for w in d["warnings"])
            rows.setdefault(name, []).append((day["reason"], d["headline"]["skill"], warned))
    print("| `day_numbers` with | `day` excluded as a date, of the seeds | named by the "
          "warning | skill: range (mean) |")
    print("|---|---|---|---|")
    for name, found in rows.items():
        skill = np.array([s for _, s, _ in found if s is not None])
        print(f"| {name} | {sum(r == 'datetime' for r, _, _ in found)} of {len(found)} | "
              f"{sum(bool(w) for _, _, w in found)} | {skill.min():+.3f} to {skill.max():+.3f} "
              f"({skill.mean():+.3f}) |")
    if args.per_seed:
        print("\nskill at each data seed (1115, 1215, ...), in that order; `d` marks a seed on "
              "which `day` was excluded as a date")
        for name, found in rows.items():
            print(f"{name}: " + " ".join(
                (f"{s:+.3f}" if s is not None else str(r)) + ("d" if r == "datetime" else "")
                for r, s, _ in found))
