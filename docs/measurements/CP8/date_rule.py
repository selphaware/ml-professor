"""CP8, section A1: the forms of the tolerant date rule that were measured (rule 2a
of step 5; D69, D72).

    python -B date_rule.py [--seeds 30] [--per-seed]

A column's non-missing values are read as step 5 reads them: a whole number, or
the stripped text of a cell. Each DISTINCT value either reads as a calendar date
written YYYYMMDD, years 1900 to 2100, or does not. With

    d = the distinct values that read as dates
    x = the distinct values that do not
    r = the share of the non-missing cells that hold a date

the forms are

    A  every value (D66, as built)        x == 0 and d >= 2
    B  95% of the cells                   r >= 0.95 and d >= 2
    C  at most two values that are not    x <= 2 and d >= 2
    D  the same, and five dates           d >= 2 and (x == 0 or (x <= 2 and d >= 5))
    E  nine distinct values in ten        d >= 2 and d >= 9 * x
    F  both: E and at most two            d >= 2 and d >= 9 * x and x <= 2
    U  the rule adopted (D72): E or D     d >= 2 and (d >= 9 * x or (x <= 2 and d >= 5))

Nothing here is part of the tool, the reference or the gate. No file is written.
The gate's datasets come from the gate's own generator, at the base seeds 1115,
1215, ... (`--seeds` of them); every other table is made here from the seed shown.
"""
import argparse
import calendar
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
YEARS = (1900, 2100)
DATE_DATASETS = ("day_numbers", "day_sentinel")     # the gate's two tables made for the rule
RULES = {
    "A every value (D66)": lambda d, x, r: x == 0 and d >= 2,
    "B 95% of cells": lambda d, x, r: r >= 0.95 and d >= 2,
    "C x<=2": lambda d, x, r: x <= 2 and d >= 2,
    "D x<=2, d>=5": lambda d, x, r: d >= 2 and (x == 0 or (x <= 2 and d >= 5)),
    "E d>=9x": lambda d, x, r: d >= 2 and d >= 9 * x,
    "F d>=9x, x<=2": lambda d, x, r: d >= 2 and d >= 9 * x and x <= 2,
    "U adopted: E or D": lambda d, x, r: d >= 2 and (d >= 9 * x or (x <= 2 and d >= 5)),
}


def gate_module():
    sys.path.insert(0, str(REPO / "verify"))
    path = REPO / "verify" / "smoke_test.py"
    spec = importlib.util.spec_from_file_location("mlprof_gate", str(path))
    module = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [str(path)]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


def is_date_number(value):
    year, month, day = value // 10000, (value // 100) % 100, value % 100
    if not (YEARS[0] <= year <= YEARS[1] and 1 <= month <= 12 and day >= 1):
        return False
    return day <= calendar.monthrange(year, month)[1]


def cells(series):
    """(tokens, kind): the non-missing cells as step 5 reads them, or (None, None)
    when rule 2a does not look at the column (a boolean, a datetime, mixed objects)."""
    values = series.dropna()
    if len(values) == 0 or pd.api.types.is_bool_dtype(values.dtype) \
            or pd.api.types.is_datetime64_any_dtype(values.dtype):
        return None, None
    if pd.api.types.is_integer_dtype(values.dtype):
        return [int(v) for v in values.tolist()], "number"
    if pd.api.types.is_float_dtype(values.dtype):
        return [float(v) for v in values.tolist()], "number"
    if pd.api.types.is_string_dtype(values.dtype) or values.dtype == object:
        out = values.tolist()
        if all(isinstance(v, str) for v in out):
            return [v.strip() for v in out], "text"
    return None, None


def reads_as_date(token, kind):
    if kind == "text":
        return len(token) == 8 and token.isascii() and token.isdigit() \
            and is_date_number(int(token))
    if isinstance(token, float):
        if not np.isfinite(token) or token != np.floor(token):
            return False
        token = int(token)
    return is_date_number(token)


def counts(series):
    """(d, x, r) of a column, or None when rule 2a does not look at it."""
    tokens, kind = cells(series)
    if tokens is None:
        return None
    verdict = {t: reads_as_date(t, kind) for t in set(tokens)}
    d = sum(verdict.values())
    x = len(verdict) - d
    r = sum(verdict[t] for t in tokens) / len(tokens)
    return d, x, r


def flagged(series):
    """rule name -> whether it calls the column a date column; and (d, x, r)."""
    found = counts(series)
    if found is None:
        return {name: False for name in RULES}, None
    return {name: bool(rule(*found)) for name, rule in RULES.items()}, found


# ------------------------------------------------------------ must be a date column --

def gate_days(gate, k):
    """The gate's `day_numbers` at data seed k: three rows a day for 600 days."""
    text = gate.dataset_text("day_numbers", gate.SEED + 100 * k)
    return pd.read_csv(io.StringIO(text), low_memory=False)["day"]


def with_cells(day, rng, share=None, cells_set=1, values=(99991231,)):
    """`day` with some cells set to sentinels: `cells_set` cells for each value, or
    `share` of the rows shared out among the values."""
    out = day.to_numpy().copy()
    n = len(out)
    picks = rng.permutation(n)
    if share is None:
        for j, value in enumerate(values):
            out[picks[j * cells_set:(j + 1) * cells_set]] = value
    else:
        chosen = picks[:round(share * n)]
        for j, value in enumerate(values):
            out[chosen[j::len(values)]] = value
    return pd.Series(out)


def ends(freq, start, periods, n=600):
    return pd.Series(np.resize(pd.date_range(start, periods=periods, freq=freq)
                               .strftime("%Y%m%d").astype("int64").to_numpy(), n))


def must_be(gate, k):
    """name -> the column, for data seed k. Every one of these is a column of dates."""
    day = gate_days(gate, k)
    rng = np.random.default_rng(8000 + k)
    out = {"clean (the gate's day_numbers)": day}
    for sentinel in (99991231, 0, 20240230):
        out[f"one cell of {sentinel}"] = with_cells(day, rng, values=(sentinel,))
    for share in (0.05, 0.30, 0.60):
        out[f"{share:.0%} of the rows 99991231"] = with_cells(day, rng, share=share)
    out["two sentinels, one cell each (0 and 99991231)"] = with_cells(
        day, rng, values=(0, 99991231))
    out["two sentinels in 30% of the rows (0 and 99991231)"] = with_cells(
        day, rng, share=0.30, values=(0, 99991231))
    out["three sentinels, one cell each (0, 99991231, 99999999)"] = with_cells(
        day, rng, values=(0, 99991231, 99999999))
    floats = with_cells(day, rng, values=(99991231,)).astype(float)
    floats[rng.random(len(floats)) < 0.05] = np.nan
    out["floats with gaps, one cell of 99991231"] = floats
    out["floats with gaps, clean"] = day.astype(float).where(rng.random(len(day)) >= 0.05)
    text = with_cells(day, rng, values=(99991231,)).astype(str)
    out["eight-digit text, one cell of 99991231"] = pd.Series(text, dtype="str")
    out["eight-digit text, one cell of 0"] = pd.Series(
        with_cells(day, rng, values=(0,)).astype(str), dtype="str")
    out["eight-digit text, 30% of the rows 00000000"] = pd.Series(
        with_cells(day, rng, share=0.30, values=(99991231,)).astype(str), dtype="str") \
        .replace("99991231", "00000000")
    out["eight-digit text, clean"] = pd.Series(day.astype(str), dtype="str")
    # the control of the CP7 measurement that D69 reverses (D72): dates with one cell of 20231301
    out["CP7 control, reversed by D69: one cell of 20231301"] = with_cells(
        day, rng, values=(20231301,))
    # shorter columns of dates
    out["96 month ends, 30% of the rows 99991231"] = with_cells(
        ends("ME", "2015-01-31", 96), rng, share=0.30)
    out["12 month ends, 30% of the rows 99991231"] = with_cells(
        ends("ME", "2023-01-31", 12), rng, share=0.30)
    out["8 quarter ends, 30% of the rows 99991231"] = with_cells(
        ends("QE", "2022-03-31", 8), rng, share=0.30)
    out["5 month ends, 0 and 99991231 in 30% of the rows"] = with_cells(
        ends("ME", "2023-01-31", 5), rng, share=0.30, values=(0, 99991231))
    return out


# -------------------------------------------------------- must not be a date column --

def cp7_controls(k):
    """The tables of docs/measurements/CP7/candidates.py that must not be flagged,
    but for the one D69 reverses (dates with one cell of 20231301)."""
    rng = np.random.default_rng(k)
    n = 600
    days = pd.date_range("2019-01-01", periods=200, freq="D")
    out = {}
    out["eight-digit identifiers drawn at random"] = pd.Series(
        rng.integers(10_000_000, 99_999_999, size=n))
    out["identifiers counting up from 20230101"] = pd.Series(20230101 + np.arange(n))
    out["whole numbers drawn at random between 19000101 and 21001231"] = pd.Series(
        rng.integers(19000101, 21001231, size=n))
    out["years only (1990 to 2024)"] = pd.Series(rng.integers(1990, 2025, size=n))
    out["year and month as YYYYMM"] = pd.Series(np.resize(
        pd.date_range("2000-01-01", periods=240, freq="MS").strftime("%Y%m").astype("int64")
        .to_numpy(), n))
    out["seconds since 1970"] = pd.Series(1_700_000_000 + 60 * np.arange(n))
    out["days since 1970"] = pd.Series(19_000 + np.arange(n) // 3)
    out["prices in whole cents"] = pd.Series(rng.integers(100, 5_000_000, size=n))
    out["counts"] = pd.Series(rng.poisson(3, size=n))
    out["ages"] = pd.Series(rng.integers(18, 90, size=n))
    out["post codes"] = pd.Series(rng.integers(10_000, 99_999, size=n))
    out["dates as DDMMYYYY whole numbers"] = pd.Series(np.repeat(
        days.strftime("%d%m%Y").astype("int64").to_numpy(), 3))
    return out


def random_date(rng):
    while True:
        value = int(rng.integers(19000101, 21001232))
        if is_date_number(value):
            return value


def random_code(rng):
    while True:
        value = int(rng.integers(10_000_000, 100_000_000))
        if not is_date_number(value):
            return value


def code_columns(k):
    """Code columns of two, three and five distinct eight-digit values, of which
    some (at least one, and not all) happen to read as dates. 600 rows."""
    rng = np.random.default_rng(9000 + k)
    out = {}
    for distinct in (2, 3, 5):
        for dates in range(1, distinct):
            codes = [random_date(rng) for _ in range(dates)] \
                + [random_code(rng) for _ in range(distinct - dates)]
            column = np.array(codes)[rng.integers(0, distinct, size=600)]
            column[:distinct] = codes                   # every code is present
            out[f"{distinct} distinct codes, {dates} of them read as dates"] = pd.Series(column)
            out[f"{distinct} distinct codes, {dates} read as dates, as text"] = pd.Series(
                column.astype(str), dtype="str")
    return out


def limits(k):
    """Columns of dates that the adopted rule does NOT name: its stated limits."""
    rng = np.random.default_rng(9500 + k)
    out = {}
    out["4 quarter ends, 30% of the rows 99991231"] = with_cells(
        ends("QE", "2022-03-31", 4), rng, share=0.30)
    out["26 days and three sentinels"] = with_cells(
        ends("D", "2023-01-01", 26), rng, values=(0, 99991231, 99999999), cells_set=20)
    return out


def report(title, tables_of, seeds, want, per_seed):
    said = {True: "all of them", False: "none", None: "shown for the record"}[want]
    print(f"\n{title} (each cell: the seeds, of {seeds}, on which the rule calls the column "
          f"a date column; it must be: {said})\n")
    print("| Column | d, x over the seeds | share of cells that are dates | "
          + " | ".join(RULES) + " |")
    print("|---|---|---|" + "---|" * len(RULES))
    names = list(tables_of(0))
    wrong = {name: 0 for name in RULES}
    for name in names:
        hits = {rule: 0 for rule in RULES}
        ds, xs, rs, rows = [], [], [], []
        for k in range(seeds):
            verdict, found = flagged(tables_of(k)[name])
            for rule, yes in verdict.items():
                hits[rule] += yes
            if found:
                ds.append(found[0]), xs.append(found[1]), rs.append(found[2])
            rows.append((k, found, verdict))
        if want is not None:
            for rule in RULES:
                wrong[rule] += hits[rule] != (seeds if want else 0)
        dx = f"d {min(ds)}..{max(ds)}, x {min(xs)}..{max(xs)}" if ds else "not looked at"
        share = f"{min(rs):.3f}..{max(rs):.3f}" if rs else ""
        print(f"| {name} | {dx} | {share} | "
              + " | ".join(f"{hits[rule]}" for rule in RULES) + " |")
        if per_seed:
            for k, found, verdict in rows:
                shown = "not looked at" if found is None else \
                    f"d {found[0]}, x {found[1]}, r {found[2]:.4f}"
                print(f"    seed {k}: {shown}: " + " ".join(
                    "1" if verdict[rule] else "0" for rule in RULES))
    if want is not None:
        print("| columns on which the rule is wrong | | | "
              + " | ".join(str(wrong[rule]) for rule in RULES) + " |")


def gate_and_example(gate, seeds):
    print("\nMUST NOT be: every column of every gate dataset but the two made for the rule "
          f"({', '.join(DATE_DATASETS)}), and of examples/customers.csv (the target included: a "
          "rule that named it would show)\n")
    print("| Table | data seeds | columns | of them, columns rule 2a looks at | "
          + " | ".join(RULES) + " |")
    print("|---|---|---|---|" + "---|" * len(RULES))
    total = {rule: 0 for rule in RULES}
    for name in gate.DATASETS:
        if name in DATE_DATASETS:
            continue
        looked = seen = loaded = 0
        named = {rule: set() for rule in RULES}
        for k in range(seeds):
            text = gate.dataset_text(name, gate.SEED + 100 * k)
            try:
                frame = pd.read_csv(io.StringIO(text), low_memory=False)
            except Exception:       # noqa: BLE001  (`long_rows` is not a table)
                continue
            loaded += 1
            for column in frame.columns:
                looked += 1
                verdict, found = flagged(frame[column])
                seen += found is not None
                for rule, yes in verdict.items():
                    if yes:
                        named[rule].add(column)
        for rule in RULES:
            total[rule] += len(named[rule])
        print(f"| gate `{name}` | {loaded} | {looked} | {seen} | "
              + " | ".join(", ".join(sorted(named[rule])) or "none" for rule in RULES) + " |")
    frame = pd.read_csv(REPO / "examples" / "customers.csv", low_memory=False)
    named = {rule: [c for c in frame.columns if flagged(frame[c])[0][rule]] for rule in RULES}
    for rule in RULES:
        total[rule] += len(named[rule])
    print(f"| examples/customers.csv | - | {len(frame.columns)} | "
          f"{sum(counts(frame[c]) is not None for c in frame.columns)} | "
          + " | ".join(", ".join(named[rule]) or "none" for rule in RULES) + " |")
    print("| columns named in all | | | | " + " | ".join(str(total[rule]) for rule in RULES) + " |")
    print("\nThe gate's two tables made for the rule, the column `day`, over the same seeds:")
    for name in DATE_DATASETS:
        hits = {rule: 0 for rule in RULES}
        for k in range(seeds):
            text = gate.dataset_text(name, gate.SEED + 100 * k)
            verdict, _ = flagged(pd.read_csv(io.StringIO(text), low_memory=False)["day"])
            for rule, yes in verdict.items():
                hits[rule] += yes
        print(f"  `{name}`: " + ", ".join(f"{rule}: {hits[rule]}" for rule in RULES))


def smallest_defeat(gate, seeds):
    """How many DISTINCT values that are no dates, one cell each, it takes before
    each rule stops calling the gate's `day_numbers` a date column."""
    print("\nThe smallest change that defeats each rule on the gate's `day_numbers` (600 "
          "distinct dates, 1,800 cells): the number of distinct values that are no dates, one "
          "cell each (99991231, 99991230, ...), at which the column is no longer a date column\n")
    print("| Rule | cells changed, min..max over the seeds |")
    print("|---|---|")
    for rule in RULES:
        needed = []
        for k in range(seeds):
            day = gate_days(gate, k).to_numpy().copy()
            rng = np.random.default_rng(7000 + k)
            picks = rng.permutation(len(day))
            found = None
            for m in range(1, len(day) + 1):
                day[picks[m - 1]] = 99991231 - (m - 1)
                if not flagged(pd.Series(day))[0][rule]:
                    found = m
                    break
            needed.append(found)
        print(f"| {rule} | {min(needed)}..{max(needed)} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--per-seed", action="store_true")
    args = parser.parse_args()
    gate = gate_module()
    print("d: distinct values that read as dates; x: distinct values that do not\n"
          + "\n".join(f"  {name}" for name in RULES))
    report("MUST BE a date column", lambda k: must_be(gate, k), args.seeds, True, args.per_seed)
    report("MUST NOT be: the controls of the CP7 measurement", cp7_controls, args.seeds, False,
           args.per_seed)
    report("MUST NOT be: code columns of a few distinct eight-digit values", code_columns,
           args.seeds, False, args.per_seed)
    report("The limits of the adopted rule: columns of dates it does not name", limits,
           args.seeds, None, args.per_seed)
    gate_and_example(gate, args.seeds)
    smallest_defeat(gate, args.seeds)
