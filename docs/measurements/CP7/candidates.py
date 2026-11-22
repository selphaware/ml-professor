"""CP7, section D: two candidate warnings, measured and not built.

    python -B candidates.py dates   [--seeds 30]
    python -B candidates.py series  [--seeds 30]

`dates` (the fifth review, finding 4): under `time_col=False`, a whole-number
column in which every value reads as a calendar date written YYYYMMDD.
`series` (finding 3): under a time column and `group_col=False`, a text or
few-valued column whose levels each form a series of their own.

Each detector is run on every dataset of the gate (30 data seeds: the gate's
base seed 1115, then 1215, and so on), on examples/customers.csv, and on tables
made here for the purpose: ones it should flag and ones it should not. Nothing
here is part of the tool, the reference or the gate. No file is written.
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
YEARS = (1900, 2100)        # the years accepted, both ends included


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


def gate_frame(gate, name, base):
    """A gate dataset as pandas loads its file (long rows cannot be loaded: None)."""
    text = gate.dataset_text(name, base)
    try:
        return pd.read_csv(io.StringIO(text), low_memory=False)
    except Exception:           # noqa: BLE001  (`long_rows` is not a table)
        return None


# ------------------------------------------------------------------ dates --

def whole_numbers(series):
    """The non-missing values as Python integers, or None when the column is not one
    of whole numbers (an integer type, or floats or digit text that are all whole)."""
    values = series.dropna()
    if len(values) == 0 or pd.api.types.is_bool_dtype(values.dtype):
        return None
    if pd.api.types.is_integer_dtype(values.dtype):
        return [int(v) for v in values.tolist()]
    if pd.api.types.is_float_dtype(values.dtype):
        array = values.to_numpy(dtype="float64")
        if not np.isfinite(array).all() or not (array == np.floor(array)).all():
            return None
        return [int(v) for v in array.tolist()]
    if pd.api.types.is_string_dtype(values.dtype) or values.dtype == object:
        out = []
        for v in values.tolist():
            if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
                out.append(int(v))
            elif isinstance(v, str) and v.strip().isdigit() and v.strip().isascii():
                out.append(int(v.strip()))
            else:
                return None
        return out
    return None


def is_yyyymmdd(value):
    year, month, day = value // 10000, (value // 100) % 100, value % 100
    if not (YEARS[0] <= year <= YEARS[1] and 1 <= month <= 12 and day >= 1):
        return False
    return day <= calendar.monthrange(year, month)[1]


def date_numbers(frame, target):
    """The candidate rule: columns other than the target in which every non-missing
    value is a whole number that reads as YYYYMMDD, with at least two distinct values."""
    flagged = []
    for name in frame.columns:
        if name == target:
            continue
        values = whole_numbers(frame[name])
        if values is None or len(set(values)) < 2:
            continue
        if all(is_yyyymmdd(v) for v in values):
            flagged.append(name)
    return flagged


def share_reading_as_dates(values):
    return sum(is_yyyymmdd(int(v)) for v in values) / len(values)


def made_for_dates(seed):
    """Tables made for the purpose: name -> (frame, target, the columns that should be flagged)."""
    rng = np.random.default_rng(seed)
    n = 600
    days = pd.date_range("2019-01-01", periods=200, freq="D")
    stamp = np.repeat(days.strftime("%Y%m%d").astype("int64").to_numpy(), 3)    # three rows a day
    y = rng.integers(0, 2, size=n)
    x = rng.normal(size=n)
    out = {}
    out["finding 4: the day as a whole number, three rows a day"] = (
        pd.DataFrame({"day": stamp, "x": x, "y": y}), "y", ["day"])
    out["the same, as floats with empty cells"] = (
        pd.DataFrame({"day": np.where(rng.random(n) < 0.05, np.nan, stamp.astype(float)),
                      "x": x, "y": y}), "y", ["day"])
    out["the same, as digit text"] = (
        pd.DataFrame({"day": stamp.astype(str), "x": x, "y": y}), "y", ["day"])
    out["one row a day, in order (id-like today)"] = (
        pd.DataFrame({"day": pd.date_range("2015-06-01", periods=n, freq="D")
                      .strftime("%Y%m%d").astype("int64"), "x": x, "y": y}), "y", ["day"])
    out["month ends over 40 years, shuffled"] = (
        pd.DataFrame({"day": rng.permutation(np.resize(
            pd.date_range("1985-01-31", periods=480, freq="ME").strftime("%Y%m%d")
            .astype("int64").to_numpy(), n)), "x": x, "y": y}), "y", ["day"])
    out["two date columns (opened, closed)"] = (
        pd.DataFrame({"opened": stamp, "closed": rng.permutation(stamp), "x": x, "y": y}),
        "y", ["opened", "closed"])
    # ones it should not flag
    out["eight-digit identifiers drawn at random"] = (
        pd.DataFrame({"account": rng.integers(10_000_000, 99_999_999, size=n), "x": x, "y": y}),
        "y", [])
    out["identifiers counting up from 20230101"] = (
        pd.DataFrame({"order": 20230101 + np.arange(n), "x": x, "y": y}), "y", [])
    out["whole numbers drawn at random between 19000101 and 21001231"] = (
        pd.DataFrame({"amount": rng.integers(19000101, 21001231, size=n), "x": x, "y": y}),
        "y", [])
    out["years only (1990 to 2024)"] = (
        pd.DataFrame({"year": rng.integers(1990, 2025, size=n), "x": x, "y": y}), "y", [])
    out["year and month as YYYYMM"] = (
        pd.DataFrame({"month": np.resize(pd.date_range("2000-01-01", periods=240, freq="MS")
                                         .strftime("%Y%m").astype("int64").to_numpy(), n),
                      "x": x, "y": y}), "y", [])
    out["seconds since 1970"] = (
        pd.DataFrame({"t": 1_700_000_000 + 60 * np.arange(n), "x": x, "y": y}), "y", [])
    out["days since 1970"] = (
        pd.DataFrame({"t": 19_000 + np.arange(n) // 3, "x": x, "y": y}), "y", [])
    out["prices in whole cents, counts, ages, post codes"] = (
        pd.DataFrame({"cents": rng.integers(100, 5_000_000, size=n),
                      "count": rng.poisson(3, size=n), "age": rng.integers(18, 90, size=n),
                      "postcode": rng.integers(10_000, 99_999, size=n), "x": x, "y": y}),
        "y", [])
    out["dates as YYYYMMDD with one cell of 20231301"] = (
        pd.DataFrame({"day": np.concatenate([[20231301], stamp[1:]]), "x": x, "y": y}), "y", [])
    out["dates as DDMMYYYY whole numbers"] = (
        pd.DataFrame({"day": np.repeat(days.strftime("%d%m%Y").astype("int64").to_numpy(), 3),
                      "x": x, "y": y}), "y", [])
    return out


def run_dates(seeds):
    gate = gate_module()
    print(f"candidate 1: a whole-number column that reads as YYYYMMDD, years {YEARS[0]} to "
          f"{YEARS[1]}\n")
    print("| Table | data seeds | columns looked at | whole-number columns | flagged |")
    print("|---|---|---|---|---|")
    total, made_for_it = 0, False
    for name, (_, target) in gate.DATASETS.items():
        looked = whole = 0
        flagged = set()
        loaded = 0
        for k in range(seeds):
            frame = gate_frame(gate, name, gate.SEED + 100 * k)
            if frame is None:
                continue
            loaded += 1
            looked += len(frame.columns) - 1
            whole += sum(whole_numbers(frame[c]) is not None for c in frame.columns if c != target)
            flagged.update(date_numbers(frame, target))
        if name == "day_numbers":           # the dataset this pass made for the rule
            assert flagged == {"day"}, flagged
            made_for_it = True
        else:
            total += len(flagged)
        note = "" if loaded else " (rows longer than the header: not a table)"
        if name == "day_numbers":
            note = " (added by this pass for the rule: its column `day` is dates)"
        print(f"| gate `{name}`{note} | {loaded} | {looked} | {whole} | "
              f"{', '.join(sorted(flagged)) or 'none'} |")
    frame = pd.read_csv(REPO / "examples" / "customers.csv", low_memory=False)
    flagged = date_numbers(frame, "churned")
    total += len(flagged)
    print(f"| examples/customers.csv | - | {len(frame.columns) - 1} | "
          f"{sum(whole_numbers(frame[c]) is not None for c in frame.columns if c != 'churned')} | "
          f"{', '.join(flagged) or 'none'} |")
    print(f"\nflagged on the gate's other datasets and the example table: {total}"
          + ("; on `day_numbers`, the one column that is dates" if made_for_it else "") + "\n")
    print("| Table made for the purpose | should flag | flagged, over the seeds | verdict |")
    print("|---|---|---|---|")
    wrong = 0
    names = list(made_for_dates(0))
    for name in names:
        hits, expected = [], None
        for k in range(seeds):
            frame, target, expected = made_for_dates(k)[name]
            hits.append(tuple(date_numbers(frame, target)))
        ok = all(sorted(h) == sorted(expected) for h in hits)
        wrong += not ok
        seen = sorted({", ".join(h) or "none" for h in hits})
        print(f"| {name} | {', '.join(expected) or 'none'} | {'; '.join(seen)} | "
              f"{'as it should' if ok else 'WRONG'} |")
    print(f"\ntables made for the purpose on which the rule is wrong: {wrong} of {len(names)}")
    rng = np.random.default_rng(0)
    draws = rng.integers(19000101, 21001232, size=1_000_000)
    print(f"a whole number drawn at random between 19000101 and 21001231 reads as a date "
          f"with probability {share_reading_as_dates(draws[:200_000]):.4f}; a column of 100 such "
          f"values all read as dates with probability about "
          f"{share_reading_as_dates(draws[:200_000]) ** 100:.1e}")


# ----------------------------------------------------------------- series --

def reference():
    sys.path.insert(0, str(REPO / "verify" / "reference"))
    from mlprofessor import core
    return core


def level_codes(series):
    """One whole number per row for a text or whole-number column (-1 for an empty
    cell), or None when the column is neither."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return None
    if pd.api.types.is_numeric_dtype(series.dtype):
        values = series.to_numpy(dtype="float64", na_value=np.nan)
        held = values[~np.isnan(values)]
        if len(held) == 0 or not np.isfinite(held).all() or not (held == np.floor(held)).all():
            return None
        return pd.factorize(values, use_na_sentinel=True)[0]
    if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
        cells = [v.strip() if isinstance(v, str) else None for v in series.tolist()]
        if any(c is None and not pd.isna(v) for c, v in zip(cells, series.tolist())):
            return None
        return pd.factorize(np.array(cells, dtype=object), use_na_sentinel=True)[0]
    return None


def series_columns(core, frame, target, time_col, min_rows_per_level=4):
    """The candidate measure, for every text or whole-number column but the target and
    the time column: how much better the previous value within a level predicts the
    target, over the training rows, than the table's own baseline does.

    Returns records {column, levels, covered, skill, skill_own_mean} for the columns
    whose levels can be series: at least two levels, at least `min_rows_per_level`
    rows a level on average, no empty cell, and no level with two rows at one
    timestamp. `skill` is 1 - (squared error of the level's previous value) /
    (squared error of the table's baseline), the baseline being the training mean
    and, where the timestamps are unique, the previous row if that is better.
    `skill_own_mean` is the same against each level's own training mean: a level
    that is a series, and not only a level, beats that too.
    """
    y_all = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype="float64")
    keys_all, missing_time, _form, _show = core._time_keys(frame[time_col], time_col)
    usable = np.flatnonzero(~np.isnan(y_all) & ~missing_time)
    order = usable[np.argsort(keys_all[usable], kind="stable")]
    y, keys = y_all[order], keys_all[order]
    split = core._split_time(keys, 0)
    train = split["train"]
    unique = len(np.unique(keys)) == len(keys)
    out = []
    for name in frame.columns:
        if name in (target, time_col):
            continue
        codes = level_codes(frame[name])
        if codes is None:
            continue
        codes = codes[order]
        if (codes < 0).any():
            continue
        levels = len(np.unique(codes))
        if levels < 2 or len(codes) < min_rows_per_level * levels:
            continue
        _, stamp = np.unique(keys, return_inverse=True)
        cell = codes.astype("int64") * (int(stamp.max()) + 1) + stamp
        if len(np.unique(cell)) != len(cell):
            continue                    # a level with two rows at one timestamp is no series
        prev = core._previous_by_group(y, keys, codes, 0)
        rows = train[~np.isnan(prev[train])]
        if len(rows) < 30 or 2 * len(rows) < len(train):
            continue
        own = float(np.mean((y[rows] - prev[rows]) ** 2))
        table = float(np.mean((y[rows] - y[train].mean()) ** 2))
        if unique:
            before = np.full(len(y), np.nan)
            before[1:] = y[:-1]
            held = rows[~np.isnan(before[rows])]
            table = min(table, float(np.mean((y[held] - before[held]) ** 2)))
        sums = np.bincount(codes[train], weights=y[train], minlength=levels)
        counts = np.maximum(np.bincount(codes[train], minlength=levels), 1)
        level_mean = float(np.mean((y[rows] - (sums / counts)[codes[rows]]) ** 2))
        out.append({"column": name, "levels": levels, "covered": len(rows) / len(train),
                    "skill": None if table == 0 else 1 - own / table,
                    "skill_own_mean": None if level_mean == 0 else 1 - own / level_mean})
    return out


def walks(rng, n_series, n_times, offset_seconds=0, start_spread=20.0, step=1.0, names=None,
          phi=1.0):
    """A panel: `n_series` series on the same `n_times` business days, each at a level
    of its own. `phi` 1 is a random walk, below 1 a series that returns to its level.
    `offset_seconds`: series k's stamps are k x that many seconds later (finding 3)."""
    days = pd.bdate_range("2021-01-04", periods=n_times)
    parts = []
    for s in range(n_series):
        centre = rng.normal(0, start_spread)
        dev, levels = 0.0, []
        for _ in range(n_times + 1):
            dev = phi * dev + rng.normal(0, step)
            levels.append(centre + dev)
        levels = np.array(levels)
        when = days + pd.Timedelta(seconds=offset_seconds * s)
        parts.append(pd.DataFrame({
            "when": when.strftime("%Y-%m-%dT%H:%M:%S"),
            "series": (names[s] if names is not None else f"S{s + 1:03d}"),
            "prev": levels[:-1], "y": levels[1:]}))
    frame = pd.concat(parts, ignore_index=True)
    return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)


def made_for_series(seed):
    """name -> (frame, target, time column, the columns that should be flagged)."""
    rng = np.random.default_rng(seed)
    out = {}
    out["finding 3: twelve walks, stamps offset by one second"] = (
        walks(rng, 12, 125, offset_seconds=1), "y", "when", ["series"])
    out["twelve walks on the same stamps"] = (walks(rng, 12, 125), "y", "when", ["series"])
    out["finding 2: 150 walks of 12 stamps"] = (walks(rng, 150, 12), "y", "when", ["series"])
    out["40 walks of 8 stamps"] = (walks(rng, 40, 8), "y", "when", ["series"])
    out["40 walks of 36 stamps"] = (walks(rng, 40, 36), "y", "when", ["series"])
    out["twelve walks, the series named by a whole number"] = (
        walks(rng, 12, 125, names=list(range(101, 113))), "y", "when", ["series"])
    out["twelve series that return to their own level (0.9 a step)"] = (
        walks(rng, 12, 125, phi=0.9), "y", "when", ["series"])
    out["twelve series that return quickly (0.5 a step)"] = (
        walks(rng, 12, 125, phi=0.5), "y", "when", ["series"])
    # ones it should not flag: one series, or levels that are not series
    n = 900
    days = pd.bdate_range("2019-01-01", periods=n)
    walk = np.cumsum(rng.normal(size=n + 1))
    one = pd.DataFrame({"when": days.strftime("%Y-%m-%d"), "weekday": days.day_name(),
                        "month": days.month, "segment": rng.choice(list("ABCD"), size=n),
                        "rating": rng.integers(1, 6, size=n), "prev": walk[:-1], "y": walk[1:]})
    out["one random walk with a weekday, a month, a segment and a rating"] = (
        one, "y", "when", [])
    effect = {"A": -6.0, "B": -2.0, "C": 2.0, "D": 6.0}
    seg = rng.choice(list("ABCD"), size=n)
    level = pd.DataFrame({"when": days.strftime("%Y-%m-%d"), "segment": seg,
                          "x": rng.normal(size=n),
                          "y": np.array([effect[s] for s in seg]) + rng.normal(size=n)})
    out["independent rows, a segment that sets the level (4 sd apart)"] = (
        level, "y", "when", [])
    season = np.array([0.0, 5.0, 10.0, 5.0, 0.0])[days.day_of_week.to_numpy()]
    out["one series with a strong weekly pattern and independent noise"] = (
        pd.DataFrame({"when": days.strftime("%Y-%m-%d"), "weekday": days.day_name(),
                      "x": rng.normal(size=n), "y": season + rng.normal(size=n)}),
        "y", "when", [])
    slow = 0.2 * walk
    out["one slow random walk under a strong weekly pattern"] = (
        pd.DataFrame({"when": days.strftime("%Y-%m-%d"), "weekday": days.day_name(),
                      "x": rng.normal(size=n), "y": season + slow[1:]}),
        "y", "when", [])
    far = {"A": -30.0, "B": -10.0, "C": 10.0, "D": 30.0}
    out["independent rows, a segment that sets the level (20 sd apart)"] = (
        pd.DataFrame({"when": days.strftime("%Y-%m-%d"), "segment": seg,
                      "x": rng.normal(size=n),
                      "y": np.array([far[s] for s in seg]) + rng.normal(size=n)}),
        "y", "when", [])
    stores = [f"store{k}" for k in range(8)]
    cross = pd.DataFrame({"when": np.repeat(days[:110].strftime("%Y-%m-%d").to_numpy(), 8),
                          "store": np.tile(stores, 110), "x": rng.normal(size=880)})
    cross["y"] = 2.0 * cross["x"] + rng.normal(size=880)
    out["eight stores a day, the target independent of the store"] = (cross, "y", "when", [])
    cross2 = cross.copy()
    cross2["y"] = cross2["y"] + np.tile(rng.normal(0, 6, size=8), 110)
    out["eight stores a day, each store at a level of its own, no memory"] = (
        cross2, "y", "when", [])
    return out


CUTS = (0.5, 0.9, 0.95, 0.98)
PER_SEED = False
SEEDS_SHOWN = []


def run_series(seeds):
    gate, core = gate_module(), reference()
    in_scope = {"random_walk": "date", "int_stamps": "t", "flat_tail": "date", "panel": "date"}
    print("candidate 2: a text or whole-number column whose levels each form a series\n")
    print("skill: 1 - squared error of the previous value within a level / squared error of "
          "the table's own baseline, over the training rows; `own mean`: the same against "
          "each level's own training mean\n")
    print("| Table | column | levels | skill, min..max over the seeds | own mean, min..max | "
          + " | ".join(f"flagged at {c}" for c in CUTS)
          + " | flagged at 0.5 on both |")
    print("|---|---|---|---|---|" + "---|" * (len(CUTS) + 1))

    def report(label, runs):
        """runs: one list of records per seed."""
        columns = sorted({r["column"] for run in runs for r in run})
        if not columns:
            print(f"| {label} | no column whose levels could be series | | | |"
                  + " |" * (len(CUTS) + 1))
            return {}
        counts = {}
        for column in columns:
            recs = [r for run in runs for r in run if r["column"] == column]
            skill = [r["skill"] for r in recs if r["skill"] is not None]
            own = [r["skill_own_mean"] for r in recs if r["skill_own_mean"] is not None]
            flags = [sum(r["skill"] is not None and r["skill"] >= c for r in recs) for c in CUTS]
            both = sum(r["skill"] is not None and r["skill_own_mean"] is not None
                       and r["skill"] >= 0.5 and r["skill_own_mean"] >= 0.5 for r in recs)
            counts[column] = (flags, both, len(runs))
            if PER_SEED:
                SEEDS_SHOWN.append((label, column, [r["skill"] for r in recs],
                                    [r["skill_own_mean"] for r in recs]))
            levels = sorted({r["levels"] for r in recs})
            print(f"| {label} | {column} | {levels[0] if len(levels) == 1 else levels} | "
                  f"{min(skill):+.3f}..{max(skill):+.3f} | {min(own):+.3f}..{max(own):+.3f} | "
                  + " | ".join(f"{f} of {len(runs)}" for f in flags)
                  + f" | {both} of {len(runs)} |")
        return counts

    for name, (_, target) in gate.DATASETS.items():
        if name not in in_scope:
            continue
        runs = []
        for k in range(seeds):
            frame = gate_frame(gate, name, gate.SEED + 100 * k)
            runs.append(series_columns(core, frame, target, in_scope[name]))
        report(f"gate `{name}` (--time-col {in_scope[name]})", runs)
    others = [n for n in gate.DATASETS if n not in in_scope]
    print(f"\nnot in scope (no regression under a time column in the gate's own run): "
          f"{', '.join(others)}\n")
    frame = pd.read_csv(REPO / "examples" / "customers.csv", low_memory=False)
    print("| Table | column | levels | skill | own mean | " + " | ".join(
        f"flagged at {c}" for c in CUTS) + " | flagged at 0.5 on both |")
    print("|---|---|---|---|---|" + "---|" * (len(CUTS) + 1))
    for target in ("monthly_spend", "age"):
        report(f"examples/customers.csv, target {target}, time column signup_date",
               [series_columns(core, frame, target, "signup_date")])
    print()
    print("| Table made for the purpose | column | levels | skill, min..max | own mean, min..max | "
          + " | ".join(f"flagged at {c}" for c in CUTS) + " | flagged at 0.5 on both |")
    print("|---|---|---|---|---|" + "---|" * (len(CUTS) + 1))
    names = list(made_for_series(0))
    for name in names:
        runs, expected = [], None
        for k in range(seeds):
            frame, target, time_col, expected = made_for_series(k)[name]
            runs.append(series_columns(core, frame, target, time_col))
        report(f"{name} [should flag: {', '.join(expected) or 'none'}]", runs)
    if PER_SEED:
        print("\nper seed (data seed 0, 1, ... for the tables made here; base seed 1115, "
              "1215, ... for the gate's): the ratio against the table's own baseline, then "
              "against each level's own mean")
        for label, column, skill, own in SEEDS_SHOWN:
            def row(values):
                return " ".join("none" if v is None else f"{v:+.3f}" for v in values)
            print(f"{label} | {column} | table: {row(skill)}")
            print(f"{label} | {column} | own mean: {row(own)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["dates", "series"])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--table", action="store_true",
                        help="series: print the ratio at every seed")
    args = parser.parse_args()
    PER_SEED = args.table
    (run_dates if args.which == "dates" else run_series)(args.seeds)
