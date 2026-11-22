"""Shared helpers of the sixth pass (CP7). Builds on `mlp_helpers`, `cp3_helpers`,
`cp5_helpers` and `cp6_helpers`; holds no test.

Unlike the helpers of the earlier passes, nothing here makes a statement the call
did not make (D63): `run`, `raised` and `run_cli` call the application exactly as
they are called. A test that is meant to finish passes `time_col` and `group_col`
itself (`**SAID` for "no" to both).

Nothing here imports `mlprofessor` at module level: each test imports the
application inside its own body.
"""
import datetime as _dt

import numpy as np
import pandas as pd

import cp3_helpers as c
import cp5_helpers as p
import mlp_helpers as h

SAID = {"time_col": False, "group_col": False}

# The four texts each message must hold (spec 2.4, step 5, 11).
TIME_TEXTS = ("time_col", "--time-col", "time_col=False", "--no-time-order")
ENTITY_TEXTS = ("group_col", "--group-col", "group_col=False", "--no-entities")

# The three fixed passages of spec 5.4, exactly, final point included.
COPIES_ADVICE = (
    "Pass duplicates='keep' (--duplicates keep) if the repeated rows are different cases "
    "that look alike, as in a table of few-valued columns. Pass duplicates='group' "
    "(--duplicates group) if they are one case recorded more than once: no combination of "
    "feature values is then on both sides of the split. Or remove the repeated rows."
)
LARGE_KEPT = ("It is kept among the training rows and the holdout is made of smaller units: "
              "the headline does not test it.")
LARGE_IS_HOLDOUT = ("The units smaller than a quarter do not together hold a quarter, so this "
                    "one unit is the holdout: the headline rests on it alone.")

KEPT = "large_group_kept_in_training"
IS_HOLDOUT = "large_group_is_holdout"


# ------------------------------------------------------------------ running

def run(frame, target="y", families=None, steps=None, seed=0, **kw):
    """`profile()` exactly as called, fast profile and one core by default, inside
    `extensions.use`. It adds no statement."""
    import mlprofessor
    from mlprofessor import extensions

    args = dict(h.FAST)
    args.update(kw)
    with extensions.use(families=families, steps=steps):
        return mlprofessor.profile(frame, target, seed=seed, **args)


def raised(frame, target="y", families=None, steps=None, **kw):
    """The `ProfileError` of the call exactly as made (no statement added)."""
    return p.raised(frame, target, families=families, steps=steps, as_called=True, **kw)


def run_cli(args, timeout=900):
    """The command line exactly as given (no flag added)."""
    return h.run_cli(args, timeout=timeout, as_given=True)


def rec():
    """A recording family on a fresh log: `(log, families)`."""
    log = h.Log()
    return log, [h.family("rec", log)]


def codes(result):
    return c.codes(result)


def messages(result, code):
    """The messages of every warning with `code`, in order."""
    return [str(r["message"]) for r in h.records(result.warnings) if r["code"] == code]


def warning_records(result, code):
    return [r for r in h.records(result.warnings) if r["code"] == code]


def numbers_in(text):
    return c.numbers_in(text)


def holds_number(text, value, within=0.1):
    """Some number written in `text` is within `within` of `value`."""
    return any(abs(n - value) <= within for n in numbers_in(text))


# ------------------------------------------------------------------ tables

def sized_entities(sizes, seed=0, labels=None):
    """One entity per entry of `sizes`, named `e000`, `e001`, ...: `who`, two noise
    measures `m1` and `m2`, and `y` (normal noise; with `labels`, one 0 or 1 per entity).
    Rows shuffled from `seed`. No two rows agree."""
    rng = np.random.default_rng(seed)
    sizes = [int(s) for s in sizes]
    if labels is not None:
        labels = [int(v) for v in labels]
        assert len(labels) == len(sizes)
    who, y = [], []
    for k, size in enumerate(sizes):
        who.extend([f"e{k:03d}"] * size)
        if labels is None:
            y.extend(rng.normal(size=size).tolist())
        else:
            y.extend([labels[k]] * size)
    n = len(who)
    frame = pd.DataFrame({"who": who, "m1": rng.normal(size=n), "m2": rng.normal(size=n),
                          "y": y})
    order = rng.permutation(n)
    return frame.iloc[order].reset_index(drop=True)


def few_valued(n=600, levels=(4, 3), seed=0):
    """Text columns `c1`, `c2`, ... of that many levels each, drawn at random, and a 0/1
    target `y` that depends on none of them."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        f"c{k + 1}": [f"v{k + 1}_{int(v)}" for v in rng.integers(0, count, size=n)]
        for k, count in enumerate(levels)
    })
    frame["y"] = rng.integers(0, 2, size=n)
    return frame


def date_number(day):
    """A `datetime.date` (or anything with year, month, day) as the whole number YYYYMMDD."""
    return day.year * 10000 + day.month * 100 + day.day


def day_numbers(n_days=200, per=3, seed=0, start="2019-01-01"):
    """`per` rows a day for `n_days` days: `day` as an int64 written YYYYMMDD, a noise
    column `x` and a 0/1 target `y`. Rows shuffled from `seed`."""
    rng = np.random.default_rng(seed)
    first = _dt.date.fromisoformat(start)
    days = [date_number(first + _dt.timedelta(days=k)) for k in range(n_days)]
    n = n_days * per
    frame = pd.DataFrame({
        "day": np.repeat(np.array(days, dtype=np.int64), per),
        "x": rng.normal(size=n),
        "y": rng.integers(0, 2, size=n),
    })
    order = rng.permutation(n)
    return frame.iloc[order].reset_index(drop=True)


# ------------------------------------------------- property 3 of step 7

def quarter_unit(sizes):
    """The smallest unit such that the units no larger than it hold, together, at least
    a quarter of the rows."""
    sizes = sorted(int(s) for s in sizes)
    total = sum(sizes)
    running = 0
    for size in sizes:
        running += size
        if 4 * running >= total:
            return size
    raise AssertionError("no units")


def _as_mapping(units):
    if isinstance(units, dict):
        return {int(r): u for r, u in units.items()}
    return {r: u for r, u in enumerate(list(units))}


def assert_property_three(holdout_rows, units, where=""):
    """Property 3 of spec step 7, from the rows alone. `units` holds the unit of every
    row split (a sequence indexed by row, or a mapping row -> unit)."""
    unit_of = _as_mapping(units)
    held = {int(r) for r in holdout_rows}
    assert held <= set(unit_of), f"{where}: holdout rows outside the rows split"
    size = {}
    for u in unit_of.values():
        size[u] = size.get(u, 0) + 1
    held_size = {}
    for r in held:
        held_size[unit_of[r]] = held_size.get(unit_of[r], 0) + 1
    for u, k in held_size.items():
        assert k == size[u], f"{where}: unit {u!r} has {k} of its {size[u]} rows held out"
    n, hold = len(unit_of), len(held)
    assert 4 * hold >= n, f"{where}: holdout {hold} of {n} rows is less than a quarter"
    for u in held_size:
        assert 4 * (hold - size[u]) < n, (
            f"{where}: without unit {u!r} ({size[u]} rows) the holdout still holds "
            f"{hold - size[u]} of {n} rows, a quarter or more")
    bound = quarter_unit(size.values())
    assert 4 * hold < n + 4 * bound, (
        f"{where}: holdout {hold} of {n} rows is not less than a quarter plus {bound}")
