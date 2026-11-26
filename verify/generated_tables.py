"""Generated tables: the application against the reference implementation (D56, D71,
D73, D74).

The gate judges the tool on a few tables whose right conclusion is known, and the
suite on tables made for one rule each. Neither looks at a table nobody thought
of. This tool does: a generator makes table number i from a seed, each table is
run through the package under test and through the reference implementation
(verify/reference/), each in processes of its own, and the two outcomes must
agree. Agreement proves neither side right. A disagreement is a finding: one
side departs from the spec, or the spec does not say.

    python verify/generated_tables.py <package folder> [--reference FOLDER] [--tables N]
           [--ninth N] [--seed S] [--only i,j] [--jobs N] [--json FILE] [--keep]
           [--describe]

`<package folder>` is the folder `mlprofessor` to judge. The reference defaults to
verify/reference/mlprofessor. Without options: tables 0 to 999 of seed 20261122,
and the first 200 of the ninth pass's series, tables 1,000,000 to 1,000,199: about
110 seconds at 24 workers on this machine (2026-11-26, idle). Before a review the
lead runs more, by hand: `--tables 10000 --ninth 2000`, or other seeds.

`--json FILE` writes the counts, the class of every table by its number
(`classes`) and, for every table that has a finding, the table and its findings
in full (`tables_with_findings`): a test reads there which tables fail, and why.

Exit codes
    0  nothing fails: every table agrees, or is on one of the two lists below
    1  at least one table fails
    3  HARNESS ERROR: an outcome is missing, or the comparison itself raised

One table again, and what it is:

    python verify/generated_tables.py mlprofessor --seed S --only N --keep
    python verify/generated_tables.py --seed S --only N --describe

The same seed and number give the same table, from now on: the generator is not
to be changed in a way that moves an existing table. `--keep` leaves the two
outputs and the CSV file in the working folder, which is printed.

The tables. Sizes from 96 rows to past the fast profile's cap; a DataFrame or a
CSV file; the four combinations of the two statements, and a statement left
unsaid; one row for each timestamp, repeated timestamps and panels; entities of
one size, one large entity of a quarter of the rows to past the cap, every row
its own; both tasks, declared rightly, wrongly or not at all; each value of
`duplicates`; gaps; both profiles; and columns of every kind the spec knows,
among them dates written YYYYMMDD with a sentinel in one cell or in a third of
the rows, and a two-valued copy of the target beside few-valued columns. The ninth
pass's series (D77, D78, D80) is the same recipe, each table numbered from 1,000,000
up, with columns of that pass's kinds drawn from a stream of their own, so that no
table below 1,000,000 moves: noise with repeated values (rounded to one decimal place,
or a whole number from 1 to 40), the target as a code (text or numbers, labels in
random order), two columns that give the target only together, an exact copy of a
feature, and a feature put into 10 or 40 bands.

What the headline rests on is a property of each side alone (5.2, step 12; D77,
D78, D80): every column record holds `shuffled_score`, `shuffled_skill` and
`skill_fall`, null where the column reached no model or a warning says its score
could not be computed, and their arithmetic holds; `headline.rests_on_floor` is
0.05 or the half-width of the skill interval, the larger, at least 0.05 where
there is no interval, and null with no skill; `headline.rests_on` is the columns
whose fall is at least the floor, at most five, largest first, ties in file order;
one `near_perfect_headline` record exactly when the skill is 0.98 or more; an
exact copy of a feature shares its fall; and one `holdout_far_above_quarter`
record exactly when a time-ordered holdout holds half of the rows or more. The
shuffled numbers are not compared between the sides: they are of the fitted
models.

What is compared, and how (spec 2.4, 3, 5.2, 5.3, 11)
  exact     what the table and the arguments alone determine: equal on both sides,
            numbers within 1e-9. The named error; `input`, `task`, the row counts
            before the split, each column's role, reason and counts, the scheme,
            the configurations, the fixed texts, the warnings the table alone
            decides, `input.duplicate_rows` and one `fold_not_scored` record for
            each fold not scored (D72). Under time order the rows of the split are
            determined too, so the folds, the baselines, the scores, the chosen
            configuration and every prediction are compared as well, for every
            family: the rows of a fit are in time order, ties in file order (D74
            point 4), so a forest is compared like the rest. So is the largest
            row's share of the squared error, on the holdout and over the
            selection folds (D74 point 3).
  property  what depends on a choice the spec leaves open is not compared across the
            sides: each side alone is held to what the spec pins. The
            reconciliation rules of 5.3 recomputed here; a quarter rounded up;
            property 3 of step 7 with the units rebuilt here; the copies rule
            against the side's own count; the null patterns of 5.2; under the
            fast profile's cap with an entity column, that an entity of a quarter
            or more of the usable rows is kept (D72); under `duplicates="group"`,
            that no combination of the features that reached the models is on
            both sides (D69); that a column of dates written YYYYMMDD with a
            value that is no date is still excluded as a date (D72). Since the
            eighth pass (D73, D74): that an exact copy of the target is
            excluded by the leak screen, under every scheme; the largest row's
            share on the holdout recomputed from predictions.csv, its null
            pattern, the same keys in `cv` and in every record of `models`, and
            one `largest_row_dominates` record for each statement above one
            half; one `selection_not_made` record exactly when, under the
            grouped scheme, a fold has fewer than 50 training rows and fewer
            training rows than validation rows, and then the first fitted
            configuration chosen; under time order with no entity column both
            baselines recomputed from the source, the previous value being the
            mean target of the timestamp `gap + 1` distinct timestamps before;
            and the generator's identifier of whole numbers id-like by rule 8b
            of step 5 or by the clause D74 adds, and by nothing else.

Not compared, and printed by kind and count, no part of the verdict
  - which of two named errors is raised when a table earns both in steps 1 to 4
    (spec 2.4: open);
  - what depends on which rows a random or a grouped split took, or which the
    fast cap kept (spec 11: open): a leak decision at the margin, a fold with
    one class, a copies stop near the rule, an interval at 29 against 30 units;
  - how many records a warning has where spec 2.5 gives no count;
  - never: `meta.timing`, the wording of a message, `columns[].dtype`, the kind
    of a column that is no feature, the ends of an interval, a p-value.

Under the fast cap the output does not say which rows were left out. So where the
cap left rows out with no time column, the units of a grouped split and the
combinations on both sides are not counted, and that an entity of a quarter or
more was kept is shown by the row counts, the holdout rows and the warnings for
a large unit, which is nearly always enough and not always.

Three classes are counted, and printed with the count of each
  agrees        the table has no finding
  stated limit  a finding the spec states as a limit, with its decision number.
                None is known today; the list is `STATED_LIMITS`
  open          a finding that is owned and not yet built: `OPEN`, by name, each
                with its decision and the pass that builds it. None since the
                eighth pass: the list is empty, and its count is printed all
                the same, 0
A finding on neither list fails the tool. So does a raw exception from either
side. Each is printed with the call that makes its table again.

The first run listed two findings as open (D73). Neither is: the order of the
rows of a fit under time order is pinned (D74 point 4), and a column that
reproduces the target is always excluded (D73 point 2, D74 point 2), so an
exact copy of the target that stays a feature, on either side, fails the tool.

Everything this tool writes goes under verify/artifacts/generated/, which the
loop's snapshot of the judge leaves out. It writes nowhere else, and no cache
file. Each side runs in worker processes of its own, since both packages are
named `mlprofessor`; that is also why this is no test of the suite. The loop runs
it through tests/test_judge_tools.py once a pass's last task is reached (D72).
"""
from __future__ import annotations

import argparse
import calendar
import csv
import datetime
import decimal
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True

EXIT_OK, EXIT_DIFF, EXIT_HARNESS = 0, 1, 3
FAST_CAP = 2000
TOO_LARGE = 3.4028234663852886e+38
DEFAULT_TABLES, DEFAULT_SEED = 1000, 20261122
# The ninth pass's series (D77, D78): tables numbered from here on carry columns of
# its kinds. Every table below is as it was.
NINTH_FROM, DEFAULT_NINTH = 1_000_000, 200
REST_KEYS = ("shuffled_score", "shuffled_skill", "skill_fall")
REST_FALL, REST_MOST, NEAR_PERFECT, FAR_ABOVE = 0.05, 5, 0.98, 0.5
CHUNK = 6                       # tables handed to one worker process
CHUNK_SECONDS = 1500
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts" / "generated"    # the one place this tool writes
LIMITATION_COUNT = 9
# The date rule of step 5, rule 2a (D72): nine distinct dates for each distinct
# value that is no date, or five beside at most two.
DATES_PER_OTHER_VALUE, FEW_DATES, FEW_OTHER_VALUES = 9, 5, 2

# A finding the spec states as a limit: name -> the decision that states it. None
# is known today. A name is added here only with the number of its decision.
STATED_LIMITS = {}
# A finding that is owned and not yet built: name -> (decision, who builds it, what
# it is). A finding that is on neither list fails the tool. Empty since the eighth
# pass (D73, D74): the class is kept so that its count is always printed, 0.
OPEN = {}
# The second clause of the id-like rule for whole numbers (step 5; D74 point 5): more
# than this share of the values distinct, and the distinct values filling at least
# that share of their range, whatever the order of the rows.
ID_DISTINCT_SHARE, ID_RANGE_SHARE = 0.95, 0.5
# A `largest_row_dominates` record for each share above this (step 11; D74 point 3).
LARGEST_ROW_SHARE = 0.5
# Under the grouped scheme, when a fold has fewer training rows than this, and fewer
# than it has validation rows, no choice between configurations is made (step 9; D74
# point 6).
SELECTION_FLOOR = 50

OPENERS = ["called about", "asked for", "complained about", "praised", "returned", "queried",
           "cancelled", "renewed", "upgraded", "paused", "reported", "requested", "disputed",
           "confirmed", "missed", "declined", "accepted", "reviewed", "forgot", "changed"]
CLOSERS = ["the invoice", "a refund", "late delivery", "the new plan", "a faulty unit",
           "the price rise", "their password", "the mobile app", "a discount", "the contract",
           "billing dates", "a missing part", "the welcome pack", "an engineer visit",
           "the loyalty scheme", "account access", "a duplicate charge", "the help pages",
           "opening hours", "a replacement"]


# ------------------------------------------------------------- the generator --

def _pick(r, weighted):
    total = sum(w for _, w in weighted)
    x = r.uniform(0, total)
    for value, w in weighted:
        x -= w
        if x <= 0:
            return value
    return weighted[-1][0]


def make_table(seed, number):
    """Table `number` of the run `seed`: the frame, the arguments, and what the
    generator knows about it (which rows have no target, the order in time, the
    entity of each row). The same two numbers always give the same table."""
    import numpy as np
    import pandas as pd

    r = random.Random(f"generated-tables:{seed}:{number}")
    g = np.random.default_rng([int(seed), int(number), 811])
    notes = []
    kind = "csv" if r.random() < 0.5 else "dataframe"
    size = _pick(r, [("tiny", 7), ("small", 46), ("medium", 26), ("large", 12), ("over_cap", 9)])
    n = {"tiny": r.randint(96, 125), "small": r.randint(126, 260), "medium": r.randint(261, 700),
         "large": r.randint(701, 1600), "over_cap": r.randint(2050, 2600)}[size]
    statement = _pick(r, [("none", 32), ("entity", 19), ("time", 24), ("time_entity", 15),
                          ("unsaid_both", 2), ("unsaid_time", 2), ("unsaid_entity", 2),
                          ("time_unsaid_entity", 2), ("unsaid_time_entity", 2)])
    if statement == "entity" and size != "over_cap" and r.random() < 0.15:
        # more tables past the cap with an entity column: where the cap of whole
        # entities decides something (D70, D72)
        size, n = "over_cap", r.randint(2050, 2600)
    declares_time = statement in ("time", "time_entity", "time_unsaid_entity")
    declares_entity = statement in ("entity", "time_entity", "unsaid_time_entity")
    wants_time = declares_time or r.random() < 0.2
    wants_entity = declares_entity or r.random() < 0.15
    task_kind = _pick(r, [("clf_text", 29), ("clf_int", 13), ("clf_bool", 5), ("reg", 40),
                          ("reg_whole", 6), ("ambiguous_int", 3), ("three_classes", 2),
                          ("constant", 2)])
    few_valued = (not wants_time) and r.random() < 0.14
    notes.append(f"{size}; statement {statement}; target {task_kind}"
                 + ("; few-valued columns only" if few_valued else ""))

    # ---- which rows there are: timestamps and entities
    t, who = None, None
    if wants_time:
        if wants_entity:
            entities = r.choice([3, 5, 8, 12, 25, 60, 300])
            stamps = max(r.choice([4, 8, 8, 12]), n // entities)
            irregular = r.random() < 0.35
            pairs = [(e, k) for e in range(entities) for k in range(stamps)
                     if not irregular or r.random() < 0.75]
            who, t = [e for e, _ in pairs], [k for _, k in pairs]
            notes.append(f"panel of {entities} entities on {stamps} timestamps"
                         + (", irregular" if irregular else ""))
        elif r.random() < 0.7:
            t = list(range(n))
            notes.append("one row for each timestamp")
        else:
            t, k = [], 0
            for _ in range(n):
                t.append(k)
                k += r.random() > 0.25
            notes.append("some timestamps hold several rows")
    elif wants_entity:
        # past the cap one large entity is the usual shape: the cap must keep one of
        # a quarter or more of the rows, and cannot keep one larger than itself (D72)
        shape = _pick(r, [("equal", 25), ("one_large", 50), ("own", 5), ("few", 5),
                          ("long_tail", 15)] if size == "over_cap" else
                      [("equal", 45), ("one_large", 22), ("own", 8), ("few", 10),
                       ("long_tail", 15)])
        if shape == "equal":
            per = r.choice([2, 3, 5, 8])
            who = [j // per for j in range(n)]
        elif shape == "one_large":
            big = int(n * r.choice([0.25, 0.3, 0.45, 0.6, 0.8, 0.92]))
            if size == "over_cap" and r.random() < 0.5:
                big = r.randint(FAST_CAP + 1, n - 30)       # larger than the cap itself
            per = r.choice([1, 2, 3])
            who = [0] * big + [1 + j // per for j in range(n - big)]
            notes.append(f"one entity of {big} of the {n} rows")
        elif shape == "own":
            who = list(range(n))
        elif shape == "few":
            count = r.randint(3, 8)
            who = [j % count for j in range(n)]
        else:
            who, e = [], 0
            while len(who) < n:
                who += [e] * max(1, int(r.lognormvariate(1.0, 1.0)))
                e += 1
            who = who[:n]
        notes.append(f"entities: {shape}, {len(set(who))} of them")
    if t is not None:
        n = len(t)
    t = None if t is None else np.asarray(t, dtype="int64")
    who = None if who is None else np.asarray(who, dtype="int64")

    # ---- latent values and the target
    f1, f2 = g.normal(size=n), g.normal(size=n)
    levels = ["A", "B", "C", "D"][: r.choice([2, 3, 4])]
    seg = g.choice(levels, size=n)
    effect = np.array([{"A": -1.0, "B": 0.0, "C": 0.5, "D": 1.2}[s] for s in seg])
    signal = r.random() < 0.7
    if who is not None and t is None and r.random() < 0.6:
        centre = g.normal(size=int(who.max()) + 1)
        f1 = centre[who] + 0.1 * f1          # the measure says which entity a row is of
        notes.append("f1 identifies the entity")
    z = 1.6 * f1 - 1.2 * f2 + effect if signal else g.normal(size=n)
    level_prev = None
    share = _pick(r, [(0.5, 60), (0.3, 20), (0.1, 12), (0.04, 8)])
    y01, y_clean = None, None
    if task_kind.startswith("clf"):
        noisy = z + g.logistic(size=n)
        y01 = (noisy > np.quantile(noisy, 1 - share)).astype(int)
        if who is not None and t is None and r.random() < 0.3:
            y01 = (g.random(int(who.max()) + 1) < share).astype(int)[who]   # one label an entity
            notes.append("one label for each entity")
    elif task_kind in ("reg", "reg_whole"):
        if t is not None and r.random() < 0.5:
            order = np.lexsort((t, who if who is not None else np.zeros(n, dtype="int64")))
            walk, before, last, cur = np.zeros(n), np.zeros(n), None, 0.0
            steps, starts = g.normal(size=n), g.normal(0, 20, size=n)
            for j in order:
                e = int(who[j]) if who is not None else 0
                if e != last:
                    cur, last = float(starts[j]), e
                before[j] = cur
                cur += float(steps[j])
                walk[j] = cur
            y_clean, level_prev = np.round(walk, 4), np.round(before, 4)
            notes.append("the target is a random walk" + (" for each entity" if who is not None
                                                          else ""))
        else:
            y_clean = np.round(2.0 * z + g.normal(size=n), 4)
        if task_kind == "reg_whole":
            y_clean = np.round(8 * y_clean)
    elif task_kind == "ambiguous_int":
        y_clean = np.clip(np.round(4 + 1.2 * f1 + g.normal(size=n)), 1, 7)
    miss_rate = _pick(r, [(0.0, 60), (0.03, 30), (0.10, 10)])
    y_missing = g.random(n) < miss_rate

    def text_series(values):
        """A text column: generic objects, or the default string type."""
        if r.random() < 0.5:
            return pd.Series(list(values), dtype=object)
        return pd.Series(list(values), dtype="str")

    target_name = "y"
    if task_kind == "clf_text":
        pair = r.choice([("no", "yes"), ("N", "Y"), ("low", "high"), ("b", "a")])
        cells = [pair[v] for v in y01]
        if r.random() < 0.25:
            cells = [c + " " if g.random() < 0.05 else (" " + c if g.random() < 0.03 else c)
                     for c in cells]
            notes.append("white space round some target values")
        blank = r.choice([None, None, "", "  "])
        target = text_series([blank if gone else c for c, gone in zip(cells, y_missing)])
    elif task_kind == "clf_int":
        if y_missing.any():
            target = pd.Series(np.where(y_missing, np.nan, y01.astype(float)))
        else:
            target = pd.Series(y01.astype("int64"))
    elif task_kind == "clf_bool":
        if y_missing.any():
            target = pd.Series([None if gone else bool(v) for v, gone in zip(y01, y_missing)],
                               dtype=object)
        else:
            target = pd.Series(y01.astype(bool))
    elif task_kind in ("reg", "reg_whole", "ambiguous_int"):
        values = y_clean.astype(float).copy()
        extreme = r.random()
        if task_kind == "reg" and extreme < 0.02:
            values[int(g.integers(n))] = 1.7976931348623157e308
            notes.append("one target cell of 1.7976931348623157e308")
        elif task_kind == "reg" and extreme < 0.03:
            values = values * 1e160
            notes.append("target times 1e160")
        elif task_kind == "reg" and extreme < 0.08:
            hit = g.random(n) < 0.02
            values[hit] = np.inf
            y_missing = y_missing | hit
            notes.append("infinite target cells")
        values[y_missing & ~np.isinf(values)] = np.nan
        if kind == "dataframe" and task_kind == "reg" and r.random() < 0.06:
            target = pd.Series([None if not math.isfinite(v) else decimal.Decimal(repr(float(v)))
                                for v in values], dtype=object)
            y_missing = ~np.isfinite(values)
            notes.append("target held as Decimal objects")
        elif task_kind != "reg" and not y_missing.any():
            target = pd.Series(values.astype("int64"))
        else:
            target = pd.Series(values)
    elif task_kind == "three_classes":
        target = text_series([None if gone else c for c, gone in
                              zip(g.choice(["red", "green", "blue"], size=n), y_missing)])
    else:
        target = text_series([None if gone else "yes" for gone in y_missing]) \
            if r.random() < 0.5 else pd.Series(np.where(y_missing, np.nan, 5.0))

    cols = {}
    if few_valued:
        for k in range(r.choice([2, 3])):
            width = r.choice([3, 4, 5])
            cols[f"c{k + 1}"] = text_series(g.choice([f"v{j}" for j in range(width)], size=n))
    else:
        cols["f1"] = pd.Series(np.round(f1, 4))
        rate = r.choice([0.0, 0.03, 0.08])
        cols["f2"] = pd.Series(np.where(g.random(n) < rate, np.nan, np.round(f2, 4)))
    if level_prev is not None and r.random() < 0.6:
        cols["level_prev"] = pd.Series(level_prev)
        notes.append("level_prev: the level before, of the same series")

    def maybe(p):
        return (not few_valued) and r.random() < p

    if maybe(0.6):
        rate = r.choice([0.0, 0.0, 0.04])
        cells = [None if g.random() < rate else (s + " " if g.random() < 0.03 else s) for s in seg]
        cols["seg"] = text_series(cells)
        if kind == "dataframe" and r.random() < 0.15:
            cols["seg"] = pd.Series(pd.Categorical(cells))
    if maybe(0.3):
        counts = g.integers(0, 500, size=n)
        cols["count"] = pd.Series(np.where(g.random(n) < 0.04, np.nan, counts)) \
            if r.random() < 0.4 else pd.Series(counts)
    if maybe(0.2):
        flags = g.random(n) < 0.4
        if kind == "dataframe" and r.random() < 0.3:
            cols["flag"] = pd.Series([pd.NA if g.random() < 0.05 else bool(v) for v in flags],
                                     dtype="boolean")
        else:
            cols["flag"] = pd.Series(flags)
    if maybe(0.15):
        cols["const"] = pd.Series([7] * n) if r.random() < 0.5 else text_series(["GB"] * n)
    if maybe(0.15):
        form = r.choice(["consecutive", "increasing", "text"])
        if form == "consecutive":
            cols["row_id"] = pd.Series(100001 + g.permutation(n))
        elif form == "increasing":
            cols["row_id"] = pd.Series(np.cumsum(g.integers(1, 9, size=n)))
        else:
            cols["row_id"] = text_series([f"ID-{j:05d}" for j in g.permutation(n)])
        notes.append(f"row_id: an identifier ({form})")
    if maybe(0.12):
        cols["notes"] = text_series([f"{r.choice(OPENERS)} {r.choice(CLOSERS)} {r.randint(1, 60)}"
                                     for _ in range(n)])
    if maybe(0.12):
        days = [datetime.date(2019, 1, 1) + datetime.timedelta(days=int(d))
                for d in g.integers(0, 1500, size=n)]
        if kind == "dataframe" and r.random() < 0.4:
            cols["signup"] = pd.Series(pd.to_datetime([d.isoformat() for d in days]))
        else:
            cols["signup"] = text_series([d.isoformat() for d in days])
    if maybe(0.12):
        days = [datetime.date(2015, 1, 1) + datetime.timedelta(days=int(d))
                for d in g.integers(0, 3000, size=n)]
        numbers = [int(d.strftime("%Y%m%d")) for d in days]
        form = r.choice(["int", "int", "text", "float_gaps"])
        # a quarter hold a value that is no date: an open-ended date, a zero (D69, D72)
        sentinel = _pick(r, [(None, 75), ("one sentinel cell", 14),
                             ("a sentinel in a third of the rows", 7),
                             ("two sentinels, one cell each", 4)])
        if sentinel == "one sentinel cell":
            numbers[int(g.integers(n))] = r.choice([99991231, 0])
        elif sentinel == "a sentinel in a third of the rows":
            value = r.choice([99991231, 0])
            for j in np.flatnonzero(g.random(n) < 0.3):
                numbers[int(j)] = value
        elif sentinel:
            first, second = g.choice(n, size=2, replace=False)
            numbers[int(first)], numbers[int(second)] = 99991231, 0
        if form == "int":
            cols["daynum"] = pd.Series(numbers, dtype="int64")
        elif form == "text":
            cols["daynum"] = text_series([f"{v:08d}" for v in numbers])
        else:
            cols["daynum"] = pd.Series([np.nan if g.random() < 0.05 else float(v)
                                        for v in numbers])
        notes.append(f"daynum: dates written YYYYMMDD ({form})"
                     + (f", {sentinel}" if sentinel else ""))
    if maybe(0.1):
        amounts = g.lognormal(7.0, 1.2, size=n)
        if r.random() < 0.6:
            cols["revenue"] = text_series([f"{a:,.2f}" for a in amounts])
        else:
            cells = [f"{min(a, 999.0):.2f}" for a in amounts]
            cells[int(g.integers(n))] = "1,200.50"
            cols["revenue"] = text_series(cells)
    if maybe(0.12):
        cells = [f"{v:.2f}" for v in g.normal(50, 10, size=n)]
        for j in g.choice(n, size=3, replace=False):
            cells[int(j)] = r.choice(["abc", "tbc", "12 units", "-"])
        cols["spend"] = text_series(cells)
    if maybe(0.08):
        cols["empty"] = pd.Series([np.nan] * n)
    if maybe(0.1):
        values = g.normal(size=n)
        values[g.random(n) < 0.03] = np.inf
        values[g.random(n) < 0.02] = 1e39
        values[g.random(n) < 0.01] = -np.inf
        cols["big"] = pd.Series(values)
    copy_of = y01 if y01 is not None else y_clean
    # beside few-valued columns, a two-valued copy of the target: with
    # `duplicates="group"` the leak screen removes it after the units were formed (D69)
    leak_in_few = few_valued and y01 is not None and r.random() < 0.4
    if leak_in_few:
        cols["y_copy"] = text_series([("open", "closed")[v] for v in y01])
        notes.append("y_copy: a two-valued copy of the target beside few-valued columns")
    elif copy_of is not None and r.random() < 0.1:
        if y01 is not None:
            cols["y_copy"] = text_series([("A", "B")[v] for v in y01]) if r.random() < 0.6 \
                else pd.Series(y01.astype("int64"))
        else:
            cols["y_copy"] = pd.Series(np.round(3.7 * y_clean + 12, 4))
        notes.append("y_copy: a copy of the target")
    if copy_of is not None and maybe(0.08):
        if y01 is not None:
            flip = np.zeros(n, dtype=bool)
            flip[g.choice(n, size=max(1, round(0.02 * n)), replace=False)] = True
            cols["y_near"] = pd.Series(np.where(flip, 1 - y01, y01).astype("int64"))
        else:
            cols["y_near"] = pd.Series(np.round(y_clean + g.normal(0, 0.2 * y_clean.std() + 1e-9,
                                                                   size=n), 4))
        notes.append("y_near: a near-copy of the target")
    if maybe(0.15):
        cols["strong"] = pd.Series(np.round(z + g.normal(0, 0.5, size=n), 4))
    if y01 is not None and maybe(0.06):
        cols["after"] = pd.Series(np.where(y01 == 1, np.round(g.normal(1000, 200, size=n), 2),
                                           np.nan))
        notes.append("after: filled in for one class only")
    if maybe(0.08):
        cols["dec"] = pd.Series([None if g.random() < 0.04
                                 else decimal.Decimal(f"{v:.2f}") for v in g.normal(20, 5, size=n)],
                                dtype=object)
    if maybe(0.06):
        cols["optint"] = pd.Series([None if g.random() < 0.06 else int(v)
                                    for v in g.integers(0, 40, size=n)], dtype=object)
    if maybe(0.08):
        form = r.choice(["complex", "timedelta", "mixed", "list", "bytes"])
        if form == "complex":
            cols["odd"] = pd.Series(g.normal(size=n) + 1j * g.normal(size=n))
        elif form == "timedelta":
            cols["odd"] = pd.Series(pd.to_timedelta(g.integers(0, 9000, size=n), unit="s"))
        elif form == "mixed":
            cols["odd"] = pd.Series([("x" if j % 3 else 3) for j in range(n)], dtype=object)
        elif form == "list":
            cols["odd"] = pd.Series([[int(v)] for v in g.integers(0, 5, size=n)], dtype=object)
        else:
            cols["odd"] = pd.Series([bytes([65 + int(v)]) for v in g.integers(0, 4, size=n)],
                                    dtype=object)
        notes.append(f"odd: a column of {form}")
    if maybe(0.03):
        cols["huge"] = pd.Series([10 ** 25 + int(v) for v in g.integers(0, 50, size=n)],
                                 dtype=object)
    if "seg" in cols and maybe(0.03):
        cols["seg=A"] = pd.Series(np.round(g.normal(size=n), 4))
        notes.append("a numeric column named seg=A beside seg")
    if number >= NINTH_FROM:
        # the ninth pass's kinds, from streams of their own
        r9 = random.Random(f"generated-tables:ninth:{seed}:{number}")
        g9 = np.random.default_rng([int(seed), int(number), 919])
        values9 = y_clean if y_clean is not None else (
            y01.astype("float64") if y01 is not None else None)
        if r9.random() < 0.5:
            if r9.random() < 0.5:
                cols["tied"] = pd.Series(np.round(g9.normal(size=n), 1))
                notes.append("tied: noise rounded to one decimal place")
            else:
                cols["tied"] = pd.Series(g9.integers(1, 41, size=n).astype("int64"))
                notes.append("tied: noise, a whole number from 1 to 40")
        if values9 is not None and r9.random() < 0.4:
            if y01 is not None:
                q = r9.choice([10, 40])
                labels9, half = g9.permutation(q), q // 2
                code = np.where(y01 == 1, labels9[half + g9.integers(0, q - half, size=n)],
                                labels9[g9.integers(0, half, size=n)])
            else:
                q = r9.choice([12, 24, 30, 40])
                ranks = np.argsort(np.argsort(values9, kind="stable"), kind="stable")
                code = g9.permutation(q)[np.minimum(q - 1, ranks * q // n)]
            if r9.random() < 0.5:
                cols["y_code"] = text_series([f"code_{int(v):03d}" for v in code])
            else:
                cols["y_code"] = pd.Series((100 + code).astype("int64"))
            notes.append(f"y_code: the target as a code of {q} levels, the labels in random "
                         "order")
        if values9 is not None and r9.random() < 0.25:
            noise9 = g9.normal(0, float(np.std(values9)) or 1.0, size=n)
            cols["pair_a"] = pd.Series(np.round(values9 + noise9, 6))
            cols["pair_b"] = pd.Series(np.round(noise9, 6))
            notes.append("pair_a, pair_b: the target plus a noise, and that noise")
        if "f1" in cols and r9.random() < 0.3:
            cols["f1_again"] = cols["f1"].copy()
            notes.append("f1_again: an exact copy of f1")
        elif "f1" in cols and r9.random() < 0.3:
            q = r9.choice([10, 40])
            held9 = cols["f1"].to_numpy(dtype="float64")
            low, high = np.nanmin(held9), np.nanmax(held9)
            width = (high - low) / q or 1.0
            cols["f1"] = pd.Series(np.round(low + (np.minimum(q - 1, np.floor((held9 - low)
                                                                              / width)) + 0.5)
                                            * width, 6))
            notes.append(f"f1 put into {q} bands of equal width")

    # ---- the time column and the entity column, as the table holds them
    oracle = {"time_form": None, "time_sentinel": False}
    t_missing = np.zeros(n, dtype=bool)
    if t is not None:
        form = _pick(r, [("iso_date", 30), ("iso_datetime", 12), ("datetime", 14), ("number", 14),
                         ("yyyymmdd_int", 12), ("yyyymmdd_text", 8), ("dayfirst", 5),
                         ("big_int", 5)])
        if form == "datetime" and kind != "dataframe":
            form = "iso_date"
        start = datetime.date(2015, 1, 1) + datetime.timedelta(days=r.randint(0, 2000))
        days = [start + datetime.timedelta(days=int(k)) for k in t]
        if form != "big_int":
            t_missing = g.random(n) < r.choice([0.0, 0.0, 0.03])
        if form == "iso_date":
            when = text_series([None if gone else d.isoformat() for d, gone in zip(days, t_missing)])
        elif form == "iso_datetime":
            tail = r.choice(["T09:15:00", " 09:15:00", "T09:15:00.5", "T09:15:00.123456"])
            when = text_series([None if gone else d.isoformat() + tail
                                for d, gone in zip(days, t_missing)])
        elif form == "datetime":
            when = pd.Series(pd.to_datetime([None if gone else d.isoformat()
                                             for d, gone in zip(days, t_missing)]))
        elif form == "number":
            step, base = r.choice([1, 10, 0.5]), r.randint(0, 1000)
            values = base + step * t.astype(float)
            when = pd.Series(np.where(t_missing, np.nan, values)) if t_missing.any() or step == 0.5 \
                else pd.Series(values.astype("int64"))
        elif form == "yyyymmdd_int":
            numbers = [int(d.strftime("%Y%m%d")) for d in days]
            when = pd.Series([np.nan if gone else float(v) for v, gone in zip(numbers, t_missing)]) \
                if t_missing.any() else pd.Series(numbers, dtype="int64")
        elif form == "yyyymmdd_text":
            when = text_series([None if gone else d.strftime("%Y%m%d")
                                for d, gone in zip(days, t_missing)])
        elif form == "dayfirst":
            when = text_series([None if gone else d.strftime("%d/%m/%Y")
                                for d, gone in zip(days, t_missing)])
        else:
            when = pd.Series(1_700_000_000_000_000_000 + 100 * t, dtype="int64")
        # Dates written YYYYMMDD with one value that is no date. As the time column
        # that stops the run (D69, D72). Planted only where the rule is sure to hold:
        # twelve distinct days or more beside the one value.
        if form in ("yyyymmdd_int", "yyyymmdd_text") and len(set(t.tolist())) >= 12 \
                and r.random() < 0.12:
            where = int(g.choice(np.flatnonzero(~t_missing)))
            value = r.choice([99991231, 0])
            when = when.copy()
            if form == "yyyymmdd_text":
                when = pd.Series([f"{value:08d}" if j == where else v
                                  for j, v in enumerate(when)], dtype=when.dtype)
            else:
                when.iloc[where] = value
            oracle["time_sentinel"] = True
            notes.append(f"when: one cell of {value}, which is no date")
        cols["when"] = when
        oracle["time_form"] = form
        notes.append(f"when: {form}" + (", some cells empty" if t_missing.any() else ""))
    who_empty = np.zeros(n, dtype=bool)
    if who is not None:
        form = _pick(r, [("text", 75), ("int", 18), ("float", 7)])
        if r.random() < 0.04:
            who_empty[int(g.integers(n))] = True
            notes.append("one entity cell empty")
        if form == "text":
            cells = [None if gone else (f"e{e:04d} " if g.random() < 0.02 else f"e{e:04d}")
                     for e, gone in zip(who, who_empty)]
            cols["who"] = text_series(cells)
        elif form == "int" and not who_empty.any():
            cols["who"] = pd.Series(who + 100)
        else:
            cols["who"] = pd.Series(np.where(who_empty, np.nan, who.astype(float) + 100))

    # ---- assemble, repeat rows, shuffle
    names = list(cols)
    r.shuffle(names)
    names.insert(r.randint(0, len(names)), target_name)
    cols[target_name] = target
    frame = pd.DataFrame({name: cols[name].reset_index(drop=True) for name in names})
    repeats = _pick(r, [("none", 70), ("few", 15), ("heavy", 15)])
    extra = np.array([], dtype="int64")
    if repeats == "few":
        extra = g.choice(n, size=r.randint(1, 5), replace=False)
    elif repeats == "heavy":
        extra = np.arange(n) if r.random() < 0.5 else g.choice(n, size=int(0.4 * n), replace=False)
    if repeats != "none":
        notes.append(f"repeated rows: {repeats} ({len(extra)})")
    rows = np.concatenate([np.arange(n), extra])
    rows = rows[g.permutation(len(rows))]
    frame = frame.iloc[rows].reset_index(drop=True)
    y_missing, t_missing, who_empty = y_missing[rows], t_missing[rows], who_empty[rows]
    t = None if t is None else t[rows]
    who = None if who is None else who[rows]
    if r.random() < 0.03:
        frame = frame.rename(columns={names[0]: ""})
        notes.append(f"a blank column name (was {names[0]})")
    elif r.random() < 0.015 and len(names) > 2:
        twice = [nm for nm in names if nm not in (target_name, "when", "who")][:2]
        if len(twice) == 2:
            frame = frame.rename(columns={twice[1]: twice[0]})
            notes.append(f"two columns named {twice[0]}")

    # ---- the arguments
    args = {"target": target_name, "n_jobs": 1}
    args["time_col"] = "when" if declares_time else (
        None if statement in ("unsaid_both", "unsaid_time", "unsaid_time_entity") else False)
    args["group_col"] = "who" if declares_entity else (
        None if statement in ("unsaid_both", "unsaid_entity", "time_unsaid_entity") else False)
    natural = "classification" if task_kind.startswith("clf") else "regression"
    declared = r.random()
    args["task"] = "auto" if declared < 0.7 else (
        natural if declared < 0.97 else ("regression" if natural == "classification"
                                         else "classification"))
    args["gap"] = _pick(r, [(0, 70), (1, 12), (2, 8), (5, 6), (40, 4)]) if declares_time else (
        2 if r.random() < 0.01 else 0)
    args["profile"] = "full" if len(frame) <= 180 and frame.shape[1] <= 8 and r.random() < 0.3 \
        else "fast"
    args["seed"] = r.choice([0, 0, 0, 1, 7, 123456, 4294967295])
    args["duplicates"] = _pick(r, [("error", 55), ("keep", 22), ("group", 23)])
    if leak_in_few and r.random() < 0.7:
        args["duplicates"] = "group"
    blank_lines =sorted(int(v) for v in g.choice(len(frame), size=r.randint(1, 3), replace=False)) \
        if kind == "csv" and r.random() < 0.15 else []
    if blank_lines:
        notes.append(f"{len(blank_lines)} blank line(s) in the file")
    oracle.update(n=len(frame), y_missing=y_missing, t_missing=t_missing, who_empty=who_empty,
                  t=t, who=who, task_kind=task_kind, declares_time=declares_time,
                  declares_entity=declares_entity, natural=natural)
    return {"number": number, "kind": kind, "frame": frame, "args": args, "oracle": oracle,
            "notes": notes, "blank_lines": blank_lines}


def write_csv(case, path):
    """The table as a CSV file: pandas' own writing, with blank lines put in after the
    data rows the generator chose (a blank line is not a row: spec section 6)."""
    text = case["frame"].to_csv(index=False, lineterminator="\n")
    if case["blank_lines"]:
        lines = text.split("\n")
        for row in reversed(case["blank_lines"]):
            lines.insert(row + 2, "")            # after the header and data row `row`
        text = "\n".join(lines)
    Path(path).write_bytes(text.encode("utf-8"))


def broken_rules(case):
    """The errors of steps 1 to 4 that the generator knows this table earns. When a
    table earns several, which is raised is open (spec 2.4)."""
    import numpy as np
    o, args, frame = case["oracle"], case["args"], case["frame"]
    names = [str(c) for c in frame.columns]
    earned = set()
    if len(set(names)) != len(names):
        earned.add("duplicate_columns")
    wanted = [args["target"]] + [args[k] for k in ("time_col", "group_col")
                                 if isinstance(args[k], str)]
    if any(name not in names for name in wanted):
        earned.add("column_not_found")
    usable = ~o["y_missing"] & (~o["t_missing"] if o["declares_time"] else True)
    if int(np.sum(usable)) < 100:
        earned.add("too_few_rows")
    if o["declares_entity"] and bool(np.any(o["who_empty"] & usable)):
        earned.add("group_column_invalid")
    if o["declares_time"] and (o["time_form"] == "dayfirst" or o["time_sentinel"]):
        earned.add("time_column_invalid")
    kind, asked = o["task_kind"], args["task"]
    if kind == "constant":
        earned.add("target_constant")
    elif kind == "three_classes":
        earned.add("task_ambiguous" if asked == "auto" else "task_unsupported")
    elif kind == "ambiguous_int":
        if asked == "auto":
            earned.add("task_ambiguous")
        elif asked == "classification":
            earned.add("task_unsupported")
    elif kind in ("clf_text", "clf_bool") and asked == "regression":
        earned.add("task_unsupported")
    elif kind in ("reg", "reg_whole") and asked == "classification":
        earned.add("task_unsupported")
    return earned


def describe(case):
    a = case["args"]
    said = ", ".join(f"{k}={a[k]!r}" for k in ("task", "time_col", "group_col", "gap", "profile",
                                                 "seed", "duplicates"))
    return (f"table {case['number']}: {case['kind']}, {len(case['frame'])} rows x "
            f"{case['frame'].shape[1]} columns {list(map(str, case['frame'].columns))}; {said}; "
            + "; ".join(case["notes"]))


def csv_path(data_dir, number):
    return Path(data_dir) / f"table_{number:05d}.csv"


# ---------------------------------------------------------------- the worker --

def worker(parent, seed, numbers, out_root, data_dir, side):
    """Run the tables `numbers` through the package under `parent`; one JSON file each."""
    sys.path.insert(0, str(parent))
    import mlprofessor
    from mlprofessor import errors
    home = Path(mlprofessor.__file__).resolve().parent
    if home.parent != Path(parent).resolve():
        raise SystemExit(f"imported mlprofessor from {home}, not from {parent}")
    for number in numbers:
        case = make_table(seed, number)
        data = str(csv_path(data_dir, number)) if case["kind"] == "csv" else case["frame"]
        out = Path(out_root) / side / f"{number:05d}"
        if out.exists():
            shutil.rmtree(out)
        record, started = {"number": number}, time.perf_counter()
        try:
            mlprofessor.profile(data, out_dir=str(out), **case["args"])
        except errors.ProfileError as exc:
            record.update(kind="error", code=str(exc.code), error_class=type(exc).__name__,
                          message=str(exc))
        except BaseException as exc:      # noqa: BLE001  (a raw exception is the finding)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            record.update(kind="raw", error_class=type(exc).__name__, message=str(exc)[:600],
                          trace="".join(traceback.format_exception(exc))[-2500:])
        else:
            record["kind"] = "result"
        record["seconds"] = round(time.perf_counter() - started, 3)
        record["written"] = sorted(p.name for p in out.iterdir()) if out.is_dir() else []
        if record["kind"] == "result":
            bad = []
            try:
                text = (out / "profile.json").read_text(encoding="utf-8")
                record["doc"] = json.loads(text, parse_constant=lambda tok: bad.append(tok) or tok)
                with open(out / "predictions.csv", newline="", encoding="utf-8") as fh:
                    table = [row for row in csv.reader(fh) if row]
                record["predictions_header"] = table[0] if table else []
                record["predictions"] = table[1:]
            except (OSError, ValueError) as exc:
                record.update(kind="raw", error_class="unreadable output",
                              message=f"{type(exc).__name__}: {exc}", trace="")
            record["bad_numbers"] = bad
        target = Path(out_root) / side / f"{number:05d}.json"
        target.write_text(json.dumps(record), encoding="utf-8")


# ------------------------------------------------- the judge's own arithmetic --

def close(a, b, tolerance=1e-9):
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def first_difference(a, b, path=""):
    """First difference between two JSON values, or None. Numbers within 1e-9."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}.{key} is present on one side only"
            found = first_difference(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} entries against {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    if is_number(a) and is_number(b):
        return None if close(a, b) else f"{path}: {a!r} != {b!r}"
    return None if a == b and type(a) is type(b) else f"{path}: {a!r} != {b!r}"


def same_cell(a, b):
    a, b = str(a).strip(), str(b).strip()
    if a == b:
        return True
    try:
        return close(float(a), float(b), 1e-12)
    except ValueError:
        return False


def auc(labels, scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks, i = [0.0] * len(scores), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    return (sum(rk for rk, y in zip(ranks, labels) if y) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def at(doc, path):
    node = doc
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return ("<absent>",)
        node = node[part]
    return node


def row_counts(doc):
    return {rec.get("item"): rec.get("count") for rec in doc.get("rows", [])
            if isinstance(rec, dict)}


def warning_counts(doc):
    counts = {}
    for w in doc.get("warnings", []):
        key = (w.get("code"), w.get("column"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def judge_frame(case, data_dir):
    """The table as the judge reads it: the frame, or the CSV file read as spec 6 pins."""
    import pandas as pd
    if case["kind"] == "csv":
        return pd.read_csv(csv_path(data_dir, case["number"]), low_memory=False)
    return case["frame"]


def source_cells(case, data_dir, name):
    """The cells of one column as the source holds them, for y_true (5.3, rule 3)."""
    if case["kind"] == "csv":
        with open(csv_path(data_dir, case["number"]), newline="", encoding="utf-8") as fh:
            table = [row for row in csv.reader(fh) if row]
        position = table[0].index(name)
        return [row[position] for row in table[1:]]
    return list(case["frame"][name])


def feature_keys(frame, doc, reached_models=False):
    """One key per row: equal for two rows that agree in every feature column of step 5
    (a column that is a feature, or was one until the leak screen excluded it), compared
    as spec step 8a compares values. With `reached_models`, the columns that are
    features in the end only: what a model was handed. The judge's own reading of
    the table."""
    import numpy as np
    import pandas as pd
    parts = []
    for rec in doc["columns"]:
        if not (rec["role"] == "feature"
                or (rec.get("reason") == "suspected_leak" and not reached_models)):
            continue
        series = frame.iloc[:, rec["position"]]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = pd.Series([v if pd.isna(v) else str(v) for v in series], dtype=object)
        cells = list(series)
        present = [v for v in cells if not _empty(v)]
        texts = [v for v in present if isinstance(v, str)]
        numeric = None
        said = rec.get("kind") if rec["role"] == "feature" else None
        if said == "categorical":
            pass
        elif said == "numeric":
            numeric = [float("nan") if _empty(v) else
                       (_text_float(v) if isinstance(v, str) else _to_float(v)) for v in cells]
        elif not texts:
            numeric = [float("nan") if _empty(v) else _to_float(v) for v in cells]
        elif len(texts) == len(present):
            parsed = pd.to_numeric(pd.Series([v.strip() for v in present], dtype=object),
                                   errors="coerce")
            if parsed.notna().mean() >= 0.9:
                numeric = [float("nan") if _empty(v) else _text_float(v) for v in cells]
        if numeric is not None:
            values = np.array(numeric, dtype="float64")
            values[~np.isfinite(values) | (np.abs(values) > TOO_LARGE)] = np.nan
            parts.append(["nan" if math.isnan(v) else repr(v + 0.0) for v in values])
        else:
            parts.append(["(missing)" if _empty(v) else str(v).strip() for v in cells])
    return [tuple(part[i] for part in parts) for i in range(len(frame))]


def _empty(value):
    """Whether a cell is empty: None, NaN, pd.NA, NaT, a decimal not-a-number."""
    import pandas as pd
    if isinstance(value, decimal.Decimal):
        return value.is_nan()
    if isinstance(value, (list, tuple, dict, set, bytes)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _text_float(text):
    try:
        return float(text.strip())
    except ValueError:
        return float("nan")


def reads_as_date(value):
    """Whether one cell reads as a calendar date written YYYYMMDD, years 1900 to 2100
    (spec step 5, rule 2a): a whole number, or text of exactly eight digits."""
    if isinstance(value, str):
        text = value.strip()
        if not (len(text) == 8 and text.isascii() and text.isdigit()):
            return False
        number = int(text)
    elif isinstance(value, bool):
        return False
    elif isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not math.isfinite(value) or value != math.floor(value):
            return False
        number = int(value)
    else:
        return False
    year, month, day = number // 10000, number // 100 % 100, number % 100
    if not (1900 <= year <= 2100 and 1 <= month <= 12 and day >= 1):
        return False
    return day <= calendar.monthrange(year, month)[1]


def is_date_column(cells):
    """Rule 2a as D72 pins it, on the cells given (those of the usable rows): by the
    distinct values, at least two that read as dates, and for each one that does
    not at least nine that do, or at least five beside at most two."""
    distinct = set()
    for value in cells:
        if _empty(value):
            continue
        if hasattr(value, "item"):
            value = value.item()
        distinct.add(value.strip() if isinstance(value, str) else value)
    dates = sum(1 for value in distinct if reads_as_date(value))
    others = len(distinct) - dates
    return dates >= 2 and (dates >= DATES_PER_OTHER_VALUE * others
                           or (others <= FEW_OTHER_VALUES and dates >= FEW_DATES))


def expected_time_split(keys, gap, cap):
    """Which rows the time-ordered scheme must use (spec steps 6 and 7). `keys` maps a
    usable row to its place in time. None when the split cannot be formed."""
    by_stamp = {}
    for row, stamp in keys.items():
        by_stamp.setdefault(stamp, []).append(row)
    order = sorted(by_stamp)
    kept = len(keys)
    if cap is not None and kept > cap:
        first, kept = len(order), 0
        while first > 0 and kept + len(by_stamp[order[first - 1]]) <= cap:
            first -= 1
            kept += len(by_stamp[order[first]])
        order = order[first:]
    if not order:
        return None
    h, held = len(order), 0
    while h > 0 and 4 * held < kept:
        h -= 1
        held += len(by_stamp[order[h]])
    if h - gap < 6:
        return None
    rows_of = lambda part: {row for s in part for row in by_stamp[s]}  # noqa: E731
    return {"capped": len(keys) - kept, "holdout": rows_of(order[h:]),
            "gap": rows_of(order[h - gap:h]), "training": rows_of(order[:h - gap]),
            "holdout_stamps": len(order) - h, "order": order, "by_stamp": by_stamp, "h": h}


def property_three(sizes, held, n=None):
    """Spec step 7, property 3, for the units `sizes` (unit -> rows) of which `held` are
    the holdout. Returns the statements that fail."""
    n = sum(sizes.values()) if n is None else n
    rows = sum(sizes[u] for u in held)
    wrong = []
    if 4 * rows < n:
        wrong.append(f"the holdout holds {rows} of {n} rows, less than a quarter")
    spare = [u for u in held if 4 * (rows - sizes[u]) >= n]
    if spare:
        wrong.append(f"the holdout could spare {len(spare)} of its {len(held)} units")
    filled, unit = 0, 0
    for size in sorted(sizes.values()):
        filled, unit = filled + size, size
        if 4 * filled >= n:
            break
    if 4 * rows >= n + 4 * unit:
        wrong.append(f"the holdout holds {rows} of {n} rows, a quarter plus a unit of {unit} or more")
    return wrong


def _clusters(labels_a, labels_b):
    """Rows joined when they share a label in either list, directly or through others."""
    parent = list(range(len(labels_a)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for labels in (labels_a, labels_b):
        first = {}
        for i, label in enumerate(labels):
            if label is None:
                continue
            if label in first:
                a, b = find(first[label]), find(i)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first[label] = i
    return [find(i) for i in range(len(parent))]


def check_side(case, out, data_dir):
    """What the spec pins about one implementation's outcome on its own. Returns a list
    of (property, what was found)."""
    import numpy as np
    wrong = []
    o, args = case["oracle"], case["args"]

    def bad(name, text):
        wrong.append((name, text))

    if out["kind"] == "error":
        if out.get("written"):
            bad("nothing is written when an error is raised (2.1)", f"wrote {out['written']}")
        return wrong
    doc, preds = out["doc"], out["predictions"]
    if out.get("bad_numbers"):
        bad("no NaN or infinity in profile.json (5.2)", str(sorted(set(out["bad_numbers"]))))
    if args["time_col"] is None or args["group_col"] is None:
        bad("a call with a statement unsaid does not finish (2.1, D63)", "it returned a result")
    if o["time_form"] == "dayfirst" and o["declares_time"]:
        bad("a day-first time column is refused (step 3)", "it returned a result")
    if o["time_sentinel"] and o["declares_time"]:
        bad("a time column of dates written YYYYMMDD with a value that is no date stops "
            "the run (step 3; D69, D72)", "it returned a result")
    if len(set(map(str, case["frame"].columns))) != case["frame"].shape[1]:
        bad("duplicate column names stop the run (2.4)", "it returned a result")
        return wrong
    frame = judge_frame(case, data_dir)
    n = o["n"]
    rows = row_counts(doc)
    names = [str(c) for c in frame.columns]
    listed = [c.get("name") for c in doc.get("columns", [])]
    if listed != names:
        bad("every source column appears once, in file order (5.3 rule 2)",
            f"columns lists {listed}; the source has {names}")
        return wrong
    scheme = at(doc, "validation.scheme")
    timed = scheme == "time_ordered"
    usable = ~o["y_missing"] & (~o["t_missing"] if o["declares_time"] else True)
    # ---- what the headline rests on (D77 points 1 and 2, D78 points 1, 3 and 7)
    skill9 = at(doc, "headline.skill")
    base9 = at(doc, "headline.baseline_score")
    clf9 = at(doc, "task.type") == "classification"
    records9 = doc.get("columns", [])
    codes9 = [w.get("code") for w in doc.get("warnings", [])]
    lacking = [c.get("name") for c in records9 if any(k not in c for k in REST_KEYS)]
    if lacking:
        bad("every column record holds shuffled_score, shuffled_skill and skill_fall (5.2; D77)",
            f"not {lacking[:6]}")
    else:
        unscored9 = {w.get("column") for w in doc.get("warnings", [])
                     if w.get("code") == "shuffled_score_not_computed"}
        falls9 = []
        for c in records9:
            s9, w9, f9 = (c[k] for k in REST_KEYS)
            label9 = f"`{c.get('name')}` ({c.get('role')}): {(s9, w9, f9)}"
            if c.get("role") != "feature" or c.get("name") in unscored9:
                if (s9, w9, f9) != (None, None, None):
                    bad("a column that reached no model, or whose shuffled score could not be "
                        "computed, has null in all three keys (5.2; D78 point 7)", label9)
                continue
            if not is_number(s9):
                bad("a feature's shuffled_score is a number, or a "
                    "`shuffled_score_not_computed` warning names it (D78 point 7)", label9)
                continue
            if skill9 is None:
                if (w9, f9) != (None, None):
                    bad("with no headline skill, shuffled_skill and skill_fall are null (5.2)",
                        label9)
                continue
            want9 = 2 * s9 - 1 if clf9 else (1 - s9 / base9 if base9 else None)
            if want9 is None or not is_number(w9) or not close(w9, want9, 1e-7):
                bad("shuffled_skill is the skill of shuffled_score against the headline "
                    "baseline (5.2; D77 point 1)", f"{label9}; expected {want9!r}")
            elif not is_number(f9) or not close(f9, skill9 - w9, 1e-7):
                bad("skill_fall is the headline skill less shuffled_skill (5.2)", label9)
            else:
                falls9.append((-f9, c.get("position"), c.get("name"), f9))
        floor9 = at(doc, "headline.rests_on_floor")
        low9, high9 = at(doc, "headline.skill_low"), at(doc, "headline.skill_high")
        if skill9 is None:
            if floor9 is not None:
                bad("with no headline skill the floor of the list is null (D80 point 1)",
                    repr(floor9))
        elif is_number(low9) and is_number(high9):
            if not (is_number(floor9) and close(floor9, max(REST_FALL, (high9 - low9) / 2), 1e-9)):
                bad("the floor of the list is 0.05 or the half-width of the headline skill's "
                    "interval, the larger (D80 point 1)",
                    f"{floor9!r}; the interval {low9!r} to {high9!r}")
        elif not (is_number(floor9) and floor9 >= REST_FALL):
            bad("where the headline skill has no interval the floor of the list is a number of "
                "at least 0.05 (D80 point 1)", repr(floor9))
        cut9 = floor9 if is_number(floor9) else REST_FALL
        want_list = [(name, f) for _, _, name, f in sorted(falls9) if f >= cut9][:REST_MOST]
        got_list = at(doc, "headline.rests_on")
        same9 = isinstance(got_list, list) and len(got_list) == len(want_list) and all(
            isinstance(e, dict) and e.get("column") == name and is_number(e.get("skill_fall"))
            and close(e["skill_fall"], f, 1e-9) for e, (name, f) in zip(got_list, want_list))
        if not same9:
            bad("headline.rests_on lists the columns whose fall is at least the floor, at most "
                "five, largest first, ties in file order (D78 point 1, D80 point 1)",
                f"{got_list!r}; the judge makes {want_list!r}")
        near9 = is_number(skill9) and skill9 >= NEAR_PERFECT
        if codes9.count("near_perfect_headline") != (1 if near9 else 0):
            bad("one near_perfect_headline record exactly when the headline skill is 0.98 or "
                "more (2.5; D77 point 2)",
                f"{codes9.count('near_perfect_headline')} record(s) at skill {skill9!r}")
        by9 = {c.get("name"): c for c in records9}
        if "f1_again" in by9 and "f1" in by9 and by9["f1"].get("role") == "feature" \
                and by9["f1_again"].get("role") == "feature":
            pair9 = [tuple(by9[x].get(k) for k in REST_KEYS) for x in ("f1", "f1_again")]
            if pair9[0] != pair9[1]:
                bad("exact copies are shuffled together and share one fall (D78 point 3)",
                    f"f1 {pair9[0]}; f1_again {pair9[1]}")
    share9 = at(doc, "validation.holdout_share")
    far9 = timed and is_number(share9) and share9 >= FAR_ABOVE
    if codes9.count("holdout_far_above_quarter") != (1 if far9 else 0):
        bad("one holdout_far_above_quarter record exactly when a time-ordered holdout holds "
            "half of the rows or more (2.5; D77 point 6)",
            f"{codes9.count('holdout_far_above_quarter')} record(s); scheme {scheme}, holdout "
            f"share {share9!r}")
    # Dates written YYYYMMDD, with or without a value that is no date, are a date
    # column (step 5, rule 2a; D66, D72): the generator knows which columns hold them.
    for rec in doc["columns"]:
        if rec.get("name") not in ("daynum", "when") or rec.get("role") in ("time", "group",
                                                                           "target"):
            continue
        if rec["name"] == "when" and o["time_form"] not in ("yyyymmdd_int", "yyyymmdd_text"):
            continue
        held_cells = [v for v, ok in zip(frame.iloc[:, rec["position"]], usable) if ok]
        if is_date_column(held_cells) and (rec.get("role"), rec.get("reason")) != (
                "excluded", "datetime"):
            bad("a column of dates written YYYYMMDD is excluded as a date, also when it holds "
                "a value that is no date (step 5, rule 2a; D66, D72)",
                f"`{rec['name']}` has role {rec.get('role')!r}, reason {rec.get('reason')!r}")
    # The generator's identifier of whole numbers (step 5, rule 8b; D19, D74 point 5):
    # id-like when every value is distinct and the values are consecutive or strictly
    # ordered in file order, or when the values are nearly all distinct and fill half
    # of their range, whatever the order of the rows. And by nothing else.
    for rec in doc["columns"]:
        if rec.get("name") != "row_id" or rec.get("role") in ("time", "group", "target"):
            continue
        present = [v for v, ok in zip(frame.iloc[:, rec["position"]], usable)
                   if ok and not _empty(v)]
        if not present or any(isinstance(v, str) for v in present):
            continue
        whole = [int(v) for v in present]
        distinct = set(whole)
        if rec.get("reason") in ("constant", "all_missing", "datetime"):
            continue                        # an earlier rule of step 5 has the column
        span = max(distinct) - min(distinct) + 1
        rising = all(b > a for a, b in zip(whole, whole[1:]))
        falling = all(b < a for a, b in zip(whole, whole[1:]))
        ordered = len(whole) > 1 and len(distinct) == len(whole) and (
            rising or falling or span == len(whole))
        dense = len(distinct) > 1 and len(distinct) > ID_DISTINCT_SHARE * len(whole) \
            and len(distinct) >= ID_RANGE_SHARE * span
        is_id = (rec.get("role"), rec.get("reason")) == ("excluded", "id_like")
        if (ordered or dense) != is_id:
            bad("a whole-number column is id-like exactly when every value is distinct and the "
                "values are consecutive or strictly ordered in file order, or more than 95% of "
                "its values are distinct and they fill at least half of their range (step 5, "
                "rule 8b; D19, D74 point 5)",
                f"`row_id` has role {rec.get('role')!r}, reason {rec.get('reason')!r}; "
                f"{len(distinct)} distinct of {len(whole)} values over a range of {span}, "
                f"{'in' if rising or falling else 'not in'} the order of the file")
    want = {"loaded": n, "dropped_missing_target": int(o["y_missing"].sum()),
            "dropped_missing_time": int((~o["y_missing"] & o["t_missing"]).sum())
            if o["declares_time"] else 0}
    for item, count in want.items():
        if rows.get(item) != count:
            bad(f"rows.{item} equals the count from the source (5.3 rule 1)",
                f"{rows.get(item)!r}; the generator counts {count}")
    try:
        dropped = sum(rows[k] for k in ("dropped_missing_target", "dropped_missing_time",
                                        "dropped_gap", "dropped_row_cap"))
        if rows["loaded"] != dropped + rows["training"] + rows["holdout"]:
            bad("the row counts add up (5.3 rule 1)", str(rows))
    except (KeyError, TypeError):
        bad("rows has its seven items (5.2)", str(rows))
        return wrong
    # predictions: one row per holdout row, y_true from the source
    try:
        held = [int(row[0]) for row in preds]
        guesses = [float(row[2]) for row in preds]
    except (ValueError, IndexError):
        bad("predictions.csv holds (row, y_true, y_pred) (5.1)", "a row does not parse")
        return wrong
    if out.get("predictions_header") != ["row", "y_true", "y_pred"]:
        bad("predictions.csv has the header row,y_true,y_pred (5.1)", str(out.get("predictions_header")))
    if held != sorted(set(held)) or len(held) != rows["holdout"]:
        bad("predictions.csv has one line per holdout row, in increasing row (5.1)",
            f"{len(held)} lines, {len(set(held))} distinct, rows.holdout {rows['holdout']}")
    if any(not 0 <= p < n or not usable[p] for p in held):
        bad("every holdout row is a usable row of the source (5.3 rule 3)",
            f"{sum(1 for p in held if not 0 <= p < n or not usable[p])} rows are not")
        return wrong
    if not all(math.isfinite(v) for v in guesses):
        bad("y_pred is always finite (5.1)", "a prediction is not")
    cells = source_cells(case, data_dir, args["target"])
    off = [p for p, row in zip(held, preds) if not same_cell(row[1], cells[p])]
    if off:
        bad("y_true equals the target of that row in the source (5.3 rule 3)",
            f"{len(off)} rows differ, the first at row {off[0]}")
    # the headline recomputed (5.3 rule 4)
    metric, score = at(doc, "headline.metric"), at(doc, "headline.score")
    base, skill = at(doc, "headline.baseline_score"), at(doc, "headline.skill")
    mine = None
    positive = at(doc, "task.positive_class")
    labels = [1 if same_cell(row[1], positive) else 0 for row in preds]
    if metric == "roc_auc":
        mine = auc(labels, guesses)
    elif metric == "mse":
        try:
            truth = [float(row[1]) for row in preds]
            mine = math.fsum((a - b) ** 2 for a, b in zip(truth, guesses)) / len(truth)
        except (ValueError, ZeroDivisionError, OverflowError):
            mine = None
    if mine is not None and (not is_number(score) or not close(score, mine, 1e-7)):
        bad("headline.score equals the metric recomputed from predictions.csv (5.3 rule 4)",
            f"{score!r}; recomputed {mine!r}")
    if is_number(score) and is_number(base):
        if metric == "roc_auc":
            follows = 2 * score - 1
        else:
            follows = None if base == 0 else 1 - score / base
        if (follows is None) != (skill is None) or (follows is not None
                                                    and not close(skill, follows, 1e-7)):
            bad("headline.skill follows from the score and the baseline (5.3 rule 4)",
                f"{skill!r}; from score and baseline {follows!r}")
    # the largest row's share of the squared error on the holdout (step 11; D74 point 3)
    share_keys = ("largest_row_share", "largest_row", "baseline_largest_row_share",
                  "baseline_largest_row")
    got = {k: at(doc, f"headline.{k}") for k in share_keys}
    absent = [k for k, v in got.items() if v == ("<absent>",)]
    if absent:
        bad("headline holds the largest row's share, for the model and for the baseline "
            "(5.2; D74 point 3)", f"absent: {absent}")
    else:
        if metric == "roc_auc" and any(v is not None for v in got.values()):
            bad("the largest row's share is null for a classification (5.2)", str(got))
        elif metric == "mse" and mine is not None:
            errors = [(x - y) ** 2 for x, y in zip(truth, guesses)]
            total = math.fsum(errors)
            if total > 0:
                top = max(errors)
                if not is_number(got["largest_row_share"]) or not close(
                        got["largest_row_share"], top / total, 1e-7):
                    bad("headline.largest_row_share is the largest squared error over their sum, "
                        "recomputed from predictions.csv (step 11; D74 point 3)",
                        f"{got['largest_row_share']!r}; recomputed {top / total!r}")
                at_top = {p for p, e in zip(held, errors) if close(e, top, 1e-9)}
                if got["largest_row"] not in at_top:
                    bad("headline.largest_row is the holdout row with the largest squared error "
                        "(step 11; D74 point 3)",
                        f"{got['largest_row']!r}; the judge finds {sorted(at_top)[:3]}")
            elif got["largest_row_share"] is not None:
                bad("the largest row's share is null when the model has no error (5.2)",
                    repr(got["largest_row_share"]))
            b_share = got["baseline_largest_row_share"]
            if (b_share is None) != (base == 0):
                bad("the baseline's largest row has a share exactly when the baseline has an "
                    "error (5.2)", f"share {b_share!r}, baseline score {base!r}")
            elif b_share is not None and (not is_number(b_share) or not 0 < b_share <= 1 + 1e-12
                                          or got["baseline_largest_row"] not in set(held)):
                bad("the baseline's largest row is a holdout row, and its share is above 0 and "
                    "at most 1 (5.2)", f"{b_share!r} at row {got['baseline_largest_row']!r}")
        above = sum(1 for k in ("largest_row_share", "baseline_largest_row_share")
                    if is_number(got[k]) and got[k] > LARGEST_ROW_SHARE)
        # The same statement over the selection folds (D74 point 3): four keys in `cv`,
        # for the chosen configuration and for the headline baseline, and two in every
        # record of `models`. The folds are inside the training rows, so no row named
        # there is a holdout row. One record when a configuration's share is above one
        # half, whichever and however many, and one when the baseline's is.
        cv = doc.get("cv", {})
        records = [m for m in doc.get("models", []) if isinstance(m, dict)]
        lacking = [f"cv.{key}" for key in share_keys if key not in cv] + [
            f"models[{m.get('config')}].{key}" for m in records
            for key in share_keys[:2] if key not in m]
        if lacking:
            bad("`cv` and every record of `models` hold the largest row's share over the "
                "selection folds (5.2; D74 point 3)", f"absent: {lacking[:6]}")
        else:
            pairs = [(f"cv.{key}", cv[key + "largest_row_share"], cv[key + "largest_row"])
                     for key in ("", "baseline_")]
            pairs += [(f"models[{m.get('config')}]", m["largest_row_share"], m["largest_row"])
                      for m in records]
            in_holdout = set(held)
            for where, share, row in pairs:
                if metric == "roc_auc":
                    sound = share is None and row is None
                else:
                    sound = (share is None and row is None) or (
                        is_number(share) and 0 < share <= 1 + 1e-12
                        and isinstance(row, int) and not isinstance(row, bool)
                        and 0 <= row < n and row not in in_holdout)
                if not sound:
                    bad("over the selection folds the largest row's share is null for a "
                        "classification; for a regression null together with its row, or "
                        "above 0 and at most 1 at a row that is no holdout row (5.2; D74 "
                        "point 3)", f"{where}: share {share!r} at row {row!r}")
            chosen = [m for m in records if m.get("chosen") is True]
            if len(chosen) == 1 and first_difference(
                    [cv["largest_row_share"], cv["largest_row"]],
                    [chosen[0]["largest_row_share"], chosen[0]["largest_row"]]):
                bad("`cv` holds the largest row's share of the chosen configuration (5.2; D74 "
                    "point 3)",
                    f"cv: {cv['largest_row_share']!r} at row {cv['largest_row']!r}; "
                    f"models[{chosen[0].get('config')}]: {chosen[0]['largest_row_share']!r} "
                    f"at row {chosen[0]['largest_row']!r}")
        above += any(is_number(m.get("largest_row_share"))
                     and m["largest_row_share"] > LARGEST_ROW_SHARE for m in records)
        over = cv.get("baseline_largest_row_share")
        above += bool(is_number(over) and over > LARGEST_ROW_SHARE)
        said = sum(count for (code, _), count in warning_counts(doc).items()
                   if code == "largest_row_dominates")
        if said != above:
            bad("one largest_row_dominates record for each statement above one half: the "
                "model and the baseline on the holdout, any configuration and the baseline "
                "over the selection folds (2.5; D74 point 3)",
                f"{said} record(s); {above} statement(s) above {LARGEST_ROW_SHARE}: {got}")
    # null patterns of 5.2
    codes = {code for code, _ in warning_counts(doc)}
    method = at(doc, "headline.interval_method")
    if (skill is None) != ("skill_undefined" in codes):
        bad("skill is null exactly with the skill_undefined warning (2.5)",
            f"skill {skill!r}, warning {'present' if 'skill_undefined' in codes else 'absent'}")
    lows = [at(doc, f"headline.{k}") for k in ("interval_low", "interval_high")]
    if (method is None) != all(v is None for v in lows) or (method is None) != (
            "no_interval" in codes) or (method is None) != bool(at(doc, "headline.interval_reason")):
        bad("no interval: null fields, a reason and the no_interval warning together (5.2)",
            f"method {method!r}, ends {lows}, no_interval {'no_interval' in codes}")
    if method is not None and all(is_number(v) for v in lows) and lows[0] > lows[1]:
        bad("the interval's low end is not above its high end", str(lows))
    if (method == "bootstrap_blocks") != ("interval_approximate" in codes):
        bad("interval_approximate exactly with bootstrap_blocks (2.5)", f"method {method!r}")
    fold_list = at(doc, "validation.folds")
    if isinstance(fold_list, list):
        unscored = sum(1 for f in fold_list if isinstance(f, dict) and f.get("scored") is False)
        records = sum(count for (code, _), count in warning_counts(doc).items()
                      if code == "fold_not_scored")
        if records != unscored:
            bad("one fold_not_scored record for each fold not scored (2.5; D72)",
                f"{records} record(s) for {unscored} fold(s) not scored")
        sized = [f for f in fold_list if isinstance(f, dict)
                 and isinstance(f.get("training_rows"), int)]
        if at(doc, "task.type") == "regression" and unscored:
            bad("every fold of a regression is scored (step 7)",
                str([(f.get("training_rows"), f.get("validation_rows"), f.get("scored"))
                     for f in fold_list if isinstance(f, dict)]))
        told = sum(count for (code, _), count in warning_counts(doc).items()
                   if code == "selection_not_made")
        few = scheme == "grouped" and len(sized) == len(fold_list) and bool(sized) \
            and any(f["training_rows"] < SELECTION_FLOOR
                    and f["training_rows"] < f["validation_rows"] for f in sized)
        if told != int(few):
            bad("one selection_not_made record exactly when, under the grouped scheme, a fold "
                "has fewer than 50 training rows and fewer training rows than validation rows "
                "(step 9; D74 point 6)",
                f"{told} record(s); the folds train on "
                f"{[f['training_rows'] for f in sized]} rows under the {scheme} scheme")
        elif few:
            fitted = [m.get("config") for m in doc.get("models", [])
                      if isinstance(m, dict) and m.get("status") == "fitted"]
            if fitted and at(doc, "cv.chosen") != fitted[0]:
                bad("with selection_not_made the first configuration in configuration order "
                    "is used (step 9; D74 point 6)",
                    f"cv.chosen is {at(doc, 'cv.chosen')!r}; the first fitted configuration is "
                    f"{fitted[0]!r}")
    p_value = at(doc, "headline.p_value")
    may_have_p = scheme == "random" and skill is not None
    if (p_value is not None) != may_have_p or (p_value is None) != bool(
            at(doc, "headline.p_value_reason")):
        bad("a p-value only under the random scheme with a skill, else a reason (5.2)",
            f"scheme {scheme!r}, skill {skill!r}, p_value {p_value!r}")
    expected_method = {"random": "bootstrap_rows", "grouped": "bootstrap_groups",
                       "time_ordered": "bootstrap_blocks"}.get(scheme)
    if method not in (None, expected_method):
        bad("the interval method is the scheme's (5.2)", f"{method!r} under {scheme!r}")
    limits = doc.get("limitations")
    if not isinstance(limits, list) or len(limits) != LIMITATION_COUNT:
        bad("limitations holds nine sentences (5.4)", f"{len(limits or [])}")
    if at(doc, "input.time_col") != args["time_col"] or at(doc, "input.group_col") != args["group_col"]:
        bad("input records both statements as passed (5.3 rule 9)",
            f"{at(doc, 'input.time_col')!r}, {at(doc, 'input.group_col')!r}")
    # the scheme the statements give (2.1, the four combinations)
    capped = rows["dropped_row_cap"]
    n_usable = int(np.sum(usable))
    if args["profile"] != "fast" and capped:
        bad("the full profile leaves no row out (step 6)", f"dropped_row_cap {capped}")
    if o["declares_time"]:
        if scheme != "time_ordered":
            bad("a time column gives the time-ordered scheme (2.1)", repr(scheme))
        elif o["time_form"] not in ("dayfirst",) and not o["time_sentinel"]:
            keys = {int(i): int(o["t"][i]) for i in np.flatnonzero(usable)}
            split = expected_time_split(keys, args["gap"],
                                        FAST_CAP if args["profile"] == "fast" else None)
            if split is None:
                bad("too few timestamps for the holdout and six blocks stop the run (step 7)",
                    "it returned a result")
            else:
                if set(held) != split["holdout"]:
                    bad("the holdout is the latest whole timestamps holding a quarter (step 7)",
                        f"{len(set(held) - split['holdout'])} unexpected rows, "
                        f"{len(split['holdout'] - set(held))} missing")
                for item, count in (("dropped_row_cap", split["capped"]),
                                    ("dropped_gap", len(split["gap"])),
                                    ("training", len(split["training"])),
                                    ("holdout", len(split["holdout"]))):
                    if rows[item] != count:
                        bad(f"rows.{item} under time order (steps 6 and 7)",
                            f"{rows[item]!r}; the generator counts {count}")
                if at(doc, "validation.n_holdout_timestamps") != split["holdout_stamps"]:
                    bad("validation.n_holdout_timestamps (5.2)",
                        f"{at(doc, 'validation.n_holdout_timestamps')!r}; counted "
                        f"{split['holdout_stamps']}")
                if at(doc, "task.type") == "regression":
                    listed_b = [x.get("name") for x in doc.get("baselines", [])
                                if isinstance(x, dict)]
                    if listed_b != ["training_mean", "previous_value"]:
                        bad("under time order a regression scores both baselines, whether the "
                            "timestamps repeat or not (step 11; D73 point 1, D74 point 1)",
                            str(listed_b))
                    elif not o["declares_entity"]:
                        # the mean target of the timestamp `gap + 1` distinct timestamps before
                        try:
                            value = {i: float(cells[i]) for i in keys}
                            place = {s: k for k, s in enumerate(split["order"])}
                            means = [math.fsum(value[i] for i in split["by_stamp"][s])
                                     / len(split["by_stamp"][s]) for s in split["order"]]
                            level = math.fsum(value[i] for i in split["training"]) \
                                / len(split["training"])
                            by_mean = math.fsum((value[i] - level) ** 2
                                                for i in split["holdout"]) / len(split["holdout"])
                            by_prev = math.fsum(
                                (value[i] - means[place[keys[i]] - args["gap"] - 1]) ** 2
                                for i in split["holdout"]) / len(split["holdout"])
                        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
                            by_mean = by_prev = None
                        said_b = {x["name"]: x.get("score") for x in doc["baselines"]}
                        for b_name, b_mine in (("training_mean", by_mean),
                                               ("previous_value", by_prev)):
                            if b_mine is not None and math.isfinite(b_mine) and (
                                    not is_number(said_b[b_name])
                                    or not close(said_b[b_name], b_mine, 1e-7)):
                                bad(f"the baseline {b_name} recomputed from the source: with no "
                                    "entity column the previous value is the mean target of the "
                                    "timestamp gap + 1 distinct timestamps before (step 11; D73 "
                                    "point 1, D74 point 1)",
                                    f"{said_b[b_name]!r}; recomputed {b_mine!r}")
                if o["declares_entity"] and o["who"] is not None:
                    ent = o["who"]
                    want_units = (len({int(ent[i]) for i in split["training"]}),
                                  len({int(ent[i]) for i in split["holdout"]}))
                    got_units = (at(doc, "validation.n_groups_training"),
                                 at(doc, "validation.n_groups_holdout"))
                    if got_units != want_units:
                        bad("the entities on each side under time order (5.2)",
                            f"{got_units}; counted {want_units}")
    else:
        if scheme not in ("random", "grouped"):
            bad("no time column gives the random or the grouped scheme (2.1)", repr(scheme))
        if o["declares_entity"] and scheme != "grouped":
            bad("an entity column and no time column give the grouped scheme (2.1)", repr(scheme))
        if not o["declares_entity"] and args["duplicates"] != "group" and scheme != "random":
            bad("no entity column and no duplicates='group' give the random scheme", repr(scheme))
        if not o["declares_entity"]:
            want_cap = max(0, n_usable - FAST_CAP) if args["profile"] == "fast" else 0
            if capped != want_cap:
                bad("the fast cap keeps 2,000 rows drawn at random (step 6)",
                    f"dropped_row_cap {capped}; expected {want_cap}")
        elif args["profile"] == "fast" and n_usable > FAST_CAP:
            # The cap of whole entities (step 6; D57, D70, D72). The output does not say
            # which rows were left out, so this goes by the counts, the holdout rows
            # and the warnings for a large unit.
            kept = n_usable - capped
            sizes = {}
            for i in np.flatnonzero(usable):
                sizes[int(o["who"][i])] = sizes.get(int(o["who"][i]), 0) + 1
            large = {e: s for e, s in sizes.items() if 4 * s >= n_usable}
            others = [s for e, s in sizes.items() if e not in large]
            named = sum(count for (code, _), count in warning_counts(doc).items()
                        if code in ("large_group_kept_in_training", "large_group_is_holdout"))
            seen_held = {int(o["who"][p]) for p in held}
            if max(sizes.values()) > FAST_CAP or sum(large.values()) > FAST_CAP:
                bad("the fast profile stops when an entity it must keep does not fit under "
                    "its cap (step 6; D70, D72)",
                    f"it returned a result; the largest entity holds {max(sizes.values())} of "
                    f"the {n_usable} usable rows, and those of a quarter or more "
                    f"{sum(large.values())} together")
            elif large and (capped > sum(others)
                            or (not named and not (set(large) & seen_held))):
                bad("an entity of a quarter or more of the usable rows is kept by the fast "
                    "cap (step 6; D72)",
                    f"dropped_row_cap {capped} of {n_usable} usable rows; the entities of a "
                    f"quarter or more hold {sorted(large.values())} rows and the others "
                    f"{sum(others)}; no row of a large one is in the holdout and no warning "
                    "names a large unit")
            elif kept > FAST_CAP or (capped and others and kept <= FAST_CAP - max(others)):
                bad("the fast cap keeps whole entities, and none left out would have fitted "
                    "(step 6)", f"kept {kept} rows; the largest entity that can be left out "
                                f"holds {max(others)}")
        split_rows = rows["training"] + rows["holdout"]
        if scheme == "random" and rows["holdout"] != -(-split_rows // 4):
            bad("the random holdout is a quarter of the rows, rounded up (step 7)",
                f"{rows['holdout']} of {split_rows}")
        share = at(doc, "validation.holdout_share")
        if is_number(share) and split_rows and not close(share, rows["holdout"] / split_rows):
            bad("validation.holdout_share is holdout over the rows split (5.2)", repr(share))
        copies, copy_share = at(doc, "validation.n_holdout_copies"), \
            at(doc, "validation.holdout_copy_share")
        if not isinstance(copies, int) or not 0 <= copies <= rows["holdout"] or not is_number(
                copy_share) or not close(copy_share, copies / max(1, rows["holdout"])):
            bad("the count and the share of holdout copies are reported (step 8a)",
                f"{copies!r}, {copy_share!r}")
        elif copies >= 10 and copy_share > 0.01:
            if args["duplicates"] != "keep":
                bad("copies above the rule stop the run unless duplicates='keep' (step 8a)",
                    f"{copies} of {rows['holdout']} and it finished")
            elif "holdout_copies_kept" not in codes:
                bad("copies above the rule under 'keep' carry holdout_copies_kept (2.5)", "absent")
        elif "holdout_copies_kept" in codes:
            bad("holdout_copies_kept only when the rule is met (2.5)", f"{copies} copies")
        folds = at(doc, "validation.folds")
        if isinstance(folds, list) and len(folds) == 5:
            valid = [f.get("validation_rows") for f in folds]
            if all(isinstance(v, int) for v in valid):
                if sum(valid) != rows["training"] or any(
                        f.get("training_rows") != rows["training"] - f.get("validation_rows")
                        for f in folds):
                    bad("the five validation parts are the training rows (step 7)", str(valid))
                elif scheme == "random" and max(valid) - min(valid) > 1:
                    bad("fold sizes differ by at most one (step 7)", str(valid))
        # `duplicates="group"`: in a finished run no combination of the features that
        # reached the models is on both sides (steps 7 and 8a; D62, D69). Counted
        # where every usable row is split: under the cap the output does not say
        # which of the other rows are training rows.
        if args["duplicates"] == "group" and not capped:
            left = feature_keys(frame, doc, reached_models=True)
            in_holdout = set(held)
            used_rows = [int(i) for i in np.flatnonzero(usable)]
            both = {left[i] for i in used_rows if i in in_holdout} \
                & {left[i] for i in used_rows if i not in in_holdout}
            if both:
                leaks = [c["name"] for c in doc["columns"] if c.get("reason") == "suspected_leak"]
                bad("with duplicates='group' no combination of the features that reached the "
                    "models is on both sides of the split (steps 7 and 8a; D62, D69)",
                    f"{len(both)} combination(s) are; excluded as suspected leaks: "
                    f"{leaks or 'none'}")
        # whole units (step 7, properties 1 and 3), where every usable row is split
        if scheme == "grouped" and not capped:
            used = np.flatnonzero(usable)
            in_holdout = set(held)
            entity = [int(o["who"][i]) if o["declares_entity"] else None for i in used]
            alike = feature_keys(frame, doc) if args["duplicates"] == "group" else None
            agree = [alike[i] for i in used] if alike is not None else [None] * len(used)
            if alike is not None:
                seen = {}
                for key in agree:
                    seen[key] = seen.get(key, 0) + 1
                same = sum(v for v in seen.values() if v > 1)
                if at(doc, "validation.n_rows_same_features") != same:
                    bad("n_rows_same_features counts the rows that agree in every feature (5.2)",
                        f"{at(doc, 'validation.n_rows_same_features')!r}; the judge counts {same}")
                if (same > 0) != ("same_features_grouped" in codes):
                    bad("same_features_grouped exactly when rows agree (2.5)", f"{same} rows agree")
            unit = _clusters(entity, agree)
            sizes, held_units = {}, set()
            for i, u in zip(used, unit):
                sizes[u] = sizes.get(u, 0) + 1
                if int(i) in in_holdout:
                    held_units.add(u)
            cut = [u for u in held_units
                   if sum(1 for i, v in zip(used, unit) if v == u and int(i) in in_holdout) != sizes[u]]
            if cut:
                bad("no unit has rows on both sides of the split (step 7, property 1)",
                    f"{len(cut)} of {len(sizes)} units do")
            else:
                got_units = (at(doc, "validation.n_groups_training"),
                             at(doc, "validation.n_groups_holdout"))
                if got_units != (len(sizes) - len(held_units), len(held_units)):
                    bad("the units counted on each side (5.2)",
                        f"{got_units}; the judge counts "
                        f"{(len(sizes) - len(held_units), len(held_units))}")
                if at(doc, "task.type") == "classification":
                    cls = [same_cell(cells[i], positive) for i in used]
                    pos = {}
                    for u, c in zip(unit, cls):
                        pos[u] = pos.get(u, 0) + int(c)
                    for stratum in (0, 1):
                        part = {u: s for u, s in sizes.items() if int(2 * pos[u] >= s) == stratum}
                        if part:
                            for text in property_three(part, {u for u in held_units if u in part}):
                                bad("the grouped holdout (step 7, property 3)",
                                    f"units of class {stratum}: {text}")
                else:
                    for text in property_three(sizes, held_units):
                        bad("the grouped holdout (step 7, property 3)", text)
    return wrong


# ------------------------------------------------------------- the comparison --

EXACT = ["meta.tool", "meta.schema_version", "meta.profile", "meta.seed", "meta.n_jobs",
         "meta.versions", "input.source", "input.kind", "input.target", "input.time_col",
         "input.task", "input.gap", "input.n_rows", "input.n_columns", "input.duplicate_rows",
         "input.duplicates", "input.group_col", "task.type", "task.decided_by", "task.n_distinct",
         "task.target_dtype", "task.classes", "task.positive_class", "validation.scheme",
         "validation.n_folds", "validation.stratified", "validation.gap", "validation.time_col",
         "validation.time_form", "validation.caveat", "validation.group_col",
         "validation.train_end", "validation.holdout_start", "validation.holdout_end",
         "validation.n_train_timestamps", "validation.n_holdout_timestamps",
         "validation.timestamps_unique", "validation.n_holdout_without_previous",
         "headline.metric", "headline.baseline", "headline.interval_level", "headline.n_resamples",
         "headline.block_length", "cv.metric", "cv.greater_is_better", "cv.n_folds",
         "cv.n_configurations", "limitations"]
# Warnings the table and the arguments alone decide, whatever rows a split picks.
TABLE_CODES = {"date_column_undeclared", "values_coerced", "numeric_with_separators",
               "whole_number_regression_target", "blank_header", "infinite_values",
               "duplicate_rows", "unsupported_type", "interval_approximate", "no_previous_value"}
SPLIT_ERRORS = {"holdout_copies_in_training", "one_class", "too_few_groups",
                "fast_profile_too_few_rows"}
# Spec 2.5 says when each of these is raised and not how many records it makes.
# (`duplicate_rows` and `fold_not_scored` left this list with D72: both are pinned.)
COUNT_NOT_PINNED = {"small_holdout", "no_interval"}
# What is listed and not compared: no part of the verdict. Why -> the spec's word on it.
NOT_COMPARED = {
    "two_errors": "the table earns two named errors of steps 1 to 4, and which is raised is "
                  "open (spec 2.4)",
    "rows": "it depends on which rows a random or a grouped split took, or the fast cap kept "
            "(spec 11)",
    "records": "both sides raise the warning; how many records it has is not pinned (spec 2.5)",
    "estimator": "which estimator of the lag-1 autocorrelation the panel warning uses is open "
                 "(spec 11)",
}
FAILS = ("raw", "outcome", "exact", "fit", "property")      # each of these fails the tool
# The column the generator makes as an exact copy of the target. A column that
# reproduces the target is always excluded (D73 point 2, D74 point 2).
TARGET_COPY = "y_copy"


def brief(out):
    if out["kind"] == "error":
        return f"error {out['code']}"
    if out["kind"] == "raw":
        return f"RAW {out['error_class']}: {out['message'][:160]}"
    if out["kind"] == "result":
        return (f"result: {at(out['doc'], 'validation.scheme')}, skill "
                f"{at(out['doc'], 'headline.skill')!r}, model {at(out['doc'], 'headline.model')}")
    return out["kind"]


def compare(case, app, ref, data_dir):
    """Every finding on one table: where the application's outcome and the reference's
    differ, and where one side alone breaks what the spec pins.

    Each finding has a class. `raw`, `outcome`, `exact`, `fit` and `property` fail
    the tool; `limit` carries the name of an entry of `STATED_LIMITS` (and `open`
    one of `OPEN`: no finding has either class while both lists are empty);
    `not_compared` carries why (`NOT_COMPARED`); `harness` is the tool's own fault.
    """
    found = []

    def add(kind, field, a, b, note="", name=None):
        found.append({"class": kind, "name": name, "field": field, "application": a,
                      "reference": b, "note": note})

    def aside(why, field, a, b, note=""):
        add("not_compared", field, a, b, (note + " | " if note else "") + NOT_COMPARED[why], why)

    for side, out in (("application", app), ("reference", ref)):
        if out["kind"] == "raw":
            add("raw", side, brief(app), brief(ref), out.get("trace", "")[-700:])
        elif out["kind"] == "missing":
            add("harness", side, brief(app), brief(ref), "the run did not finish")
    if {app["kind"], ref["kind"]} & {"raw", "missing"}:
        return found
    for side, out in (("application", app), ("reference", ref)):
        try:
            for name, text in check_side(case, out, data_dir):
                add("property", f"{side}: {name}", text if side == "application" else "",
                    text if side == "reference" else "")
        except Exception as exc:  # noqa: BLE001  (a malformed document is a finding too)
            add("property", f"{side}: the output could be read by the judge",
                f"{type(exc).__name__}: {exc}" if side == "application" else "",
                f"{type(exc).__name__}: {exc}" if side == "reference" else "",
                traceback.format_exc()[-600:])
        # An exact copy of the target that the leak screen leaves a feature: a property
        # of one side, under every scheme, and it fails the tool (D73 point 2).
        if out["kind"] == "result":
            kept = [c for c in out["doc"].get("columns", [])
                    if c.get("name") == TARGET_COPY and c.get("role") == "feature"]
            if kept:
                text = (f"`{TARGET_COPY}` is a feature; skill "
                        f"{at(out['doc'], 'headline.skill')!r}, warned: "
                        f"{any(k[1] == TARGET_COPY for k in warning_counts(out['doc']))}")
                add("property", f"{side}: an exact copy of the target is excluded by the leak "
                                "screen (step 8; D73 point 2, D74 point 2)",
                    text if side == "application" else "", text if side == "reference" else "")
    if app["kind"] == "error" and ref["kind"] == "error":
        if app["code"] != ref["code"]:
            earned = broken_rules(case)
            words = f"application: {app['message'][:300]} | reference: {ref['message'][:300]}"
            if app["code"] in earned and ref["code"] in earned:
                aside("two_errors", "the named error", app["code"], ref["code"], words)
            else:
                add("outcome", "the named error", app["code"], ref["code"], words)
        return found
    if app["kind"] != ref["kind"]:
        stopped, finished = (app, ref) if app["kind"] == "error" else (ref, app)
        by_rows = stopped["code"] in SPLIT_ERRORS \
            and at(finished["doc"], "validation.scheme") != "time_ordered"
        if by_rows:
            aside("rows", "a result on one side, a named error on the other", brief(app),
                  brief(ref), stopped["message"][:400])
        else:
            add("outcome", "a result on one side, a named error on the other", brief(app),
                brief(ref), stopped["message"][:400])
        return found

    a, b = app["doc"], ref["doc"]

    def exact(path, kind="exact", note=""):
        """Equal on both sides, or a finding of class `kind`; a kind of `NOT_COMPARED`
        is listed and no part of the verdict."""
        x, y = at(a, path), at(b, path)
        if not first_difference(x, y):
            return True
        if kind in NOT_COMPARED:
            aside(kind, path, x, y, note)
        else:
            add(kind, path, x, y, note)
        return False

    for path in EXACT:
        exact(path)
    scheme_a, scheme_b = at(a, "validation.scheme"), at(b, "validation.scheme")
    if scheme_a != scheme_b:
        return found
    timed = scheme_a == "time_ordered"
    ra, rb = row_counts(a), row_counts(b)
    capped = bool(ra.get("dropped_row_cap") or rb.get("dropped_row_cap"))
    entity_cap = scheme_a == "grouped" and at(a, "input.group_col") is not False
    for item in ("loaded", "dropped_missing_target", "dropped_missing_time", "dropped_gap",
                 "dropped_row_cap", "training", "holdout"):
        if item == "dropped_row_cap" and entity_cap:
            continue                        # which entities the cap keeps is open (11)
        if item in ("training", "holdout") and scheme_a == "grouped":
            continue                        # property 3 is the contract, not the count
        if ra.get(item) != rb.get(item):
            add("exact", f"rows.{item}", ra.get(item), rb.get(item))
    # columns
    ca, cb = a.get("columns", []), b.get("columns", [])
    if len(ca) != len(cb):
        add("exact", "columns", f"{len(ca)} records", f"{len(cb)} records")
    for x, y in zip(ca, cb):
        name = x.get("name")
        for key in ("name", "position", "n_missing", "n_coerced", "n_infinite"):
            if first_difference(x.get(key), y.get(key)):
                add("exact", f"columns[{name}].{key}", x.get(key), y.get(key))
        role_a = (x.get("role"), x.get("reason"), x.get("leak_rule"))
        role_b = (y.get("role"), y.get("reason"), y.get("leak_rule"))
        if role_a != role_b:
            leak = "suspected_leak" in (x.get("reason"), y.get("reason"))
            if leak and not timed:
                aside("rows", f"columns[{name}] role, reason, leak rule", role_a, role_b,
                      "the leak screen ran on different folds")
            else:
                add("exact", f"columns[{name}] role, reason, leak rule", role_a, role_b)
        elif x.get("role") == "feature":
            if x.get("kind") != y.get("kind"):
                add("exact", f"columns[{name}].kind", x.get("kind"), y.get("kind"))
            if (timed or not capped) and x.get("n_imputed") != y.get("n_imputed"):
                add("exact", f"columns[{name}].n_imputed", x.get("n_imputed"), y.get("n_imputed"))
        elif x.get("n_imputed") != y.get("n_imputed"):
            add("exact", f"columns[{name}].n_imputed", x.get("n_imputed"), y.get("n_imputed"))
    # validation
    if scheme_a != "grouped":
        exact("validation.holdout_share")
    if timed:
        exact("validation.folds")
        for path in ("validation.n_holdout_copies", "validation.holdout_copy_share",
                     "validation.n_groups_training", "validation.n_groups_holdout"):
            exact(path)
    if not capped or timed:
        exact("validation.n_rows_same_features")
    if scheme_a == "random":
        exact("validation.n_groups_training")
        exact("validation.n_groups_holdout")
    # headline, baselines, cv, models
    names_a = [x.get("name") for x in a.get("baselines", [])]
    names_b = [x.get("name") for x in b.get("baselines", [])]
    if names_a != names_b:
        add("exact", "baselines (names)", names_a, names_b)
    for path in ("headline.interval_method",):
        exact(path, "exact" if timed else "rows",
              "" if timed else "the resamples, or the units in the holdout, differ")
    for path in ("headline.p_value_reason", "headline.interval_reason"):
        if (at(a, path) is None) != (at(b, path) is None):
            if timed:
                add("exact", path + " (null or not)", at(a, path), at(b, path))
            else:
                aside("rows", path + " (null or not)", at(a, path), at(b, path))
    ma, mb = a.get("models", []), b.get("models", [])
    # Under time order the rows of every fit are determined, and so is their order:
    # time order, ties in file order (D74 point 4). A forest draws its rows by their
    # place in the matrix it is handed, so it is compared like every other family.
    if timed:
        exact("headline.baseline_score")
        exact("baselines")
        exact("cv.n_scored_folds")
        for path in ("headline.baseline_largest_row_share", "headline.baseline_largest_row",
                     "cv.baseline_largest_row_share", "cv.baseline_largest_row"):
            exact(path)
        for path in ("headline.score", "headline.skill", "headline.model", "cv.chosen",
                     "cv.chosen_mean", "cv.chosen_std", "headline.largest_row_share",
                     "headline.largest_row", "cv.largest_row_share", "cv.largest_row"):
            exact(path, "fit", "the same rows on both sides: only the fits can differ")
    elif at(a, "cv.n_scored_folds") != at(b, "cv.n_scored_folds"):
        aside("rows", "cv.n_scored_folds", at(a, "cv.n_scored_folds"),
              at(b, "cv.n_scored_folds"))
    shape_a = [(m.get("config"), m.get("family"), json.dumps(m.get("params"), sort_keys=True))
               for m in ma]
    shape_b = [(m.get("config"), m.get("family"), json.dumps(m.get("params"), sort_keys=True))
               for m in mb]
    if shape_a != shape_b:
        add("exact", "models (configurations, families, params)", shape_a, shape_b)
    elif timed:
        keys = ["status", "error", "fold_scores", "cv_mean", "cv_std", "chosen",
                "largest_row_share", "largest_row"]
        mine = [{k: m.get(k) for k in keys} for m in ma]
        theirs = [{k: m.get(k) for k in keys} for m in mb]
        for x, y in zip(mine, theirs):          # the error text is wording: null or not
            x["error"], y["error"] = x["error"] is None, y["error"] is None
        difference = first_difference(mine, theirs, "models")
        if difference:
            add("fit", "models", difference, "", "the same rows on both sides")
    else:
        status_a, status_b = [m.get("status") for m in ma], [m.get("status") for m in mb]
        if status_a != status_b:
            aside("rows", "models (status)", status_a, status_b)
    # warnings, as (code, column) with their counts
    wa, wb = warning_counts(a), warning_counts(b)
    for key in sorted(set(wa) | set(wb), key=str):
        if wa.get(key, 0) == wb.get(key, 0):
            continue
        code = key[0]
        hard = timed or code in TABLE_CODES
        if code in ("same_features_grouped", "large_group_kept_in_training",
                    "large_group_is_holdout"):
            hard = not capped
        if code == "small_holdout" and at(a, "task.type") == "regression":
            hard = scheme_a != "grouped"      # there the holdout's size is a property, not a count
        field = f"warnings[{code}, {key[1]}] (count)"
        if code == "autocorrelated_panel_target":
            aside("estimator", field, wa.get(key, 0), wb.get(key, 0))
        elif not hard:
            aside("rows", field, wa.get(key, 0), wb.get(key, 0))
        elif code in COUNT_NOT_PINNED and wa.get(key, 0) and wb.get(key, 0):
            aside("records", field, wa.get(key, 0), wb.get(key, 0))
        else:
            add("exact", field, wa.get(key, 0), wb.get(key, 0))
    # predictions
    if timed:
        pa, pb = app["predictions"], ref["predictions"]
        if [row[0] for row in pa] != [row[0] for row in pb]:
            add("exact", "predictions.csv (rows)", f"{len(pa)} rows", f"{len(pb)} rows")
        else:
            off = [row[0] for row, other in zip(pa, pb) if not same_cell(row[1], other[1])]
            if off:
                add("exact", "predictions.csv (y_true)", f"{len(off)} rows differ", "", off[:5])
            far = [row[0] for row, other in zip(pa, pb)
                   if not close(float(row[2]), float(other[2]))]
            if far:
                add("fit", "predictions.csv (y_pred)", f"{len(far)} of {len(pa)} rows differ",
                    "", "the same rows on both sides: only the fits can differ")
    return found


# ---------------------------------------------------------------------- main --

def _env():
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def _run_chunk(parent, seed, numbers, out_root, data_dir, side):
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(parent),
           "--seed", str(seed), "--only", ",".join(map(str, numbers)), "--work", str(out_root),
           "--data", str(data_dir), "--side", side]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(out_root), env=_env(), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=CHUNK_SECONDS)
        tail = (proc.stderr or "")[-800:]
    except subprocess.TimeoutExpired:
        tail = f"the worker did not finish within {CHUNK_SECONDS} seconds"
    return side, numbers, tail, time.perf_counter() - started


def _copy_roles(out):
    """What became of the columns that copy the target, for the record."""
    if out["kind"] != "result":
        return None
    return {c["name"]: [c["role"], c["reason"]] for c in out["doc"].get("columns", [])
            if c.get("name") in ("y_copy", "y_near", "after")}


def _load(out_root, side, number):
    path = Path(out_root) / side / f"{number:05d}.json"
    if not path.is_file():
        return {"number": number, "kind": "missing", "seconds": 0.0}
    return json.loads(path.read_text(encoding="utf-8"))


def table_class(findings):
    """The class of one table, by its worst finding: `harness`, `fails`, `open`,
    `stated limit`, or `agrees`. What is not compared does not count."""
    classes = {f["class"] for f in findings}
    if "harness" in classes:
        return "harness"
    if classes & set(FAILS):
        return "fails"
    if "open" in classes:
        return "open"
    if "limit" in classes:
        return "stated limit"
    return "agrees"


def again(package, reference, seed, number):
    """The call that makes one table again and runs both sides on it."""
    default = HERE / "reference" / "mlprofessor"
    other = "" if reference == default else f" --reference {reference}"
    return (f"python verify/generated_tables.py {package}{other} --seed {seed} "
            f"--only {number} --keep")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(
        prog="generated_tables.py",
        description="Generated tables: a package against the reference implementation.")
    parser.add_argument("package", nargs="?", help="the package folder to judge (mlprofessor)")
    parser.add_argument("--reference", default=None,
                        help="the reference's package folder (default verify/reference/mlprofessor)")
    parser.add_argument("--tables", type=int, default=DEFAULT_TABLES,
                        help=f"tables 0 to N-1 (default {DEFAULT_TABLES})")
    parser.add_argument("--ninth", type=int, default=DEFAULT_NINTH,
                        help="tables of the ninth pass's series, from 1,000,000 (default 200)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"the generator's seed (default {DEFAULT_SEED})")
    parser.add_argument("--only", default=None, help="table numbers, comma-separated")
    parser.add_argument("--jobs", type=int, default=max(2, min(24, (os.cpu_count() or 4) // 2)),
                        help="worker processes side by side")
    parser.add_argument("--json", default=None,
                        help="write every table's class and every finding to this file")
    parser.add_argument("--keep", action="store_true", help="keep the working folder")
    parser.add_argument("--describe", action="store_true", help="print the tables and stop")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--work", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--data", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--side", default=None, help=argparse.SUPPRESS)
    try:
        args = parser.parse_args(argv)
        numbers = [int(x) for x in args.only.split(",") if x.strip()] if args.only \
            else list(range(args.tables)) + [NINTH_FROM + i for i in range(args.ninth)]
    except SystemExit as exc:
        if exc.code in (0, None):
            return EXIT_OK
        print("HARNESS ERROR: bad arguments (this says nothing about the package)")
        return EXIT_HARNESS
    except ValueError:
        print("HARNESS ERROR: --only takes table numbers, comma-separated")
        return EXIT_HARNESS
    if args.worker:
        worker(Path(args.worker), args.seed, numbers, args.work, args.data, args.side)
        return EXIT_OK
    if args.describe:
        for number in numbers:
            print(describe(make_table(args.seed, number)))
        return EXIT_OK
    if not args.package:
        print("HARNESS ERROR: no package folder given")
        return EXIT_HARNESS
    package = Path(args.package).resolve()
    default_reference = HERE / "reference" / "mlprofessor"
    reference = Path(args.reference).resolve() if args.reference else default_reference
    for folder in (package, reference):
        if not (folder / "__init__.py").is_file() or folder.name != "mlprofessor":
            print(f"HARNESS ERROR: {folder} is not a package folder named mlprofessor")
            return EXIT_HARNESS
    if args.jobs < 1 or not numbers:
        print("HARNESS ERROR: --jobs must be at least 1, and there must be a table to run")
        return EXIT_HARNESS
    try:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="run-", dir=ARTIFACTS)).resolve()
        data_dir = work / "data"
        data_dir.mkdir()
        for side in ("application", "reference"):
            (work / side).mkdir()
    except OSError as exc:
        print(f"HARNESS ERROR: {ARTIFACTS} is not writable: {exc}")
        return EXIT_HARNESS
    started = time.perf_counter()
    cases = {}
    for number in numbers:
        case = make_table(args.seed, number)
        cases[number] = case
        if case["kind"] == "csv":
            write_csv(case, csv_path(data_dir, number))
    made = time.perf_counter() - started
    chunks = [numbers[i:i + CHUNK] for i in range(0, len(numbers), CHUNK)]
    jobs = [(package.parent, "application", c) for c in chunks] \
        + [(reference.parent, "reference", c) for c in chunks]
    print(f"generated tables: {len(numbers)} table(s) from seed {args.seed}, {len(jobs)} worker "
          f"run(s), {args.jobs} at a time, under {work}")
    print(f"  package:   {package}\n  reference: {reference}")
    problems = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for side, chunk, tail, _ in pool.map(
                lambda j: _run_chunk(j[0], args.seed, j[2], work, data_dir, j[1]), jobs):
            missing = [k for k in chunk if not (work / side / f"{k:05d}.json").is_file()]
            if missing:
                problems.append((side, missing, tail))
    # a worker that died takes the rest of its chunk with it: those tables run again, alone
    for side, missing, tail in problems:
        parent = package.parent if side == "application" else reference.parent
        for number in missing:
            _, _, tail, _ = _run_chunk(parent, args.seed, [number], work, data_dir, side)
            if not (work / side / f"{number:05d}.json").is_file():
                print(f"  table {number} ({side}): no outcome. {tail.strip()[-300:]}")
    wall = time.perf_counter() - started

    report, seconds = [], {"application": 0.0, "reference": 0.0}
    outcomes = {side: {"result": 0, "error": 0, "other": 0} for side in seconds}
    codes, schemes = {side: {} for side in seconds}, {side: {} for side in seconds}
    tables = {"agrees": 0, "stated limit": 0, "open": 0, "fails": 0, "harness": 0}
    by_name, by_fail, asides = {}, {}, {}
    for number in numbers:
        case = cases[number]
        outs = {"application": _load(work, "application", number),
                "reference": _load(work, "reference", number)}
        for side, out in outs.items():
            seconds[side] += out.get("seconds", 0.0)
            outcomes[side][out["kind"] if out["kind"] in outcomes[side] else "other"] += 1
            if out["kind"] == "error":
                codes[side][out["code"]] = codes[side].get(out["code"], 0) + 1
            elif out["kind"] == "result":
                scheme = str(at(out["doc"], "validation.scheme"))
                schemes[side][scheme] = schemes[side].get(scheme, 0) + 1
        try:
            found = compare(case, outs["application"], outs["reference"], data_dir)
        except Exception as exc:  # noqa: BLE001
            found = [{"class": "harness", "name": None, "field": "the comparison itself",
                      "application": f"{type(exc).__name__}: {exc}", "reference": "",
                      "note": traceback.format_exc()[-900:]}]
        for item in found:
            if item["class"] in ("open", "limit"):
                by_name[item["name"]] = by_name.get(item["name"], set()) | {number}
            elif item["class"] == "not_compared":
                asides[item["name"]] = asides.get(item["name"], 0) + 1
            elif item["class"] in FAILS:
                by_fail[item["class"]] = by_fail.get(item["class"], 0) + 1
        kind = table_class(found)
        tables[kind] += 1
        report.append({"number": number, "class": kind, "description": describe(case),
                       "args": case["args"], "application": brief(outs["application"]),
                       "reference": brief(outs["reference"]),
                       "copies": [_copy_roles(outs["application"]), _copy_roles(outs["reference"])],
                       "seconds": [outs["application"].get("seconds"),
                                   outs["reference"].get("seconds")],
                       "findings": found})

    # The tables that fail in full; then the two lists, one line a finding; then what
    # was not compared, by kind.
    for wanted, title in (("harness", "HARNESS: no verdict on these tables"),
                          ("fails", "FAILS: on neither list")):
        entries = [e for e in report if e["class"] == wanted]
        if entries:
            print(f"\n{title}")
        for entry in entries:
            print(f"\n{entry['description']}")
            print(f"  application: {entry['application']}\n  reference:   {entry['reference']}")
            for f in entry["findings"]:
                label = f["class"] + (f" {f['name']}" if f["name"] else "")
                print(f"  [{label}] {f['field']}: {f['application']!r} | {f['reference']!r}"
                      + (f"  ({str(f['note'])[:500]})" if f["note"] else ""))
            print(f"  again: {again(package, reference, args.seed, entry['number'])}")
    for names, title in ((STATED_LIMITS, "STATED LIMITS"), (OPEN, "OPEN: owned, not yet built")):
        print(f"\n{title}")
        if not names:
            print("  none is known")
        for name, about in names.items():
            hit = sorted(by_name.get(name, ()))
            owner = about if isinstance(about, str) else f"{about[0]}, {about[1]}"
            print(f"  {name} ({owner}): {len(hit)} table(s)"
                  + (f": {', '.join(map(str, hit[:25]))}{' ...' if len(hit) > 25 else ''}"
                     if hit else ""))
            if not isinstance(about, str):
                print(f"      {about[2]}")
    print("\nNOT COMPARED (no part of the verdict): differences by kind")
    for why, text in NOT_COMPARED.items():
        print(f"  {asides.get(why, 0):>5}  {text}")

    print()
    for side in ("application", "reference"):
        print(f"{side}: {outcomes[side]['result']} result(s) {dict(sorted(schemes[side].items()))}"
              f", {outcomes[side]['error']} named error(s) {dict(sorted(codes[side].items()))}"
              f", {outcomes[side]['other']} other")
    print(f"time: {wall:.0f} s wall-clock at {args.jobs} worker(s) ({made:.0f} s making the "
          f"tables); in the tool: application {seconds['application']:.0f} s, reference "
          f"{seconds['reference']:.0f} s")
    counts = (f"{len(numbers)} tables from seed {args.seed}: {tables['agrees']} agree, "
              f"{tables['stated limit']} stated limit, {tables['open']} open, "
              f"{tables['fails']} fail")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"package": str(package), "reference": str(reference), "seed": args.seed,
             "tables": len(numbers), "jobs": args.jobs, "wall_seconds": round(wall, 1),
             "tool_seconds": seconds, "outcomes": outcomes, "errors": codes, "schemes": schemes,
             "tables_by_class": tables,
             "open": {name: sorted(by_name.get(name, ())) for name in OPEN},
             "stated_limits": {name: sorted(by_name.get(name, ())) for name in STATED_LIMITS},
             "failing_findings_by_class": by_fail, "not_compared": asides,
             "classes": {str(e["number"]): e["class"] for e in report},
             "tables_with_findings": [e for e in report if e["findings"]]},
            indent=1, default=str), encoding="utf-8")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"kept: {work}")
    if tables["harness"]:
        print(f"HARNESS ERROR: {tables['harness']} table(s) have no outcome on one side, or "
              "could not be compared. This says nothing about the package")
        print(f"GENERATED TABLES: HARNESS ERROR ({counts})")
        return EXIT_HARNESS
    if tables["fails"]:
        print(f"GENERATED TABLES: FAIL ({counts})")
        return EXIT_DIFF
    print(f"GENERATED TABLES: PASS ({counts})")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
