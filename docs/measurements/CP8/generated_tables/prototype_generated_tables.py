"""Generated tables: the application against the reference implementation (D56).

PROTOTYPE of CP8 section C2. A generator makes table number i from a seed: the
kinds of column the spec knows, missing values, sizes, both tasks, every
combination of the two statements, `duplicates` in each value. Each table is run
through the application and through the reference, each in a process of its own,
and the two outcomes must agree: the same named error, or the same profile.json
and predictions.csv apart from what the spec leaves open. Agreement proves
neither right; a disagreement is a finding.

    python -B generated_tables.py <package folder> [--reference FOLDER] [--tables N]
           [--seed S] [--only i,j] [--jobs N] [--json FILE] [--keep] [--describe]

Exit codes: 0 no disagreement, 1 at least one, 3 HARNESS ERROR.

What is compared, and how (spec 2.4, 3, 5.2, 5.3, 11):
  exact     a field the table and the arguments alone determine: equal, numbers within 1e-9
  fit       a field determined through fits on rows both must agree on (time order): equal
  property  a field that depends on an open choice (which rows a random or grouped split or
            the fast cap picks, the resamples): each side is held to what the spec pins
  split     reported for information only: the two differ where different rows may explain it
  open      reported for information only: the two differ in something the spec leaves open
            (which of several earned errors is raised, 2.4; a forest's fit, whose rows are
            handed over in an order nothing pins)
Never compared: meta.timing, message wording, columns[].dtype, kind of a non-feature.
"""
from __future__ import annotations

import argparse
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
DEFAULT_TABLES, DEFAULT_SEED = 300, 20261122
CHUNK = 6                       # tables handed to one worker process
CHUNK_SECONDS = 1500
HERE = Path(__file__).resolve().parent
LIMITATION_COUNT = 9

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
        shape = _pick(r, [("equal", 45), ("one_large", 22), ("own", 8), ("few", 10),
                          ("long_tail", 15)])
        if shape == "equal":
            per = r.choice([2, 3, 5, 8])
            who = [j // per for j in range(n)]
        elif shape == "one_large":
            big = int(n * r.choice([0.3, 0.6, 0.8, 0.92]))
            per = r.choice([1, 2, 3])
            who = [0] * big + [1 + j // per for j in range(n - big)]
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
        sentinel = r.random() < 0.25
        if sentinel:
            numbers[int(g.integers(n))] = r.choice([99991231, 0])
        if form == "int":
            cols["daynum"] = pd.Series(numbers, dtype="int64")
        elif form == "text":
            cols["daynum"] = text_series([f"{v:08d}" for v in numbers])
        else:
            cols["daynum"] = pd.Series([np.nan if g.random() < 0.05 else float(v)
                                        for v in numbers])
        notes.append(f"daynum: dates written YYYYMMDD ({form})"
                     + (", one sentinel cell" if sentinel else ""))
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
    if copy_of is not None and r.random() < 0.1:
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

    # ---- the time column and the entity column, as the table holds them
    oracle = {"time_form": None}
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
    blank_lines = sorted(int(v) for v in g.choice(len(frame), size=r.randint(1, 3), replace=False)) \
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
    if o["declares_time"] and o["time_form"] == "dayfirst":
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


def feature_keys(frame, doc):
    """One key per row: equal for two rows that agree in every feature column of step 5
    (a column that is a feature, or was one until the leak screen excluded it), compared
    as spec step 8a compares values. The judge's own reading of the table."""
    import numpy as np
    import pandas as pd
    parts = []
    for rec in doc["columns"]:
        if not (rec["role"] == "feature" or rec.get("reason") == "suspected_leak"):
            continue
        series = frame.iloc[:, rec["position"]]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = pd.Series([v if pd.isna(v) else str(v) for v in series], dtype=object)
        cells = list(series)
        present = [v for v in cells if not _empty(v)]
        texts = [v for v in present if isinstance(v, str)]
        numeric = None
        if not texts:
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
        elif o["time_form"] not in ("dayfirst",):
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
        elif capped:
            kept = n_usable - capped
            sizes = {}
            for i in np.flatnonzero(usable):
                sizes[int(o["who"][i])] = sizes.get(int(o["who"][i]), 0) + 1
            if kept > FAST_CAP or kept <= FAST_CAP - max(sizes.values()):
                bad("the fast cap keeps whole entities, and none left out would have fitted "
                    "(step 6)", f"kept {kept} rows; the largest entity holds {max(sizes.values())}")
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
# A forest draws its rows by their place in the matrix it is handed, and the order of
# the rows of one fit is pinned nowhere (2.6 pins the index, not the order). Measured:
# the application hands a fit its rows in file order, the reference in time order.
ROW_ORDER_FAMILIES = {"forest"}
INFORMATION = {"split", "open"}       # listed, and no part of the verdict
# Spec 2.5 says when each of these is raised and not how many records it makes.
COUNT_NOT_PINNED = {"fold_not_scored", "small_holdout", "duplicate_rows", "no_interval"}


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
    """Every difference between the application's outcome and the reference's."""
    found = []

    def add(kind, field, a, b, note=""):
        found.append({"class": kind, "field": field, "application": a, "reference": b,
                      "note": note})

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
    if app["kind"] == "error" and ref["kind"] == "error":
        if app["code"] != ref["code"]:
            earned = broken_rules(case)
            both = app["code"] in earned and ref["code"] in earned
            add("open" if both else "outcome", "the named error", app["code"], ref["code"],
                ("the table earns both, and which is raised is then open (2.4) | " if both else "")
                + f"application: {app['message'][:300]} | reference: {ref['message'][:300]}")
        return found
    if app["kind"] != ref["kind"]:
        stopped, finished = (app, ref) if app["kind"] == "error" else (ref, app)
        by_rows = stopped["code"] in SPLIT_ERRORS \
            and at(finished["doc"], "validation.scheme") != "time_ordered"
        add("split" if by_rows else "outcome", "a result on one side, a named error on the other",
            brief(app), brief(ref),
            ("with no time column this error can depend on which rows the split or the cap "
             "picked, which is open (11) | " if by_rows else "") + stopped["message"][:400])
        return found

    a, b = app["doc"], ref["doc"]

    def exact(path, kind="exact", note=""):
        x, y = at(a, path), at(b, path)
        if first_difference(x, y):
            add(kind, path, x, y, note)
            return False
        return True

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
            add("split" if leak and not timed else "exact", f"columns[{name}] role, reason, leak rule",
                role_a, role_b, "the leak screen ran on different folds" if leak and not timed else "")
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
        exact(path, "exact" if timed else "split",
              "" if timed else "the resamples, or the units in the holdout, differ")
    for path in ("headline.p_value_reason", "headline.interval_reason"):
        if (at(a, path) is None) != (at(b, path) is None):
            add("exact" if timed else "split", path + " (null or not)", at(a, path), at(b, path))
    ma, mb = a.get("models", []), b.get("models", [])
    by_order = {m.get("config") for m in ma + mb if m.get("family") in ROW_ORDER_FAMILIES}
    order_note = ("a forest draws its rows by their place in the matrix, and the order of the "
                  "rows of one fit is not pinned (2.6)")
    if timed:
        exact("headline.baseline_score")
        exact("baselines")
        exact("cv.n_scored_folds")
        # The same rows on both sides, so only the fits can differ. Where a forest is
        # among the configurations the choice itself may rest on one.
        soft = bool(by_order)
        for path in ("headline.score", "headline.skill", "headline.model", "cv.chosen",
                     "cv.chosen_mean", "cv.chosen_std"):
            exact(path, "open" if soft else "fit",
                  order_note if soft else "the same rows on both sides: only the fits can differ")
    elif at(a, "cv.n_scored_folds") != at(b, "cv.n_scored_folds"):
        add("split", "cv.n_scored_folds", at(a, "cv.n_scored_folds"), at(b, "cv.n_scored_folds"))
    shape_a = [(m.get("config"), m.get("family"), json.dumps(m.get("params"), sort_keys=True))
               for m in ma]
    shape_b = [(m.get("config"), m.get("family"), json.dumps(m.get("params"), sort_keys=True))
               for m in mb]
    if shape_a != shape_b:
        add("exact", "models (configurations, families, params)", shape_a, shape_b)
    elif timed:
        # `chosen` follows the ranking, which a forest's score can move: compared
        # for the families whose fit does not depend on the order of the rows only
        # when no forest is in the table.
        keys = ["status", "error", "fold_scores", "cv_mean", "cv_std"] + ([] if by_order
                                                                          else ["chosen"])
        steady = [{k: m.get(k) for k in keys} for m in ma if m.get("config") not in by_order]
        other = [{k: m.get(k) for k in keys} for m in mb if m.get("config") not in by_order]
        for x, y in zip(steady, other):         # the error text is wording: null or not
            x["error"], y["error"] = x["error"] is None, y["error"] is None
        difference = first_difference(steady, other, "models (linear, tree, boosting)")
        if difference:
            add("fit", "models", difference, "", "the same rows on both sides")
        moved = [m.get("config") for m, o in zip(ma, mb) if m.get("config") in by_order
                 and first_difference(m.get("fold_scores"), o.get("fold_scores"))]
        if moved:
            add("open", "models (fold scores of the forests)", f"{len(moved)} configurations "
                "differ", "", order_note)
    else:
        status_a, status_b = [m.get("status") for m in ma], [m.get("status") for m in mb]
        if status_a != status_b:
            add("split", "models (status)", status_a, status_b)
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
        if code == "autocorrelated_panel_target":
            hard = False
        note = "" if hard else "may depend on the rows a split picked, or on an open choice"
        if code == "autocorrelated_panel_target":
            note = "which estimator of the autocorrelation is used is open (11)"
        kind = "exact" if hard else "split"
        if hard and code in COUNT_NOT_PINNED and wa.get(key, 0) and wb.get(key, 0):
            kind, note = "open", "both raise it; how many records of this code is not pinned (2.5)"
        add(kind, f"warnings[{code}, {key[1]}] (count)", wa.get(key, 0), wb.get(key, 0), note)
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
                add("open" if by_order else "fit", "predictions.csv (y_pred)",
                    f"{len(far)} of {len(pa)} rows differ", "",
                    order_note if by_order else "the same rows on both sides: only the fits "
                                                "can differ")
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


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="Generated tables: a package against the "
                                                 "reference implementation.")
    parser.add_argument("package", nargs="?", help="the package folder to judge (mlprofessor)")
    parser.add_argument("--reference", default=None, help="the reference's package folder")
    parser.add_argument("--tables", type=int, default=DEFAULT_TABLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--only", default=None, help="table numbers, comma-separated")
    parser.add_argument("--jobs", type=int, default=max(2, min(24, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--json", default=None, help="write every outcome and finding here")
    parser.add_argument("--keep", action="store_true", help="keep the working folder")
    parser.add_argument("--describe", action="store_true", help="print the tables and stop")
    parser.add_argument("--tmp", default=None, help="where the working folder is made")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--work", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--data", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--side", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    numbers = [int(x) for x in args.only.split(",") if x.strip()] if args.only \
        else list(range(args.tables))
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
    reference = Path(args.reference).resolve() if args.reference \
        else HERE / "reference" / "mlprofessor"
    for folder in (package, reference):
        if not (folder / "__init__.py").is_file() or folder.name != "mlprofessor":
            print(f"HARNESS ERROR: {folder} is not a package folder named mlprofessor")
            return EXIT_HARNESS
    work = Path(tempfile.mkdtemp(prefix="mlprof-generated-", dir=args.tmp)).resolve()
    data_dir = work / "data"
    data_dir.mkdir()
    for side in ("application", "reference"):
        (work / side).mkdir()
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
            _run_chunk(parent, args.seed, [number], work, data_dir, side)
            if not (work / side / f"{number:05d}.json").is_file():
                print(f"  table {number} ({side}): no outcome. {tail.strip()[-300:]}")
    wall = time.perf_counter() - started
    report, counts, seconds = [], {}, {"application": 0.0, "reference": 0.0}
    outcomes = {"result": 0, "error": 0, "other": 0}
    codes = {}
    for number in numbers:
        case = cases[number]
        app, ref = _load(work, "application", number), _load(work, "reference", number)
        seconds["application"] += app.get("seconds", 0.0)
        seconds["reference"] += ref.get("seconds", 0.0)
        outcomes[ref["kind"] if ref["kind"] in outcomes else "other"] += 1
        if ref["kind"] == "error":
            codes[ref["code"]] = codes.get(ref["code"], 0) + 1
        try:
            found = compare(case, app, ref, data_dir)
        except Exception as exc:  # noqa: BLE001
            found = [{"class": "harness", "field": "the comparison itself",
                      "application": f"{type(exc).__name__}: {exc}", "reference": "",
                      "note": traceback.format_exc()[-900:]}]
        for item in found:
            counts[item["class"]] = counts.get(item["class"], 0) + 1
        report.append({"number": number, "description": describe(case), "args": case["args"],
                       "application": brief(app), "reference": brief(ref),
                       "copies": [_copy_roles(app), _copy_roles(ref)],
                       "seconds": [app.get("seconds"), ref.get("seconds")], "findings": found})
    counted = [e for e in report if any(f["class"] not in INFORMATION for f in e["findings"])]
    for entry in report:
        if not entry["findings"]:
            continue
        print(f"\n{entry['description']}")
        print(f"  application: {entry['application']}\n  reference:   {entry['reference']}")
        for f in entry["findings"]:
            print(f"  [{f['class']}] {f['field']}: {f['application']!r} | {f['reference']!r}"
                  + (f"  ({str(f['note'])[:500]})" if f["note"] else ""))
        print(f"  again: python -B {Path(__file__).name} {package} --reference {reference} "
              f"--seed {args.seed} --only {entry['number']} --keep")
    print(f"\nreference outcomes: {outcomes['result']} result(s), {outcomes['error']} named "
          f"error(s) {dict(sorted(codes.items()))}, {outcomes['other']} other")
    print(f"findings by class: {dict(sorted(counts.items())) or 'none'}")
    print(f"time: {wall:.0f} s wall-clock ({made:.0f} s making the tables); in the tool: "
          f"application {seconds['application']:.0f} s, reference {seconds['reference']:.0f} s")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"seed": args.seed, "tables": len(numbers), "wall_seconds": round(wall, 1),
             "tool_seconds": seconds, "reference_outcomes": outcomes, "reference_errors": codes,
             "findings_by_class": counts, "tables_with_findings": report}, indent=1, default=str),
            encoding="utf-8")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"kept: {work}")
    if counts.get("harness"):
        print("GENERATED TABLES: HARNESS ERROR (an outcome is missing, or the comparison failed)")
        return EXIT_HARNESS
    if counted:
        print(f"GENERATED TABLES: {len(counted)} of {len(numbers)} table(s) with a disagreement")
        return EXIT_DIFF
    print(f"GENERATED TABLES: PASS ({len(numbers)} tables, no disagreement)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
