r"""Make the reference implementation of the ninth pass (D77, D78, D80) from the
reference as it stood: the rules as ruled, with no prototype switch, and a hook for each
new broken variant (F29 to F35, M52 to M69).

    python -B make_reference10.py <reference folder as it stood> <out folder>

Every replacement must match exactly once, or the script stops. `_variant.py` is
written separately.
"""
import shutil
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
DST.mkdir(parents=True, exist_ok=True)
for name in ("__init__.py", "__main__.py", "errors.py", "extensions.py", "stages.py"):
    shutil.copyfile(SRC / name, DST / name)
text = (SRC / "core.py").read_bytes().decode("utf-8")
assert "\r" not in text


def sub(old, new, count=1):
    global text
    found = text.count(old)
    assert found == count, f"{found} matches, expected {count}, for:\n{old[:200]}"
    text = text.replace(old, new)


def splice(start, end, new):
    """Replace everything from `start` up to, not including, `end`."""
    global text
    assert text.count(start) == 1 and text.count(end) == 1, (start[:60], end[:60])
    a, b = text.index(start), text.index(end)
    assert a < b
    text = text[:a] + new + text[b:]


# ------------------------------------------------------------- the header --
sub('''Judge fixture: the gate measures its bands on this code and breaks it in thirty-seven
named ways; the test suite is proved against those and against fifty-one smaller
faults (see _variant.py).''', '''Judge fixture: the gate measures its bands on this code and breaks it in forty-four
named ways; the test suite is proved against those and against sixty-nine smaller
faults (see _variant.py).''')
sub("import warnings as _pywarnings\n",
    "import warnings as _pywarnings\nfrom concurrent.futures import ThreadPoolExecutor\n")

# ---------------------------------------------------------- pinned values --
sub('''# D74: under the grouped scheme, a fold that trains on fewer rows than this and on
# fewer rows than it validates on: no choice between configurations is made.
SELECTION_FLOOR = 50
''', '''# D74: under the grouped scheme, a fold that trains on fewer rows than this and on
# fewer rows than it validates on: no choice between configurations is made.
SELECTION_FLOOR = 50
# D77, D78, D80: what the headline rests on. Each feature's holdout values are shuffled
# REST_SHUFFLES times; a column is named when its fall is at least REST_FALL and at least
# the half-width of the headline skill's interval (or of its stand-in), at most REST_MOST
# of them. A headline skill of NEAR_PERFECT or more carries a warning. Two numeric columns
# of the final fit's matrix are copies when they agree, or one agrees with the other's
# negation, to within COPY_TOLERANCE.
REST_SHUFFLES = 5
REST_FALL = 0.05
REST_MOST = 5
NEAR_PERFECT = 0.98
COPY_TOLERANCE = 1e-9
# D77, D78: the leak screen's reading by levels. A numeric column is read by its levels
# when its training rows hold at least LEVEL_ROWS rows for each distinct value. The
# reading has a value when at least LEVEL_SCORED validation rows are scored and, in a
# classification, at least LEVEL_SCORED_CLASS of each class (or every validation row of
# that class).
LEVEL_ROWS = 5
LEVEL_SCORED = 20
LEVEL_SCORED_CLASS = 10
# D77: under time order, a holdout of this share of the rows or more is named.
FAR_ABOVE = 0.5
''')
sub('''# D66: a whole number reads as a calendar date written YYYYMMDD when its year is in''',
    '''# D77: what is said of a time-ordered holdout far above a quarter. A fixed text.
LARGE_HOLDOUT = ("The models were fitted on the rows before it and the headline is measured "
                 "on it: fewer rows were learned from than the split usually leaves.")
# D66: a whole number reads as a calendar date written YYYYMMDD when its year is in''')

# ---------------------------------------------------- the column records --
sub('''               "n_missing", "n_coerced", "n_infinite", "n_imputed"]''',
    '''               "n_missing", "n_coerced", "n_infinite", "n_imputed",
               "shuffled_score", "shuffled_skill", "skill_fall"]''')
sub("                   n_coerced=0, n_infinite=0, n_imputed=0)",
    "                   n_coerced=0, n_infinite=0, n_imputed=0,\n"
    "                   shuffled_score=None, shuffled_skill=None, skill_fall=None)")

# ------------------------------------------------------- the summary line --
sub('''    for w in d["warnings"]:                 # one line each: the code, then the column''',
    '''    # what the headline rests on (D78, D80): the columns named, or that there is none.
    # M65: nothing is printed when the list is empty
    rests, floor = h.get("rests_on") or [], h.get("rests_on_floor")
    if rests or BROKEN != "M65":
        lines.append("rests on" + ("" if floor is None else f" (columns whose fall is at least "
                                                             f"{floor:.6g})") + ": "
                     + (", ".join(f"{e['column']!r} (fall {e['skill_fall']:.6g})"
                                  for e in rests) if rests else "none"))
    for w in d["warnings"]:                 # one line each: the code, then the column''')

# ---------------------------------------------------- a prediction alone --
sub('''def _auc(y, score, ties_share=True):''',
    '''def _predict(model, x_new, classification):
    """What a fitted model predicts for `x_new`, read as `_fit_predict` reads it."""
    if classification:
        proba = np.asarray(model.predict_proba(x_new), dtype="float64")
        if proba.ndim != 2 or proba.shape != (len(x_new), 2):
            raise ValueError(f"predict_proba returned shape {proba.shape}; expected "
                             f"({len(x_new)}, 2)")
        return proba[:, 0 if BROKEN == "F16" else 1].copy()
    values = np.asarray(model.predict(x_new), dtype="float64").reshape(-1)
    if len(values) != len(x_new):
        raise ValueError(f"predict returned {len(values)} values for {len(x_new)} rows")
    return values


def _auc(y, score, ties_share=True):''')

# ----------------------------------------------------- D: the rows marked --
sub('''    Returns (folds, rows set aside, whether every row is to be scored). F25:
    nothing is ever set aside and every row is scored. M42: the rows are left out
    of the validation rows only, and stay among each fold's training rows.
    """''', '''    When more rows are marked than may be set aside (D77 point 5, D78 point 7), none is
    set aside; rule 2 still leaves its rows unscored, and its reading is also taken
    against each fold's training median, which the rows far off cannot move.

    Returns (folds, rows set aside, whether every row is to be scored, whether rule 2 is
    also read against the fold's training median). F25: nothing is ever set aside and
    every row is scored. M42: the rows are left out of the validation rows only, and stay
    among each fold's training rows. F34: where too many rows are marked, every row is
    scored, as D74 had it. M63: there rule 2 leaves its rows unscored and is not read
    against the median.
    """''')
sub('''    none = np.array([], dtype=int)
    if classification:
        return folds, none, False
    if BROKEN == "F25":
        return folds, none, True
    pooled, base_preds = _screen_baselines(y, folds, prev, by_entity)
    for _, pred in base_preds:
        if not math.isfinite(_mse(y[pooled], pred)):
            return folds, none, False       # the run stops where a column is measured (D44)''',
    '''    none = np.array([], dtype=int)
    if classification:
        return folds, none, False, False
    if BROKEN == "F25":
        return folds, none, True, False
    pooled, base_preds = _screen_baselines(y, folds, prev, by_entity)
    for _, pred in base_preds:
        if not math.isfinite(_mse(y[pooled], pred)):
            return folds, none, False, False    # the run stops where a column is measured (D44)''')
sub('''    if gone.sum() > allowed:
        return folds, none, True
    if not gone.any():
        return folds, none, False''',
    '''    if gone.sum() > allowed:
        if BROKEN == "F34":
            return folds, none, True, False
        return folds, none, False, BROKEN != "M63"
    if not gone.any():
        return folds, none, False, False''')
sub('''    return kept, np.flatnonzero(gone), False
''', '''    return kept, np.flatnonzero(gone), False, False
''')

# -------------------------------------------------- B: the reading by levels --
sub('''def _screen_column(values, numeric, y, folds, classification, seed, prev, also_fit=None,''',
    '''def _level_predict(keys_fit, y_fit, keys_new, numeric):
    """D77: a column read by its levels. For each new row, the mean target of its level
    among the rows of a fit (for a class, the share of the positive class); not-a-number
    for a level those rows do not hold. A missing cell of a numeric column is no level;
    for a categorical column `(missing)` is a level like any other."""
    out = np.full(len(keys_new), np.nan)
    have = ~np.isnan(keys_fit) if numeric else np.ones(len(keys_fit), dtype=bool)
    if not have.any():
        return out
    levels, inverse = np.unique(keys_fit[have], return_inverse=True)
    with np.errstate(over="ignore", invalid="ignore"):
        means = np.bincount(inverse, weights=y_fit[have], minlength=len(levels)) \\
            / np.bincount(inverse, minlength=len(levels))
    at = np.minimum(np.searchsorted(levels, keys_new), len(levels) - 1)
    seen = levels[at] == keys_new
    out[seen] = means[at[seen]]
    return out


def _levels_scored(truth, seen, classification):
    """D78, D80: whether a reading by levels rests on enough validation rows: at least
    LEVEL_SCORED scored and, in a classification, of each class at least
    LEVEL_SCORED_CLASS or every validation row of it. M59: any number will do."""
    if BROKEN == "M59":
        return bool(seen.any())
    if int(seen.sum()) < LEVEL_SCORED:
        return False
    if classification:
        for one in (0, 1):
            scored, held = int((truth[seen] == one).sum()), int((truth == one).sum())
            if scored < LEVEL_SCORED_CLASS and scored != held:
                return False
    return True


def _screen_column(values, numeric, y, folds, classification, seed, prev, also_fit=None,''')
sub('''                   by_entity=False, every_row=False):''',
    '''                   by_entity=False, every_row=False, by_levels=False, median=False):''')
sub('''    own (D57). F24: no line. M47: no direct reading.
    """''', '''    own. F24: no line. M47: no direct reading.

    `by_levels` (D77 point 3): the column is also read by its levels, the mean target of
    each level among a fold's training rows; validation rows whose level the fold's
    training rows do not hold are not scored for that reader (D78 point 5), and with too
    few scored rows the reading has no value. In a classification a categorical column
    is always so read: it is D74's direct reading. `median` (D78 point 7): rule 2's
    reading is also taken against each fold's training median. F32: no reading by levels,
    and a classification's categorical column read directly as D74 had it, an unseen level
    given the fold's share. M58: an unseen level is given the baseline's prediction and
    scored.
    """''')
sub('''    truth, guess, other, mean_guess, rows = [], [], [], [], []
    for fold in folds:''', '''    truth, guess, other, mean_guess, rows, level, median_guess = [], [], [], [], [], [], []
    old_direct = BROKEN == "F32" or not by_levels and not (classification and not numeric)
    for fold in folds:''')
sub('''        mean_guess.append(np.full(len(b), fold_mean))
        rows.append(b)
        if use_line:''', '''        mean_guess.append(np.full(len(b), fold_mean))
        rows.append(b)
        if median:
            median_guess.append(np.full(len(b), float(np.median(y[fold["train"]]))))
        if use_line:''')
sub('''        elif direct:
            codes_a, codes_b = xa[:, 0].astype(int), xb[:, 0].astype(int)
            share = np.bincount(codes_a, weights=y[a], minlength=len(levels)) \\
                / np.maximum(np.bincount(codes_a, minlength=len(levels)), 1)
            other.append(np.where(codes_b >= 0, share[np.maximum(codes_b, 0)], fold_mean))
    truth, guess = np.concatenate(truth), np.concatenate(guess)''',
    '''        elif direct and old_direct:
            codes_a, codes_b = xa[:, 0].astype(int), xb[:, 0].astype(int)
            share = np.bincount(codes_a, weights=y[a], minlength=len(levels)) \\
                / np.maximum(np.bincount(codes_a, minlength=len(levels)), 1)
            other.append(np.where(codes_b >= 0, share[np.maximum(codes_b, 0)], fold_mean))
        if BROKEN != "F32" and (by_levels or (classification and not numeric)) \\
                and (direct or not classification):
            level.append(_level_predict(xa[:, 0], y[a].astype("float64"), xb[:, 0], numeric))
    truth, guess = np.concatenate(truth), np.concatenate(guess)
    by_level = np.concatenate(level) if level else None
    seen = None if by_level is None else ~np.isnan(by_level)''')
sub('''        skill = 2 * auc - 1
        if other:
            second = _auc(truth, np.concatenate(other))
            if second is not None:
                skill = max(skill, 2 * second - 1)
        return skill, skill''', '''        skill = 2 * auc - 1
        if other:
            second = _auc(truth, np.concatenate(other))
            if second is not None:
                skill = max(skill, 2 * second - 1)
        if by_level is not None:
            if BROKEN == "M58":
                third = _auc(truth, np.where(seen, by_level, np.concatenate(mean_guess)))
            else:
                third = _auc(truth[seen], by_level[seen]) \\
                    if _levels_scored(truth, seen, True) else None
            if third is not None:
                skill = max(skill, 2 * third - 1)
        return skill, skill''')
sub('''    readers = [guess]
    if other:
        line = np.concatenate(other)
        if np.isfinite(line).all() and math.isfinite(_mse(truth, line)):
            readers.append(line)
    with np.errstate(over="ignore", invalid="ignore"):
        base_errors = [(truth - pred) ** 2 for _, pred in base_preds]

        def skill_of(reader, unscored):
            """The lower of the skills against each baseline; None when one has no error."""
            own_errors = (truth - reader) ** 2
            lowest = None
            for base_error in base_errors:
                keep = _scored_rows(own_errors, base_error) if unscored \\
                    else np.ones(len(truth), dtype=bool)
                base_sum = float(base_error[keep].sum())
                if not base_sum > 0:
                    return None             # a baseline with no error cannot be beaten (D36)
                value = 1 - float(own_errors[keep].sum()) / base_sum
                lowest = value if lowest is None else min(lowest, value)
            return lowest

        def best(unscored):
            found = [skill_of(reader, unscored) for reader in readers]
            return None if any(f is None for f in found) else max(found)''',
    '''    # each reader: its predictions for a baseline's predictions, and the rows it scores
    readers = [(lambda base: guess, None)]
    if other:
        line = np.concatenate(other)
        if np.isfinite(line).all() and math.isfinite(_mse(truth, line)):
            readers.append((lambda base: line, None))
    if by_level is not None and np.isfinite(by_level[seen]).all():
        if BROKEN == "M58":
            readers.append((lambda base: np.where(seen, by_level, base), None))
        elif _levels_scored(truth, seen, False):
            readers.append((lambda base: np.where(seen, by_level, base), seen))
    medians = [np.concatenate(median_guess)] if median_guess else []
    with np.errstate(over="ignore", invalid="ignore"):
        base_errors = [(truth - pred) ** 2 for _, pred in base_preds]

        def skill_of(reader, unscored):
            """The lower of the skills against each baseline; None when one has no error.
            Rule 2's reading (`unscored`) is also taken against the fold's training
            median where `median` is set."""
            make, only = reader
            lowest = None
            for k, base_pred in enumerate([p for _, p in base_preds]
                                          + (medians if unscored else [])):
                base_error = (truth - base_pred) ** 2
                own_errors = (truth - make(base_pred)) ** 2
                if only is not None:
                    own_errors, base_error = own_errors[only], base_error[only]
                keep = _scored_rows(own_errors, base_error) if unscored \\
                    else np.ones(len(own_errors), dtype=bool)
                base_sum = float(base_error[keep].sum())
                if not base_sum > 0:
                    if only is not None:
                        return False        # the rows scored hold no error to beat: no reading
                    return None             # a baseline with no error cannot be beaten (D36)
                value = 1 - float(own_errors[keep].sum()) / base_sum
                lowest = value if lowest is None else min(lowest, value)
            return lowest

        def best(unscored):
            found = [skill_of(reader, unscored) for reader in readers]
            found = [f for f in found if f is not False]
            return None if any(f is None for f in found) else max(found)''')

# ---------------------------------------------- C: the order reading by pairs --
sub('''def _far_outside(values):''', '''def _order_pairs(x, y):
    """D77 point 4, D78 point 7: the share against the order with rows of equal column value
    never put in the target's order. For each two neighbouring distinct values of `x`,
    every row of the one is compared with every row of the other: the share of those
    comparisons in which the target is higher in the row of the larger value, and the
    share in which it is lower. Each is averaged over the neighbouring pairs of values, and
    the smaller of the two is returned. With fewer than ORDER_PAIRS neighbouring pairs of
    values it has no value. On a column whose values are all distinct this is the reading
    of D76. F33: rows of equal value in the target's order, as D76 built it. M62: one point
    for each distinct value, the mean target of its rows."""
    if BROKEN == "F33":
        return _against_order(x, y)
    have = ~(np.isnan(x) | np.isnan(y))
    x, y = x[have], y[have]
    if not len(x):
        return None
    levels, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(levels) - 1 < ORDER_PAIRS:
        return None
    if BROKEN == "M62":
        with np.errstate(over="ignore", invalid="ignore"):
            point = np.bincount(inverse, weights=y) / counts
        return min(float((point[1:] > point[:-1]).mean()), float((point[1:] < point[:-1]).mean()))
    _, rank = np.unique(y, return_inverse=True)
    width = int(rank.max()) + 2
    key = np.sort(inverse.astype("int64") * width + rank)
    starts = np.r_[0, np.cumsum(counts)]
    inner = inverse < len(levels) - 1
    this, nxt, r = inverse[inner], inverse[inner] + 1, rank[inner]
    below = np.searchsorted(key, nxt * width + r, side="left") - starts[nxt]
    above = starts[nxt + 1] - np.searchsorted(key, nxt * width + r, side="right")
    weight = 1.0 / (counts[this] * counts[nxt])
    up = np.bincount(this, weights=above * weight, minlength=len(levels) - 1)
    down = np.bincount(this, weights=below * weight, minlength=len(levels) - 1)
    return min(float(up.mean()), float(down.mean()))


def _far_outside(values):''')

# ------------------------------------------------------ the screen in _run --
sub('''        screen_folds, _, every_row = _screen_folds(y, train, split["folds"], screen_prev,
                                                   by_entity, classification)
''', '''        screen_folds, _, every_row, median = _screen_folds(y, train, split["folds"],
                                                           screen_prev, by_entity,
                                                           classification)
''')
sub('''                alone, told = _screen_column(values, numeric, y, screen_folds, classification,
                                             seed, screen_prev,
                                             holdout if BROKEN == "M3" else None, by_entity,
                                             every_row)
''', '''                # D77 point 3: a text column, and a numeric column whose training rows hold
                # at least LEVEL_ROWS rows for each distinct value, is also read by its
                # levels. M60: "few" is counted over every row split, not the training rows
                by_levels = not numeric
                if numeric:
                    held = values[np.arange(len(rows)) if BROKEN == "M60" else train]
                    held = held[~np.isnan(held)]
                    distinct = len(np.unique(held))
                    by_levels = distinct > 0 and len(held) >= LEVEL_ROWS * distinct
                by_its_levels[name] = by_levels and BROKEN != "F32"
                alone, told = _screen_column(values, numeric, y, screen_folds, classification,
                                             seed, screen_prev,
                                             holdout if BROKEN == "M3" else None, by_entity,
                                             every_row, by_levels, median)
''')
sub('''        for name in list(prepared):
            values, numeric = prepared[name][rows], by_name[name]["kind"] == "numeric"
            rule = None''', '''        by_its_levels = {}
        for name in list(prepared):
            values, numeric = prepared[name][rows], by_name[name]["kind"] == "numeric"
            rule = None''')
sub('''                share = _against_order(values[train], y[train])
                before = None if screen_prev is None else _against_order(screen_prev[train],
                                                                         y[train])''',
    '''                share = _order_pairs(values[train], y[train])
                before = None if screen_prev is None else _order_pairs(screen_prev[train],
                                                                       y[train])''')
sub('''                _warn(warnings, "suspected_leak",
                      f"column {name!r} is excluded as a suspected leak: {why}", name)''',
    '''                # D78 point 6: a column of codes may name the entity each row belongs to.
                # M61: the message does not say so
                entity = rule == "predicts_target" and by_its_levels.get(name) \\
                    and BROKEN != "M61"
                _warn(warnings, "suspected_leak",
                      f"column {name!r} is excluded as a suspected leak: {why}"
                      + (". If it names the entity each row belongs to, declare it as "
                         "group_col (--group-col): the rows of one entity are then kept "
                         "together." if entity else ""), name)''')

# ------------------------------------------------ E: a large time holdout --
sub('''    train, holdout = split["train"], split["holdout"]
    for w in warnings:
''', '''    train, holdout = split["train"], split["holdout"]
    if time_ordered and BROKEN not in ("F7", "F35"):
        # D77 point 6: whole timestamps can make the holdout far more than a quarter. M64:
        # only a share above one half is named
        split_rows = len(train) + len(holdout) + len(split["gap"])
        held_share = len(holdout) / split_rows
        if held_share > FAR_ABOVE if BROKEN == "M64" else held_share >= FAR_ABOVE:
            first = int((keys[holdout] == keys[holdout].min()).sum())
            _warn(warnings, "holdout_far_above_quarter",
                  f"the holdout holds {len(holdout)} of the {split_rows} rows "
                  f"({100 * held_share:.1f}%), far more than a quarter: it is made of whole "
                  f"timestamps, and its earliest timestamp alone holds {first} rows. "
                  + LARGE_HOLDOUT)
    for w in warnings:
''')

# --------------------------------------- A: the final fit, kept for the statement --
sub('''    step_makers = extensions.active()["steps"] or []
''', '''    step_makers = extensions.active()["steps"] or []
    final_fit = {}
''')
sub('''        x_fit, x_new = prep.transform(fit_frame), prep.transform(new_frame)
        names = prep.names()
''', '''        x_fit, x_new = prep.transform(fit_frame), prep.transform(new_frame)
        names = prep.names()
        if final:
            final_fit.update(prep=prep, names=names)
''')
sub("    chosen = pred = None\n", "    chosen = pred = chosen_model = chosen_family = None\n")
sub('''        chosen = models[i]
        break
''', '''        chosen, chosen_model, chosen_family = models[i], model, family
        break
''')

# ------------------------------------------------ A: what the headline rests on --
sub('''    # 13. assemble the result
''', '''    # 12a. what the headline rests on (D77 points 1 and 2, D78, D80). For each feature, the
    # chosen configuration's score on the holdout with that feature's part of the holdout
    # matrix moved among the holdout rows, the mean over REST_SHUFFLES shuffles; nothing is
    # refitted and no other number changes. Copies of one another, as the models receive
    # them, are moved together and share one fall.
    rests_on, floor = [], None
    prep, names = final_fit["prep"], final_fit["names"]
    blocks, at = {}, len(prep.numeric)
    for j, name in enumerate(prep.numeric):
        blocks[name] = [j]
    for name, levels in zip(prep.categorical, prep.levels):
        blocks[name] = list(range(at, at + len(levels)))
        at += len(levels)
    custom = bool(chosen_family.get("custom"))
    copy_sets = _copy_sets(prepared, kinds, blocks, x_fit, x_holdout,
                           [values[rows[np.concatenate([train, holdout])]]
                            for values in prepared.values()])

    def shuffled(members):
        """The scores of REST_SHUFFLES shuffles of the holdout values of `members`, or the
        text of what the model raised. M52: the stream depends on the number of cores. F29:
        nothing is moved."""
        rng = np.random.default_rng([seed, 8, by_name[members[0]]["position"]]
                                    + ([n_jobs] if BROKEN == "M52" else []))
        cols = [c for name in members for c in blocks[name]]
        got = []
        for _ in range(REST_SHUFFLES):
            order = rng.permutation(len(holdout))
            moved = x_holdout.copy()
            if BROKEN != "F29":
                moved[:, cols] = x_holdout[order][:, cols]
            new = pd.DataFrame(moved, index=frames[1].index, columns=names) if custom else moved
            try:
                guess = finite(_predict(chosen_model, new, classification), len(holdout))
            except Exception as exc:            # the model cannot predict this matrix
                if BROKEN == "M66":
                    raise
                return f"{type(exc).__name__}: {exc}"
            got.append(_auc(truth, guess) if classification else _mse(truth, guess))
        return got

    # Forests predict on one thread (D25), so their shuffles are spread over the cores; the
    # stream of each set of columns is its own, so the result is the same at any n_jobs.
    if not custom and hasattr(chosen_model, "estimators_") and n_jobs > 1 and len(copy_sets) > 1:
        with ThreadPoolExecutor(max_workers=min(n_jobs, len(copy_sets))) as pool:
            every = list(pool.map(shuffled, copy_sets))
    else:
        every = [shuffled(members) for members in copy_sets]
    for members, got in zip(copy_sets, every):
        mean_score = None
        if not isinstance(got, str):
            with np.errstate(over="ignore", invalid="ignore"):
                mean_score = float(np.mean(got))
            if not math.isfinite(mean_score):
                got, mean_score = f"its score is not finite ({mean_score!r})", None
        for name in members:
            rec = by_name[name]
            if mean_score is None:
                _warn(warnings, "shuffled_score_not_computed",
                      f"what the headline rests on is not stated for column {name!r}: the "
                      f"chosen configuration {chosen['config']} could not score the holdout "
                      f"with its values shuffled ({got})", name)
                continue
            rec["shuffled_score"] = mean_score
            if BROKEN == "F4":
                continue
            if classification:
                without = 2 * mean_score - 1
            else:
                without = 1 - mean_score / baseline["score"] if baseline["score"] > 0 else None
            if skill is not None and without is not None:
                rec["shuffled_skill"], rec["skill_fall"] = without, skill - without
    if BROKEN == "M53":                         # M53: a column that reached no model as well
        for rec in records:
            if rec["role"] == "excluded":
                rec.update(shuffled_score=score, shuffled_skill=skill,
                           skill_fall=None if skill is None else 0.0)
    if skill is not None:
        # D80 point 1: the floor of the list. The half-width of the headline skill's
        # interval; where there is none, that of a bootstrap of the holdout rows taken as
        # if they were independent; never below REST_FALL. M67: REST_FALL alone. M68: no
        # stand-in where there is no interval
        floor = REST_FALL
        if uncertainty["skill_low"] is not None and uncertainty["skill_high"] is not None:
            if BROKEN != "M67":
                floor = max(REST_FALL, (uncertainty["skill_high"] - uncertainty["skill_low"]) / 2)
        elif BROKEN not in ("M67", "M68"):
            half = _rows_half_width(truth, pred, base_pred, classification,
                                    RESAMPLES[profile], seed)
            if half is not None:
                floor = max(REST_FALL, half)
        if BROKEN == "M54":                     # M54: no floor at all
            floor = -math.inf
        order = sorted((r for r in records if r["skill_fall"] is not None),
                       key=(lambda r: (-r["skill_fall"], r["name"])) if BROKEN == "M55"
                       else (lambda r: (-r["skill_fall"], r["position"])))
        named = [r for r in order if r["skill_fall"] >= floor][:REST_MOST]
        rests_on = [{"column": r["name"], "skill_fall": r["skill_fall"]} for r in named]
        if BROKEN == "M54":
            floor = REST_FALL
        # M57: a regression's near-perfect headline carries no warning
        near = skill >= NEAR_PERFECT and not (BROKEN == "M57" and not classification)
        if near and BROKEN not in ("F4", "F30"):
            if named:
                rests = ("It rests most on " + "; ".join(
                    f"{r['name']!r} (skill {r['shuffled_skill']:.3f} with its holdout values "
                    f"shuffled, a fall of {r['skill_fall']:.3f})" for r in named)
                    + ". Check that each of these columns is known before the outcome")
            else:
                rests = (f"No column's fall reaches the floor of {floor:.4g}: shuffling any one "
                         "column's holdout values leaves the skill almost where it is. Check "
                         "that the columns are known before the outcome")
            _warn(warnings, "near_perfect_headline",
                  f"the headline skill is {skill:.4f}, at or above {NEAR_PERFECT}: the level at "
                  "which one column alone is excluded as a suspected leak. " + rests)

    # 13. assemble the result
''')
sub('''    headline.update(largest)
''', '''    headline.update(largest)
    headline["rests_on"] = rests_on
    headline["rests_on_floor"] = floor
''')

# ------------------------------------------------- A: helpers of the statement --
sub('''def _largest_row(errors):''', '''def _copy_sets(prepared, kinds, blocks, x_fit, x_holdout, loaded):
    """D78 point 3, D80 point 4: the features in sets of copies of one another, as the
    models receive them, in file order. Two numeric features are copies when their columns
    of the final fit's matrix agree, or one agrees with the other's negation, to within
    COPY_TOLERANCE in every training row and every holdout row; two categorical features
    when their blocks of level columns hold the same columns in any order. Features of
    different kinds are never copies. F31: every feature is a set of its own. M56: copies
    are features whose values, as loaded, are the same. M69: a negated column is not a
    copy."""
    names = list(prepared)
    if BROKEN == "F31":
        return [[name] for name in names]
    if BROKEN == "M56":
        found = {}
        for name, held in zip(names, loaded):
            key = (kinds[name], held.astype(str).tobytes() if held.dtype == object
                   else np.where(np.isnan(held), np.nan, held + 0.0).tobytes())
            found.setdefault(key, []).append(name)
        return list(found.values())
    whole = np.vstack([x_fit, x_holdout])
    head = min(16, len(whole))

    def block(name):
        part = whole[:, blocks[name]]
        if kinds[name] == "numeric" or not part.shape[1]:
            return part
        return part[:, np.lexsort(part[::-1])]

    seen = {name: block(name) for name in names}
    signs = (1.0,) if BROKEN == "M69" else (1.0, -1.0)

    def alike(a, b):
        x, z = seen[a], seen[b]
        if kinds[a] != kinds[b] or x.shape != z.shape:
            return False
        if kinds[a] == "numeric":
            return any(np.all(np.abs(x[:head] - s * z[:head]) <= COPY_TOLERANCE)
                       and np.all(np.abs(x - s * z) <= COPY_TOLERANCE) for s in signs)
        return bool(np.array_equal(x[:head], z[:head]) and np.array_equal(x, z))

    sets, placed = [], set()
    for name in names:
        if name in placed:
            continue
        members = [name] + [other for other in names if other != name
                            and other not in placed and alike(name, other)]
        placed.update(members)
        sets.append(members)
    return sets


def _rows_half_width(truth, pred, base_pred, classification, n_resamples, seed):
    """D80 point 1: the stand-in for the half-width of the headline skill's interval
    where there is none: the half-width of the middle 95% of the skill over a bootstrap
    of the holdout rows taken as if they were independent, on a stream of its own. None
    when the skill can be computed in fewer than half of the resamples."""
    rng = np.random.default_rng([seed, 10])
    found = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(truth), len(truth))
        if classification:
            auc = _auc(truth[pick], pred[pick])
            if auc is not None:
                found.append(2 * auc - 1)
        else:
            base = _mse(truth[pick], base_pred[pick])
            if base > 0:
                found.append(1 - _mse(truth[pick], pred[pick]) / base)
    if len(found) * 2 < n_resamples:
        return None
    low, high = np.percentile(found, [2.5, 97.5])
    return float(high - low) / 2


def _largest_row(errors):''')

(DST / "core.py").write_bytes(text.encode("utf-8"))
print("wrote", DST / "core.py", len(text))
