"""Make the CP10 prototype of the reference: the reference as it stands, with the
proposed rules of sections A to E written in behind the switches of `_proto.py`.

    python -B make_proto.py <reference folder (mlprofessor)> <prototype folder (mlprofessor)>

Every replacement must match exactly once, or the script stops.
"""
import shutil
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
HERE = Path(__file__).resolve().parent
DST.mkdir(parents=True, exist_ok=True)
for name in ("__init__.py", "__main__.py", "_variant.py", "errors.py", "extensions.py", "stages.py"):
    shutil.copyfile(SRC / name, DST / name)
shutil.copyfile(HERE / "_proto.py", DST / "_proto.py")
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


# ------------------------------------------------------------------ imports --
sub("from ._variant import BROKEN\n",
    "from ._variant import BROKEN\nfrom . import _proto as P\n")
sub("import warnings as _pywarnings\n",
    "import warnings as _pywarnings\nfrom concurrent.futures import ThreadPoolExecutor\n")

# ------------------------------------------------------------ A: the columns --
sub('''               "n_missing", "n_coerced", "n_infinite", "n_imputed"]''',
    '''               "n_missing", "n_coerced", "n_infinite", "n_imputed",
               "shuffled_score", "shuffled_skill", "skill_fall"]''')
sub("                   n_coerced=0, n_infinite=0, n_imputed=0)",
    "                   n_coerced=0, n_infinite=0, n_imputed=0,\n"
    "                   shuffled_score=None, shuffled_skill=None, skill_fall=None)")

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

# ------------------------------------------------- D: the rows set aside --
sub('''    if gone.sum() > allowed:
        return folds, none, True
''', '''    if P.DEBUG:
        P.LOG["marked"] = (int(gone.sum()), int(allowed), int(len(train)))
    if gone.sum() > allowed:
        if P.D == "always":         # rule 2 leaves its rows unscored all the same
            return folds, none, False, None
        if P.D == "unmarked":       # ... but never a row that is marked
            return folds, none, False, gone
        if P.D == "median":         # ... and against a baseline the rows left in cannot move
            return folds, none, False, "median"
        return folds, none, True, None
''')
sub("        return folds, none, False\n", "        return folds, none, False, None\n", 2)
sub("            return folds, none, False       # the run stops",
    "            return folds, none, False, None     # the run stops")
sub("        return folds, none, True\n", "        return folds, none, True, None\n")
sub("    return kept, np.flatnonzero(gone), False\n",
    "    return kept, np.flatnonzero(gone), False, None\n")

splice("def _scored_rows(own_errors, base_errors):", "def _screen_column(", '''def _scored_rows(own_errors, base_errors, eligible=None):
    """Step 8 (D74): which pooled validation rows count for rule 2. The
    `SCREEN_UNSCORED` of the rows that the column misses by most are not scored,
    and the same share of the rows that the baseline misses by most: k = that
    share of the rows, rounded up, and at least one, from each. Among equal errors
    the earlier row goes. With too few rows for that (2k >= n) every row counts.
    `eligible` (D77): when rows are marked and could not be set aside, only a row
    that is not marked may be left unscored."""
    n = len(own_errors)
    keep = np.ones(n, dtype=bool)
    k = max(1, math.ceil(SCREEN_UNSCORED * n))
    if 2 * k < n:
        for errors in (own_errors, base_errors):
            order = np.argsort(-errors, kind="stable")
            if eligible is not None:
                order = order[eligible[order]]
            keep[order[:k]] = False
    return keep


def _level_predict(keys_fit, y_fit, keys_new, numeric):
    """D77: a column read by its levels. The mean target of each level among the rows
    of a fit, for each new row; not-a-number for a level those rows do not hold. A
    missing cell of a numeric column is no level."""
    out = np.full(len(keys_new), np.nan)
    have = ~np.isnan(keys_fit) if numeric else np.ones(len(keys_fit), dtype=bool)
    if not have.any():
        return out
    levels, inverse = np.unique(keys_fit[have], return_inverse=True)
    with np.errstate(over="ignore", invalid="ignore"):
        means = np.bincount(inverse, weights=y_fit[have], minlength=len(levels)) \\
            / np.bincount(inverse, minlength=len(levels))
    at = np.searchsorted(levels, keys_new)
    at = np.minimum(at, len(levels) - 1)
    seen = levels[at] == keys_new
    out[seen] = means[at[seen]]
    return out


''')

splice("def _screen_column(", "def _against_order(x, y):", '''def _screen_column(values, numeric, y, folds, classification, seed, prev, also_fit=None,
                   by_entity=False, every_row=False, level_mode=0, marked=None, log=None):
    """D11, D74, D77: one column's cross-validated skill over the folds of the screen.

    `level_mode`: 0, the column is not read by its levels; 1, it is, for the log
    only; 2, it is. `marked`: the rows that may not be left unscored (D77).
    """
    cls = DecisionTreeClassifier if classification else DecisionTreeRegressor
    use_line = numeric and not classification and BROKEN != "F24"
    direct = classification and BROKEN != "M47"
    truth, guess, other, mean_guess, rows, level, median_guess = [], [], [], [], [], [], []
    robust = isinstance(marked, str)
    if robust:
        marked = None
    for fold in folds:
        if not fold["scored"] or not len(fold["train"]) or not len(fold["valid"]):
            continue
        a, b = fold["train"], fold["valid"]
        fold_mean = _mean(y[a])
        if also_fit is not None:
            a = np.concatenate([a, also_fit])
        if numeric:
            xa, xb = values[a].reshape(-1, 1), values[b].reshape(-1, 1)
        else:
            levels = {lv: i for i, lv in enumerate(sorted(set(values[a].astype(str))))}
            xa = np.array([levels[v] for v in values[a].astype(str)], dtype="float64").reshape(-1, 1)
            xb = np.array([levels.get(v, -1) for v in values[b].astype(str)],
                          dtype="float64").reshape(-1, 1)
        model = cls(max_depth=6, min_samples_leaf=5, random_state=seed).fit(xa, y[a])
        guess.append(model.predict_proba(xb)[:, 1] if classification else model.predict(xb))
        truth.append(y[b])
        mean_guess.append(np.full(len(b), fold_mean))
        rows.append(b)
        if robust:
            median_guess.append(np.full(len(b), float(np.median(y[fold["train"]]))))
        if use_line:
            other.append(_line_predict(values[a], y[a], values[b], fold_mean))
        elif direct and numeric:
            stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1,
                                           random_state=seed).fit(xa, y[a])
            other.append(stump.predict_proba(xb)[:, 1])
        elif direct:
            codes_a, codes_b = xa[:, 0].astype(int), xb[:, 0].astype(int)
            share = np.bincount(codes_a, weights=y[a], minlength=len(levels)) \\
                / np.maximum(np.bincount(codes_a, minlength=len(levels)), 1)
            other.append(np.where(codes_b >= 0, share[np.maximum(codes_b, 0)], fold_mean))
        if level_mode and not (classification and not numeric):
            level.append(_level_predict(xa[:, 0], y[a].astype("float64"), xb[:, 0], numeric))
    truth, guess = np.concatenate(truth), np.concatenate(guess)
    if classification:
        auc = _auc(truth, guess)
        if auc is None:
            return None, None
        found = {"tree": 2 * auc - 1}
        if other:
            second = _auc(truth, np.concatenate(other))
            if second is not None:
                found["direct"] = 2 * second - 1
        if level:
            by_level = np.concatenate(level)
            third = _auc(truth, np.where(np.isnan(by_level), np.concatenate(mean_guess), by_level))
            if third is not None:
                found["levels"] = 2 * third - 1
        if log is not None:
            log.update({name: (value, value) for name, value in found.items()})
        if level_mode < 2:
            found.pop("levels", None)
        skill = max(found.values())
        return skill, skill
    mean_pred = np.concatenate(mean_guess)
    base_preds = [("training_mean", mean_pred)]
    if prev is not None and by_entity:
        before = prev[np.concatenate(rows)]
        base_preds.append(("previous_value", np.where(np.isnan(before), mean_pred, before)))
    elif prev is not None:
        base_preds.append(("previous_value",
                           _previous_of(prev, np.concatenate(rows), "a validation block")))
    for name, pred in base_preds:
        error = _mse(truth, pred)
        if not math.isfinite(error):
            raise _not_finite(f"In the leak screen, the squared error of the baseline {name} "
                              "over the validation rows of the selection folds", error,
                              TARGET_CAUSE)
    own = _mse(truth, guess)
    if not math.isfinite(own):
        raise _not_finite("In the leak screen, the squared error of a single-column tree over "
                          "the validation rows of the selection folds", own, TARGET_CAUSE)
    readers = [("tree", lambda base: guess)]
    only = {}
    if other:
        line = np.concatenate(other)
        if np.isfinite(line).all() and math.isfinite(_mse(truth, line)):
            readers.append(("line", lambda base: line))
    if level:
        by_level = np.concatenate(level)
        seen = ~np.isnan(by_level)

        def from_levels(base):
            return np.where(seen, by_level, base if P.B_UNSEEN == "base" else mean_pred)
        if np.isfinite(by_level[seen]).all() \\
                and all(math.isfinite(_mse(truth, from_levels(pred))) for _, pred in base_preds):
            readers.append(("levels", from_levels))
            if P.B_UNSEEN == "skip":
                only[from_levels] = seen
    eligible = None if marked is None else ~marked[np.concatenate(rows)]
    robust_preds = [("training_median", np.concatenate(median_guess))] if robust else []
    with np.errstate(over="ignore", invalid="ignore"):

        def skill_of(reader, unscored):
            """The lower of the skills against each baseline; None when one has no error."""
            lowest = None
            for _, base_pred in base_preds + (robust_preds if unscored else []):
                base_error = (truth - base_pred) ** 2
                own_errors = (truth - reader(base_pred)) ** 2
                may = eligible
                if reader in only:
                    if only[reader].sum() < 20:
                        return -1.0     # too few rows of a level seen: no reading by levels
                    own_errors, base_error = own_errors[only[reader]], base_error[only[reader]]
                    may = None if eligible is None else eligible[only[reader]]
                keep = _scored_rows(own_errors, base_error, may) if unscored \\
                    else np.ones(len(own_errors), dtype=bool)
                base_sum = float(base_error[keep].sum())
                if not base_sum > 0:
                    return None             # a baseline with no error cannot be beaten (D36)
                value = 1 - float(own_errors[keep].sum()) / base_sum
                lowest = value if lowest is None else min(lowest, value)
            return lowest

        each = {name: (skill_of(reader, not every_row), skill_of(reader, False))
                for name, reader in readers}
        if log is not None:
            log.update(each)
            log["seen"] = float(seen.mean()) if level else None
        if level_mode < 2:
            each.pop("levels", None)

        def best(which):
            found = [pair[which] for pair in each.values()]
            return None if any(f is None for f in found) else max(found)

        told = best(1)
        decided = told if every_row else best(0)
    return decided, told


''')

# ------------------------------------------------- C: the order reading --
sub('''def _far_outside(values):''', '''def _order_share(x, y, form, seed=0):
    """D77: the share against the order, under each form measured for the plan.

    old     rows of equal column value in the target's order (D76 as built)
    random  rows of equal column value in an order drawn from the seed
    mean    one point for each distinct value: the mean target of its rows
    median  one point for each distinct value: the median target of its rows
    pairs   for each two neighbouring distinct values, every row of the one with
            every row of the other; the shares are averaged over the neighbours
    """
    if form == "old":
        return _against_order(x, y)
    have = ~(np.isnan(x) | np.isnan(y))
    x, y = x[have], y[have]
    if not len(x):
        return None
    if form == "random":
        order = np.lexsort((np.random.default_rng([seed, 9]).random(len(x)), x))
        xs, ys = x[order], y[order]
        moved = xs[1:] > xs[:-1]
        if int(moved.sum()) < ORDER_PAIRS:
            return None
        return min(float((ys[1:][moved] > ys[:-1][moved]).mean()),
                   float((ys[1:][moved] < ys[:-1][moved]).mean()))
    levels, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(levels) - 1 < ORDER_PAIRS:
        return None
    if form in ("mean", "median"):
        if form == "mean":
            with np.errstate(over="ignore", invalid="ignore"):
                point = np.bincount(inverse, weights=y) / counts
        else:
            ys = y[np.lexsort((y, inverse))]
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            point = (ys[starts + (counts - 1) // 2] + ys[starts + counts // 2]) / 2
        return min(float((point[1:] > point[:-1]).mean()), float((point[1:] < point[:-1]).mean()))
    # pairs: every row of a value with every row of the next value
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

# -------------------------------------------------------- the screen in _run --
sub('''        screen_folds, _, every_row = _screen_folds(y, train, split["folds"], screen_prev,
                                                   by_entity, classification)
''', '''        screen_folds, _, every_row, marked = _screen_folds(y, train, split["folds"],
                                                           screen_prev, by_entity,
                                                           classification)
        if P.DEBUG:
            P.LOG["screen"], P.LOG["order"], P.LOG["levels"] = {}, {}, {}
''')
sub('''                alone, told = _screen_column(values, numeric, y, screen_folds, classification,
                                             seed, screen_prev,
                                             holdout if BROKEN == "M3" else None, by_entity,
                                             every_row)
''', '''                # D77: a text column, and a numeric column of few distinct values, is
                # also read by its levels
                few = True
                if numeric:
                    held = values[train]
                    held = held[~np.isnan(held)]
                    distinct = len(np.unique(held))
                    few = distinct > 0 and len(held) >= P.B_ROWS * distinct \\
                        and (P.B_MAX == 0 or distinct <= P.B_MAX)
                    if P.DEBUG:
                        P.LOG["levels"][name] = (int(distinct), int(len(held)))
                level_mode = 0 if P.B != "levels" else 2 if few else (1 if P.DEBUG else 0)
                seen_log = {} if P.DEBUG else None
                alone, told = _screen_column(values, numeric, y, screen_folds, classification,
                                             seed, screen_prev,
                                             holdout if BROKEN == "M3" else None, by_entity,
                                             every_row, level_mode, marked, seen_log)
                if P.DEBUG:
                    P.LOG["screen"][name] = dict(seen_log, alone=alone, told=told,
                                                 level_mode=level_mode)
''')
sub('''                share = _against_order(values[train], y[train])
                before = None if screen_prev is None else _against_order(screen_prev[train],
                                                                         y[train])
''', '''                share = _order_share(values[train], y[train], P.C, seed)
                before = None if screen_prev is None else _order_share(screen_prev[train],
                                                                       y[train], P.C, seed)
                if P.DEBUG:
                    P.LOG["order"][name] = {
                        form: (_order_share(values[train], y[train], form, seed),
                               None if screen_prev is None
                               else _order_share(screen_prev[train], y[train], form, seed))
                        for form in P.C_FORMS}
''')

# timing of the screen, for the plan
sub("""    # 8. from here on every fitted step sees training rows only: the leak screen first
    if BROKEN != "F2":
""", """    # 8. from here on every fitted step sees training rows only: the leak screen first
    screen_started = time.perf_counter()
    if BROKEN != "F2":
""")
sub("""    # D74, D76: a numeric feature, or a regression target, that holds a value far
    # outside all its other values, among the rows that are split.""",
    """    if P.DEBUG:
        P.LOG["screen_seconds"] = time.perf_counter() - screen_started
    # D74, D76: a numeric feature, or a regression target, that holds a value far
    # outside all its other values, among the rows that are split.""")

# ------------------------------------------------- E: a large holdout --
sub('''    train, holdout = split["train"], split["holdout"]
    for w in warnings:
''', '''    train, holdout = split["train"], split["holdout"]
    if time_ordered and BROKEN != "F7" and P.E > 0:
        held_share = len(holdout) / (len(train) + len(holdout) + len(split["gap"]))
        if held_share >= P.E:
            first = int((keys[holdout] == keys[holdout].min()).sum())
            _warn(warnings, "large_holdout",
                  f"The holdout holds {len(holdout)} of the "
                  f"{len(train) + len(holdout) + len(split['gap'])} rows "
                  f"({100 * held_share:.1f}%), far more than a quarter: it is made of whole "
                  f"timestamps, and its earliest timestamp alone holds {first} rows. "
                  + LARGE_HOLDOUT)
    for w in warnings:
''')
sub('''# D66: a whole number reads as a calendar date written YYYYMMDD when its year is in''',
    '''# D77: what is said of a time-ordered holdout far above a quarter. A fixed text.
LARGE_HOLDOUT = ("The models were fitted on the rows before it, and the headline is measured "
                 "on it: fewer rows were learned from than the split usually leaves.")
# D66: a whole number reads as a calendar date written YYYYMMDD when its year is in''')

# ------------------------------------------- A: what the headline rests on --
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
sub('''    # 13. assemble the result
''', '''    # D77: what the headline rests on. For each feature, the score of the chosen
    # configuration on the holdout with that feature's part of the holdout matrix moved
    # to other holdout rows, the mean over `A_K` shuffles; nothing is refitted.
    if P.A:
        started_a = time.perf_counter()
        prep, names = final_fit["prep"], final_fit["names"]
        blocks, at = {}, len(prep.numeric)
        for j, name in enumerate(prep.numeric):
            blocks[name] = [j]
        for name, levels in zip(prep.categorical, prep.levels):
            blocks[name] = list(range(at, at + len(levels)))
            at += len(levels)
        custom = bool(chosen_family.get("custom"))

        def shuffled(name):
            rng = np.random.default_rng([seed, 8, by_name[name]["position"]])
            cols, got = blocks[name], []
            for _ in range(P.A_K):
                order = rng.permutation(len(holdout))
                moved = x_holdout.copy()
                moved[:, cols] = x_holdout[order][:, cols]
                new = pd.DataFrame(moved, index=frames[1].index, columns=names) if custom \\
                    else moved
                guess = finite(_predict(chosen_model, new, classification), len(holdout))
                got.append(_auc(truth, guess) if classification else _mse(truth, guess))
            return got

        feature_names = list(prepared)
        if not custom and hasattr(chosen_model, "estimators_") and n_jobs > 1 \\
                and len(feature_names) > 1:
            with ThreadPoolExecutor(max_workers=min(n_jobs, len(feature_names))) as pool:
                every = list(pool.map(shuffled, feature_names))
        else:
            every = [shuffled(name) for name in feature_names]
        for name, got in zip(feature_names, every):
            with np.errstate(over="ignore", invalid="ignore"):
                mean_score = float(np.mean(got))
            if not math.isfinite(mean_score):
                raise _not_finite(f"The score of the chosen configuration with the holdout "
                                  f"values of {name!r} shuffled", mean_score, TARGET_CAUSE)
            rec = by_name[name]
            rec["shuffled_score"] = mean_score
            if BROKEN == "F4":
                continue
            if classification:
                without = 2 * mean_score - 1
            else:
                without = 1 - mean_score / baseline["score"] if baseline["score"] > 0 else None
            if skill is not None and without is not None:
                rec["shuffled_skill"], rec["skill_fall"] = without, skill - without
            if P.DEBUG:
                P.LOG.setdefault("rests", {})[name] = [float(g) for g in got]
        if skill is not None and skill >= P.A_WARN and BROKEN != "F4":
            order = sorted((r for r in records if r["skill_fall"] is not None),
                           key=lambda r: (-r["skill_fall"], r["position"]))
            named = [r for r in order if r["skill_fall"] >= P.A_MIN][:P.A_TOP]
            if named:
                rests = ("It rests most on " + "; ".join(
                    f"{r['name']!r} (skill {r['shuffled_skill']:.3f} with its holdout values "
                    f"shuffled, a fall of {r['skill_fall']:.3f})" for r in named)
                    + ". Check that each of these columns is known before the outcome")
            else:
                rests = (f"No column's fall reaches {P.A_MIN}: shuffling any one column's "
                         "holdout values leaves the skill almost where it is, as it does when "
                         "several columns carry the same information. Check that the columns "
                         "are known before the outcome")
            _warn(warnings, "near_perfect_headline",
                  f"The headline skill is {skill:.4f}, at or above {P.A_WARN}: the level at "
                  "which one column alone is excluded as a suspected leak. " + rests)
        if P.DEBUG:
            P.LOG["a_seconds"] = time.perf_counter() - started_a

    # 13. assemble the result
''')

(DST / "core.py").write_bytes(text.encode("utf-8"))
print("wrote", DST / "core.py", len(text))
