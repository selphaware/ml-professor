r"""Make make_proto5.py from make_proto.py: the prototype brought up to D78.

R1  `headline.rests_on` and the summary's line; the warning names the same columns
R2  exact copies are shuffled together and share one fall
R4  a level not seen in a fold's training rows is not scored; a floor of scored rows
R5  the `suspected_leak` message of a column read by its levels names `group_col`
R6  nulls with a warning when a model cannot predict a shuffled matrix
R7  the code name `holdout_far_above_quarter`
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
t = (HERE / "make_proto.py").read_text(encoding="utf-8")


def sub(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:100])
    t = t.replace(old, new)


# ------------------------------------------------------------------ R4 --
# classification: the reading by levels of a text column is the direct reading of D74,
# and a level the fold's training rows do not hold is no longer given the fold's share
sub(r'''        elif direct:
            codes_a, codes_b = xa[:, 0].astype(int), xb[:, 0].astype(int)
            share = np.bincount(codes_a, weights=y[a], minlength=len(levels)) \\
                / np.maximum(np.bincount(codes_a, minlength=len(levels)), 1)
            other.append(np.where(codes_b >= 0, share[np.maximum(codes_b, 0)], fold_mean))
        if level_mode and not (classification and not numeric):
            level.append(_level_predict(xa[:, 0], y[a].astype("float64"), xb[:, 0], numeric))''',
    r'''        elif direct and P.B_UNSEEN == "base":
            codes_a, codes_b = xa[:, 0].astype(int), xb[:, 0].astype(int)
            share = np.bincount(codes_a, weights=y[a], minlength=len(levels)) \\
                / np.maximum(np.bincount(codes_a, minlength=len(levels)), 1)
            other.append(np.where(codes_b >= 0, share[np.maximum(codes_b, 0)], fold_mean))
        elif direct:
            level.append(_level_predict(xa[:, 0], y[a].astype("float64"), xb[:, 0], False))
        if level_mode and not (classification and not numeric):
            level.append(_level_predict(xa[:, 0], y[a].astype("float64"), xb[:, 0], numeric))''')
sub(r'''        if level:
            by_level = np.concatenate(level)
            third = _auc(truth, np.where(np.isnan(by_level), np.concatenate(mean_guess), by_level))
            if third is not None:
                found["levels"] = 2 * third - 1
        if log is not None:
            log.update({name: (value, value) for name, value in found.items()})
        if level_mode < 2:
            found.pop("levels", None)''',
    r'''        if level:
            by_level = np.concatenate(level)
            seen = ~np.isnan(by_level)
            if P.B_UNSEEN == "base":
                third = _auc(truth, np.where(seen, by_level, np.concatenate(mean_guess)))
            else:
                # D78: rows whose level was not seen are not scored; with too few rows
                # scored the reading by levels has no value
                raw = _auc(truth[seen], by_level[seen]) if seen.any() else None
                rarer = int(min(truth[seen].sum(), seen.sum() - truth[seen].sum()))
                enough = int(seen.sum()) >= P.B_MIN
                for one in (0, 1):
                    scored, held = int((truth[seen] == one).sum()), int((truth == one).sum())
                    enough = enough and (scored >= P.B_MIN_CLASS or scored == held)
                third = raw if enough else None
                if log is not None:
                    log["levels_raw"] = (None if raw is None else 2 * raw - 1, int(seen.sum()),
                                         int(len(seen)), rarer)
            if third is not None:
                found["levels"] = 2 * third - 1
        if log is not None:
            log.update({name: (value, value) for name, value in found.items()})
        if level_mode < 2 and numeric:
            found.pop("levels", None)''')
# regression
sub(r'''        if np.isfinite(by_level[seen]).all() \\
                and all(math.isfinite(_mse(truth, from_levels(pred))) for _, pred in base_preds):
            readers.append(("levels", from_levels))
            if P.B_UNSEEN == "skip":
                only[from_levels] = seen''',
    r'''        if np.isfinite(by_level[seen]).all() \\
                and all(math.isfinite(_mse(truth, from_levels(pred))) for _, pred in base_preds):
            readers.append(("levels", from_levels))
            if P.B_UNSEEN == "skip":
                only[from_levels] = seen''')
sub(r'''                if reader in only:
                    if only[reader].sum() < 20:
                        return -1.0     # too few rows of a level seen: no reading by levels
                    own_errors, base_error = own_errors[only[reader]], base_error[only[reader]]''',
    r'''                if reader in only:
                    if not only[reader].any():
                        return None
                    own_errors, base_error = own_errors[only[reader]], base_error[only[reader]]''')
sub(r'''        if log is not None:
            log.update(each)
            log["seen"] = float(seen.mean()) if level else None
        if level_mode < 2:
            each.pop("levels", None)''',
    r'''        if log is not None:
            log.update(each)
            log["seen"] = float(seen.mean()) if level else None
            if level and "levels" in each:
                log["levels_raw"] = (each["levels"][1], int(seen.sum()), int(len(seen)),
                                     each["levels"][0])
        if level_mode < 2:
            each.pop("levels", None)
        if "levels" in each and only and (int(seen.sum()) < P.B_MIN or None in each["levels"]):
            # D78: too few rows scored (or no error to beat among them): no reading by levels
            each.pop("levels")
            if log is not None:
                log.pop("levels", None)''')

# ------------------------------------------------------------------ R5 --
sub(r'''                if P.DEBUG:
                    P.LOG["screen"][name] = dict(seen_log, alone=alone, told=told,
                                                 level_mode=level_mode)
''', r'''                if P.DEBUG:
                    P.LOG["screen"][name] = dict(seen_log, alone=alone, told=told,
                                                 level_mode=level_mode)
                by_its_levels[name] = level_mode == 2 or (classification and not numeric
                                                          and BROKEN != "M47")
''')
sub(r'''        if P.DEBUG:
            P.LOG["screen"], P.LOG["order"], P.LOG["levels"] = {}, {}, {}
''', r'''        if P.DEBUG:
            P.LOG["screen"], P.LOG["order"], P.LOG["levels"] = {}, {}, {}
        by_its_levels = {}
''')
t = t.replace(r'''# timing of the screen, for the plan''', r'''# R5: the message of a column read by its levels
sub("""                _warn(warnings, "suspected_leak",
                      f"column {name!r} is excluded as a suspected leak: {why}", name)""",
    """                _warn(warnings, "suspected_leak",
                      f"column {name!r} is excluded as a suspected leak: {why}"
                      + (". If it names the entity each row belongs to, declare it as "
                         "group_col (--group-col)"
                         if rule == "predicts_target" and by_its_levels.get(name) else ""),
                      name)""")

# timing of the screen, for the plan''')

# ------------------------------------------------------------------ R7 --
sub(r'''            _warn(warnings, "large_holdout",''', r'''            _warn(warnings, "holdout_far_above_quarter",''')

# ------------------------------------------------------- R1, R2, R6: A --
start = t.index("sub('''    # 13. assemble the result\n''', '''    # D77: what the headline rests on.")
end = t.index("(DST / \"core.py\").write_bytes")
t = t[:start] + r"""sub('''    # 13. assemble the result
''', '''    # D77, D78: what the headline rests on. For each feature, the score of the chosen
    # configuration on the holdout with that feature's part of the holdout matrix moved
    # to other holdout rows, the mean over `A_K` shuffles; nothing is refitted. Columns
    # that are exact copies of one another are moved together and share one fall.
    rests_on, floor = [], None
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
        # D78 (3): two features are copies when the parts of the final fit's matrix they
        # give are the same in every training row and every holdout row: a numeric
        # feature's column agreeing to within COPY_TOLERANCE, a categorical feature's
        # block of level columns holding the same columns in any order
        if P.A_COPIES == "values":
            reach = rows[np.concatenate([train, holdout])]
            copy_sets = {}
            for name, values in prepared.items():
                held = values[reach]
                if kinds[name] == "numeric":
                    held = np.where(np.isnan(held) | (np.abs(held) > TOO_LARGE), np.nan,
                                    held) + 0.0
                    key = ("numeric", held.tobytes())
                else:
                    key = ("categorical", "\\x00".join(held.astype(str)))
                copy_sets.setdefault(key, []).append(name)
            copy_sets = list(copy_sets.values())
        else:
            whole = np.vstack([x_fit, x_holdout])
            head = min(16, len(whole))

            def block(name):
                part = whole[:, blocks[name]]
                if kinds[name] == "numeric":
                    return part
                return part[:, np.lexsort(part[::-1])] if part.shape[1] else part

            seen_blocks = {name: block(name) for name in prepared}

            def alike(a, b):
                x, z = seen_blocks[a], seen_blocks[b]
                if kinds[a] != kinds[b] or x.shape != z.shape:
                    return False
                if kinds[a] == "numeric":
                    for sign in ((1.0, -1.0) if P.COPY_SIGN else (1.0,)):
                        if np.all(np.abs(x[:head] - sign * z[:head]) <= P.COPY_TOLERANCE) \
                                and np.all(np.abs(x - sign * z) <= P.COPY_TOLERANCE):
                            return True
                    return False
                return bool(np.array_equal(x[:head], z[:head]) and np.array_equal(x, z))

            copy_sets, placed = [], set()
            for name in prepared:
                if name in placed:
                    continue
                members = [name] + [other for other in prepared
                                    if other != name and other not in placed
                                    and P.A_COPIES != "none" and alike(name, other)]
                placed.update(members)
                copy_sets.append(members)

        def shuffled(members):
            rng = np.random.default_rng([seed, 8, by_name[members[0]]["position"]])
            cols = [c for name in members for c in blocks[name]]
            got = []
            for _ in range(P.A_K):
                order = rng.permutation(len(holdout))
                moved = x_holdout.copy()
                moved[:, cols] = x_holdout[order][:, cols]
                new = pd.DataFrame(moved, index=frames[1].index, columns=names) if custom \\
                    else moved
                try:
                    guess = finite(_predict(chosen_model, new, classification), len(holdout))
                except Exception as exc:            # the model cannot predict this matrix
                    return f"{type(exc).__name__}: {exc}"
                got.append(_auc(truth, guess) if classification else _mse(truth, guess))
            return got

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
                if P.DEBUG:
                    P.LOG.setdefault("rests", {})[name] = [float(g) for g in got]
        if P.DEBUG:
            P.LOG["copies"] = [members for members in copy_sets if len(members) > 1]
        # D80 (1): the floor of the list. The half-width of the headline skill's
        # interval, or a stand-in where there is none; never below A_MIN.
        half = None
        if uncertainty["skill_low"] is not None and uncertainty["skill_high"] is not None:
            half = (uncertainty["skill_high"] - uncertainty["skill_low"]) / 2
        rows_half = None
        if skill is not None and (half is None or P.DEBUG):
            # the half-width of a bootstrap of the holdout rows, as if they were independent
            rng_rows = np.random.default_rng([seed, 10])
            found_rows = []
            for _ in range(RESAMPLES[profile]):
                pick = rng_rows.integers(0, len(holdout), len(holdout))
                if classification:
                    a_rows = _auc(truth[pick], pred[pick])
                    if a_rows is not None:
                        found_rows.append(2 * a_rows - 1)
                else:
                    b_rows = _mse(truth[pick], base_pred[pick])
                    if b_rows > 0:
                        found_rows.append(1 - _mse(truth[pick], pred[pick]) / b_rows)
            if len(found_rows) * 2 >= RESAMPLES[profile]:
                lo_rows, hi_rows = np.percentile(found_rows, [2.5, 97.5])
                rows_half = float(hi_rows - lo_rows) / 2
        if half is not None and P.A_HALF:
            floor = max(P.A_MIN, half)
        elif P.A_STANDIN == "rows" and rows_half is not None and P.A_HALF:
            floor = max(P.A_MIN, rows_half)
        elif P.A_STANDIN not in ("rows", "none") and P.A_HALF:
            floor = max(P.A_MIN, float(P.A_STANDIN))
        else:
            floor = P.A_MIN
        if P.DEBUG:
            P.LOG["floor"] = {"half": half, "rows_half": rows_half, "floor": floor}
        order = sorted((r for r in records if r["skill_fall"] is not None),
                       key=lambda r: (-r["skill_fall"], r["position"]))
        named = [r for r in order if r["skill_fall"] >= floor][:P.A_TOP]
        rests_on = [{"column": r["name"], "skill_fall": r["skill_fall"]} for r in named]
        if skill is not None and skill >= P.A_WARN and BROKEN != "F4":
            if named:
                rests = ("It rests most on " + "; ".join(
                    f"{r['name']!r} (skill {r['shuffled_skill']:.3f} with its holdout values "
                    f"shuffled, a fall of {r['skill_fall']:.3f})" for r in named)
                    + ". Check that each of these columns is known before the outcome")
            else:
                rests = (f"No column's fall reaches {P.A_MIN}: shuffling any one column's "
                         "holdout values leaves the skill almost where it is. Check that the "
                         "columns are known before the outcome")
            _warn(warnings, "near_perfect_headline",
                  f"The headline skill is {skill:.4f}, at or above {P.A_WARN}: the level at "
                  "which one column alone is excluded as a suspected leak. " + rests)
        if P.DEBUG:
            P.LOG["a_seconds"] = time.perf_counter() - started_a

    # 13. assemble the result
''')
sub('''    headline.update(largest)
''', '''    headline.update(largest)
    headline["rests_on"] = rests_on
    headline["rests_on_floor"] = floor if P.A else None
''')
sub('''    for w in d["warnings"]:                 # one line each: the code, then the column''',
    '''    rests = h.get("rests_on") or []
    floor = h.get("rests_on_floor")
    lines.append("rests on" + ("" if floor is None else f" (columns whose fall is at least "
                                                         f"{floor:.4g})") + ": "
                 + (", ".join(f"{e['column']!r} (fall {e['skill_fall']:.6g})" for e in rests)
                    if rests else "none"))
    for w in d["warnings"]:                 # one line each: the code, then the column''')

""" + t[end:]

(HERE / "make_proto5.py").write_text(t, encoding="utf-8", newline="\n")
p = HERE / "_proto.py"
s = p.read_text(encoding="utf-8")
if "B_MIN" not in s:
    s = s.replace('B_UNSEEN = _get("MLP_B_UNSEEN", "base")',
                  'B_UNSEEN = _get("MLP_B_UNSEEN", "skip")        # D78: not scored\n'
                  '# D78: the fewest scored validation rows for a reading by levels to have a value\n'
                  'B_MIN = int(_get("MLP_B_MIN", "20"))\n'
                  '# D78: exact copies are shuffled together and share one fall\n'
                  'A_COPIES = _get("MLP_A_COPIES", "1") == "1"')
    assert "B_MIN" in s
    p.write_text(s, encoding="utf-8", newline="\n")
print("wrote", HERE / "make_proto5.py")
