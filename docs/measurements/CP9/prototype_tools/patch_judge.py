"""Apply the proposed changes to a COPY of verify/generated_tables.py (never the repo's).

    python -B patch_judge.py <path to the copy of generated_tables.py>

Each change is one exact replacement, asserted to match once. The list at the end of
REPORT.md names them C1 to C8 in the same order.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
assert not str(path.resolve()).lower().replace("/", chr(92)).startswith("e:" + chr(92) + "code"), "refusing to patch a file inside the repo"
s = path.read_text(encoding="utf-8")
CHANGES = []


def change(label, old, new):
    global s
    assert s.count(old) == 1, f"{label}: the text to replace occurs {s.count(old)} times"
    s = s.replace(old, new)
    CHANGES.append(label)


# C1. The open list is empty.
change("C1 OPEN is empty", '''OPEN = {
    "forests_under_time_order": (
        "D73 point 4", "pass 8",
        "under time order the two sides hand a fit its rows in different orders, so a "
        "forest's fold scores differ, and with them whatever follows when a forest is "
        "chosen on either side: the chosen configuration, the score, the predictions"),
    "target_copy_kept": (
        "D73 point 2", "pass 8",
        "an exact copy of the target that the leak screen leaves a feature, on either side"),
}
''', '''# Empty since pass 8 (D73): the order of the rows of a fit under time order is pinned,
# so a forest is compared like every other family; and a column that reproduces the
# target is always excluded, so one that stays a feature fails the tool.
OPEN = {}
# The id-like rule for whole numbers as pass 8 repairs it (step 5, rule 8b; D73): more
# than this share of the values distinct, and the distinct values filling at least
# that share of their range. PROPOSED VALUES: they follow the rule that is approved.
ID_DISTINCT_SHARE, ID_RANGE_SHARE = 0.95, 0.5
# The cut-off of the `largest_row_dominates` warning (step 11; D73 point 3). PROPOSED.
LARGEST_ROW_SHARE = 0.5
# Under the grouped scheme, a fold that trains on fewer rows than this: no choice between
# configurations is made (step 9; D73). PROPOSED.
SELECTION_FLOOR = 50
''')

# C2. No family is set apart for the order of its rows.
change("C2 no family set apart for row order", '''ROW_ORDER_FAMILIES = {"forest"}
''', '''ROW_ORDER_FAMILIES = set()     # none since pass 8: the order is pinned (D73 point 4)
''')

# C3. An exact copy of the target that stays a feature fails the tool.
change("C3 a kept copy of the target is a failing property", '''                add("open", f"{side}: an exact copy of the target is excluded by the leak "
                            "screen (step 8)", text if side == "application" else "",
                    text if side == "reference" else "", name="target_copy_kept")
''', '''                add("property", f"{side}: an exact copy of the target is excluded by the leak "
                                f"screen (step 8; D73 point 2)",
                    text if side == "application" else "", text if side == "reference" else "")
''')

# C4. Under time order the largest row's share is compared like the score.
change("C4 the four headline keys compared under time order", '''    if timed:
        exact("headline.baseline_score")
        exact("baselines")
        exact("cv.n_scored_folds")
''', '''    if timed:
        exact("headline.baseline_score")
        exact("baselines")
        exact("cv.n_scored_folds")
        for path in ("headline.baseline_largest_row_share", "headline.baseline_largest_row"):
            exact(path)
        for path in ("headline.largest_row_share", "headline.largest_row"):
            exact(path, "fit", "the same rows on both sides: only the fits can differ")
''')

# C5. One side alone: the largest row's share, recomputed from predictions.csv.
change("C5 the largest row's share as a property of one side", '''    # null patterns of 5.2
    codes = {code for code, _ in warning_counts(doc)}
''', '''    # the largest row's share of the squared error on the holdout (step 11; D73 point 3)
    share_keys = ("largest_row_share", "largest_row", "baseline_largest_row_share",
                  "baseline_largest_row")
    got = {k: at(doc, f"headline.{k}") for k in share_keys}
    absent = [k for k, v in got.items() if v == ("<absent>",)]
    if absent:
        bad("headline holds the largest row's share, for the model and for the baseline "
            "(5.2; D73 point 3)", f"absent: {absent}")
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
                        "recomputed from predictions.csv (step 11; D73 point 3)",
                        f"{got['largest_row_share']!r}; recomputed {top / total!r}")
                at_top = {p for p, e in zip(held, errors) if close(e, top, 1e-9)}
                if got["largest_row"] not in at_top:
                    bad("headline.largest_row is the holdout row with the largest squared error "
                        "(step 11; D73 point 3)",
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
        # D74 (3): the same statement over the selection folds: one record when a
        # configuration's share is above one half, one when the baseline's is
        cv = doc.get("cv", {})
        for key in ("largest_row_share", "largest_row", "baseline_largest_row_share",
                    "baseline_largest_row"):
            if key not in cv:
                bad("`cv` holds the largest row's share over the selection folds (5.2; D74)",
                    f"no `cv.{key}`")
        shares = [m.get("largest_row_share") for m in doc.get("models", [])
                  if isinstance(m, dict)]
        above += any(is_number(v) and v > LARGEST_ROW_SHARE for v in shares)
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
''')

# C6. One side alone: no fold is left unscored for its size, and under the grouped
# scheme no choice between configurations is made when a fold trains on too few rows.
change("C6 selection_not_made under the grouped scheme", '''        if records != unscored:
            bad("one fold_not_scored record for each fold not scored (2.5; D72)",
                f"{records} record(s) for {unscored} fold(s) not scored")
''', '''        if records != unscored:
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
        few = scheme == "grouped" and len(sized) == len(fold_list) and bool(sized) \\
            and any(f["training_rows"] < SELECTION_FLOOR
                    and f["training_rows"] < f["validation_rows"] for f in sized)
        if told != int(few):
            bad("one selection_not_made record exactly when, under the grouped scheme, a fold "
                "trains on fewer than 50 rows and on fewer than it validates on (step 9; D73)",
                f"{told} record(s); the folds train on "
                f"{[f['training_rows'] for f in sized]} rows under the {scheme} scheme")
        elif few:
            fitted = [m.get("config") for m in doc.get("models", [])
                      if isinstance(m, dict) and m.get("status") == "fitted"]
            if fitted and at(doc, "cv.chosen") != fitted[0]:
                bad("with selection_not_made the first configuration in configuration order "
                    "is used (step 9; D73)",
                    f"cv.chosen is {at(doc, 'cv.chosen')!r}; the first fitted configuration is "
                    f"{fitted[0]!r}")
''')

# C7. One side alone: both baselines under time order, recomputed from the source
# where no entity column is declared (the rule D73 point 1 extends).
change("C7 both baselines under time order, recomputed with no entities", '''                if at(doc, "validation.n_holdout_timestamps") != split["holdout_stamps"]:
                    bad("validation.n_holdout_timestamps (5.2)",
                        f"{at(doc, 'validation.n_holdout_timestamps')!r}; counted "
                        f"{split['holdout_stamps']}")
''', '''                if at(doc, "validation.n_holdout_timestamps") != split["holdout_stamps"]:
                    bad("validation.n_holdout_timestamps (5.2)",
                        f"{at(doc, 'validation.n_holdout_timestamps')!r}; counted "
                        f"{split['holdout_stamps']}")
                if at(doc, "task.type") == "regression":
                    listed_b = [x.get("name") for x in doc.get("baselines", [])
                                if isinstance(x, dict)]
                    if listed_b != ["training_mean", "previous_value"]:
                        bad("under time order a regression scores both baselines, whether the "
                            "timestamps repeat or not (step 11; D73 point 1)", str(listed_b))
                    elif not o["declares_entity"]:
                        # the mean target of the timestamp `gap + 1` distinct timestamps before
                        try:
                            value = {i: float(cells[i]) for i in keys}
                            place = {s: k for k, s in enumerate(split["order"])}
                            means = [math.fsum(value[i] for i in split["by_stamp"][s])
                                     / len(split["by_stamp"][s]) for s in split["order"]]
                            level = math.fsum(value[i] for i in split["training"]) \\
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
                                    "point 1)", f"{said_b[b_name]!r}; recomputed {b_mine!r}")
''')

# C8. One side alone: the generator's identifier of whole numbers, by the repaired rule.
change("C8 the identifier row_id by the repaired rule", '''    want = {"loaded": n, "dropped_missing_target": int(o["y_missing"].sum()),
''', '''    # The generator's identifier of whole numbers (step 5, rule 8b as D73 repairs it):
    # judged by its distinct values among the usable rows, whatever their order.
    for rec in doc["columns"]:
        if rec.get("name") != "row_id" or rec.get("role") in ("time", "group", "target"):
            continue
        present = [v for v, ok in zip(frame.iloc[:, rec["position"]], usable)
                   if ok and not _empty(v)]
        if not present or any(isinstance(v, str) for v in present):
            continue
        whole = [int(v) for v in present]
        distinct = set(whole)
        dense = len(distinct) > 1 and len(distinct) > ID_DISTINCT_SHARE * len(whole) \\
            and len(distinct) >= ID_RANGE_SHARE * (max(distinct) - min(distinct) + 1)
        rising = all(b > a for a, b in zip(whole, whole[1:]))
        falling = all(b < a for a, b in zip(whole, whole[1:]))
        d19 = len(whole) > 1 and len(distinct) == len(whole) and (
            rising or falling or max(distinct) - min(distinct) + 1 == len(whole))
        dense = dense or d19
        is_id = (rec.get("role"), rec.get("reason")) == ("excluded", "id_like")
        if rec.get("reason") in ("constant", "all_missing", "datetime"):
            continue
        if dense != is_id:
            bad("a whole-number column is id-like exactly when every value is distinct and the "
                "values are consecutive or strictly ordered in file order, or more than 95% of "
                "its values are distinct and they fill at least half of their range (step 5, "
                "rule 8b; D19, D74)",
                f"`row_id` has role {rec.get('role')!r}, reason {rec.get('reason')!r}; "
                f"{len(distinct)} distinct of {len(whole)} values over a range of "
                f"{max(distinct) - min(distinct) + 1}")
    want = {"loaded": n, "dropped_missing_target": int(o["y_missing"].sum()),
''')

path.write_bytes(s.encode("utf-8"))
print("applied:", *CHANGES, sep="\n  ")
