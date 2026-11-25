"""Bring patch_judge.py (the changes the generated-table judge needs) up to D74:
the largest row's share over the selection folds, and the id-like rule as the
union of D19's clause and the new one."""
from pathlib import Path

p = Path(__file__).resolve().parent / "patch_judge.py"
s = p.read_text(encoding="utf-8")


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


rep('''        above = sum(1 for k in ("largest_row_share", "baseline_largest_row_share")
                    if is_number(got[k]) and got[k] > LARGEST_ROW_SHARE)
''', '''        above = sum(1 for k in ("largest_row_share", "baseline_largest_row_share")
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
''')
rep('''            bad("one largest_row_dominates record for each of the two shares above one half "
                "(2.5; D73 point 3)", f"{said} record(s); {above} share(s) above "
                                      f"{LARGEST_ROW_SHARE}: {got}")''',
    '''            bad("one largest_row_dominates record for each statement above one half: the "
                "model and the baseline on the holdout, any configuration and the baseline "
                "over the selection folds (2.5; D74 point 3)",
                f"{said} record(s); {above} statement(s) above {LARGEST_ROW_SHARE}: {got}")''')
rep('''        is_id = (rec.get("role"), rec.get("reason")) == ("excluded", "id_like")
        if dense != is_id:
            bad("a whole-number column is id-like exactly when more than 95% of its values are "
                "distinct and they fill at least half of their range (step 5, rule 8b; D73)",''',
    '''        rising = all(b > a for a, b in zip(whole, whole[1:]))
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
                "rule 8b; D19, D74)",''')
p.write_bytes(s.encode("utf-8"))
print("patch_judge.py brought up to D74")
