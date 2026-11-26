r"""edit_make_proto5.py brought up to D80:

  (1) the rests-on floor: a column is named when its fall is at least 0.05 and at least
      the half-width of the headline skill's interval; where there is no such interval,
      at least a stand-in, `MLP_A_STANDIN`: a number, or `rows` for the half-width of a
      bootstrap of the holdout rows taken as if they were independent. The floor is
      written as `headline.rests_on_floor`, and the summary line says it.
  (4) copies up to sign: a numeric column is a copy of another when it agrees with it,
      or with its negation, to within the tolerance.
For the measurement, the log keeps the half-width, the bootstrap of rows and the falls.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = HERE / "edit_make_proto5.py"
t = p.read_text(encoding="utf-8")


def sub(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:90])
    t = t.replace(old, new)


# (4) up to sign
sub('''                if kinds[a] == "numeric":
                    return bool(np.all(np.abs(x[:head] - z[:head]) <= P.COPY_TOLERANCE)
                                and np.all(np.abs(x - z) <= P.COPY_TOLERANCE))''',
    '''                if kinds[a] == "numeric":
                    for sign in ((1.0, -1.0) if P.COPY_SIGN else (1.0,)):
                        if np.all(np.abs(x[:head] - sign * z[:head]) <= P.COPY_TOLERANCE) \\
                                and np.all(np.abs(x - sign * z) <= P.COPY_TOLERANCE):
                            return True
                    return False''')

# (1) the floor
sub('''        order = sorted((r for r in records if r["skill_fall"] is not None),''',
    '''        # D80 (1): the floor of the list. The half-width of the headline skill's
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
        order = sorted((r for r in records if r["skill_fall"] is not None),''')
sub('''        named = [r for r in order if r["skill_fall"] >= P.A_MIN][:P.A_TOP]''',
    '''        named = [r for r in order if r["skill_fall"] >= floor][:P.A_TOP]''')
sub('''    headline["rests_on"] = rests_on
''', '''    headline["rests_on"] = rests_on
    headline["rests_on_floor"] = floor if P.A else None
''')
sub('''    rests_on = []
    if P.A:''', '''    rests_on, floor = [], None
    if P.A:''')
sub('''    lines.append("rests on: " + (", ".join(f"{e['column']!r} (fall {e['skill_fall']:.6g})"
                                           for e in rests)
                                 if rests else "none (no column's fall is 0.05 or more)"))''',
    '''    floor = h.get("rests_on_floor")
    lines.append("rests on" + ("" if floor is None else f" (columns whose fall is at least "
                                                         f"{floor:.4g})") + ": "
                 + (", ".join(f"{e['column']!r} (fall {e['skill_fall']:.6g})" for e in rests)
                    if rests else "none"))''')
p.write_text(t, encoding="utf-8", newline="\n")

q = HERE / "_proto.py"
s = q.read_text(encoding="utf-8")
if "A_STANDIN" not in s:
    s += '''
# D80 (1): the floor of the list: the half-width of the headline skill's interval, and
# where there is none a stand-in (a number, `rows`, or `none` for A_MIN alone)
A_HALF = _get("MLP_A_HALF", "1") == "1"
A_STANDIN = _get("MLP_A_STANDIN", "rows")
# D80 (4): copies up to sign
COPY_SIGN = _get("MLP_COPY_SIGN", "1") == "1"
'''
    q.write_text(s, encoding="utf-8", newline="\n")
print("edited")
