"""Make the CP10 prototype of the generated-table test from verify/generated_tables.py:

  - a ninth-pass series of tables, numbered from 1,000,000 (tables 0 to 999,999 are as
    they were), each the same recipe plus columns of the kinds the pass is about: noise
    with repeated values, a coded copy of the target, two columns that give the target
    together, an exact copy of a feature, a feature rounded to few values. The default
    run is tables 0 to 999 and the first 200 of the series;
  - properties of the new output, held on each side alone: the three keys of every
    column record, their arithmetic and null pattern, `headline.rests_on`, one
    `near_perfect_headline` record exactly at a skill of 0.98 or more, an exact copy
    sharing its column's fall, and one `holdout_far_above_quarter` record exactly when a
    time-ordered holdout holds half of the rows or more.

    python -B patch_generated10.py <verify/generated_tables.py> <out/generated_tables.py>
"""
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
text = SRC.read_bytes().decode("utf-8")
assert "\r" not in text


def sub(old, new, count=1):
    global text
    found = text.count(old)
    assert found == count, f"{found} matches, expected {count}, for:\n{old[:200]}"
    text = text.replace(old, new)


sub('''DEFAULT_TABLES, DEFAULT_SEED = 1000, 20261122
''', '''DEFAULT_TABLES, DEFAULT_SEED = 1000, 20261122
# The ninth pass's series (D77, D78): tables numbered from here on carry columns of
# its kinds. Every table below is as it was.
NINTH_FROM, DEFAULT_NINTH = 1_000_000, 200
REST_KEYS = ("shuffled_score", "shuffled_skill", "skill_fall")
REST_FALL, REST_MOST, NEAR_PERFECT, FAR_ABOVE = 0.05, 5, 0.98, 0.5
''')

sub('''    if "seg" in cols and maybe(0.03):
        cols["seg=A"] = pd.Series(np.round(g.normal(size=n), 4))
        notes.append("a numeric column named seg=A beside seg")
''', '''    if "seg" in cols and maybe(0.03):
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
''')

# ---- properties, on every finished run that lists its columns
sub('''    scheme = at(doc, "validation.scheme")
    timed = scheme == "time_ordered"
    usable = ~o["y_missing"] & (~o["t_missing"] if o["declares_time"] else True)
''', '''    scheme = at(doc, "validation.scheme")
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
        if "f1_again" in by9 and "f1" in by9 and by9["f1"].get("role") == "feature" \\
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
''')

# ---- the default run
sub('''        numbers = [int(x) for x in args.only.split(",") if x.strip()] if args.only \\
            else list(range(args.tables))''',
    '''        numbers = [int(x) for x in args.only.split(",") if x.strip()] if args.only \\
            else list(range(args.tables)) + [NINTH_FROM + i for i in range(args.ninth)]''')
sub('''    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,''',
    '''    parser.add_argument("--ninth", type=int, default=DEFAULT_NINTH,
                        help="tables of the ninth pass's series, from 1,000,000 (default 200)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,''')

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_bytes(text.encode("utf-8"))
print("wrote", DST, len(text))
