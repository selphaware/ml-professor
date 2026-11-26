r"""Make the ninth pass's gate (D77, D78, D80, D81) from verify/smoke_test.py as it stood:
six datasets and six runs added last, so no seed moves; seven checks; seven broken
copies, F29 to F35.

    python -B make_gate10.py <smoke_test.py as it stood> <out smoke_test.py>
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


# ---------------------------------------------------------------- docstring --
sub('''  F28 under the grouped scheme the choice between configurations is made however few
      rows the folds beside one large entity train on
''', '''  F28 under the grouped scheme the choice between configurations is made however few
      rows the folds beside one large entity train on

The ninth pass added seven (D77, D78, D80, D81): the result now says what its headline
rests on, and the leak screen reads a column by its levels. Each has a run of its own
but the first, a check that names it and a broken copy:
  F29 the statement of what the headline rests on is not of a shuffle of the holdout:
      every column's holdout values are left where they are
  F30 a headline skill of 0.98 or more carries no `near_perfect_headline` warning
  F31 columns that are copies of one another are shuffled one at a time
  F32 the leak screen does not read a column by its levels: a coded copy of the
      target, its labels in random order, is a feature
  F33 the order reading puts rows of equal column value in the target's order: a
      column of noise with repeated values is warned about
  F34 where more training rows are marked than may be set aside, rule 2 scores every
      row: a copy of a long-tailed target with five cells moved is kept
  F35 under time order a holdout of half of the rows or more is not named
''')

# ------------------------------------------------------------ the makers --
sub('''# name -> (builder, target column). Seeds are SEED + position.
DATASETS = {''', '''# ------------------------------------------- the ninth pass (D77, D78, D80, D81) --

def make_pair_together(seed):
    """`signal_reg`, with two columns that give the target only together (D77): `pair_a`
    is the target plus a noise of the target's own spread, `pair_b` is that noise, each
    written to six places, from a stream of their own.

    Neither column alone predicts the target well enough for the leak screen (`pair_a`
    explains about half of it, `pair_b` none), and no reading of single columns can see
    the two together. A line through both recovers the target, and the headline skill is
    1.000. Measured on the reference as it was (CP10, 30 data seeds): skill 1.000 on
    every seed and nothing said about either column. The headline now carries a
    `near_perfect_headline` warning that names both.
    """
    header, rows, blank_after = make_signal_reg(seed)
    rng = random.Random(f"pair_together.noise.{seed}")
    values = [float(row[-1]) for row in rows]
    centre = mean(values)
    spread = math.sqrt(mean([(v - centre) ** 2 for v in values]))
    made = []
    for row, value in zip(rows, values):
        noise = rng.gauss(0, spread)
        made.append(row[:-1] + [_f(value + noise, 6), _f(noise, 6), row[-1]])
    return header[:-1] + ["pair_a", "pair_b", header[-1]], made, blank_after


def make_signal_reg_copied(seed):
    """`signal_reg`, with `f1_scaled`: the feature `f1` on another scale, 2 x f1 + 3,
    exactly (D78 point 3, D80 point 4).

    The models receive the two as one column: imputed and scaled, they are the same
    numbers. Shuffled one at a time, each shows a part of what they carry together (the
    other is still there): on the reference's own prototype each showed about a quarter
    of the fall of `f1` alone. Shuffled together, as copies, they share one fall.
    """
    header, rows, blank_after = make_signal_reg(seed)
    made = [row[:-1] + [_f(2 * float(row[0]) + 3), row[-1]] for row in rows]
    return header[:-1] + ["f1_scaled", header[-1]], made, blank_after


CODED_BANDS = 30


def make_leak_reg_coded(seed):
    """`signal_reg`, with `amount_band`: the target cut into 30 bands of equal count,
    each named by a label drawn at random from a stream of its own (D77 point 3).

    Read by a tree on codes of its sorted labels, as the screen read a categorical
    column of a regression, the column is noise: measured on the reference as it was
    (CP10, 30 data seeds), kept on 29 of 30, in silence on 23, with headline skill up to
    0.99. Read by its levels, the mean target of each level fitted on a fold's training
    rows, it predicts the target at about 0.99 and is excluded as a suspected leak.
    """
    header, rows, blank_after = make_signal_reg(seed)
    labels = [f"band_{k:02d}" for k in range(CODED_BANDS)]
    random.Random(f"leak_reg_coded.labels.{seed}").shuffle(labels)
    order = sorted(range(len(rows)), key=lambda i: (float(rows[i][-1]), i))
    band_of = {i: labels[rank * CODED_BANDS // len(rows)] for rank, i in enumerate(order)}
    made = [row[:-1] + [band_of[i], row[-1]] for i, row in enumerate(rows)]
    return header[:-1] + ["amount_band", header[-1]], made, blank_after


def make_noise_tied(seed):
    """`signal_reg`, with two columns of noise whose values repeat (D77 point 4): `score`, a
    whole number from 1 to 40, and `noise_rounded`, noise rounded to one decimal place,
    from a stream of their own.

    The order reading of D76 sorted rows of equal column value by the target, so across
    two neighbouring values the target nearly always fell once and rose everywhere
    else: such a column read as one the target follows, and was warned about. Rows of
    equal value are now never put in the target's order, and neither column is named.
    """
    header, rows, blank_after = make_signal_reg(seed)
    rng = random.Random(f"noise_tied.{seed}")
    made = [row[:-1] + [str(rng.randint(1, 40)), f"{rng.gauss(0, 1):.1f}", row[-1]]
            for row in rows]
    return header[:-1] + ["score", "noise_rounded", header[-1]], made, blank_after


SKEW_CELLS, SKEW_SIZE = 5, 20


def make_skewed_copy(seed, n=1500):
    """A regression whose target has a long right tail (log-normal, written to two
    places), two honest features, one of noise, and `amount_copy`: 1.07 x amount + 3,
    with five cells, drawn from a stream of their own, moved up by 20 times the target's
    spread (D77 point 5).

    The tail marks more than 1% of the training rows as far from the median, so none is
    set aside, and as D74 had it every row was then scored: the five moved cells were
    nearly all of the column's error, and the copy was warned about and kept (measured
    on the reference as it was, CP10, 30 data seeds: kept on 30 of 30). Rule 2 now
    leaves its rows unscored all the same and reads against the fold's training median
    too, and the copy is excluded.
    """
    rng = random.Random(seed)
    made = []
    for _ in range(n):
        f1, f2 = rng.gauss(0, 1), rng.gauss(0, 1)
        value = round(math.exp(0.6 * f1 - 0.4 * f2 + 0.7 * rng.gauss(0, 1)) * 100, 2)
        made.append((f1, f2, rng.gauss(0, 1), value))
    values = [v for *_, v in made]
    centre = mean(values)
    spread = math.sqrt(mean([(v - centre) ** 2 for v in values]))
    moved = set(random.Random(f"skewed_copy.cells.{seed}").sample(range(n), SKEW_CELLS))
    rows = []
    for i, (f1, f2, n1, value) in enumerate(made):
        copy = COPY_SCALE * value + COPY_SHIFT + (COPY_SCALE * SKEW_SIZE * spread
                                                   if i in moved else 0.0)
        rows.append([_f(f1), _f(f2), _f(n1), _f(copy, 6), f"{value:.2f}"])
    return ["f1", "f2", "n1", "amount_copy", "amount"], rows, ()


LAST_STAMP_SHARE = 0.6


def make_last_stamp(seed, n=1000, stamps=200):
    """Time order, with 60% of the rows on the last date (D77 point 6): 400 rows over 200
    earlier dates, each date held at least once, and 600 on the date after them. Noise.

    The holdout is the latest whole timestamps that make a quarter, and here the last
    date alone is three fifths of the rows: the models learn from 400 rows and are
    measured on 600. A `holdout_far_above_quarter` warning says so.
    """
    rng = random.Random(seed)
    late = round(LAST_STAMP_SHARE * n)
    days = list(range(stamps)) + [rng.randrange(stamps) for _ in range(n - late - stamps)]
    days += [stamps] * late
    start = datetime.date(2019, 1, 1)
    rows = [[(start + datetime.timedelta(days=d)).isoformat()]
            + [_f(rng.gauss(0, 1)) for _ in range(3)] for d in days]
    rng.shuffle(rows)
    return ["date", "n1", "n2", "amount"], rows, ()


# name -> (builder, target column). Seeds are SEED + position.
DATASETS = {''')
sub('''    "messy_shuffled": (lambda s: make_messy_shuffled(s), "outcome"),
}
''', '''    "messy_shuffled": (lambda s: make_messy_shuffled(s), "outcome"),
    # Added with the ninth pass (D77, D78, D80, D81), last, so no seed above moves.
    "pair_together": (lambda s: make_pair_together(s), "amount"),
    "signal_reg_copied": (lambda s: make_signal_reg_copied(s), "amount"),
    "leak_reg_coded": (lambda s: make_leak_reg_coded(s), "amount"),
    "noise_tied": (lambda s: make_noise_tied(s), "amount"),
    "skewed_copy": (lambda s: make_skewed_copy(s), "amount"),
    "last_stamp": (lambda s: make_last_stamp(s), "amount"),
}
''')

# ------------------------------------------------------------------ runs --
sub('''    "messy_shuffled": ("messy_shuffled", NEITHER, "fast", 1),
}
''', '''    "messy_shuffled": ("messy_shuffled", NEITHER, "fast", 1),
    # Added with the ninth pass (D77, D78, D80, D81), last, so the order above stays.
    "pair_together": ("pair_together", NEITHER, "fast", 1),
    "signal_reg_copied": ("signal_reg_copied", NEITHER, "fast", 1),
    "leak_reg_coded": ("leak_reg_coded", NEITHER, "fast", 1),
    "noise_tied": ("noise_tied", NEITHER, "fast", 1),
    "skewed_copy": ("skewed_copy", NEITHER, "fast", 1),
    "last_stamp": ("last_stamp", ["--time-col", "date"] + NO_ENTITIES, "fast", 1),
}
''')
sub('''    "groups.choice_not_made": ["dominant_entity"],
}
''', '''    "groups.choice_not_made": ["dominant_entity"],
    "result.rests_on_recomputed": ["random_walk"],
    "rests_on.near_perfect_named": ["pair_together"],
    "result.copies_share_a_fall": ["signal_reg_copied"],
    "leak.coded_copy_excluded": ["leak_reg_coded"],
    "leak.tied_noise_quiet": ["noise_tied"],
    "leak.changed_cells_excluded": ["skewed_copy"],
    "time.large_holdout_named": ["last_stamp"],
}
''')

# ---------------------------------------------------------------- checks --
sub('''# ---------------------------------------------------------------- fixtures --
''', '''# The seven checks of the ninth pass (D77, D78, D80, D81). Each reads a run of its own,
# but for the first, which reads `random_walk`: a run with one feature, where the gate
# can recompute what a shuffle of that feature's holdout values gives.

REST_KEYS = ("shuffled_score", "shuffled_skill", "skill_fall")
REST_FALL, REST_MOST, REST_SHUFFLES, NEAR_PERFECT, FAR_ABOVE = 0.05, 5, 5, 0.98, 0.5
# |z| of the reported shuffled score of `level_prev` on `random_walk`, against its exact
# mean over every shuffle of the holdout rows, in standard deviations of the mean of five
# shuffles. Measured on the reference (CP10, 30 data seeds): good |z| at most SHUFFLE_Z_GOOD;
# F29 (nothing shuffled) z of SHUFFLE_Z_BROKEN. The cut-off sits between.
SHUFFLE_Z = 5.0


def _rests_on_checked(ctx, result, label):
    """The statement of what the headline rests on, as the gate recomputes it from the
    record's own numbers (spec.md 5.2 and step 12; D77, D78, D80): the three keys of
    every column record and their arithmetic, the floor of the list, and the list.
    Returns the column records by name and the list as (column, fall)."""
    columns = records(result, "columns", ["name", "position", "role"] + list(REST_KEYS))
    skill = at(result, "headline.skill", "number")
    base = at(result, "headline.baseline_score", "number")
    classification = at(result, "task.type") == "classification"
    falls = []
    for c in columns:
        values = tuple(c[k] for k in REST_KEYS)
        if c["role"] != "feature":
            ctx.require(values == (None, None, None),
                        f"`{c['name']}` on `{label}` has role {c['role']!r} and shuffled_score, "
                        f"shuffled_skill, skill_fall {values}: a column that reached no model "
                        "has null in all three (spec.md 5.2; D77)")
            continue
        score, without, fall = values
        ctx.require(all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        for v in values),
                    f"the feature `{c['name']}` on `{label}` has shuffled_score, shuffled_skill, "
                    f"skill_fall {values}: each is a number on a finished run whose skill has a "
                    "value (spec.md 5.2; D77)")
        want = 2 * score - 1 if classification else 1 - score / base
        ctx.require(close(without, want, 1e-7),
                    f"`{c['name']}` on `{label}`: shuffled_skill {without!r} is not the skill of "
                    f"its shuffled_score {score!r} against the headline baseline ({want!r})")
        ctx.require(close(fall, skill - without, 1e-7),
                    f"`{c['name']}` on `{label}`: skill_fall {fall!r} is not headline.skill "
                    f"{skill!r} less shuffled_skill {without!r}")
        falls.append((-fall, c["position"], c["name"], fall))
    low, high = at(result, "headline.skill_low"), at(result, "headline.skill_high")
    floor = at(result, "headline.rests_on_floor", "number")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        ctx.require(close(floor, max(REST_FALL, (high - low) / 2), 1e-9),
                    f"headline.rests_on_floor on `{label}` is {floor!r}; with the skill interval "
                    f"{low!r} to {high!r} it is the larger of {REST_FALL} and the half-width, "
                    f"{max(REST_FALL, (high - low) / 2)!r} (D80 point 1)")
    else:
        ctx.require(floor >= REST_FALL,
                    f"headline.rests_on_floor on `{label}` is {floor!r}, below {REST_FALL}")
    want_list = [(name, fall) for _, _, name, fall in sorted(falls) if fall >= floor]
    want_list = want_list[:REST_MOST]
    got = records(result, "headline.rests_on", ["column", "skill_fall"])
    ctx.record(f"{label} rests_on", ",".join(str(e["column"]) for e in got))
    ctx.require([e["column"] for e in got] == [name for name, _ in want_list]
                and all(close(e["skill_fall"], fall, 1e-12)
                        for e, (_, fall) in zip(got, want_list)),
                f"headline.rests_on on `{label}` is {got!r}; the columns whose fall is at least "
                f"the floor {floor:.6g}, at most {REST_MOST}, largest first and ties in file "
                f"order, are {want_list!r} (D78 point 1, D80 point 1)")
    return {c["name"]: c for c in columns}, want_list


@gate.check("result.rests_on_recomputed", catches=["F29"],
            doc="what the headline rests on: every column record holds its shuffled score, "
                "its skill and its fall, the list and its floor follow from them, the console "
                "prints the list, and the one feature of `random_walk` scores, shuffled, what "
                "a shuffle of the holdout rows gives")
def result_rests_on_recomputed(ctx):
    result = _with_hint(ctx, "random_walk", WALK_HINT)
    by_name, listed = _rests_on_checked(ctx, result, "random_walk")
    lag = by_name.get("level_prev", {})
    # One feature: moving its holdout values to other holdout rows moves the predictions
    # with them, whatever the model. So the shuffled score is the squared error of the
    # holdout targets against a shuffle of predictions.csv, whose mean over every shuffle
    # and whose variance are exact.
    truth = [float(t) for _, t, _ in result["predictions"]]
    guess = [p for _, _, p in result["predictions"]]
    n = len(truth)
    want = (mean([t * t for t in truth]) - 2 * mean(truth) * mean(guess)
            + mean([p * p for p in guess]))
    s_tt = sum((t - mean(truth)) ** 2 for t in truth)
    s_pp = sum((p - mean(guess)) ** 2 for p in guess)
    sd = math.sqrt(4 / n ** 2 * s_tt * s_pp / (n - 1) / REST_SHUFFLES)
    got = lag.get("shuffled_score")
    z = (got - want) / sd if isinstance(got, (int, float)) and sd > 0 else None
    ctx.record("random_walk level_prev shuffled z", None if z is None else round(z, 3))
    ctx.require(z is not None and abs(z) <= SHUFFLE_Z,
                f"the shuffled_score of `level_prev` on `random_walk` is {got!r}. It is the one "
                f"feature, so a shuffle of its holdout values among the holdout rows gives the "
                f"squared error of the holdout targets against a shuffle of the predictions in "
                f"predictions.csv: over every shuffle its mean is {want:.6g}, and the mean of "
                f"{REST_SHUFFLES} shuffles lies within {SHUFFLE_Z} standard deviations "
                f"({sd:.4g} each) of it; this is {z if z is None else round(z, 2)!r} of them. "
                "The statement is of the chosen model's score on the holdout with that "
                "column's holdout values shuffled among the holdout rows, the mean of five "
                "shuffles, nothing refitted (D77 point 1; spec.md step 12)")
    ctx.require([name for name, _ in listed] == ["level_prev"],
                f"headline.rests_on on `random_walk` names {[n for n, _ in listed]}: the "
                "headline rests on yesterday's level, whose fall is far above the floor")
    printed = [line for line in result["stdout"].splitlines() if line.startswith("rests on")]
    ctx.require(len(printed) == 1 and "'level_prev'" in printed[0],
                f"the console summary on `random_walk` prints {printed or 'no line'} for what "
                "the headline rests on; it prints one line that begins `rests on` and names "
                "each column of the list (spec.md 5.5; D78 point 1)")


@gate.check("rests_on.near_perfect_named", catches=["F30"],
            doc="two columns that give the target only together: the headline is near "
                "perfect, and one `near_perfect_headline` warning names both")
def rests_on_near_perfect_named(ctx):
    result = outputs(ctx, "pair_together")
    by_name, listed = _rests_on_checked(ctx, result, "pair_together")
    skill = at(result, "headline.skill", "number")
    ctx.record("pair_together skill", round(skill, 4))
    for name in ("pair_a", "pair_b"):
        ctx.require(by_name.get(name, {}).get("role") == "feature",
                    f"`{name}` on `pair_together` has role {by_name.get(name, {}).get('role')!r}. "
                    "Alone it predicts the target far below the leak screen's cut-offs, so it "
                    "is a feature; the two together give the target")
    ctx.require(skill >= NEAR_PERFECT,
                f"headline.skill on `pair_together` is {skill:.4f}; `pair_a` less `pair_b` is "
                "the target, and a line through the two recovers it: skill about 1.000")
    warned = warnings_of(result, "near_perfect_headline")
    ctx.record("pair_together near_perfect_headline", len(warned))
    ctx.require(len(warned) == 1 and warned[0]["column"] is None,
                f"{len(warned)} warning(s) with the code `near_perfect_headline` on "
                f"`pair_together`, at skill {skill:.4f}. A headline skill at or above "
                f"{NEAR_PERFECT}, the level at which one column alone is excluded as a "
                "suspected leak, always carries one such warning, for the run, which names the "
                "columns the headline rests on and asks whether each is known before the "
                "outcome (D77 point 2; spec.md 2.5)")
    named = [name for name, _ in listed]
    ctx.require({"pair_a", "pair_b"} <= set(named)
                and all(f"'{name}'" in warned[0]["message"] for name in named),
                f"on `pair_together` headline.rests_on names {named} and the warning says: "
                f"{warned[0]['message']!r}. Both columns carry the headline; the warning names "
                "every column of the list")


@gate.check("result.copies_share_a_fall", catches=["F31"],
            doc="a feature and its copy on another scale are shuffled together: their "
                "records hold the same three numbers, and the list names both")
def result_copies_share_a_fall(ctx):
    result = outputs(ctx, "signal_reg_copied")
    by_name, listed = _rests_on_checked(ctx, result, "signal_reg_copied")
    one, two = (tuple(by_name.get(n, {}).get(k) for k in REST_KEYS)
                for n in ("f1", "f1_scaled"))
    ctx.record("signal_reg_copied f1 fall", one[2])
    ctx.require(one == two and isinstance(one[2], (int, float)),
                f"on `signal_reg_copied` `f1` has shuffled_score, shuffled_skill, skill_fall "
                f"{one} and `f1_scaled` {two}. `f1_scaled` is 2 x f1 + 3: imputed and scaled, "
                "the models receive the same numbers twice. Columns that are copies of one "
                "another as the models receive them are shuffled together and share one fall "
                "(D78 point 3, D80 point 4); shuffled one at a time each shows a part of it")
    named = [name for name, _ in listed]
    ctx.require({"f1", "f1_scaled"} <= set(named),
                f"headline.rests_on on `signal_reg_copied` names {named}: `f1` and its copy "
                "carry most of the headline, and their shared fall is far above the floor")


@gate.check("leak.coded_copy_excluded", catches=["F32"],
            doc="the target cut into 30 bands with labels in random order is excluded as a "
                "suspected leak, read by its levels")
def leak_coded_copy_excluded(ctx):
    result = outputs(ctx, "leak_reg_coded")
    columns = {c["name"]: c for c in records(result, "columns", ["name", "role", "reason",
                                                                 "leak_rule"])}
    band_col = columns.get("amount_band", {})
    got = (band_col.get("role"), band_col.get("reason"), band_col.get("leak_rule"))
    ctx.record("leak_reg_coded amount_band", "/".join(map(str, got)))
    ctx.require(got == ("excluded", "suspected_leak", "predicts_target"),
                f"`amount_band` on `leak_reg_coded` has role, reason, leak_rule {got}; skill is "
                f"{at(result, 'headline.skill')!r}. The column is the target cut into "
                f"{CODED_BANDS} bands of equal count, each named by a label drawn at random: "
                "read by a tree on codes of its sorted labels it is noise, and read by its "
                "levels, the mean target of each level among a fold's training rows, it "
                "predicts the target at about 0.99. A text column, and a numeric one with at "
                "least five training rows for each distinct value, is also read by its levels "
                "(D77 point 3, D78 point 5; spec.md step 8)")
    warned = [w for w in warnings_of(result, "suspected_leak") if w["column"] == "amount_band"]
    ctx.require(len(warned) == 1 and "group_col" in warned[0]["message"],
                f"the `suspected_leak` warnings for `amount_band` on `leak_reg_coded` are "
                f"{[w['message'] for w in warned]}: one names it, and for a column read by its "
                "levels the message says to declare it as group_col if it names the entity "
                "each row belongs to (D78 point 6)")
    kept = [n for n in ("f1", "f2", "n1", "n2", "n3", "region")
            if columns.get(n, {}).get("role") != "feature"]
    ctx.require(not kept, f"{kept} on `leak_reg_coded` are not features: only the coded copy "
                          "of the target carries it")


@gate.check("leak.tied_noise_quiet", catches=["F33"],
            doc="two columns of noise whose values repeat stay features that no warning of "
                "the leak screen names")
def leak_tied_noise_quiet(ctx):
    result = outputs(ctx, "noise_tied")
    columns = {c["name"]: c for c in records(result, "columns", ["name", "role", "reason"])}
    named = {w["column"] for w in warnings_of(result, "strong_single_column")
             + warnings_of(result, "suspected_leak")}
    for name in ("score", "noise_rounded"):
        got = (columns.get(name, {}).get("role"), columns.get(name, {}).get("reason"))
        ctx.record(f"noise_tied {name}", "/".join(map(str, got)))
        ctx.require(got == ("feature", None) and name not in named,
                    f"`{name}` on `noise_tied` has role, reason {got}, and the leak screen's "
                    f"warnings name {sorted(map(str, named))}. It is noise whose values repeat "
                    "(a whole number from 1 to 40; noise rounded to one decimal place). The "
                    "order reading compares, between every two neighbouring distinct values, "
                    "every row of one with every row of the other: rows of equal value are "
                    "never put in the target's order, which made the target seem to follow "
                    "the column nearly everywhere (D77 point 4, D78 point 7; spec.md step 8)")


@gate.check("leak.changed_cells_excluded", catches=["F34"],
            doc="a copy of a long-tailed target with five cells moved far off is excluded as "
                "a suspected leak")
def leak_changed_cells_excluded(ctx):
    result = outputs(ctx, "skewed_copy")
    columns = {c["name"]: c for c in records(result, "columns", ["name", "role", "reason",
                                                                 "leak_rule"])}
    copy = columns.get("amount_copy", {})
    got = (copy.get("role"), copy.get("reason"), copy.get("leak_rule"))
    ctx.record("skewed_copy amount_copy", "/".join(map(str, got)))
    ctx.require(got == ("excluded", "suspected_leak", "predicts_target"),
                f"`amount_copy` on `skewed_copy` has role, reason, leak_rule {got}; skill is "
                f"{at(result, 'headline.skill')!r}. It is 1.07 x amount + 3 with {SKEW_CELLS} "
                f"cells moved up by {SKEW_SIZE} times the target's spread. The target's long "
                "tail marks more than 1% of the training rows as far from the median, so none "
                "is set aside; rule 2 still leaves unscored the rows the column misses by "
                "most, and is read against the fold's training median as well (D77 point 5, "
                "D78 point 7; spec.md step 8)")
    kept = [n for n in ("f1", "f2", "n1") if columns.get(n, {}).get("role") != "feature"]
    ctx.require(not kept, f"{kept} on `skewed_copy` are not features: only the copy of the "
                          "target is excluded")


@gate.check("time.large_holdout_named", catches=["F35"],
            doc="under time order a holdout of 60% of the rows, the last date, is named by "
                "one `holdout_far_above_quarter` warning")
def time_large_holdout_named(ctx):
    result = _with_hint(ctx, "last_stamp", WALK_HINT)
    held = set(holdout_check(ctx, result, "last_stamp"))
    at_date = result["header"].index("date")
    last = max(row[at_date] for row in result["source"])
    want = {i for i, row in enumerate(result["source"]) if row[at_date] == last}
    rows = len(result["source"])
    ctx.require(held == want,
                f"the holdout on `last_stamp` holds {len(held)} rows; it is the latest whole "
                f"timestamps that make a quarter, here the last date alone, {len(want)} of the "
                f"{rows} rows (spec.md step 7)")
    warned = warnings_of(result, "holdout_far_above_quarter")
    ctx.record("last_stamp holdout_far_above_quarter", len(warned))
    ctx.require(len(warned) == 1 and warned[0]["column"] is None
                and f"{len(want)} of the {rows} rows" in warned[0]["message"],
                f"the `holdout_far_above_quarter` warnings on `last_stamp` are "
                f"{[w['message'] for w in warned]}. Whole timestamps make the holdout "
                f"{len(want)} of the {rows} rows ({len(want) / rows:.0%}); at a share of "
                f"{FAR_ABOVE} or more, one warning for the run says how many rows of how many "
                "and why (D77 point 6, D78 point 7; spec.md 2.5)")


# ---------------------------------------------------------------- fixtures --
''')

# ------------------------------------------------------------- fixtures --
sub('''if __name__ == "__main__":
    gate.main()''', '''# The seven of the ninth pass (D77, D78, D80, D81): the statement of what the headline
# rests on and the screen's new readings, each with a rule taken out.

@gate.failure_mode("F29", "the statement of what the headline rests on is not of a shuffle "
                          "of the holdout: every column's holdout values are left in place")
def broken_statement_unshuffled(tmp):
    return _fixture(tmp, "F29")


@gate.failure_mode("F30", "a headline skill of 0.98 or more carries no near_perfect_headline "
                          "warning")
def broken_no_near_perfect_warning(tmp):
    return _fixture(tmp, "F30")


@gate.failure_mode("F31", "columns that are copies of one another are shuffled one at a time")
def broken_copies_shuffled_apart(tmp):
    return _fixture(tmp, "F31")


@gate.failure_mode("F32", "the leak screen does not read a column by its levels: a coded copy "
                          "of the target is a feature")
def broken_no_reading_by_levels(tmp):
    return _fixture(tmp, "F32")


@gate.failure_mode("F33", "the order reading puts rows of equal column value in the target's "
                          "order: noise with repeated values is warned about")
def broken_ties_in_target_order(tmp):
    return _fixture(tmp, "F33")


@gate.failure_mode("F34", "where more training rows are marked than may be set aside, rule 2 "
                          "scores every row: a copy with five cells moved is kept")
def broken_every_row_scored_when_marked(tmp):
    return _fixture(tmp, "F34")


@gate.failure_mode("F35", "under time order a holdout of half of the rows or more is not named")
def broken_large_holdout_unnamed(tmp):
    return _fixture(tmp, "F35")


if __name__ == "__main__":
    gate.main()''')
sub('''    and thirty-seven broken copies of it through the same checks; it writes to a folder''',
    '''    and forty-four broken copies of it through the same checks; it writes to a folder''')

DST.write_bytes(text.encode("utf-8"))
print("wrote", DST, len(text))
