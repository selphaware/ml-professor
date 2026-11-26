"""Make the CP10 prototype of the perturbation battery: verify/battery.py as it stands,
extended as section G of the change prompt says.

    python -B patch_battery10.py <verify/battery.py> <out/battery.py>

Every replacement must match exactly once, or the script stops.
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


# ------------------------------------------------- the stated limits of D80 --
sub("""    ("panel", "copy_noise_5pct"): "D76",
}""", """    ("panel", "copy_noise_5pct"): "D76",
    # where the headline skill has no value (a baseline with no error), nothing can be
    # said of how far a carrier moves it, and no fall is computed (D36, D80 point 2)
    ("flat_tail", "two_columns_together"): "D36",
}""")

# ------------------------------------------------------------- the case --
sub('''        self.rounds = 1                         # calls of its own the judge may make
''', '''        self.rounds = 1                         # calls of its own the judge may make
        self.outcome = {}                       # a planted carrier of the target -> what became of it
        self.carrier = None                     # its skill, the run's without it, the floor
''')

# ------------------------------------------------------------ the changes --
sub('''def build_pairs(only=None):''', '''# --------------------------------------- the changes of the ninth pass --
# Added with D77 (section G of CP10). The judge missed the seventh review's defects for
# three reasons: every numeric column of the gate and of this battery was continuous,
# every altered-cell pair used one size of change, and no pair added two columns at
# once.

NEAR_PERFECT = 0.98         # headline skill at or above this: a `near_perfect_headline` warning
NAMED_FALL = 0.05           # ... that names the columns whose fall is at least this
NAMED_MOST = 5              # ... at most this many, in order of their fall
BANDS_ADDED, CODES_ADDED = 30, 40


def _left_to_other_pairs(case):
    if case.run in ("dup_noise", "few_valued_leak"):
        return ("an added column that differs from row to row: on this run that is "
                "`noise_added`, which is listed and has its own words")
    return None


def _noise_tied(name, make, what):
    def change(case, rng):
        if _left_to_other_pairs(case):
            return _left_to_other_pairs(case)
        case.add_column(name, [make(rng) for _ in case.rows])
        _added(case, name, "feature", None)
        _holds(case, what + ": a feature like any other, which predicts nothing and is named by "
                            "no warning of the leak screen")
        if not _stops(case):
            case.extra.append(("quiet", name))
        return None
    return change


def _carrier(case, names, what):
    """The change plants a carrier of the target. It holds when each planted column is
    excluded as a suspected leak, or warned about by name, or the headline is at or
    above 0.98 and the `near_perfect_headline` warning names it (D77 point 9)."""
    if _stops(case):
        case.kind, case.must = "stop", f"the same stop, `{case.stop}`: {what}"
        return
    case.claims = [c for c in case.claims if not c.startswith("band")
                   and c not in ("walk_baselines", "panel_baselines", "flat", "int_keys",
                                 "p_floor", "majority_margin", "blocks_interval")]
    case.extra.append(("carrier",) + tuple(names))
    case.kind = "carrier"
    case.must = ("each of " + ", ".join(f"`{n}`" for n in names) + " is excluded as a suspected "
                 "leak, or named by a `strong_single_column` warning, or named by the "
                 "`near_perfect_headline` warning, or named in `headline.rests_on`: " + what)


def _coded(kind):
    def change(case, rng):
        if case.info.get("malformed"):
            return "the table cannot be read, so no column of it is judged"
        if case.duplicates == "group":
            return ("under duplicates='group' a column that the leak screen removes stops the "
                    "run (D69): `copy_two_valued` and `copy_distinct` are the pairs for that")
        usable, t = _usable_rows(case), case.header.index(case.target)
        cells = [""] * len(case.rows)
        classes = case.classes()
        if classes:
            names = list(range(CODES_ADDED))
            rng.shuffle(names)
            half = CODES_ADDED // 2
            of_class = {classes[0]: names[:half], classes[1]: names[half:]}
            for i in usable:
                name = rng.choice(of_class[case.rows[i][t].strip()])
                cells[i] = f"code_{name:03d}" if kind == "text" else str(100 + name)
            what = (f"a code of {CODES_ADDED} levels, each of which holds one class, the labels "
                    "in random order, as " + ("text" if kind == "text" else "numbers"))
        elif case.regression():
            names = list(range(BANDS_ADDED))
            rng.shuffle(names)
            order = sorted(usable, key=lambda i: float(case.rows[i][t]))
            for position, i in enumerate(order):
                band = names[min(BANDS_ADDED - 1, position * BANDS_ADDED // len(order))]
                cells[i] = f"tier_{band:03d}" if kind == "text" else str(100 + band)
            what = (f"{BANDS_ADDED} bands of the target of equal count, the labels in random "
                    "order, as " + ("text" if kind == "text" else "a numeric code"))
        else:
            return "the target is neither two-valued nor a regression target"
        case.add_column("target_code", cells)
        _carrier(case, ["target_code"], what)
        return None
    return change


def ch_two_together(case, rng):
    if case.info.get("malformed"):
        return "the table cannot be read, so no column of it is judged"
    if _left_to_other_pairs(case):
        return _left_to_other_pairs(case)
    if case.duplicates == "group":
        return ("two columns drawn afresh in every row leave no two rows that agree: nothing "
                "is left of what this run is about")
    usable, t = _usable_rows(case), case.header.index(case.target)
    classes = case.classes()
    if classes:
        values = {i: float(classes.index(case.rows[i][t].strip())) for i in usable}
    elif case.regression():
        values = {i: float(case.rows[i][t]) for i in usable}
    else:
        return "the target is neither two-valued nor a regression target"
    mean = sum(values.values()) / len(values)
    spread = 0.5 if classes else \
        math.sqrt(sum((v - mean) ** 2 for v in values.values()) / len(values)) or 1.0
    a, b = [""] * len(case.rows), [""] * len(case.rows)
    for i in usable:
        z = rng.gauss(0, spread)
        a[i], b[i] = g._f(values[i] + z, 9), g._f(z, 9)
    case.add_column("pair_a", a)
    case.add_column("pair_b", b)
    _carrier(case, ["pair_a", "pair_b"],
             "`pair_a` is the target plus a noise z of the target's own spread and `pair_b` is "
             "z: neither says much alone, and the two together give the target exactly")
    return None


def _typical(case, usable):
    """The target's typical size: under time order its median absolute step, within an
    entity when there is an entity column; else its standard deviation."""
    t = case.header.index(case.target)
    values = {i: float(case.rows[i][t]) for i in usable}
    if isinstance(case.time, str):
        at = case.col(case.time)
        who = case.col(case.group) if isinstance(case.group, str) else None
        series = {}
        try:
            for i in usable:
                if g.is_missing(case.rows[i][at]):
                    continue
                series.setdefault(case.rows[i][who] if who is not None else "", []).append(
                    (time_key(case.rows[i][at]), values[i]))
            steps = []
            for points in series.values():
                points.sort()
                steps += [abs(q[1] - p[1]) for p, q in zip(points, points[1:])]
            steps.sort()
            if steps and steps[len(steps) // 2] > 0:
                return steps[len(steps) // 2], "step"
        except ValueError:
            pass
    mean = sum(values.values()) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values.values()) / len(values)), "spread"


def _cells_changed(size, words):
    def change(case, rng):
        if not case.regression():
            return "the target is not a regression target"
        if case.duplicates == "group":
            return "under duplicates='group' a removed leak stops the run (D69)"
        usable, t = _usable_rows(case), case.header.index(case.target)
        unit, kind = _typical(case, usable)
        values = {i: float(case.rows[i][t]) for i in usable}
        picks = rng.sample(usable, 5)
        for i in picks:
            if size == "other":
                values[i] = float(case.rows[rng.choice([j for j in usable if j != i])][t])
            elif size == "x1.5":
                values[i] = 1.5 * values[i]
            else:
                values[i] = values[i] + size * unit
        case.touched += picks
        cells = [""] * len(case.rows)
        for i in usable:
            cells[i] = g._f(3.7 * values[i] + 12, 9)
        case.add_column("target_copy", cells)
        _carrier(case, ["target_copy"],
                 "a rescaled copy of the target with five cells changed " + words
                 + (f" (the target's typical {kind})" if isinstance(size, (int, float)) else ""))
        return None
    return change


def _feature_rounded(distinct):
    def change(case, rng):
        name = case.info.get("num")
        if not name or case.info.get("malformed"):
            return "no numeric feature"
        if case.info.get("cell") == "feature":
            return (f"`{name}` holds one cell far outside its other values: bands of equal "
                    "width would put every other value into one")
        at = case.col(name)
        held = {}
        for i, row in enumerate(case.rows):
            try:
                held[i] = float(row[at])
            except ValueError:
                pass
        if len(set(held.values())) <= distinct:
            return f"`{name}` has no more than {distinct} distinct values as it is"
        low, high = min(held.values()), max(held.values())
        width = (high - low) / distinct
        for i, v in held.items():
            case.rows[i][at] = g._f(low + (min(distinct - 1, int((v - low) / width)) + 0.5) * width, 6)
        _holds(case, f"the numeric feature `{name}` is put into {distinct} bands of equal width: "
                     f"a column of {distinct} distinct values. It says a little less than it "
                     "did, and nothing else changes")
        if not _stops(case) and not case.stop_ok:
            # a coarser feature can only say less: where the conclusion is "no skill" the
            # band's lower end is not asserted
            case.claims = ["not_above:" + c[5:] if c.startswith("band:")
                           and c[5:] in ("random_walk", "panel") else c for c in case.claims]
            case.claims = [c for c in case.claims if not c.startswith("band_or_named")]
            case.extra.append(("roles_as_base",))
        return None
    return change


def ch_feature_copied(case, rng):
    name = case.info.get("num")
    if not name or case.info.get("malformed"):
        return "no numeric feature"
    if case.duplicates == "group":
        return "the run is about rows that agree; a column written twice changes none of that"
    at = case.col(name)
    case.add_column(name + "_again", [row[at] for row in case.rows])
    _added(case, name + "_again", "feature", None)
    _holds(case, f"the numeric feature `{name}` is written twice: `{name}_again` holds the same "
                 "cells. Two exact copies are shuffled together and share one fall (D78 point 3)")
    if not _stops(case) and not case.stop_ok:
        case.claims = [c for c in case.claims if not c.startswith("band_or_named")]
        case.extra.append(("same_fall", name, name + "_again"))
    return None


CHANGES.update({
    "feature_written_twice": ch_feature_copied,
    "noise_rounded_1dp": _noise_tied("noise_rounded", lambda rng: f"{rng.gauss(0, 1):.1f}",
                                     "a column of noise rounded to one decimal place, so "
                                     "that its values repeat"),
    "noise_score_1_to_40": _noise_tied("noise_score", lambda rng: str(rng.randint(1, 40)),
                                       "a whole-number score from 1 to 40 drawn at random"),
    "coded_copy_text": _coded("text"),
    "coded_copy_number": _coded("number"),
    "two_columns_together": ch_two_together,
    "copy_5_cells_plus_1": _cells_changed(1.0, "by once the target's typical size"),
    "copy_5_cells_plus_20": _cells_changed(20.0, "by 20 times the target's typical size"),
    "copy_5_cells_plus_1000": _cells_changed(1000.0, "by 1,000 times the target's typical size"),
    "copy_5_cells_times_1.5": _cells_changed("x1.5", "to one and a half times their value"),
    "copy_5_cells_other_row": _cells_changed("other", "to the value of another row"),
    "feature_rounded_40": _feature_rounded(40),
    "feature_rounded_10": _feature_rounded(10),
})


def build_pairs(only=None):''')

# ----------------------------------------------------------- the judging --
sub('''                     "u_copies", "u_finite", "u_largest_row"):''',
    '''                     "u_copies", "u_finite", "u_largest_row", "u_rests_on"):''')

sub('''    def c_p_floor(self):''', '''    def c_not_above(self, name):
        skill = self.at("headline.skill")
        high = BANDS[name][1]
        self.require(isinstance(skill, (int, float)) and skill <= high,
                     f"skill is {skill if skill is None else round(skill, 4)}, expected at most "
                     f"{high} (the upper end of the band of `{name}`)")

    def c_p_floor(self):''')

sub('''    def x_roles_as_base(self):''', '''    def named_by_near_perfect(self):
        """The columns the `near_perfect_headline` warning names, in the order of the message."""
        found = self.warned("near_perfect_headline")
        if not found:
            return []
        message = found[0].get("message") or ""
        names = [(message.find(repr(c["name"])), c["name"]) for c in self.at("columns")]
        return [name for at, name in sorted(names) if at >= 0]

    def x_quiet(self, name):
        codes = sorted({w["code"] for w in self.at("warnings")
                        if w.get("column") == name
                        and w["code"] in ("strong_single_column", "suspected_leak")})
        self.require(not codes, f"the added column of noise `{name}` is named by {codes}")
        self.require(name not in self.named_by_near_perfect(),
                     f"the `near_perfect_headline` warning names the column of noise `{name}`")

    def x_same_fall(self, first, second):
        columns = self.columns()
        a, b = columns.get(self.c.name(first), {}), columns.get(second, {})
        keys = ("shuffled_score", "shuffled_skill", "skill_fall")
        self.require(a.get("role") == b.get("role") == "feature",
                     f"`{first}` and `{second}` are {a.get('role')!r} and {b.get('role')!r}")
        self.require(all(a.get(k) == b.get(k) for k in keys),
                     f"`{first}` and `{second}` hold the same cells and do not share one fall: "
                     f"{[a.get(k) for k in keys]} against {[b.get(k) for k in keys]}")

    def x_carrier(self, *names):
        columns, skill = self.columns(), self.at("headline.skill")
        base = self.base["doc"]["headline"].get("skill") \
            if self.base and self.base.get("finished") else None
        floor = self.at("headline.rests_on_floor")
        lift = skill - base if isinstance(skill, (int, float)) \
            and isinstance(base, (int, float)) else None
        base_floor = self.base["doc"]["headline"].get("rests_on_floor") \
            if self.base and self.base.get("finished") else None
        self.c.carrier = {"skill": skill, "base": base, "floor": floor, "lift": lift,
                          "base_floor": base_floor,
                          "falls": {n: columns.get(n, {}).get("skill_fall") for n in names}}
        near = isinstance(skill, (int, float)) and skill >= NEAR_PERFECT
        named = self.named_by_near_perfect() if near else []
        listed = [entry.get("column") for entry in self.at("headline.rests_on")]
        silent = []
        for name in names:
            got = columns.get(name, {})
            if (got.get("role"), got.get("reason")) == ("excluded", "suspected_leak"):
                self.c.outcome[name] = "excluded"
            elif name in self.names_warned("strong_single_column"):
                self.c.outcome[name] = "warned"
            elif name in named:
                self.c.outcome[name] = "named"
            elif name in listed:
                self.c.outcome[name] = "listed"
            else:
                self.c.outcome[name] = "silent"
                silent.append(name)
        if len(silent) < len(names):
            return                  # of two planted columns, either one named counts (D80)
        # D81: the yardstick is the unperturbed run's floor, and only a rise above it fails
        if lift is not None and isinstance(base_floor, (int, float)) and lift <= base_floor:
            for name in names:
                self.c.outcome[name] = "within" if lift >= -base_floor else "lowered"
            return
        self.require(not silent,
                     f"the headline skill is {skill!r}, {lift!r} from the unperturbed run's "
                     f"{base!r}, whose rests-on floor is {base_floor!r}; "
                     f"and {silent} carr{'ies' if len(silent) == 1 else 'y'} the target and "
                     f"{'is' if len(silent) == 1 else 'are'} a feature that nothing names: not "
                     "excluded, no `strong_single_column` warning, "
                     + (f"the `near_perfect_headline` warning names {named}" if near
                        else f"no `near_perfect_headline` warning (skill {skill!r})")
                     + f", and `headline.rests_on` names {listed}"
                     + f"; baseline {self.at('headline.baseline')!r}, model "
                       f"{self.at('headline.model')}, outcomes {self.c.outcome}")

    def u_rests_on(self):
        """D77: what the headline rests on, on every finished run."""
        skill, base = self.at("headline.skill"), self.at("headline.baseline_score")
        classification = self.at("task.type") == "classification"
        falls = []
        for col in self.at("columns"):
            for key in ("shuffled_score", "shuffled_skill", "skill_fall"):
                self.require(key in col, f"the record of `{col['name']}` has no key `{key}`")
            score, without, fall = col["shuffled_score"], col["shuffled_skill"], col["skill_fall"]
            if col["role"] != "feature":
                self.require(score is None and without is None and fall is None,
                             f"`{col['name']}` reached no model and has a statement: "
                             f"{(score, without, fall)}")
                continue
            self.require(isinstance(score, (int, float)) and math.isfinite(score),
                         f"`{col['name']}` is a feature and its `shuffled_score` is {score!r}")
            if skill is None:
                self.require(without is None and fall is None,
                             f"skill has no value and `{col['name']}` has {(without, fall)}")
                continue
            want = 2 * score - 1 if classification else 1 - score / base
            self.require(isinstance(without, (int, float)) and g.close(without, want, 1e-7),
                         f"`{col['name']}`: shuffled_skill {without!r} is not the skill of its "
                         f"shuffled_score ({want!r})")
            self.require(isinstance(fall, (int, float)) and g.close(fall, skill - without, 1e-7),
                         f"`{col['name']}`: skill_fall {fall!r} is not headline.skill - "
                         f"shuffled_skill ({skill - without!r})")
            falls.append((-fall, col["position"], col["name"]))
        floor = self.at("headline.rests_on_floor")
        low, high = self.at("headline.skill_low"), self.at("headline.skill_high")
        if skill is None:
            self.require(floor is None, f"skill has no value and the floor is {floor!r}")
        elif isinstance(low, (int, float)) and isinstance(high, (int, float)):
            self.require(isinstance(floor, (int, float))
                         and g.close(floor, max(NAMED_FALL, (high - low) / 2), 1e-9),
                         f"`headline.rests_on_floor` is {floor!r}; with the skill interval "
                         f"{low!r} to {high!r} it is {max(NAMED_FALL, (high - low) / 2)!r}")
        else:
            self.require(isinstance(floor, (int, float)) and floor >= NAMED_FALL,
                         f"`headline.rests_on_floor` is {floor!r}, below {NAMED_FALL}")
        cut = floor if isinstance(floor, (int, float)) else NAMED_FALL
        want = [name for fall, _, name in sorted(falls) if -fall >= cut][:NAMED_MOST]
        listed = self.at("headline.rests_on")
        self.require(isinstance(listed, list)
                     and [entry.get("column") for entry in listed] == want,
                     f"`headline.rests_on` names {listed!r}; the columns whose fall is at least "
                     f"the floor {cut!r}, the first {NAMED_MOST} in order of their fall, are "
                     f"{want}")
        by_name = self.columns()
        for entry in listed:
            self.require(g.close(entry.get("skill_fall"), by_name[entry["column"]]["skill_fall"]),
                         f"`headline.rests_on` gives {entry!r}; the column's record says "
                         f"{by_name[entry['column']]['skill_fall']!r}")
        near = isinstance(skill, (int, float)) and skill >= NEAR_PERFECT
        found = self.warned("near_perfect_headline")
        self.require(len(found) == (1 if near else 0),
                     f"{len(found)} `near_perfect_headline` warning(s) with skill {skill!r}")
        if near:
            got = self.named_by_near_perfect()
            self.require(got == want and found[0].get("column") is None,
                         f"the `near_perfect_headline` warning names {got}; the columns whose "
                         f"fall is at least the floor {cut!r}, the first {NAMED_MOST} in order "
                         f"of their fall, are {want}")

    def x_roles_as_base(self):''')

# -------------------------------------------------------------- the report --
sub('''        watch = {"const_added", "row_id_added", "ref_added", "added_date", "noise_added",
                 "status_copy", "target_copy",''', '''        watch = {"const_added", "row_id_added", "ref_added", "added_date", "noise_added",
                 "status_copy", "target_copy", "noise_rounded", "noise_score", "target_code",
                 "pair_a", "pair_b",''')
sub('''    "measure_sentinel_-999": "one cell of the whole-number measure `visits` is set to -999",
}''', '''    "measure_sentinel_-999": "one cell of the whole-number measure `visits` is set to -999",
    "noise_rounded_1dp": "a column `noise_rounded` is added: noise rounded to one decimal place",
    "noise_score_1_to_40": "a column `noise_score` is added: a whole number from 1 to 40, at random",
    "coded_copy_text": "a column `target_code` is added: the target as a code, in text, the "
                       "labels in random order",
    "coded_copy_number": "a column `target_code` is added: the target as a numeric code in "
                         "random order",
    "two_columns_together": "columns `pair_a` and `pair_b` are added: the target plus a noise, "
                            "and that noise",
    "copy_5_cells_plus_1": "a column `target_copy` is added: the target rescaled, five cells "
                           "changed by once the target's typical size",
    "copy_5_cells_plus_20": "a column `target_copy` is added: the target rescaled, five cells "
                            "changed by 20 times the target's typical size",
    "copy_5_cells_plus_1000": "a column `target_copy` is added: the target rescaled, five cells "
                              "changed by 1,000 times the target's typical size",
    "copy_5_cells_times_1.5": "a column `target_copy` is added: the target rescaled, five cells "
                              "at one and a half times their value",
    "copy_5_cells_other_row": "a column `target_copy` is added: the target rescaled, five cells "
                              "holding another row's value",
    "feature_written_twice": "a numeric feature is written twice, under a second name",
    "feature_rounded_40": "a numeric feature is put into 40 bands of equal width",
    "feature_rounded_10": "a numeric feature is put into 10 bands of equal width",
}''')
sub('''                  "args": case.args(), "touched": list(case.touched)}''',
    '''                  "args": case.args(), "touched": list(case.touched),
                  "outcome": dict(case.outcome), "carrier": case.carrier}''')
sub('''    if args.json:
        try:
            Path(args.json).write_text(json.dumps(''', '''    planted = [r for r in report if r["outcome"]]
    # a pair counts under the strongest outcome among its planted columns: of two, either
    # one named counts as named (D80 point 2)
    strongest = ("excluded", "warned", "named", "listed", "within", "lowered", "silent")
    tally = {o: 0 for o in strongest}
    for r in planted:
        tally[min(r["outcome"].values(), key=strongest.index)] += 1
    print(f"pairs that plant a carrier of the target: {len(planted)}. Excluded: "
          f"{tally['excluded']}; warned about by name: {tally['warned']}; named by the "
          f"`near_perfect_headline` warning: {tally['named']}; named in `headline.rests_on`: "
          f"{tally['listed']}; the headline within the unperturbed run's floor of it: "
          f"{tally['within']}; the headline lowered by more than that floor: "
          f"{tally['lowered']}; none of these: {tally['silent']}")
    if args.json:
        try:
            Path(args.json).write_text(json.dumps(''')

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_bytes(text.encode("utf-8"))
print("wrote", DST, len(text))
