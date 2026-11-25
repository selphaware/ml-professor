"""CP9 plan, prototype of the changed battery. Outside the repo; never run on the repo's file.

    python -B patch_battery.py <battery.py as it is in the repo> <battery.py to write>

Reads the first file, applies the changes below, writes the second. Every
replacement must match exactly once, so a battery that has moved on stops this
script instead of being patched wrongly.

What changes:
  1. the open list is empty: every pair on it is a must-hold pair (FORMERLY_OPEN
     keeps the names, for the report only);
  2. the judge's own arithmetic follows the proposed rules: the previous value at
     repeated timestamps (the mean target of the timestamp gap + 1 before), the
     headline baseline as the lower of the two by the judge's count;
  3. on every finished run the judge recomputes the largest row's share of the
     squared error, for the model and for the headline baseline (u_largest_row);
  4. the pairs of one extreme cell and of one empty cell assert `band_or_row`;
  5. `starved_fold` on the runs with an entity column and no time column;
  6. the extension: the changes at the end of CHANGES.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
source, target = Path(sys.argv[1]), Path(sys.argv[2])
text = source.read_text(encoding="utf-8")


def swap(old, new, count=1):
    global text
    found = text.count(old)
    if found != count:
        raise SystemExit(f"expected {count} match(es), found {found}, for:\n{old[:300]}")
    text = text.replace(old, new)


# ------------------------------------------------------------------ 1. lists --
swap('''OPEN = {pair: finding for finding, pairs in _OPEN.items() for pair in pairs}
''', '''FORMERLY_OPEN = {pair: finding for finding, pairs in _OPEN.items() for pair in pairs}
OPEN = {}                   # CP9 prototype: nothing is open; every pair above must hold
''')

# ------------------------------------------------------------- the run specs --
swap('''    "entities": dict(claims=["band:entities", "entities_apart", "grouped_share"], num="m1"),''',
     '''    "entities": dict(claims=["band:entities", "entities_apart", "grouped_share",
                             "fold_rule"], num="m1"),''')
swap('''    "dominant_entity": dict(claims=["entities_apart", "holdout_bounded", "cap_keeps_large"],
                            num="n1"),''',
     '''    "dominant_entity": dict(claims=["band:dominant_entity", "entities_apart",
                                    "holdout_bounded", "cap_keeps_large", "fold_rule"],
                            num="n1"),''')
swap('''g = gate_module()
''', '''g = gate_module()
# CP9 prototype: a band the gate does not have yet. `dominant_entity` under the rule
# proposed for A6 (no choice of configuration when a fold trains on fewer than 50
# rows), measured over 30 data seeds by the session's lead: -0.0117 to +0.0030
# (mean -0.0037, sd 0.0040).
BANDS = dict(g.BANDS, dominant_entity=(-0.03, 0.02))
''')
swap('''        low, high = g.BANDS[name]
        self.require(isinstance(skill, (int, float)) and low <= skill <= high,''',
     '''        low, high = BANDS[name]
        self.require(isinstance(skill, (int, float)) and low <= skill <= high,''')
swap('''    "quarter_entity": dict(claims=["entities_apart", "holdout_bounded", "cap_keeps_large"],
                           num="n1", kept=("customer", "C0000")),''',
     '''    "quarter_entity": dict(claims=["entities_apart", "holdout_bounded", "cap_keeps_large",
                                   "fold_rule"],
                           num="n1", kept=("customer", "C0000")),''')

swap('''        self.same = False                       # nothing may move against the unperturbed run
''', '''        self.same = False                       # nothing may move against the unperturbed run
        self.touched = []                       # the rows a change altered, for the report
''')

# ------------------------------------------------- 4. the pairs of one cell --
swap('''def _added(case, name, role, reason, rule=None, warned=None):''',
     '''def _named_row(case):
    """CP9 A3: the pass reports a row that carries most of the error and does not change
    the metric, so the conclusion of such a pair is not the band alone. `band:X`
    becomes `band_or_row:X`, and the roles must be those of the unperturbed run."""
    if _stops(case) or case.stop_ok:
        return
    case.claims = ["band_or_row:" + c[5:] if c.startswith("band:") else c for c in case.claims]
    case.extra.append(("roles_as_base",))
    if any(c.startswith("band_or_row:") for c in case.claims):
        case.kind = "band-or-row"
        case.must += (". Skill is inside the run's band; or a `largest_row_dominates` warning "
                      "is there and the skill the judge recomputes with the named row left out "
                      "is inside the band. The roles are those of the unperturbed run")


def _added(case, name, role, reason, rule=None, warned=None):''')

swap('''    name = _feature(case)
    case.rows[case.pick(rng)][case.col(name)] = ""
    _holds(case, f"one empty cell in the feature `{name}` is counted in `n_missing` and "
                 "imputed")
    if not _stops(case):
        case.extra.append(("n_missing", name))
''', '''    name = _feature(case)
    k = case.pick(rng)
    case.rows[k][case.col(name)] = ""
    case.touched.append(k)
    _holds(case, f"one empty cell in the feature `{name}` is counted in `n_missing` and "
                 "imputed")
    if not _stops(case):
        case.extra.append(("n_missing", name))
    _named_row(case)
''')

swap('''        case.rows[case.pick(rng)][case.col(name)] = value
        _holds(case, f"one cell of the numeric feature `{name}` set to {value}")
''', '''        k = case.pick(rng)
        case.rows[k][case.col(name)] = value
        case.touched.append(k)
        _holds(case, f"one cell of the numeric feature `{name}` set to {value}")
        if abs(float(value)) <= TOO_LARGE:
            _named_row(case)
''')

swap('''    case.rows[case.pick(rng)][case.header.index(case.target)] = "-999"
    _holds(case, "one target cell set to -999 (the spec has no rule for an outlier in the "
                 "target; docs/BACKLOG.md, 'One extreme value can dominate')")
    return None
''', '''    k = case.pick(rng)
    case.rows[k][case.header.index(case.target)] = "-999"
    case.touched.append(k)
    _holds(case, "one target cell set to -999 (the spec has no rule for an outlier in the "
                 "target; docs/BACKLOG.md, 'One extreme value can dominate')")
    _named_row(case)
    return None
''')

# ------------------------------------------------------- 6. the extension --
EXTENSION = r'''

# ------------------------------------------------- CP9: the extension --
# Changes judged most likely to find the next fault in the rules of section A.

def _usable_rows(case):
    t = case.header.index(case.target)
    return [i for i, row in enumerate(case.rows) if not g.is_missing(row[t])]


def _repeat_stop(case):
    """Two rows of one entity at one timestamp, in a regression under time order."""
    if isinstance(case.time, str) and isinstance(case.group, str) and not _stops(case):
        case.stop, case.claims, case.extra, case.kind = "group_column_invalid", [], [], "stop"
        case.mentions = [case.group]
        case.must = ("the stop `group_column_invalid`, naming the entity column: two rows of "
                     "one entity at one timestamp, in a regression under time order (spec "
                     "step 11)")


def _rows_repeated(share):
    def change(case, rng):
        if not isinstance(case.time, str):
            return ("the run declares no time column: repeated rows are there the business of "
                    "the copies rule (step 8a), which has pairs of its own")
        usable = _usable_rows(case)
        picks = rng.sample(usable, max(1, round(share * len(usable))))
        for row in [list(case.rows[i]) for i in picks]:
            case.rows.insert(rng.randrange(len(case.rows) + 1), row)
        _holds(case, f"{round(100 * share)}% of the rows written twice, each copy somewhere in "
                     "the file: under time order with no entities the previous value is the "
                     "mean target of the timestamp before, so the baselines the judge makes by "
                     "hand and the band hold (D73)")
        if not _stops(case):
            case.extra.append(("duplicate_rows",))
        _repeat_stop(case)
        return None
    return change


def ch_row_other_target(case, rng):
    if not isinstance(case.time, str):
        return "the run declares no time column"
    k = case.pick(rng)
    row, t = list(case.rows[k]), case.header.index(case.target)
    classes = case.classes()
    if classes:
        row[t] = classes[1] if row[t].strip() == classes[0] else classes[0]
    else:
        try:
            row[t] = g._f(float(row[t]) + 1.0)
        except ValueError:
            return "the target is neither two-valued nor a number"
    case.rows.insert(rng.randrange(len(case.rows) + 1), row)
    _holds(case, "one row written twice, the second time with another target (the other "
                 "class; for a regression the target plus 1): two rows at one timestamp that "
                 "disagree. The previous value of the rows after them is their mean (D73)")
    _repeat_stop(case)
    return None


def ch_stamp_shared(case, rng):
    if not isinstance(case.time, str):
        return "the run declares no time column"
    at, t = case.col(case.time), case.header.index(case.target)
    usable = [i for i in _usable_rows(case) if not g.is_missing(case.rows[i][at])]
    k = rng.choice(usable)
    try:
        keys = {i: time_key(case.rows[i][at]) for i in usable}
        earlier = [i for i in usable if keys[i] < keys[k]]
        other = max(earlier, key=lambda i: keys[i]) if earlier \
            else min((i for i in usable if keys[i] > keys[k]), key=lambda i: keys[i])
    except ValueError:                      # a time column in no accepted form: any other row
        other = rng.choice([i for i in usable if i != k])
    case.rows[k][at] = case.rows[other][at]
    _holds(case, "one row is given the timestamp before its own: two different rows at one "
                 "timestamp. The split is by whole timestamps and the previous value of the "
                 "rows after them is their mean (D73)")
    _repeat_stop(case)
    return None


def _gap_two(repeat):
    def change(case, rng):
        if case.run not in ("random_walk", "int_stamps", "flat_tail"):
            return "not a regression under time order with no entities that finishes"
        case.gap = 2
        if repeat:
            case.rows.append(list(case.rows[case.pick(rng)]))
        case.claims = [c for c in case.claims if not c.startswith("band:")]
        case.kind = "structure"
        case.must = ("the call passes --gap 2" + (" and one row is written twice" if repeat
                                                   else "")
                     + ": the split and both baselines as the judge makes them by hand, the "
                       "previous value being the mean target of the timestamp three distinct "
                       "timestamps before (no band: with a gap the feature is closer than the "
                       "baseline)")
        if repeat:
            case.extra.append(("duplicate_rows",))
        return None
    return change


def _copy_change(make, what, rotate=0, cell=None, target_cell=None):
    def change(case, rng):
        if not case.regression():
            return ("the target is not a regression target (a two-valued target has "
                    "`copy_two_valued` and `copy_distinct`)")
        t, usable = case.header.index(case.target), _usable_rows(case)
        cells = ["" if g.is_missing(c) else make(float(c), i, c)
                 for i, c in enumerate(case.target_cells())]
        if rotate:
            picks = rng.sample(usable, rotate)
            moved = [cells[i] for i in picks]
            for i, value in zip(picks, moved[1:] + moved[:1]):
                cells[i] = value
            case.touched += picks
        if cell is not None:
            k = rng.choice(usable)
            cells[k] = cell
            case.touched.append(k)
        case.add_column("target_copy", cells)
        _added(case, "target_copy", "excluded", "suspected_leak", "predicts_target",
               "suspected_leak")
        if target_cell is not None:
            k = case.pick(rng)
            case.rows[k][t] = target_cell
            case.touched.append(k)
            _holds(case, what)
            _named_row(case)
        else:
            _same(case, what)
            if not _stops(case):
                case.extra.append(("roles_as_base",))
        return None
    return change


def _rescaled(value, i, cell):
    return g._f(3.7 * value + 12 + i * 1e-9, 9)


def ch_target_sentinel_big(case, rng):
    if not case.regression():
        return "the target is not a regression target"
    k = case.pick(rng)
    case.rows[k][case.header.index(case.target)] = "999999"
    case.touched.append(k)
    _holds(case, "one target cell set to 999999")
    _named_row(case)
    return None


def _two_rows(case, rng):
    first = case.pick(rng)
    second = case.pick(rng)
    while second == first:
        second = case.pick(rng)
    return [first, second]


def ch_target_two_sentinels(case, rng):
    if not case.regression():
        return "the target is not a regression target"
    for k in _two_rows(case, rng):
        case.rows[k][case.header.index(case.target)] = "-999"
        case.touched.append(k)
    _holds(case, "two target cells, in two rows, set to -999")
    _named_row(case)
    return None


def ch_number_two_sentinels(case, rng):
    name = case.info.get("num")
    if not name:
        return "no numeric feature"
    for k in _two_rows(case, rng):
        case.rows[k][case.col(name)] = "999999"
        case.touched.append(k)
    _holds(case, f"two cells of the numeric feature `{name}`, in two rows, set to 999999")
    _named_row(case)
    return None


def _placed(what, late):
    def change(case, rng):
        if not (isinstance(case.time, str) and case.regression()):
            return "not a regression under time order"
        name = case.info.get("num")
        at, t = case.col(case.time), case.header.index(case.target)
        usable = [i for i in _usable_rows(case) if not g.is_missing(case.rows[i][at])]
        try:
            order = sorted(usable, key=lambda i: time_key(case.rows[i][at]))
        except ValueError:
            return "the time column is in no accepted form"
        k = order[-1] if late else order[int(0.4 * len(order))]
        if what == "feature":
            case.rows[k][case.col(name)] = "999999"
        else:
            case.rows[k][t] = "-999"
        case.touched.append(k)
        _holds(case, ("the numeric feature `" + name + "` set to 999999" if what == "feature"
                      else "the target set to -999")
               + (" in the latest row: a holdout row for sure" if late
                  else " in the row four tenths of the way through time: a training row for "
                       "sure"))
        _named_row(case)
        return None
    return change


def _messy_only(case):
    return None if case.run == "messy" else \
        "about the identifier and the whole-number measure of `messy`"


def ch_id_shuffled_and_repeated(case, rng):
    if _messy_only(case):
        return _messy_only(case)
    rng.shuffle(case.rows)
    case.rows.insert(rng.randrange(len(case.rows) + 1), list(case.rows[case.pick(rng)]))
    _holds(case, "the rows are written in another order and one of them twice: `record_id` "
                 "is an identifier all the same")
    case.extra.append(("duplicate_rows",))
    return None


def ch_rows_deleted_shuffled(case, rng):
    if _messy_only(case):
        return _messy_only(case)
    gone = set(rng.sample(range(len(case.rows)), round(0.03 * len(case.rows))))
    case.rows = [row for i, row in enumerate(case.rows) if i not in gone]
    rng.shuffle(case.rows)
    case.blank_after = {i for i in case.blank_after if i < len(case.rows) - 1}
    _holds(case, "3% of the rows are deleted and the rest written in another order: "
                 "`record_id` has gaps and no order, and is an identifier all the same")
    return None


def ch_id_floats_shuffled(case, rng):
    if _messy_only(case):
        return _messy_only(case)
    for name in ("record_id", "visits"):
        i = case.col(name)
        for row in case.rows:
            if not g.is_missing(row[i]):
                row[i] = f"{int(row[i])}.0"
    rng.shuffle(case.rows)
    _holds(case, "the identifier and the measure are written as floats (100001.0) and the "
                 "rows in another order: both are judged by value")
    return None


def ch_sorted_by_measure(case, rng):
    if _messy_only(case):
        return _messy_only(case)
    i = case.col("visits")
    case.rows.sort(key=lambda row: int(row[i]))
    _holds(case, "the table is sorted by the whole-number measure `visits`: it stays a "
                 "feature (D19's own control; today the ordered clause calls it `id_like`, "
                 "the limit D31 accepted)")
    return None


def ch_id_steps_by_7(case, rng):
    if _messy_only(case):
        return _messy_only(case)
    i = case.col("record_id")
    for k, row in enumerate(case.rows):
        row[i] = str(100001 + 7 * k)
    _holds(case, "`record_id` steps by 7 down the file, 100001, 100008, ...: strictly "
                 "ordered and not consecutive, `id_like` by D19's ordered clause")
    return None


def ch_id_added_gaps(case, rng):
    n = len(case.rows)
    case.add_column("row_id_gaps", rng.sample(range(100001, 100002 + int(n / 0.97)), n))
    _added(case, "row_id_gaps", "excluded", "id_like")
    _same(case, "whole numbers distinct in every row, in no order and with 3% of the numbers "
                "of their range missing, are an identifier and reach nothing")
    return None


def ch_entities_merged(case, rng):
    if case.run != "entities":
        return "only on `entities`: the one classification with an entity column and no time"
    at = case.col(case.group)
    names = sorted({row[at] for row in case.rows})
    big = set(names[:round(0.45 * len(names))])
    for row in case.rows:
        if row[at] in big:
            row[at] = "CBIG"
    case.claims = ["band:entities", "entities_apart", "fold_rule"]
    case.kind = "band"
    case.must = ("45% of the entities are given one name: one entity of 675 rows, of both "
                 "classes, beside 165 of five. No entity on both sides, the rule for a fold "
                 "with few training rows as the judge counts it, and no skill (the band of "
                 "`entities`): the label of a held-out entity is still its own")
    return None


EXTENSION_CHANGES = {
    "rows_repeated_1pct": _rows_repeated(0.01),
    "rows_repeated_10pct": _rows_repeated(0.10),
    "row_repeated_other_target": ch_row_other_target,
    "stamp_shared": ch_stamp_shared,
    "gap_2": _gap_two(False),
    "gap_2_row_repeated": _gap_two(True),
    "copy_exact": _copy_change(lambda value, i, cell: cell.strip(),
                               "an exact copy of the target is excluded by the leak screen, "
                               "after the split, so nothing else moves"),
    "copy_negative_scale": _copy_change(lambda value, i, cell: g._f(-2.5 * value + 40, 9),
                                        "a copy of the target with a negative scale "
                                        "(-2.5 x target + 40) is excluded by the leak screen, "
                                        "so nothing else moves"),
    "copy_5_cells_differ": _copy_change(_rescaled,
                                        "a rescaled copy of the target in which the values of "
                                        "five rows are rotated among them is excluded by the "
                                        "leak screen, so nothing else moves", rotate=5),
    "copy_one_cell_999999": _copy_change(_rescaled,
                                         "a rescaled copy of the target with one cell set to "
                                         "999999 is excluded by the leak screen, so nothing "
                                         "else moves", cell="999999"),
    "copy_and_target_sentinel": _copy_change(_rescaled,
                                             "a rescaled copy of the target, and then one "
                                             "target cell set to -999: the copy is excluded by "
                                             "the leak screen", target_cell="-999"),
    "target_sentinel_999999": ch_target_sentinel_big,
    "target_two_sentinels": ch_target_two_sentinels,
    "number_two_sentinels": ch_number_two_sentinels,
    "feature_999999_latest_row": _placed("feature", True),
    "feature_999999_early_row": _placed("feature", False),
    "target_-999_latest_row": _placed("target", True),
    "target_-999_early_row": _placed("target", False),
    "id_shuffled_and_repeated": ch_id_shuffled_and_repeated,
    "rows_deleted_shuffled": ch_rows_deleted_shuffled,
    "id_floats_shuffled": ch_id_floats_shuffled,
    "sorted_by_measure": ch_sorted_by_measure,
    "id_steps_by_7": ch_id_steps_by_7,
    "id_added_gaps": ch_id_added_gaps,
    "entities_merged_45pct": ch_entities_merged,
}
CHANGES.update(EXTENSION_CHANGES)


def build_pairs(only=None):'''
swap('''

def build_pairs(only=None):''', EXTENSION)

# ------------------------------------------- 2, 3, 5. the judge's arithmetic --
swap('''        for name in ("u_rows", "u_columns", "u_headline", "u_statements", "u_random_counts",
                     "u_copies", "u_finite"):''',
     '''        for name in ("u_rows", "u_columns", "u_headline", "u_statements", "u_random_counts",
                     "u_copies", "u_finite", "u_largest_row"):''')

OLD_BY_HAND = text[text.index("    def series_before(self, by_entity=False):"):
                   text.index("    def check_baselines(self, base_mean, base_previous):")]
NEW_BY_HAND = r'''    def kept_in_time(self):
        """Under time order: the timestamp of each row the cap keeps, and the judge's split."""
        stamps, want = self.time_expectation()
        kept = want["training"] | want["gap"] | want["holdout"]
        return {i: stamps[i] for i in kept}, want

    def previous_values(self, by_entity=False):
        """Each kept row's previous value as step 11 has it (D10, D57, D73), where it has one.

        With no entity column: the mean target of the rows at the timestamp `gap + 1`
        distinct timestamps before the row's own; with unique timestamps that is the
        row before. With one: the target of the latest row of the same entity at
        least `gap + 1` distinct timestamps before. Returns (the previous value by
        row, whether the timestamps are unique within each series, the rows of each
        series, the target by row)."""
        header, source, c = self.r["header"], self.r["source"], self.c
        t = header.index(c.target)
        stamps, _ = self.kept_in_time()
        index = {s: k for k, s in enumerate(sorted(set(stamps.values())))}
        level = {i: float(source[i][t]) for i in stamps}
        groups = {}
        for i in sorted(stamps):
            entity = source[i][header.index(c.group)].strip() if by_entity else None
            groups.setdefault(entity, []).append(i)
        unique = all(len({stamps[i] for i in members}) == len(members)
                     for members in groups.values())
        previous = {}
        if by_entity:
            for members in groups.values():
                members.sort(key=lambda i: stamps[i])
                for position, i in enumerate(members):
                    limit = index[stamps[i]] - c.gap - 1
                    for j in reversed(members[:position]):
                        if index[stamps[j]] <= limit:
                            previous[i] = level[j]
                            break
        else:
            at = {}
            for i, s in stamps.items():
                at.setdefault(index[s], []).append(level[i])
            mean_at = {k: g.mean(values) for k, values in at.items()}
            for i, s in stamps.items():
                k = index[s] - c.gap - 1
                if k >= 0:
                    previous[i] = mean_at[k]
        return previous, unique, groups, level

    def baselines_by_hand(self, by_entity=False):
        previous, unique, groups, level = self.previous_values(by_entity)
        _, want = self.kept_in_time()
        held = self.held()
        stray = [p for p in held if p not in level]
        self.require(not stray, f"{len(stray)} holdout rows are not among the rows the judge "
                                "expects to be split under time order")
        truth = [level[p] for p in held]
        training_mean = g.mean([level[i] for i in want["training"]])
        base_mean = g.mse(truth, [training_mean] * len(truth))
        without = [p for p in held if p not in previous]
        base_previous = g.mse(truth, [previous.get(p, training_mean) for p in held])
        mine = g.mse(truth, [guess for _, _, guess in self.r["predictions"]])
        return base_mean, base_previous, mine, unique, without, groups

    def baseline_predictions(self):
        """The headline baseline's prediction for each holdout row, by the judge's own
        count; None where the output does not say which rows were split (a row cap
        under the random or the grouped scheme)."""
        c, name = self.c, self.at("headline.baseline")
        header, source = self.r["header"], self.r["source"]
        t, held = header.index(c.target), self.held()
        if isinstance(c.time, str):
            previous, _, _, level = self.previous_values(isinstance(c.group, str))
            _, want = self.kept_in_time()
            mean = g.mean([level[i] for i in want["training"]])
            if name == "training_mean":
                return [mean] * len(held)
            return [previous.get(p, mean) for p in held]
        sides = self.split_rows()
        if sides is None or name != "training_mean":
            return None
        return [g.mean([float(source[i][t]) for i in sides[0]])] * len(held)

    def row_errors(self):
        """(holdout rows, the model's squared error on each, the headline baseline's or None)."""
        held = self.held()
        truths = [float(truth) for _, truth, _ in self.r["predictions"]]
        own = [(t - guess) ** 2 for t, (_, _, guess) in zip(truths, self.r["predictions"])]
        base = self.baseline_predictions()
        theirs = None if base is None else [(t - b) ** 2 for t, b in zip(truths, base)]
        return held, own, theirs

    @staticmethod
    def largest(errors, held):
        """(the largest row's share of the squared error, that row, whether it is alone)."""
        total = math.fsum(errors)
        if not total > 0:
            return None, None, False
        top = max(errors)
        return top / total, held[errors.index(top)], errors.count(top) == 1

    def u_largest_row(self):
        """CP9 A3: the share of the squared error on the holdout that its largest row
        carries, for the chosen model and for the headline baseline, and one
        `largest_row_dominates` warning for each share above one half."""
        h = self.at("headline")
        keys = ("largest_row_share", "largest_row", "baseline_largest_row_share",
                "baseline_largest_row")
        missing = [key for key in keys if key not in h]
        self.require(not missing, f"`headline` has no {missing}")
        warned = len(self.warned("largest_row_dominates"))
        if self.at("task.type") != "regression":
            self.require(all(h[key] is None for key in keys) and warned == 0,
                         f"a classification reports {[h[key] for key in keys]} and {warned} "
                         "`largest_row_dominates` warning(s); squared error is not its metric")
            return
        held, own, theirs = self.row_errors()
        expected = 0
        for label, errors, key in (("the chosen model", own, ""),
                                   ("the headline baseline", theirs, "baseline_")):
            got, got_row = h[key + "largest_row_share"], h[key + "largest_row"]
            if errors is None:              # the judge cannot make this baseline's predictions
                self.require(got is None or (isinstance(got, (int, float)) and 0 < got <= 1),
                             f"`headline.{key}largest_row_share` is {got!r}")
                expected += bool(got is not None and got > 0.5)
                continue
            share, row, alone = self.largest(errors, held)
            if share is None:
                self.require(got is None and got_row is None,
                             f"`headline.{key}largest_row_share` is {got!r} where {label} has "
                             "no error on the holdout")
                continue
            self.require(isinstance(got, (int, float)) and not isinstance(got, bool)
                         and g.close(got, share, 1e-7),
                         f"`headline.{key}largest_row_share` is {got!r}; the judge computes "
                         f"{share!r} for {label} (row {row})")
            if alone:
                self.require(got_row == row, f"`headline.{key}largest_row` is {got_row!r}; the "
                                             f"judge finds row {row}")
            expected += share > 0.5
        self.require(warned == expected,
                     f"{warned} `largest_row_dominates` warning(s); the judge expects "
                     f"{expected}: largest_row_share {h['largest_row_share']!r}, "
                     f"baseline_largest_row_share {h['baseline_largest_row_share']!r}")

    def where_touched(self):
        """Where the rows the change altered fell, for the report."""
        rows = list(getattr(self.c, "touched", []))
        if not rows:
            return "nowhere (the change altered no single row)"
        held, want = set(self.held()), None
        if isinstance(self.c.time, str):
            try:
                _, want = self.time_expectation()
            except Exception:  # noqa: BLE001  a time column the judge cannot read
                want = None
        out = []
        for k in rows:
            if k in held:
                where = "holdout"
            elif want is not None:
                where = "training" if k in want["training"] else "gap" if k in want["gap"] \
                    else "left out by the cap"
            else:
                where = "training, or left out by the cap" if self.capped() else "training"
            out.append(f"row {k}: {where}")
        return "; ".join(out)

    def c_band_or_row(self, name):
        """CP9 A3: skill inside the band; or a `largest_row_dominates` warning, and the
        skill the judge recomputes with the named row left out inside the band."""
        skill = self.at("headline.skill")
        low, high = BANDS[name]
        if isinstance(skill, (int, float)) and low <= skill <= high:
            return
        shown, h = skill if skill is None else round(skill, 4), self.at("headline")
        where = self.where_touched()
        self.require(self.warned("largest_row_dominates"),
                     f"skill is {shown}, outside the band of `{name}` ({low} to {high}), and no "
                     f"`largest_row_dominates` warning says why. The changed row(s): {where}. "
                     f"largest_row_share {h.get('largest_row_share')!r}, "
                     f"baseline_largest_row_share {h.get('baseline_largest_row_share')!r}; "
                     f"model {h.get('model')}, baseline {h.get('baseline')}")
        held, own, theirs = self.row_errors()
        self.require(theirs is not None, "the judge cannot make the baseline's predictions "
                                         "here, so it cannot recompute skill without the row")
        named = set()
        for errors in (own, theirs):
            share, row, _ = self.largest(errors, held)
            if share is not None and share > 0.5:
                named.add(row)
        keep = [i for i, p in enumerate(held) if p not in named]
        rest = math.fsum(theirs[i] for i in keep)
        self.require(rest > 0, f"with row(s) {sorted(named)} left out the baseline has no error")
        without = 1 - math.fsum(own[i] for i in keep) / rest
        self.require(low <= without <= high,
                     f"skill is {shown}, outside the band of `{name}` ({low} to {high}); a "
                     f"`largest_row_dominates` warning names row(s) {sorted(named)}, and with "
                     f"them left out the judge computes skill {without:.4f}: still outside the "
                     f"band. The changed row(s): {where}. largest_row_share "
                     f"{h.get('largest_row_share')!r}, baseline_largest_row_share "
                     f"{h.get('baseline_largest_row_share')!r}; model {h.get('model')}, "
                     f"baseline {h.get('baseline')}")

    def c_fold_rule(self):
        """CP9 A6, the rule of the final prototype: under the grouped scheme every fold
        is scored as it is, and when one trains on fewer than 50 rows no choice between
        configurations is made: the first configuration is used, and one
        `selection_not_made` warning says so. By the judge's own count there is such
        a fold when one entity outside the holdout holds all but fewer than 50 of the
        training rows. (BAT_A6=units asserts the rule the first snapshot had: a fold
        that trains on fewer rows than it validates on is not scored, with one
        `fold_not_scored` warning.)"""
        folds = self.at("validation.folds")
        shape = [(f["training_rows"], f["validation_rows"], f["scored"]) for f in folds]
        if os.environ.get("BAT_A6") != "units":
            floor = 50
            starved = [f["training_rows"] for f in folds
                       if f["training_rows"] < floor and f["training_rows"] < f["validation_rows"]]
            fewest = min(starved) if starved else floor
            said = len(self.warned("selection_not_made"))
            first = next(m["config"] for m in self.at("models") if m["cv_mean"] is not None)
            if fewest < floor:
                self.require(said == 1 and self.at("cv.chosen") == first
                             and self.at("headline.model") == first,
                             f"one fold trains on {fewest} rows, fewer than {floor}: "
                             f"{said} `selection_not_made` warning(s), chosen "
                             f"{self.at('cv.chosen')!r}, the first configuration is {first!r}; "
                             f"the folds are {shape}")
            else:
                self.require(said == 0, f"{said} `selection_not_made` warning(s) though every "
                                        f"fold trains on at least {floor} rows: {shape}")
            size, held_entities, _ = self.entity_sizes()
            training, usable = self.counts()["training"], sum(size.values())
            big = max((rows for e, rows in size.items() if e not in held_entities), default=0)
            if (not self.capped() or 4 * big >= usable) and training - big < floor \
                    and training - big < big:
                self.require(fewest < floor,
                             f"an entity of {big} rows is among the {training} training rows, "
                             f"so the fold that validates on it trains on fewer than {floor}; "
                             f"the folds are {shape}")
            unscored = [f for f in folds if not f["scored"]]
            self.require(len(self.warned("fold_not_scored")) == len(unscored),
                         f"{len(self.warned('fold_not_scored'))} `fold_not_scored` warning(s) "
                         f"for {len(unscored)} fold(s) not scored")
            if self.at("task.type") == "regression":
                self.require(not unscored, f"fold(s) {[f['fold'] for f in unscored]} of a "
                                           "regression are not scored")
            return
        starved = [f for f in folds if f["training_rows"] < f["validation_rows"]]
        scored = [f["fold"] for f in starved if f["scored"]]
        self.require(not scored, f"fold(s) {scored} train on fewer rows than they validate on "
                                 f"and are scored; the folds (training rows, validation rows, "
                                 f"scored) are {shape}")
        size, held_entities, _ = self.entity_sizes()
        training, usable = self.counts()["training"], sum(size.values())
        big = max((rows for e, rows in size.items() if e not in held_entities), default=0)
        if (not self.capped() or 4 * big >= usable) and 2 * big > training:
            self.require(len(starved) == 1 and starved[0]["validation_rows"] >= big,
                         f"an entity of {big} rows is among the {training} training rows, so "
                         f"one fold validates on at least {big} rows and trains on fewer; the "
                         f"folds are {shape}")
        unscored = [f for f in folds if not f["scored"]]
        self.require(len(self.warned("fold_not_scored")) == len(unscored),
                     f"{len(self.warned('fold_not_scored'))} `fold_not_scored` warning(s) for "
                     f"{len(unscored)} fold(s) not scored")
        if self.at("task.type") == "regression":
            idle = [f["fold"] for f in unscored if f["training_rows"] >= f["validation_rows"]]
            self.require(not idle, f"fold(s) {idle} of a regression are not scored though each "
                                   "trains on at least as many rows as it validates on")

    def lower_baseline(self, base_mean, base_previous, mine):
        """The headline baseline is the one with the lower error on the holdout by the
        judge's own count, a tie going to the training mean (step 11)."""
        name, best = ("previous_value", base_previous) if base_previous < base_mean \
            else ("training_mean", base_mean)
        skill = self.at("headline.skill")
        self.require(self.at("headline.baseline") == name,
                     f"the headline baseline is {self.at('headline.baseline')!r} (skill "
                     f"{skill!r}); the previous value scores {base_previous:.6g} against "
                     f"{base_mean:.6g} for the training mean")
        if best == 0:
            self.require(skill is None, f"skill is {skill!r} with a baseline error of 0")
        else:
            self.require(skill is not None and g.close(skill, 1 - mine / best, 1e-7),
                         f"headline.skill {skill!r} is not 1 - {mine!r} / {best!r}")

'''
swap(OLD_BY_HAND, NEW_BY_HAND)

swap('''        base_mean, base_previous, mine, unique, _, _ = self.baselines_by_hand()
        if not unique:
            baselines = sorted(b["name"] for b in self.at("baselines"))
            self.require("previous_value" in baselines,
                         f"the timestamps of the source are not all distinct, the baselines "
                         f"are {baselines} and the headline is skill "
                         f"{self.at('headline.skill')!r} against "
                         f"{self.at('headline.baseline')!r}: a random walk scored against the "
                         "training mean; warnings "
                         f"{sorted({w['code'] for w in self.at('warnings')})}")
            return
        self.check_baselines(base_mean, base_previous)
        skill = self.at("headline.skill")
        self.require(self.at("headline.baseline") == "previous_value",
                     f"the headline baseline is {self.at('headline.baseline')!r} (skill "
                     f"{skill!r}); the previous value scores {base_previous:.4g} against "
                     f"{base_mean:.4g} for the training mean")
        self.require(g.close(skill, 1 - mine / base_previous, 1e-7),
                     "headline.skill is not 1 - error / previous-value error")
''', '''        # CP9 A1: with repeated timestamps too, the previous value is scored: the mean
        # target of the timestamp before (D73).
        base_mean, base_previous, mine, unique, _, _ = self.baselines_by_hand()
        self.check_baselines(base_mean, base_previous)
        self.lower_baseline(base_mean, base_previous, mine)
''')

swap('''        base_mean, base_previous, mine, unique, _, _ = self.baselines_by_hand()
        self.require(unique, "the timestamps of the source are not all distinct; baselines "
                             f"{sorted(b['name'] for b in self.at('baselines'))}, skill "
                             f"{self.at('headline.skill')!r} against "
                             f"{self.at('headline.baseline')!r}")
        self.check_baselines(base_mean, base_previous)
''', '''        base_mean, base_previous, mine, unique, _, _ = self.baselines_by_hand()
        self.check_baselines(base_mean, base_previous)
''')

swap('''        self.check_baselines(base_mean, base_previous)
        skill = self.at("headline.skill")
        self.require(self.at("headline.baseline") == "previous_value",
                     f"the headline baseline is {self.at('headline.baseline')!r}, skill {skill!r}")
        self.require(g.close(skill, 1 - mine / base_previous, 1e-7),
                     "headline.skill is not 1 - error / previous-value error")
        got = (self.at("validation.group_col"), self.at("validation.n_groups_training"),''',
     '''        self.check_baselines(base_mean, base_previous)
        self.lower_baseline(base_mean, base_previous, mine)
        got = (self.at("validation.group_col"), self.at("validation.n_groups_training"),''')

swap('''    def x_duplicate_rows(self):''', '''    def x_roles_as_base(self):
        base = self.base
        self.require(base is not None and base.get("finished"),
                     "the unperturbed run did not finish, so there is nothing to compare with")
        theirs = {c["name"]: (c["role"], c["reason"]) for c in base["doc"]["columns"]}
        back = {new: old for old, new in self.c.alias.items()}
        moved = [f"`{c['name']}` {theirs[back.get(c['name'], c['name'])]} -> "
                 f"{(c['role'], c['reason'])}" for c in self.at("columns")
                 if back.get(c["name"], c["name"]) in theirs
                 and theirs[back.get(c["name"], c["name"])] != (c["role"], c["reason"])]
        self.require(not moved, "roles moved against the unperturbed run: " + "; ".join(moved[:6]))

    def x_duplicate_rows(self):''')

# ------------------------------------------------------------- the report --
swap('''        rows = {r["item"]: r["count"] for r in doc["rows"]}
        seen["cap"] = rows.get("dropped_row_cap")
''', '''        rows = {r["item"]: r["count"] for r in doc["rows"]}
        seen["cap"] = rows.get("dropped_row_cap")
        for key, label in (("largest_row_share", "share"),
                           ("baseline_largest_row_share", "base_share")):
            value = doc["headline"].get(key)
            seen[label] = None if value is None else round(value, 4)
        seen["folds"] = [(f["training_rows"], f["validation_rows"], f["scored"])
                         for f in doc["validation"].get("folds", [])] \\
            if doc["validation"].get("scheme") == "grouped" else None
        if case.touched:
            try:
                seen["changed"] = Judge(case, result, None).where_touched()
            except Exception as exc:  # noqa: BLE001
                seen["changed"] = f"(not placed: {type(exc).__name__})"
''')

swap('''    "duplicates_group": "the call passes --duplicates group",
}''', '''    "duplicates_group": "the call passes --duplicates group",
    "rows_repeated_1pct": "1% of the rows are written twice",
    "rows_repeated_10pct": "10% of the rows are written twice",
    "row_repeated_other_target": "one row is written twice, the second time with another target",
    "stamp_shared": "one row is given the timestamp before its own",
    "gap_2": "the call passes --gap 2",
    "gap_2_row_repeated": "the call passes --gap 2 and one row is written twice",
    "copy_exact": "a column `target_copy` is added: the target as it is",
    "copy_negative_scale": "a column `target_copy` is added: -2.5 x the target + 40",
    "copy_5_cells_differ": "a column `target_copy` is added: the target rescaled, the values "
                           "of five rows rotated among them",
    "copy_one_cell_999999": "a column `target_copy` is added: the target rescaled, one cell "
                            "set to 999999",
    "copy_and_target_sentinel": "a column `target_copy` is added, the target rescaled, and "
                                "then one target cell is set to -999",
    "target_sentinel_999999": "one cell of the regression target is set to 999999",
    "target_two_sentinels": "two cells of the regression target, in two rows, are set to -999",
    "number_two_sentinels": "two cells of a numeric feature, in two rows, are set to 999999",
    "feature_999999_latest_row": "the numeric feature of the latest row is set to 999999",
    "feature_999999_early_row": "the numeric feature of a row four tenths of the way through "
                                "time is set to 999999",
    "target_-999_latest_row": "the target of the latest row is set to -999",
    "target_-999_early_row": "the target of a row four tenths of the way through time is set "
                             "to -999",
    "id_shuffled_and_repeated": "the rows are written in another order and one of them twice",
    "rows_deleted_shuffled": "3% of the rows are deleted and the rest written in another order",
    "id_floats_shuffled": "the identifier and the measure are written as floats and the rows "
                          "in another order",
    "sorted_by_measure": "the table is sorted by the whole-number measure `visits`",
    "id_steps_by_7": "the identifier steps by 7 down the file",
    "id_added_gaps": "a column `row_id_gaps` is added: distinct whole numbers in no order, "
                     "3% of their range missing",
    "entities_merged_45pct": "45% of the entities are given one name",
}''')

swap('''                  "seconds": result.get("seconds"), "attempts": result.get("attempts"),
                  "args": case.args()}''',
     '''                  "seconds": result.get("seconds"), "attempts": result.get("attempts"),
                  "args": case.args(), "touched": list(case.touched),
                  "formerly_open": FORMERLY_OPEN.get((case.run, case.change)),
                  "extension": case.change in EXTENSION_CHANGES}''')

target.write_bytes(text.encode("utf-8"))
print(f"wrote {target}: {len(text.splitlines())} lines")
