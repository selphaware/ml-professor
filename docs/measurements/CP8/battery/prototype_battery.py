"""The perturbation battery (prototype; D71).

For every gate run whose right conclusion is known, apply each of a fixed list
of small changes to the table (or to the call) and assert that the conclusion
still holds: the same stop with the same code, the same columns excluded for
the same reasons, the same structural property counted by the judge from the
output, skill inside the same band, or, where the change cannot move a number
at all, the same headline and predictions as the unperturbed run.

    python battery.py <package folder> [--only run[:change],...] [--jobs N]
                      [--list] [--json FILE] [--work FOLDER]

The tool is run as the gate runs it: a new process for each table,
`python -m mlprofessor profile ...` from the package's parent, fast profile,
seed 0, one core. The datasets, the bands and the arithmetic are the gate's
own (verify/smoke_test.py, loaded as a module); nothing is asserted that the
gate could not recompute from the two files and the source.

Exit codes: 0 every pair holds, 1 at least one does not, 3 harness error.
Standard library only.
"""
import argparse
import datetime
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
GATE_FILE = Path(os.environ.get("MLPROF_GATE_FILE",
                                HERE.parent / "before" / "verify" / "smoke_test.py"))
EXIT_PASS, EXIT_FAIL, EXIT_HARNESS = 0, 1, 3
RUN_SECONDS = 180
DATA = Path(os.environ.get("MLPROF_BATTERY_DATA", HERE / "data"))    # the tables, written once
TOO_LARGE = 3.4028234663852886e+38
MISSING = object()


def gate_module():
    """The gate script as a module: its datasets, bands and arithmetic."""
    sys.path.insert(0, str(GATE_FILE.parent))
    spec = importlib.util.spec_from_file_location("mlprof_gate", str(GATE_FILE))
    module = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [str(GATE_FILE)]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


g = gate_module()

# ------------------------------------------------------------ the runs --
# run -> what its conclusion is. `stop`: the named error it must end in.
# `claims`: what the judge recomputes on a finished run. `num`, `cat`: a numeric
# and a categorical feature of the table. `date`: a date column that is not the
# declared time column. `ymd`: a column of dates written YYYYMMDD.
FOUR = ["--time-col", "--no-time-order", "--group-col", "--no-entities"]
RUNSPEC = {
    "signal_clf": dict(claims=["band:signal_clf", "families_fast"], num="f1", cat="seg"),
    "signal_reg": dict(claims=["band:signal_reg"], num="f1", cat="region"),
    "noise_clf": dict(claims=["band:noise_clf", "p_floor"], num="n001"),
    "noise_reg": dict(claims=["band:noise_reg"], num="n001"),
    "imbalanced": dict(claims=["band:imbalanced", "majority_margin"], num="n001", cat="region"),
    "leak": dict(claims=["band:signal_clf", "leak_roles"], num="f1", cat="seg"),
    "leak_reg": dict(claims=["band:signal_reg", "leak_reg_roles"], num="f1", cat="region"),
    "near_leak": dict(claims=["near_leak_roles"], num="f1", cat="seg"),
    "binary_numeric": dict(claims=["band:signal_clf", "task_binary"], num="f1", cat="seg"),
    "ambiguous_target": dict(stop="task_ambiguous", num="f1", cat="seg"),
    "messy": dict(claims=["band:messy", "messy_roles", "messy_coercion",
                          "undeclared_date:signup_date"], num="f1", cat="seg",
                  date="signup_date"),
    "long_rows": dict(stop="input_error", num="f1", cat="seg", malformed=True),
    "dup_noise": dict(stop="holdout_copies_in_training", num="n001"),
    "time_blocks": dict(claims=["band:time_blocks", "time_split", "blocks_interval"], num="b1"),
    "time_blocks_gap": dict(claims=["time_split", "gap_folds"], num="b1"),
    "time_blocks_undeclared": dict(claims=["undeclared_date:date"], num="b1", date="date"),
    "time_blocks_unstated": dict(stop="time_order_undeclared",
                                 flags=["--time-col", "--no-time-order"], num="b1", date="date"),
    "time_dayfirst": dict(stop="time_column_invalid", num="b1"),
    "random_walk": dict(claims=["band:random_walk", "time_split", "walk_baselines"],
                        num="level_prev"),
    "int_stamps": dict(claims=["time_split", "walk_baselines", "int_keys"], num="level_prev"),
    "flat_tail": dict(claims=["time_split", "flat"], num="footfall"),
    "entities": dict(claims=["band:entities", "entities_apart", "grouped_share"], num="m1"),
    "panel": dict(claims=["band:panel", "time_split", "panel_baselines"], num="level_prev"),
    "unsaid_both": dict(stop="time_order_undeclared", flags=FOUR, num="f1", cat="seg"),
    "unsaid_entities": dict(stop="entities_undeclared", flags=["--group-col", "--no-entities"],
                            num="level_prev"),
    "few_valued": dict(claims=["band:few_valued", "combos_apart", "group_units"], cat="branch"),
    "dominant_entity": dict(claims=["entities_apart", "holdout_bounded"], num="n1"),
    "day_numbers": dict(claims=["day_role", "undeclared_date:day"], num="n1", ymd="day",
                        date="day"),
}
# Runs of the gate that are left out, and why.
LEFT_OUT_RUNS = {
    "api_signal_clf": "the same table as `signal_clf` through the Python function; its "
                      "conclusion is that two accounts of one run agree, which no change to "
                      "the table bears on",
    "full_noise_small": "full profile: 18 configurations for each pair; the fast run "
                        "`noise_clf` carries the same conclusion",
    "full_signal_clf_j1": "full profile; about the core count, not about the table",
    "full_signal_clf_j4": "full profile; about the core count, not about the table",
    "full_signal_reg_j1": "full profile; about the core count, not about the table",
    "full_signal_reg_j4": "full profile; about the core count, not about the table",
}


def parse_statements(extra):
    time_col = group = dup = None
    gap, words = 0, iter(extra)
    for word in words:
        if word == "--time-col":
            time_col = next(words)
        elif word == "--no-time-order":
            time_col = False
        elif word == "--group-col":
            group = next(words)
        elif word == "--no-entities":
            group = False
        elif word == "--gap":
            gap = int(next(words))
        elif word == "--duplicates":
            dup = next(words)
    return time_col, group, gap, dup


class Case:
    """One table and one call: a gate run, as it is or with one change."""

    def __init__(self, run):
        dataset, extra, _, _ = g.RUNS[run]
        header, rows, blank_after = g.DATASETS[dataset][0](g.dataset_seed(dataset))
        self.run, self.change, self.dataset = run, "base", dataset
        self.header = [str(h) for h in header]
        self.rows = [[str(c) for c in row] for row in rows]
        self.blank_after = set(blank_after)
        self.target = g.DATASETS[dataset][1]
        self.time, self.group, self.gap, self.duplicates = parse_statements(extra)
        info = RUNSPEC[run]
        self.info = info
        self.stop = info.get("stop")            # the named error the run must end in
        self.stop_ok = None                     # a stop that is as good as the conclusion
        self.flags = list(info.get("flags", []))
        self.mentions = []                      # texts the message of the stop must hold
        self.claims = list(info.get("claims", []))
        self.extra = []                         # claims the change adds
        self.alias = {}                         # a column's name in the gate -> its name here
        self.same = False                       # nothing may move against the unperturbed run
        self.kind, self.must, self.decided = "", "", None

    def name(self, original):
        return self.alias.get(original, original)

    def col(self, original):
        return self.header.index(self.name(original))

    def rename(self, original, new):
        i = self.col(original)
        self.header[i] = new
        self.alias[original] = new

    def add_column(self, name, values):
        self.header.append(name)
        for row, value in zip(self.rows, values):
            row.append(str(value))

    def text(self):
        lines = [",".join(g._quote(h) for h in self.header)]
        for i, row in enumerate(self.rows):
            lines.append(",".join(g._quote(cell) for cell in row))
            if i in self.blank_after:
                lines.append("")
        return "\n".join(lines) + "\n"

    def args(self):
        words = []
        if isinstance(self.time, str):
            words += ["--time-col", self.time]
        elif self.time is False:
            words += ["--no-time-order"]
        if self.gap:
            words += ["--gap", str(self.gap)]
        if isinstance(self.group, str):
            words += ["--group-col", self.group]
        elif self.group is False:
            words += ["--no-entities"]
        if self.duplicates:
            words += ["--duplicates", self.duplicates]
        return words

    # what the table holds, for the changes
    def target_cells(self):
        t = self.header.index(self.target)
        return [row[t] for row in self.rows]

    def classes(self):
        """The two classes in sorted order, or None when the target is not two-valued."""
        if self.info.get("malformed"):
            return None
        values = sorted({c.strip() for c in self.target_cells() if not g.is_missing(c)})
        return values if len(values) == 2 else None

    def regression(self):
        if self.info.get("malformed") or self.classes():
            return False
        values = {c.strip() for c in self.target_cells() if not g.is_missing(c)}
        return len(values) > 20

    def pick(self, rng):
        """A row that has a target (any row of a malformed table)."""
        if self.info.get("malformed"):
            return rng.randrange(len(self.rows))
        t = self.header.index(self.target)
        return rng.choice([i for i, row in enumerate(self.rows) if not g.is_missing(row[t])])


# ---------------------------------------------------------- the changes --
# Each takes a fresh copy of a run's case and a generator seeded for the pair.
# It returns a text when the pair is left out (the reason), and otherwise
# changes the case and says what must hold.

def _stops(case):
    return case.stop is not None


def _same(case, what):
    """The change cannot move a number: the same rows, split and features reach the models."""
    if _stops(case):
        case.kind, case.must = "stop", f"the same stop, `{case.stop}`: {what}"
    else:
        case.same = True
        case.kind = "same"
        case.must = ("headline, row counts and predictions equal to the unperturbed run, and "
                     f"the run's own conclusion: {what}")


def _holds(case, what):
    """The table is another draw of the same kind: the conclusion holds, the numbers move."""
    if _stops(case):
        case.kind, case.must = "stop", f"the same stop, `{case.stop}`: {what}"
    else:
        case.kind = "band" if any(c.startswith("band:") for c in case.claims) else "structure"
        case.must = f"the run's own conclusion (its band, its roles, its counts): {what}"


def _added(case, name, role, reason, rule=None, warned=None):
    if not _stops(case):
        case.extra.append(("added", name, role, reason, rule, warned))


def ch_constant_added(case, rng):
    case.add_column("const_added", ["GB"] * len(case.rows))
    _added(case, "const_added", "excluded", "constant")
    _same(case, "a column of one value is excluded as `constant` at step 5 and reaches nothing")


def ch_id_added_int(case, rng):
    case.add_column("row_id_added", [100001 + i for i in range(len(case.rows))])
    _added(case, "row_id_added", "excluded", "id_like")
    _same(case, "consecutive whole numbers in file order are `id_like` (rule 8b) and reach "
                "nothing; an excluded column plays no part in the copies rule (step 8a)")


def ch_id_added_text(case, rng):
    case.add_column("ref_added", [f"R{i + 1:06d}" for i in range(len(case.rows))])
    _added(case, "ref_added", "excluded", "id_like")
    _same(case, "text distinct in every row is `id_like` (rule 6a) and reaches nothing")


def ch_second_date_added(case, rng):
    start = datetime.date(2019, 1, 1)
    case.add_column("added_date", [(start + datetime.timedelta(days=rng.randrange(1500)))
                                   .isoformat() for _ in case.rows])
    warned = "date_column_undeclared" if case.time is False else "no:date_column_undeclared"
    _added(case, "added_date", "excluded", "datetime", None, warned)
    _same(case, "a text column of ISO dates is excluded as `datetime` (rule 4); under "
                "--no-time-order one `date_column_undeclared` warning names it, under a time "
                "column none does")
    if case.stop == "time_order_undeclared":
        case.mentions.append("'added_date'")        # spec step 5: every date column, as repr
    if case.stop == "holdout_copies_in_training":
        case.mentions += ["added_date", "time_col"]  # spec 11


def ch_noise_added(case, rng):
    case.add_column("noise_added", [g._f(rng.gauss(0, 1)) for _ in case.rows])
    _added(case, "noise_added", "feature", None)
    _holds(case, "one more column of noise is a feature like any other and predicts nothing")
    if case.run == "dup_noise":
        case.stop_ok, case.stop = case.stop, None
        case.claims = ["band:noise_clf"]
        case.kind = "stop-or-band"
        case.must = ("the stop `holdout_copies_in_training`, or no skill (the band of "
                     "`noise_clf`, borrowed): every row is still present twice and the target "
                     "is noise. By spec step 8a the rows are no longer copies and the run goes "
                     "ahead")
        case.decided = "D60: a copy that differs in a feature is a stated limit (pass 8)"


def ch_copy_two_valued(case, rng):
    classes = case.classes()
    if not classes:
        return "the target is not two-valued, so it has no two-valued copy"
    names = {classes[0]: "closed", classes[1]: "open"}
    case.add_column("status_copy", ["" if g.is_missing(c) else names[c.strip()]
                                    for c in case.target_cells()])
    _added(case, "status_copy", "excluded", "suspected_leak", "predicts_target",
           "suspected_leak")
    _same(case, "a relabelled copy of the target is excluded by the leak screen (step 8, "
                "rule 2), after the split, so nothing else moves")
    if case.duplicates == "group" and not _stops(case):
        case.same, case.kind = False, "stop-or-structure"
        case.stop_ok = "holdout_copies_in_training"
        case.mentions = ["status_copy"]
        case.must = ("the run stops and names `status_copy`, or it finishes with no "
                     "combination of the features that reached the models on both sides of "
                     "the split and skill in the band of `few_valued`")
        case.decided = "CP8 finding 2 (D69 point 2)"


def ch_copy_distinct(case, rng):
    if case.info.get("malformed"):
        cells = [str(i) for i in range(len(case.rows))]
    elif case.classes():
        code = {c: i for i, c in enumerate(case.classes())}
        cells = ["" if g.is_missing(c) else f"{code[c.strip()] + i * 1e-6:.6f}"
                 for i, c in enumerate(case.target_cells())]
    else:
        cells = ["" if g.is_missing(c) else g._f(3.7 * float(c) + 12 + i * 1e-9, 9)
                 for i, c in enumerate(case.target_cells())]
    case.add_column("target_copy", cells)
    _added(case, "target_copy", "excluded", "suspected_leak", "predicts_target",
           "suspected_leak")
    _same(case, "a copy of the target that is distinct in every row (the class plus a "
                "millionth of the row number; for regression the target rescaled) is excluded "
                "by the leak screen, after the split, so nothing else moves")
    if case.duplicates == "group" and not _stops(case):
        case.same, case.kind = False, "stop"
        case.stop, case.claims, case.extra = "holdout_copies_in_training", [], []
        case.mentions = ["target_copy"]
        case.must = ("the stop `holdout_copies_in_training`, naming `target_copy`: with it no "
                     "two rows agree at step 5, the screen then excludes it, and the rows "
                     "that now agree are copies on both sides (spec step 8a, the pinned case)")


def ch_rows_shuffled(case, rng):
    rng.shuffle(case.rows)
    _holds(case, "the order of the rows in the file carries nothing in this table (under "
                 "time order the rows are put in time order by the tool)")


def ch_row_repeated(case, rng):
    case.rows.append(list(case.rows[case.pick(rng)]))
    _holds(case, "one row written twice is one more row; `input.duplicate_rows` counts it "
                 "and a `duplicate_rows` warning says so")
    if not _stops(case):
        case.extra.append(("duplicate_rows",))
    if isinstance(case.time, str) and isinstance(case.group, str) and not _stops(case):
        case.stop, case.claims, case.extra, case.kind = "group_column_invalid", [], [], "stop"
        case.mentions = [case.group]
        case.must = ("the stop `group_column_invalid`, naming the entity column: two rows of "
                     "one entity at one timestamp, in a regression under time order (spec "
                     "step 11)")


def ch_blank_line_added(case, rng):
    spare = [i for i in range(len(case.rows) - 1) if i not in case.blank_after]
    case.blank_after.add(rng.choice(spare))
    _same(case, "a blank line is not a row (spec section 6), so every row keeps its position")


def _feature(case):
    return case.info.get("num") or case.info.get("cat")


def ch_missing_cell_feature(case, rng):
    name = _feature(case)
    case.rows[case.pick(rng)][case.col(name)] = ""
    _holds(case, f"one empty cell in the feature `{name}` is counted in `n_missing` and "
                 "imputed")
    if not _stops(case):
        case.extra.append(("n_missing", name))


def ch_missing_cell_entity(case, rng):
    if not isinstance(case.group, str):
        return "the run declares no entity column"
    case.rows[case.pick(rng)][case.col(case.group)] = ""
    if not _stops(case):
        case.stop, case.claims, case.extra = "group_column_invalid", [], []
        case.mentions = [case.group]
    case.kind = "stop"
    case.must = (f"the stop `{case.stop}`: an empty entity cell in a row that has a target "
                 "(spec step 2); the message names the entity column")


def ch_missing_cell_date(case, rng):
    if isinstance(case.time, str):
        case.rows[case.pick(rng)][case.col(case.time)] = ""
        _holds(case, "a row with no time value is dropped and counted in "
                     "`dropped_missing_time` (step 2)")
        if case.run == "int_stamps":
            case.decided = ("D55 point 3: a CSV whole-number column with an empty cell loads "
                            "as floats, already rounded (a stated limit)")
        return None
    if not case.info.get("date"):
        return "the table has no date or time column"
    case.rows[case.pick(rng)][case.col(case.info["date"])] = ""
    _same(case, "an empty cell is not looked at by the date rules (step 5), and the column "
                "reaches nothing either way")
    return None


def ch_missing_cell_target(case, rng):
    if case.info.get("malformed"):
        return "malformed by design: the header's target name is over another field"
    case.rows[case.pick(rng)][case.header.index(case.target)] = ""
    _holds(case, "a row with no target is dropped and counted in `dropped_missing_target`")


def ch_feature_renamed(case, rng):
    case.rename(_feature(case), "renamed_feature")
    _same(case, "a column's name carries nothing")


def ch_feature_renamed_awkward(case, rng):
    case.rename(_feature(case), "net value, £ (2024)")
    _same(case, "a column's name carries nothing, a name with a comma, a space and a "
                "non-ASCII letter included")


def ch_target_renamed(case, rng):
    case.rename(case.target, "the outcome")
    case.target = "the outcome"
    _same(case, "the target's name carries nothing")


def ch_role_renamed(case, rng):
    done = []
    if isinstance(case.time, str):
        case.rename(case.time, "when")
        case.time = "when"
        done.append("time")
    elif case.info.get("date"):
        case.rename(case.info["date"], "when")
        done.append("date")
    if isinstance(case.group, str):
        case.rename(case.group, "who")
        case.group = "who"
        done.append("entity")
    if not done:
        return "the table has no time, date or entity column"
    _same(case, f"the name of the {' and the '.join(done)} column carries nothing")
    return None


def ch_target_first(case, rng):
    t = case.header.index(case.target)
    case.header.insert(0, case.header.pop(t))
    for row in case.rows:
        row.insert(0, row.pop(t))
    _same(case, "where the target stands among the columns carries nothing: the features "
                "keep their order")


def ch_target_respelt(case, rng, flipped=False):
    classes = case.classes()
    if not classes:
        return "the target is not two-valued"
    numeric = all(re.fullmatch(r"-?\d+", c) for c in classes)
    new = (["2", "1"] if numeric else ["zeta", "alpha"]) if flipped else \
        (["1", "2"] if numeric else ["alpha", "beta"])
    names = dict(zip(classes, new))
    t = case.header.index(case.target)
    for row in case.rows:
        if not g.is_missing(row[t]):
            row[t] = names[row[t].strip()]
    if flipped:
        _holds(case, "the classes respelt so that the other one sorts second: the positive "
                     "class changes and skill, 2 x AUC - 1, does not depend on which it is")
        if not _stops(case):
            case.extra.append(("classes", sorted(new), max(new)))
    else:
        _same(case, "the classes respelt in the same sorted order: the same 0 and 1 reach "
                    "every step")
    return None


def ch_target_respelt_flipped(case, rng):
    return ch_target_respelt(case, rng, flipped=True)


def ch_white_space_cells(case, rng):
    where = []
    if case.classes() and not all(re.fullmatch(r"-?\d+", c) for c in case.classes()):
        where.append((case.header.index(case.target), "{} "))
    if case.info.get("cat") and not case.info.get("malformed"):
        where.append((case.col(case.info["cat"]), " {} "))
    if isinstance(case.group, str):
        where.append((case.col(case.group), "{} "))
    if not where:
        return "no text target, categorical feature or entity column to pad"
    for i, pad in where:
        for k in rng.sample(range(len(case.rows)), max(5, len(case.rows) // 100)):
            if not g.is_missing(case.rows[k][i]):
                case.rows[k][i] = pad.format(case.rows[k][i])
    _same(case, "white space round a target value, a categorical level or an entity value "
                "is not part of it (D42, D46, step 5)")
    return None


FLOAT_WHOLE = {"day_numbers": ["day"], "messy": ["record_id", "visits"],
               "binary_numeric": ["outcome"], "ambiguous_target": ["rating"]}


def ch_float_whole_numbers(case, rng):
    names = FLOAT_WHOLE.get(case.run)
    if not names:
        return "no whole-number column whose reading the spec pins by value"
    for name in names:
        i = case.col(name)
        for row in case.rows:
            if not g.is_missing(row[i]):
                row[i] = f"{int(row[i])}.0"
    _same(case, f"whole numbers written as floats ({', '.join(names)}) are judged by value: "
                "a date number, an identifier, a two-valued target (steps 4 and 5)")
    return None


def _sentinel_ymd(value):
    def change(case, rng):
        name = case.info.get("ymd")
        if not name:
            return "no column of dates written YYYYMMDD"
        case.rows[case.pick(rng)][case.col(name)] = value
        _holds(case, f"one cell of the date column `{name}` set to {value}: the column is "
                     "still a column of dates (D69 point 1)")
        case.decided = "CP8 finding 1 (D69 point 1)"
        return None
    return change


def ch_ymd_sentinel_in_dropped_row(case, rng):
    name = case.info.get("ymd")
    if not name:
        return "no column of dates written YYYYMMDD"
    k = case.pick(rng)
    case.rows[k][case.col(name)] = "99991231"
    case.rows[k][case.header.index(case.target)] = ""
    _holds(case, "a value that is no date in a row with no target does not count: rule 2a "
                 "looks at the usable rows (spec step 5)")
    return None


def _sentinel_iso(value, in_range):
    def change(case, rng):
        declared = isinstance(case.time, str) and case.run != "int_stamps"
        name = case.time if declared else case.info.get("date")
        if not name or case.info.get("ymd") or (not declared and case.run == "int_stamps"):
            return "no column of ISO dates"
        case.rows[case.pick(rng)][case.col(name)] = value
        if not declared:
            _same(case, f"one cell of the undeclared date column set to {value!r}: at least "
                        "95% of its values are still in a date form (rule 4), and it reaches "
                        "nothing either way")
        elif in_range:
            _holds(case, f"one time value set to {value}: a valid date, so that row is the "
                         "earliest and the split, counted by the judge, is by time as before")
        elif case.stop != "time_column_invalid":
            case.stop, case.claims, case.extra, case.flags = "time_column_invalid", [], [], []
            case.kind = "stop"
            case.must = (f"the stop `time_column_invalid`: {value!r} in the time column is "
                         + ("a date no nanosecond timestamp can hold" if value[:1].isdigit()
                            and "-" in value else "text in no accepted form")
                         + " (spec step 3); never ordered as it is, never dropped in silence")
        else:
            _holds(case, "the time column was in no accepted form already")
        return None
    return change


def _sentinel_int_time(value):
    def change(case, rng):
        if case.run != "int_stamps":
            return "no whole-number time column"
        case.rows[case.pick(rng)][case.col(case.time)] = value
        _holds(case, f"one time value set to {value}: a numeric time column is ordered as "
                     "numbers whatever it holds, so that row is the earliest")
        return None
    return change


def _sentinel_number(value):
    def change(case, rng):
        name = case.info.get("num")
        if not name:
            return "no numeric feature"
        case.rows[case.pick(rng)][case.col(name)] = value
        _holds(case, f"one cell of the numeric feature `{name}` set to {value}")
        if float(value) > TOO_LARGE and not _stops(case):
            case.extra.append(("too_large", name))
            case.must += ("; a value above 3.4e38 is read as missing, counted in `n_infinite` "
                          "and named by `infinite_values` (D48)")
        return None
    return change


def ch_target_sentinel_neg(case, rng):
    if not case.regression():
        return "the target is not a regression target"
    case.rows[case.pick(rng)][case.header.index(case.target)] = "-999"
    _holds(case, "one target cell set to -999 (the spec has no rule for an outlier in the "
                 "target; docs/BACKLOG.md, 'One extreme value can dominate')")
    return None


def ch_target_sentinel_max(case, rng):
    if not case.regression():
        return "the target is not a regression target"
    case.rows[case.pick(rng)][case.header.index(case.target)] = "1.7976931348623157e308"
    if not _stops(case):
        case.stop, case.claims, case.extra = "non_finite_score", [], []
    case.kind = "stop"
    case.must = (f"the stop `{case.stop}`"
                 + (": one target cell of 1.7976931348623157e308 makes a baseline's squared "
                    "error infinite, before any fit (D44, rule 1)"
                    if case.stop == "non_finite_score" else
                    ": the statement is asked for at step 5, before the baselines"))
    return None


def _duplicates(value):
    def change(case, rng):
        if case.run not in ("dup_noise", "few_valued", "time_blocks_undeclared"):
            return "the run is not about repeated rows"
        if (case.duplicates or "error") == value:
            return "this is the run as the gate makes it"
        case.duplicates, case.stop, case.same = value, None, False
        date = [case.info["date"], "time_col"] if case.info.get("date") else []
        if value == "error":
            case.stop, case.claims, case.kind = "holdout_copies_in_training", [], "stop"
            case.mentions = date
            case.must = ("the stop `holdout_copies_in_training`: every holdout row has a row "
                         "like it among the training rows (step 8a)"
                         + ("; the message names the date column and `time_col`" if date else ""))
        elif value == "keep":
            case.claims = {"dup_noise": [], "few_valued": ["band:few_valued"]}[case.run]
            case.kind = "structure" if case.run == "dup_noise" else "band"
            case.must = ("the run goes ahead with one `holdout_copies_kept` warning and the "
                         "count the judge makes for itself"
                         + ("; the target is noise in sixteen combinations, so no skill"
                            if case.run == "few_valued" else
                            " (no band: with `keep` the caller has taken the inflation)"))
        else:
            borrowed = {"dup_noise": "noise_clf", "time_blocks_undeclared": "time_blocks"}
            case.claims = [f"band:{borrowed[case.run]}", "combos_apart", "group_units"] \
                + (["undeclared_date:date"] if case.run == "time_blocks_undeclared" else [])
            case.kind = "band"
            case.must = ("the rows that agree in every feature are kept together: no "
                         "combination on both sides, the counts the judge makes, and no skill "
                         f"(the band of `{borrowed[case.run]}`, borrowed: the target is noise "
                         "for each set)")
        return None
    return change


CHANGES = {
    "ymd_sentinel_99991231": _sentinel_ymd("99991231"),
    "ymd_sentinel_0": _sentinel_ymd("0"),
    "ymd_sentinel_20240230": _sentinel_ymd("20240230"),
    "ymd_sentinel_in_dropped_row": ch_ymd_sentinel_in_dropped_row,
    "date_sentinel_9999": _sentinel_iso("9999-12-31", False),
    "date_sentinel_1900": _sentinel_iso("1900-01-01", True),
    "date_sentinel_zero": _sentinel_iso("0", False),
    "int_time_sentinel_0": _sentinel_int_time("0"),
    "int_time_sentinel_99991231": _sentinel_int_time("99991231"),
    "number_sentinel_-999": _sentinel_number("-999"),
    "number_sentinel_999999": _sentinel_number("999999"),
    "number_sentinel_1e308": _sentinel_number("1e308"),
    "target_sentinel_-999": ch_target_sentinel_neg,
    "target_sentinel_max_double": ch_target_sentinel_max,
    "copy_two_valued": ch_copy_two_valued,
    "copy_distinct": ch_copy_distinct,
    "constant_added": ch_constant_added,
    "id_added_int": ch_id_added_int,
    "id_added_text": ch_id_added_text,
    "second_date_added": ch_second_date_added,
    "noise_added": ch_noise_added,
    "rows_shuffled": ch_rows_shuffled,
    "row_repeated": ch_row_repeated,
    "blank_line_added": ch_blank_line_added,
    "missing_cell_feature": ch_missing_cell_feature,
    "missing_cell_entity": ch_missing_cell_entity,
    "missing_cell_date": ch_missing_cell_date,
    "missing_cell_target": ch_missing_cell_target,
    "feature_renamed": ch_feature_renamed,
    "feature_renamed_awkward": ch_feature_renamed_awkward,
    "target_renamed": ch_target_renamed,
    "role_column_renamed": ch_role_renamed,
    "target_first": ch_target_first,
    "target_respelt": ch_target_respelt,
    "target_respelt_flipped": ch_target_respelt_flipped,
    "white_space_cells": ch_white_space_cells,
    "float_whole_numbers": ch_float_whole_numbers,
    "duplicates_error": _duplicates("error"),
    "duplicates_keep": _duplicates("keep"),
    "duplicates_group": _duplicates("group"),
}


def build_pairs(only=None):
    """Every pair in scope, and every pair left out with its reason."""
    cases, left_out = [], []
    for run in RUNSPEC:
        base = Case(run)
        base.kind = "stop" if base.stop else "base"
        base.must = (f"the stop `{base.stop}`" if base.stop else "the gate's own conclusion")
        cases.append(base)
        for change, apply in CHANGES.items():
            case = Case(run)
            case.change = change
            reason = apply(case, random.Random(f"battery:{run}:{change}"))
            if reason:
                left_out.append((run, change, reason))
            else:
                cases.append(case)
    if only:
        wanted = [w.strip() for w in only.split(",") if w.strip()]

        def chosen(case):
            return any(w == case.run or w == f"{case.run}:{case.change}"
                       or w == f":{case.change}" for w in wanted)
        runs = {c.run for c in cases if chosen(c)}
        cases = [c for c in cases if chosen(c) or (c.change == "base" and c.run in runs)]
    return cases, left_out


# ------------------------------------------------------------ tool runs --

def launch(case, package, work):
    label = f"{case.run}__{case.change}"
    data = DATA / f"{label}.csv"
    out = work / "out" / label
    data.parent.mkdir(parents=True, exist_ok=True)
    if not data.is_file():
        data.write_bytes(case.text().encode("utf-8"))
    cmd = [sys.executable, "-B", "-m", "mlprofessor", "profile", "--data", str(data),
           "--target", case.target, "--out", str(out), "--profile", "fast", "--seed", "0",
           "--n-jobs", "1"] + case.args()
    result = {"label": label, "data": data, "out": out, "attempts": 0}
    for _ in range(2):                      # one more go for a run that did not finish
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        started = time.perf_counter()
        result["attempts"] += 1
        try:
            proc = subprocess.run(cmd, cwd=str(package.parent), env=g._tool_env(1),
                                  capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=RUN_SECONDS)
            rc, stdout, stderr = proc.returncode, proc.stdout or "", proc.stderr or ""
        except subprocess.TimeoutExpired:
            rc, stdout, stderr = None, "", f"did not finish within {RUN_SECONDS} seconds"
        result.update(rc=rc, stdout=stdout, stderr=stderr,
                      seconds=round(time.perf_counter() - started, 2))
        if rc is not None:
            break
    return result


def read_outputs(result):
    """Parse what a run wrote. `finished` is whether it exited 0 with both files."""
    profile, predictions = result["out"] / "profile.json", result["out"] / "predictions.csv"
    result["header"], result["source"] = g.read_source(result["data"])
    found = re.search(r"error: (\w+):", result["stderr"] or "")
    result["code"] = found.group(1) if found else None
    result["finished"] = result["rc"] == 0 and profile.is_file() and predictions.is_file()
    if not result["finished"]:
        return result
    bad = []
    result["doc"] = json.loads(profile.read_text(encoding="utf-8"),
                               parse_constant=lambda token: bad.append(token) or token)
    result["bad_numbers"] = bad
    _, lines = g.read_source(predictions)
    result["predictions"] = [(int(p), truth, float(guess)) for p, truth, guess in lines]
    return result


# ------------------------------------------------------------ the judge --

class Fail(Exception):
    pass


def norm_number(cell):
    try:
        value = float(cell.strip())
    except ValueError:
        return None
    return None if value != value or abs(value) > TOO_LARGE else value + 0.0


def norm_level(cell):
    return "(missing)" if g.is_missing(cell) else cell.strip()


def norm_target(cell):
    value = norm_number(cell)
    return cell.strip() if value is None else value


def time_key(cell):
    cell = cell.strip()
    if re.fullmatch(r"-?\d+", cell):
        return int(cell)
    return datetime.datetime.fromisoformat(cell)


class Judge:
    def __init__(self, case, result, base):
        self.c, self.r, self.base, self.fails = case, result, base, []

    # ---- reading
    def at(self, path, doc=None):
        node = self.r["doc"] if doc is None else doc
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise Fail(f"profile.json has no key `{path}`")
            node = node[part]
        return node

    def number(self, path):
        value = self.at(path)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            raise Fail(f"`{path}` is {value!r}, not a finite number")
        return value

    def warned(self, code):
        return [w for w in self.at("warnings") if w.get("code") == code]

    def columns(self):
        return {c["name"]: c for c in self.at("columns")}

    def counts(self):
        return {r["item"]: r["count"] for r in self.at("rows")}

    def require(self, condition, message):
        if not condition:
            raise Fail(message)

    # ---- the source, as the judge counts it
    def usable(self):
        header, source, c = self.r["header"], self.r["source"], self.c
        t = header.index(c.target)
        rows = [i for i, row in enumerate(source) if not g.is_missing(row[t])]
        if isinstance(c.time, str):
            at = header.index(c.time)
            rows = [i for i in rows if not g.is_missing(source[i][at])]
        return rows

    def held(self):
        return [p for p, _, _ in self.r["predictions"]]

    def capped(self):
        got = self.counts()
        return (got.get("dropped_row_cap") or 0) + (got.get("dropped_gap") or 0)

    def split_rows(self):
        """(training rows, holdout rows) when every usable row is on one side; else None."""
        if self.capped():
            return None
        return self.sides()

    def sides(self):
        """(usable rows not held out, holdout rows). Under a row cap the first holds the
        rows the cap left out as well: the output does not say which they are."""
        held = set(self.held())
        return [i for i in self.usable() if i not in held], sorted(held)

    def feature_columns(self, with_leaks=False):
        header = self.r["header"]
        return [(header.index(c["name"]), c["kind"]) for c in self.at("columns")
                if c["role"] == "feature" or (with_leaks and c["reason"] == "suspected_leak")]

    def combination(self, row, features):
        cells = self.r["source"][row]
        return tuple(norm_number(cells[i]) if kind == "numeric" else norm_level(cells[i])
                     for i, kind in features)

    def time_expectation(self):
        header, source, c = self.r["header"], self.r["source"], self.c
        at = header.index(c.time)
        stamps = {i: time_key(source[i][at]) for i in self.usable()}
        return stamps, g.expected_time_split(stamps, gap=c.gap)

    # ---- the verdict
    def run(self):
        c, r = self.c, self.r
        if r["rc"] is None:
            self.fails.append(("finish", r["stderr"]))
            return self.fails
        if c.stop:
            self.claim("stop", lambda: self.stopped(c.stop))
            return self.fails
        if c.stop_ok and not r["finished"] and r["code"] == c.stop_ok:
            self.claim("stop", lambda: self.stopped(c.stop_ok))
            return self.fails
        if not r["finished"]:
            self.fails.append(("finish", f"exit {r['rc']}, `{r['code']}`: "
                                         + g._tail(r["stderr"], 300)))
            return self.fails
        for name in ("u_rows", "u_columns", "u_headline", "u_statements", "u_random_counts",
                     "u_copies", "u_finite"):
            self.claim(name, getattr(self, name))
        for name in c.claims:
            head, _, arg = name.partition(":")
            self.claim(name, (lambda h=head, a=arg: getattr(self, "c_" + h)(a)) if arg
                       else getattr(self, "c_" + head))
        for extra in c.extra:
            self.claim(":".join(str(e) for e in extra[:2]),
                       lambda e=extra: getattr(self, "x_" + e[0])(*e[1:]))
        if c.same:
            self.claim("same_as_base", self.same_as_base)
        return self.fails

    def claim(self, name, check):
        try:
            check()
        except (Fail, g.CheckFailed) as exc:
            self.fails.append((name, str(exc)))
        except Exception as exc:  # noqa: BLE001  a judge that cannot read the output
            self.fails.append((name, f"JUDGE ERROR {type(exc).__name__}: {exc}"))

    def stopped(self, code):
        r, c = self.r, self.c
        self.require(r["rc"] == 3, f"exit code {r['rc']} (a named error is exit 3); stderr: "
                                   + g._tail(r["stderr"], 300))
        self.require(r["code"] == code, f"the error is `{r['code']}`, expected `{code}`: "
                                        + g._tail(r["stderr"], 300))
        self.require(not (r["out"] / "profile.json").exists(), "profile.json was written")
        missing = [text for text in c.flags + c.mentions if text not in r["stderr"]]
        self.require(not missing, f"the message does not hold {missing}: "
                                  + g._tail(r["stderr"], 400))

    # ---- universal: the reconciliation rules of spec 5.3, on any finished run
    def u_rows(self):
        header, source, c = self.r["header"], self.r["source"], self.c
        t, got = header.index(c.target), self.counts()
        self.require(got.get("loaded") == len(source),
                     f"rows.loaded is {got.get('loaded')}; the source has {len(source)} rows")
        dropped = sum(v for k, v in got.items() if k.startswith("dropped_"))
        self.require(got["loaded"] == dropped + got["training"] + got["holdout"],
                     f"the row counts do not add up: {got}")
        no_target = sum(g.is_missing(row[t]) for row in source)
        self.require(got["dropped_missing_target"] == no_target,
                     f"rows.dropped_missing_target is {got['dropped_missing_target']}; the "
                     f"judge counts {no_target}")
        held = self.held()
        self.require(len(set(held)) == len(held) == got["holdout"],
                     f"predictions.csv has {len(held)} rows, {len(set(held))} distinct; "
                     f"rows.holdout is {got['holdout']}")
        for position, truth, _ in self.r["predictions"]:
            self.require(0 <= position < len(source), f"predictions.csv names row {position}")
            self.require(g.same_cell(truth, source[position][t]),
                         f"y_true at row {position} is {truth!r}; the source has "
                         f"{source[position][t]!r}")

    def u_columns(self):
        names = [col["name"] for col in self.at("columns")]
        self.require(names == self.r["header"],
                     f"the column accounting lists {names[:6]}...; the source has "
                     f"{self.r['header'][:6]}...")

    def u_headline(self):
        score, kind = self.number("headline.score"), self.at("task.type")
        truths = [truth for _, truth, _ in self.r["predictions"]]
        guesses = [guess for _, _, guess in self.r["predictions"]]
        if kind == "classification":
            positive = str(self.at("task.positive_class"))
            if isinstance(self.at("task.positive_class"), bool):
                positive = positive.lower()
            mine = g.auc([1 if g.same_cell(t, positive) else 0 for t in truths], guesses)
            self.require(self.at("headline.metric") == "roc_auc", "the metric is not roc_auc")
            self.require(g.close(score, mine), f"headline.score {score!r}; ROC AUC of "
                                               f"predictions.csv is {mine!r}")
            self.require(g.close(self.number("headline.skill"), 2 * mine - 1),
                         "headline.skill is not 2 x AUC - 1")
            return
        mine = g.mse([float(t) for t in truths], guesses)
        self.require(self.at("headline.metric") == "mse", "the metric is not mse")
        self.require(g.close(score, mine), f"headline.score {score!r}; the squared error of "
                                           f"predictions.csv is {mine!r}")
        base, skill = self.at("headline.baseline_score"), self.at("headline.skill")
        if base == 0:
            self.require(skill is None, f"skill is {skill!r} with a baseline error of 0")
        else:
            self.require(skill is not None and g.close(skill, 1 - mine / base, 1e-7),
                         f"headline.skill {skill!r} is not 1 - {mine!r} / {base!r}")
        sides = self.split_rows()
        if sides and self.at("headline.baseline") == "training_mean":
            t = self.r["header"].index(self.c.target)
            training = [float(self.r["source"][i][t]) for i in sides[0]]
            want = g.mse([float(self.r["source"][i][t]) for i in sides[1]],
                         [g.mean(training)] * len(sides[1]))
            self.require(g.close(base, want, 1e-7), f"the training-mean baseline scores "
                                                    f"{base!r}; the judge computes {want!r}")

    def expected_scheme(self):
        c = self.c
        if isinstance(c.time, str):
            return "time_ordered"
        if isinstance(c.group, str):
            return "grouped"
        if c.duplicates == "group":
            return "grouped" if self.rows_alike() else "random"
        return "random"

    def rows_alike(self):
        """Rows, among those split, that agree with another in every feature of step 5."""
        features = self.feature_columns(with_leaks=True)
        count = {}
        for i in self.usable():
            key = self.combination(i, features)
            count[key] = count.get(key, 0) + 1
        return sum(n for n in count.values() if n > 1)

    def u_statements(self):
        c = self.c
        self.require(self.at("input.time_col") == c.time,
                     f"input.time_col is {self.at('input.time_col')!r}; the call said {c.time!r}")
        self.require(self.at("input.group_col") == c.group,
                     f"input.group_col is {self.at('input.group_col')!r}; the call said "
                     f"{c.group!r}")
        self.require(self.at("input.duplicates") == (c.duplicates or "error"),
                     f"input.duplicates is {self.at('input.duplicates')!r}")
        want = self.expected_scheme()
        self.require(self.at("validation.scheme") == want,
                     f"validation.scheme is {self.at('validation.scheme')!r}; expected {want!r}")

    def u_random_counts(self):
        if self.expected_scheme() != "random":
            return
        got, usable = self.counts(), len(self.usable())
        capped = max(0, usable - g.FAST_CAP)
        holdout = -(-(usable - capped) // 4)
        want = {"dropped_row_cap": capped, "holdout": holdout,
                "training": usable - capped - holdout, "dropped_gap": 0,
                "dropped_missing_time": 0}
        wrong = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
        self.require(not wrong, f"row counts (got, expected): {wrong}")

    def copies(self):
        """Holdout rows with a copy among the training rows, by the judge's own count."""
        sides = self.split_rows()
        if sides is None:
            return None
        features, t = self.feature_columns(), self.r["header"].index(self.c.target)

        def key(i):
            return self.combination(i, features) + (norm_target(self.r["source"][i][t]),)
        seen = {key(i) for i in sides[0]}
        return sum(key(i) in seen for i in sides[1]), len(sides[1])

    def u_copies(self):
        c = self.c
        if isinstance(c.time, str):
            self.require(self.at("validation.n_holdout_copies") is None,
                         "validation.n_holdout_copies is not null under time order")
            return
        found = self.copies()
        if found is None:
            return                          # the cap left rows out: the judge cannot say which
        n, held = found
        self.require(self.at("validation.n_holdout_copies") == n,
                     f"validation.n_holdout_copies is {self.at('validation.n_holdout_copies')!r}"
                     f"; the judge counts {n} of {held}")
        met = n >= 10 and n / held > 0.01
        kept = len(self.warned("holdout_copies_kept"))
        if (c.duplicates or "error") == "keep":
            self.require(kept == (1 if met else 0),
                         f"{kept} `holdout_copies_kept` warning(s) with {n} of {held} holdout "
                         "rows having a copy")
        else:
            self.require(not met and kept == 0,
                         f"the run finished with {n} of {held} holdout rows having a copy "
                         f"among the training rows, under duplicates={c.duplicates or 'error'}")

    def u_finite(self):
        self.require(not self.r["bad_numbers"], f"profile.json holds {self.r['bad_numbers']}")
        if self.at("headline.skill") is None:
            self.require(len(self.warned("skill_undefined")) == 1,
                         "skill is null without one `skill_undefined` warning")

    # ---- the conclusions of the runs
    def c_band(self, name):
        skill = self.at("headline.skill")
        low, high = g.BANDS[name]
        self.require(isinstance(skill, (int, float)) and low <= skill <= high,
                     f"skill is {skill if skill is None else round(skill, 4)}, expected "
                     f"between {low} and {high} (the band of `{name}`)")

    def c_p_floor(self):
        self.require(self.number("headline.p_value") >= g.P_VALUE_FLOOR,
                     f"the p-value is {self.at('headline.p_value')!r}, at its floor: noise is "
                     "reported as significant")

    def c_families_fast(self):
        families = sorted({m["family"] for m in self.at("models")})
        self.require(families == ["linear", "tree"], f"the models table lists {families}")

    def c_majority_margin(self):
        truths = [t.strip() for _, t, _ in self.r["predictions"]]
        majority = max(truths.count(v) for v in set(truths)) / len(truths)
        self.require(self.at("headline.baseline_score") == 0.5, "the baseline is not 0.5")
        self.require(abs(self.number("headline.score") - majority) > g.MAJORITY_MARGIN,
                     f"headline.score is within {g.MAJORITY_MARGIN} of the majority share")

    def role_is(self, original, role, reason, rule=MISSING, kind=None):
        name = self.c.name(original)
        got = self.columns().get(name, {})
        self.require((got.get("role"), got.get("reason")) == (role, reason),
                     f"`{name}` has role {got.get('role')!r}, reason {got.get('reason')!r}; "
                     f"expected {role!r}, {reason!r}")
        if rule is not MISSING:
            self.require(got.get("leak_rule") == rule,
                         f"`{name}` has leak_rule {got.get('leak_rule')!r}; expected {rule!r}")
        if kind:
            self.require(got.get("kind") == kind, f"`{name}` has kind {got.get('kind')!r}")

    def names_warned(self, code):
        return sorted(str(w["column"]) for w in self.warned(code))

    def c_leak_roles(self):
        self.role_is("outcome_copy", "excluded", "suspected_leak", "predicts_target")
        self.role_is("settlement_amount", "excluded", "suspected_leak", "missing_by_class")
        self.role_is("strong_feature", "feature", None)
        leaks = self.names_warned("suspected_leak")
        for name in ("outcome_copy", "settlement_amount"):
            self.require(self.c.name(name) in leaks, f"no `suspected_leak` warning names {name}")
        self.require(self.c.name("strong_feature") not in self.names_warned("strong_single_column"),
                     "`strong_feature` carries a `strong_single_column` warning")

    def c_leak_reg_roles(self):
        self.role_is("amount_scaled", "excluded", "suspected_leak", "predicts_target")
        self.require(self.c.name("amount_scaled") in self.names_warned("suspected_leak"),
                     "no `suspected_leak` warning names amount_scaled")

    def c_near_leak_roles(self):
        self.role_is("outcome_noisy", "feature", None)
        warned = self.names_warned("strong_single_column")
        self.require(warned == [self.c.name("outcome_noisy")],
                     f"`strong_single_column` warnings name {warned}")

    def c_task_binary(self):
        got = (self.at("task.type"), self.at("task.decided_by"), self.at("task.n_distinct"),
               self.at("headline.metric"))
        self.require(got == ("classification", "rule", 2, "roc_auc"), f"the task is {got}")

    def c_messy_roles(self):
        for original, (role, reason, kind) in g.MESSY_ROLES.items():
            self.role_is(original, role, reason, kind=kind)
        want = sorted(self.c.name(n) for n in ("fee", "revenue"))
        self.require(self.names_warned("numeric_with_separators") == want,
                     f"`numeric_with_separators` names {self.names_warned('numeric_with_separators')}")

    def c_messy_coercion(self):
        header, source, columns = self.r["header"], self.r["source"], self.columns()
        t = header.index(self.c.target)
        for original in ("spend", "n1", "f2", "seg"):
            name = self.c.name(original)
            cells = [(row[header.index(name)], row[t]) for row in source]
            missing = sum(g.is_missing(cell) for cell, _ in cells)
            coerced = sum(1 for cell, _ in cells if original == "spend"
                          and not g.is_missing(cell) and not g._parses(cell))
            imputed = sum(1 for cell, y in cells if not g.is_missing(y)
                          and (g.is_missing(cell) or (original == "spend"
                                                      and not g._parses(cell))))
            got = columns[name]
            for key, want in (("n_missing", missing), ("n_coerced", coerced),
                              ("n_imputed", imputed)):
                self.require(got[key] == want, f"`{name}` reports {key} = {got[key]!r}; the "
                                               f"judge counts {want}")
        self.require(self.c.name("spend") in self.names_warned("values_coerced"),
                     "no `values_coerced` warning names spend")

    def c_undeclared_date(self, original):
        name = self.c.name(original)
        self.role_is(original, "excluded", "datetime")
        self.require(name in self.names_warned("date_column_undeclared"),
                     f"no `date_column_undeclared` warning names `{name}`")
        self.require(self.at("input.time_col") is False and self.at("validation.caveat"),
                     "the statement or the caveat is not recorded")

    def c_day_role(self):
        self.role_is("day", "excluded", "datetime")

    def c_time_split(self):
        stamps, want = self.time_expectation()
        got, held = self.counts(), set(self.held())
        self.require(held == want["holdout"],
                     f"the holdout is {len(held)} rows; the judge expects the "
                     f"{len(want['holdout'])} latest ({len(held - want['holdout'])} unexpected, "
                     f"{len(want['holdout'] - held)} missing)")
        header, source, c = self.r["header"], self.r["source"], self.c
        t, at = header.index(c.target), header.index(c.time)
        no_time = sum(1 for row in source if not g.is_missing(row[t]) and g.is_missing(row[at]))
        for item, count in (("dropped_row_cap", want["capped"]), ("dropped_gap", len(want["gap"])),
                            ("training", len(want["training"])), ("dropped_missing_time", no_time)):
            self.require(got.get(item) == count,
                         f"rows.{item} is {got.get(item)!r}; the judge counts {count}")
        for key, expected in (("validation.train_end", want["train_end"]),
                              ("validation.holdout_start", want["holdout_start"])):
            value = self.at(key)
            if isinstance(expected, int):
                self.require(value == expected, f"`{key}` is {value!r}; expected {expected}")
            else:
                self.require(isinstance(value, str)
                             and datetime.datetime.fromisoformat(value) == expected,
                             f"`{key}` is {value!r}; expected {expected.isoformat()}")

    def c_gap_folds(self):
        self.require(self.at("validation.gap") == self.c.gap, "validation.gap is not the gap")
        folds = self.at("validation.folds")
        self.require(len(folds) == 5 and all(f["gap_rows"] >= self.c.gap for f in folds),
                     "a fold reports fewer rows removed by the gap than the gap")

    def c_blocks_interval(self):
        _, want = self.time_expectation()
        self.require(self.at("headline.interval_method") == "bootstrap_blocks",
                     f"interval_method is {self.at('headline.interval_method')!r}")
        self.require(len(self.warned("interval_approximate")) == 1,
                     "not exactly one `interval_approximate` warning")
        self.require(self.at("headline.block_length") == g.cube_root_up(want["holdout_stamps"]),
                     "block_length is not the cube root of the holdout timestamps, rounded up")
        self.require(self.at("headline.p_value") is None, "a p-value under time order")

    def series_before(self, by_entity=False):
        """order of the usable rows in time, and each row's previous row (of its entity)."""
        header, source, c = self.r["header"], self.r["source"], self.c
        at = header.index(c.time)
        rows = self.usable()
        stamps = {i: time_key(source[i][at]) for i in rows}
        groups = {}
        for i in rows:
            entity = source[i][header.index(c.group)].strip() if by_entity else None
            groups.setdefault(entity, []).append(i)
        before, unique = {}, True
        for members in groups.values():
            members.sort(key=lambda i: stamps[i])
            unique = unique and len({stamps[i] for i in members}) == len(members)
            for earlier, later in zip(members, members[1:]):
                before[later] = earlier
        return rows, before, unique, groups

    def baselines_by_hand(self, by_entity=False):
        header, source = self.r["header"], self.r["source"]
        t = header.index(self.c.target)
        rows, before, unique, groups = self.series_before(by_entity)
        held = self.held()
        level = {i: float(source[i][t]) for i in rows}
        in_holdout = set(held)
        truth = [level[p] for p in held]
        training_mean = g.mean([level[i] for i in rows if i not in in_holdout])
        base_mean = g.mse(truth, [training_mean] * len(truth))
        without = [p for p in held if p not in before]
        base_previous = g.mse(truth, [level[before[p]] if p in before else training_mean
                                      for p in held])
        mine = g.mse(truth, [guess for _, _, guess in self.r["predictions"]])
        return base_mean, base_previous, mine, unique, without, groups

    def check_baselines(self, base_mean, base_previous):
        baselines = {b["name"]: b for b in self.at("baselines")}
        self.require({"training_mean", "previous_value"} <= set(baselines),
                     f"baselines lists {sorted(baselines)}; headline baseline "
                     f"{self.at('headline.baseline')!r}, skill {self.at('headline.skill')!r}")
        for name, want in (("training_mean", base_mean), ("previous_value", base_previous)):
            self.require(g.close(baselines[name]["score"], want, 1e-7),
                         f"the {name} baseline scores {baselines[name]['score']!r}; the judge "
                         f"computes {want!r}")

    def c_walk_baselines(self):
        base_mean, base_previous, mine, unique, _, _ = self.baselines_by_hand()
        if not unique:
            baselines = sorted(b["name"] for b in self.at("baselines"))
            self.require("previous_value" in baselines,
                         f"the timestamps of the source are not all distinct, the baselines "
                         f"are {baselines} and the headline is skill "
                         f"{self.at('headline.skill')!r} against "
                         f"{self.at('headline.baseline')!r}: a random walk scored against the "
                         f"training mean; warnings {sorted({w['code'] for w in self.at('warnings')})}")
            return
        self.check_baselines(base_mean, base_previous)
        skill = self.at("headline.skill")
        self.require(self.at("headline.baseline") == "previous_value",
                     f"the headline baseline is {self.at('headline.baseline')!r} (skill "
                     f"{skill!r}); the previous value scores {base_previous:.4g} against "
                     f"{base_mean:.4g} for the training mean")
        self.require(g.close(skill, 1 - mine / base_previous, 1e-7),
                     "headline.skill is not 1 - error / previous-value error")

    def c_int_keys(self):
        stamps, want = self.time_expectation()
        distinct = len(set(stamps.values())) == len(stamps)
        got = (self.at("validation.n_train_timestamps"), self.at("validation.n_holdout_timestamps"),
               self.at("validation.timestamps_unique"), self.at("validation.time_form"))
        expected = (len({stamps[i] for i in want["training"]}), want["holdout_stamps"],
                    distinct, "numeric")
        self.require(got == expected, f"training and holdout timestamps, unique, form: {got}; "
                                      f"the judge counts {expected}")
        for key in ("validation.train_end", "validation.holdout_start"):
            self.require(isinstance(self.at(key), int), f"`{key}` is {self.at(key)!r}, not a "
                                                        "whole number written exactly")

    def c_flat(self):
        base_mean, base_previous, mine, unique, _, _ = self.baselines_by_hand()
        self.require(unique, "the timestamps of the source are not all distinct; baselines "
                             f"{sorted(b['name'] for b in self.at('baselines'))}, skill "
                             f"{self.at('headline.skill')!r} against "
                             f"{self.at('headline.baseline')!r}")
        self.check_baselines(base_mean, base_previous)
        skill = self.at("headline.skill")
        if base_previous == 0:
            self.require(skill is None and self.at("headline.baseline") == "previous_value"
                         and self.at("headline.baseline_score") == 0,
                         f"skill is {skill!r} against {self.at('headline.baseline')!r}; the "
                         "previous value is exact on every holdout row, so skill has no value")
            self.require(self.at("headline.skill_low") is None
                         and self.at("headline.n_resamples_skill_used") == 0,
                         "an interval on skill where skill has no value")
            self.require(self.at("headline.interval_method") == "bootstrap_blocks"
                         and self.at("headline.n_resamples_used") == 200,
                         "the score lost its interval")
        else:
            best = min(base_mean, base_previous)
            self.require(skill is not None and g.close(skill, 1 - mine / best, 1e-7),
                         f"skill is {skill!r}; the judge computes {1 - mine / best!r}")

    def c_panel_baselines(self):
        base_mean, base_previous, mine, unique, without, groups = self.baselines_by_hand(True)
        self.check_baselines(base_mean, base_previous)
        skill = self.at("headline.skill")
        self.require(self.at("headline.baseline") == "previous_value",
                     f"the headline baseline is {self.at('headline.baseline')!r}, skill {skill!r}")
        self.require(g.close(skill, 1 - mine / base_previous, 1e-7),
                     "headline.skill is not 1 - error / previous-value error")
        got = (self.at("validation.group_col"), self.at("validation.n_groups_training"),
               self.at("validation.n_groups_holdout"),
               self.at("validation.n_holdout_without_previous"))
        self.require(got == (self.c.group, len(groups), len(groups), len(without)),
                     f"group_col, entities on each side, rows without a previous value: {got}; "
                     f"the judge counts {(self.c.group, len(groups), len(groups), len(without))}")

    def entity_sides(self):
        header, source = self.r["header"], self.r["source"]
        at = header.index(self.c.group)
        sides = self.sides()
        entity = {i: source[i][at].strip() for i in sides[0] + sides[1]}
        return entity, {entity[i] for i in sides[0]}, {entity[i] for i in sides[1]}, sides

    def c_entities_apart(self):
        entity, training, held, sides = self.entity_sides()
        both = sorted(training & held)
        self.require(not both, f"{len(both)} entities have rows on both sides (for example "
                               f"{both[:3]})")
        got = (self.at("validation.group_col"), self.at("validation.n_groups_training"),
               self.at("validation.n_groups_holdout"))
        if self.capped():                   # whole entities left out: fewer in training
            self.require(got[0] == self.c.group and got[2] == len(held)
                         and got[1] <= len(training),
                         f"group_col and entities on each side: {got}; the judge counts "
                         f"{self.c.group}, at most {len(training)}, {len(held)}")
        else:
            self.require(got == (self.c.group, len(training), len(held)),
                         f"group_col and entities on each side: {got}; the judge counts "
                         f"{(self.c.group, len(training), len(held))}")
        self.require(self.columns()[self.c.group]["role"] == "group",
                     "the entity column does not have the role `group`")

    def c_grouped_share(self):
        entity, _, _, sides = self.entity_sides()
        n, size = len(entity), {}
        for name in entity.values():
            size[name] = size.get(name, 0) + 1
        share = self.number("validation.holdout_share")
        self.require(0.25 <= share <= 0.25 + 2 * max(size.values()) / n
                     and g.close(share, len(sides[1]) / n),
                     f"holdout_share is {share!r}: {len(sides[1])} of {n} rows")

    def c_holdout_bounded(self):
        entity, _, held_entities, sides = self.entity_sides()
        n, size, held = len(entity), {}, len(sides[1])
        for name in entity.values():
            size[name] = size.get(name, 0) + 1
        left = self.capped()
        if left:
            # The cap leaves whole entities out and the output does not say which. Only
            # their sizes bear on property 3: take entities outside the holdout, the
            # smallest first, whose rows add up to the count.
            for name in sorted((e for e in size if e not in held_entities), key=size.get):
                if size[name] <= left:
                    left, n = left - size[name], n - size.pop(name)
            self.require(left == 0, f"the cap left {self.capped()} rows out, which no set of "
                                    "whole entities outside the holdout adds up to")
        filled, unit = 0, 0
        for rows in sorted(size.values()):
            filled, unit = filled + rows, rows
            if 4 * filled >= n:
                break
        self.require(4 * held >= n and 4 * held < n + 4 * unit,
                     f"the holdout holds {held} of {n} rows; property 3 of step 7 asks for "
                     f"{-(-n // 4)} to {(n + 4 * unit - 1) // 4}")
        spare = sorted(e for e in held_entities if 4 * (held - size[e]) >= n)
        self.require(not spare, f"the holdout could spare {len(spare)} of its entities")
        large = [e for e, rows in size.items() if 4 * rows >= n]
        small = sum(rows for rows in size.values() if 4 * rows < n)
        kept, alone = len(self.warned("large_group_kept_in_training")), \
            len(self.warned("large_group_is_holdout"))
        want = (len(large), 0) if 4 * small >= n else (len(large) - 1, 1)
        self.require((kept, alone) == want,
                     f"{kept} `large_group_kept_in_training` and {alone} `large_group_is_holdout`"
                     f" warning(s); the judge expects {want}")

    def c_combos_apart(self):
        sides, features = self.sides(), self.feature_columns()
        held = {self.combination(i, features) for i in sides[1]}
        if self.capped():
            # Rows outside the holdout whose combination is held out can only be rows
            # the cap left out: there are no more of them than that.
            stray = sum(self.combination(i, features) in held for i in sides[0])
            self.require(stray <= self.capped(),
                         f"{stray} rows outside the holdout have a combination of feature "
                         f"values that is held out; the cap left only {self.capped()} out")
            return
        training = {self.combination(i, features) for i in sides[0]}
        both = held & training
        self.require(not both,
                     f"{len(both)} of the {len(held | training)} combinations of the features "
                     f"that reached the models ({[self.r['header'][i] for i, _ in features]}) "
                     f"have rows on both sides; skill {self.at('headline.skill')!r}, "
                     f"n_holdout_copies {self.at('validation.n_holdout_copies')!r}")

    def c_group_units(self):
        alike, v = self.rows_alike(), self.at("validation")
        if self.capped():                   # the count is of the rows kept, which are not known
            self.require(isinstance(v.get("n_rows_same_features"), int)
                         and 0 < v["n_rows_same_features"] <= alike
                         and len(self.warned("same_features_grouped")) == 1,
                         f"n_rows_same_features is {v.get('n_rows_same_features')!r} under the "
                         f"cap; the judge counts {alike} among all usable rows")
            return
        self.require(v.get("n_rows_same_features") == alike,
                     f"n_rows_same_features is {v.get('n_rows_same_features')!r}; the judge "
                     f"counts {alike}")
        self.require(len(self.warned("same_features_grouped")) == (1 if alike else 0),
                     f"{len(self.warned('same_features_grouped'))} `same_features_grouped` "
                     f"warning(s) with {alike} rows that agree")
        if not alike:
            return
        sides, features = self.split_rows(), self.feature_columns(with_leaks=True)
        units = tuple(len({self.combination(i, features) for i in side}) for side in sides)
        got = (v.get("n_groups_training"), v.get("n_groups_holdout"))
        self.require(got == units, f"units on each side: {got}; the judge counts {units}")

    # ---- what a change adds
    def x_added(self, name, role, reason, rule, warned):
        got = self.columns().get(name, {})
        self.require((got.get("role"), got.get("reason")) == (role, reason),
                     f"the added column `{name}` has role {got.get('role')!r}, reason "
                     f"{got.get('reason')!r}, leak_rule {got.get('leak_rule')!r}; expected "
                     f"{role!r}, {reason!r}; skill {self.at('headline.skill')!r}, baseline "
                     f"{self.at('headline.baseline')!r}, warnings "
                     f"{sorted({w['code'] for w in self.at('warnings') if w['column'] == name})}")
        if rule:
            self.require(got.get("leak_rule") == rule, f"`{name}` has leak_rule "
                                                       f"{got.get('leak_rule')!r}")
        if warned and warned.startswith("no:"):
            self.require(not self.warned(warned[3:]), f"a `{warned[3:]}` warning under a "
                                                      "declared time column")
        elif warned:
            self.require(name in self.names_warned(warned),
                         f"no `{warned}` warning names `{name}`")

    def x_duplicate_rows(self):
        self.require(self.at("input.duplicate_rows") >= 1
                     and len(self.warned("duplicate_rows")) == 1,
                     f"input.duplicate_rows is {self.at('input.duplicate_rows')!r} with "
                     f"{len(self.warned('duplicate_rows'))} `duplicate_rows` warning(s)")

    def x_n_missing(self, original):
        name = self.c.name(original)
        i = self.r["header"].index(name)
        want = sum(g.is_missing(row[i]) for row in self.r["source"])
        got = self.columns()[name]["n_missing"]
        self.require(got == want, f"`{name}` reports n_missing {got!r}; the judge counts {want}")

    def x_too_large(self, original):
        name = self.c.name(original)
        got = self.columns()[name]
        self.require(got["n_infinite"] == 1 and name in self.names_warned("infinite_values"),
                     f"`{name}` reports n_infinite {got['n_infinite']!r}; `infinite_values` "
                     f"names {self.names_warned('infinite_values')}")

    def x_classes(self, classes, positive):
        got = [str(v) for v in self.at("task.classes")]
        self.require(got == classes and str(self.at("task.positive_class")) == positive,
                     f"task.classes is {got}, positive {self.at('task.positive_class')!r}; "
                     f"expected {classes}, {positive!r}")

    def same_as_base(self):
        base = self.base
        self.require(base is not None and base.get("finished"),
                     "the unperturbed run did not finish, so there is nothing to compare with")
        moved = []
        for key in ("metric", "score", "model", "baseline", "baseline_score", "skill",
                    "interval_method", "interval_low", "interval_high", "skill_low",
                    "skill_high", "n_resamples_used", "p_value"):
            a, b = self.at(f"headline.{key}"), self.at(f"headline.{key}", base["doc"])
            numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b))
            if not (g.close(a, b) if numbers else a == b):
                moved.append(f"headline.{key} {b!r} -> {a!r}")
        mine = self.counts()
        theirs = {r["item"]: r["count"] for r in base["doc"]["rows"]}
        if mine != theirs:
            moved.append(f"rows {theirs} -> {mine}")
        if self.at("cv.chosen") != base["doc"]["cv"]["chosen"]:
            moved.append(f"cv.chosen {base['doc']['cv']['chosen']} -> {self.at('cv.chosen')}")
        a, b = self.r["predictions"], base["predictions"]
        if [p for p, _, _ in a] != [p for p, _, _ in b]:
            moved.append("the holdout rows of predictions.csv")
        elif not all(g.close(x[2], y[2]) for x, y in zip(a, b)):
            moved.append("y_pred in predictions.csv")
        self.require(not moved, "moved against the unperturbed run: " + "; ".join(moved[:6]))


def observed(result, case):
    """What was seen, in a few values, for the report."""
    seen = {"rc": result["rc"], "error": result["code"]}
    if result.get("finished"):
        doc = result["doc"]
        skill = doc["headline"].get("skill")
        seen.update(skill=None if skill is None else round(skill, 4),
                    scheme=doc["validation"].get("scheme"),
                    baseline=doc["headline"].get("baseline"),
                    model=doc["headline"].get("model"),
                    holdout=len(result["predictions"]),
                    copies=doc["validation"].get("n_holdout_copies"),
                    warnings=sorted({w["code"] for w in doc["warnings"]}))
        watch = {"const_added", "row_id_added", "ref_added", "added_date", "noise_added",
                 "status_copy", "target_copy", case.name("day"), case.name("record_id")}
        seen["roles"] = {c["name"]: f"{c['role']}/{c['reason']}" for c in doc["columns"]
                         if c["name"] in watch}
    return seen


# ---------------------------------------------------------------- main --

def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="The perturbation battery (D71).")
    parser.add_argument("package", nargs="?", help="the package folder `mlprofessor`")
    parser.add_argument("--only", default=None, help="run, run:change or :change, comma-separated")
    parser.add_argument("--jobs", type=int, default=16, help="tool runs side by side")
    parser.add_argument("--list", action="store_true", help="print every pair and stop")
    parser.add_argument("--json", default=None, help="write every pair's verdict to this file")
    parser.add_argument("--work", default=None, help="folder for the tables and the outputs")
    args = parser.parse_args(argv)

    cases, left_out = build_pairs(args.only)
    if args.list:
        for case in cases:
            print(f"{case.run}:{case.change}  [{case.kind}]"
                  + (f"  ({case.decided})" if case.decided else "") + f"  {case.must}")
        for run, change, reason in left_out:
            print(f"LEFT OUT {run}:{change}  {reason}")
        for run, reason in LEFT_OUT_RUNS.items():
            print(f"LEFT OUT {run}:*  {reason}")
        print(f"{len(cases)} tool runs: {sum(c.change == 'base' for c in cases)} gate runs as "
              f"they are and {sum(c.change != 'base' for c in cases)} pairs; "
              f"{len(left_out)} pairs left out")
        return EXIT_PASS
    if not args.package:
        print("HARNESS ERROR: no package folder given")
        return EXIT_HARNESS
    package = Path(args.package).resolve()
    if package.name != "mlprofessor" or not (package / "__main__.py").is_file():
        print(f"HARNESS ERROR: {package} is not a package folder named mlprofessor")
        return EXIT_HARNESS
    work = Path(args.work).resolve() if args.work else HERE / "work"
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(lambda c: read_outputs(launch(c, package, work)), cases))
    wall = time.perf_counter() - started
    for result in results:
        text = (result.get("stderr") or "")
        for module in ("pandas", "numpy", "sklearn", "threadpoolctl"):
            if f"No module named '{module}'" in text:
                print(f"HARNESS ERROR: the interpreter cannot import {module}")
                return EXIT_HARNESS
    base = {c.run: r for c, r in zip(cases, results) if c.change == "base"}
    report, failing = [], 0
    for case, result in zip(cases, results):
        fails = Judge(case, result, base.get(case.run)).run()
        failing += bool(fails)
        report.append({"run": case.run, "change": case.change, "kind": case.kind,
                       "must": case.must, "decided": case.decided, "ok": not fails,
                       "fails": [list(f) for f in fails], "observed": observed(result, case),
                       "seconds": result.get("seconds"), "attempts": result.get("attempts"),
                       "args": case.args()})
        if fails:
            print(f"FAIL {case.run}:{case.change} [{case.kind}]"
                  + (f" ({case.decided})" if case.decided else ""))
            for name, message in fails:
                print(f"     {name}: {message}")
            seen = report[-1]["observed"]
            print("     seen: " + ", ".join(f"{k}={v}" for k, v in seen.items()
                                            if k not in ("warnings",)))
    total = sum(r.get("seconds") or 0 for r in results)
    print(f"{len(cases)} tool runs in {wall:.0f} s at {args.jobs} side by side "
          f"({total:.0f} s of tool time); {len(left_out)} pairs left out")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"package": str(package), "wall_seconds": round(wall, 1),
             "tool_seconds": round(total, 1), "jobs": args.jobs, "pairs": report,
             "left_out": [list(x) for x in left_out], "left_out_runs": LEFT_OUT_RUNS},
            indent=1), encoding="utf-8")
    if failing:
        print(f"BATTERY: FAIL ({failing} of {len(cases)} do not hold)")
        return EXIT_FAIL
    print(f"BATTERY: PASS ({len(cases)} runs, {len(cases) - len(base)} pairs)")
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
