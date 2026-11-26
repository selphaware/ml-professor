"""The perturbation battery: does a conclusion survive a small change? (D71)

The gate proves each of its conclusions on one table. Four reviews in a row
found, by changing one cell or adding one column to such a table, another form
of a fault the gate had proved in one form only. This script is that method
made permanent. For every fast run of the gate whose right conclusion is
known, it applies each of a fixed list of small changes to the table, or to
the call, and asserts that the conclusion still holds:

    stop       the same stop, with the same code and the texts spec 11 pins
    same       the change cannot move a number (a constant or an identifier
               column added, a column renamed, a blank line): headline, row
               counts, holdout rows and predictions equal those of the
               unperturbed run of the same package
    band       the table is another draw of the same kind (rows shuffled, a
               column of noise, one row repeated): the run's band and its
               structural claims
    band-or-named
               one cell is set to an extreme value: skill is inside the run's
               band; or the result names the row (`largest_row_dominates`) or
               the value (`extreme_value`), and the same call on the table
               without the named rows is inside the band. The judge makes
               that second call itself (D74 point 3)
    structure  roles, counts, baselines and splits, recomputed by the judge

and, on every run that finishes, the reconciliation rules of spec 5.3 and the
share of the squared error that its largest row carries, as the judge counts
it from predictions.csv (D74 point 3).

    python verify/battery.py <package folder> [--only run[:change],...]
                             [--jobs N] [--list] [--json FILE] [--work FOLDER]

    python verify/battery.py mlprofessor                     the build
    python verify/battery.py mlprofessor --only messy        one run, every change
    python verify/battery.py mlprofessor --only day_numbers:ymd_sentinel_0
    python verify/battery.py --list                          every pair, and every
                                                             pair left out, with why

Every pair is in one of three classes, and the last lines give the count of each:

    holds          what was asserted holds
    stated limit   it does not, and a decision says so: the pair is named in
                   STATED_LIMITS below with the decision's number
    open           it does not, and a decision has given the fault an owner:
                   the pair is named in OPEN below

A pair on a list that fails is reported under its class and does not fail the
tool. One that holds is counted as holding and named on a line of its own, so
that whoever owns it sees what moved. **A pair that fails and is on neither
list fails the tool.** The two lists are fixed here, by run and by change, and
change only by a decision in docs/DECISIONS.md: a pair is never added to one
because a build misses it. A miss is a finding.

**Nothing is open.** D73 gave every fault of the first run to pass 8, and the
rules of D74 and D76 make each of those pairs hold: OPEN is empty, and the
last lines still give its count, 0. The stated limits are the four kinds of
D76, six pairs: a copy that differs by an added column (D60), a whole-number
time column with an empty cell (D55), a distinct whole-number measure in a
table sorted by it (D31), and under time order a copy of the target with
noise above 1% of the target's spread (D76); and one pair where the headline
skill has no value, so that nothing can be said of what a carrier does to it
(D36, D80 point 2). **A pair asserts what the spec intends, never an outcome
known to be wrong:** each of the seven is asserted as it should come out,
fails, and is counted as a stated limit, not as holding.

**A pair that plants a carrier of the target** (a coded copy, two columns
that give the target only together, a copy with cells altered, and the
others: D77 point 9, D78, D80, D81) holds when each planted column ends in
one of four outcomes, or the headline does not rise: the column is excluded
as a suspected leak; it is named by a `strong_single_column` warning; the
headline skill is 0.98 or more and the `near_perfect_headline` warning names
it; or `headline.rests_on` names it. Of two planted columns, either one named
counts. Where none of the four holds, the pair holds when its headline skill
is no more than the unperturbed run's `headline.rests_on_floor` above the
unperturbed run's headline skill (D81): `within` that floor of it either way,
or `lowered` by more than the floor. The last lines count each of the six
outcomes, and the pairs that end in none.

Exit codes
    0  nothing fails outside the two lists
    1  at least one pair fails and is on neither list
    3  HARNESS ERROR: the tool itself, or the machine, could not do its work

Four things to know.

- It is not part of the gate, and no gate check runs it: it starts 1,229
  processes and, for the pairs whose skill is outside the band, about 140
  more of the judge's own. On the reference, with nothing else on the
  machine, that takes 274 to 296 seconds at 32 side by side, the default, and
  about 8,000 seconds of tool time (two runs on 2026-11-25; at 16 side by
  side it took 716 seconds, with the same verdict on every pair; until the
  eighth pass it was 794 processes and 190 to 230 seconds), where the
  per-edit gate has 20 seconds. The
  loop runs it through tests/test_judge_tools.py once a pass's last task is
  reached, and the lead runs it by hand before a review (D72). A builder runs
  it, or the pairs of one run, when a task changes a rule of steps 1 to 8.
- The tool is run as the gate runs it: a new process for each table,
  `python -m mlprofessor profile ...` from the package's parent, fast profile,
  seed 0, one core. The datasets, the bands and the arithmetic are the gate's
  own (verify/smoke_test.py, loaded as a module): nothing is asserted that the
  judge cannot recompute from the two files and the source. A borrowed band is
  said to be borrowed.
- Everything it writes is under verify/artifacts/battery/: the tables in
  `data/`, each written once and left alone while its bytes are what they
  should be, and the outputs of each run in `runs/<package>/<run>__<change>/`,
  where <package> is `build`, `reference` or the name of a broken copy.
  `--work FOLDER` puts both elsewhere.
- Under the fast profile's row cap the output does not say which rows were
  left out. Where a claim needs them (property 3 of step 7, the entities the
  cap must keep), the judge accepts the run when some set of whole entities
  outside the holdout accounts for the rows left out and meets the claim.

Standard library only.
"""
import argparse
import copy
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
GATE_FILE = HERE / "smoke_test.py"
WORK = HERE / "artifacts" / "battery"       # ignored by the loop's snapshot, like the gate's
EXIT_PASS, EXIT_FAIL, EXIT_HARNESS = 0, 1, 3
RUN_SECONDS = 180
TOO_LARGE = 3.4028234663852886e+38
SELECTION_FLOOR = 50        # training rows of a fold under the grouped scheme (D74 point 6)
MISSING = object()

# ------------------------------------------------------ the two fixed lists --
# (run, change) -> the decision that states the limit. The pair is asserted as the
# spec's intent has it, fails, and is reported as a stated limit. Exactly the four
# kinds of D76 (point 6), each with its decision, and D36 where the headline skill has
# no value (D80 point 2, D81), and no other.
STATED_LIMITS = {
    # a copy of a row that differs by an added column is not a copy (D60)
    ("dup_noise", "noise_added"): "D60",
    # a CSV whole-number column with an empty cell loads as floats, already rounded (D55)
    ("int_stamps", "missing_cell_date"): "D55",
    # a distinct whole-number measure in a table sorted by it is called id-like (D31)
    ("messy", "sorted_by_measure"): "D31",
    # under time order, a copy of the target with noise above 1% of the target's
    # spread cannot be told from a strong feature measured at the same moment (D76
    # point 2); on the three runs where the previous value is the baseline in use
    ("random_walk", "copy_noise_5pct"): "D76",
    ("int_stamps", "copy_noise_5pct"): "D76",
    ("panel", "copy_noise_5pct"): "D76",
    # where the headline skill has no value (a baseline with no error), nothing can be
    # said of how far a carrier moves it, and no fall is computed (D36, D80 point 2)
    ("flat_tail", "two_columns_together"): "D36",
}
# (run, change) -> (the finding, its owner). Empty, by D73 and D74: the 30 pairs the
# first run listed here were owned by pass 8, and every one of them must now hold.
# The class is kept so that the last lines always say how many pairs are open: 0.
OPEN = {}
# For the judge's second call (`c_band_or_named`): the package, the work folder, and
# every call that was made, as (label, seconds).
_SECOND_CALL = {"made": []}


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
BANDS = g.BANDS             # the gate's own; `dominant_entity` has one since D74 point 6

# ------------------------------------------------------------ the runs --
# run -> what its conclusion is. `stop`: the named error it must end in, and
# `mentions`: texts its message must hold. `claims`: what the judge recomputes on a
# finished run. `or_claims`: for a run that stops because of where the split fell,
# what must hold if another draw of the table lets it finish. `num`, `cat`: a numeric
# and a categorical feature of the table. `date`: a date column that is not the
# declared time column. `ymd`: a column of dates written YYYYMMDD. `kept`: for a run
# under the fast profile's cap, a column and the value of it whose rows the cap is
# sure to keep. `cell`: the table holds one extreme cell of its own, in the target or
# in a feature.
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
    "entities": dict(claims=["band:entities", "entities_apart", "grouped_share",
                             "fold_rule"], num="m1"),
    "panel": dict(claims=["band:panel", "time_split", "panel_baselines"], num="level_prev"),
    "unsaid_both": dict(stop="time_order_undeclared", flags=FOUR, num="f1", cat="seg"),
    "unsaid_entities": dict(stop="entities_undeclared", flags=["--group-col", "--no-entities"],
                            num="level_prev"),
    "few_valued": dict(claims=["band:few_valued", "combos_apart", "group_units"], cat="branch"),
    "dominant_entity": dict(claims=["band:dominant_entity", "entities_apart",
                                    "holdout_bounded", "cap_keeps_large", "fold_rule"],
                            num="n1"),
    "day_numbers": dict(claims=["day_role", "undeclared_date:day"], num="n1", ymd="day",
                        date="day"),
    # Added with the seventh pass (D69, D70, D72).
    "day_sentinel": dict(claims=["day_role", "undeclared_date:day"], num="n1", ymd="day",
                         date="day"),
    "few_valued_leak": dict(stop="holdout_copies_in_training", mentions=["status"],
                            cat="branch",
                            or_claims=["band:few_valued", "combos_apart", "group_units",
                                       "leak_status"]),
    "large_entity": dict(stop="fast_profile_group_too_large", mentions=["full"], num="n1"),
    "quarter_entity": dict(claims=["entities_apart", "holdout_bounded", "cap_keeps_large",
                                   "fold_rule"],
                           num="n1", kept=("customer", "C0000")),
    # Added with the eighth pass (D73, D74, D76): the gate's own runs of five changes
    # that this battery makes too, each on a table of its own.
    # `random_walk` with one row written twice: both baselines as the judge makes them,
    # the previous value being the mean target of the timestamp before, and no skill.
    "walk_repeated": dict(claims=["band:random_walk", "time_split", "walk_baselines"],
                          num="level_prev"),
    # `random_walk` with the target on another scale beside it: the copy is excluded as
    # a leak, the lag stays a feature, and what is left is `random_walk`.
    "walk_copy": dict(claims=["band:random_walk", "time_split", "walk_baselines",
                              "walk_copy_roles"], num="level_prev"),
    # `walk_copy` with one target cell of -999 in a training row: the copy is excluded
    # all the same, and the value is named. The row is in every fit, so skill is far
    # below the band (the gate asserts none): it is in band without the named rows.
    "walk_copy_cell": dict(claims=["band_or_named:random_walk", "time_split",
                                   "walk_baselines", "walk_copy_roles", "named:level"],
                           num="level_prev", cell="target"),
    # `random_walk` with one cell of the lag set to 999999 in a holdout row: that row
    # carries the model's whole error, the result says so and names the value, and
    # without the row skill is in band.
    "walk_late_cell": dict(claims=["band_or_named:random_walk", "time_split",
                                   "walk_baselines", "named:level_prev"], num="level_prev",
                           cell="feature"),
    # `messy` with its rows in another order: the identifier is still one, the measure
    # still a feature, and the conclusion of `messy` holds.
    "messy_shuffled": dict(claims=["band:messy", "messy_roles", "messy_coercion",
                                   "undeclared_date:signup_date"], num="f1", cat="seg",
                           date="signup_date"),
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
    # The six runs of the ninth pass (D77, D78, D80, D81). Each table is a gate table with
    # one of the battery's own changes of that pass already made, or a table whose
    # conclusion no change of the battery bears on.
    "pair_together": "`signal_reg` with two columns that give the target only together: "
                     "the pair `signal_reg:two_columns_together`, and that change on every "
                     "other run, carry its conclusion",
    "signal_reg_copied": "`signal_reg` with `f1` written again on another scale: the pair "
                         "`signal_reg:feature_written_twice`, and that change on every other "
                         "run, carry its conclusion",
    "leak_reg_coded": "`signal_reg` with the target as a code of 30 levels: the pairs "
                      "`signal_reg:coded_copy_text` and `:coded_copy_number`, and those "
                      "changes on every other run, carry its conclusion",
    "noise_tied": "`signal_reg` with two columns of noise whose values repeat: the pairs "
                  "`signal_reg:noise_rounded_1dp` and `:noise_score_1_to_40`, and those "
                  "changes on every other run, carry its conclusion",
    "skewed_copy": "a copy of a long-tailed target with five cells moved: the pairs "
                   "`copy_5_cells_*` on every run carry the conclusion, at several sizes of "
                   "change; the gate check reads the long tail itself",
    "last_stamp": "its conclusion is the size of a holdout of whole timestamps, which no "
                  "change of the battery moves: a change that adds or alters a column leaves "
                  "the timestamps as they are",
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

    _tables = {}            # dataset -> (header, rows, blank lines): each is made once

    def __init__(self, run):
        dataset, extra, _, _ = g.RUNS[run]
        if dataset not in Case._tables:
            header, rows, blank_after = g.DATASETS[dataset][0](g.dataset_seed(dataset))
            Case._tables[dataset] = ([str(h) for h in header],
                                     [[str(c) for c in row] for row in rows], set(blank_after))
        header, rows, blank_after = Case._tables[dataset]
        self.run, self.change, self.dataset = run, "base", dataset
        self.header = list(header)
        self.rows = [list(row) for row in rows]
        self.blank_after = set(blank_after)
        self.target = g.DATASETS[dataset][1]
        self.time, self.group, self.gap, self.duplicates = parse_statements(extra)
        info = RUNSPEC[run]
        self.info = info
        self.stop = info.get("stop")            # the named error the run must end in
        self.stop_ok = None                     # a stop that is as good as the conclusion
        self.stop_also = {}                     # another stop that is as right: its texts
        self.flags = list(info.get("flags", []))
        self.mentions = list(info.get("mentions", []))  # texts the stop's message must hold
        self.claims = list(info.get("claims", []))
        self.extra = []                         # claims the change adds
        self.alias = {}                         # a column's name in the gate -> its name here
        self.same = False                       # nothing may move against the unperturbed run
        self.touched = []                       # the rows a change altered, for the report
        self.rounds = 1                         # calls of its own the judge may make
        self.outcome = {}                       # a planted carrier of the target -> what became of it
        self.carrier = None                     # its skill, the run's without it, the floor
        self.kind, self.must = "", ""

    @property
    def listed(self):
        """(class, what) when the pair is on one of the two fixed lists, else None."""
        pair = (self.run, self.change)
        if pair in STATED_LIMITS:
            return "stated limit", STATED_LIMITS[pair]
        if pair in OPEN:
            return "open", OPEN[pair][1]
        return None

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

    def pick(self, rng, kept=False):
        """A row that has a target (any row of a malformed table). `kept`: one that the
        fast profile's cap is sure to keep, where the run is under the cap: a row of
        the entity that holds a quarter or more of the rows (D72)."""
        if self.info.get("malformed"):
            return rng.randrange(len(self.rows))
        t = self.header.index(self.target)
        rows = [i for i, row in enumerate(self.rows) if not g.is_missing(row[t])]
        if kept and self.info.get("kept"):
            column, value = self.info["kept"]
            at = self.col(column)
            rows = [i for i in rows if self.rows[i][at].strip() == value]
        return rng.choice(rows)


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
    if _stops(case) and case.info.get("or_claims"):
        # The run stops because of where its split fell. Another draw may let it
        # finish, and then what it writes must be sound.
        case.stop_ok, case.stop = case.stop, None
        case.claims = list(case.info["or_claims"])
        case.kind = "stop-or-structure"
        named = " and ".join(f"`{text}`" for text in case.mentions)
        case.must = (f"the same stop, `{case.stop_ok}`, naming {named}; or, if this draw puts "
                     "no combination of the features that are left on both sides, a run that "
                     "finishes with none on both sides, `status` excluded as a leak and no "
                     f"skill (the band of `few_valued`): {what}")
    elif _stops(case):
        case.kind, case.must = "stop", f"the same stop, `{case.stop}`: {what}"
    else:
        case.kind = "band" if any(c.startswith("band:") for c in case.claims) \
            else "band-or-named" if any(c.startswith("band_or_named:") for c in case.claims) \
            else "structure"
        case.must = f"the run's own conclusion (its band, its roles, its counts): {what}"


def _or_named(case):
    """One cell is set to an extreme value. The tool changes no number for it: it says
    which row carries most of the squared error, and which value is far outside its
    column (D74 point 3). So the conclusion of such a pair is not the band alone:
    `band:X` becomes `band_or_named:X`, and the roles must be those of the
    unperturbed run."""
    if _stops(case) or case.stop_ok:
        return
    case.claims = ["band_or_named:" + c[5:] if c.startswith("band:") else c
                   for c in case.claims]
    case.extra.append(("roles_as_base",))
    if case.info.get("cell"):
        # The run holds an extreme cell of its own, so this table holds two: the table
        # without the rows the result names is one that holds one, and the same claim
        # is made of it.
        case.rounds = 2
    if any(c.startswith("band_or_named:") for c in case.claims):
        case.kind = "band-or-named"
        case.must += (". Skill is inside the run's band; or the result names the row "
                      "(`largest_row_dominates`) or the value (`extreme_value`), and the "
                      "same call on the table without the named rows is inside the band. "
                      "The roles are those of the unperturbed run")


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
                     "ahead: a copy that differs in one feature is not detected (D60)")
    if case.run == "few_valued_leak":
        # The noise is written to four decimals, so a few rows of one branch agree in
        # it by chance, and one such pair on opposite sides is a combination on both
        # sides: the stop has no threshold (D72). Either end is right, and which it is
        # depends on the draw; a run that finishes must have none on both sides.
        case.extra = [("added", "noise_added", "feature", None, None, None)]
        case.must = ("with a column of noise almost no two rows agree, so there is little to "
                     "keep together and little for the stop of step 8a to find. The run "
                     "finishes with `status` excluded as a leak, no combination of the "
                     "features that are left on both sides and no skill (the band of "
                     "`few_valued`, borrowed); or, when two rows that agree in region, branch "
                     "and the noise as written fell on opposite sides, it stops with "
                     "`holdout_copies_in_training`, naming `status`: one combination on both "
                     "sides is enough (D72; spec step 8a)")


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
    if case.duplicates == "group" and _stops(case):
        case.mentions.append("status_copy")         # every column excluded as a leak (spec 11)
    if case.duplicates == "group" and not _stops(case):
        case.same, case.kind = False, "stop-or-structure"
        case.stop_ok = "holdout_copies_in_training"
        case.mentions = ["status_copy"]
        case.must = ("the run stops with `holdout_copies_in_training` and names `status_copy`: "
                     "once the leak screen has excluded it, a combination of the features that "
                     "are left is on both sides (D69, D72; spec step 8a). Or, if this draw puts "
                     "none on both sides, it finishes with none on both sides and skill in the "
                     "band of `few_valued`")


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
    if case.duplicates == "group" and _stops(case):
        case.mentions.append("target_copy")         # every column excluded as a leak (spec 11)
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
    k = case.pick(rng)
    case.rows[k][case.col(name)] = ""
    case.touched.append(k)
    _holds(case, f"one empty cell in the feature `{name}` is counted in `n_missing` and "
                 "imputed")
    if not _stops(case):
        case.extra.append(("n_missing", name))
    _or_named(case)


def ch_missing_cell_entity(case, rng):
    if not isinstance(case.group, str):
        return "the run declares no entity column"
    case.rows[case.pick(rng)][case.col(case.group)] = ""
    if _stops(case):
        # the table earns two errors, and which of them is raised is open (spec 2.4)
        case.stop_also = {case.stop: case.flags + case.mentions}
    case.stop, case.claims, case.extra, case.flags = "group_column_invalid", [], [], []
    case.mentions = [case.group]
    case.kind = "stop"
    case.must = ("the stop `group_column_invalid`: an empty entity cell in a row that has a "
                 "target (spec step 2); the message names the entity column"
                 + (f". The run's own stop, `{next(iter(case.stop_also))}`, is as right: "
                    "which error a table that earns two is given is open (spec 2.4)"
                    if case.stop_also else ""))


def ch_missing_cell_date(case, rng):
    if isinstance(case.time, str):
        case.rows[case.pick(rng)][case.col(case.time)] = ""
        _holds(case, "a row with no time value is dropped and counted in "
                     "`dropped_missing_time` (step 2)")
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


FLOAT_WHOLE = {"day_numbers": ["day"], "day_sentinel": ["day"],
               "messy": ["record_id", "visits"],
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
        _holds(case, f"one cell of the date column `{name}` set to {value}, a value that is "
                     "no date: the column is still a column of dates, since a value that is "
                     "no date counts once and hundreds of distinct dates stand beside it "
                     "(D69, D72; spec step 5, rule 2a)")
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
        k = case.pick(rng)
        case.rows[k][case.col(name)] = value
        case.touched.append(k)
        _holds(case, f"one cell of the numeric feature `{name}` set to {value}")
        if abs(float(value)) <= TOO_LARGE:
            _or_named(case)
        if float(value) > TOO_LARGE and not _stops(case):
            case.extra.append(("too_large", name))
            case.must += ("; a value above 3.4e38 is read as missing, counted in `n_infinite` "
                          "and named by `infinite_values` (D48)")
        return None
    return change


def ch_target_sentinel_neg(case, rng):
    if not case.regression():
        return "the target is not a regression target"
    k = case.pick(rng)
    case.rows[k][case.header.index(case.target)] = "-999"
    case.touched.append(k)
    _holds(case, "one target cell set to -999 (the spec has no rule for an outlier in the "
                 "target; docs/BACKLOG.md, 'One extreme value can dominate')")
    _or_named(case)
    return None


def ch_target_sentinel_max(case, rng):
    if not case.regression():
        return "the target is not a regression target"
    # in a row the cap is sure to keep: a cell the cap leaves out stops nothing
    case.rows[case.pick(rng, kept=True)][case.header.index(case.target)] = \
        "1.7976931348623157e308"
    if not _stops(case):
        case.stop, case.claims, case.extra = "non_finite_score", [], []
    elif case.stop not in ("time_order_undeclared", "entities_undeclared"):
        # a stop of a later step: which of two earned errors is raised is open (spec 2.4)
        case.stop_also = {"non_finite_score": []}
    case.kind = "stop"
    case.must = (f"the stop `{case.stop}`"
                 + (": one target cell of 1.7976931348623157e308 makes a baseline's squared "
                    "error infinite, before any fit (D44, rule 1)"
                    if case.stop == "non_finite_score" else
                    ": a statement is asked for at step 5, before the baselines (spec 2.4)"
                    if not case.stop_also else
                    ", or `non_finite_score`: the table earns both, and which is raised is "
                    "open (spec 2.4)"))
    return None


def _duplicates(value):
    def change(case, rng):
        if case.run not in ("dup_noise", "few_valued", "time_blocks_undeclared",
                            "few_valued_leak"):
            return "the run is not about repeated rows"
        if (case.duplicates or "error") == value:
            return "this is the run as the gate makes it"
        case.duplicates, case.stop, case.same = value, None, False
        case.stop_ok, case.mentions = None, []
        leak = ["leak_status"] if case.run == "few_valued_leak" else []
        date = [case.info["date"], "time_col"] if case.info.get("date") else []
        if value == "error":
            case.stop, case.claims, case.kind = "holdout_copies_in_training", [], "stop"
            case.mentions = date
            case.must = ("the stop `holdout_copies_in_training`: every holdout row has a row "
                         "like it among the training rows (step 8a)"
                         + ("; the message names the date column and `time_col`" if date else ""))
        elif value == "keep":
            case.claims = {"dup_noise": []}.get(case.run, ["band:few_valued"]) + leak
            case.kind = "structure" if case.run == "dup_noise" else "band"
            case.must = ("the run goes ahead with one `holdout_copies_kept` warning and the "
                         "count the judge makes for itself"
                         + ("; the target is noise in sixteen combinations, so no skill"
                            + ("; `status` is excluded as a leak" if leak else "")
                            if case.run != "dup_noise" else
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


# --------------------------------------- the changes of the eighth pass --
# Added with the rules of D73, D74 and D76, each where the next fault in one of them
# is most likely: repeated rows and shared timestamps under time order, a gap, copies
# of the target in other forms, extreme cells by where they fall, the identifier and
# the whole-number measure of `messy`, and one entity of nearly half the rows.

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
            _or_named(case)
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
    _or_named(case)
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
    _or_named(case)
    return None


def ch_number_two_sentinels(case, rng):
    name = case.info.get("num")
    if not name:
        return "no numeric feature"
    for k in _two_rows(case, rng):
        case.rows[k][case.col(name)] = "999999"
        case.touched.append(k)
    _holds(case, f"two cells of the numeric feature `{name}`, in two rows, set to 999999")
    _or_named(case)
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
        _or_named(case)
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
    _holds(case, "the table is sorted by the whole-number measure `visits`: a measure is "
                 "a feature whatever the order of the rows. A stated limit (D31): every "
                 "value is distinct and in the order of the file, so the first clause of "
                 "the id-like rule (D19) excludes it")
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


NOISE_LIMIT_RUNS = ("random_walk", "int_stamps", "panel")


def _told_copy(make_cells, what, only=None, by_spread=False):
    """Under time order, a column derived from the target that must be excluded as a
    suspected leak or at least named by a `strong_single_column` warning (D74 point
    2; D76 points 1 and 2)."""
    def change(case, rng):
        if not case.regression():
            return "the target is not a regression target"
        if not isinstance(case.time, str):
            return ("asserted under time order, where the previous value is a baseline; "
                    "under the other schemes the screen already excludes or warns")
        if only is not None and case.run not in only:
            return ("a stated limit, asserted on three runs where the previous value is the "
                    "baseline in use: " + ", ".join(only))
        if by_spread and case.info.get("cell") == "target":
            return ("the target of this run holds one cell far outside the rest, so its "
                    "spread is no measure for the noise")
        usable = _usable_rows(case)
        values = [None if g.is_missing(c) else float(c) for c in case.target_cells()]
        cells = make_cells(values, usable, rng, case)
        case.add_column("target_copy", cells)
        case.claims = [c for c in case.claims if not c.startswith("band")
                       and c not in ("walk_baselines", "panel_baselines", "flat", "int_keys")]
        case.extra.append(("copy_told", "target_copy"))
        case.kind = "structure"
        case.must = ("the added column `target_copy` is excluded as a suspected leak, or at "
                     "least named by a `strong_single_column` warning: " + what)
        return None
    return change


def _cells_altered(values, usable, rng, case):
    cells = ["" if v is None else g._f(3.7 * v + 12, 9) for v in values]
    picks = rng.sample(usable, max(1, round(0.02 * len(usable))))
    moved = [cells[i] for i in picks]
    for i, value in zip(picks, moved[1:] + moved[:1]):
        cells[i] = value
    case.touched += picks
    return cells


def _cells_noisy(share):
    def make(values, usable, rng, case):
        present = [values[i] for i in usable]
        mean = sum(present) / len(present)
        spread = math.sqrt(sum((v - mean) ** 2 for v in present) / len(present))
        return ["" if v is None else g._f(3.7 * v + 12 + rng.gauss(0, 3.7 * share * spread), 9)
                for v in values]
    return make


def _cells_curve(values, usable, rng, case):
    return ["" if v is None else g._f((v + 1000.0) ** 2, 6) for v in values]


def ch_measure_sentinel(case, rng):
    """-999 in `visits` of `messy`, whole numbers from 1,000 to 100,000. The value is
    not far outside the column and moves no number, so the warning need not name it
    (D76 point 3): the run is in band, or the value is named."""
    if _messy_only(case):
        return _messy_only(case)
    k = case.pick(rng)
    case.rows[k][case.col("visits")] = "-999"
    case.touched.append(k)
    _holds(case, "one cell of the whole-number measure `visits` set to -999, just below "
                 "its smallest value")
    _or_named(case)
    return None


CHANGES.update({
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
    "copy_2pct_cells_altered": _told_copy(_cells_altered, "a rescaled copy of the target in "
                                          "which the values of 2% of the rows are rotated "
                                          "among them"),
    "copy_curve": _told_copy(_cells_curve, "the square of the target plus 1,000, a curve that "
                             "rises with it"),
    "copy_noise_1pct": _told_copy(_cells_noisy(0.01), "a rescaled copy of the target with "
                                  "noise of 1% of the target's spread added", by_spread=True),
    "copy_noise_5pct": _told_copy(_cells_noisy(0.05), "a rescaled copy of the target with "
                                  "noise of 5% of the target's spread added. A stated limit "
                                  "(D76 point 2): above 1% such a column cannot be told from "
                                  "a strong feature measured at the same moment, and whether "
                                  "it is known before the outcome is the caller's to say",
                                  only=NOISE_LIMIT_RUNS, by_spread=True),
    "measure_sentinel_-999": ch_measure_sentinel,
})


# --------------------------------------- the changes of the ninth pass --
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
    """The change plants a carrier of the target. It holds when a planted column is
    excluded as a suspected leak, or warned about by name, or the headline is at or
    above 0.98 and the `near_perfect_headline` warning names it, or `headline.rests_on`
    names it (D77 point 9, D78 point 2); or, where none of these holds, when the
    headline skill is no more than the unperturbed run's rests-on floor above the
    unperturbed run's (D80 point 2, D81)."""
    if _stops(case):
        case.kind, case.must = "stop", f"the same stop, `{case.stop}`: {what}"
        return
    case.claims = [c for c in case.claims if not c.startswith("band")
                   and c not in ("walk_baselines", "panel_baselines", "flat", "int_keys",
                                 "p_floor", "majority_margin", "blocks_interval")]
    case.extra.append(("carrier",) + tuple(names))
    case.kind = "carrier"
    case.must = ("one of " + ", ".join(f"`{n}`" for n in names) + " is excluded as a suspected "
                 "leak, or named by a `strong_single_column` warning, or named by the "
                 "`near_perfect_headline` warning, or named in `headline.rests_on`; or the "
                 "headline skill is no more than the unperturbed run's rests-on floor above "
                 "the unperturbed run's (D81): " + what)


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


def build_pairs(only=None):
    """Every pair in scope, and every pair left out with its reason."""
    wanted = [w.strip() for w in (only or "").split(",") if w.strip()]

    def chosen(run, change):
        return not wanted or any(w in (run, f"{run}:{change}", f":{change}") for w in wanted)

    cases, left_out = [], []
    for run in RUNSPEC:
        pairs = []
        for change, apply in CHANGES.items():
            if not chosen(run, change):
                continue                    # not asked for: its table is not made
            case = Case(run)
            case.change = change
            reason = apply(case, random.Random(f"battery:{run}:{change}"))
            if reason:
                left_out.append((run, change, reason))
            else:
                pairs.append(case)
        if pairs or chosen(run, "base"):
            base = Case(run)                # the run as it is: what `same` is compared with
            base.kind = "stop" if base.stop else "base"
            base.must = (f"the stop `{base.stop}`" if base.stop
                         else "the gate's own conclusion")
            cases += [base] + pairs
    return cases, left_out


# ------------------------------------------------------------ tool runs --

def package_label(package):
    """`build`, `reference`, or the fault a broken copy of the reference carries."""
    name = g.fixture_name(package)
    return "build" if name is None else "reference" if name == "good" else name


def write_table(path, data):
    """Make sure the table is on disk, byte for byte; a file that is right is left alone."""
    if path.is_file() and path.stat().st_size == len(data) and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(f"{path.name}.{os.getpid()}.{id(data)}.tmp")
    scratch.write_bytes(data)
    os.replace(scratch, path)


def launch(case, package, work):
    """Run the tool once on one pair's table, as the gate runs it."""
    label = f"{case.run}__{case.change}"
    data = work / "data" / f"{label}.csv"
    out = work / "runs" / package_label(package) / label
    write_table(data, case.text().encode("utf-8"))
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
        if c.stop and r["code"] in c.stop_also:
            self.claim("stop", lambda: self.stopped(r["code"], c.stop_also[r["code"]]))
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
                     "u_copies", "u_finite", "u_largest_row", "u_rests_on"):
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

    def stopped(self, code, texts=None):
        r, c = self.r, self.c
        if r["rc"] == 0:
            skill = None
            if r.get("finished"):
                skill = r["doc"].get("headline", {}).get("skill")
            raise Fail(f"the run finished (exit 0, skill {skill!r}) where it must stop with "
                       f"`{code}`")
        self.require(r["rc"] == 3, f"exit code {r['rc']} (a named error is exit 3; a larger "
                                   "one is the interpreter's own, a crash); stderr: "
                                   + (g._tail(r["stderr"], 300) or "(empty)"))
        self.require(r["code"] == code, f"the error is `{r['code']}`, expected `{code}`: "
                                        + g._tail(r["stderr"], 300))
        self.require(not (r["out"] / "profile.json").exists(),
                     "profile.json was written although the run stopped")
        wanted = c.flags + c.mentions if texts is None else texts
        missing = [text for text in wanted if text not in r["stderr"]]
        self.require(not missing, f"the message of `{code}` does not hold {missing} (what a "
                                  "message must name is pinned in spec 11): "
                                  + g._tail(r["stderr"], 500))

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
        low, high = BANDS[name]
        self.require(isinstance(skill, (int, float)) and low <= skill <= high,
                     f"skill is {skill if skill is None else round(skill, 4)}, expected "
                     f"between {low} and {high} (the band of `{name}`)")

    def c_not_above(self, name):
        skill = self.at("headline.skill")
        high = BANDS[name][1]
        self.require(isinstance(skill, (int, float)) and skill <= high,
                     f"skill is {skill if skill is None else round(skill, 4)}, expected at most "
                     f"{high} (the upper end of the band of `{name}`)")

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

    def c_walk_copy_roles(self):
        self.role_is("level_scaled", "excluded", "suspected_leak", "predicts_target")
        self.require(self.c.name("level_scaled") in self.names_warned("suspected_leak"),
                     "no `suspected_leak` warning names level_scaled")
        self.role_is("level_prev", "feature", None)

    def c_named(self, original):
        """An `extreme_value` warning names the column (D74 point 3)."""
        name, named = self.c.name(original), self.names_warned("extreme_value")
        self.require(name in named,
                     f"`extreme_value` warnings name {named or 'no column'}: `{name}` holds one "
                     "value far outside all its other values, and the result must name it")

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
                     "`numeric_with_separators` names "
                     f"{self.names_warned('numeric_with_separators')}")

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

    def c_leak_status(self):
        self.role_is("status", "excluded", "suspected_leak", "predicts_target")
        self.require(self.c.name("status") in self.names_warned("suspected_leak"),
                     "no `suspected_leak` warning names status")

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

    def kept_in_time(self):
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
        """The share of the squared error that its largest row carries (D74 point 3). On
        the holdout, for the chosen model and for the headline baseline: the share
        and the row as the judge counts them from predictions.csv. Over the selection
        folds: the same four keys in `cv` and two in every record of `models`. And
        one `largest_row_dominates` warning for each statement above one half, the
        configurations of `models` sharing one. Null for a classification."""
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
        cv = self.at("cv")
        absent = [key for key in keys if key not in cv] + [
            f"models[{m.get('config')}].{key}" for m in self.at("models")
            for key in ("largest_row_share", "largest_row") if key not in m]
        self.require(not absent, f"the selection folds' statement is missing: {absent[:6]}")
        ruled = [m["config"] for m in self.at("models")
                 if isinstance(m["largest_row_share"], (int, float))
                 and m["largest_row_share"] > 0.5]
        expected += bool(ruled)
        over = cv["baseline_largest_row_share"]
        expected += bool(isinstance(over, (int, float)) and over > 0.5)
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

    def named_rows(self):
        """The rows the result names: the largest row of a squared error where its
        share is above 0.5 (the holdout or the selection folds, a configuration or
        the baseline), and every row that holds a value an `extreme_value` warning
        names."""
        rows = set()
        parts = [self.at("headline"), self.at("cv")] + list(self.at("models"))
        for part in parts:
            for key in ("", "baseline_"):
                share, row = part.get(key + "largest_row_share"), part.get(key + "largest_row")
                if isinstance(share, (int, float)) and share > 0.5 and row is not None:
                    rows.add(int(row))
        header, source = self.r["header"], self.r["source"]
        for w in self.warned("extreme_value"):
            found = re.search(r"holds the value (\S+) in (\d+) row", w["message"])
            if not found or w["column"] not in header:
                continue
            value, at = float(found.group(1)), header.index(w["column"])
            for i, line in enumerate(source):
                try:
                    cell = float(line[at])
                except ValueError:
                    continue
                if cell == value:
                    rows.add(i)
        return rows

    def c_band_or_named(self, name):
        """Skill inside the band; or the result names the row or the value, and the
        same call on the table without the named rows is inside the band (D74 point
        3). That call is the judge's own: one more run of the tool, on the table as
        it was changed with the named rows taken out, written beside the pair's own
        as `<run>__<change>__without`. On a table that holds two cells that were set,
        the run's own and the change's, the claim is made of that second table too:
        `rounds` calls at most."""
        low, high = BANDS[name]
        judge, where, story = self, self.where_touched(), []
        for calls in range(self.c.rounds + 1):
            skill = judge.at("headline.skill")
            if isinstance(skill, (int, float)) and low <= skill <= high:
                return
            shown = skill if skill is None else round(skill, 4)
            story.append(f"skill is {shown}" if not story
                         else f"the same call without them gives skill {shown}")
            told = "; ".join(story)
            self.require(calls < self.c.rounds,
                         f"{told}: still outside the band of `{name}` ({low} to {high}). The "
                         f"changed row(s): {where}")
            named = judge.named_rows()
            codes = sorted({w["code"] for w in judge.at("warnings")
                            if w["code"] in ("largest_row_dominates", "extreme_value")})
            self.require(named, f"{told}, outside the band of `{name}` ({low} to {high}), and "
                                f"the result names no row and no value. The changed row(s): "
                                f"{where}. model {judge.at('headline.model')}, baseline "
                                f"{judge.at('headline.baseline')}")
            self.require(len(named) <= 20, f"{told}, outside the band; the result names "
                                           f"{len(named)} rows, too many to be the changed cell")
            story.append(f"the result names row(s) {sorted(named)} ({codes})")
            case = copy.copy(judge.c)
            case.change = judge.c.change + "__without"
            case.rows = [list(row) for i, row in enumerate(judge.c.rows) if i not in named]
            case.blank_after = set()
            again = read_outputs(launch(case, _SECOND_CALL["package"], _SECOND_CALL["work"]))
            _SECOND_CALL["made"].append((again["label"], again.get("seconds") or 0))
            self.require(again.get("finished"),
                         "; ".join(story) + "; the same call without them did not finish: "
                         f"exit {again.get('rc')}, `{again.get('code')}`")
            judge = Judge(case, again, None)

    def c_fold_rule(self):
        """Under the grouped scheme every fold is scored as it is. When some fold has
        fewer than 50 training rows, and fewer training rows than validation rows, no
        choice between configurations is made: the first configuration that was
        fitted is used, and one `selection_not_made` warning says so (D74 point 6).
        By the judge's own count there is such a fold when one entity outside the
        holdout holds more than the rest of the training rows, and all but fewer
        than 50 of them."""
        floor = SELECTION_FLOOR
        folds = self.at("validation.folds")
        shape = [(f["training_rows"], f["validation_rows"], f["scored"]) for f in folds]
        starved = [f["training_rows"] for f in folds
                   if f["training_rows"] < floor and f["training_rows"] < f["validation_rows"]]
        said = len(self.warned("selection_not_made"))
        first = next(m["config"] for m in self.at("models") if m["cv_mean"] is not None)
        if starved:
            self.require(said == 1 and self.at("cv.chosen") == first
                         and self.at("headline.model") == first,
                         f"one fold trains on {min(starved)} rows, fewer than {floor} and "
                         f"fewer than it validates on: {said} `selection_not_made` "
                         f"warning(s), chosen {self.at('cv.chosen')!r}, the first "
                         f"configuration is {first!r}; the folds (training rows, validation "
                         f"rows, scored) are {shape}")
        else:
            self.require(said == 0, f"{said} `selection_not_made` warning(s) though no fold "
                                    f"trains on fewer than {floor} rows and on fewer than it "
                                    f"validates on: {shape}")
        size, held_entities, _ = self.entity_sizes()
        training, usable = self.counts()["training"], sum(size.values())
        big = max((rows for e, rows in size.items() if e not in held_entities), default=0)
        if (not self.capped() or 4 * big >= usable) and training - big < floor \
                and training - big < big:
            self.require(starved,
                         f"an entity of {big} rows is among the {training} training rows, so "
                         f"the fold that validates on it trains on fewer than {floor} and on "
                         f"fewer than it validates on; the folds are {shape}")
        unscored = [f for f in folds if not f["scored"]]
        self.require(len(self.warned("fold_not_scored")) == len(unscored),
                     f"{len(self.warned('fold_not_scored'))} `fold_not_scored` warning(s) for "
                     f"{len(unscored)} fold(s) not scored")
        if self.at("task.type") == "regression":
            self.require(not unscored, f"fold(s) {[f['fold'] for f in unscored]} of a "
                                       "regression are not scored")

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
        # With repeated timestamps too the previous value is scored: the mean target
        # of the timestamp before (D73, D74 point 1).
        base_mean, base_previous, mine, unique, _, _ = self.baselines_by_hand()
        self.check_baselines(base_mean, base_previous)
        self.lower_baseline(base_mean, base_previous, mine)

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
        self.lower_baseline(base_mean, base_previous, mine)
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

    def entity_sizes(self):
        """(rows of each entity among the usable rows, entities of the holdout, holdout rows)."""
        entity, _, held_entities, sides = self.entity_sides()
        size = {}
        for name in entity.values():
            size[name] = size.get(name, 0) + 1
        return size, held_entities, len(sides[1])

    def left_out_choices(self, size, held_entities, never=(), limit=3000):
        """The sets of whole entities, outside the holdout, that the rows the cap left
        out can be: the output does not say which they are. Entities of one size are
        alike for every claim made here, so a set is chosen by size. One empty set
        when the cap left nothing out. `never`: entities that may not be left out."""
        left = self.counts().get("dropped_row_cap") or 0
        if not left:
            yield set()
            return
        free = {}
        for name in sorted(size):
            if name not in held_entities and name not in never:
                free.setdefault(size[name], []).append(name)
        sizes, found = sorted(free), [0]

        def walk(i, remaining, chosen):
            if found[0] >= limit:
                return
            if remaining == 0:
                found[0] += 1
                yield {name for rows, k in chosen for name in free[rows][:k]}
                return
            if i == len(sizes):
                return
            rows = sizes[i]
            for k in range(min(len(free[rows]), remaining // rows), -1, -1):
                yield from walk(i + 1, remaining - k * rows, chosen + [(rows, k)])

        yield from walk(0, left, [])

    def property_three(self, size, held_entities, held):
        """What is wrong with the holdout against property 3 of step 7 and the two
        warnings of D66, given the rows of each entity that was split; None if nothing."""
        n = sum(size.values())
        filled, unit = 0, 0
        for rows in sorted(size.values()):
            filled, unit = filled + rows, rows
            if 4 * filled >= n:
                break
        if not (4 * held >= n and 4 * held < n + 4 * unit):
            return (f"the holdout holds {held} of {n} rows; property 3 of step 7 asks for "
                    f"{-(-n // 4)} to {(n + 4 * unit - 1) // 4}")
        spare = sorted(e for e in held_entities if 4 * (held - size[e]) >= n)
        if spare:
            return f"the holdout could spare {len(spare)} of its entities (step 7, property 3)"
        large = [e for e, rows in size.items() if 4 * rows >= n]
        small = sum(rows for rows in size.values() if 4 * rows < n)
        kept, alone = len(self.warned("large_group_kept_in_training")), \
            len(self.warned("large_group_is_holdout"))
        want = (len(large), 0) if 4 * small >= n else (len(large) - 1, 1)
        if (kept, alone) != want:
            return (f"{kept} `large_group_kept_in_training` and {alone} "
                    f"`large_group_is_holdout` warning(s); the judge expects {want} (D66)")
        return None

    def c_holdout_bounded(self):
        size, held_entities, held = self.entity_sizes()
        first = None
        for left_out in self.left_out_choices(size, held_entities):
            split = {e: rows for e, rows in size.items() if e not in left_out}
            wrong = self.property_three(split, held_entities, held)
            if wrong is None:
                return
            first = first or wrong
        raise Fail(first or f"the cap left {self.counts().get('dropped_row_cap')} rows out, "
                            "which no set of whole entities outside the holdout adds up to")

    def c_cap_keeps_large(self):
        """Step 6 with an entity column (D57, D70, D72): the cap keeps whole entities,
        at most 2,000 rows, none left out that would have fitted, and never leaves out
        one that holds a quarter or more of the usable rows."""
        got = self.counts()
        left = got.get("dropped_row_cap") or 0
        size, held_entities, _ = self.entity_sizes()
        usable = sum(size.values())
        if usable <= g.FAST_CAP:
            self.require(left == 0, f"rows.dropped_row_cap is {left} with {usable} usable rows, "
                                    f"no more than the cap of {g.FAST_CAP}")
            return
        kept = usable - left
        self.require(0 < kept <= g.FAST_CAP and kept == got["training"] + got["holdout"],
                     f"the cap kept {kept} of {usable} usable rows (training "
                     f"{got['training']}, holdout {got['holdout']}); it keeps at most "
                     f"{g.FAST_CAP}")
        large = {e for e, rows in size.items() if 4 * rows >= usable}
        fitted = False
        for left_out in self.left_out_choices(size, held_entities, never=large):
            fitted = True
            if kept + min(size[e] for e in left_out) > g.FAST_CAP:
                return
        biggest = max(size.values())
        self.require(fitted,
                     f"the cap left {left} of the {usable} usable rows out, and no set of whole "
                     "entities that each hold less than a quarter of the usable rows adds up "
                     f"to that: an entity of a quarter or more was left out (the largest holds "
                     f"{biggest} rows, {biggest / usable:.0%}). Such an entity is always kept by "
                     "the cap, and the run stops with `fast_profile_group_too_large` when it "
                     "cannot be (D70, D72; spec step 6)")
        raise Fail(f"the cap kept {kept} rows and left out an entity that would have fitted "
                   f"beside them under {g.FAST_CAP} (spec step 6)")

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

    def x_copy_told(self, name):
        got = self.columns().get(name, {})
        out = (got.get("role"), got.get("reason")) == ("excluded", "suspected_leak")
        warned = name in self.names_warned("strong_single_column")
        self.require(out or warned,
                     f"the added column `{name}` is a feature (role {got.get('role')!r}, reason "
                     f"{got.get('reason')!r}) and no `strong_single_column` warning names it; "
                     f"skill {self.at('headline.skill')!r} against "
                     f"{self.at('headline.baseline')!r}, model {self.at('headline.model')}")

    def named_by_near_perfect(self):
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
        finished = bool(self.base and self.base.get("finished"))
        base = self.base["doc"]["headline"].get("skill") if finished else None
        floor = self.at("headline.rests_on_floor")
        lift = skill - base if isinstance(skill, (int, float)) \
            and isinstance(base, (int, float)) else None
        base_floor = self.base["doc"]["headline"].get("rests_on_floor") if finished else None
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

    def x_roles_as_base(self):
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
                 "status_copy", "target_copy", "noise_rounded", "noise_score", "target_code",
                 "pair_a", "pair_b", case.name("day"), case.name("record_id"),
                 case.name("status")}
        rows = {r["item"]: r["count"] for r in doc["rows"]}
        seen["cap"] = rows.get("dropped_row_cap")
        for key, label in (("largest_row_share", "share"),
                           ("baseline_largest_row_share", "base_share")):
            value = doc["headline"].get(key)
            seen[label] = None if value is None else round(value, 4)
        seen["folds"] = [(f["training_rows"], f["validation_rows"], f["scored"])
                         for f in doc["validation"].get("folds", [])] \
            if doc["validation"].get("scheme") == "grouped" else None
        if case.touched:
            try:
                seen["changed"] = Judge(case, result, None).where_touched()
            except Exception as exc:  # noqa: BLE001
                seen["changed"] = f"(not placed: {type(exc).__name__})"
        seen["roles"] = {c["name"]: f"{c['role']}/{c['reason']}" for c in doc["columns"]
                         if c["name"] in watch}
    return seen


# ---------------------------------------------------------------- main --

CHANGE_WORDS = {
    "base": "the gate's run as it is",
    "ymd_sentinel_99991231": "one cell of the YYYYMMDD column is set to 99991231",
    "ymd_sentinel_0": "one cell of the YYYYMMDD column is set to 0",
    "ymd_sentinel_20240230": "one cell of the YYYYMMDD column is set to 20240230, no date",
    "ymd_sentinel_in_dropped_row": "one cell of the YYYYMMDD column is set to 99991231 in a "
                                   "row whose target is emptied",
    "date_sentinel_9999": "one cell of the ISO date column is set to 9999-12-31",
    "date_sentinel_1900": "one cell of the ISO date column is set to 1900-01-01",
    "date_sentinel_zero": "one cell of the ISO date column is set to 0",
    "int_time_sentinel_0": "one cell of the whole-number time column is set to 0",
    "int_time_sentinel_99991231": "one cell of the whole-number time column is set to 99991231",
    "number_sentinel_-999": "one cell of a numeric feature is set to -999",
    "number_sentinel_999999": "one cell of a numeric feature is set to 999999",
    "number_sentinel_1e308": "one cell of a numeric feature is set to 1e308",
    "target_sentinel_-999": "one cell of the regression target is set to -999",
    "target_sentinel_max_double": "one cell of the regression target is set to "
                                  "1.7976931348623157e308",
    "copy_two_valued": "a column `status_copy` is added: the target relabelled closed and open",
    "copy_distinct": "a column `target_copy` is added: the target made distinct in every row "
                     "(the class plus a millionth of the row number; for regression the "
                     "target rescaled)",
    "constant_added": "a column `const_added` of one value is added",
    "id_added_int": "a column `row_id_added` of consecutive whole numbers is added",
    "id_added_text": "a column `ref_added` of text distinct in every row is added",
    "second_date_added": "a column `added_date` of ISO dates is added",
    "noise_added": "a column `noise_added` of normal noise is added",
    "rows_shuffled": "the rows are written in another order",
    "row_repeated": "one row is written twice",
    "blank_line_added": "one blank line is added between two rows",
    "missing_cell_feature": "one cell of a feature is emptied",
    "missing_cell_entity": "one cell of the entity column is emptied",
    "missing_cell_date": "one cell of the time column, or of a date column, is emptied",
    "missing_cell_target": "one cell of the target is emptied",
    "feature_renamed": "a feature is renamed `renamed_feature`",
    "feature_renamed_awkward": "a feature is renamed `net value, \u00a3 (2024)`",
    "target_renamed": "the target is renamed `the outcome`",
    "role_column_renamed": "the time or date column is renamed `when`, the entity column `who`",
    "target_first": "the target is moved to the first column",
    "target_respelt": "the two classes are respelt, in the same sorted order",
    "target_respelt_flipped": "the two classes are respelt so that the other sorts second",
    "white_space_cells": "white space is put round some target values, levels and entity "
                         "values",
    "float_whole_numbers": "whole numbers are written as floats (20150101.0)",
    "duplicates_error": "the call passes --duplicates error (the default)",
    "duplicates_keep": "the call passes --duplicates keep",
    "duplicates_group": "the call passes --duplicates group",
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
    "copy_2pct_cells_altered": "a column `target_copy` is added: the target rescaled, the "
                               "values of 2% of the rows rotated among them",
    "copy_curve": "a column `target_copy` is added: the square of the target plus 1,000",
    "copy_noise_1pct": "a column `target_copy` is added: the target rescaled, with noise of "
                       "1% of the target's spread",
    "copy_noise_5pct": "a column `target_copy` is added: the target rescaled, with noise of "
                       "5% of the target's spread",
    "measure_sentinel_-999": "one cell of the whole-number measure `visits` is set to -999",
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
}


def show(case, record, result, package, lead):
    """One pair that does not hold, as a builder needs it."""
    print(f"{lead} {case.run}:{case.change} [{case.kind}]")
    print(f"     the change: {CHANGE_WORDS[case.change]}")
    print(f"     must hold: {case.must}")
    for name, message in record["fails"]:
        print(f"     {name}: {message}")
    print("     seen: " + ", ".join(f"{k}={v}" for k, v in record["observed"].items()
                                    if k != "warnings"))
    print(f"     table: {result['data']}")
    print(f"     output: {result['out']}")
    print(f"     again: python verify/battery.py {package} --only {case.run}:{case.change}")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="The perturbation battery (D71): does a "
                                                 "conclusion survive a small change?")
    parser.add_argument("package", nargs="?", help="the package folder `mlprofessor`")
    parser.add_argument("--only", default=None, help="run, run:change or :change, comma-separated")
    parser.add_argument("--jobs", type=int, default=32, help="tool runs side by side (default 32)")
    parser.add_argument("--list", action="store_true", help="print every pair and stop")
    parser.add_argument("--json", default=None, help="write every pair's verdict to this file")
    parser.add_argument("--work", default=None,
                        help="folder for the tables and the outputs (default "
                             "verify/artifacts/battery)")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code in (0, None):
            return EXIT_PASS
        print("HARNESS ERROR: bad arguments (this says nothing about the package)")
        return EXIT_HARNESS
    try:
        return run_battery(args)
    except HarnessError as exc:
        print(f"HARNESS ERROR - this says nothing about the package: {exc}")
        return EXIT_HARNESS


class HarnessError(Exception):
    """The battery itself, or the machine, could not do its work."""


def run_battery(args):
    unknown = [pair for pair in list(STATED_LIMITS) + list(OPEN)
               if pair[0] not in RUNSPEC or pair[1] not in CHANGES]
    absent = [run for run in RUNSPEC if run not in g.RUNS]
    unspoken = [run for run, (_, extra, profile, _) in g.RUNS.items()
                if run not in RUNSPEC and run not in LEFT_OUT_RUNS]
    if unknown or absent or unspoken:
        raise HarnessError(f"the battery and the gate do not agree: listed pairs that do not "
                           f"exist {unknown}; runs the gate does not have {absent}; runs of "
                           f"the gate the battery says nothing of {unspoken}")
    try:
        cases, left_out = build_pairs(args.only)
    except Exception as exc:  # noqa: BLE001  a table the battery could not make
        raise HarnessError(f"could not make the tables: {type(exc).__name__}: {exc}") from exc
    if args.only and not cases:
        why = "; ".join(f"{run}:{change} is left out ({reason})"
                        for run, change, reason in left_out[:3])
        raise HarnessError(f"--only {args.only} names no pair that is run"
                           + (f": {why}" if why else "") + ". See --list")
    if args.list:
        for case in cases:
            listed = case.listed
            print(f"{case.run}:{case.change}  [{case.kind}]"
                  + (f"  ({listed[0]}: {listed[1]})" if listed else "") + f"  {case.must}")
        for run, change, reason in left_out:
            print(f"LEFT OUT {run}:{change}  {reason}")
        for run, reason in LEFT_OUT_RUNS.items():
            print(f"LEFT OUT {run}:*  {reason}")
        print(f"{len(cases)} tool runs: {sum(c.change == 'base' for c in cases)} gate runs as "
              f"they are and {sum(c.change != 'base' for c in cases)} pairs; "
              f"{len(left_out)} pairs left out; {len(STATED_LIMITS)} pairs listed as a stated "
              f"limit and {len(OPEN)} as open")
        return EXIT_PASS
    if not args.package:
        raise HarnessError("no package folder given")
    package = Path(args.package).resolve()
    if package.name != "mlprofessor" or not (package / "__main__.py").is_file():
        raise HarnessError(f"{package} is not a package folder named mlprofessor with a "
                           "__main__.py")
    if args.jobs < 1:
        raise HarnessError("--jobs must be at least 1")
    work = Path(args.work).resolve() if args.work else WORK
    _SECOND_CALL.update(package=package, work=work)
    try:
        (work / "data").mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(lambda c: read_outputs(launch(c, package, work)), cases))
    except OSError as exc:
        raise HarnessError(f"could not write under {work} or start {sys.executable}: {exc}") \
            from exc
    for result in results:
        for module in ("pandas", "numpy", "sklearn", "threadpoolctl", "scipy", "joblib"):
            if f"No module named '{module}'" in (result.get("stderr") or ""):
                raise HarnessError(f"the interpreter {sys.executable} cannot import {module}; "
                                   "the modelling stack recorded in CLAUDE.md is not installed")
    base = {c.run: r for c, r in zip(cases, results) if c.change == "base"}
    # The judging, side by side too: a claim for an extreme cell may run the tool once
    # more, and those second calls should not wait in line. A judge reads its own
    # pair and the unperturbed run, and writes nothing that another reads.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        verdicts = list(pool.map(lambda pair: Judge(pair[0], pair[1],
                                                    base.get(pair[0].run)).run(),
                                 zip(cases, results)))
    wall = time.perf_counter() - started
    second = sorted(_SECOND_CALL["made"])
    report, shown = [], {"fails": [], "stated limit": [], "open": [], "moved": []}
    for case, result, fails in zip(cases, results, verdicts):
        listed = case.listed
        verdict = "holds" if not fails else listed[0] if listed else "fails"
        record = {"run": case.run, "change": case.change, "kind": case.kind, "must": case.must,
                  "class": verdict, "listed": f"{listed[0]}: {listed[1]}" if listed else None,
                  "fails": [list(f) for f in fails], "observed": observed(result, case),
                  "seconds": result.get("seconds"), "attempts": result.get("attempts"),
                  "args": case.args(), "touched": list(case.touched),
                  "outcome": dict(case.outcome), "carrier": case.carrier}
        report.append(record)
        if fails:
            shown[verdict].append((case, record, result))
        elif listed:
            shown["moved"].append((case, listed))
    for case, record, result in shown["fails"]:
        show(case, record, result, args.package, "FAIL")
    for label in ("stated limit", "open"):
        for case, record, result in shown[label]:
            show(case, record, result, args.package, f"{label.upper()} ({case.listed[1]}):")
            if label == "open":
                print(f"     the finding: {OPEN[(case.run, case.change)][0]}")
    for case, listed in shown["moved"]:
        print(f"LISTED AND HOLDS HERE {case.run}:{case.change} ({listed[0]}: {listed[1]})")
    count = {label: sum(r["class"] == label for r in report)
             for label in ("holds", "stated limit", "open", "fails")}
    total = sum(r.get("seconds") or 0 for r in results) + sum(s for _, s in second)
    print(f"{len(cases)} tool runs, and {len(second)} second calls by the judge, in "
          f"{wall:.0f} s at {args.jobs} side by side ({total:.0f} s of tool time): "
          f"{len(base)} gate runs as they are and {len(cases) - len(base)} pairs; "
          f"{len(left_out)} pairs left out (--list says why)")
    print(f"holds {count['holds']}; stated limit {count['stated limit']} of "
          f"{len(STATED_LIMITS)} listed; open {count['open']} of {len(OPEN)} listed; "
          f"on neither list and failing {count['fails']}")
    planted = [r for r in report if r["outcome"]]
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
            Path(args.json).write_text(json.dumps(
                {"package": str(package), "wall_seconds": round(wall, 1),
                 "tool_seconds": round(total, 1), "jobs": args.jobs, "counts": count,
                 "second_calls": [label for label, _ in second],
                 "stated_limits": {f"{run}:{change}": decision
                                   for (run, change), decision in STATED_LIMITS.items()},
                 "open": {f"{run}:{change}": list(what) for (run, change), what in OPEN.items()},
                 "pairs": report, "left_out": [list(x) for x in left_out],
                 "left_out_runs": LEFT_OUT_RUNS}, indent=1), encoding="utf-8")
        except OSError as exc:
            raise HarnessError(f"could not write {args.json}: {exc}") from exc
    summary = (f"holds {count['holds']}, stated limit {count['stated limit']}, "
               f"open {count['open']}")
    if count["fails"]:
        print(f"BATTERY: FAIL ({count['fails']} of {len(cases)} do not hold and are on neither "
              f"list; {summary})")
        return EXIT_FAIL
    print(f"BATTERY: PASS ({summary})")
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
