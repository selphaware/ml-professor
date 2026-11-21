"""The ML Professor gate (phase 1), built on gatekit.

Deliverable: the package folder `mlprofessor`. The gate runs it the way a user
would, `python -m mlprofessor profile ...` with the working directory set to
the folder's parent, in a new process per dataset, and reads the two files the
tool writes. One check drives the Python function instead.

A modelling tool has no right answer on real data, so nothing here asserts a
score on real data. Every dataset is synthetic, generated below from a fixed
seed into verify/artifacts/data/, and built so the right conclusion is known.

Named silent failures (docs/DECISIONS.md D13; spec.md "Failure modes"):
  F1 a fitted step sees the holdout          F5 the task is misread
  F2 a leak is used as a feature             F6 something is lost unreported
  F3 the headline is a cross-validation score F7 time order is ignored
  F4 accuracy is the headline                F8 only the training-mean baseline
  F9 copies of holdout rows in training are scored with no error (D49)
  F10 an entity has rows on both sides of the split although its column was declared (D57)

F6, F7 and F8 each have more than one form, and one broken fixture cannot fail
every check of its family. The other forms have fixtures of their own (D46):
  F6b a column is lost from the accounting    F6d a long row is read silently
  F6c a model family is lost from the table   F7b a day-first date is parsed leniently
  F7c whole-number time keys are read as floats, so distinct instants merge (D50)
  F7d a table with a date column is split at random without the caller's say (D58)
  F8b a panel is scored against the training mean although its entities were declared (D57)

The self-test proves more than gatekit asks. gatekit accepts a failure mode as
soon as one check that names it fails on its fixture. Here every check that
names a mode must fail on that mode's own fixture, and the self-test prints
the table that shows it (see `ProvenGate`).

The judge's own arithmetic (ROC AUC, squared error, row counts, which rows are
the holdout) is standard library only, so it shares no bug with pandas, numpy
or scikit-learn. Every score is asserted as a band; every band was measured on
the reference implementation in verify/reference/ over 30 data seeds, and the
measurements are written beside it.

The files a run reads are kept: verify/artifacts/runs/<run>/ for a gate run on
a build, verify/artifacts/selftest_runs/<fixture>/<run>/ for the self-test, so
that a self-test never overwrites the files a builder is told to read (D46).

Usage
    python verify/smoke_test.py mlprofessor [--only id,id] [--fast] [--json]
    python verify/smoke_test.py --selftest | --list | --matrix
"""
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gatekit import EXIT_FAIL, EXIT_PASS, CheckFailed, Gate, HarnessError  # noqa: E402


def fixture_name(deliverable):
    """Which fixture a deliverable is: `good`, a failure-mode id, or None for a build.

    A fixture is a copy of the reference implementation, and its `_variant.py`
    says which fault it carries. A build has no such file.
    """
    try:
        text = (Path(deliverable) / "_variant.py").read_text(encoding="utf-8")
    except OSError:
        return None
    found = re.search(r"^BROKEN\s*=\s*(None|'[^']*'|\"[^\"]*\")\s*$", text, re.M)
    if not found:
        return None
    return "good" if found.group(1) == "None" else found.group(1)[1:-1]


class _HeldVerdict:
    """Standard output for the length of a self-test.

    Every line goes straight through, except gatekit's verdict, its last line,
    which is kept back: the proof table is printed above it, and the verdict
    stays the last line of the self-test, where the loop's doctor reads it.
    """

    def __init__(self, stream):
        self.stream, self.verdict, self._partial = stream, [], ""

    def write(self, text):
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            if line.startswith("SELFTEST: "):
                self.verdict.append(line)
            else:
                self.stream.write(line + "\n")
        return len(text)

    def flush(self):
        self.stream.flush()

    def release(self):
        if self._partial:
            self.stream.write(self._partial)
            self._partial = ""
        self.stream.flush()


class ProvenGate(Gate):
    """gatekit's gate, with a self-test that also proves every `catches`.

    gatekit's self-test passes a failure mode when any one check that names it
    fails on its fixture. A check could therefore name a mode and never have
    been seen to fail: the second review found four (D46). This self-test adds
    the stricter rule: **every check that names a failure mode fails on that
    mode's own fixture, in every run of it** (a fixture is run from a temporary
    folder and from the project folder). It prints the table that proves it,
    and a second table, for information only, of the checks that give a `why`
    in place of a failure mode and the fixtures they failed on.

    gatekit is a kit file and is not changed: this class only watches the runs
    the self-test makes, and keeps gatekit's verdict line back so that it is
    still the last line printed.
    """

    _watching = None        # fixture name -> the checks that failed, one set per full run

    def run(self, deliverable, only=None, fast=False):
        results = super().run(deliverable, only=only, fast=fast)
        if self._watching is not None and only is None and not fast:
            self._watching.setdefault(fixture_name(deliverable), []).append(
                {r["id"] for r in results if not r["ok"]})
        return results

    def selftest(self):
        self._watching = {}
        held = _HeldVerdict(sys.stdout)
        sys.stdout = held
        try:
            code = super().selftest()
        finally:
            sys.stdout = held.stream
            held.release()
            seen, self._watching = self._watching, None
        unproved = self._print_proof(seen) if seen else []
        if code == EXIT_PASS and unproved:
            for check_id, mode in unproved:
                print(f"SELFTEST DEFECT: check '{check_id}' names {mode} and did not fail on "
                      f"the {mode} fixture in every run of it. A check that names a failure "
                      "mode must be seen to fail on it: fix the check, give the form a "
                      "fixture of its own, or correct catches=[...].")
            print(f"SELFTEST: FAIL ({len(unproved)} defect(s))")
            return EXIT_FAIL
        for line in held.verdict:
            print(line)
        return code

    def _print_proof(self, seen):
        """Print both tables. Returns the (check, mode) pairs that were not proved."""
        modes = list(self._modes)
        runs = {mode: seen.get(mode, []) for mode in modes}
        always = {m: set.intersection(*runs[m]) if runs[m] else set() for m in modes}
        ever = {m: set().union(*runs[m]) if runs[m] else set() for m in modes}
        unproved = []
        print("\nPROOF: every check that names a failure mode, run on that mode's own fixture")
        print("| Check | Names | Failed on that fixture | Also failed on the fixture of |")
        print("|---|---|---|---|")
        for entry in self._checks.values():
            if not entry["catches"]:
                continue
            verdicts = []
            for mode in entry["catches"]:
                proved = bool(runs[mode]) and entry["id"] in always[mode]
                verdicts.append(f"{mode}: yes, {len(runs[mode])} of {len(runs[mode])} runs"
                                if proved else f"{mode}: NO")
                if not proved:
                    unproved.append((entry["id"], mode))
            others = [m for m in modes if m not in entry["catches"] and entry["id"] in ever[m]]
            print(f"| {entry['id']} | {', '.join(entry['catches'])} | {'; '.join(verdicts)} | "
                  f"{', '.join(others) or '-'} |")
        pairs = sum(len(c["catches"]) for c in self._checks.values())
        print(f"{pairs - len(unproved)} of {pairs} pairs of a check and a failure mode it names "
              f"were seen failing, over {len(modes)} broken fixtures.")
        print("\nFOR INFORMATION: checks that give a reason in place of a failure mode, and the "
              "broken fixtures they failed on (no part of the verdict)")
        print("| Check | Failed on the fixture of |")
        print("|---|---|")
        for entry in self._checks.values():
            if not entry["catches"]:
                on = [m for m in modes if entry["id"] in ever[m]]
                print(f"| {entry['id']} | {', '.join(on) or 'none: never seen to fail here'} |")
        return unproved


gate = ProvenGate("mlprofessor")

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference" / "mlprofessor"
PARALLEL = 8                 # tool runs side by side, each single-threaded
# Seconds between process starts. Measured on this machine on 2026-11-15, when the
# interpreter's base and standard library sat on A:, a drive mapped to a loopback
# network share: interpreters started at the same instant (eight, four, or just
# two) all blocked for exactly 35 or 70 seconds and then finished together; 35
# seconds is the share's lock-break timeout. Rounds of eight starts that blocked,
# by spacing: none 6 of 6, 0.1 s 2 of 2, 0.2 s 3 of 30, 0.3 s 0 of 30, 0.5 s 0 of 30.
# Since D32 the base is a local path. Measured again on 2026-11-16: eight starts at
# the same instant, 30 rounds, twice, and none blocked. The spacing is kept.
START_SPACING = 0.5
FAST_CAP = 2000              # spec.md: the fast profile's row cap
SEED = 1115                  # base seed of every dataset the gate uses
MISSING_TOKENS = {"", "n/a"}  # the only missing-value spellings the gate writes

LIMITATIONS = [
    "Predictability here means how well these model families did on this data under this "
    "validation scheme.",
    "Low skill is not proof that the target cannot be predicted.",
    "High skill is not proof of cause.",
    "A random split is optimistic when rows are ordered in time or one entity appears on "
    "many rows.",
    "Look-ahead inside the caller's own features cannot be detected.",
    "Classification skill and regression skill are on different scales and are not comparable.",
    "A CSV file is read with pandas' types, so text identifiers can lose leading zeros; "
    "pass a DataFrame to control the types.",
    "Intervals are approximate: they tend to be too narrow when the holdout is short and, "
    "under time order, when errors are serially correlated.",
    "Negative skill means the models did worse than the baseline on the holdout; read it as "
    "no evidence of predictability.",
]

# ---------------------------------------------------------------- bands --
# Each band: (low, high) on headline.skill. Measured on the reference
# implementation ("good") and on each broken copy the check is meant to catch,
# over 30 data seeds, fast profile unless stated. Each band is the good mean
# plus or minus about 4.5 standard deviations, so that a correct implementation
# whose random split differs from the reference's still lands inside.
# Do not widen a band because a build misses it: a miss is a finding.
BANDS = {
    # good 0.651..0.807 (mean 0.727, sd 0.040). `leak` with both leaks excluded:
    # 0.645..0.791. F2 (leak screen removed) on `leak`: 1.0000 on every seed.
    "signal_clf": (0.54, 0.91),
    # good 0.811..0.878 (mean 0.844, sd 0.018). `leak_reg` with the copy excluded:
    # 0.810..0.882. F2 on `leak_reg`: 0.9994..0.9996.
    "signal_reg": (0.76, 0.93),
    # good 0.630..0.824 (mean 0.731, sd 0.039). Re-measured on 30 other seeds when
    # `revenue` and `fee` were added and three text cells of `spend` were reworded:
    # 0.670..0.826 (mean 0.746, sd 0.035), identical seed by seed before and after,
    # since neither new column reaches a model and `spend` parses as it did. Band
    # unchanged (docs/measurements/CP1).
    "messy": (0.55, 0.91),
    # good -0.099..+0.141 (mean +0.010, sd 0.058). F1 (refit on all rows):
    # 0.389..0.559 when the linear model is chosen, 1.0000 when the tree is.
    "noise_clf": (-0.27, 0.27),
    # good -0.502..-0.253 (mean -0.357, sd 0.062): 300 useless columns on 1,125
    # training rows cost a linear model about a third of the baseline's error.
    # F1: +0.126..+0.271. The unlimited tree, if wrongly chosen: about -1.
    "noise_reg": (-0.65, -0.05),
    # good -0.185..+0.182 (mean -0.001, sd 0.092); about 25 positives in the holdout.
    "imbalanced": (-0.42, 0.42),
    # full profile, 400 rows: good -0.223..+0.189 (mean -0.008, sd 0.102). F3 lands
    # inside this band (-0.233..+0.218); it is caught by the exact comparison instead.
    "selection_full": (-0.47, 0.47),
    # good -0.167..+0.270 (mean +0.012, sd 0.104). F7 (shuffled split): 0.994..1.000.
    "time_blocks": (-0.48, 0.48),
    # good -0.118..+0.009 (mean -0.008, sd 0.024, skewed down). F8 (training mean
    # only): 0.881..1.000.
    "random_walk": (-0.30, 0.05),
    # 300 entities of 5 rows, a label per entity, `--group-col customer` (CP6):
    # good -0.260..+0.168 (mean -0.027, sd 0.119). F10 (the split ignores the entity column): +0.760..+0.882.
    "entities": (-0.57, 0.52),
    # twelve random walks on the same dates, `--group-col series` (CP6): good
    # -0.023..+0.006 (mean -0.003, sd 0.007). F8b (scored against the training
    # mean): 0.995..0.999.
    "panel": (-0.04, 0.03),
}
# The fast profile's p-value takes the values k/201. On noise_clf: good 0.0100..0.965
# (never at the floor in 30 seeds); F1 exactly 1/201 = 0.004975 on all 30. The cut-off
# sits between 1/201 and 2/201. A correct implementation lands on the floor by chance
# once in 201 runs: that is the price of asserting a p-value at all.
P_VALUE_FLOOR = 0.0075
# |headline.score - majority share| on `imbalanced`: good 0.363..0.549 (mean 0.452,
# sd 0.047); F4 (accuracy as the headline) 0.000..0.078.
MAJORITY_MARGIN = 0.20


# ------------------------------------------------- synthetic ground truth --

def _f(value, places=4):
    return f"{value:.{places}f}"


SEG_EFFECT = {"A": -1.0, "B": 0.0, "C": 0.5, "D": 1.2}
REGIONS = ["north", "south", "east", "west"]
OPENERS = ["called about", "asked for", "complained about", "praised", "returned", "queried",
           "cancelled", "renewed", "upgraded", "downgraded", "paused", "reported", "requested",
           "disputed", "confirmed", "missed", "rescheduled", "declined", "accepted", "reviewed",
           "forgot", "changed", "lost", "found", "shared", "ignored", "escalated", "closed",
           "reopened", "mentioned", "compared", "tested", "wanted", "needed", "liked",
           "disliked", "questioned", "approved", "rejected", "noticed"]
CLOSERS = ["the invoice", "a refund", "late delivery", "the new plan", "a faulty unit",
           "the price rise", "their password", "the mobile app", "a discount", "the contract",
           "billing dates", "a missing part", "the welcome pack", "an engineer visit",
           "the loyalty scheme", "account access", "a duplicate charge", "the help pages",
           "opening hours", "a replacement", "the annual review", "data usage", "the survey",
           "a callback", "the direct debit", "an address change", "the paper bill",
           "a second user", "the trial period", "delivery slots", "the warranty",
           "a price match", "the renewal letter", "an outage", "roaming charges",
           "the handset", "a late fee", "the upgrade offer", "store staff", "the website"]


def _signal_row(rng):
    """Two numeric features and one categorical decide the outcome; the rest is noise."""
    f1, f2, seg = rng.gauss(0, 1), rng.gauss(0, 1), rng.choice("ABCD")
    z = 1.6 * f1 - 1.2 * f2 + SEG_EFFECT[seg]
    y = 1 if rng.random() < 1 / (1 + math.exp(-z)) else 0
    return {"f1": f1, "f2": f2, "seg": seg, "z": z, "y": y,
            "noise": [rng.gauss(0, 1) for _ in range(3)], "region": rng.choice(REGIONS)}


def make_signal_clf(seed, n=1500, numeric_target=False, leak=False):
    rng = random.Random(seed)
    header = ["f1", "f2", "seg", "n1", "n2", "n3", "region"]
    if leak:
        header += ["outcome_copy", "settlement_amount", "strong_feature"]
    rows = []
    for _ in range(n):
        r = _signal_row(rng)
        row = [_f(r["f1"]), _f(r["f2"]), r["seg"]] + [_f(v) for v in r["noise"]] + [r["region"]]
        if leak:
            row += [_f(r["y"] + rng.gauss(0, 0.15)),                  # a noisy copy of the target
                    _f(rng.gauss(1000, 200), 2) if r["y"] else "",    # filled only after the event
                    _f(r["z"] + rng.gauss(0, 0.5))]                   # strong, and legitimate
        rows.append(row + [r["y"] if numeric_target else ("yes" if r["y"] else "no")])
    return header + ["outcome"], rows, ()


def make_near_leak(seed, n=1500, share=0.02):
    """signal_clf plus `outcome_noisy`: the target with exactly 2% of its labels flipped.

    Not a leak by the 0.98 rule, and far too good to pass without comment (D30). The
    rows to flip come from their own stream, and their number is fixed at 30 of
    1,500. Measured over 30 data seeds, the column's single-column skill: 0.938 to
    0.960 (mean 0.949, sd 0.005) with exactly 2% of the rows flipped; 0.924 to 0.959
    (mean 0.948, sd 0.008) with each row flipped at a 2% chance, where the share
    itself varies with the seed and 0.98 is only 4.2 sd above the mean.
    """
    header, rows, blank_after = make_signal_clf(seed, n=n)
    flipped = set(random.Random(f"near_leak.outcome_noisy.{seed}")
                  .sample(range(n), round(share * n)))
    other = {"yes": "no", "no": "yes"}
    rows = [row[:-1] + [other[row[-1]] if i in flipped else row[-1], row[-1]]
            for i, row in enumerate(rows)]
    return header[:-1] + ["outcome_noisy", "outcome"], rows, blank_after


def make_long_rows(seed, n=1500):
    """signal_clf with a row number from 1 leading every data row, and no name for it.

    The header has eight names and every row nine fields: a table exported with
    its row numbers. pandas reads the file without complaint. It takes the leading
    field as the index, which for row numbers is a `RangeIndex`, and gives the
    eight names to the eight fields that follow, so the table it returns looks
    like `signal_clf` and a column of the file is gone with nothing said. The
    tool must stop instead (spec.md section 6; docs/measurements/CP2).
    """
    header, rows, blank_after = make_signal_clf(seed, n=n)
    return header, [[i + 1] + row for i, row in enumerate(rows)], blank_after


def make_signal_reg(seed, n=1500, leak=False):
    rng = random.Random(seed)
    header = ["f1", "f2", "n1", "n2", "n3", "region"] + (["amount_scaled"] if leak else [])
    rows = []
    for _ in range(n):
        f1, f2 = rng.gauss(0, 1), rng.gauss(0, 1)
        y = 2.0 * f1 - 1.2 * f2 + rng.gauss(0, 1)
        row = [_f(f1), _f(f2)] + [_f(rng.gauss(0, 1)) for _ in range(3)] + [rng.choice(REGIONS)]
        if leak:
            row.append(_f(3.7 * y + 12 + rng.gauss(0, 0.2)))          # the target, rescaled
        rows.append(row + [_f(y)])
    return header + ["amount"], rows, ()


def make_noise(seed, n=1500, columns=300, kind="clf", share=0.5, region=False):
    """Nothing here predicts the target. Many columns, so that a model refitted on
    the holdout rows (F1) shows skill whichever model the tool chose."""
    rng = random.Random(seed)
    header = [f"n{i + 1:03d}" for i in range(columns)] + (["region"] if region else [])
    rows = []
    for _ in range(n):
        row = [_f(rng.gauss(0, 1)) for _ in range(columns)]
        if region:
            row.append(rng.choice(REGIONS))
        if kind == "clf":
            row.append("yes" if rng.random() < share else "no")
        else:
            row.append(_f(rng.gauss(0, 1)))
        rows.append(row)
    return header + ["outcome" if kind == "clf" else "amount"], rows, ()


def make_ambiguous(seed, n=1500):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        f1 = rng.gauss(0, 1)
        rating = max(1, min(7, round(4 + 1.2 * f1 + rng.gauss(0, 1))))
        rows.append([_f(f1), _f(rng.gauss(0, 1)), rng.choice("ABCD"), rating])
    return ["f1", "f2", "seg", "rating"], rows, ()


def make_messy(seed, n=1540):
    """signal_clf with everything a real export does to a table.

    `spend` is numeric with eight text cells, none of them a number with thousands
    separators, so it goes on testing coercion (D34). Two columns test D28 and D34,
    each drawn from its own stream and written after the target, so no other column
    of this table depends on them:

    `revenue` is written with thousands separators throughout. About one value in
    five is below 1,000 and so carries no separator: fewer than 90% of the values
    parse as plain numbers, fewer than 90% carry a separator, and only the two
    together reach 90%. Nearly every value is distinct, so without the rule the
    column would be called id-like.

    `fee` is plain numbers below 1,000 except for six cells of 1,000 or more, which
    carry a separator. More than 99% of its values parse, so the coercion rule
    alone would keep it as numeric with its six largest values turned into missing.
    """
    rng = random.Random(seed)
    amounts = random.Random(f"messy.revenue.{seed}")
    fees = random.Random(f"messy.fee.{seed}")
    large_fee = set(fees.sample(range(n), 6))
    base = [_signal_row(rng) for _ in range(n)]
    visits = rng.sample(range(1000, 100000), n)          # distinct, whole, not in order
    special = rng.sample(range(n), 18)
    # Until D34 the second, fourth and seventh of these were `1,200`, `3,450` and
    # `1,200`. They were coerced to missing then and are now: every parsed value of
    # the column is what it was.
    text_cells = dict(zip(special[:8], ["abc", "tbc", "12 units", "pending", "unknown", "abc",
                                        "tbc", "-"]))
    na_cells, blank_cells = set(special[8:14]), set(special[14:])
    no_target = set(rng.sample(range(n), 40))
    blank_after = set(rng.sample(range(n - 1), 5))       # blank lines: not rows
    start = datetime.date(2019, 1, 1)
    rows = []
    for i, r in enumerate(base):
        spend = text_cells.get(i, "n/a" if i in na_cells else "" if i in blank_cells
                               else _f(rng.gauss(50, 10), 2))
        rows.append([
            _f(r["f1"]),
            "" if rng.random() < 0.05 else _f(r["f2"]),
            "" if rng.random() < 0.04 else r["seg"],
            "n/a" if rng.random() < 0.03 else _f(r["noise"][0]),
            spend, visits[i], "GB", 100001 + i,
            f"{rng.choice(OPENERS)} {rng.choice(CLOSERS)}",
            (start + datetime.timedelta(days=rng.randrange(1500))).isoformat(),
            "" if i in no_target else ("yes" if r["y"] else "no"),
            f"{(-1 if amounts.random() < 0.03 else 1) * amounts.lognormvariate(7.92, 1.2):,.2f}",
            f"{fees.uniform(1000, 9000) if i in large_fee else fees.uniform(20, 980):,.2f}",
        ])
    header = ["f1", "f2", "seg", "n1", "spend", "visits", "const", "record_id", "notes",
              "signup_date", "outcome", "revenue", "fee"]
    return header, rows, blank_after


def make_time_blocks(seed, stamps=1800, dayfirst=False, block=5, features=8):
    """The label is random per block of consecutive dates; the features identify
    the block (jointly, never one alone) and carry nothing else. Rows are written
    in shuffled order, so the tool has to sort by the time column itself.

    Measured, 30 seeds: with blocks of 20 dates and noisy features the holdout
    held about 20 blocks, honest skill ranged -0.40 to +0.37 (sd 0.21) and the
    shuffled-split fault only 0.74 to 0.95, because a tree splits inside the
    feature noise. Blocks of 5 dates and exact feature values give -0.17 to +0.27
    (sd 0.10) against 0.994 to 1.000.
    """
    rng = random.Random(seed)
    start = datetime.date(2015, 1, 1)
    rows = []
    for t in range(stamps):
        if t % block == 0:
            label = rng.choice(["low", "high"])
            centre = [rng.choice((-1.0, 0.0, 1.0)) for _ in range(features)]
        day = start + datetime.timedelta(days=t)
        text = day.strftime("%d/%m/%Y") if dayfirst else day.isoformat()
        for _ in range(2 if rng.random() < 0.22 else 1):     # some dates hold two rows
            rows.append([text] + [_f(c, 1) for c in centre] + [label])
    rng.shuffle(rows)
    return ["date"] + [f"b{i + 1}" for i in range(features)] + ["regime"], rows, ()


def make_random_walk(seed, n=1500):
    """A random walk on business days; the one feature is yesterday's level."""
    rng = random.Random(seed)
    day, level, rows = datetime.date(2018, 1, 1), 0.0, []
    while len(rows) < n:
        if day.weekday() < 5:
            new = level + rng.gauss(0, 1)
            rows.append([day.isoformat(), _f(level), _f(new)])
            level = float(_f(new))
        day += datetime.timedelta(days=1)
    rng.shuffle(rows)
    return ["date", "level_prev", "level"], rows, ()


def make_flat_tail(seed, n=1500, flat_from=0.65):
    """Daily sales that stop: the target is exactly zero from 65% of the way through
    the series to its end. One row per date, written in shuffled order.

    The holdout is the last quarter of the dates. Every one of them is inside the
    flat stretch, and so is the last training date, so the previous-value baseline
    predicts every holdout row exactly: its squared error on the holdout is 0, and
    skill, 1 - error / baseline error, has no value (D36). The training mean is
    about 17 and scores about 300; a model fitted on the earlier dates predicts
    about 20 on days when nothing was sold.
    """
    rng = random.Random(seed)
    start = datetime.date(2019, 1, 1)
    first_flat = round(flat_from * n)
    rows = []
    for t in range(n):
        footfall, promo, noise = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
        sales = 0.0 if t >= first_flat else max(0.5, 20 + 6 * footfall - 3 * promo + 2 * noise)
        rows.append([(start + datetime.timedelta(days=t)).isoformat(), _f(footfall), _f(promo),
                     _f(sales)])
    rng.shuffle(rows)
    return ["date", "footfall", "promo", "sales"], rows, ()


def make_dup_noise(seed, n=750, columns=10):
    """Pure noise with every row present twice, in shuffled order (D49).

    Nothing predicts the target, so the right skill is about zero. But a quarter
    of the rows go to the holdout at random, so about three quarters of the
    holdout rows have their copy among the training rows, and a model that can
    recall a row scores on those: the third review measured skill 0.918 on such
    a table under the full profile, p = 0.001, against 0.009 without the copies.
    The tool must stop instead. Measured over 30 data seeds on the reference
    (docs/measurements/CP5): 263 to 305 of the 375 holdout rows have a copy,
    70% to 81%; the rule is more than 1% and at least 10 rows (D53). On every
    other dataset of this gate under the random scheme, and on the README's
    example table, the count is 0 on all 30 seeds, but for `time_blocks` run
    with no time column, where it is 494 to 500 of 500. The order of the rows
    comes from its own stream.
    """
    header, rows, blank_after = make_noise(seed, n=n, columns=columns, kind="clf")
    rows = rows + [list(row) for row in rows]
    random.Random(f"dup_noise.order.{seed}").shuffle(rows)
    return header, rows, blank_after


INT_STAMP_START = 1_700_000_000_000_000_000     # nanoseconds since 1970: November 2023
INT_STAMP_STEP = 100                            # nanoseconds between two rows


def make_int_stamps(seed, n=1500):
    """A random walk whose time column is whole numbers: nanoseconds since 1970 as
    64-bit integers, 100 nanoseconds apart. The one feature is the level before.

    Every timestamp is distinct and far above 2**53, where a 64-bit float holds
    only every 256th whole number: read as floats, these 1,500 instants are 587
    (D50; the third review, finding 3). The timestamps are then not unique, the
    previous-value baseline is gone, and a random walk is scored against the
    training mean, which is the outcome of F8 by another road. Read as the
    integers they are, this is `random_walk` again: both baselines, the previous
    value the one used, and skill about zero.
    """
    rng = random.Random(seed)
    level, rows = 0.0, []
    for k in range(n):
        new = level + rng.gauss(0, 1)
        rows.append([INT_STAMP_START + INT_STAMP_STEP * k, _f(level), _f(new)])
        level = float(_f(new))
    rng.shuffle(rows)
    return ["t", "level_prev", "level"], rows, ()


ENTITY_ROWS = 5                                 # rows of each entity in `entities`


def make_entities(seed, entities=300, per=ENTITY_ROWS, features=6):
    """Many rows for each entity, alike but for a little noise, and one label for
    the entity, drawn at random (D57).

    Nothing about one entity says anything of another, so the right skill is
    about zero. But the six measures identify the entity, and every row of an
    entity has its label: with a row of an entity on each side of the split a
    model scores by knowing the entity again. No two rows agree in every column
    (each value has its own noise), so the count of copies (D49) is zero here:
    the control is the entity column, `customer`, declared as `--group-col`.
    Rows are written in shuffled order.
    """
    rng = random.Random(seed)
    rows = []
    for e in range(entities):
        label = rng.choice(["no", "yes"])
        centre = [rng.gauss(0, 1) for _ in range(features)]
        for _ in range(per):
            rows.append([f"C{e + 1:04d}"] + [_f(c + rng.gauss(0, 0.05)) for c in centre]
                        + [label])
    rng.shuffle(rows)
    return ["customer"] + [f"m{i + 1}" for i in range(features)] + ["outcome"], rows, ()


def make_panel(seed, series=12, stamps=125):
    """Twelve random walks on the same business days, each at a level of its own;
    the one feature is the level the day before, of the same series (D57, D58).

    A panel: every date holds twelve rows. Against the training mean each
    series looks almost perfectly predictable, since a level resembles the one
    before (the real-data run measured 0.979 on five stocks). With the series
    declared as `--group-col`, each has its own previous value, the baseline is
    that, and skill is about zero. Rows are written in shuffled order.
    """
    rng = random.Random(seed)
    day, days = datetime.date(2021, 1, 4), []
    while len(days) < stamps:
        if day.weekday() < 5:
            days.append(day)
        day += datetime.timedelta(days=1)
    rows = []
    for s in range(series):
        level = float(_f(rng.gauss(0, 20)))
        for d in days:
            new = level + rng.gauss(0, 1)
            rows.append([d.isoformat(), f"S{s + 1:02d}", _f(level), _f(new)])
            level = float(_f(new))
    rng.shuffle(rows)
    return ["date", "series", "level_prev", "level"], rows, ()


# name -> (builder, target column). Seeds are SEED + position.
DATASETS = {
    "signal_clf": (lambda s: make_signal_clf(s), "outcome"),
    "signal_reg": (lambda s: make_signal_reg(s), "amount"),
    "noise_clf": (lambda s: make_noise(s, kind="clf"), "outcome"),
    "noise_reg": (lambda s: make_noise(s, kind="reg"), "amount"),
    "imbalanced": (lambda s: make_noise(s, n=2600, columns=6, share=0.05, region=True),
                   "outcome"),
    "leak": (lambda s: make_signal_clf(s, leak=True), "outcome"),
    "leak_reg": (lambda s: make_signal_reg(s, leak=True), "amount"),
    "binary_numeric": (lambda s: make_signal_clf(s, numeric_target=True), "outcome"),
    "ambiguous_target": (lambda s: make_ambiguous(s), "rating"),
    "messy": (lambda s: make_messy(s), "outcome"),
    "time_blocks": (lambda s: make_time_blocks(s), "regime"),
    "time_dayfirst": (lambda s: make_time_blocks(s, dayfirst=True), "regime"),
    "random_walk": (lambda s: make_random_walk(s), "level"),
    # Small tables for the full profile, which fits 18 configurations x 6 times.
    "noise_small": (lambda s: make_noise(s, n=400, columns=10, kind="clf"), "outcome"),
    "signal_clf_small": (lambda s: make_signal_clf(s, n=500), "outcome"),
    "signal_reg_small": (lambda s: make_signal_reg(s, n=500), "amount"),
    # Added last, so that the seed of every dataset above stays what it was.
    "near_leak": (lambda s: make_near_leak(s), "outcome"),
    "long_rows": (lambda s: make_long_rows(s), "outcome"),
    "flat_tail": (lambda s: make_flat_tail(s), "sales"),
    # Added last again (CP5), for the same reason.
    "dup_noise": (lambda s: make_dup_noise(s), "outcome"),
    "int_stamps": (lambda s: make_int_stamps(s), "level"),
    # Added last again (CP6), for the same reason.
    "entities": (lambda s: make_entities(s), "outcome"),
    "panel": (lambda s: make_panel(s), "level"),
}


def dataset_seed(name, base=SEED):
    return base + list(DATASETS).index(name)


def dataset_text(name, base=SEED):
    header, rows, blank_after = DATASETS[name][0](dataset_seed(name, base))
    lines = [",".join(header)]
    for i, row in enumerate(rows):
        lines.append(",".join(_quote(cell) for cell in row))
        if i in blank_after:
            lines.append("")
    return "\n".join(lines) + "\n"


def _quote(cell):
    text = str(cell)
    return '"' + text.replace('"', '""') + '"' if any(c in text for c in ',"\n') else text


_GATE_SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def write_dataset(name, folder, base=SEED):
    """Make sure one dataset is on disk, byte for byte. Returns the path.

    Generating all of them costs about 1.5 seconds, so a stamp beside each file
    records the gate source and the file's hash; a file that still matches its
    stamp is left alone, and anything else is regenerated.
    """
    path = Path(folder) / f"{name}.csv"
    stamp = path.with_suffix(".stamp")
    key = f"{_GATE_SOURCE}:{base}:"
    if path.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8") == \
            key + hashlib.sha256(path.read_bytes()).hexdigest():
        return path
    data = dataset_text(name, base).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(f".{os.getpid()}.tmp")
    scratch.write_bytes(data)
    os.replace(scratch, path)
    stamp.write_text(key + hashlib.sha256(data).hexdigest(), encoding="utf-8")
    return path


# ------------------------------------------------ the judge's own arithmetic --

def read_source(path):
    """Header and rows of a CSV as the gate counts them: blank lines are not rows."""
    with open(path, newline="", encoding="utf-8") as fh:
        table = [row for row in csv.reader(fh) if row]
    return table[0], table[1:]


def is_missing(cell):
    return cell.strip() in MISSING_TOKENS


def auc(labels, scores):
    """ROC AUC from ranks, ties sharing their average rank."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks, i = [0.0] * len(scores), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        raise CheckFailed("the holdout in predictions.csv holds a single class, so ROC AUC "
                          "cannot be recomputed")
    return (sum(r for r, y in zip(ranks, labels) if y) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def mse(truth, guess):
    return math.fsum((t - g) ** 2 for t, g in zip(truth, guess)) / len(truth)


def mean(values):
    return math.fsum(values) / len(values)


def cube_root_up(n):
    length = 1
    while length ** 3 < n:
        length += 1
    return length


def close(a, b, tolerance=1e-9):
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def same_cell(a, b):
    """A value in predictions.csv against the source cell it must equal."""
    a, b = str(a).strip(), str(b).strip()
    if a == b:
        return True
    try:
        return close(float(a), float(b), 1e-12)
    except ValueError:
        return False


def expected_time_split(stamps, gap=0, cap=FAST_CAP):
    """Which rows the time-ordered scheme must use (spec.md, steps 6 and 7).

    `stamps` maps row position -> sortable time value, usable rows only.
    """
    by_stamp = {}
    for row, stamp in stamps.items():
        by_stamp.setdefault(stamp, []).append(row)
    order = sorted(by_stamp)
    first, kept = len(order), 0
    while first > 0 and kept + len(by_stamp[order[first - 1]]) <= cap:
        first -= 1
        kept += len(by_stamp[order[first]])
    order = order[first:]                       # the most recent whole timestamps
    h, held = len(order), 0
    while h > 0 and 4 * held < kept:            # the latest holding at least 25%
        h -= 1
        held += len(by_stamp[order[h]])
    rows_of = lambda part: {r for s in part for r in by_stamp[s]}  # noqa: E731
    return {
        "capped": len(stamps) - kept,
        "holdout": rows_of(order[h:]), "gap": rows_of(order[h - gap:h]),
        "training": rows_of(order[:h - gap]),
        "holdout_start": order[h], "train_end": order[h - gap - 1],
        "holdout_stamps": len(order) - h,
    }


# ------------------------------------------------------------ tool runs --

API_SCRIPT = r'''
import json, sys
import pandas as pd
import mlprofessor

def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value

data, target, out = sys.argv[1:4]
result = mlprofessor.profile(data, target=target, profile="fast", seed=0, out_dir=out, n_jobs=1)
result.to_json(out + "/via_to_json.json")
frames = {}
for name in ("rows", "columns", "models", "warnings", "predictions"):
    frame = getattr(result, name)
    if not isinstance(frame, pd.DataFrame):
        raise SystemExit("ProfileResult.%s is %s, not a pandas DataFrame"
                         % (name, type(frame).__name__))
    frames[name] = plain(frame.to_dict(orient="records"))
print("@@RESULT@@" + json.dumps({"frames": frames, "headline": plain(dict(result.headline)),
                                 "to_dict": plain(result.to_dict())}))
'''

# run name -> dataset, extra command-line arguments, profile, n_jobs
RUNS = {
    "signal_clf": ("signal_clf", [], "fast", 1),
    "signal_reg": ("signal_reg", [], "fast", 1),
    "noise_clf": ("noise_clf", [], "fast", 1),
    "noise_reg": ("noise_reg", [], "fast", 1),
    "imbalanced": ("imbalanced", [], "fast", 1),
    "leak": ("leak", [], "fast", 1),
    "leak_reg": ("leak_reg", [], "fast", 1),
    "near_leak": ("near_leak", [], "fast", 1),
    "binary_numeric": ("binary_numeric", [], "fast", 1),
    "ambiguous_target": ("ambiguous_target", [], "fast", 1),
    # `messy` has a column of dates, `signup_date`, and its rows have no order in
    # time: the run says so (D58). Nothing else about it changes.
    "messy": ("messy", ["--no-time-order"], "fast", 1),
    "long_rows": ("long_rows", [], "fast", 1),
    "dup_noise": ("dup_noise", [], "fast", 1),
    "time_blocks": ("time_blocks", ["--time-col", "date"], "fast", 1),
    "time_blocks_gap": ("time_blocks", ["--time-col", "date", "--gap", "3"], "fast", 1),
    # With `date` excluded, the rows of one block of dates differ in nothing the
    # models see, so every holdout row has a copy among the training rows (500 of
    # 500 at the gate seed). The run is about the warning for an undeclared date
    # column, so it says the copies are meant: `--duplicates keep` (D49, D53). Since
    # D58 the warning is reached only when the caller has said that the rows have no
    # time order: `--no-time-order`.
    "time_blocks_undeclared": ("time_blocks", ["--duplicates", "keep", "--no-time-order"],
                               "fast", 1),
    # The same table and nothing said about time order: the run must stop (D58).
    # `--duplicates keep`, so that the copies rule is not what stops it: without
    # the new stop this run finishes, under a random split, at skill near 1.
    "time_blocks_unstated": ("time_blocks", ["--duplicates", "keep"], "fast", 1),
    "time_dayfirst": ("time_dayfirst", ["--time-col", "date"], "fast", 1),
    "random_walk": ("random_walk", ["--time-col", "date"], "fast", 1),
    "int_stamps": ("int_stamps", ["--time-col", "t"], "fast", 1),
    "flat_tail": ("flat_tail", ["--time-col", "date"], "fast", 1),
    "entities": ("entities", ["--group-col", "customer"], "fast", 1),
    "panel": ("panel", ["--time-col", "date", "--group-col", "series"], "fast", 1),
    "api_signal_clf": ("signal_clf", "api", "fast", 1),
    "full_noise_small": ("noise_small", [], "full", 1),
    "full_signal_clf_j1": ("signal_clf_small", [], "full", 1),
    "full_signal_clf_j4": ("signal_clf_small", [], "full", 4),
    "full_signal_reg_j1": ("signal_reg_small", [], "full", 1),
    "full_signal_reg_j4": ("signal_reg_small", [], "full", 4),
}
# Time limits per run. Measured on the reference: a fast run 3 to 6 seconds, a full
# run 9 to 17 alone and about 20 when eight run side by side. Two waves of hung fast
# runs must still end inside the per-edit hook's 150 seconds.
SLOW_RUN_SECONDS, FAST_RUN_SECONDS = 180, 60

# check id -> the runs it reads. Setup starts exactly these, side by side.
CHECK_RUNS = {
    "data.rows_reconcile": ["messy", "imbalanced"],
    "data.columns_accounted": ["messy"],
    "data.coercion_reported": ["messy"],
    "data.long_rows_stop": ["long_rows"],
    "data.holdout_copies_stop": ["dup_noise"],
    "models.all_accounted": ["signal_clf"],
    "task.binary_numeric": ["binary_numeric"],
    "task.ambiguous_stops": ["ambiguous_target"],
    "leak.excluded": ["leak"],
    "leak.control_kept": ["leak"],
    "leak.regression": ["leak_reg"],
    "leak.near_leak_warned": ["near_leak"],
    "model.signal_clf": ["signal_clf"],
    "model.signal_reg": ["signal_reg"],
    "model.noise_clf": ["noise_clf"],
    "model.noise_p_value": ["noise_clf"],
    "model.noise_reg": ["noise_reg"],
    "model.headline_recomputed": ["noise_clf", "signal_reg"],
    "model.metric_clf": ["imbalanced"],
    "model.selection_full": ["full_noise_small"],
    "time.order_respected": ["time_blocks"],
    "time.no_skill_on_blocks": ["time_blocks"],
    "time.gap_applied": ["time_blocks_gap"],
    "time.dayfirst_stops": ["time_dayfirst"],
    "time.persistence_baseline": ["random_walk"],
    "time.integer_keys_distinct": ["int_stamps"],
    "time.undeclared_warns": ["time_blocks_undeclared"],
    "time.undeclared_stops": ["time_blocks_unstated"],
    "time.panel_previous_value": ["panel"],
    "groups.kept_together": ["entities"],
    "groups.no_skill_on_entities": ["entities"],
    "time.flat_holdout_reported": ["flat_tail"],
    "uncertainty.reported": ["signal_clf", "time_blocks"],
    "result.matches_json": ["api_signal_clf"],
    "output.complete": ["signal_clf"],
    "output.deterministic": ["signal_clf", "api_signal_clf"],
    "output.same_at_any_n_jobs": ["full_signal_clf_j1", "full_signal_clf_j4",
                                  "full_signal_reg_j1", "full_signal_reg_j4"],
}
SLOW_CHECKS = {"model.selection_full", "output.same_at_any_n_jobs"}


def _selected_checks(ctx):
    """The check ids this invocation will run (gatekit does not pass them to setup)."""
    ids, argv, only = list(CHECK_RUNS), sys.argv[1:], None
    for i, arg in enumerate(argv):
        if arg == "--only" and i + 1 < len(argv):
            only = argv[i + 1]
        elif arg.startswith("--only="):
            only = arg.split("=", 1)[1]
    if only and "--selftest" not in argv:
        wanted = {part.strip() for part in only.split(",")}
        ids = [i for i in ids if i in wanted]
    return [i for i in ids if not (ctx.fast and i in SLOW_CHECKS)]


def _tool_env(n_jobs):
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if n_jobs == 1:
            env[name] = "1"             # single-threaded inside the gate
        else:
            env.pop(name, None)         # the tool itself must govern its threads (D17)
    return env


_START_LOCK = threading.Lock()
_last_start = [0.0]


def _spaced_start():
    """Keep process starts START_SPACING seconds apart (see the note at START_SPACING)."""
    with _START_LOCK:
        wait = _last_start[0] + START_SPACING - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_start[0] = time.monotonic()


def _launch(ctx, name):
    """Run the tool once. Worker-thread safe: touches only its own output folder."""
    dataset, extra, profile, n_jobs = RUNS[name]
    data = ctx.state["data"] / f"{dataset}.csv"
    out = ctx.state["runs_dir"] / name
    target = DATASETS[dataset][1]
    try:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
    except OSError as exc:
        return {"harness": f"cannot prepare the output folder {out}: {exc}"}
    python = sys.executable
    if extra == "api":
        cmd = [python, "-c", API_SCRIPT, str(data), target, str(out)]
    else:
        cmd = [python, "-m", "mlprofessor", "profile", "--data", str(data), "--target", target,
               "--out", str(out), "--profile", profile, "--seed", "0",
               "--n-jobs", str(n_jobs)] + list(extra)
    shown = " ".join(["python"] + cmd[1:2] + (["<script>"] if extra == "api" else []) +
                     [str(c) for c in cmd[(3 if extra == "api" else 2):]])
    limit = FAST_RUN_SECONDS if profile == "fast" else SLOW_RUN_SECONDS
    _spaced_start()
    try:
        proc = subprocess.run(cmd, cwd=str(ctx.state["deliverable"].parent), env=_tool_env(n_jobs),
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=limit)
        rc, stdout, stderr = proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        rc, stdout, stderr = None, "", f"the tool did not finish within {limit} seconds"
    except OSError as exc:
        return {"harness": f"could not start the interpreter {python}: {exc}"}
    for module in ("pandas", "numpy", "sklearn", "threadpoolctl", "scipy", "joblib"):
        if f"No module named '{module}'" in stderr:
            return {"harness": f"the interpreter {python} cannot import {module}; the modelling "
                               "stack recorded in CLAUDE.md is not installed"}
    return {"name": name, "dataset": dataset, "data": data, "out": out, "rc": rc,
            "stdout": stdout, "stderr": stderr, "cmd": shown, "target": target}


def run(ctx, name):
    """The outcome of one tool run, started now if setup did not already start it."""
    if ctx.state.get("fatal"):
        raise CheckFailed(ctx.state["fatal"])
    done = ctx.state["runs"]
    if name not in done:
        write_dataset(RUNS[name][0], ctx.state["data"])
        done[name] = _launch(ctx, name)
    result = done[name]
    if "harness" in result:
        ctx.harness_error(result["harness"])
    return result


def _tail(text, limit=900):
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def outputs(ctx, name):
    """A run that must have succeeded: its parsed profile.json and predictions.csv."""
    result = run(ctx, name)
    if "doc" in result:
        return result
    if result["rc"] is None:
        raise CheckFailed(
            f"the tool did not finish on dataset `{result['dataset']}`: {result['stderr']}"
            + (", on two attempts" if result.get("attempts") else "") + ". The reference "
            f"implementation takes a few seconds here.\n  command: {result['cmd']}")
    if result["rc"] != 0:
        raise CheckFailed(
            f"the tool exited {result['rc']} on dataset `{result['dataset']}` where it should "
            f"have succeeded.\n  command: {result['cmd']}\n  stderr: {_tail(result['stderr'])}")
    profile_path, predictions_path = result["out"] / "profile.json", result["out"] / "predictions.csv"
    for path in (profile_path, predictions_path):
        if not path.is_file():
            raise CheckFailed(f"the tool exited 0 on `{result['dataset']}` but did not write "
                              f"{path.name} into the --out folder ({result['cmd']})")
    text = profile_path.read_text(encoding="utf-8")
    bad_numbers = []
    try:
        doc = json.loads(text, parse_constant=lambda token: bad_numbers.append(token) or token)
    except ValueError as exc:
        raise CheckFailed(f"profile.json on `{result['dataset']}` is not valid JSON: {exc}")
    with open(predictions_path, newline="", encoding="utf-8") as fh:
        table = [row for row in csv.reader(fh) if row]
    if not table or table[0] != ["row", "y_true", "y_pred"]:
        raise CheckFailed(f"predictions.csv on `{result['dataset']}` must have exactly the "
                          f"columns row, y_true, y_pred; found {table[0] if table else 'nothing'}")
    predictions = []
    for line in table[1:]:
        try:
            position, guess = int(line[0]), float(line[2])
        except (ValueError, IndexError):
            raise CheckFailed(f"predictions.csv on `{result['dataset']}` has a row that is not "
                              f"(whole number, value, number): {line}")
        if not math.isfinite(guess):
            raise CheckFailed(f"predictions.csv on `{result['dataset']}` holds a non-finite "
                              f"y_pred at row {position}")
        predictions.append((position, line[1], guess))
    header, rows = read_source(result["data"])
    result.update(doc=doc, bad_numbers=bad_numbers, predictions=predictions, header=header,
                  source=rows, profile_text=text,
                  predictions_bytes=predictions_path.read_bytes())
    return result


def at(result, path, kind=None):
    """A value from profile.json by dotted path, with a failure that names the key."""
    node = result["doc"]
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise CheckFailed(f"profile.json on `{result['dataset']}` has no key `{path}` "
                              "(spec.md, 'profile.json')")
        node = node[part]
    if kind == "number":
        if isinstance(node, bool) or not isinstance(node, (int, float)) or not math.isfinite(node):
            raise CheckFailed(f"`{path}` in profile.json on `{result['dataset']}` is {node!r}, "
                              "not a finite number")
    elif kind is not None and not isinstance(node, kind):
        raise CheckFailed(f"`{path}` in profile.json on `{result['dataset']}` is {node!r}, "
                          f"expected {getattr(kind, '__name__', kind)}")
    return node


def records(result, key, fields):
    """A list-of-records part of profile.json, each record holding `fields`."""
    items = at(result, key, list)
    for item in items:
        if not isinstance(item, dict) or any(f not in item for f in fields):
            raise CheckFailed(f"every entry of `{key}` in profile.json must have the keys "
                              f"{', '.join(fields)}; found {item!r}")
    return items


def row_counts(result):
    return {r["item"]: r["count"] for r in records(result, "rows", ["item", "count"])}


def warnings_of(result, code):
    return [w for w in records(result, "warnings", ["code", "column", "message"])
            if w["code"] == code]


def band(ctx, value, name, label, hint):
    """A measured band, with the likely cause attached to the failure message."""
    low, high = BANDS[name]
    try:
        ctx.require_between(value, low, high, label)
    except CheckFailed as exc:
        raise CheckFailed(f"{exc}: {hint}") from None


def stopped(ctx, name, code, what):
    """A run that must have stopped with a named error and written no headline."""
    result = run(ctx, name)
    ctx.record(f"{name} exit code", result["rc"])
    ctx.require(result["rc"] not in (0, None),
                f"the tool exited {result['rc']} on `{result['dataset']}`; it must stop with the "
                f"named error `{code}` ({what})")
    ctx.require(code in result["stderr"],
                f"the tool stopped on `{result['dataset']}` without naming the error `{code}` "
                f"on stderr. stderr was: {_tail(result['stderr'], 400) or '(empty)'}")
    ctx.require(not (result["out"] / "profile.json").exists(),
                f"the tool named the error `{code}` on `{result['dataset']}` but still wrote "
                "profile.json: no headline may be written when the run stops")


def holdout_check(ctx, result, label):
    """Reconciliation rule 3: one prediction per holdout row, y_true from the source."""
    target = result["header"].index(result["target"])
    positions = [p for p, _, _ in result["predictions"]]
    ctx.require(len(set(positions)) == len(positions),
                f"predictions.csv on `{label}` lists a source row more than once")
    for position, truth, _ in result["predictions"]:
        ctx.require(0 <= position < len(result["source"]),
                    f"predictions.csv on `{label}` names row {position}; the source has rows 0 "
                    f"to {len(result['source']) - 1} (blank lines are not rows)")
        cell = result["source"][position][target]
        ctx.require(same_cell(truth, cell),
                    f"predictions.csv on `{label}`: y_true at row {position} is {truth!r} but "
                    f"the target in the source row is {cell!r}. `row` must be the zero-based "
                    "position in the loaded source, counted before any row is dropped")
    return positions


def recomputed_auc(result, positive):
    return auc([1 if truth.strip() == positive else 0 for _, truth, _ in result["predictions"]],
               [guess for _, _, guess in result["predictions"]])


def _runs_dir(ctx):
    """The folder this invocation's tool runs write into.

    A gate run on a build writes to verify/artifacts/runs, the folder a failure
    message and CLAUDE.md send the builder to. The self-test runs the reference
    and seventeen broken copies of it through the same checks; it writes to a folder
    per fixture under verify/artifacts/selftest_runs, so that after it `runs/`
    still holds the last gate run on the build and not a broken fixture's output
    (D46; second review, finding 6).
    """
    if "--selftest" in sys.argv[1:]:
        return ctx.artifacts / "selftest_runs" / (fixture_name(ctx.deliverable) or "unnamed")
    return ctx.artifacts / "runs"


@gate.setup
def prepare(ctx):
    for module in ("pandas", "numpy", "sklearn", "threadpoolctl"):
        if importlib.util.find_spec(module) is None:
            ctx.harness_error(f"the interpreter {sys.executable} cannot import {module}; the "
                              "modelling stack recorded in CLAUDE.md is not installed")
    data, runs_dir = ctx.artifacts / "data", _runs_dir(ctx)
    try:
        for folder in (data, runs_dir):
            folder.mkdir(parents=True, exist_ok=True)
        probe = ctx.artifacts / f".write-probe-{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        ctx.harness_error(f"verify/artifacts is not writable: {exc}")
    deliverable = Path(ctx.deliverable).resolve()
    ctx.state.update(data=data, runs_dir=runs_dir, deliverable=deliverable, runs={})
    if not deliverable.is_dir():
        ctx.state["fatal"] = (f"the package folder {deliverable} does not exist yet, so there is "
                              "nothing to run. Build the package (spec.md, 'Interface')")
        return
    if deliverable.name != "mlprofessor":
        ctx.harness_error(f"the deliverable must be a folder named mlprofessor, got {deliverable}")
    if not (deliverable / "__main__.py").is_file():
        ctx.state["fatal"] = (f"{deliverable} has no __main__.py, so `python -m mlprofessor` "
                              "cannot run (spec.md, 'Interface')")
        return
    needed = []
    for check_id in _selected_checks(ctx):
        needed += [name for name in CHECK_RUNS[check_id] if name not in needed]
    try:
        for dataset in {RUNS[name][0] for name in needed}:
            write_dataset(dataset, data)
    except OSError as exc:
        ctx.harness_error(f"could not write a synthetic dataset under {data}: {exc}")
    slow_first = sorted(needed, key=lambda name: RUNS[name][2] != "full")
    _launch_all(ctx, slow_first)
    done = ctx.state["runs"]
    held_up = [n for n in slow_first if "harness" not in done[n] and done[n]["rc"] is None]
    finished = [n for n in slow_first if done[n].get("rc") is not None]
    if held_up and finished:
        # Measured on this machine before D32: when eight interpreters started
        # together they sometimes all waited a minute or more on file reads, because
        # the interpreter's base sat on a mapped drive (spec.md section 9). It is a
        # local path now and the wait was not seen again in 60 rounds; the retry is
        # kept all the same. The tool finished other runs in time, so a run that did
        # not gets one more go before it is called a failure. If nothing finished,
        # nothing is retried.
        _launch_all(ctx, held_up)
        for name in held_up:
            done[name]["attempts"] = 2
    for result in done.values():
        if "harness" in result:
            ctx.harness_error(result["harness"])


def _launch_all(ctx, names):
    """Start the named runs side by side, at most PARALLEL at once."""
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        for name, result in zip(names, pool.map(lambda n: _launch(ctx, n), names)):
            ctx.state["runs"][name] = result


# ------------------------------------------------------------------ checks --

@gate.check("data.rows_reconcile", catches=["F6"],
            doc="every row count equals the gate's own count from the source, and they add up")
def data_rows_reconcile(ctx):
    for name in ("messy", "imbalanced"):
        result = outputs(ctx, name)
        target = result["header"].index(result["target"])
        loaded = len(result["source"])
        no_target = sum(is_missing(row[target]) for row in result["source"])
        usable = loaded - no_target
        capped = max(0, usable - FAST_CAP)
        holdout = -(-(usable - capped) // 4)
        want = {"loaded": loaded, "dropped_missing_target": no_target, "dropped_missing_time": 0,
                "dropped_gap": 0, "dropped_row_cap": capped,
                "training": usable - capped - holdout, "holdout": holdout}
        got = row_counts(result)
        for item, count in want.items():
            ctx.record(f"{name} rows {item}", got.get(item))
            ctx.require(got.get(item) == count,
                        f"rows.{item} on `{name}` is {got.get(item)!r}; the gate counts {count} "
                        f"from the source file ({loaded} rows, blank lines not counted; "
                        f"{no_target} with no target"
                        + (f"; the fast profile keeps {FAST_CAP}" if capped else "")
                        + "). A row is being lost, or dropped without an entry in the accounting")
        ctx.require(set(got) == set(want), f"rows on `{name}` must have exactly the items "
                    f"{', '.join(want)}; found {', '.join(map(str, got))}")
        dropped = sum(v for k, v in got.items() if k.startswith("dropped_"))
        ctx.require(got["loaded"] == dropped + got["training"] + got["holdout"],
                    f"rows on `{name}` do not add up: loaded {got['loaded']} != dropped "
                    f"{dropped} + training {got['training']} + holdout {got['holdout']}")
        positions = holdout_check(ctx, result, name)
        ctx.require(len(positions) == holdout,
                    f"predictions.csv on `{name}` has {len(positions)} rows; the holdout is "
                    f"{holdout} rows (a quarter of the usable rows, rounded up)")
        ctx.require(not any(is_missing(result["source"][p][target]) for p in positions),
                    f"predictions.csv on `{name}` includes a row whose target is missing")


MESSY_ROLES = {
    "f1": ("feature", None, "numeric"), "f2": ("feature", None, "numeric"),
    "seg": ("feature", None, "categorical"), "n1": ("feature", None, "numeric"),
    "spend": ("feature", None, "numeric"), "visits": ("feature", None, "numeric"),
    "const": ("excluded", "constant", None), "record_id": ("excluded", "id_like", None),
    "notes": ("excluded", "free_text", None), "signup_date": ("excluded", "datetime", None),
    "outcome": ("target", None, None),
    "revenue": ("excluded", "numeric_with_separators", None),
    "fee": ("excluded", "numeric_with_separators", None),
}
ROLE_HINTS = {
    "visits": "a whole-number measure with every value distinct but not in order is a feature, "
              "not an identifier (D19)",
    "record_id": "a whole-number column whose values are all distinct and consecutive is an "
                 "identifier (D19)",
    "spend": "a text column in which at least 90% of the values parse as numbers is numeric; "
             "its eight text cells are words, none of them a number with thousands separators",
    "n1": "`n/a` is a missing value on load, so this column is an ordinary numeric column",
    "revenue": "every value is a number written with thousands separators (`1,200.50`) or, "
               "below 1,000, a plain number. Such a column is never parsed, and is excluded "
               "with its own reason ahead of the id-like and free-text rules (D28)",
    "fee": "six of its cells are numbers written with thousands separators and the rest are "
           "plain numbers. One such cell is enough to exclude the column, ahead of the "
           "coercion rule: kept as numeric it would be missing its six largest values (D34)",
}


@gate.check("data.columns_accounted", catches=["F6b"],
            doc="every source column appears exactly once, with the expected role and reason")
def data_columns_accounted(ctx):
    result = outputs(ctx, "messy")
    columns = records(result, "columns", ["name", "role", "reason", "kind"])
    names = [c["name"] for c in columns]
    ctx.require(sorted(names) == sorted(result["header"]),
                f"the column accounting on `messy` lists {names}; the source has "
                f"{result['header']}. Every source column must appear exactly once")
    for column in columns:
        role, reason, kind = MESSY_ROLES[column["name"]]
        ctx.record(f"messy role {column['name']}", f"{column['role']}/{column['reason']}")
        hint = ROLE_HINTS.get(column["name"], "see spec.md, 'Column roles'")
        ctx.require((column["role"], column["reason"]) == (role, reason),
                    f"column `{column['name']}` on `messy` has role {column['role']!r} and "
                    f"reason {column['reason']!r}; expected {role!r} and {reason!r}: {hint}")
        if kind:
            ctx.require(column["kind"] == kind,
                        f"feature `{column['name']}` on `messy` has kind {column['kind']!r}; "
                        f"expected {kind!r}: {hint}")
    separated = sorted(str(w["column"]) for w in warnings_of(result, "numeric_with_separators"))
    ctx.record("messy numeric_with_separators", ",".join(separated))
    for name in ("revenue", "fee"):
        ctx.require(name in separated,
                    f"no warning with code `numeric_with_separators` names the column `{name}` "
                    "on `messy`: a column excluded because it holds numbers written with "
                    "thousands separators must be reported, so the caller knows to parse it "
                    "(D28, D34)")
    ctx.require(separated == ["fee", "revenue"],
                f"`numeric_with_separators` warnings on `messy` name {separated}; the warning "
                "has one form, one per column excluded with that reason: `fee` and `revenue` "
                "(D34). A numeric column whose text cells are words, such as `spend`, is not "
                "named")
    band(ctx, at(result, "headline.skill", "number"), "messy", "skill on messy",
         "the signal is still there under the mess; a wrong column role or lost rows remove it")


@gate.check("data.coercion_reported", catches=["F6"],
            doc="text cells in a numeric column are counted as coerced, n/a cells as missing")
def data_coercion_reported(ctx):
    result = outputs(ctx, "messy")
    keys = ["name", "role", "kind", "n_missing", "n_coerced", "n_imputed"]
    columns = {c["name"]: c for c in records(result, "columns", keys)}
    target = result["header"].index("outcome")
    for name in ("spend", "n1", "f2", "seg"):
        ctx.require(name in columns, f"column `{name}` is absent from the accounting on `messy`")
        cells = [(row[result["header"].index(name)], row[target]) for row in result["source"]]
        missing = sum(is_missing(cell) for cell, _ in cells)
        coerced = 0
        if name == "spend":
            for cell, _ in cells:
                try:
                    float(cell) if not is_missing(cell) else 0.0
                except ValueError:
                    coerced += 1
        imputed = sum(1 for cell, y in cells if not is_missing(y)
                      and (is_missing(cell) or (name == "spend" and not _parses(cell))))
        got = columns[name]
        for key, want, what in (("n_missing", missing, "blank or n/a cells"),
                                ("n_coerced", coerced, "text cells that are not numbers"),
                                ("n_imputed", imputed, "missing or coerced cells in rows that "
                                                       "reach a model")):
            ctx.record(f"messy {name} {key}", got[key])
            ctx.require(got[key] == want,
                        f"`{name}` on `messy` reports {key} = {got[key]!r}; the gate counts "
                        f"{want} {what} in the source")
    spend = columns["spend"]
    ctx.require(spend["role"] == "feature" and spend["kind"] == "numeric",
                f"`spend` on `messy` has role {spend['role']!r} and kind {spend['kind']!r}: a "
                "numeric column with a few text cells must stay a numeric feature, not become "
                "all-missing or a set of categories")
    ctx.require(any(w["column"] == "spend" for w in warnings_of(result, "values_coerced")),
                "no warning with code `values_coerced` names the column `spend` on `messy`: "
                "values turned into missing must be reported")
    ctx.require(not any(w["column"] == "spend"
                        for w in warnings_of(result, "numeric_with_separators")),
                "a warning with code `numeric_with_separators` names the column `spend` on "
                "`messy`. None of its eight text cells is a number with thousands separators "
                "(`abc`, `tbc`, `12 units`, `pending`, `unknown`, `-`), so it is an ordinary "
                "case of coercion: kept as numeric, with `values_coerced` alone (D34)")


def _parses(cell):
    try:
        float(cell)
        return True
    except ValueError:
        return False


@gate.check("data.long_rows_stop", catches=["F6d"],
            doc="data rows one field longer than the header stop with the named input error")
def data_long_rows_stop(ctx):
    stopped(ctx, "long_rows", "input_error",
            "every data row has nine fields under a header of eight names: a row number leads "
            "each row and the header does not name it. Data rows with more fields than the "
            "header are an input error (D22). pandas reads this file without complaint: it "
            "takes the leading field as the index, here a `RangeIndex` like the default one, "
            "and that column is lost with nothing said. The index cannot show the fault")


@gate.check("data.holdout_copies_stop", catches=["F9"],
            doc="noise with every row present twice stops with the named copies error, and "
                "writes no headline")
def data_holdout_copies_stop(ctx):
    stopped(ctx, "dup_noise", "holdout_copies_in_training",
            "every row of this table is present twice, so about three quarters of the holdout "
            "rows have a copy among the training rows: the same value in every feature column "
            "and the same target. A model scores on those rows by recalling them, and pure "
            "noise is reported as predictable. Under the random scheme the run stops when more "
            "than 1% of the holdout rows, and at least 10 of them, have such a copy, unless "
            "`--duplicates keep` is given (D49, D53; spec.md step 8)")


@gate.check("models.all_accounted", catches=["F6c"],
            doc="the models table lists every family of the profile, each with a status")
def models_all_accounted(ctx):
    result = outputs(ctx, "signal_clf")
    models = records(result, "models", ["config", "family", "status", "error", "fold_scores",
                                        "cv_mean", "chosen"])
    families = sorted({m["family"] for m in models})
    ctx.record("signal_clf families", ",".join(families))
    ctx.require(families == ["linear", "tree"],
                f"the fast profile runs the families linear and tree; the models table on "
                f"`signal_clf` lists {families}. A family that fails must stay in the table "
                "with status `failed`, never be dropped")
    for m in models:
        ctx.require(m["status"] in ("fitted", "failed"),
                    f"model {m['config']} has status {m['status']!r}; expected fitted or failed")
        ctx.require(m["status"] == "fitted" or m["error"],
                    f"model {m['config']} failed without an error text")
        ctx.require(m["status"] == "failed" or (isinstance(m["fold_scores"], list)
                                                 and len(m["fold_scores"]) == 5),
                    f"model {m['config']} must carry one score per fold (5)")
    chosen = [m for m in models if m["chosen"]]
    ctx.require(len(chosen) == 1 and chosen[0]["status"] == "fitted",
                f"exactly one fitted configuration must be marked chosen; found "
                f"{[m['config'] for m in chosen]}")
    ctx.require(chosen[0]["config"] == at(result, "headline.model"),
                "headline.model is not the configuration marked chosen in the models table")


@gate.check("task.binary_numeric", catches=["F5"],
            doc="a target stored as the integers 0 and 1 is classification, decided by the rule")
def task_binary_numeric(ctx):
    result = outputs(ctx, "binary_numeric")
    kind, how = at(result, "task.type"), at(result, "task.decided_by")
    metric = at(result, "headline.metric")
    ctx.record("binary_numeric task", f"{kind}/{how}/{metric}")
    ctx.require(kind == "classification",
                f"task on `binary_numeric` is {kind!r}: a target with two distinct values is "
                "classification whatever its type (D4). It was read from the column's type")
    ctx.require(how == "rule", f"task.decided_by is {how!r}; the task was not declared, so the "
                               "rule decided it")
    ctx.require(at(result, "task.n_distinct") == 2, "task.n_distinct must be 2 on `binary_numeric`")
    ctx.require(metric == "roc_auc", f"headline.metric is {metric!r} on a classification task; "
                                     "expected roc_auc")


@gate.check("task.ambiguous_stops", catches=["F5"],
            doc="an integer target with 7 distinct values stops with the named task error")
def task_ambiguous_stops(ctx):
    stopped(ctx, "ambiguous_target", "task_ambiguous",
            "a numeric target with 7 distinct values is neither two classes nor more than 20 "
            "values, so the tool must not guess (D4)")


@gate.check("leak.excluded", catches=["F2"],
            doc="both leak columns are excluded with their reasons; skill is not near 1")
def leak_excluded(ctx):
    result = outputs(ctx, "leak")
    columns = {c["name"]: c for c in records(result, "columns", ["name", "role", "reason",
                                                                 "leak_rule"])}
    skill = at(result, "headline.skill", "number")
    for name, rule, why in (("outcome_copy", "predicts_target",
                             "it is a noisy copy of the target"),
                            ("settlement_amount", "missing_by_class",
                             "it is filled in only where the outcome is yes")):
        got = columns.get(name, {})
        ctx.record(f"leak {name}", f"{got.get('role')}/{got.get('reason')}/{got.get('leak_rule')}")
        ctx.require((got.get("role"), got.get("reason")) == ("excluded", "suspected_leak"),
                    f"`{name}` on `leak` has role {got.get('role')!r} and reason "
                    f"{got.get('reason')!r}: {why}, so it must be excluded as a suspected leak. "
                    f"Skill is {skill:.4f}; with a leak in the features it is the leak that is "
                    "being reported as predictability")
        ctx.require(got.get("leak_rule") == rule,
                    f"`{name}` on `leak` is excluded with leak_rule {got.get('leak_rule')!r}; "
                    f"expected {rule!r} ({why})")
        ctx.require(any(w["column"] == name for w in warnings_of(result, "suspected_leak")),
                    f"no warning with code `suspected_leak` names `{name}` on `leak`")
    band(ctx, skill, "signal_clf", "skill on leak",
         "with both leaks excluded this is the skill of `signal_clf`; near 1 means a leak "
         "is still a feature")


@gate.check("leak.control_kept",
            why="a screen that flags every strong column would pass leak.excluded while "
                "throwing real signal away (D11: merely predictive is not a leak)",
            doc="a strongly predictive but legitimate column stays a feature, with no warning")
def leak_control_kept(ctx):
    result = outputs(ctx, "leak")
    got = {c["name"]: c for c in records(result, "columns", ["name", "role", "reason"])} \
        .get("strong_feature", {})
    ctx.record("leak strong_feature", f"{got.get('role')}/{got.get('reason')}")
    ctx.require(got.get("role") == "feature",
                f"`strong_feature` on `leak` has role {got.get('role')!r} (reason "
                f"{got.get('reason')!r}): it is strongly predictive but legitimate, and must "
                "stay a feature. The leak cut-off is too low or is not measured as spec.md says")
    # Measured, 30 data seeds: its single-column skill is 0.61 to 0.75 (mean 0.66,
    # sd 0.03), the highest of any legitimate column in any dataset of this gate,
    # against a warning cut-off of 0.90 (spec.md 10.2).
    warned = [w for w in warnings_of(result, "strong_single_column")
              if w["column"] == "strong_feature"]
    ctx.record("leak strong_feature warned", len(warned))
    ctx.require(not warned,
                "`strong_feature` on `leak` carries a `strong_single_column` warning. On its "
                "own it predicts the outcome with skill of about 0.65, far below the warning "
                "cut-off of 0.90 (D30): the cut-off is too low, or the skill is not measured "
                "as spec.md says (the pinned tree, over the selection folds, training rows "
                "only)")


@gate.check("leak.regression", catches=["F2"],
            doc="a rescaled copy of a numeric target is excluded and listed")
def leak_regression(ctx):
    result = outputs(ctx, "leak_reg")
    got = {c["name"]: c for c in records(result, "columns", ["name", "role", "reason",
                                                             "leak_rule"])} \
        .get("amount_scaled", {})
    skill = at(result, "headline.skill", "number")
    ctx.record("leak_reg amount_scaled", f"{got.get('role')}/{got.get('reason')}")
    ctx.require((got.get("role"), got.get("reason"), got.get("leak_rule")) ==
                ("excluded", "suspected_leak", "predicts_target"),
                f"`amount_scaled` on `leak_reg` has role {got.get('role')!r}, reason "
                f"{got.get('reason')!r}, leak_rule {got.get('leak_rule')!r}: it is the target "
                f"rescaled with a little noise and must be excluded (skill is {skill:.4f})")
    ctx.require(any(w["column"] == "amount_scaled" for w in warnings_of(result, "suspected_leak")),
                "no warning with code `suspected_leak` names `amount_scaled` on `leak_reg`")
    band(ctx, skill, "signal_reg", "skill on leak_reg",
         "with the copy excluded this is the skill of `signal_reg`")


@gate.check("leak.near_leak_warned", catches=["F2"],
            doc="a copy of the target with 2% of its labels flipped stays a feature, with a warning")
def leak_near_leak_warned(ctx):
    # No score is asserted here: with `outcome_noisy` among the features the headline
    # is that column, which is the reason for the warning. Measured, 30 data seeds: its
    # single-column skill is 0.938 to 0.960 (mean 0.949, sd 0.005), between the warning
    # cut-off of 0.90 and the exclusion cut-off of 0.98; the next strongest column of
    # this table, `f1`, is at 0.37 to 0.49.
    result = outputs(ctx, "near_leak")
    got = {c["name"]: c for c in records(result, "columns", ["name", "role", "reason",
                                                             "leak_rule"])} \
        .get("outcome_noisy", {})
    ctx.record("near_leak outcome_noisy", f"{got.get('role')}/{got.get('reason')}")
    ctx.require(got.get("role") == "feature",
                f"`outcome_noisy` on `near_leak` has role {got.get('role')!r} (reason "
                f"{got.get('reason')!r}, leak_rule {got.get('leak_rule')!r}): it agrees with "
                "the outcome on 98% of the rows, so its single-column skill is about 0.95, "
                "below the exclusion cut-off of 0.98. It stays a feature and is warned about "
                "(D30); lowering the exclusion cut-off would remove legitimately strong columns")
    warned = sorted(str(w["column"]) for w in warnings_of(result, "strong_single_column"))
    ctx.record("near_leak strong_single_column", ",".join(warned))
    ctx.require("outcome_noisy" in warned,
                "no warning with code `strong_single_column` names `outcome_noisy` on "
                "`near_leak`. The column is the outcome with 2% of its labels flipped: on its "
                "own it predicts the target with skill of about 0.95, at least the warning "
                "cut-off of 0.90 and below 0.98, so it is kept and the caller is told (D30). "
                "The leak screen is missing, or it has no warning tier")
    ctx.require(warned == ["outcome_noisy"],
                f"`strong_single_column` warnings on `near_leak` name {warned}; only "
                "`outcome_noisy` is at or above the warning cut-off of 0.90 (the next "
                "strongest column, `f1`, is at about 0.43). One warning per column, and "
                "none for an ordinary feature")


@gate.check("model.signal_clf",
            why="proves the tool can find skill at all, so that 'none' elsewhere means something",
            doc="clear skill where a binary target depends on three features")
def model_signal_clf(ctx):
    result = outputs(ctx, "signal_clf")
    band(ctx, at(result, "headline.skill", "number"), "signal_clf", "skill on signal_clf",
         "the outcome depends on f1, f2 and seg. Too low: the model is not fitted on them, or "
         "the positive class is inverted. Too high: something leaks")


@gate.check("model.signal_reg",
            why="proves the tool can find skill at all, so that 'none' elsewhere means something",
            doc="clear skill where a numeric target is a known function of two features")
def model_signal_reg(ctx):
    result = outputs(ctx, "signal_reg")
    band(ctx, at(result, "headline.skill", "number"), "signal_reg", "skill on signal_reg",
         "amount = 2.0*f1 - 1.2*f2 + noise. Too low: the model is not fitted on them, the "
         "worst configuration was chosen (squared error: lower is better), or the baseline "
         "is wrong")


@gate.check("model.noise_clf", catches=["F1"],
            doc="no skill where the target is a coin flip")
def model_noise_clf(ctx):
    result = outputs(ctx, "noise_clf")
    band(ctx, at(result, "headline.skill", "number"), "noise_clf", "skill on noise_clf",
         "the target is a coin flip, so honest skill is about zero. The chosen model was "
         "refitted on rows that include the holdout, or a fitted step saw the holdout")


@gate.check("model.noise_p_value", catches=["F1"],
            doc="a coin flip is not reported as significant")
def model_noise_p_value(ctx):
    result = outputs(ctx, "noise_clf")
    p_value = at(result, "headline.p_value", "number")
    ctx.record("p-value on noise_clf", p_value)
    ctx.require(p_value >= P_VALUE_FLOOR,
                f"the permutation p-value on `noise_clf` is {p_value:.4g}, at its floor of "
                "1/201: the tool reports a coin flip as significant. The holdout predictions "
                "were made by a model that had seen the holdout targets")


@gate.check("model.noise_reg", catches=["F1"],
            doc="no positive skill where a numeric target is independent of every feature")
def model_noise_reg(ctx):
    result = outputs(ctx, "noise_reg")
    band(ctx, at(result, "headline.skill", "number"), "noise_reg", "skill on noise_reg",
         "the target is independent of all 300 columns, so honest skill is below zero (a "
         "linear model pays for 300 useless columns). Above the band: the chosen model "
         "was refitted on rows that include the holdout. Far below: the worst configuration "
         "was chosen (squared error: lower is better)")


def _not_a_cv_score(ctx, result, score, label):
    models = records(result, "models", ["config", "cv_mean", "fold_scores"])
    means = [m["cv_mean"] for m in models if isinstance(m["cv_mean"], (int, float))]
    folds = [s for m in models for s in (m["fold_scores"] or []) if isinstance(s, (int, float))]
    ctx.require(means, f"no configuration on `{label}` has a cross-validation mean")
    for name, values in (("best mean cross-validation score", (max(means), min(means))),
                         ("best single fold score", (max(folds), min(folds)))):
        ctx.require(not any(close(score, v, 1e-12) for v in values),
                    f"headline.score on `{label}` ({score:.6g}) equals the {name} of the models "
                    "tried: the headline must be the chosen model's score on the holdout, never "
                    "a cross-validation score")


@gate.check("model.headline_recomputed", catches=["F3", "F4"],
            doc="the gate recomputes the metric and the skill from predictions.csv and the source")
def model_headline_recomputed(ctx):
    result = outputs(ctx, "noise_clf")
    holdout_check(ctx, result, "noise_clf")
    score = at(result, "headline.score", "number")
    mine = recomputed_auc(result, "yes")
    ctx.record("recomputed AUC on noise_clf", mine)
    ctx.require(at(result, "headline.metric") == "roc_auc",
                f"headline.metric on `noise_clf` is {at(result, 'headline.metric')!r}; a "
                "classification headline is roc_auc, never accuracy")
    ctx.require(close(score, mine),
                f"headline.score on `noise_clf` is {score:.6g}; ROC AUC recomputed by the gate "
                f"from predictions.csv is {mine:.6g}. The headline is not the holdout score of "
                "the predictions written (a cross-validation score, or another metric)")
    ctx.require(close(at(result, "headline.skill", "number"), 2 * mine - 1),
                f"headline.skill on `noise_clf` is {at(result, 'headline.skill'):.6g}; "
                f"2 x AUC - 1 is {2 * mine - 1:.6g}")
    _not_a_cv_score(ctx, result, score, "noise_clf")

    result = outputs(ctx, "signal_reg")
    held = holdout_check(ctx, result, "signal_reg")
    target = result["header"].index("amount")
    truth = [float(result["source"][p][target]) for p in held]
    in_holdout = set(held)
    training = [float(row[target]) for i, row in enumerate(result["source"]) if i not in in_holdout]
    mine = mse(truth, [guess for _, _, guess in result["predictions"]])
    base = mse(truth, [mean(training)] * len(truth))
    ctx.record("recomputed squared error on signal_reg", mine)
    score = at(result, "headline.score", "number")
    ctx.require(at(result, "headline.metric") == "mse",
                f"headline.metric on `signal_reg` is {at(result, 'headline.metric')!r}; "
                "expected mse")
    ctx.require(close(score, mine),
                f"headline.score on `signal_reg` is {score:.6g}; mean squared error recomputed "
                f"by the gate from predictions.csv is {mine:.6g}. The headline is not the "
                "holdout score of the predictions written")
    ctx.require(at(result, "headline.baseline") == "training_mean"
                and close(at(result, "headline.baseline_score", "number"), base),
                f"the baseline on `signal_reg` is {at(result, 'headline.baseline')!r} with score "
                f"{at(result, 'headline.baseline_score')!r}; the squared error of the training "
                f"mean on the holdout, recomputed by the gate, is {base:.6g}")
    ctx.require(close(at(result, "headline.skill", "number"), 1 - mine / base),
                f"headline.skill on `signal_reg` is {at(result, 'headline.skill'):.6g}; "
                f"1 - error / baseline error is {1 - mine / base:.6g}")
    _not_a_cv_score(ctx, result, score, "signal_reg")


@gate.check("model.metric_clf", catches=["F4"],
            doc="on a 95/5 target the metric is ROC AUC against 0.5, not accuracy")
def model_metric_clf(ctx):
    result = outputs(ctx, "imbalanced")
    metric, baseline = at(result, "headline.metric"), at(result, "headline.baseline_score")
    score = at(result, "headline.score", "number")
    labels = [truth.strip() for _, truth, _ in result["predictions"]]
    majority = max(labels.count("yes"), labels.count("no")) / len(labels)
    ctx.record("imbalanced metric", f"{metric}/{baseline}")
    ctx.require(metric == "roc_auc",
                f"headline.metric on `imbalanced` is {metric!r}: on a rare class accuracy "
                "rewards predicting the majority; the headline is roc_auc (D5)")
    ctx.require(baseline == 0.5, f"headline.baseline_score on `imbalanced` is {baseline!r}; "
                                 "the ROC AUC baseline is 0.5")
    ctx.require(abs(score - majority) > MAJORITY_MARGIN,
                f"headline.score on `imbalanced` is {score:.4f}, within {MAJORITY_MARGIN} of the "
                f"majority share {majority:.4f}: that is accuracy from predicting the majority, "
                "presented as predictability")
    band(ctx, at(result, "headline.skill", "number"), "imbalanced", "skill on imbalanced",
         "the target is noise with 5% positives, so honest skill is about zero")
    ctx.require(close(score, recomputed_auc(result, "yes")),
                "headline.score on `imbalanced` is not the ROC AUC of predictions.csv")


@gate.check("model.selection_full", catches=["F3"], slow=True,
            doc="full profile on noise: the headline is the holdout score, not the best CV score")
def model_selection_full(ctx):
    result = outputs(ctx, "full_noise_small")
    models = records(result, "models", ["config", "family", "status", "cv_mean"])
    families = {}
    for m in models:
        families[m["family"]] = families.get(m["family"], 0) + 1
    ctx.record("full profile configurations", len(models))
    ctx.require(sorted(families) == ["boosting", "forest", "linear"],
                f"the full profile runs the families linear, forest and boosting; the models "
                f"table lists {sorted(families)}")
    ctx.require(all(1 <= n <= 8 for n in families.values()) and len(models) <= 24,
                f"each family has at most 8 configurations, 24 in all (D16); found {families}")
    score = at(result, "headline.score", "number")
    mine = recomputed_auc(result, "yes")
    ctx.require(close(score, mine),
                f"headline.score under the full profile is {score:.6g}; ROC AUC recomputed from "
                f"predictions.csv is {mine:.6g}. With {len(models)} configurations tried, the "
                "best cross-validation score is optimistic and must not be the headline")
    _not_a_cv_score(ctx, result, score, "noise_small (full profile)")
    band(ctx, at(result, "headline.skill", "number"), "selection_full",
         "skill on noise_small (full profile)",
         "the target is a coin flip; skill outside the band means the holdout was used in "
         "fitting or in choosing the model")


def _time_expectation(result, gap):
    time_at, target = result["header"].index("date"), result["header"].index(result["target"])
    stamps = {i: datetime.date.fromisoformat(row[time_at])
              for i, row in enumerate(result["source"])
              if not is_missing(row[target]) and not is_missing(row[time_at])}
    return expected_time_split(stamps, gap=gap)


def _stamp(result, key):
    text = at(result, key, str)
    try:
        return datetime.datetime.fromisoformat(text).date()
    except ValueError:
        raise CheckFailed(f"`{key}` is {text!r}; a timestamp is written in ISO 8601 form")


@gate.check("time.order_respected", catches=["F7"],
            doc="with a time column the holdout is exactly the latest rows, cut at a whole date")
def time_order_respected(ctx):
    result = outputs(ctx, "time_blocks")
    want = _time_expectation(result, gap=0)
    ctx.require(at(result, "validation.scheme") == "time_ordered",
                f"validation.scheme is {at(result, 'validation.scheme')!r} although --time-col "
                "was given")
    train_end, holdout_start = _stamp(result, "validation.train_end"), \
        _stamp(result, "validation.holdout_start")
    ctx.record("time_blocks holdout_start", holdout_start.isoformat())
    ctx.require(train_end < holdout_start,
                f"the last training timestamp ({train_end}) is not earlier than the first "
                f"holdout timestamp ({holdout_start}): training rows are at or after the "
                "holdout, so the split ignores time order")
    held = set(holdout_check(ctx, result, "time_blocks"))
    ctx.require(held == want["holdout"],
                f"the holdout on `time_blocks` is {len(held)} rows; the gate expects exactly the "
                f"{len(want['holdout'])} source rows dated {want['holdout_start']} or later "
                f"({len(held - want['holdout'])} unexpected, {len(want['holdout'] - held)} "
                "missing). The holdout must be the latest whole timestamps that together hold "
                "at least 25% of the rows, the smallest such set; the rows were shuffled, or "
                "the file order was used instead of the time column")
    ctx.require(holdout_start == want["holdout_start"] and train_end == want["train_end"],
                f"validation reports train_end {train_end} and holdout_start {holdout_start}; "
                f"the gate computes {want['train_end']} and {want['holdout_start']}")
    got = row_counts(result)
    for item, count in (("dropped_row_cap", want["capped"]), ("training", len(want["training"])),
                        ("holdout", len(want["holdout"])), ("dropped_gap", 0)):
        ctx.record(f"time_blocks rows {item}", got.get(item))
        ctx.require(got.get(item) == count,
                    f"rows.{item} on `time_blocks` is {got.get(item)!r}; the gate counts {count}. "
                    f"Under time order the fast profile keeps the most recent whole timestamps "
                    f"holding at most {FAST_CAP} rows, and counts the rest as dropped_row_cap")
    share = at(result, "validation.holdout_share", "number")
    ctx.require(share >= 0.25 and close(share, len(held) / (got["training"] + got["holdout"]
                                                             + got["dropped_gap"])),
                f"validation.holdout_share is {share:.4f}; it must be holdout rows over "
                "training + holdout + gap rows, and at least 0.25")


@gate.check("time.no_skill_on_blocks", catches=["F7"],
            doc="no skill where only a shuffled split could find any")
def time_no_skill_on_blocks(ctx):
    result = outputs(ctx, "time_blocks")
    band(ctx, at(result, "headline.skill", "number"), "time_blocks", "skill on time_blocks",
         "the label is random per block of 5 dates and the features only identify the block, "
         "so nothing learned on earlier dates helps on later ones. Skill here means rows of "
         "the same block are on both sides of the split: it was shuffled")


@gate.check("time.gap_applied", catches=["F7"],
            doc="--gap 3 removes exactly the 3 timestamps before the holdout from training")
def time_gap_applied(ctx):
    result = outputs(ctx, "time_blocks_gap")
    want = _time_expectation(result, gap=3)
    got = row_counts(result)
    ctx.record("time_blocks gap rows", got.get("dropped_gap"))
    ctx.require(got.get("dropped_gap") == len(want["gap"]),
                f"rows.dropped_gap with --gap 3 is {got.get('dropped_gap')!r}; the gate counts "
                f"{len(want['gap'])} rows in the 3 distinct timestamps before the holdout")
    ctx.require(got.get("training") == len(want["training"]),
                f"rows.training with --gap 3 is {got.get('training')!r}; the gate counts "
                f"{len(want['training'])} rows before the gap. The gap rows must leave the "
                "training side")
    train_end = _stamp(result, "validation.train_end")
    ctx.require(train_end == want["train_end"],
                f"validation.train_end with --gap 3 is {train_end}; the last training timestamp "
                f"must be {want['train_end']}, three distinct timestamps short of the holdout "
                f"({want['holdout_start']}): a row inside the gap is still a training row")
    ctx.require(set(holdout_check(ctx, result, "time_blocks (gap 3)")) == want["holdout"],
                "the holdout changed when --gap 3 was given: the gap comes off the training "
                "side only")
    ctx.require(at(result, "validation.gap") == 3, "validation.gap must record the gap used (3)")
    folds = records(result, "validation.folds", ["fold", "training_rows", "validation_rows",
                                                 "gap_rows"])
    ctx.require(len(folds) == 5 and all(f["gap_rows"] >= 3 for f in folds),
                "every one of the five folds must report at least 3 rows removed by the gap "
                "before its validation block (validation.folds[].gap_rows): the gap applies "
                "to the folds as well as to the holdout")


@gate.check("time.dayfirst_stops", catches=["F7b"],
            doc="a day/month/year time column stops with the named time-column error")
def time_dayfirst_stops(ctx):
    stopped(ctx, "time_dayfirst", "time_column_invalid",
            "a text time column that is not ISO 8601 with the year first is never guessed "
            "(D18): 01/12/2023 read month-first puts the wrong rows in the holdout")


@gate.check("time.persistence_baseline", catches=["F8"],
            doc="on a random walk the baseline is the previous value and skill is about zero")
def time_persistence_baseline(ctx):
    result = outputs(ctx, "random_walk")
    held = holdout_check(ctx, result, "random_walk")
    time_at, target = result["header"].index("date"), result["header"].index("level")
    order = sorted(range(len(result["source"])), key=lambda i: result["source"][i][time_at])
    level = {i: float(result["source"][i][target]) for i in order}
    before = {row: order[k - 1] for k, row in enumerate(order) if k}
    in_holdout = set(held)
    truth = [level[p] for p in held]
    training_mean = mean([level[i] for i in order if i not in in_holdout])
    base_mean = mse(truth, [training_mean] * len(truth))
    base_previous = mse(truth, [level[before[p]] for p in held])
    mine = mse(truth, [guess for _, _, guess in result["predictions"]])
    baselines = {b["name"]: b for b in records(result, "baselines", ["name", "score", "used"])}
    ctx.record("random_walk baseline", at(result, "headline.baseline"))
    ctx.require({"training_mean", "previous_value"} <= set(baselines),
                f"baselines on `random_walk` lists {sorted(baselines)}; with unique timestamps "
                "both training_mean and previous_value are scored and reported (D10)")
    for name, want in (("training_mean", base_mean), ("previous_value", base_previous)):
        ctx.require(isinstance(baselines[name]["score"], (int, float))
                    and close(baselines[name]["score"], want, 1e-7),
                    f"the {name} baseline on `random_walk` scores {baselines[name]['score']!r}; "
                    f"recomputed by the gate from the source it is {want:.6g}")
    skill = at(result, "headline.skill", "number")
    ctx.require(at(result, "headline.baseline") == "previous_value",
                f"the headline baseline on `random_walk` is {at(result, 'headline.baseline')!r}, "
                f"giving skill {skill:.4f}. The previous value has the lower error "
                f"({base_previous:.4g} against {base_mean:.4g} for the training mean), so it is "
                "the baseline (D10): against the mean a random walk looks almost perfectly "
                "predictable")
    ctx.require(close(skill, 1 - mine / base_previous, 1e-7),
                f"headline.skill on `random_walk` is {skill:.6g}; 1 - error / previous-value "
                f"error, recomputed by the gate, is {1 - mine / base_previous:.6g}")
    band(ctx, skill, "random_walk", "skill on random_walk",
         "tomorrow's level is today's plus noise, so nothing beats the previous value")


@gate.check("time.integer_keys_distinct", catches=["F7c"],
            doc="64-bit integer timestamps above 2**53 stay distinct: the timestamp counts "
                "equal the gate's own, and the baseline is the previous value")
def time_integer_keys_distinct(ctx):
    why = ("On `int_stamps` the time column `t` holds 1,500 distinct whole numbers, nanoseconds "
           "since 1970, 100 apart and far above 2**53. A 64-bit float holds only every 256th "
           "whole number there, so keys read as floats merge into 587 timestamps: they are no "
           "longer unique, the previous-value baseline is lost, and a random walk is scored "
           "against the training mean. Integer time keys above 2**53 are compared as the "
           "integers they are (D50, D53; spec.md step 3)")
    try:
        result = outputs(ctx, "int_stamps")
    except CheckFailed as exc:
        raise CheckFailed(f"{exc}\n  {why}") from None
    time_at, target = result["header"].index("t"), result["header"].index("level")
    stamps = {i: int(row[time_at]) for i, row in enumerate(result["source"])}
    as_floats = {float(s) for s in stamps.values()}
    if len(set(stamps.values())) != len(stamps) or min(stamps.values()) <= 2 ** 53 \
            or len(as_floats) == len(stamps):
        ctx.harness_error("the dataset `int_stamps` is not what the gate built it to be: "
                          f"{len(set(stamps.values()))} distinct timestamps in {len(stamps)} "
                          f"rows, the smallest {min(stamps.values())}, and {len(as_floats)} "
                          "distinct values as floats; expected every one distinct, above "
                          "2**53, and fewer as floats")
    want = expected_time_split(stamps, gap=0)
    ctx.require(at(result, "validation.scheme") == "time_ordered"
                and at(result, "validation.time_form") == "numeric",
                f"validation on `int_stamps` reports scheme {at(result, 'validation.scheme')!r} "
                f"and time_form {at(result, 'validation.time_form')!r}; a whole-number time "
                "column gives the time-ordered scheme and the form 'numeric'")
    n_train = at(result, "validation.n_train_timestamps")
    n_holdout = at(result, "validation.n_holdout_timestamps")
    ctx.record("int_stamps training timestamps", n_train)
    ctx.record("int_stamps holdout timestamps", n_holdout)
    ctx.require((n_train, n_holdout) == (len(want["training"]), want["holdout_stamps"]),
                f"validation on `int_stamps` reports {n_train!r} training and {n_holdout!r} "
                f"holdout timestamps; the gate counts {len(want['training'])} and "
                f"{want['holdout_stamps']}, one per row ({len(as_floats)} of the "
                f"{len(stamps)} are left when the keys are read as floats). {why}")
    unique = at(result, "validation.timestamps_unique")
    ctx.record("int_stamps timestamps_unique", unique)
    ctx.require(unique is True,
                f"validation.timestamps_unique on `int_stamps` is {unique!r}; every one of the "
                f"{len(stamps)} timestamps is distinct. {why}")
    held = holdout_check(ctx, result, "int_stamps")
    ctx.require(set(held) == want["holdout"],
                f"the holdout on `int_stamps` is {len(held)} rows; the gate expects exactly the "
                f"{len(want['holdout'])} source rows with t >= {want['holdout_start']} "
                f"({len(set(held) - want['holdout'])} unexpected, "
                f"{len(want['holdout'] - set(held))} missing). {why}")
    for key, expected in (("validation.train_end", want["train_end"]),
                          ("validation.holdout_start", want["holdout_start"])):
        got = at(result, key)
        ctx.record(f"int_stamps {key}", got)
        ctx.require(isinstance(got, int) and not isinstance(got, bool) and got == expected,
                    f"`{key}` on `int_stamps` is {got!r}; the gate computes {expected}. A "
                    "whole-number timestamp is written as the whole number it is, exactly: "
                    "one that has been through a float is a multiple of 256 here")

    order = sorted(stamps, key=stamps.get)
    level = {i: float(result["source"][i][target]) for i in order}
    before = {row: order[k - 1] for k, row in enumerate(order) if k}
    in_holdout = set(held)
    truth = [level[p] for p in held]
    base_mean = mse(truth, [mean([level[i] for i in order if i not in in_holdout])] * len(truth))
    base_previous = mse(truth, [level[before[p]] for p in held])
    mine = mse(truth, [guess for _, _, guess in result["predictions"]])
    baselines = {b["name"]: b for b in records(result, "baselines", ["name", "score", "used"])}
    ctx.record("int_stamps baselines", ",".join(sorted(baselines)))
    ctx.require({"training_mean", "previous_value"} <= set(baselines),
                f"baselines on `int_stamps` lists {sorted(baselines)}; with unique timestamps "
                f"both training_mean and previous_value are scored and reported (D10). {why}")
    for name, expected in (("training_mean", base_mean), ("previous_value", base_previous)):
        got = baselines[name]["score"]
        ctx.require(isinstance(got, (int, float)) and not isinstance(got, bool)
                    and close(got, expected, 1e-7),
                    f"the {name} baseline on `int_stamps` scores {got!r}; recomputed by the "
                    f"gate from the source, in the order of the integer timestamps, it is "
                    f"{expected:.6g}")
    skill = at(result, "headline.skill", "number")
    ctx.record("int_stamps baseline", at(result, "headline.baseline"))
    ctx.require(at(result, "headline.baseline") == "previous_value",
                f"the headline baseline on `int_stamps` is {at(result, 'headline.baseline')!r}, "
                f"giving skill {skill:.4f}. The previous value has the lower error "
                f"({base_previous:.4g} against {base_mean:.4g} for the training mean), so it is "
                "the baseline (D10)")
    ctx.require(close(skill, 1 - mine / base_previous, 1e-7),
                f"headline.skill on `int_stamps` is {skill:.6g}; 1 - error / previous-value "
                f"error, recomputed by the gate, is {1 - mine / base_previous:.6g}")


@gate.check("time.undeclared_warns",
            why="a date column with no declared time column is the commonest way to get a "
                "random split on ordered data without noticing",
            doc="a date column and --no-time-order: random scheme, with a warning, and the "
                "statement recorded")
def time_undeclared_warns(ctx):
    try:
        result = outputs(ctx, "time_blocks_undeclared")
    except CheckFailed as exc:
        raise CheckFailed(
            f"{exc}\n  This run passes `--no-time-order` and `--duplicates keep` (spec.md 2.1, "
            "steps 5 and 8a; D49, D53, D58). `--no-time-order` is the caller's statement that "
            "the rows have no time order: without it a table with a date column stops with "
            "`time_order_undeclared`. With `date` excluded, the rows of one block of dates "
            "differ in nothing the models see, so every holdout row has a copy among the "
            "training rows; without `--duplicates keep` the run stops with "
            "`holdout_copies_in_training`. The command line must accept both, and with both "
            "the run goes ahead") from None
    scheme = at(result, "validation.scheme")
    ctx.record("undeclared scheme", scheme)
    ctx.require(scheme == "random",
                f"validation.scheme is {scheme!r} with no --time-col: the time column is never "
                "inferred (D7)")
    ctx.require(at(result, "input.time_col") is False,
                f"input.time_col is {at(result, 'input.time_col')!r} on a run given "
                "`--no-time-order`: the caller's statement is recorded as `false` (D58; spec.md "
                "5.2), so that the output says who chose the random split")
    ctx.require(any(w["column"] == "date" for w in warnings_of(result, "date_column_undeclared")),
                "no warning with code `date_column_undeclared` names the column `date`: a date "
                "column is present and the rows were split at random")
    ctx.require(at(result, "validation.caveat"), "validation.caveat must carry the fixed "
                                                 "random-split caveat (D6)")


@gate.check("time.undeclared_stops", catches=["F7d"],
            doc="a date column, no --time-col and no --no-time-order: the run stops with the "
                "named error and writes no headline")
def time_undeclared_stops(ctx):
    stopped(ctx, "time_blocks_unstated", "time_order_undeclared",
            "`time_blocks` has a column of dates, `date`, and this run declares no time column "
            "and does not say that the rows have no time order. A table that holds dates is "
            "very often ordered in time, and under a random split such a table is reported as "
            "predictable when it is not: here the label is random per block of five dates, and "
            "a random split scores near 1. So the run stops and asks: `--time-col date`, or "
            "`--no-time-order` (D58; spec.md step 5). The run passes `--duplicates keep`, so "
            "the copies rule of step 8a is not what stops it")


def _with_hint(ctx, name, hint):
    """`outputs`, with a word on the arguments of the run when it did not succeed."""
    try:
        return outputs(ctx, name)
    except CheckFailed as exc:
        raise CheckFailed(f"{exc}\n  {hint}") from None


def _all_rows_split(ctx, result, label):
    """The holdout rows, for a run in which every source row is on one side of the
    split: none dropped, none capped. Returns (holdout positions in file order, set)."""
    held = holdout_check(ctx, result, label)
    got, n = row_counts(result), len(result["source"])
    ctx.require(got.get("loaded") == n and got.get("holdout") == len(held)
                and got.get("training") == n - len(held),
                f"rows on `{label}`: loaded {got.get('loaded')!r}, training "
                f"{got.get('training')!r}, holdout {got.get('holdout')!r}; the gate counts {n} "
                f"source rows, none of them dropped, and {len(held)} rows in predictions.csv, "
                f"so {n - len(held)} training rows")
    return held, set(held)


@gate.check("time.panel_previous_value", catches=["F8b"],
            doc="twelve random walks with --group-col: the baseline is each series' own "
                "previous value, recomputed by the gate, and skill is about zero")
def time_panel_previous_value(ctx):
    result = _with_hint(
        ctx, "panel",
        "This run passes `--time-col date --group-col series` (spec.md 2.1; D57). The command "
        "line must accept `--group-col`.")
    held, in_holdout = _all_rows_split(ctx, result, "panel")
    header, source = result["header"], result["source"]
    time_at, series_at, target = header.index("date"), header.index("series"), header.index("level")
    level = [float(row[target]) for row in source]
    of_series = {}
    for i, row in enumerate(source):
        of_series.setdefault(row[series_at], []).append((row[time_at], i))
    before = {}                             # a row -> the row of its series on the date before
    for dated in of_series.values():
        dated.sort()
        for (_, earlier), (_, later) in zip(dated, dated[1:]):
            before[later] = earlier
    ctx.require(at(result, "validation.scheme") == "time_ordered",
                f"validation.scheme on `panel` is {at(result, 'validation.scheme')!r}: with "
                "--time-col the split is by time, with or without an entity column (D57)")
    ctx.require(all(p in before for p in held),
                "a holdout row of `panel` is the first row of its series: every series runs "
                "over every date, so the holdout must be the latest dates")
    truth = [level[p] for p in held]
    training_mean = mean([v for i, v in enumerate(level) if i not in in_holdout])
    base_mean = mse(truth, [training_mean] * len(truth))
    base_previous = mse(truth, [level[before[p]] for p in held])
    mine = mse(truth, [guess for _, _, guess in result["predictions"]])
    baselines = {b["name"]: b for b in records(result, "baselines", ["name", "score", "used"])}
    ctx.record("panel baseline", at(result, "headline.baseline"))
    ctx.require({"training_mean", "previous_value"} <= set(baselines),
                f"baselines on `panel` lists {sorted(baselines)}. Every date holds twelve rows, "
                "one for each series, and the series are declared with `--group-col series`: "
                "each row's previous value is the target of its own series on the timestamp "
                "before, so both training_mean and previous_value are scored and reported "
                "(D57; spec.md step 11). Without it the only baseline is the training mean, "
                "against which twelve random walks look almost perfectly predictable")
    for name, want in (("training_mean", base_mean), ("previous_value", base_previous)):
        ctx.require(isinstance(baselines[name]["score"], (int, float))
                    and close(baselines[name]["score"], want, 1e-7),
                    f"the {name} baseline on `panel` scores {baselines[name]['score']!r}; "
                    f"recomputed by the gate from the source it is {want:.6g}. The previous "
                    "value of a row is the target of the same series on the date before")
    skill = at(result, "headline.skill", "number")
    ctx.require(at(result, "headline.baseline") == "previous_value",
                f"the headline baseline on `panel` is {at(result, 'headline.baseline')!r}, giving "
                f"skill {skill:.4f}. The previous value has the lower error "
                f"({base_previous:.4g} against {base_mean:.4g} for the training mean), so it is "
                "the baseline (D10, D57)")
    ctx.require(close(skill, 1 - mine / base_previous, 1e-7),
                f"headline.skill on `panel` is {skill:.6g}; 1 - error / previous-value error, "
                f"recomputed by the gate, is {1 - mine / base_previous:.6g}")
    ctx.require(at(result, "validation.group_col") == "series"
                and at(result, "validation.n_groups_training") == len(of_series)
                and at(result, "validation.n_groups_holdout") == len(of_series),
                f"validation on `panel` reports group_col "
                f"{at(result, 'validation.group_col')!r} and "
                f"{at(result, 'validation.n_groups_training')!r} and "
                f"{at(result, 'validation.n_groups_holdout')!r} entities in the training and "
                f"the holdout rows; the gate counts the column `series` and {len(of_series)} "
                "series on each side")
    ctx.require(at(result, "validation.n_holdout_without_previous") == 0,
                f"validation.n_holdout_without_previous on `panel` is "
                f"{at(result, 'validation.n_holdout_without_previous')!r}; every holdout row has "
                "an earlier row of its own series, so it is 0")
    band(ctx, skill, "panel", "skill on panel",
         "each series is a random walk: tomorrow's level is today's plus noise, so nothing "
         "beats the series' own previous value")


@gate.check("groups.kept_together", catches=["F10"],
            doc="with --group-col no entity has rows on both sides of the split, by the gate's "
                "own count from predictions.csv, and the output reports the grouped scheme")
def groups_kept_together(ctx):
    result = _with_hint(
        ctx, "entities",
        "This run passes `--group-col customer` (spec.md 2.1; D57). The command line must "
        "accept `--group-col`.")
    held, in_holdout = _all_rows_split(ctx, result, "entities")
    entity_at = result["header"].index("customer")
    entity = [row[entity_at] for row in result["source"]]
    held_entities = {entity[i] for i in in_holdout}
    training_entities = {e for i, e in enumerate(entity) if i not in in_holdout}
    both = sorted(held_entities & training_entities)
    ctx.record("entities on both sides", len(both))
    ctx.require(not both,
                f"{len(both)} of the {len(set(entity))} entities of `entities` have rows on both "
                f"sides of the split (for example {', '.join(both[:3])}). The run declares "
                "`--group-col customer`: the rows of one entity stay on one side of the holdout "
                "split, and of every fold (D57; spec.md step 7). Here the features identify the "
                "entity and the label belongs to the entity, so a row on each side lets a model "
                "score by knowing the entity again")
    scheme = at(result, "validation.scheme")
    ctx.require(scheme == "grouped",
                f"validation.scheme on `entities` is {scheme!r}; with an entity column and no "
                "time column it is `grouped`")
    ctx.require(at(result, "validation.group_col") == "customer"
                and at(result, "input.group_col") == "customer",
                "input.group_col and validation.group_col must both record the entity column "
                "the run used (`customer`)")
    counts = (at(result, "validation.n_groups_training"), at(result, "validation.n_groups_holdout"))
    ctx.require(counts == (len(training_entities), len(held_entities)),
                f"validation reports {counts[0]!r} entities in the training rows and "
                f"{counts[1]!r} in the holdout; the gate counts {len(training_entities)} and "
                f"{len(held_entities)}")
    roles = {c["name"]: c["role"] for c in records(result, "columns", ["name", "role"])}
    ctx.require(roles.get("customer") == "group",
                f"the column `customer` has role {roles.get('customer')!r}; the entity column "
                "has the role `group` and is not a feature (spec.md 5.2)")
    share = at(result, "validation.holdout_share", "number")
    largest = max(entity.count(e) for e in set(entity))
    ctx.require(0.25 <= share <= 0.25 + 2 * largest / len(entity) and close(
        share, len(held) / len(entity)),
                f"validation.holdout_share on `entities` is {share:.4f}, {len(held)} rows. The "
                "holdout is whole entities that together hold at least a quarter of the rows "
                "and, for each class, less than one entity more than that (spec.md step 7): "
                f"between 0.25 and {0.25 + 2 * largest / len(entity):.4f} here")


@gate.check("groups.no_skill_on_entities", catches=["F10"],
            doc="no skill where only an entity on both sides of the split could give any")
def groups_no_skill_on_entities(ctx):
    result = _with_hint(
        ctx, "entities",
        "This run passes `--group-col customer` (spec.md 2.1; D57). The command line must "
        "accept `--group-col`.")
    band(ctx, at(result, "headline.skill", "number"), "entities", "skill on entities",
         "the label is random per entity and the features only identify the entity, so "
         "nothing learned on one entity helps on another. Skill here means rows of one entity "
         "are on both sides of the split")


@gate.check("time.flat_holdout_reported",
            why="a baseline that scores exactly zero is where skill, 1 - error / baseline "
                "error, has no value. The first build died there with a ZeroDivisionError and "
                "wrote nothing; a zero or an infinity in its place would read as a measurement "
                "(D36)",
            doc="a target flat across the holdout: the run finishes, skill is null with a "
                "warning, the score and both baselines are reported")
def time_flat_holdout_reported(ctx):
    why = ("On `flat_tail` nothing is sold from 65% of the way through the series to its end, "
           "so the previous value predicts every holdout row exactly and that baseline's squared "
           "error on the holdout is 0. Skill, 1 - error / baseline error, then has no value: "
           "`headline.skill` is null, a `skill_undefined` warning says why, and the score, both "
           "baselines and the rest of the output are reported as usual (D36)")
    try:
        result = outputs(ctx, "flat_tail")
    except CheckFailed as exc:
        raise CheckFailed(f"{exc}\n  {why}") from None
    ctx.require(not result["bad_numbers"],
                f"profile.json on `flat_tail` contains {sorted(set(result['bad_numbers']))}: "
                f"NaN and infinity are not JSON and never appear. {why}")
    held = holdout_check(ctx, result, "flat_tail")
    want = _time_expectation(result, gap=0)
    ctx.require(set(held) == want["holdout"],
                f"the holdout on `flat_tail` is {len(held)} rows; the gate expects exactly the "
                f"{len(want['holdout'])} source rows dated {want['holdout_start']} or later")
    time_at, target = result["header"].index("date"), result["header"].index("sales")
    order = sorted(range(len(result["source"])), key=lambda i: result["source"][i][time_at])
    sales = {i: float(result["source"][i][target]) for i in order}
    before = {row: order[k - 1] for k, row in enumerate(order) if k}
    truth = [sales[p] for p in held]
    base_mean = mse(truth, [mean([sales[i] for i in want["training"]])] * len(truth))
    base_previous = mse(truth, [sales[before[p]] for p in held])
    if base_previous != 0 or base_mean <= 0:
        ctx.harness_error("the dataset `flat_tail` is not what the gate built it to be: the "
                          f"previous-value baseline scores {base_previous!r} on the holdout and "
                          f"the training mean {base_mean!r}; expected exactly 0 and above 0")
    mine = mse(truth, [guess for _, _, guess in result["predictions"]])

    baselines = {b["name"]: b for b in records(result, "baselines", ["name", "score", "used"])}
    ctx.record("flat_tail baselines", ",".join(sorted(baselines)))
    ctx.require({"training_mean", "previous_value"} <= set(baselines),
                f"baselines on `flat_tail` lists {sorted(baselines)}; with unique timestamps "
                "both training_mean and previous_value are scored and reported (D10), also "
                "when one of them scores zero")
    for name, expected in (("training_mean", base_mean), ("previous_value", base_previous)):
        got = baselines[name]["score"]
        ctx.record(f"flat_tail {name} score", got)
        ctx.require(isinstance(got, (int, float)) and not isinstance(got, bool)
                    and close(got, expected, 1e-7),
                    f"the {name} baseline on `flat_tail` scores {got!r}; recomputed by the gate "
                    f"from the source it is {expected:.6g}")
    ctx.require(at(result, "headline.baseline") == "previous_value"
                and at(result, "headline.baseline_score") == 0,
                f"the headline baseline on `flat_tail` is {at(result, 'headline.baseline')!r} "
                f"with score {at(result, 'headline.baseline_score')!r}; the previous value has "
                f"the lower error (0 against {base_mean:.4g} for the training mean), so it is "
                "the baseline (D10) and its score, 0, is reported as measured")
    ctx.require(close(at(result, "headline.score", "number"), mine),
                f"headline.score on `flat_tail` is {at(result, 'headline.score'):.6g}; mean "
                f"squared error recomputed by the gate from predictions.csv is {mine:.6g}")

    skill = at(result, "headline.skill")
    ctx.record("flat_tail skill", skill)
    ctx.require(skill is None,
                f"headline.skill on `flat_tail` is {skill!r}; it must be null. {why}. A number "
                "here, zero included, is not a measurement")
    for key in ("headline.skill_low", "headline.skill_high"):
        ctx.require(at(result, key) is None,
                    f"`{key}` on `flat_tail` is {at(result, key)!r}; with no skill there is no "
                    "interval on it, so it is null")
    warned = warnings_of(result, "skill_undefined")
    ctx.record("flat_tail skill_undefined warnings", len(warned))
    ctx.require(len(warned) == 1 and warned[0]["column"] is None,
                f"`flat_tail` has {len(warned)} warning(s) with code `skill_undefined`; there "
                f"is exactly one, with a null column. {why}")

    # The score still has its interval: only skill is undefined (spec.md, step 12).
    method = at(result, "headline.interval_method")
    ctx.record("flat_tail interval method", method)
    ctx.require(method == "bootstrap_blocks",
                f"headline.interval_method on `flat_tail` is {method!r}; expected "
                "'bootstrap_blocks'. The holdout holds 375 timestamps and the squared error can "
                "be computed in every resample: a resample whose baseline error is zero is left "
                "out of the interval on skill only")
    low, high = at(result, "headline.interval_low", "number"), \
        at(result, "headline.interval_high", "number")
    ctx.record("flat_tail interval low", low)
    ctx.record("flat_tail interval high", high)
    ctx.require(low <= mine <= high,
                f"the interval on `flat_tail` ({low:.6g} to {high:.6g}) does not contain the "
                f"headline score {mine:.6g}")
    used, used_for_skill = at(result, "headline.n_resamples_used"), \
        at(result, "headline.n_resamples_skill_used")
    ctx.require(used == 200 and used_for_skill == 0,
                f"`flat_tail` reports n_resamples_used {used!r} and n_resamples_skill_used "
                f"{used_for_skill!r}; the score can be computed in all 200 resamples and skill "
                "in none of them, and the two are counted apart")


@gate.check("uncertainty.reported",
            why="a headline without a valid interval reads as more certain than it is",
            doc="the interval contains the score; row bootstrap with a p-value, or block "
                "bootstrap labelled approximate")
def uncertainty_reported(ctx):
    for name, method in (("signal_clf", "bootstrap_rows"), ("time_blocks", "bootstrap_blocks")):
        result = outputs(ctx, name)
        score, skill = at(result, "headline.score", "number"), at(result, "headline.skill", "number")
        got = at(result, "headline.interval_method")
        ctx.record(f"{name} interval method", got)
        ctx.require(got == method, f"headline.interval_method on `{name}` is {got!r}; expected "
                                   f"{method!r} under the {at(result, 'validation.scheme')} scheme")
        low, high = at(result, "headline.interval_low", "number"), \
            at(result, "headline.interval_high", "number")
        ctx.record(f"{name} interval low", low)
        ctx.record(f"{name} interval high", high)
        ctx.require(low < high, f"the interval on `{name}` is degenerate: {low} to {high}")
        ctx.require(low <= score <= high,
                    f"the interval on `{name}` ({low:.4f} to {high:.4f}) does not contain the "
                    f"headline score {score:.4f}: it was computed from other predictions")
        ctx.require(at(result, "headline.skill_low", "number") <= skill
                    <= at(result, "headline.skill_high", "number"),
                    f"the skill interval on `{name}` does not contain the skill {skill:.4f}")
        ctx.require(at(result, "headline.n_resamples") == 200,
                    "the fast profile uses 200 resamples (headline.n_resamples)")
        approximate = warnings_of(result, "interval_approximate")
        ctx.record(f"{name} interval_approximate warnings", len(approximate))
        if method == "bootstrap_blocks":
            ctx.require(len(approximate) == 1 and approximate[0]["column"] is None,
                        f"`{name}` has {len(approximate)} warning(s) with code "
                        "`interval_approximate`; whenever the interval is a block bootstrap "
                        "there is exactly one, with a null column: that interval tends to be "
                        "too narrow, and the output says so (D27)")
        else:
            ctx.require(not approximate,
                        f"`{name}` carries an `interval_approximate` warning under the random "
                        "scheme: it belongs to the block bootstrap of the time-ordered scheme "
                        "only (D27)")
        if method == "bootstrap_rows":
            p_value = at(result, "headline.p_value", "number")
            ctx.record(f"{name} p-value", p_value)
            ctx.require(0 < p_value <= 1, f"headline.p_value on `{name}` is {p_value}")
            ctx.require(close(p_value, 1 / 201),
                        f"headline.p_value on `{name}` is {p_value:.4g}; with a clear signal no "
                        "shuffle of the holdout targets matches the score, so it is "
                        "(1 + 0) / (1 + 200)")
        else:
            want = cube_root_up(_time_expectation(result, gap=0)["holdout_stamps"])
            ctx.require(at(result, "headline.block_length") == want,
                        f"headline.block_length on `{name}` is "
                        f"{at(result, 'headline.block_length')!r}; the cube root of the number "
                        f"of distinct holdout timestamps, rounded up, is {want}")
            ctx.require(at(result, "headline.p_value") is None
                        and at(result, "headline.p_value_reason"),
                        f"under time order the p-value is absent and headline.p_value_reason "
                        "says why (shuffling assumes independent rows)")


def _same_json(a, b, path=""):
    """First difference between two JSON values, or None. Numbers within 1e-9."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}.{key} is present on one side only"
            found = _same_json(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path} has {len(a)} entries on one side and {len(b)} on the other"
        for i, (x, y) in enumerate(zip(a, b)):
            found = _same_json(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    numbers = [isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b)]
    if all(numbers):
        return None if close(a, b) else f"{path}: {a!r} != {b!r}"
    return None if a == b and type(a) is type(b) else f"{path}: {a!r} != {b!r}"


def _api_payload(ctx):
    result = outputs(ctx, "api_signal_clf")
    if "payload" not in result:
        lines = [l for l in result["stdout"].splitlines() if l.startswith("@@RESULT@@")]
        ctx.require(lines, "the script that calls mlprofessor.profile() printed no result")
        result["payload"] = json.loads(lines[-1][len("@@RESULT@@"):])
    return result, result["payload"]


@gate.check("result.matches_json",
            why="two accounts of one run that disagree make every other check meaningless "
                "to the user",
            doc="through the Python function, every part of ProfileResult equals profile.json")
def result_matches_json(ctx):
    result, payload = _api_payload(ctx)
    doc = result["doc"]
    for name in ("rows", "columns", "models", "warnings"):
        found = _same_json(payload["frames"][name], doc.get(name), name)
        ctx.require(found is None,
                    f"ProfileResult.{name} (as records) differs from profile.json: {found}")
    found = _same_json(payload["headline"], doc.get("headline"), "headline")
    ctx.require(found is None, f"ProfileResult.headline differs from profile.json: {found}")
    found = _same_json(payload["to_dict"], doc, "to_dict()")
    ctx.require(found is None, f"ProfileResult.to_dict() differs from profile.json: {found}")
    written = json.loads((result["out"] / "via_to_json.json").read_text(encoding="utf-8"))
    found = _same_json(written, doc, "to_json()")
    ctx.require(found is None, f"ProfileResult.to_json(path) differs from profile.json: {found}")
    frame = payload["frames"]["predictions"]
    ctx.record("api predictions", len(frame))
    ctx.require(len(frame) == len(result["predictions"])
                and all(isinstance(r, dict) and set(r) == {"row", "y_true", "y_pred"}
                        for r in frame),
                "ProfileResult.predictions must have the columns row, y_true, y_pred and one "
                "row per row of predictions.csv")
    for mine, (position, truth, guess) in zip(frame, result["predictions"]):
        ctx.require(mine["row"] == position and same_cell(mine["y_true"], truth)
                    and close(mine["y_pred"], guess),
                    f"ProfileResult.predictions differs from predictions.csv at row {position}")


REQUIRED = {
    "meta": ["tool", "schema_version", "profile", "seed", "n_jobs", "versions", "timing"],
    "input": ["source", "kind", "target", "time_col", "task", "gap", "n_rows", "n_columns",
              "duplicate_rows"],
    "task": ["type", "decided_by", "n_distinct", "target_dtype", "classes", "positive_class"],
    "validation": ["scheme", "holdout_share", "n_folds", "stratified", "gap", "time_col",
                   "time_form", "train_end", "holdout_start", "holdout_end",
                   "n_train_timestamps", "n_holdout_timestamps", "timestamps_unique", "folds",
                   "caveat"],
    "headline": ["metric", "score", "model", "baseline", "baseline_score", "skill",
                 "interval_method", "interval_level", "interval_low", "interval_high",
                 "skill_low", "skill_high", "n_resamples", "n_resamples_used", "block_length",
                 "interval_reason", "p_value", "n_permutations", "p_value_reason"],
    "cv": ["metric", "greater_is_better", "n_folds", "n_scored_folds", "n_configurations",
           "chosen", "chosen_mean", "chosen_std"],
}
RECORD_KEYS = {
    "rows": ["item", "count"],
    "columns": ["name", "position", "dtype", "role", "kind", "reason", "leak_rule", "n_missing",
                "n_coerced", "n_infinite", "n_imputed"],
    "baselines": ["name", "score", "used"],
    "models": ["config", "family", "params", "status", "error", "fold_scores", "cv_mean",
               "cv_std", "chosen"],
    "warnings": ["code", "column", "message"],
}
# On signal_clf (random scheme, classification) these may be null; everything else may not.
MAY_BE_NULL = {"input.time_col", "validation.time_col", "validation.time_form",
               "validation.train_end", "validation.holdout_start", "validation.holdout_end",
               "validation.n_train_timestamps", "validation.n_holdout_timestamps",
               "validation.timestamps_unique", "headline.block_length",
               "headline.interval_reason", "headline.p_value_reason"}


@gate.check("output.complete",
            why="a blank or NaN field looks like a result",
            doc="every required key is present and populated; no NaN or infinity anywhere")
def output_complete(ctx):
    result = outputs(ctx, "signal_clf")
    doc = result["doc"]
    ctx.require(not result["bad_numbers"],
                f"profile.json on `signal_clf` contains {sorted(set(result['bad_numbers']))}: "
                "NaN and infinity are not JSON and never appear; write null and say why")
    top = list(REQUIRED) + list(RECORD_KEYS) + ["limitations"]
    missing = [k for k in top if k not in doc]
    ctx.require(not missing, f"profile.json has no top-level key(s) {missing} (spec.md, "
                             "'profile.json')")
    for section, keys in REQUIRED.items():
        for key in keys:
            path = f"{section}.{key}"
            ctx.require(isinstance(doc[section], dict) and key in doc[section],
                        f"profile.json has no key `{path}` (spec.md, 'profile.json')")
            value = doc[section][key]
            ctx.require(path in MAY_BE_NULL or (value is not None and value != ""),
                        f"`{path}` in profile.json is {value!r} on `signal_clf`: it must be "
                        "populated on a classification run under the random scheme")
    for section, keys in RECORD_KEYS.items():
        items = records(result, section, keys)
        ctx.require(items or section == "warnings", f"`{section}` in profile.json is empty")
    ctx.require(at(result, "meta.schema_version") == 1 and at(result, "meta.tool") == "mlprofessor",
                "meta.tool must be 'mlprofessor' and meta.schema_version must be 1")
    ctx.require(at(result, "meta.n_jobs") == 1 and at(result, "meta.seed") == 0
                and at(result, "meta.profile") == "fast",
                "meta must record the profile, seed and n_jobs the run used (fast, 0, 1)")
    ctx.require(len(at(result, "validation.folds", list)) == 5, "validation.folds must list 5 folds")
    got = [" ".join(str(s).split()) for s in at(result, "limitations", list)]
    ctx.record("limitations", len(got))
    differs = [i + 1 for i, (a, b) in enumerate(zip(got, LIMITATIONS)) if a != b]
    ctx.require(got == LIMITATIONS,
                "the limitations statement in profile.json is not the fixed text of spec.md, "
                f"'Limitations' ({len(LIMITATIONS)} sentences, in order); found {len(got)} "
                f"entries, and sentence(s) {differs or 'beyond the shorter list'} differ")


def _without(doc, *paths):
    doc = json.loads(json.dumps(doc))
    for path in paths:
        section, key = path.split(".")
        if isinstance(doc.get(section), dict):
            doc[section].pop(key, None)
    return doc


@gate.check("output.deterministic",
            why="a number that moves between identical runs cannot be checked",
            doc="two runs with the same seed give identical files apart from meta.timing")
def output_deterministic(ctx):
    first, second = outputs(ctx, "signal_clf"), outputs(ctx, "api_signal_clf")
    a, b = _without(first["doc"], "meta.timing"), _without(second["doc"], "meta.timing")
    ctx.record("deterministic", a == b)
    found = _same_json(a, b, "profile.json") if a != b else None
    ctx.require(a == b,
                "two runs on `signal_clf` with seed 0 (one through the command line, one through "
                f"mlprofessor.profile) wrote different profile.json files: {found or 'values differ beyond 1e-9 nowhere, but the files are not identical'}. "
                "Only meta.timing may differ: pin every seed, and keep clock values, absolute "
                "paths other than the one given, and unordered sets out of the output")
    ctx.require(first["predictions_bytes"] == second["predictions_bytes"],
                "two runs on `signal_clf` with seed 0 wrote different predictions.csv files")


@gate.check("output.same_at_any_n_jobs", slow=True,
            why="the default uses every core, so a result that depends on the core count "
                "differs between machines",
            doc="full profile at --n-jobs 1 and 4: identical files apart from timing and n_jobs")
def output_same_at_any_n_jobs(ctx):
    # Measured on copies of the reference, 500 rows, 4 trials each. A copy whose
    # forests predict on n_jobs threads: the two runs differ in 4 of 4 trials on
    # signal_reg (its fold scores are squared errors, which carry the last bits of
    # every forest prediction) and in 0 of 4 on signal_clf (ROC AUC only ranks).
    # The reference itself: 0 of 4 on both. The regression half is where this
    # check bites. A tool that leaves BLAS at many threads is NOT caught on tables
    # this small (it shows at 20,000 x 100); the test suite proves that limit.
    for name in ("signal_clf", "signal_reg"):
        one, four = outputs(ctx, f"full_{name}_j1"), outputs(ctx, f"full_{name}_j4")
        ctx.require(at(one, "meta.n_jobs") == 1 and at(four, "meta.n_jobs") == 4,
                    f"meta.n_jobs must record the cores used: got {at(one, 'meta.n_jobs')!r} "
                    f"with --n-jobs 1 and {at(four, 'meta.n_jobs')!r} with --n-jobs 4")
        a = _without(one["doc"], "meta.timing", "meta.n_jobs")
        b = _without(four["doc"], "meta.timing", "meta.n_jobs")
        ctx.record(f"{name} same at 1 and 4 cores", a == b)
        ctx.require(a == b,
                    f"the full profile on `{name}_small` gives different results with --n-jobs 1 and "
                    f"--n-jobs 4: {_same_json(a, b, 'profile.json')}. The same seed must give "
                    "the same result at any n_jobs (D17): hold linear algebra to one thread "
                    "with threadpoolctl, and make forests predict on one thread")
        ctx.require(one["predictions_bytes"] == four["predictions_bytes"],
                    f"predictions.csv on `{name}_small` differs between --n-jobs 1 and "
                    "--n-jobs 4")
        ctx.record(f"{name} full-profile skill", at(one, "headline.skill", "number"))


# ---------------------------------------------------------------- fixtures --

def _fixture(tmp, broken):
    """A copy of the reference implementation, carrying one named failure or none."""
    if not (REFERENCE / "core.py").is_file():
        raise HarnessError(f"the reference implementation is missing from {REFERENCE}")
    target = Path(tmp) / "mlprofessor"
    shutil.copytree(REFERENCE, target, ignore=shutil.ignore_patterns("__pycache__"))
    (target / "_variant.py").write_text(f"BROKEN = {broken!r}\n", encoding="utf-8")
    return target


@gate.good_fixture
def good(tmp):
    return _fixture(tmp, None)


@gate.failure_mode("F1", "the chosen model is refitted on all rows before the holdout is scored")
def broken_refit_on_all_rows(tmp):
    return _fixture(tmp, "F1")


@gate.failure_mode("F2", "the leak screen is removed, so a copy of the target is a feature")
def broken_no_leak_screen(tmp):
    return _fixture(tmp, "F2")


@gate.failure_mode("F3", "the headline is the best cross-validation score, not the holdout score")
def broken_headline_is_cv(tmp):
    return _fixture(tmp, "F3")


@gate.failure_mode("F4", "the headline metric is accuracy")
def broken_accuracy(tmp):
    return _fixture(tmp, "F4")


@gate.failure_mode("F5", "the task is decided from the column's type alone")
def broken_task_from_type(tmp):
    return _fixture(tmp, "F5")


@gate.failure_mode("F6", "rows with any missing value are dropped before counting")
def broken_rows_dropped(tmp):
    return _fixture(tmp, "F6")


@gate.failure_mode("F7", "the split is shuffled even when a time column is declared")
def broken_shuffled_split(tmp):
    return _fixture(tmp, "F7")


@gate.failure_mode("F8", "the regression baseline is always the training mean")
def broken_mean_baseline(tmp):
    return _fixture(tmp, "F8")


# The other forms of F6 and F7 (D46). The F6 fixture loses rows and the F7 fixture
# shuffles the split; neither fails the four checks below, which had therefore never
# been seen to fail (second review, finding 4). Each form has a fixture of its own,
# under an id that shows its family, and each check names the form it was seen to
# catch. Registered after F1 to F8, so the order of those in the matrix stays.

@gate.failure_mode("F6b", "a column is lost from the accounting (constant columns are dropped "
                          "with no record)")
def broken_column_lost(tmp):
    return _fixture(tmp, "F6b")


@gate.failure_mode("F6c", "a model family is lost from the models table (only the chosen "
                          "configuration's family is listed)")
def broken_family_lost(tmp):
    return _fixture(tmp, "F6c")


@gate.failure_mode("F6d", "data rows longer than the header are read silently, with no error")
def broken_long_rows_read(tmp):
    return _fixture(tmp, "F6d")


@gate.failure_mode("F7b", "a day-first time column is parsed leniently, so the run goes ahead "
                          "on the wrong order")
def broken_lenient_dates(tmp):
    return _fixture(tmp, "F7b")


# The two routes to a wrong headline that the third review found (D49, D50), each
# with a dataset on which the right conclusion is known and a fixture that takes
# the wrong road. Registered last, so the order of the modes above stays.

@gate.failure_mode("F7c", "whole-number time keys are read as floats, so distinct instants "
                          "above 2**53 become one timestamp")
def broken_float_time_keys(tmp):
    return _fixture(tmp, "F7c")


@gate.failure_mode("F9", "holdout rows with copies among the training rows are scored with no "
                         "error")
def broken_copies_ignored(tmp):
    return _fixture(tmp, "F9")


# The three inflated headlines of the real-data run that the fifth pass removes
# (D57, D58), each with a dataset on which the right conclusion is known and a
# fixture that takes the wrong road. Registered last, so the order above stays.

@gate.failure_mode("F10", "an entity has rows on both sides of the split although the entity "
                          "column was declared")
def broken_entities_split(tmp):
    return _fixture(tmp, "F10")


@gate.failure_mode("F8b", "a panel is scored against the training mean although the entity "
                          "column was declared")
def broken_panel_mean_baseline(tmp):
    return _fixture(tmp, "F8b")


@gate.failure_mode("F7d", "a table with an undeclared date column is profiled under a random "
                          "split without the caller's say")
def broken_undeclared_date_runs(tmp):
    return _fixture(tmp, "F7d")


if __name__ == "__main__":
    gate.main()
