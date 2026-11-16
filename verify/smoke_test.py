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

The judge's own arithmetic (ROC AUC, squared error, row counts, which rows are
the holdout) is standard library only, so it shares no bug with pandas, numpy
or scikit-learn. Every score is asserted as a band; every band was measured on
the reference implementation in verify/reference/ over 30 data seeds, and the
measurements are written beside it.

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
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gatekit import CheckFailed, Gate, HarnessError  # noqa: E402

gate = Gate("mlprofessor")

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference" / "mlprofessor"
PARALLEL = 8                 # tool runs side by side, each single-threaded
# Seconds between process starts. Measured on this machine: interpreters started
# at the same instant (eight, four, or just two) all block for exactly 35 or 70
# seconds and then finish together, because the interpreter's base and standard
# library sit on A:, a drive mapped to a loopback network share, and 35 seconds
# is the share's lock-break timeout. Rounds of eight starts that blocked, by
# spacing: none 6 of 6, 0.1 s 2 of 2, 0.2 s 3 of 30, 0.3 s 0 of 30, 0.5 s 0 of 30.
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
    # good 0.630..0.824 (mean 0.731, sd 0.039).
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
    """signal_clf with everything a real export does to a table."""
    rng = random.Random(seed)
    base = [_signal_row(rng) for _ in range(n)]
    visits = rng.sample(range(1000, 100000), n)          # distinct, whole, not in order
    special = rng.sample(range(n), 18)
    text_cells = dict(zip(special[:8], ["abc", "1,200", "12 units", "3,450", "unknown", "abc",
                                        "1,200", "-"]))
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
        ])
    header = ["f1", "f2", "seg", "n1", "spend", "visits", "const", "record_id", "notes",
              "signup_date", "outcome"]
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
    "binary_numeric": ("binary_numeric", [], "fast", 1),
    "ambiguous_target": ("ambiguous_target", [], "fast", 1),
    "messy": ("messy", [], "fast", 1),
    "time_blocks": ("time_blocks", ["--time-col", "date"], "fast", 1),
    "time_blocks_gap": ("time_blocks", ["--time-col", "date", "--gap", "3"], "fast", 1),
    "time_blocks_undeclared": ("time_blocks", [], "fast", 1),
    "time_dayfirst": ("time_dayfirst", ["--time-col", "date"], "fast", 1),
    "random_walk": ("random_walk", ["--time-col", "date"], "fast", 1),
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
    "models.all_accounted": ["signal_clf"],
    "task.binary_numeric": ["binary_numeric"],
    "task.ambiguous_stops": ["ambiguous_target"],
    "leak.excluded": ["leak"],
    "leak.control_kept": ["leak"],
    "leak.regression": ["leak_reg"],
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
    "time.undeclared_warns": ["time_blocks_undeclared"],
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


@gate.setup
def prepare(ctx):
    for module in ("pandas", "numpy", "sklearn", "threadpoolctl"):
        if importlib.util.find_spec(module) is None:
            ctx.harness_error(f"the interpreter {sys.executable} cannot import {module}; the "
                              "modelling stack recorded in CLAUDE.md is not installed")
    data, runs_dir = ctx.artifacts / "data", ctx.artifacts / "runs"
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
        # Measured on this machine: when eight interpreters start together they
        # sometimes all wait a minute or more on file reads, because the
        # interpreter's base sits on a mapped drive (spec.md section 9). The tool
        # finished other runs in time, so a run that did not gets one more go
        # before it is called a failure. If nothing finished, nothing is retried.
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
}
ROLE_HINTS = {
    "visits": "a whole-number measure with every value distinct but not in order is a feature, "
              "not an identifier (D19)",
    "record_id": "a whole-number column whose values are all distinct and consecutive is an "
                 "identifier (D19)",
    "spend": "a text column in which at least 90% of the values parse as numbers is numeric",
    "n1": "`n/a` is a missing value on load, so this column is an ordinary numeric column",
}


@gate.check("data.columns_accounted", catches=["F6"],
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


def _parses(cell):
    try:
        float(cell)
        return True
    except ValueError:
        return False


@gate.check("models.all_accounted", catches=["F6"],
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
            doc="a strongly predictive but legitimate column stays a feature")
def leak_control_kept(ctx):
    result = outputs(ctx, "leak")
    got = {c["name"]: c for c in records(result, "columns", ["name", "role", "reason"])} \
        .get("strong_feature", {})
    ctx.record("leak strong_feature", f"{got.get('role')}/{got.get('reason')}")
    ctx.require(got.get("role") == "feature",
                f"`strong_feature` on `leak` has role {got.get('role')!r} (reason "
                f"{got.get('reason')!r}): it is strongly predictive but legitimate, and must "
                "stay a feature. The leak cut-off is too low or is not measured as spec.md says")


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


@gate.check("time.dayfirst_stops", catches=["F7"],
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


@gate.check("time.undeclared_warns",
            why="a date column with no declared time column is the commonest way to get a "
                "random split on ordered data without noticing",
            doc="a date column with no --time-col: random scheme, with a warning")
def time_undeclared_warns(ctx):
    result = outputs(ctx, "time_blocks_undeclared")
    scheme = at(result, "validation.scheme")
    ctx.record("undeclared scheme", scheme)
    ctx.require(scheme == "random",
                f"validation.scheme is {scheme!r} with no --time-col: the time column is never "
                "inferred (D7)")
    ctx.require(any(w["column"] == "date" for w in warnings_of(result, "date_column_undeclared")),
                "no warning with code `date_column_undeclared` names the column `date`: a date "
                "column is present and the rows were split at random")
    ctx.require(at(result, "validation.caveat"), "validation.caveat must carry the fixed "
                                                 "random-split caveat (D6)")


@gate.check("uncertainty.reported",
            why="a headline without a valid interval reads as more certain than it is",
            doc="the interval contains the score; row bootstrap with a p-value, or block bootstrap")
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


if __name__ == "__main__":
    gate.main()
