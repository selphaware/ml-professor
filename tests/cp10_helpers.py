"""Shared helpers of the ninth pass (CP10). Builds on `mlp_helpers`, `cp7_helpers` and
`cp9_helpers`; holds no test.

As in the passes before, nothing here makes a statement the call did not make (D63): the
tests of this pass call through `cp7_helpers.run`, `raised` and `run_cli`, and pass
`time_col` and `group_col` in every call that is meant to finish.

The rules of this pass that are a function of the output or of a few columns are written
here a second time, in plain arithmetic (`list_by_hand`, `floor_by_hand`,
`standin_half_width`, `against_by_pairs`, `against_by_means`), so that a test compares
the application with them.

Nothing here imports `mlprofessor` at module level.
"""
import math

import numpy as np
import pandas as pd

import cp7_helpers as s
import cp9_helpers as q
import mlp_helpers as h

# --------------------------------------------------------- pinned values (4.2, 4.3)

REST_SHUFFLES = 5
REST_FALL = 0.05
REST_MOST = 5
NEAR_PERFECT = 0.98
COPY_TOLERANCE = 1e-9
LEVEL_ROWS = 5
LEVEL_SCORED = 20
LEVEL_SCORED_CLASS = 10
FAR_ABOVE = 0.5
STANDIN_RESAMPLES_FAST = 200

STATEMENT_KEYS = ("shuffled_score", "shuffled_skill", "skill_fall")

NEAR = "near_perfect_headline"
NOT_COMPUTED = "shuffled_score_not_computed"
FAR = "holdout_far_above_quarter"
STRONG = "strong_single_column"
LEAK = "suspected_leak"

# The fixed passages of spec 5.4 added by the ninth pass (D77 point 6, D78 point 6).
FAR_ABOVE_PASSAGE = (
    "The models were fitted on the rows before it and the headline is measured on it: "
    "fewer rows were learned from than the split usually leaves."
)
GROUP_COL_SENTENCE = (
    "If it names the entity each row belongs to, declare it as group_col (--group-col): "
    "the rows of one entity are then kept together."
)

RANDOM = {"time_col": False, "group_col": False}


# ------------------------------------------------------------------ readers

def doc(result):
    return result.to_dict()


def record(result, name):
    """The record of `columns` for `name`, from `to_dict()`."""
    found = [r for r in doc(result)["columns"] if r["name"] == name]
    assert len(found) == 1, f"no single record for column {name!r}"
    return found[0]


def statement(result, name):
    """`(shuffled_score, shuffled_skill, skill_fall)` of the column `name`."""
    rec = record(result, name)
    return tuple(rec[k] for k in STATEMENT_KEYS)


def fall(result, name):
    return record(result, name)["skill_fall"]


def rests_on(result):
    """`headline.rests_on` as a list of `(column, skill_fall)`."""
    return [(r["column"], r["skill_fall"]) for r in doc(result)["headline"]["rests_on"]]


def listed(result):
    return [name for name, _ in rests_on(result)]


def floor(result):
    return doc(result)["headline"]["rests_on_floor"]


def features(result):
    return [r["name"] for r in doc(result)["columns"] if r["role"] == "feature"]


def told(result, code, name=None):
    return q.told(result, code, name)


def warning_records(result, code):
    return s.warning_records(result, code)


def leaks(result):
    return q.leaks(result)


def training_rows_of(result, frame):
    """The rows of `frame` that are not in the holdout (no row dropped)."""
    return q.training_rows(result, frame)


def same(a, b, rel=1e-12):
    """Two statements (or numbers) agree: both None, or both numbers within `rel`."""
    if isinstance(a, tuple):
        return len(a) == len(b) and all(same(x, y, rel) for x, y in zip(a, b))
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-15)


def quiet_feature(result, name):
    """A feature that no rule of the screen names: no exclusion, no warning on it."""
    rec = record(result, name)
    return (rec["role"] == "feature" and rec["reason"] is None
            and not told(result, STRONG, name) and not told(result, LEAK, name))


# ------------------------------------------------------- rules, by hand

def skill_of(score, headline):
    """Skill of a score against the headline baseline, as step 11 computes it."""
    if headline["metric"] == "roc_auc":
        return 2.0 * float(score) - 1.0
    return 1.0 - float(score) / float(headline["baseline_score"])


def assert_rule_17(result):
    """5.3 rule 17, every record."""
    d = doc(result)
    head = d["headline"]
    named = {r["column"] for r in d["warnings"] if r["code"] == NOT_COMPUTED}
    for rec in d["columns"]:
        keys = tuple(rec[k] for k in STATEMENT_KEYS)
        if rec["role"] != "feature" or rec["name"] in named:
            assert keys == (None, None, None), rec
            continue
        score, shuffled, dropped = keys
        assert isinstance(score, (int, float)) and math.isfinite(score), rec
        if head["skill"] is None:
            assert shuffled is None and dropped is None, rec
            continue
        assert math.isclose(shuffled, skill_of(score, head), rel_tol=1e-9, abs_tol=1e-12), rec
        assert math.isclose(dropped, head["skill"] - shuffled, rel_tol=1e-9, abs_tol=1e-12), rec


def floor_by_hand(headline):
    """5.3 rule 18 where the skill has an interval; None where it has none (the stand-in
    is then the application's to draw)."""
    if headline["skill"] is None:
        return None
    low, high = headline["skill_low"], headline["skill_high"]
    if low is None or high is None:
        return None
    return max(REST_FALL, (float(high) - float(low)) / 2.0)


def list_by_hand(result):
    """5.3 rule 19 from the column records and the floor: `(column, skill_fall)`."""
    d = doc(result)
    cut = d["headline"]["rests_on_floor"]
    if cut is None:
        return []
    rows = [(r["position"], r["name"], r["skill_fall"]) for r in d["columns"]
            if r["role"] == "feature" and r["skill_fall"] is not None
            and r["skill_fall"] >= cut]
    rows.sort(key=lambda t: (-t[2], t[0]))
    return [(name, value) for _, name, value in rows[:REST_MOST]]


def _auc(y, p):
    return h.auc_by_hand(np.asarray(y), np.asarray(p))


def standin_half_width(result, frame=None, target="y", resamples=2000, seed=0):
    """Half the middle 95% of the headline skill over a bootstrap of the holdout rows of
    `result.predictions`, taken as if independent, with the test's own generator. A
    regression's baseline is the training mean, taken from `frame` (no row dropped)."""
    preds = result.predictions
    truth = np.asarray(preds["y_true"], dtype=float)
    guess = np.asarray(preds["y_pred"], dtype=float)
    head = doc(result)["headline"]
    if head["metric"] == "mse":
        held = set(int(r) for r in preds["row"])
        y = np.asarray(frame[target], dtype=float)
        mean = float(np.mean([y[i] for i in range(len(y)) if i not in held]))
    rng = np.random.default_rng(seed)
    n = len(truth)
    skills = []
    for _ in range(resamples):
        pick = rng.integers(0, n, size=n)
        t, g = truth[pick], guess[pick]
        if head["metric"] == "roc_auc":
            if t.min() == t.max():
                continue
            skills.append(2.0 * _auc(t, g) - 1.0)
        else:
            base = float(np.sum((t - mean) ** 2))
            if base == 0:
                continue
            skills.append(1.0 - float(np.sum((t - g) ** 2)) / base)
    low, high = np.quantile(skills, [0.025, 0.975])
    return (float(high) - float(low)) / 2.0


def _clean_pairs(x, y):
    pairs = []
    for a, b in zip(x, y):
        if a is None or b is None:
            continue
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b):
            continue
        pairs.append((a, b))
    return pairs


def against_by_pairs(x, y):
    """The share against the order of step 8 (D77 point 4): for each two neighbouring
    distinct values of `x`, every row of the one compared with every row of the next;
    the shares in which `y` rises and falls, each averaged over the pairs of values; the
    smaller of the two averages. None with fewer than 20 pairs of values."""
    groups = {}
    for a, b in _clean_pairs(x, y):
        groups.setdefault(a, []).append(b)
    values = sorted(groups)
    if len(values) - 1 < q.ORDER_PAIRS:
        return None
    rise_sum = fall_sum = 0.0
    for v, w in zip(values, values[1:]):
        low, high = groups[v], groups[w]
        rises = sum(1 for a in low for b in high if b > a)
        falls = sum(1 for a in low for b in high if b < a)
        total = len(low) * len(high)
        rise_sum += rises / total
        fall_sum += falls / total
    pairs = len(values) - 1
    return min(rise_sum, fall_sum) / pairs


def against_by_means(x, y):
    """The form of M62, for contrast: one point for each distinct value of `x`, the mean
    of its `y`, read by the order of D76."""
    groups = {}
    for a, b in _clean_pairs(x, y):
        groups.setdefault(a, []).append(b)
    values = sorted(groups)
    means = [sum(groups[v]) / len(groups[v]) for v in values]
    return q.against_order(values, means)


def level_means(levels, target, rows):
    """The mean target of the rows `rows` for each level among them."""
    sums, counts = {}, {}
    for r in rows:
        sums[levels[r]] = sums.get(levels[r], 0.0) + float(target[r])
        counts[levels[r]] = counts.get(levels[r], 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


# ------------------------------------------------------------------- running

def run(frame, statements=None, **kw):
    """`cp7_helpers.run` with the statements given (the random scheme by default)."""
    said = dict(RANDOM if statements is None else statements)
    said.update(kw)
    return s.run(frame, **said)


def holdout_of(frame, statements=None, **kw):
    """The holdout rows of a run of `frame`. A test that then changes a column other than
    the target, the time or the entity column and runs again asserts that the rows it
    chose are still where it chose them."""
    return sorted(h.holdout_rows(run(frame, statements, **kw)))


def training_of(frame, statements=None, **kw):
    held = set(holdout_of(frame, statements, **kw))
    return [r for r in range(len(frame)) if r not in held]


def write_csv(frame, folder, name="table.csv"):
    path = folder / name
    frame.to_csv(path, index=False)
    return path


def summary_line(stdout):
    """The lines of the console summary that begin with `rests on`."""
    return [line.strip() for line in stdout.splitlines()
            if line.strip().lower().startswith("rests on")]


def holds_value(text, value):
    """Some number printed in `text` agrees with `value` to four significant digits
    (D46): within 0.0005 x |value|."""
    return any(abs(n - value) <= 0.0005 * abs(value) + 1e-12 for n in s.numbers_in(text))


# ------------------------------------------------------------------- tables

def pair_table(task="regression", n=400, seed=0):
    """`z1` noise; `pair_a` = the target plus a noise of its spread (0.5 for a class),
    `pair_b` that noise; the target `y`. Neither column alone gives the target."""
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    if task == "regression":
        y = rng.normal(size=n)
        noise = rng.normal(size=n) * float(np.std(y))
    else:
        y = rng.integers(0, 2, size=n)
        noise = 0.5 * rng.normal(size=n)
    return pd.DataFrame({"z1": z1, "pair_a": y + noise, "pair_b": noise, "y": y})


def bands_of(values, k):
    """The band, 0 to k - 1, of each value: k bands of equal count, by rank."""
    order = np.argsort(np.asarray(values, dtype=float), kind="stable")
    band = np.empty(len(order), dtype=int)
    band[order] = (np.arange(len(order)) * k) // len(order)
    return band


def coded(band, k, seed=0, text=True):
    """A code of `band` (0 to k - 1) with labels in random order: text, or whole
    numbers."""
    rng = np.random.default_rng(seed)
    label = rng.permutation(k)
    if text:
        names = [f"L{v:03d}" for v in label]
        return [names[b] for b in band]
    return (label[np.asarray(band)] * 7 + 11).astype(np.int64)


def coded_copy_table(task="regression", n=600, seed=0, text=True, k=None):
    """`x1` carries some of the target, `z1` is noise, `code` is a coded copy: the target
    in `k` bands (30 by default) for a regression; for a class, `k` levels (40 by
    default) each of one class."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    z1 = rng.normal(size=n)
    if task == "regression":
        k = 30 if k is None else k
        y = x1 + rng.normal(size=n)
        band = bands_of(y, k)
    else:
        k = 40 if k is None else k
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-1.5 * x1))).astype(int)
        band = level_of_class(y, k, rng)
    return pd.DataFrame({"x1": x1, "z1": z1,
                         "code": coded(band, k, seed=seed + 1, text=text), "y": y})


def few_scored_table(rare=1, n=600, seed=9305):
    """A classification: the rows of class `1 - rare` hold one of five levels that sort
    first (`a0` to `a4`); six rows of class `rare` hold `p0`, and every other row of that
    class a level of its own. Read by its levels the column is perfect on the rows it
    scores, which hold six rows of class `rare`."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    y = np.full(n, 1 - rare)
    y[rng.choice(n, size=n // 4, replace=False)] = rare
    names = rng.permutation(100_000)[:n]
    code = []
    seen_rare = 0
    for i in range(n):
        if y[i] != rare:
            code.append(f"a{rng.integers(0, 5)}")
        elif seen_rare < 6:
            code.append("p0")
            seen_rare += 1
        else:
            code.append(f"q{names[i]:05d}")
    return pd.DataFrame({"x1": x1, "z1": rng.normal(size=n), "code": code, "y": y})


def level_of_class(y, k, rng):
    """`k` levels, the first half for class 0 and the rest for class 1, each row given at
    random one level of its class."""
    y = np.asarray(y)
    half = k // 2
    out = np.empty(len(y), dtype=int)
    zeros, ones = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    out[zeros] = rng.integers(0, half, size=len(zeros))
    out[ones] = half + rng.integers(0, k - half, size=len(ones))
    return out


def repeated_noise(n, rng):
    """Two columns of noise with repeated values: a whole number from 1 to 40, and a
    normal draw rounded to one decimal place."""
    return rng.integers(1, 41, size=n).astype(np.int64), np.round(rng.normal(size=n), 1)


def lognormal_table(n=600, seed=0):
    """A log-normal target `y` and two noise columns: its tail lies more than 10 median
    deviations from the median on several percent of the rows."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"n1": rng.normal(size=n), "n2": rng.normal(size=n),
                         "y": np.exp(rng.normal(size=n))})


def moved_copy(y, rows, spreads):
    """1.07 x y + 3, with the cells `rows` moved up by `spreads` times the target's
    standard deviation."""
    y = np.asarray(y, dtype=float)
    out = 1.07 * y + 3.0
    out[list(rows)] += spreads * float(np.std(y))
    return out


def last_date_table(n=1000, share=0.6, seed=0):
    """`date`, `x1`, `y`: the first n - k rows one a day, the last k = share x n rows all
    on the day after."""
    rng = np.random.default_rng(seed)
    k = int(round(share * n))
    days = list(pd.date_range("2020-01-01", periods=n - k + 1, freq="D"))
    when = days[:n - k] + [days[n - k]] * k
    x1 = rng.normal(size=n)
    return pd.DataFrame({"date": when, "x1": x1, "z1": rng.normal(size=n),
                         "y": 2.0 * x1 + rng.normal(size=n)})


def entities(n_entities, per, seed=0):
    """`who` (`w000`, ...) with `per` rows each, in shuffled order."""
    rng = np.random.default_rng(seed)
    who = np.repeat([f"w{k:03d}" for k in range(n_entities)], per)
    return list(who[rng.permutation(len(who))])


# -------------------------------------------------------------- seam families

class RecordingModel(h.WrapEstimator):
    """A real linear model that records every matrix it predicts on, and what it
    predicted, in `log.matrices` as `(oid, matrix, prediction)`. Its predictions are a
    function of the matrix it is handed and of its fit."""

    def _keep(self, X, out):
        if not hasattr(self.log, "matrices"):
            self.log.matrices = []
        self.log.matrices.append((self.oid, X.copy(), np.array(out, dtype=float)))

    def predict_proba(self, X):
        out = super().predict_proba(X)
        self._keep(X, np.asarray(out)[:, 1])
        return out

    def predict(self, X):
        out = super().predict(X)
        self._keep(X, out)
        return out


def recording():
    """`(log, families)` with one recording linear family named `rec`."""
    log = h.Log()
    log.matrices = []
    return log, [h.family("rec", log, estimator=RecordingModel)]


def holdout_calls(log, hold):
    """The recorded `(oid, matrix, prediction)` whose rows are holdout rows, in order."""
    hold = set(hold)
    return [m for m in log.matrices if set(int(i) for i in m[1].index) & hold]


class ShuffleFailingModel(h.WrapEstimator):
    """A real linear model that, once it has predicted, fails on any later matrix whose
    column `column` (every column when None) differs from the one it was first handed:
    it raises, or with `infinite` predicts infinity."""

    def __init__(self, *args, column=None, infinite=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.column, self.infinite, self.first = column, infinite, None

    def _check(self, X):
        if self.first is None:
            self.first = X.copy()
            return True
        if list(X.index) != list(self.first.index):
            return True
        names = list(X.columns) if self.column is None else [
            c for c in X.columns if c == self.column or str(c).startswith(f"{self.column}=")]
        return all(np.array_equal(np.asarray(X[c], dtype=float),
                                  np.asarray(self.first[c], dtype=float)) for c in names)

    def predict_proba(self, X):
        if self._check(X):
            return super().predict_proba(X)
        if self.infinite:
            return np.full((len(X), 2), np.inf)
        raise RuntimeError("deliberate failure on a shuffled matrix")

    def predict(self, X):
        if self._check(X):
            return super().predict(X)
        if self.infinite:
            return np.full(len(X), np.inf)
        raise RuntimeError("deliberate failure on a shuffled matrix")


def failing_on_shuffle(column=None, infinite=False):
    """`(log, families)`: one family named `rec` of `ShuffleFailingModel`."""
    log = h.Log()
    return log, [h.family("rec", log, estimator=ShuffleFailingModel, column=column,
                          infinite=infinite)]


def plain_family():
    """`(log, families)`: one family named `rec`, the same linear model, failing never."""
    log = h.Log()
    return log, [h.family("rec", log)]


# ---------------------------------------------- what pass 9 owns (D78 point 8)
# The pairs of the battery and the tables of the generated-table test that fail on the
# application as the eighth pass left it, as docs/measurements/CP10/owned/
# owned_by_pass9.json lists them (keys `battery` and `tables`). Not shortened, not
# added to.

BATTERY_OWNED = frozenset({
    'binary_numeric:base', 'binary_numeric:blank_line_added', 'binary_numeric:coded_copy_number', 'binary_numeric:coded_copy_text', 'binary_numeric:constant_added', 'binary_numeric:copy_distinct', 'binary_numeric:copy_two_valued', 'binary_numeric:feature_renamed',
    'binary_numeric:feature_renamed_awkward', 'binary_numeric:feature_rounded_10', 'binary_numeric:feature_rounded_40', 'binary_numeric:feature_written_twice', 'binary_numeric:float_whole_numbers', 'binary_numeric:id_added_gaps', 'binary_numeric:id_added_int', 'binary_numeric:id_added_text',
    'binary_numeric:missing_cell_feature', 'binary_numeric:missing_cell_target', 'binary_numeric:noise_added', 'binary_numeric:noise_rounded_1dp', 'binary_numeric:noise_score_1_to_40', 'binary_numeric:number_sentinel_-999', 'binary_numeric:number_sentinel_1e308', 'binary_numeric:number_sentinel_999999',
    'binary_numeric:number_two_sentinels', 'binary_numeric:row_repeated', 'binary_numeric:rows_shuffled', 'binary_numeric:second_date_added', 'binary_numeric:target_first', 'binary_numeric:target_renamed', 'binary_numeric:target_respelt', 'binary_numeric:target_respelt_flipped',
    'binary_numeric:two_columns_together', 'binary_numeric:white_space_cells', 'day_numbers:base', 'day_numbers:blank_line_added', 'day_numbers:coded_copy_number', 'day_numbers:coded_copy_text', 'day_numbers:constant_added', 'day_numbers:copy_distinct',
    'day_numbers:copy_two_valued', 'day_numbers:feature_renamed', 'day_numbers:feature_renamed_awkward', 'day_numbers:feature_rounded_10', 'day_numbers:feature_rounded_40', 'day_numbers:feature_written_twice', 'day_numbers:float_whole_numbers', 'day_numbers:id_added_gaps',
    'day_numbers:id_added_int', 'day_numbers:id_added_text', 'day_numbers:missing_cell_date', 'day_numbers:missing_cell_feature', 'day_numbers:missing_cell_target', 'day_numbers:noise_added', 'day_numbers:noise_rounded_1dp', 'day_numbers:noise_score_1_to_40',
    'day_numbers:number_sentinel_-999', 'day_numbers:number_sentinel_1e308', 'day_numbers:number_sentinel_999999', 'day_numbers:number_two_sentinels', 'day_numbers:role_column_renamed', 'day_numbers:row_repeated', 'day_numbers:rows_shuffled', 'day_numbers:second_date_added',
    'day_numbers:target_first', 'day_numbers:target_renamed', 'day_numbers:target_respelt', 'day_numbers:target_respelt_flipped', 'day_numbers:two_columns_together', 'day_numbers:white_space_cells', 'day_numbers:ymd_sentinel_0', 'day_numbers:ymd_sentinel_20240230',
    'day_numbers:ymd_sentinel_99991231', 'day_numbers:ymd_sentinel_in_dropped_row', 'day_sentinel:base', 'day_sentinel:blank_line_added', 'day_sentinel:coded_copy_number', 'day_sentinel:coded_copy_text', 'day_sentinel:constant_added', 'day_sentinel:copy_distinct',
    'day_sentinel:copy_two_valued', 'day_sentinel:feature_renamed', 'day_sentinel:feature_renamed_awkward', 'day_sentinel:feature_rounded_10', 'day_sentinel:feature_rounded_40', 'day_sentinel:feature_written_twice', 'day_sentinel:float_whole_numbers', 'day_sentinel:id_added_gaps',
    'day_sentinel:id_added_int', 'day_sentinel:id_added_text', 'day_sentinel:missing_cell_date', 'day_sentinel:missing_cell_feature', 'day_sentinel:missing_cell_target', 'day_sentinel:noise_added', 'day_sentinel:noise_rounded_1dp', 'day_sentinel:noise_score_1_to_40',
    'day_sentinel:number_sentinel_-999', 'day_sentinel:number_sentinel_1e308', 'day_sentinel:number_sentinel_999999', 'day_sentinel:number_two_sentinels', 'day_sentinel:role_column_renamed', 'day_sentinel:row_repeated', 'day_sentinel:rows_shuffled', 'day_sentinel:second_date_added',
    'day_sentinel:target_first', 'day_sentinel:target_renamed', 'day_sentinel:target_respelt', 'day_sentinel:target_respelt_flipped', 'day_sentinel:two_columns_together', 'day_sentinel:white_space_cells', 'day_sentinel:ymd_sentinel_0', 'day_sentinel:ymd_sentinel_20240230',
    'day_sentinel:ymd_sentinel_99991231', 'day_sentinel:ymd_sentinel_in_dropped_row', 'dominant_entity:base', 'dominant_entity:blank_line_added', 'dominant_entity:coded_copy_number', 'dominant_entity:coded_copy_text', 'dominant_entity:constant_added', 'dominant_entity:copy_5_cells_differ',
    'dominant_entity:copy_5_cells_other_row', 'dominant_entity:copy_5_cells_plus_1', 'dominant_entity:copy_5_cells_plus_1000', 'dominant_entity:copy_5_cells_plus_20', 'dominant_entity:copy_5_cells_times_1.5', 'dominant_entity:copy_and_target_sentinel', 'dominant_entity:copy_distinct', 'dominant_entity:copy_exact',
    'dominant_entity:copy_negative_scale', 'dominant_entity:copy_one_cell_999999', 'dominant_entity:feature_renamed', 'dominant_entity:feature_renamed_awkward', 'dominant_entity:feature_rounded_10', 'dominant_entity:feature_rounded_40', 'dominant_entity:feature_written_twice', 'dominant_entity:id_added_gaps',
    'dominant_entity:id_added_int', 'dominant_entity:id_added_text', 'dominant_entity:missing_cell_feature', 'dominant_entity:missing_cell_target', 'dominant_entity:noise_added', 'dominant_entity:noise_rounded_1dp', 'dominant_entity:noise_score_1_to_40', 'dominant_entity:number_sentinel_-999',
    'dominant_entity:number_sentinel_1e308', 'dominant_entity:number_sentinel_999999', 'dominant_entity:number_two_sentinels', 'dominant_entity:role_column_renamed', 'dominant_entity:row_repeated', 'dominant_entity:rows_shuffled', 'dominant_entity:second_date_added', 'dominant_entity:target_first',
    'dominant_entity:target_renamed', 'dominant_entity:target_sentinel_-999', 'dominant_entity:target_sentinel_999999', 'dominant_entity:target_two_sentinels', 'dominant_entity:two_columns_together', 'dominant_entity:white_space_cells', 'dup_noise:duplicates_group', 'dup_noise:duplicates_keep',
    'entities:base', 'entities:blank_line_added', 'entities:coded_copy_number', 'entities:coded_copy_text', 'entities:constant_added', 'entities:copy_distinct', 'entities:copy_two_valued', 'entities:entities_merged_45pct',
    'entities:feature_renamed', 'entities:feature_renamed_awkward', 'entities:feature_rounded_10', 'entities:feature_rounded_40', 'entities:feature_written_twice', 'entities:id_added_gaps', 'entities:id_added_int', 'entities:id_added_text',
    'entities:missing_cell_feature', 'entities:missing_cell_target', 'entities:noise_added', 'entities:noise_rounded_1dp', 'entities:noise_score_1_to_40', 'entities:number_sentinel_-999', 'entities:number_sentinel_1e308', 'entities:number_sentinel_999999',
    'entities:number_two_sentinels', 'entities:role_column_renamed', 'entities:row_repeated', 'entities:rows_shuffled', 'entities:second_date_added', 'entities:target_first', 'entities:target_renamed', 'entities:target_respelt',
    'entities:target_respelt_flipped', 'entities:two_columns_together', 'entities:white_space_cells', 'few_valued:base', 'few_valued:blank_line_added', 'few_valued:constant_added', 'few_valued:duplicates_keep', 'few_valued:feature_renamed',
    'few_valued:feature_renamed_awkward', 'few_valued:id_added_gaps', 'few_valued:id_added_int', 'few_valued:id_added_text', 'few_valued:missing_cell_feature', 'few_valued:missing_cell_target', 'few_valued:noise_added', 'few_valued:noise_rounded_1dp',
    'few_valued:noise_score_1_to_40', 'few_valued:row_repeated', 'few_valued:rows_shuffled', 'few_valued:second_date_added', 'few_valued:target_first', 'few_valued:target_renamed', 'few_valued:target_respelt', 'few_valued:target_respelt_flipped',
    'few_valued:white_space_cells', 'few_valued_leak:duplicates_keep', 'flat_tail:base', 'flat_tail:blank_line_added', 'flat_tail:coded_copy_number', 'flat_tail:coded_copy_text', 'flat_tail:constant_added', 'flat_tail:copy_2pct_cells_altered',
    'flat_tail:copy_5_cells_differ', 'flat_tail:copy_5_cells_other_row', 'flat_tail:copy_5_cells_plus_1', 'flat_tail:copy_5_cells_plus_1000', 'flat_tail:copy_5_cells_plus_20', 'flat_tail:copy_5_cells_times_1.5', 'flat_tail:copy_and_target_sentinel', 'flat_tail:copy_curve',
    'flat_tail:copy_distinct', 'flat_tail:copy_exact', 'flat_tail:copy_negative_scale', 'flat_tail:copy_noise_1pct', 'flat_tail:copy_one_cell_999999', 'flat_tail:date_sentinel_1900', 'flat_tail:feature_999999_early_row', 'flat_tail:feature_999999_latest_row',
    'flat_tail:feature_renamed', 'flat_tail:feature_renamed_awkward', 'flat_tail:feature_rounded_10', 'flat_tail:feature_rounded_40', 'flat_tail:feature_written_twice', 'flat_tail:gap_2', 'flat_tail:gap_2_row_repeated', 'flat_tail:id_added_gaps',
    'flat_tail:id_added_int', 'flat_tail:id_added_text', 'flat_tail:missing_cell_date', 'flat_tail:missing_cell_feature', 'flat_tail:missing_cell_target', 'flat_tail:noise_added', 'flat_tail:noise_rounded_1dp', 'flat_tail:noise_score_1_to_40',
    'flat_tail:number_sentinel_-999', 'flat_tail:number_sentinel_1e308', 'flat_tail:number_sentinel_999999', 'flat_tail:number_two_sentinels', 'flat_tail:role_column_renamed', 'flat_tail:row_repeated', 'flat_tail:row_repeated_other_target', 'flat_tail:rows_repeated_10pct',
    'flat_tail:rows_repeated_1pct', 'flat_tail:rows_shuffled', 'flat_tail:second_date_added', 'flat_tail:stamp_shared', 'flat_tail:target_-999_early_row', 'flat_tail:target_-999_latest_row', 'flat_tail:target_first', 'flat_tail:target_renamed',
    'flat_tail:target_sentinel_-999', 'flat_tail:target_sentinel_999999', 'flat_tail:target_two_sentinels', 'imbalanced:base', 'imbalanced:blank_line_added', 'imbalanced:coded_copy_number', 'imbalanced:coded_copy_text', 'imbalanced:constant_added',
    'imbalanced:copy_distinct', 'imbalanced:copy_two_valued', 'imbalanced:feature_renamed', 'imbalanced:feature_renamed_awkward', 'imbalanced:feature_rounded_10', 'imbalanced:feature_rounded_40', 'imbalanced:feature_written_twice', 'imbalanced:id_added_gaps',
    'imbalanced:id_added_int', 'imbalanced:id_added_text', 'imbalanced:missing_cell_feature', 'imbalanced:missing_cell_target', 'imbalanced:noise_added', 'imbalanced:noise_rounded_1dp', 'imbalanced:noise_score_1_to_40', 'imbalanced:number_sentinel_-999',
    'imbalanced:number_sentinel_1e308', 'imbalanced:number_sentinel_999999', 'imbalanced:number_two_sentinels', 'imbalanced:row_repeated', 'imbalanced:rows_shuffled', 'imbalanced:second_date_added', 'imbalanced:target_first', 'imbalanced:target_renamed',
    'imbalanced:target_respelt', 'imbalanced:target_respelt_flipped', 'imbalanced:two_columns_together', 'imbalanced:white_space_cells', 'int_stamps:base', 'int_stamps:blank_line_added', 'int_stamps:coded_copy_number', 'int_stamps:coded_copy_text',
    'int_stamps:constant_added', 'int_stamps:copy_2pct_cells_altered', 'int_stamps:copy_5_cells_differ', 'int_stamps:copy_5_cells_other_row', 'int_stamps:copy_5_cells_plus_1', 'int_stamps:copy_5_cells_plus_1000', 'int_stamps:copy_5_cells_plus_20', 'int_stamps:copy_5_cells_times_1.5',
    'int_stamps:copy_and_target_sentinel', 'int_stamps:copy_curve', 'int_stamps:copy_distinct', 'int_stamps:copy_exact', 'int_stamps:copy_negative_scale', 'int_stamps:copy_noise_1pct', 'int_stamps:copy_one_cell_999999', 'int_stamps:feature_999999_early_row',
    'int_stamps:feature_999999_latest_row', 'int_stamps:feature_renamed', 'int_stamps:feature_renamed_awkward', 'int_stamps:feature_rounded_10', 'int_stamps:feature_rounded_40', 'int_stamps:feature_written_twice', 'int_stamps:gap_2', 'int_stamps:gap_2_row_repeated',
    'int_stamps:id_added_gaps', 'int_stamps:id_added_int', 'int_stamps:id_added_text', 'int_stamps:int_time_sentinel_0', 'int_stamps:int_time_sentinel_99991231', 'int_stamps:missing_cell_feature', 'int_stamps:missing_cell_target', 'int_stamps:noise_added',
    'int_stamps:noise_rounded_1dp', 'int_stamps:noise_score_1_to_40', 'int_stamps:number_sentinel_-999', 'int_stamps:number_sentinel_1e308', 'int_stamps:number_sentinel_999999', 'int_stamps:number_two_sentinels', 'int_stamps:role_column_renamed', 'int_stamps:row_repeated',
    'int_stamps:row_repeated_other_target', 'int_stamps:rows_repeated_10pct', 'int_stamps:rows_repeated_1pct', 'int_stamps:rows_shuffled', 'int_stamps:second_date_added', 'int_stamps:stamp_shared', 'int_stamps:target_-999_early_row', 'int_stamps:target_-999_latest_row',
    'int_stamps:target_first', 'int_stamps:target_renamed', 'int_stamps:target_sentinel_-999', 'int_stamps:target_sentinel_999999', 'int_stamps:target_two_sentinels', 'int_stamps:two_columns_together', 'leak:base', 'leak:blank_line_added',
    'leak:coded_copy_number', 'leak:coded_copy_text', 'leak:constant_added', 'leak:copy_distinct', 'leak:copy_two_valued', 'leak:feature_renamed', 'leak:feature_renamed_awkward', 'leak:feature_rounded_10',
    'leak:feature_rounded_40', 'leak:feature_written_twice', 'leak:id_added_gaps', 'leak:id_added_int', 'leak:id_added_text', 'leak:missing_cell_feature', 'leak:missing_cell_target', 'leak:noise_added',
    'leak:noise_rounded_1dp', 'leak:noise_score_1_to_40', 'leak:number_sentinel_-999', 'leak:number_sentinel_1e308', 'leak:number_sentinel_999999', 'leak:number_two_sentinels', 'leak:row_repeated', 'leak:rows_shuffled',
    'leak:second_date_added', 'leak:target_first', 'leak:target_renamed', 'leak:target_respelt', 'leak:target_respelt_flipped', 'leak:two_columns_together', 'leak:white_space_cells', 'leak_reg:base',
    'leak_reg:blank_line_added', 'leak_reg:coded_copy_number', 'leak_reg:coded_copy_text', 'leak_reg:constant_added', 'leak_reg:copy_5_cells_differ', 'leak_reg:copy_5_cells_other_row', 'leak_reg:copy_5_cells_plus_1', 'leak_reg:copy_5_cells_plus_1000',
    'leak_reg:copy_5_cells_plus_20', 'leak_reg:copy_5_cells_times_1.5', 'leak_reg:copy_and_target_sentinel', 'leak_reg:copy_distinct', 'leak_reg:copy_exact', 'leak_reg:copy_negative_scale', 'leak_reg:copy_one_cell_999999', 'leak_reg:feature_renamed',
    'leak_reg:feature_renamed_awkward', 'leak_reg:feature_rounded_10', 'leak_reg:feature_rounded_40', 'leak_reg:feature_written_twice', 'leak_reg:id_added_gaps', 'leak_reg:id_added_int', 'leak_reg:id_added_text', 'leak_reg:missing_cell_feature',
    'leak_reg:missing_cell_target', 'leak_reg:noise_added', 'leak_reg:noise_rounded_1dp', 'leak_reg:noise_score_1_to_40', 'leak_reg:number_sentinel_-999', 'leak_reg:number_sentinel_1e308', 'leak_reg:number_sentinel_999999', 'leak_reg:number_two_sentinels',
    'leak_reg:row_repeated', 'leak_reg:rows_shuffled', 'leak_reg:second_date_added', 'leak_reg:target_first', 'leak_reg:target_renamed', 'leak_reg:target_sentinel_-999', 'leak_reg:target_sentinel_999999', 'leak_reg:target_two_sentinels',
    'leak_reg:two_columns_together', 'leak_reg:white_space_cells', 'messy:base', 'messy:blank_line_added', 'messy:coded_copy_number', 'messy:coded_copy_text', 'messy:constant_added', 'messy:copy_distinct',
    'messy:copy_two_valued', 'messy:date_sentinel_1900', 'messy:date_sentinel_9999', 'messy:date_sentinel_zero', 'messy:feature_renamed', 'messy:feature_renamed_awkward', 'messy:feature_rounded_10', 'messy:feature_rounded_40',
    'messy:feature_written_twice', 'messy:float_whole_numbers', 'messy:id_added_gaps', 'messy:id_added_int', 'messy:id_added_text', 'messy:id_floats_shuffled', 'messy:id_shuffled_and_repeated', 'messy:id_steps_by_7',
    'messy:measure_sentinel_-999', 'messy:missing_cell_date', 'messy:missing_cell_feature', 'messy:missing_cell_target', 'messy:noise_added', 'messy:noise_rounded_1dp', 'messy:noise_score_1_to_40', 'messy:number_sentinel_-999',
    'messy:number_sentinel_1e308', 'messy:number_sentinel_999999', 'messy:number_two_sentinels', 'messy:role_column_renamed', 'messy:row_repeated', 'messy:rows_deleted_shuffled', 'messy:rows_shuffled', 'messy:second_date_added',
    'messy:target_first', 'messy:target_renamed', 'messy:target_respelt', 'messy:target_respelt_flipped', 'messy:two_columns_together', 'messy:white_space_cells', 'messy_shuffled:base', 'messy_shuffled:blank_line_added',
    'messy_shuffled:coded_copy_number', 'messy_shuffled:coded_copy_text', 'messy_shuffled:constant_added', 'messy_shuffled:copy_distinct', 'messy_shuffled:copy_two_valued', 'messy_shuffled:date_sentinel_1900', 'messy_shuffled:date_sentinel_9999', 'messy_shuffled:date_sentinel_zero',
    'messy_shuffled:feature_renamed', 'messy_shuffled:feature_renamed_awkward', 'messy_shuffled:feature_rounded_10', 'messy_shuffled:feature_rounded_40', 'messy_shuffled:feature_written_twice', 'messy_shuffled:id_added_gaps', 'messy_shuffled:id_added_int', 'messy_shuffled:id_added_text',
    'messy_shuffled:missing_cell_date', 'messy_shuffled:missing_cell_feature', 'messy_shuffled:missing_cell_target', 'messy_shuffled:noise_added', 'messy_shuffled:noise_rounded_1dp', 'messy_shuffled:noise_score_1_to_40', 'messy_shuffled:number_sentinel_-999', 'messy_shuffled:number_sentinel_1e308',
    'messy_shuffled:number_sentinel_999999', 'messy_shuffled:number_two_sentinels', 'messy_shuffled:role_column_renamed', 'messy_shuffled:row_repeated', 'messy_shuffled:rows_shuffled', 'messy_shuffled:second_date_added', 'messy_shuffled:target_first', 'messy_shuffled:target_renamed',
    'messy_shuffled:target_respelt', 'messy_shuffled:target_respelt_flipped', 'messy_shuffled:two_columns_together', 'messy_shuffled:white_space_cells', 'near_leak:base', 'near_leak:blank_line_added', 'near_leak:coded_copy_number', 'near_leak:coded_copy_text',
    'near_leak:constant_added', 'near_leak:copy_distinct', 'near_leak:copy_two_valued', 'near_leak:feature_renamed', 'near_leak:feature_renamed_awkward', 'near_leak:feature_rounded_10', 'near_leak:feature_rounded_40', 'near_leak:feature_written_twice',
    'near_leak:id_added_gaps', 'near_leak:id_added_int', 'near_leak:id_added_text', 'near_leak:missing_cell_feature', 'near_leak:missing_cell_target', 'near_leak:noise_added', 'near_leak:noise_rounded_1dp', 'near_leak:noise_score_1_to_40',
    'near_leak:number_sentinel_-999', 'near_leak:number_sentinel_1e308', 'near_leak:number_sentinel_999999', 'near_leak:number_two_sentinels', 'near_leak:row_repeated', 'near_leak:rows_shuffled', 'near_leak:second_date_added', 'near_leak:target_first',
    'near_leak:target_renamed', 'near_leak:target_respelt', 'near_leak:target_respelt_flipped', 'near_leak:two_columns_together', 'near_leak:white_space_cells', 'noise_clf:base', 'noise_clf:blank_line_added', 'noise_clf:coded_copy_number',
    'noise_clf:coded_copy_text', 'noise_clf:constant_added', 'noise_clf:copy_distinct', 'noise_clf:copy_two_valued', 'noise_clf:feature_renamed', 'noise_clf:feature_renamed_awkward', 'noise_clf:feature_rounded_10', 'noise_clf:feature_rounded_40',
    'noise_clf:feature_written_twice', 'noise_clf:id_added_gaps', 'noise_clf:id_added_int', 'noise_clf:id_added_text', 'noise_clf:missing_cell_feature', 'noise_clf:missing_cell_target', 'noise_clf:noise_added', 'noise_clf:noise_rounded_1dp',
    'noise_clf:noise_score_1_to_40', 'noise_clf:number_sentinel_-999', 'noise_clf:number_sentinel_1e308', 'noise_clf:number_sentinel_999999', 'noise_clf:number_two_sentinels', 'noise_clf:row_repeated', 'noise_clf:rows_shuffled', 'noise_clf:second_date_added',
    'noise_clf:target_first', 'noise_clf:target_renamed', 'noise_clf:target_respelt', 'noise_clf:target_respelt_flipped', 'noise_clf:two_columns_together', 'noise_clf:white_space_cells', 'noise_reg:base', 'noise_reg:blank_line_added',
    'noise_reg:coded_copy_number', 'noise_reg:coded_copy_text', 'noise_reg:constant_added', 'noise_reg:copy_5_cells_differ', 'noise_reg:copy_5_cells_other_row', 'noise_reg:copy_5_cells_plus_1', 'noise_reg:copy_5_cells_plus_1000', 'noise_reg:copy_5_cells_plus_20',
    'noise_reg:copy_5_cells_times_1.5', 'noise_reg:copy_and_target_sentinel', 'noise_reg:copy_distinct', 'noise_reg:copy_exact', 'noise_reg:copy_negative_scale', 'noise_reg:copy_one_cell_999999', 'noise_reg:feature_renamed', 'noise_reg:feature_renamed_awkward',
    'noise_reg:feature_rounded_10', 'noise_reg:feature_rounded_40', 'noise_reg:feature_written_twice', 'noise_reg:id_added_gaps', 'noise_reg:id_added_int', 'noise_reg:id_added_text', 'noise_reg:missing_cell_feature', 'noise_reg:missing_cell_target',
    'noise_reg:noise_added', 'noise_reg:noise_rounded_1dp', 'noise_reg:noise_score_1_to_40', 'noise_reg:number_sentinel_-999', 'noise_reg:number_sentinel_1e308', 'noise_reg:number_sentinel_999999', 'noise_reg:number_two_sentinels', 'noise_reg:row_repeated',
    'noise_reg:rows_shuffled', 'noise_reg:second_date_added', 'noise_reg:target_first', 'noise_reg:target_renamed', 'noise_reg:target_sentinel_-999', 'noise_reg:target_sentinel_999999', 'noise_reg:target_two_sentinels', 'noise_reg:two_columns_together',
    'panel:base', 'panel:blank_line_added', 'panel:coded_copy_number', 'panel:coded_copy_text', 'panel:constant_added', 'panel:copy_2pct_cells_altered', 'panel:copy_5_cells_differ', 'panel:copy_5_cells_other_row',
    'panel:copy_5_cells_plus_1', 'panel:copy_5_cells_plus_1000', 'panel:copy_5_cells_plus_20', 'panel:copy_5_cells_times_1.5', 'panel:copy_and_target_sentinel', 'panel:copy_curve', 'panel:copy_distinct', 'panel:copy_exact',
    'panel:copy_negative_scale', 'panel:copy_noise_1pct', 'panel:copy_one_cell_999999', 'panel:date_sentinel_1900', 'panel:feature_999999_early_row', 'panel:feature_999999_latest_row', 'panel:feature_renamed', 'panel:feature_renamed_awkward',
    'panel:feature_rounded_10', 'panel:feature_rounded_40', 'panel:feature_written_twice', 'panel:id_added_gaps', 'panel:id_added_int', 'panel:id_added_text', 'panel:missing_cell_date', 'panel:missing_cell_feature',
    'panel:missing_cell_target', 'panel:noise_added', 'panel:noise_rounded_1dp', 'panel:noise_score_1_to_40', 'panel:number_sentinel_-999', 'panel:number_sentinel_1e308', 'panel:number_sentinel_999999', 'panel:number_two_sentinels',
    'panel:role_column_renamed', 'panel:rows_shuffled', 'panel:second_date_added', 'panel:target_-999_early_row', 'panel:target_-999_latest_row', 'panel:target_first', 'panel:target_renamed', 'panel:target_sentinel_-999',
    'panel:target_sentinel_999999', 'panel:target_two_sentinels', 'panel:two_columns_together', 'panel:white_space_cells', 'quarter_entity:base', 'quarter_entity:blank_line_added', 'quarter_entity:coded_copy_number', 'quarter_entity:coded_copy_text',
    'quarter_entity:constant_added', 'quarter_entity:copy_5_cells_differ', 'quarter_entity:copy_5_cells_other_row', 'quarter_entity:copy_5_cells_plus_1', 'quarter_entity:copy_5_cells_plus_1000', 'quarter_entity:copy_5_cells_plus_20', 'quarter_entity:copy_5_cells_times_1.5', 'quarter_entity:copy_and_target_sentinel',
    'quarter_entity:copy_distinct', 'quarter_entity:copy_exact', 'quarter_entity:copy_negative_scale', 'quarter_entity:copy_one_cell_999999', 'quarter_entity:feature_renamed', 'quarter_entity:feature_renamed_awkward', 'quarter_entity:feature_rounded_10', 'quarter_entity:feature_rounded_40',
    'quarter_entity:feature_written_twice', 'quarter_entity:id_added_gaps', 'quarter_entity:id_added_int', 'quarter_entity:id_added_text', 'quarter_entity:missing_cell_feature', 'quarter_entity:missing_cell_target', 'quarter_entity:noise_added', 'quarter_entity:noise_rounded_1dp',
    'quarter_entity:noise_score_1_to_40', 'quarter_entity:number_sentinel_-999', 'quarter_entity:number_sentinel_1e308', 'quarter_entity:number_sentinel_999999', 'quarter_entity:number_two_sentinels', 'quarter_entity:role_column_renamed', 'quarter_entity:row_repeated', 'quarter_entity:rows_shuffled',
    'quarter_entity:second_date_added', 'quarter_entity:target_first', 'quarter_entity:target_renamed', 'quarter_entity:target_sentinel_-999', 'quarter_entity:target_sentinel_999999', 'quarter_entity:target_two_sentinels', 'quarter_entity:two_columns_together', 'quarter_entity:white_space_cells',
    'random_walk:base', 'random_walk:blank_line_added', 'random_walk:coded_copy_number', 'random_walk:coded_copy_text', 'random_walk:constant_added', 'random_walk:copy_2pct_cells_altered', 'random_walk:copy_5_cells_differ', 'random_walk:copy_5_cells_other_row',
    'random_walk:copy_5_cells_plus_1', 'random_walk:copy_5_cells_plus_1000', 'random_walk:copy_5_cells_plus_20', 'random_walk:copy_5_cells_times_1.5', 'random_walk:copy_and_target_sentinel', 'random_walk:copy_curve', 'random_walk:copy_distinct', 'random_walk:copy_exact',
    'random_walk:copy_negative_scale', 'random_walk:copy_noise_1pct', 'random_walk:copy_one_cell_999999', 'random_walk:date_sentinel_1900', 'random_walk:feature_999999_early_row', 'random_walk:feature_999999_latest_row', 'random_walk:feature_renamed', 'random_walk:feature_renamed_awkward',
    'random_walk:feature_rounded_10', 'random_walk:feature_rounded_40', 'random_walk:feature_written_twice', 'random_walk:gap_2', 'random_walk:gap_2_row_repeated', 'random_walk:id_added_gaps', 'random_walk:id_added_int', 'random_walk:id_added_text',
    'random_walk:missing_cell_date', 'random_walk:missing_cell_feature', 'random_walk:missing_cell_target', 'random_walk:noise_added', 'random_walk:noise_rounded_1dp', 'random_walk:noise_score_1_to_40', 'random_walk:number_sentinel_-999', 'random_walk:number_sentinel_1e308',
    'random_walk:number_sentinel_999999', 'random_walk:number_two_sentinels', 'random_walk:role_column_renamed', 'random_walk:row_repeated', 'random_walk:row_repeated_other_target', 'random_walk:rows_repeated_10pct', 'random_walk:rows_repeated_1pct', 'random_walk:rows_shuffled',
    'random_walk:second_date_added', 'random_walk:stamp_shared', 'random_walk:target_-999_early_row', 'random_walk:target_-999_latest_row', 'random_walk:target_first', 'random_walk:target_renamed', 'random_walk:target_sentinel_-999', 'random_walk:target_sentinel_999999',
    'random_walk:target_two_sentinels', 'random_walk:two_columns_together', 'signal_clf:base', 'signal_clf:blank_line_added', 'signal_clf:coded_copy_number', 'signal_clf:coded_copy_text', 'signal_clf:constant_added', 'signal_clf:copy_distinct',
    'signal_clf:copy_two_valued', 'signal_clf:feature_renamed', 'signal_clf:feature_renamed_awkward', 'signal_clf:feature_rounded_10', 'signal_clf:feature_rounded_40', 'signal_clf:feature_written_twice', 'signal_clf:id_added_gaps', 'signal_clf:id_added_int',
    'signal_clf:id_added_text', 'signal_clf:missing_cell_feature', 'signal_clf:missing_cell_target', 'signal_clf:noise_added', 'signal_clf:noise_rounded_1dp', 'signal_clf:noise_score_1_to_40', 'signal_clf:number_sentinel_-999', 'signal_clf:number_sentinel_1e308',
    'signal_clf:number_sentinel_999999', 'signal_clf:number_two_sentinels', 'signal_clf:row_repeated', 'signal_clf:rows_shuffled', 'signal_clf:second_date_added', 'signal_clf:target_first', 'signal_clf:target_renamed', 'signal_clf:target_respelt',
    'signal_clf:target_respelt_flipped', 'signal_clf:two_columns_together', 'signal_clf:white_space_cells', 'signal_reg:base', 'signal_reg:blank_line_added', 'signal_reg:coded_copy_number', 'signal_reg:coded_copy_text', 'signal_reg:constant_added',
    'signal_reg:copy_5_cells_differ', 'signal_reg:copy_5_cells_other_row', 'signal_reg:copy_5_cells_plus_1', 'signal_reg:copy_5_cells_plus_1000', 'signal_reg:copy_5_cells_plus_20', 'signal_reg:copy_5_cells_times_1.5', 'signal_reg:copy_and_target_sentinel', 'signal_reg:copy_distinct',
    'signal_reg:copy_exact', 'signal_reg:copy_negative_scale', 'signal_reg:copy_one_cell_999999', 'signal_reg:feature_renamed', 'signal_reg:feature_renamed_awkward', 'signal_reg:feature_rounded_10', 'signal_reg:feature_rounded_40', 'signal_reg:feature_written_twice',
    'signal_reg:id_added_gaps', 'signal_reg:id_added_int', 'signal_reg:id_added_text', 'signal_reg:missing_cell_feature', 'signal_reg:missing_cell_target', 'signal_reg:noise_added', 'signal_reg:noise_rounded_1dp', 'signal_reg:noise_score_1_to_40',
    'signal_reg:number_sentinel_-999', 'signal_reg:number_sentinel_1e308', 'signal_reg:number_sentinel_999999', 'signal_reg:number_two_sentinels', 'signal_reg:row_repeated', 'signal_reg:rows_shuffled', 'signal_reg:second_date_added', 'signal_reg:target_first',
    'signal_reg:target_renamed', 'signal_reg:target_sentinel_-999', 'signal_reg:target_sentinel_999999', 'signal_reg:target_two_sentinels', 'signal_reg:two_columns_together', 'signal_reg:white_space_cells', 'time_blocks:base', 'time_blocks:blank_line_added',
    'time_blocks:coded_copy_number', 'time_blocks:coded_copy_text', 'time_blocks:constant_added', 'time_blocks:copy_distinct', 'time_blocks:copy_two_valued', 'time_blocks:date_sentinel_1900', 'time_blocks:feature_renamed', 'time_blocks:feature_renamed_awkward',
    'time_blocks:feature_written_twice', 'time_blocks:id_added_gaps', 'time_blocks:id_added_int', 'time_blocks:id_added_text', 'time_blocks:missing_cell_date', 'time_blocks:missing_cell_feature', 'time_blocks:missing_cell_target', 'time_blocks:noise_added',
    'time_blocks:noise_rounded_1dp', 'time_blocks:noise_score_1_to_40', 'time_blocks:number_sentinel_-999', 'time_blocks:number_sentinel_1e308', 'time_blocks:number_sentinel_999999', 'time_blocks:number_two_sentinels', 'time_blocks:role_column_renamed', 'time_blocks:row_repeated',
    'time_blocks:row_repeated_other_target', 'time_blocks:rows_repeated_10pct', 'time_blocks:rows_repeated_1pct', 'time_blocks:rows_shuffled', 'time_blocks:second_date_added', 'time_blocks:stamp_shared', 'time_blocks:target_first', 'time_blocks:target_renamed',
    'time_blocks:target_respelt', 'time_blocks:target_respelt_flipped', 'time_blocks:two_columns_together', 'time_blocks:white_space_cells', 'time_blocks_gap:base', 'time_blocks_gap:blank_line_added', 'time_blocks_gap:coded_copy_number', 'time_blocks_gap:coded_copy_text',
    'time_blocks_gap:constant_added', 'time_blocks_gap:copy_distinct', 'time_blocks_gap:copy_two_valued', 'time_blocks_gap:date_sentinel_1900', 'time_blocks_gap:feature_renamed', 'time_blocks_gap:feature_renamed_awkward', 'time_blocks_gap:feature_written_twice', 'time_blocks_gap:id_added_gaps',
    'time_blocks_gap:id_added_int', 'time_blocks_gap:id_added_text', 'time_blocks_gap:missing_cell_date', 'time_blocks_gap:missing_cell_feature', 'time_blocks_gap:missing_cell_target', 'time_blocks_gap:noise_added', 'time_blocks_gap:noise_rounded_1dp', 'time_blocks_gap:noise_score_1_to_40',
    'time_blocks_gap:number_sentinel_-999', 'time_blocks_gap:number_sentinel_1e308', 'time_blocks_gap:number_sentinel_999999', 'time_blocks_gap:number_two_sentinels', 'time_blocks_gap:role_column_renamed', 'time_blocks_gap:row_repeated', 'time_blocks_gap:row_repeated_other_target', 'time_blocks_gap:rows_repeated_10pct',
    'time_blocks_gap:rows_repeated_1pct', 'time_blocks_gap:rows_shuffled', 'time_blocks_gap:second_date_added', 'time_blocks_gap:stamp_shared', 'time_blocks_gap:target_first', 'time_blocks_gap:target_renamed', 'time_blocks_gap:target_respelt', 'time_blocks_gap:target_respelt_flipped',
    'time_blocks_gap:two_columns_together', 'time_blocks_gap:white_space_cells', 'time_blocks_undeclared:base', 'time_blocks_undeclared:blank_line_added', 'time_blocks_undeclared:coded_copy_number', 'time_blocks_undeclared:coded_copy_text', 'time_blocks_undeclared:constant_added', 'time_blocks_undeclared:copy_distinct',
    'time_blocks_undeclared:copy_two_valued', 'time_blocks_undeclared:date_sentinel_1900', 'time_blocks_undeclared:date_sentinel_9999', 'time_blocks_undeclared:date_sentinel_zero', 'time_blocks_undeclared:duplicates_group', 'time_blocks_undeclared:feature_renamed', 'time_blocks_undeclared:feature_renamed_awkward', 'time_blocks_undeclared:feature_written_twice',
    'time_blocks_undeclared:id_added_gaps', 'time_blocks_undeclared:id_added_int', 'time_blocks_undeclared:id_added_text', 'time_blocks_undeclared:missing_cell_date', 'time_blocks_undeclared:missing_cell_feature', 'time_blocks_undeclared:missing_cell_target', 'time_blocks_undeclared:noise_added', 'time_blocks_undeclared:noise_rounded_1dp',
    'time_blocks_undeclared:noise_score_1_to_40', 'time_blocks_undeclared:number_sentinel_-999', 'time_blocks_undeclared:number_sentinel_1e308', 'time_blocks_undeclared:number_sentinel_999999', 'time_blocks_undeclared:number_two_sentinels', 'time_blocks_undeclared:role_column_renamed', 'time_blocks_undeclared:row_repeated', 'time_blocks_undeclared:rows_shuffled',
    'time_blocks_undeclared:second_date_added', 'time_blocks_undeclared:target_first', 'time_blocks_undeclared:target_renamed', 'time_blocks_undeclared:target_respelt', 'time_blocks_undeclared:target_respelt_flipped', 'time_blocks_undeclared:two_columns_together', 'time_blocks_undeclared:white_space_cells', 'walk_copy:base',
    'walk_copy:blank_line_added', 'walk_copy:coded_copy_number', 'walk_copy:coded_copy_text', 'walk_copy:constant_added', 'walk_copy:copy_2pct_cells_altered', 'walk_copy:copy_5_cells_differ', 'walk_copy:copy_5_cells_other_row', 'walk_copy:copy_5_cells_plus_1',
    'walk_copy:copy_5_cells_plus_1000', 'walk_copy:copy_5_cells_plus_20', 'walk_copy:copy_5_cells_times_1.5', 'walk_copy:copy_and_target_sentinel', 'walk_copy:copy_curve', 'walk_copy:copy_distinct', 'walk_copy:copy_exact', 'walk_copy:copy_negative_scale',
    'walk_copy:copy_noise_1pct', 'walk_copy:copy_one_cell_999999', 'walk_copy:date_sentinel_1900', 'walk_copy:feature_999999_early_row', 'walk_copy:feature_999999_latest_row', 'walk_copy:feature_renamed', 'walk_copy:feature_renamed_awkward', 'walk_copy:feature_rounded_10',
    'walk_copy:feature_rounded_40', 'walk_copy:feature_written_twice', 'walk_copy:id_added_gaps', 'walk_copy:id_added_int', 'walk_copy:id_added_text', 'walk_copy:missing_cell_date', 'walk_copy:missing_cell_feature', 'walk_copy:missing_cell_target',
    'walk_copy:noise_added', 'walk_copy:noise_rounded_1dp', 'walk_copy:noise_score_1_to_40', 'walk_copy:number_sentinel_-999', 'walk_copy:number_sentinel_1e308', 'walk_copy:number_sentinel_999999', 'walk_copy:number_two_sentinels', 'walk_copy:role_column_renamed',
    'walk_copy:row_repeated', 'walk_copy:row_repeated_other_target', 'walk_copy:rows_repeated_10pct', 'walk_copy:rows_repeated_1pct', 'walk_copy:rows_shuffled', 'walk_copy:second_date_added', 'walk_copy:stamp_shared', 'walk_copy:target_-999_early_row',
    'walk_copy:target_-999_latest_row', 'walk_copy:target_first', 'walk_copy:target_renamed', 'walk_copy:target_sentinel_-999', 'walk_copy:target_sentinel_999999', 'walk_copy:target_two_sentinels', 'walk_copy:two_columns_together', 'walk_copy_cell:base',
    'walk_copy_cell:blank_line_added', 'walk_copy_cell:coded_copy_number', 'walk_copy_cell:coded_copy_text', 'walk_copy_cell:constant_added', 'walk_copy_cell:copy_2pct_cells_altered', 'walk_copy_cell:copy_5_cells_differ', 'walk_copy_cell:copy_5_cells_other_row', 'walk_copy_cell:copy_5_cells_plus_1',
    'walk_copy_cell:copy_5_cells_plus_1000', 'walk_copy_cell:copy_5_cells_plus_20', 'walk_copy_cell:copy_5_cells_times_1.5', 'walk_copy_cell:copy_and_target_sentinel', 'walk_copy_cell:copy_curve', 'walk_copy_cell:copy_distinct', 'walk_copy_cell:copy_exact', 'walk_copy_cell:copy_negative_scale',
    'walk_copy_cell:copy_one_cell_999999', 'walk_copy_cell:date_sentinel_1900', 'walk_copy_cell:feature_999999_early_row', 'walk_copy_cell:feature_999999_latest_row', 'walk_copy_cell:feature_renamed', 'walk_copy_cell:feature_renamed_awkward', 'walk_copy_cell:feature_rounded_10', 'walk_copy_cell:feature_rounded_40',
    'walk_copy_cell:feature_written_twice', 'walk_copy_cell:id_added_gaps', 'walk_copy_cell:id_added_int', 'walk_copy_cell:id_added_text', 'walk_copy_cell:missing_cell_date', 'walk_copy_cell:missing_cell_feature', 'walk_copy_cell:missing_cell_target', 'walk_copy_cell:noise_added',
    'walk_copy_cell:noise_rounded_1dp', 'walk_copy_cell:noise_score_1_to_40', 'walk_copy_cell:number_sentinel_-999', 'walk_copy_cell:number_sentinel_1e308', 'walk_copy_cell:number_sentinel_999999', 'walk_copy_cell:number_two_sentinels', 'walk_copy_cell:role_column_renamed', 'walk_copy_cell:row_repeated',
    'walk_copy_cell:row_repeated_other_target', 'walk_copy_cell:rows_repeated_10pct', 'walk_copy_cell:rows_repeated_1pct', 'walk_copy_cell:rows_shuffled', 'walk_copy_cell:second_date_added', 'walk_copy_cell:stamp_shared', 'walk_copy_cell:target_-999_early_row', 'walk_copy_cell:target_-999_latest_row',
    'walk_copy_cell:target_first', 'walk_copy_cell:target_renamed', 'walk_copy_cell:target_sentinel_-999', 'walk_copy_cell:target_sentinel_999999', 'walk_copy_cell:target_two_sentinels', 'walk_copy_cell:two_columns_together', 'walk_late_cell:base', 'walk_late_cell:blank_line_added',
    'walk_late_cell:coded_copy_number', 'walk_late_cell:coded_copy_text', 'walk_late_cell:constant_added', 'walk_late_cell:copy_2pct_cells_altered', 'walk_late_cell:copy_5_cells_differ', 'walk_late_cell:copy_5_cells_other_row', 'walk_late_cell:copy_5_cells_plus_1', 'walk_late_cell:copy_5_cells_plus_1000',
    'walk_late_cell:copy_5_cells_plus_20', 'walk_late_cell:copy_5_cells_times_1.5', 'walk_late_cell:copy_and_target_sentinel', 'walk_late_cell:copy_curve', 'walk_late_cell:copy_distinct', 'walk_late_cell:copy_exact', 'walk_late_cell:copy_negative_scale', 'walk_late_cell:copy_noise_1pct',
    'walk_late_cell:copy_one_cell_999999', 'walk_late_cell:date_sentinel_1900', 'walk_late_cell:feature_999999_early_row', 'walk_late_cell:feature_999999_latest_row', 'walk_late_cell:feature_renamed', 'walk_late_cell:feature_renamed_awkward', 'walk_late_cell:feature_written_twice', 'walk_late_cell:id_added_gaps',
    'walk_late_cell:id_added_int', 'walk_late_cell:id_added_text', 'walk_late_cell:missing_cell_date', 'walk_late_cell:missing_cell_feature', 'walk_late_cell:missing_cell_target', 'walk_late_cell:noise_added', 'walk_late_cell:noise_rounded_1dp', 'walk_late_cell:noise_score_1_to_40',
    'walk_late_cell:number_sentinel_-999', 'walk_late_cell:number_sentinel_1e308', 'walk_late_cell:number_sentinel_999999', 'walk_late_cell:number_two_sentinels', 'walk_late_cell:role_column_renamed', 'walk_late_cell:row_repeated', 'walk_late_cell:row_repeated_other_target', 'walk_late_cell:rows_repeated_10pct',
    'walk_late_cell:rows_repeated_1pct', 'walk_late_cell:rows_shuffled', 'walk_late_cell:second_date_added', 'walk_late_cell:stamp_shared', 'walk_late_cell:target_-999_early_row', 'walk_late_cell:target_-999_latest_row', 'walk_late_cell:target_first', 'walk_late_cell:target_renamed',
    'walk_late_cell:target_sentinel_-999', 'walk_late_cell:target_sentinel_999999', 'walk_late_cell:target_two_sentinels', 'walk_late_cell:two_columns_together', 'walk_repeated:base', 'walk_repeated:blank_line_added', 'walk_repeated:coded_copy_number', 'walk_repeated:coded_copy_text',
    'walk_repeated:constant_added', 'walk_repeated:copy_2pct_cells_altered', 'walk_repeated:copy_5_cells_differ', 'walk_repeated:copy_5_cells_other_row', 'walk_repeated:copy_5_cells_plus_1', 'walk_repeated:copy_5_cells_plus_1000', 'walk_repeated:copy_5_cells_plus_20', 'walk_repeated:copy_5_cells_times_1.5',
    'walk_repeated:copy_and_target_sentinel', 'walk_repeated:copy_curve', 'walk_repeated:copy_distinct', 'walk_repeated:copy_exact', 'walk_repeated:copy_negative_scale', 'walk_repeated:copy_noise_1pct', 'walk_repeated:copy_one_cell_999999', 'walk_repeated:date_sentinel_1900',
    'walk_repeated:feature_999999_early_row', 'walk_repeated:feature_999999_latest_row', 'walk_repeated:feature_renamed', 'walk_repeated:feature_renamed_awkward', 'walk_repeated:feature_rounded_10', 'walk_repeated:feature_rounded_40', 'walk_repeated:feature_written_twice', 'walk_repeated:id_added_gaps',
    'walk_repeated:id_added_int', 'walk_repeated:id_added_text', 'walk_repeated:missing_cell_date', 'walk_repeated:missing_cell_feature', 'walk_repeated:missing_cell_target', 'walk_repeated:noise_added', 'walk_repeated:noise_rounded_1dp', 'walk_repeated:noise_score_1_to_40',
    'walk_repeated:number_sentinel_-999', 'walk_repeated:number_sentinel_1e308', 'walk_repeated:number_sentinel_999999', 'walk_repeated:number_two_sentinels', 'walk_repeated:role_column_renamed', 'walk_repeated:row_repeated', 'walk_repeated:row_repeated_other_target', 'walk_repeated:rows_repeated_10pct',
    'walk_repeated:rows_repeated_1pct', 'walk_repeated:rows_shuffled', 'walk_repeated:second_date_added', 'walk_repeated:stamp_shared', 'walk_repeated:target_-999_early_row', 'walk_repeated:target_-999_latest_row', 'walk_repeated:target_first', 'walk_repeated:target_renamed',
    'walk_repeated:target_sentinel_-999', 'walk_repeated:target_sentinel_999999', 'walk_repeated:target_two_sentinels', 'walk_repeated:two_columns_together',
})

TABLES_OWNED = frozenset({
    0, 1, 2, 5, 6, 8, 11, 13,
    14, 18, 21, 25, 29, 31, 32, 33,
    34, 36, 38, 39, 41, 42, 47, 49,
    50, 51, 52, 53, 54, 55, 56, 58,
    63, 64, 66, 67, 69, 70, 72, 74,
    75, 76, 77, 78, 79, 80, 81, 83,
    84, 87, 91, 93, 95, 96, 97, 100,
    101, 102, 103, 105, 106, 108, 110, 111,
    112, 113, 114, 116, 119, 120, 122, 128,
    130, 132, 134, 135, 136, 138, 141, 143,
    151, 153, 155, 156, 159, 164, 165, 166,
    167, 168, 169, 170, 171, 172, 173, 174,
    175, 176, 179, 180, 182, 183, 185, 186,
    187, 189, 190, 191, 196, 197, 198, 199,
    201, 206, 210, 211, 212, 213, 214, 216,
    217, 220, 221, 223, 226, 227, 228, 231,
    232, 234, 235, 238, 239, 240, 241, 242,
    247, 249, 251, 252, 253, 254, 255, 256,
    259, 261, 262, 269, 271, 272, 276, 277,
    278, 279, 281, 282, 284, 285, 286, 288,
    289, 291, 294, 296, 298, 300, 301, 303,
    304, 306, 310, 311, 312, 313, 314, 315,
    316, 319, 321, 322, 323, 324, 326, 327,
    328, 329, 330, 332, 335, 337, 339, 340,
    341, 342, 345, 346, 347, 349, 350, 351,
    353, 354, 356, 360, 361, 362, 365, 366,
    368, 372, 375, 376, 377, 378, 379, 381,
    382, 383, 385, 388, 393, 394, 396, 397,
    398, 402, 403, 405, 406, 407, 409, 410,
    411, 412, 413, 416, 417, 418, 419, 421,
    422, 423, 424, 427, 429, 430, 431, 432,
    433, 435, 437, 438, 439, 441, 442, 443,
    444, 445, 447, 450, 451, 452, 453, 454,
    457, 458, 459, 461, 462, 463, 465, 466,
    467, 469, 471, 472, 478, 480, 481, 482,
    484, 485, 489, 490, 491, 492, 494, 498,
    500, 504, 508, 510, 511, 515, 517, 518,
    519, 520, 523, 524, 525, 526, 527, 528,
    529, 533, 534, 536, 538, 540, 542, 546,
    547, 548, 549, 550, 551, 553, 555, 556,
    557, 564, 567, 570, 571, 572, 573, 574,
    575, 576, 577, 578, 579, 580, 581, 585,
    586, 587, 590, 591, 597, 600, 601, 605,
    606, 607, 608, 609, 610, 611, 612, 615,
    616, 619, 624, 626, 628, 630, 632, 633,
    634, 635, 636, 639, 642, 644, 645, 647,
    648, 649, 650, 651, 653, 654, 655, 656,
    658, 659, 661, 662, 663, 666, 667, 669,
    670, 674, 675, 676, 677, 678, 679, 680,
    682, 684, 686, 689, 690, 692, 693, 694,
    695, 696, 697, 699, 700, 701, 702, 704,
    706, 708, 709, 710, 712, 713, 716, 717,
    718, 719, 720, 721, 725, 726, 727, 729,
    730, 731, 732, 734, 736, 737, 738, 739,
    742, 745, 747, 748, 750, 751, 752, 753,
    754, 755, 758, 760, 762, 764, 765, 768,
    769, 772, 774, 775, 776, 778, 779, 781,
    783, 785, 786, 788, 789, 791, 794, 795,
    796, 798, 800, 802, 804, 805, 806, 808,
    809, 810, 812, 815, 816, 819, 820, 821,
    822, 824, 825, 828, 829, 830, 832, 834,
    835, 836, 837, 839, 840, 841, 846, 847,
    853, 854, 856, 857, 859, 861, 863, 865,
    866, 867, 870, 871, 873, 874, 876, 878,
    880, 881, 882, 883, 884, 888, 890, 893,
    895, 896, 899, 900, 901, 904, 907, 909,
    911, 912, 914, 915, 916, 917, 918, 923,
    924, 925, 927, 929, 930, 931, 932, 934,
    935, 936, 937, 938, 939, 941, 942, 946,
    948, 949, 950, 953, 955, 959, 961, 962,
    963, 964, 965, 966, 967, 968, 969, 971,
    972, 973, 974, 975, 978, 979, 980, 982,
    984, 985, 987, 988, 992, 996, 997, 998,
    1000001, 1000004, 1000005, 1000006, 1000007, 1000008, 1000009, 1000010,
    1000011, 1000012, 1000015, 1000016, 1000017, 1000018, 1000020, 1000025,
    1000028, 1000029, 1000032, 1000033, 1000034, 1000035, 1000037, 1000038,
    1000040, 1000041, 1000042, 1000043, 1000044, 1000045, 1000046, 1000049,
    1000050, 1000051, 1000053, 1000054, 1000056, 1000057, 1000061, 1000062,
    1000064, 1000065, 1000066, 1000069, 1000070, 1000071, 1000073, 1000074,
    1000078, 1000079, 1000083, 1000085, 1000087, 1000088, 1000089, 1000094,
    1000095, 1000097, 1000099, 1000100, 1000101, 1000102, 1000103, 1000104,
    1000105, 1000108, 1000109, 1000111, 1000112, 1000114, 1000117, 1000118,
    1000119, 1000120, 1000121, 1000122, 1000126, 1000127, 1000128, 1000131,
    1000132, 1000133, 1000134, 1000135, 1000136, 1000137, 1000140, 1000144,
    1000145, 1000146, 1000150, 1000152, 1000154, 1000156, 1000157, 1000158,
    1000159, 1000161, 1000162, 1000164, 1000166, 1000168, 1000170, 1000176,
    1000177, 1000180, 1000182, 1000183, 1000184, 1000185, 1000186, 1000187,
    1000188, 1000189, 1000190, 1000191, 1000192, 1000194, 1000198, 1000199,
})
