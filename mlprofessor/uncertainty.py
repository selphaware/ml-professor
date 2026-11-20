"""Interval and p-value of the headline from the holdout predictions alone (spec 3, step 12).

Nothing is refitted. Every resample is a vector of row counts over the holdout
rows, so the score of a resample is a weighted score of the fixed predictions:
a row drawn twice counts twice. The row bootstrap draws the rows; the
moving-block bootstrap draws blocks of consecutive distinct timestamps and
takes every row of each drawn timestamp.
"""
import math

import numpy as np

from mlprofessor import metrics
from mlprofessor.errors import NonFiniteScoreError

INTERVAL_LEVEL = 0.95
MIN_BLOCK_TIMESTAMPS = 30
RESAMPLES = {"full": 1000, "fast": 200}

# Generator streams drawn from the one seed, apart from every other stream.
_BOOTSTRAP_STREAM = 2
_PERMUTATION_STREAM = 3

P_VALUE_REASON_TIME = (
    "No p-value under time order: shuffling the holdout targets assumes the rows are "
    "independent, and rows in time order are not.")
P_VALUE_REASON_NO_SKILL = (
    "No p-value: the baseline predicts every holdout row exactly, so skill has no value "
    "and there is nothing to compare the shuffled targets with.")


def _stream(seed, purpose):
    return np.random.default_rng(np.random.SeedSequence([int(seed), purpose]))


def block_length(n_timestamps):
    """The smallest whole number L with L**3 >= n_timestamps."""
    length = max(1, round(n_timestamps ** (1.0 / 3.0)))
    while length ** 3 < n_timestamps:
        length += 1
    while length > 1 and (length - 1) ** 3 >= n_timestamps:
        length -= 1
    return length


class _HoldoutScorer:
    """Score and skill of the fixed holdout predictions under row weights."""

    def __init__(self, metric, y, pred, baseline_pred):
        self.metric = metric
        self.pred = np.asarray(pred, dtype=float)
        if metric == "roc_auc":
            # Groups of tied predictions, numbered in increasing order of value.
            _, self.group = np.unique(self.pred, return_inverse=True)
            self.n_groups = int(self.group.max()) + 1 if len(self.pred) else 0
        else:
            self.baseline_pred = np.asarray(baseline_pred, dtype=float)
        self.set_targets(y)

    def set_targets(self, y):
        """Score these holdout targets against the fixed predictions."""
        y = np.asarray(y)
        if self.metric == "roc_auc":
            self.positive = (y == 1).astype(float)
            self.negative = 1.0 - self.positive
        else:
            y = y.astype(float)
            with np.errstate(over="ignore", invalid="ignore"):
                self.model_error = (y - self.pred) ** 2
                self.base_error = (y - self.baseline_pred) ** 2

    def score_and_skill(self, weights):
        """(score, skill) under `weights`, or None when the score cannot be computed.

        Skill is None when only it cannot be computed: the baseline is exact on
        the rows drawn.
        """
        if self.metric == "roc_auc":
            pos = np.bincount(self.group, weights=weights * self.positive,
                              minlength=self.n_groups)
            neg = np.bincount(self.group, weights=weights * self.negative,
                              minlength=self.n_groups)
            n_pos, n_neg = pos.sum(), neg.sum()
            if n_pos == 0 or n_neg == 0:
                return None
            below = np.cumsum(neg) - neg
            score = float((pos * (below + 0.5 * neg)).sum() / (n_pos * n_neg))
            return score, metrics.skill("roc_auc", score, 0.5)
        total = weights.sum()
        if total == 0:
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            score = float((weights * self.model_error).sum() / total)
            base = float((weights * self.base_error).sum() / total)
        if not (np.isfinite(score) and np.isfinite(base)):
            raise NonFiniteScoreError(
                "headline.interval_low, headline.interval_high and headline.p_value: the "
                "squared error of a resample or a shuffle of the holdout is not finite "
                f"(model {score!r}, baseline {base!r}): the target values are likely too "
                "large for squared error.")
        skill = metrics.skill("mse", score, base)
        if skill is not None and not np.isfinite(skill):
            raise NonFiniteScoreError(
                "headline.skill_low, headline.skill_high and headline.p_value: the skill of "
                f"a resample or a shuffle of the holdout is not finite ({skill!r}): the "
                "baseline's error on the rows drawn is too small beside the model's.")
        return score, skill


def _percentiles(values):
    tail = 100.0 * (1.0 - INTERVAL_LEVEL) / 2.0
    low, high = np.percentile(np.asarray(values, dtype=float), [tail, 100.0 - tail])
    return float(low), float(high)


def _row_weights(n_rows, rng):
    return np.bincount(rng.integers(0, n_rows, size=n_rows), minlength=n_rows).astype(float)


def _block_weights(row_timestamp, n_timestamps, length, rng):
    n_blocks = math.ceil(n_timestamps / length)
    starts = rng.integers(0, n_timestamps - length + 1, size=n_blocks)
    drawn = (starts[:, None] + np.arange(length)[None, :]).ravel()[:n_timestamps]
    per_timestamp = np.bincount(drawn, minlength=n_timestamps).astype(float)
    return per_timestamp[row_timestamp]


def _p_value(metric, y, pred, baseline_pred, observed_skill, n_shuffles, rng):
    """(1 + shuffles whose skill is at least the observed skill) / (1 + shuffles)."""
    y = np.asarray(y)
    scorer = _HoldoutScorer(metric, y, pred, baseline_pred)
    ones = np.ones(len(y))
    tolerance = 1e-12 * max(1.0, abs(observed_skill))
    at_least = 0
    for _ in range(n_shuffles):
        scorer.set_targets(y[rng.permutation(len(y))])
        result = scorer.score_and_skill(ones)
        if result is not None and result[1] is not None                 and result[1] >= observed_skill - tolerance:
            at_least += 1
    return (1 + at_least) / (1 + n_shuffles)


def headline_uncertainty(metric, y, pred, baseline_pred, skill, profile, seed,
                         holdout_keys=None):
    """The interval and p-value fields of `headline` (spec 5.2), and warning codes.

    `y`, `pred` and `baseline_pred` are the holdout targets, the chosen model's
    holdout predictions and the baseline's (for ROC AUC the baseline is chance
    and `baseline_pred` is not used). `holdout_keys` are the holdout rows'
    timestamps under time order, `None` under the random scheme. Returns
    `(fields, warnings)`, `warnings` a list of `(code, message)`.
    """
    y = np.asarray(y)
    n_resamples = RESAMPLES[profile]
    scorer = _HoldoutScorer(metric, y, pred, baseline_pred)
    fields = {
        "interval_method": None, "interval_level": INTERVAL_LEVEL,
        "interval_low": None, "interval_high": None, "skill_low": None, "skill_high": None,
        "n_resamples": n_resamples, "n_resamples_used": 0, "n_resamples_skill_used": 0,
        "block_length": None,
        "interval_reason": None, "p_value": None, "n_permutations": None,
        "p_value_reason": None,
    }
    warnings = []
    rng = _stream(seed, _BOOTSTRAP_STREAM)

    if holdout_keys is None:
        method = "bootstrap_rows"
        draw = lambda: _row_weights(len(y), rng)  # noqa: E731
    else:
        stamps, row_timestamp = np.unique(np.asarray(holdout_keys), return_inverse=True)
        n_timestamps = len(stamps)
        method = None
        if n_timestamps < MIN_BLOCK_TIMESTAMPS:
            fields["interval_reason"] = (
                f"The holdout holds {n_timestamps} distinct timestamps; the block bootstrap "
                f"needs at least {MIN_BLOCK_TIMESTAMPS}.")
        else:
            method = "bootstrap_blocks"
            length = block_length(n_timestamps)
            fields["block_length"] = length
            draw = lambda: _block_weights(row_timestamp, n_timestamps, length, rng)  # noqa: E731
        fields["p_value_reason"] = P_VALUE_REASON_TIME

    if method is not None:
        scores, skills = [], []
        for _ in range(n_resamples):
            result = scorer.score_and_skill(draw())
            if result is not None:
                scores.append(result[0])
                if result[1] is not None:
                    skills.append(result[1])
        fields["n_resamples_used"] = len(scores)
        fields["n_resamples_skill_used"] = len(skills)
        if 2 * len(scores) < n_resamples:
            fields["block_length"] = None
            fields["interval_reason"] = (
                f"The score could be computed in {len(scores)} of {n_resamples} resamples "
                "(the others held one class); at least half are needed.")
        else:
            fields["interval_method"] = method
            fields["interval_low"], fields["interval_high"] = _percentiles(scores)
            if 2 * len(skills) >= n_resamples:
                fields["skill_low"], fields["skill_high"] = _percentiles(skills)
            elif skill is not None:
                # A null headline skill is covered by its own warning (step 11).
                warnings.append((
                    "no_skill_interval",
                    f"No interval on skill: it could be computed in {len(skills)} of "
                    f"{n_resamples} resamples (in the others the baseline is exact on every "
                    "row drawn); at least half are needed. The interval on the score stands."))

    if fields["interval_method"] is None:
        warnings.append(("no_interval", "No interval: " + fields["interval_reason"]))
    elif fields["interval_method"] == "bootstrap_blocks":
        warnings.append((
            "interval_approximate",
            "The interval comes from a moving-block bootstrap over the holdout timestamps; "
            "it tends to be too narrow, more so on a short holdout or when errors are "
            "serially correlated."))

    if skill is None:
        if holdout_keys is None:
            fields["p_value_reason"] = P_VALUE_REASON_NO_SKILL
    elif holdout_keys is None:
        fields["p_value"] = _p_value(metric, y, pred, baseline_pred, skill, n_resamples,
                                     _stream(seed, _PERMUTATION_STREAM))
        fields["n_permutations"] = n_resamples
    return fields, warnings
