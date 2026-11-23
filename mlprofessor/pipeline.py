"""The run of `profile()` (spec 3): load, count, decide, split, select, score once.

Every fitted step sees training rows only. Selection uses cross-validation
scores; the headline uses the one holdout score of the chosen configuration,
refitted on the training rows alone.
"""
import os
import platform
import time
from numbers import Integral

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from mlprofessor import errors, extensions, leakage, metrics, models, uncertainty
from mlprofessor.columns import (feature_numbers, numeric_values, parse_numbers,
                                 read_entities, roles_of_read_columns, stripped_text,
                                 type_family)
from mlprofessor.load import load_table, read_number_columns, usable_rows
from mlprofessor.preprocess import MISSING_LEVEL, Preprocessor, check_feature_names
from mlprofessor.result import (GROUPED_CAVEAT, LIMITATIONS, RANDOM_CAVEAT, ROWS_ITEMS,
                                ProfileResult)
from mlprofessor.split import (MAX_SEED, N_FOLDS, large_units, read_time, split_random,
                               split_time, units_of)
from mlprofessor.task import TASKS, class_value, decide_task, read_target_values

PROFILES = ("full", "fast")
DUPLICATES = ("error", "keep", "group")
COPIES_MIN_ROWS = 10
COPIES_MAX_SHARE = 0.01
MIN_USABLE_ROWS = 100
MIN_CLASS_ROWS = 10
ROW_CAP = 2000
SMALL_HOLDOUT = 50
PANEL_AUTOCORRELATION = 0.5

# Independent generator streams drawn from the one seed. The split itself is
# `split_random(n, classes, seed)`, so that `stages` and `profile()` agree.
_ROW_CAP_STREAM = 1


def _stream(seed, purpose):
    return np.random.default_rng(np.random.SeedSequence([int(seed), purpose]))


# ------------------------------------------------------------ arguments

def _is_whole(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def check_arguments(data, target, task, time_col, gap, profile, seed, out_dir, n_jobs,
                    duplicates="error", group_col=None):
    """Raise `InputError` for any argument value outside spec 2.1, before loading."""
    if not isinstance(data, (pd.DataFrame, str, os.PathLike)):
        raise errors.InputError(
            f"data must be a pandas DataFrame or the path of a CSV file; "
            f"got {type(data).__name__}.")
    if not isinstance(target, str):
        raise errors.InputError(f"target must be a column name (text); got {target!r}.")
    if task not in TASKS:
        raise errors.InputError(f"task must be one of {', '.join(TASKS)}; got {task!r}.")
    if time_col is not None and time_col is not False and not isinstance(time_col, str):
        raise errors.InputError(
            f"time_col must be a column name, None or False; got {time_col!r}.")
    if not _is_whole(gap) or gap < 0:
        raise errors.InputError(f"gap must be a whole number, at least 0; got {gap!r}.")
    if gap > 0 and not isinstance(time_col, str):
        raise errors.InputError(
            f"gap={gap} needs a time column; pass time_col, or leave gap at 0.")
    if isinstance(time_col, str) and time_col == target:
        raise errors.InputError("time_col and target name the same column; name two columns.")
    if profile not in PROFILES:
        raise errors.InputError(f"profile must be 'full' or 'fast'; got {profile!r}.")
    if not _is_whole(seed) or not 0 <= seed <= MAX_SEED:
        raise errors.InputError(
            f"seed must be a whole number from 0 to {MAX_SEED}; got {seed!r}.")
    if out_dir is not None and not isinstance(out_dir, (str, os.PathLike)):
        raise errors.InputError(f"out_dir must be a folder path or None; got {out_dir!r}.")
    if not _is_whole(n_jobs) or not (n_jobs == -1 or n_jobs >= 1):
        raise errors.InputError(
            f"n_jobs must be -1 (every core) or a positive whole number; got {n_jobs!r}.")
    if not isinstance(duplicates, str) or duplicates not in DUPLICATES:
        raise errors.InputError(
            f"duplicates must be 'error', 'keep' or 'group'; got {duplicates!r}.")
    if group_col is not None and group_col is not False and not isinstance(group_col, str):
        raise errors.InputError(
            f"group_col must be a column name, None or False; got {group_col!r}.")
    if isinstance(group_col, str) and group_col == target:
        raise errors.InputError("group_col and target name the same column; name two columns.")
    if isinstance(group_col, str) and isinstance(time_col, str) and group_col == time_col:
        raise errors.InputError(
            "group_col and time_col name the same column; name two columns.")
    if out_dir is not None:
        _check_out_dir(os.fspath(out_dir))


def _check_out_dir(path):
    """The path, or the nearest folder above it that exists, must be a folder."""
    probe = os.path.abspath(path)
    while not os.path.lexists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return
        probe = parent
    if not os.path.isdir(probe):
        raise errors.InputError(
            f"out_dir {path!r} cannot be used: {probe!r} exists and is not a folder.")


def _date_columns(roles):
    """The columns excluded as dates, in file order."""
    return [r["name"] for r in roles.to_dict(orient="records") if r["reason"] == "datetime"]


ENTITIES_WAYS_ON = (
    "Pass the column that names the entity each row belongs to (a customer, a patient, "
    "a device, one series of a panel) as group_col (--group-col COLUMN), so that the "
    "rows of one entity are kept together; or pass group_col=False (--no-entities) to "
    "state that no rows belong together.")


def _time_order_message(dates, entities_unsaid):
    """Step 5: what the caller must still say, when time order was not said."""
    message = (
        "Nothing was said about time order. Pass the time column as time_col "
        "(--time-col COLUMN) for a time-ordered split, or pass time_col=False "
        "(--no-time-order) to state that the order of the rows in time does not matter.")
    if dates:
        message += (
            " The column(s) " + ", ".join(repr(name) for name in dates) + " hold dates "
            "and would be excluded; a table that holds dates is often ordered in time, "
            "and a random split of it can report as predictable what is not.")
    if entities_unsaid:
        message += " Nothing was said about entities either. " + ENTITIES_WAYS_ON
    return message


def _entities_message(time_col, usable_keys):
    """Step 5: the message of `entities_undeclared`, with time order stated."""
    message = "Nothing was said about entities. " + ENTITIES_WAYS_ON
    if usable_keys is not None:
        n_stamps = len(np.unique(usable_keys))
        if n_stamps < len(usable_keys):
            message += (
                f" The timestamps of {time_col!r} repeat: the {len(usable_keys)} usable "
                f"rows hold {n_stamps} distinct timestamps, as several series in one "
                "table do.")
    return message


def _excluded_text(roles):
    """Each excluded column with its reason, as `columns[].reason` spells it."""
    return "; ".join(f"{r['name']} ({r['reason']})"
                     for r in roles.to_dict(orient="records") if r["role"] == "excluded")


def resolve_n_jobs(n_jobs):
    return int(os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)


# ------------------------------------------------------------ helpers

def _warning(code, column, message):
    return {"code": code, "column": column, "message": message}


def _large_unit_warnings(units, n_rows, classes, holdout):
    """Step 7 (D66): one warning for each unit of a quarter or more of the rows it is
    drawn among, saying whether it is kept in training or is the holdout."""
    found = []
    for record in large_units(units, n_rows, classes, holdout):
        share = 100.0 * record["rows"] / record["among"]
        stated = (f"One unit holds {record['rows']} of the {record['among']} rows it is drawn "
                  f"among ({share:.1f}%), a quarter or more of them.")
        if record["held"]:
            found.append(_warning(
                "large_group_is_holdout", None,
                stated + " The units smaller than a quarter do not together hold a quarter, "
                "so this one unit is the holdout: the headline rests on it alone."))
        else:
            found.append(_warning(
                "large_group_kept_in_training", None,
                stated + " It is kept among the training rows and the holdout is made of "
                "smaller units: the headline does not test it."))
    return found


def _column_warnings(roles, time_col):
    """The warnings of step 5, one per column, in column order."""
    found = []
    for record in roles.to_dict(orient="records"):
        name = record["name"]
        if record["n_coerced"] > 0:
            found.append(_warning(
                "values_coerced", name,
                f"{record['n_coerced']} text cell(s) of {name!r} did not parse as numbers "
                "and were read as missing."))
        if record["reason"] == "numeric_with_separators":
            found.append(_warning(
                "numeric_with_separators", name,
                f"{name!r} holds numbers written with thousands separators, which are "
                "never parsed; the column is excluded. Parse it and pass a DataFrame "
                "to use it."))
        if record["reason"] == "datetime" and time_col is None:
            found.append(_warning(
                "date_column_undeclared", name,
                f"{name!r} looks like a date and is excluded; the split is random. "
                "Pass it as time_col for a time-ordered split."))
        if record["reason"] == "unsupported_type":
            found.append(_warning(
                "unsupported_type", name,
                f"{name!r} is of a type the tool does not read as numbers or categories "
                f"(dtype {record['dtype']}); the column is excluded."))
        if record["role"] == "feature" and record["kind"] == "numeric" and record["n_infinite"] > 0:
            found.append(_warning(
                "infinite_values", name,
                f"{record['n_infinite']} value(s) of {name!r} are infinite or too large for the "
                "tree models (magnitude above 3.4028234663852886e+38) and were read as "
                "missing."))
    return found


def _count_duplicate_rows(frame):
    try:
        return int(frame.duplicated().sum())
    except TypeError:
        as_text = frame.apply(lambda col: col.map(repr))
        return int(as_text.duplicated().sum())


def _feature_values(series, kind):
    """A feature as loaded and typed: floats with NaN, or text with `(missing)`."""
    if kind == "numeric":
        if type_family(series) == "numeric":
            values = numeric_values(series)
        else:
            values = parse_numbers(stripped_text(series))
        return pd.Series(feature_numbers(values), index=series.index, dtype=float)
    text = stripped_text(series)
    return text.where(text.notna(), MISSING_LEVEL).astype(object)


def _comparable_codes(values):
    """Integer codes of one column such that equal codes mean equal values (step 8a).

    Numbers are compared as floats with `-0.0` read as `0.0` and every missing
    cell equal to every other; levels are compared as text.
    """
    values = np.asarray(values)
    if values.dtype.kind == "f":
        values = values + 0.0
    codes, _ = pd.factorize(values, use_na_sentinel=False)
    return np.asarray(codes, dtype=np.int64)


def count_holdout_copies(features, y, training_rows, holdout_rows):
    """Step 8a: the holdout rows with a training row equal in every feature and, unless
    `y` is None, the target."""
    rows = np.concatenate([training_rows, holdout_rows])
    columns = [features[name].to_numpy()[rows] for name in features.columns]
    if y is not None:
        columns.append(np.asarray(y)[rows])
    codes = np.column_stack([_comparable_codes(values) for values in columns])
    _, row_key = np.unique(codes, axis=0, return_inverse=True)
    row_key = row_key.reshape(-1)
    training_keys = row_key[:len(training_rows)]
    holdout_keys = row_key[len(training_rows):]
    return int(np.isin(holdout_keys, training_keys).sum())


def same_feature_sets(features, rows):
    """Step 7 under `duplicates="group"`: `rows` numbered 0.. by the set of rows that
    agree with them in every feature column (step 8a's comparison; the target plays no
    part); and the number of those rows that agree with another of them."""
    columns = [features[name].to_numpy()[rows] for name in features.columns]
    codes = np.column_stack([_comparable_codes(values) for values in columns])
    _, key = np.unique(codes, axis=0, return_inverse=True)
    key = key.reshape(-1)
    sizes = np.bincount(key)
    return key, int((sizes[key] > 1).sum())


def joined_units(first, second):
    """Two rows in one unit when they share a label of `first` or of `second`, directly
    or through other rows; units numbered 0.. by their first row."""
    parent = list(range(len(first)))

    def root(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    for labels in (first, second):
        leader = {}
        for r, label in enumerate(np.asarray(labels).tolist()):
            if label in leader:
                a, b = root(leader[label]), root(r)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                leader[label] = r
    roots = np.array([root(r) for r in range(len(first))], dtype=np.int64)
    _, units = np.unique(roots, return_inverse=True)
    return units.reshape(-1).astype(np.int64)


COPIES_ADVICE = (
    "Pass duplicates='keep' (--duplicates keep) if the repeated rows are different cases "
    "that look alike, as in a table of few-valued columns. Pass duplicates='group' "
    "(--duplicates group) if they are one case recorded more than once: no combination of "
    "feature values is then on both sides of the split. Or remove the repeated rows.")


def _dates_text(roles):
    dates = _date_columns(roles)
    if not dates:
        return ""
    return (
        " The column(s) " + ", ".join(repr(name) for name in dates) + " were excluded "
        "as dates because time_col=False was passed, so rows that differ only in "
        "their date count as copies; if the rows are ordered in time, pass the column "
        "as time_col for a time-ordered split, under which this rule does not apply.")


def _copies_message(n_copies, n_holdout, roles):
    share = 100.0 * n_copies / n_holdout
    message = (
        f"{n_copies} of the {n_holdout} holdout rows ({share:.1f}%) have a copy among the "
        "training rows: the same value in every feature column and the same target. A model "
        "that recalls a row scores on them, so the headline would be inflated.")
    return message + " " + COPIES_ADVICE + _dates_text(roles)


def _alike_after_leak_message(n_alike, n_holdout, roles):
    """Step 8a under `duplicates="group"` once the leak screen has excluded a column."""
    share = 100.0 * n_alike / n_holdout
    leaks = roles.loc[roles["reason"] == "suspected_leak", "name"].tolist()
    names = ", ".join(repr(str(name)) for name in leaks)
    message = (
        f"{n_alike} of the {n_holdout} holdout rows ({share:.1f}%) agree with a training row "
        "in every feature column that is left, whatever their targets. duplicates='group' "
        "kept together the rows that agree in every feature column, but the leak screen "
        f"then excluded the column(s) {names} as suspected leaks, and rows that differed "
        "only in them are now on both sides of the split, so the headline would not "
        "measure what duplicates='group' promises. Remove the column(s) " + names
        + " from the table and run again.")
    return message + " " + COPIES_ADVICE + _dates_text(roles)


def build_features(frame, roles):
    """The feature columns, in file order, typed for preprocessing; and their kinds."""
    feature_records = roles[roles["role"] == "feature"]
    kinds = dict(zip(feature_records["name"], feature_records["kind"]))
    columns = {name: _feature_values(frame[name], kind) for name, kind in kinds.items()}
    features = pd.DataFrame(columns, index=frame.index, columns=list(kinds))
    missing = {name: (features[name].isna() if kinds[name] == "numeric"
                      else frame[name].isna()).to_numpy(dtype=bool)
               for name in kinds}
    return features, kinds, missing


def encode_target(values, task_info):
    """0/1 with 1 the positive class (classification), or floats (regression)."""
    if task_info["type"] == "classification":
        positive = task_info["positive_class"]
        return np.array([value == positive for value in values], dtype=np.int64)
    # Step 4 has made sure the target is of integer or float type: nothing to coerce.
    return np.asarray(values, dtype=float)


def _entity_row_cap(usable, units, profile, seed):
    """Step 6 with an entity column: whole entities, file order kept.

    Every entity of a quarter or more of the usable rows is kept; the others are
    taken in an order drawn from the seed, each kept when all of its rows still
    fit under the cap. When an entity holds more than the cap, or the large ones
    do not fit under it together, `FastProfileGroupError` is raised before any
    row is left out (the caller first asks whether the full profile would stop).
    """
    if profile != "fast" or len(usable) <= ROW_CAP:
        return usable
    sizes = np.bincount(units)
    largest = int(sizes.max())
    large = 4 * sizes >= len(usable)
    held_by_large = int(sizes[large].sum())
    if largest > ROW_CAP:
        raise errors.FastProfileGroupError(
            f"One entity holds {largest} of the {len(usable)} usable rows, more than the "
            f"fast profile's cap of {ROW_CAP}; the cap keeps whole entities, so it cannot "
            "keep this one, and leaving it out would describe another table. "
            "Use profile='full'.")
    if held_by_large > ROW_CAP:
        raise errors.FastProfileGroupError(
            f"{int(large.sum())} entities each hold a quarter or more of the {len(usable)} "
            f"usable rows, {held_by_large} rows together, more than the fast profile's cap "
            f"of {ROW_CAP}; the cap never leaves out an entity of a quarter or more of the "
            "rows, so it cannot be applied. Use profile='full'.")
    keep_unit = large.copy()
    total = held_by_large
    for u in _stream(seed, _ROW_CAP_STREAM).permutation(len(sizes)):
        if not keep_unit[u] and total + sizes[u] <= ROW_CAP:
            keep_unit[u] = True
            total += int(sizes[u])
    return usable[keep_unit[units]]


def _read_entity_column(values, name, usable):
    """The entity codes of every loaded row (-1 where empty); an empty usable cell stops."""
    codes = read_entities(values, name=name)
    n_empty = int((codes[usable] < 0).sum())
    if n_empty:
        raise errors.GroupColumnError(
            f"The entity column {name!r} is empty in {n_empty} usable row(s) (missing, or "
            "text of only white space); every row with a target must name its entity. "
            "Fill the cells in or remove the rows.")
    return codes


def _row_cap(usable, profile, seed):
    """Step 6, random scheme: a simple random sample of 2,000 usable rows, file order kept."""
    if profile != "fast" or len(usable) <= ROW_CAP:
        return usable
    rng = _stream(seed, _ROW_CAP_STREAM)
    kept = rng.choice(len(usable), size=ROW_CAP, replace=False)
    return usable[np.sort(kept)]


def _fast_error(all_keys, gap, message):
    """The fast profile's error, but only if the full profile would form the split.

    The split is worked out on every usable row, as the full profile would; if
    that stops too, its own error is raised instead (`too_few_rows`, `gap_too_large`).
    """
    split_time(all_keys, gap=gap)
    return errors.FastProfileError(message)


def _time_row_cap(keys, profile, gap=0):
    """Step 6, time order: a mask of the most recent whole timestamps holding at most 2,000 rows."""
    keep = np.ones(len(keys), dtype=bool)
    if profile != "fast" or len(keys) <= ROW_CAP:
        return keep
    stamps, stamp_of_row, counts = np.unique(keys, return_inverse=True, return_counts=True)
    held_from_latest = np.cumsum(counts[::-1])
    n_kept_stamps = int(np.searchsorted(held_from_latest, ROW_CAP, side="right"))
    if n_kept_stamps == 0:
        raise _fast_error(
            keys, gap,
            f"The most recent timestamp alone holds {int(counts[-1])} rows, more than the fast "
            f"profile's cap of {ROW_CAP}; use profile='full'.")
    return stamp_of_row.reshape(-1) >= len(stamps) - n_kept_stamps


def _stamp_value(key, form):
    """A timestamp as written in `validation`: ISO text, or a number for the numeric form.

    An exact integer key (D50) is written as the whole number it is.
    """
    if form == "numeric":
        return int(key) if isinstance(key, np.integer) else float(key)
    return pd.Timestamp(key).isoformat()


def _panel_autocorrelation(keys, y):
    """Lag-1 autocorrelation of the per-timestamp mean of `y`, timestamps in order.

    None when those means do not vary: it has no value. Raises
    `NonFiniteScoreError` when it cannot be computed because a number overflows.
    """
    stamps, stamp_of_row = np.unique(keys, return_inverse=True)
    stamp_of_row = stamp_of_row.reshape(-1)
    with np.errstate(over="ignore", invalid="ignore"):
        sums = np.bincount(stamp_of_row, weights=y, minlength=len(stamps))
        sizes = np.bincount(stamp_of_row, minlength=len(stamps))
        means = sums / sizes
    if np.all(np.isfinite(means)) and (len(means) < 2 or np.all(means == means[0])):
        return None
    r = metrics.lag1_autocorrelation(means)
    if not np.isfinite(r):
        raise errors.NonFiniteScoreError(
            "The lag-1 autocorrelation of the per-timestamp mean of the target over the "
            "training timestamps cannot be computed: a number on the way is not finite. "
            "Likely cause: target values too large for squared error.")
    return r


def _check_one_row_per_stamp(keys, codes, group_col):
    """Under time order in a regression, an entity has at most one row per timestamp."""
    n_repeated = int(metrics.repeated_in_groups(keys, codes).sum())
    if n_repeated:
        raise errors.GroupColumnError(
            f"{n_repeated} of the rows split share their entity of {group_col!r} and their "
            "timestamp with another row; under time order each entity must be one series, "
            "with one row per timestamp, to have one previous value. Remove or combine the "
            "repeated rows.")


TARGET_TOO_LARGE = ("a target value is likely too large for squared error (its square is "
                    "beyond the largest double, about 1.8e308).")


def holdout_baselines(task_type, y_all, training_rows, holdout_rows, previous_holdout=None):
    """Step 11's baselines scored on the holdout: `(baselines, predictions by name)`.

    They need no fit, so they are scored as soon as the split is made; one whose
    score is not finite stops the run (spec 3, "Numbers that are not finite", rule 1).
    """
    if task_type == "classification":
        return [{"name": "chance", "score": 0.5, "used": True}], {"chance": None}
    y_holdout = y_all[holdout_rows]
    training_mean = float(np.mean(y_all[training_rows]))
    preds = {"training_mean": np.full(len(holdout_rows), training_mean)}
    if previous_holdout is not None:
        if np.isnan(previous_holdout).any():
            raise RuntimeError(
                "the previous-value baseline has no value for a holdout row; the split or "
                "the baseline is wrong")
        preds["previous_value"] = previous_holdout
    baselines = [{"name": name, "score": metrics.mse(y_holdout, pred), "used": False}
                 for name, pred in preds.items()]
    bad = [b for b in baselines if not np.isfinite(b["score"])]
    if bad:
        raise errors.NonFiniteScoreError(
            "The " + " and ".join(f"{b['name']} baseline's ({b['score']!r})" for b in bad)
            + " squared error on the holdout is not finite: " + TARGET_TOO_LARGE)
    # The lower squared error is the headline baseline; a tie goes to training_mean.
    used = min(range(len(baselines)), key=lambda i: (baselines[i]["score"], i))
    baselines[used]["used"] = True
    return baselines, preds


# ------------------------------------------------------------ fitting

class FitData:
    """The preprocessed matrices of one fit: its training rows and the rows it predicts."""

    def __init__(self, features, kinds, y, train_rows, predict_rows):
        step_factories = extensions.current_overrides()["steps"] or []
        train_x = features.iloc[train_rows].copy()
        predict_x = features.iloc[predict_rows].copy()
        train_y = pd.Series(y[train_rows], index=train_x.index)
        for factory in step_factories:
            step = factory()
            step.fit(train_x, train_y)
            train_x = step.transform(train_x)
            predict_x = step.transform(predict_x)
        preprocessor = Preprocessor(kinds).fit(train_x)
        self.train_x = preprocessor.transform(train_x)
        self.predict_x = preprocessor.transform(predict_x)
        self.train_y = train_y
        self.predict_y = y[predict_rows]


def fit_and_predict(config, task, data, seed, n_jobs):
    """A fresh estimator of `config`, fitted on the fit's training rows; its predictions."""
    estimator = config["make"](task, dict(config["params"]), seed, n_jobs)
    estimator.fit(data.train_x, data.train_y)
    return models.predict_values(estimator, data.predict_x, task)


def _error_text(exc):
    return f"{type(exc).__name__}: {exc}"


def _score(metric, y, values):
    return metrics.roc_auc(y, values) if metric == "roc_auc" else metrics.mse(y, values)


def _mean_and_std(scores):
    """Mean and sample deviation of finite fold scores, computed so that neither overflows."""
    scores = np.asarray(scores, dtype=float)
    mean = float(np.mean(scores))
    if not np.isfinite(mean):
        mean = float(np.sum(scores / len(scores)))
    if len(scores) < 2:
        return mean, None
    spread = float(np.std(scores, ddof=1))
    if not np.isfinite(spread):
        # Scaled first: the squares of large deviations overflow before their mean does.
        deviations = scores - mean
        scale = float(np.max(np.abs(deviations)))
        spread = float(scale * np.sqrt(np.sum((deviations / scale) ** 2) / (len(scores) - 1)))
    return mean, spread


def _fail(record, error):
    """Record a configuration as failed in selection: no fold score is kept (D44)."""
    record["status"] = "failed"
    record["error"] = error
    record["fold_scores"] = [None] * N_FOLDS


def _fold_is_scored(task_type, y, fold):
    if task_type != "classification":
        return True
    return (len(np.unique(y[fold["training"]])) == 2
            and len(np.unique(y[fold["validation"]])) == 2)


def select(configs, features, kinds, y, folds, scored, task_type, metric, seed, n_jobs):
    """Step 9: every configuration on every scored fold. Returns the models records."""
    records = [{
        "config": c["config"], "family": c["family"], "params": dict(c["params"]),
        "status": "fitted", "error": None, "fold_scores": [None] * N_FOLDS,
        "cv_mean": None, "cv_std": None, "chosen": False,
    } for c in configs]
    for k, fold in enumerate(folds):
        if not scored[k]:
            continue
        data = FitData(features, kinds, y, fold["training"], fold["validation"])
        for config, record in zip(configs, records):
            if record["status"] == "failed":
                continue
            # Only what the model does can fail it: making it, its fit, its
            # prediction and the reading of what it handed back (step 9). The
            # scoring below is the tool's own; if it raises, the run ends.
            try:
                values = fit_and_predict(config, task_type, data, seed, n_jobs)
            except errors.ProfileError:
                raise
            except Exception as exc:  # a configuration that raises is recorded, never dropped
                _fail(record, _error_text(exc))
                continue
            fold_score = _score(metric, data.predict_y, values)
            if not np.isfinite(fold_score):
                _fail(record,
                      f"its {metric} on fold {k + 1} is not finite ({fold_score!r}), from "
                      "predictions that are finite; its predictions are likely too large "
                      "for squared error")
                continue
            record["fold_scores"][k] = fold_score
    for record in records:
        if record["status"] != "fitted":
            continue
        fold_scores = [s for s in record["fold_scores"] if s is not None]
        record["cv_mean"], record["cv_std"] = _mean_and_std(fold_scores)
    return records


# ------------------------------------------------------------ the run

def run(data, target, task="auto", time_col=None, gap=0, profile="full", seed=0,
        out_dir=None, n_jobs=-1, duplicates="error", group_col=None):
    """`profile()` itself: see spec 2.1 and section 3."""
    check_arguments(data, target, task, time_col, gap, profile, seed, out_dir, n_jobs,
                    duplicates, group_col)
    jobs = resolve_n_jobs(n_jobs)
    # `False` states that time order does not matter; past step 5 it reads as `None`.
    time_name = time_col if isinstance(time_col, str) else None
    stated = time_col is False
    started = time.perf_counter()
    # Arithmetic that overflows is checked where it matters (spec 3, "Numbers that are
    # not finite"); numpy's own messages are held back, and the caller's settings restored.
    with threadpool_limits(limits={"blas": 1, "openmp": jobs}),             np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        document, predictions = _run_profile(data, target, task, time_name, int(gap),
                                             profile, int(seed), jobs, duplicates,
                                             stated, group_col)
    document["meta"]["n_jobs"] = jobs
    document["meta"]["timing"] = {"seconds": round(time.perf_counter() - started, 3)}
    result = ProfileResult(document, predictions)
    if out_dir is not None:
        result.write(out_dir)
    return result


def _run_profile(data, target, task, time_col, gap, profile, seed, n_jobs, duplicates,
                 time_order_stated, group_arg):
    """The run. `time_col` is a name or None; `time_order_stated` says whether None
    came from `time_col=False`. `group_arg` is `group_col` as passed."""
    # Step 1: load and count.
    loaded_frame, info = load_table(data)
    # Step 1a: object columns of numbers are read as numbers, once; every later
    # step reads the read table. The types as loaded are what the output reports.
    loaded_dtypes = {name: str(loaded_frame[name].dtype) for name in loaded_frame.columns}
    frame = read_number_columns(loaded_frame)
    warnings = list(info["warnings"])
    n_loaded = len(frame)
    duplicate_rows = _count_duplicate_rows(frame)
    if duplicate_rows:
        warnings.append(_warning(
            "duplicate_rows", None,
            f"{duplicate_rows} row(s) exactly repeat an earlier row; they are kept."))

    # Step 2: usable rows.
    counts = usable_rows(frame, target, time_col)
    usable = counts["usable"]
    # `group_col` as passed is recorded; a name is the entity column (D57).
    group_col = group_arg if isinstance(group_arg, str) else None
    if group_col is not None and group_col not in frame.columns:
        raise errors.ColumnNotFoundError(
            f"No column named {group_col!r}. The columns are: "
            f"{', '.join(repr(str(c)) for c in frame.columns)}.")
    if len(usable) < MIN_USABLE_ROWS:
        with_what = "a target" if time_col is None else "a target and a time value"
        raise errors.TooFewRowsError(
            f"{len(usable)} usable rows (with {with_what}); at least {MIN_USABLE_ROWS} "
            "are needed.")

    # The entity column, read once over every loaded row as it was loaded.
    entity = (None if group_col is None
              else _read_entity_column(loaded_frame[group_col], group_col, usable))
    grouped = group_col is not None and time_col is None

    # Step 3: under time order, the time column is read over all its loaded rows
    # (step 1a, and the 2**53 rule), and the keys of the usable rows are kept.
    time_ordered = time_col is not None
    if time_ordered:
        usable_keys, time_form = read_time(frame[time_col], rows=usable)
        no_key = (np.isnat(usable_keys) if np.issubdtype(usable_keys.dtype, np.datetime64)
                  else np.isnan(usable_keys))
        if no_key.any():
            raise errors.TimeColumnError(
                f"time column: {int(no_key.sum())} value(s) of {time_col!r} could not be read "
                "as a time.")

    # Step 4: the task, on the usable rows, before any cap. The target is read
    # once, text without the white space round it, and every later step uses it.
    target_read = read_target_values(frame[target]).reset_index(drop=True)
    task_info = decide_task(frame[target], task, rows=usable)
    for code in task_info["warnings"]:
        warnings.append(_warning(
            code, target,
            f"Every value of {target!r} is a whole number and the rule chose regression; "
            "declare task='classification' only if it has two values."))

    # Step 5: column roles. Then both statements are asked for, whatever the table
    # holds, before the name collisions, the cap, the split and any fit (D63).
    roles = roles_of_read_columns(frame, target, time_col, loaded_dtypes, group_col)
    dates = _date_columns(roles)
    if time_col is None and not time_order_stated:
        raise errors.TimeOrderUndeclaredError(
            _time_order_message(dates, entities_unsaid=group_arg is None))
    if group_arg is None:
        raise errors.EntitiesUndeclaredError(
            _entities_message(time_col, usable_keys if time_ordered else None))
    warnings.extend(_column_warnings(roles, time_col))
    features, kinds, feature_missing = build_features(frame, roles)
    if not kinds:
        raise errors.NoFeaturesError(
            "No column can be used as a feature. Excluded: " + _excluded_text(roles) + ".")
    usable_features = features.iloc[usable]
    check_feature_names(kinds, {
        name: sorted(set(usable_features[name]))
        for name, kind in kinds.items() if kind == "categorical"})

    task_type = task_info["type"]

    def _units_among(rows):
        """Step 7's units among `rows` (or None: split one by one), and with
        `duplicates="group"` the number of those rows that agree with another of them
        in every feature column, whatever their targets."""
        if time_ordered:
            return None, None
        if duplicates != "group":
            return (None if entity is None else entity[rows]), None
        sets, n_same = same_feature_sets(features, rows)
        if entity is not None:
            return joined_units(entity[rows], sets), n_same
        return (sets if n_same else None), n_same

    def _full_profile_stop():
        """Raise the full profile's own error when, on every usable row, it would
        stop before its split is formed (`too_few_rows`, `gap_too_large`,
        `too_few_groups`)."""
        if task_type == "classification":
            y_usable = encode_target(target_read.iloc[usable].to_numpy(), task_info)
            n_pos = int(y_usable.sum())
            if min(n_pos, len(usable) - n_pos) < MIN_CLASS_ROWS:
                raise errors.TooFewRowsError(
                    f"The rarer class has {min(n_pos, len(usable) - n_pos)} rows; "
                    f"at least {MIN_CLASS_ROWS} of each class are needed.")
        if time_ordered:
            split_time(usable_keys, gap=gap)
        usable_units, _ = _units_among(usable)
        if usable_units is not None:
            # Raises the full profile's own `too_few_groups` when it would stop too.
            usable_classes = (target_read.iloc[usable].to_numpy()
                              if task_type == "classification" else None)
            split_random(len(usable), classes=usable_classes, seed=seed,
                         groups=usable_units)

    def _fast_rows_error(reason):
        """Under a cap that left rows out: the full profile's error, or the cap's own."""
        _full_profile_stop()
        return errors.FastProfileRowsError(
            f"Under the fast profile's cap of {ROW_CAP} rows: {reason}; the full "
            "profile would run: use profile='full'.")

    # Step 6: the fast profile's row cap.
    if time_ordered:
        keep = _time_row_cap(usable_keys, profile, gap)
        kept, kept_keys = usable[keep], usable_keys[keep]
    elif grouped:
        try:
            kept = _entity_row_cap(usable, units_of(entity[usable], len(usable)),
                                   profile, seed)
        except errors.FastProfileGroupError:
            _full_profile_stop()
            raise
    else:
        kept = _row_cap(usable, profile, seed)
    dropped_row_cap = len(usable) - len(kept)
    y_all = np.zeros(n_loaded, dtype=np.int64 if task_type == "classification" else float)
    y_all[kept] = encode_target(target_read.iloc[kept].to_numpy(), task_info)

    if len(kept) < MIN_USABLE_ROWS:
        message = f"{len(kept)} rows are left; at least {MIN_USABLE_ROWS} are needed."
        if dropped_row_cap:
            raise _fast_rows_error(message)
        raise errors.TooFewRowsError(message)
    if task_type == "classification":
        n_positive = int(y_all[kept].sum())
        if min(n_positive, len(kept) - n_positive) < MIN_CLASS_ROWS:
            message = (f"The rarer class has {min(n_positive, len(kept) - n_positive)} rows; "
                       f"at least {MIN_CLASS_ROWS} of each class are needed.")
            if dropped_row_cap:
                raise _fast_rows_error(message)
            raise errors.TooFewRowsError(message)

    # Step 7: the holdout and the folds, as positions of the kept rows. Under
    # `duplicates="group"` the rows that agree in every feature column of step 5 are
    # put together here, whatever their targets.
    units, n_rows_same_features = _units_among(kept)
    if n_rows_same_features:
        warnings.append(_warning(
            "same_features_grouped", None,
            f"{n_rows_same_features} of the {len(kept)} rows split agree with another of "
            "them in every feature column; with duplicates='group' the rows that agree "
            "were kept together, whatever their targets, on one side of the holdout split "
            "and in one fold, so no combination of feature values is on both sides. "
            "With duplicates='group' the headline measures whether a combination of feature "
            "values that the model has not seen can be predicted. On a table of few-valued "
            "columns that can give negative skill where duplicates='keep' finds signal."))
    if time_ordered:
        try:
            split = split_time(kept_keys, gap=gap)
        except errors.TooFewRowsError as exc:
            if dropped_row_cap:
                raise _fast_error(
                    usable_keys, gap,
                    f"Under the fast profile's cap of {ROW_CAP} rows: {exc}; use "
                    "profile='full'.") from exc
            raise
        timestamps_unique = len(np.unique(kept_keys)) == len(kept_keys)
    else:
        classes = target_read.iloc[kept].to_numpy() if task_type == "classification" else None
        if units is not None:
            try:
                split = split_random(len(kept), classes=classes, seed=seed,
                                     groups=units)
            except errors.TooFewGroupsError as exc:
                if dropped_row_cap:
                    raise _fast_rows_error(str(exc)) from exc
                raise
        else:
            split = split_random(len(kept), classes=classes, seed=seed)
        if units is not None:
            warnings.extend(_large_unit_warnings(units, len(kept), classes, split["holdout"]))
        split["gap"] = split["holdout"][:0]
        for fold in split["folds"]:
            fold["gap"] = fold["training"][:0]
    training_rows = kept[split["training"]]
    holdout_rows = kept[split["holdout"]]
    gap_rows = kept[split["gap"]]
    folds = [{"training": kept[f["training"]], "validation": kept[f["validation"]],
              "gap": kept[f["gap"]]} for f in split["folds"]]
    if task_type == "classification":
        for name, rows in (("holdout", holdout_rows), ("training rows", training_rows)):
            if len(np.unique(y_all[rows])) < 2:
                raise errors.OneClassError(f"The {name} hold one class only.")
    scored = [_fold_is_scored(task_type, y_all, fold) for fold in folds]
    for k, is_scored in enumerate(scored):
        if not is_scored:
            warnings.append(_warning(
                "fold_not_scored", None,
                f"Fold {k + 1}: its training or validation rows hold one class; it is "
                "not scored."))
    if not any(scored):
        raise errors.OneClassError("No fold holds both classes in training and validation.")

    # Step 11 under time order in a regression: each entity one series; a panel with
    # no entities (`group_col=False`) whose per-timestamp mean persists is warned of.
    by_entity = task_type == "regression" and time_ordered and entity is not None
    if by_entity:
        _check_one_row_per_stamp(kept_keys, entity[kept], group_col)
    panel_warning = None
    if task_type == "regression" and time_ordered and entity is None and not timestamps_unique:
        r = _panel_autocorrelation(kept_keys[split["training"]], y_all[training_rows])
        if r is not None and r > PANEL_AUTOCORRELATION:
            panel_warning = _warning(
                "autocorrelated_panel_target", target,
                f"The per-timestamp mean of {target!r} has lag-1 autocorrelation {r:.2f} "
                f"over the training timestamps (above {PANEL_AUTOCORRELATION}). With "
                "repeated timestamps the baseline is the training mean, which a "
                "persistent target beats easily; profile its changes instead.")

    # Step 11's baselines need no fit: scored on the holdout now, before any (D44).
    previous = previous_holdout = None
    n_without_previous = None
    if by_entity:
        # Each entity's latest row `gap + 1` distinct timestamps earlier, among the kept rows.
        previous = np.full(n_loaded, np.nan)
        previous[kept] = metrics.previous_value(y_all[kept], kept_keys, gap=gap,
                                                groups=entity[kept])
        previous_holdout = previous[holdout_rows]
        no_previous = np.isnan(previous_holdout)
        n_without_previous = int(no_previous.sum())
        # A row whose entity has no earlier row is predicted by what is known: the mean.
        previous_holdout = np.where(no_previous, float(np.mean(y_all[training_rows])),
                                    previous_holdout)
        if n_without_previous:
            warnings.append(_warning(
                "no_previous_value", None,
                f"{n_without_previous} holdout row(s) belong to an entity of {group_col!r} "
                f"with no row at least {gap + 1} timestamp(s) earlier; for them the "
                "previous-value baseline predicts the training mean."))
    elif task_type == "regression" and time_ordered and timestamps_unique:
        # The target `gap + 1` timestamps earlier among the kept rows in time order.
        previous = np.full(n_loaded, np.nan)
        previous[kept] = metrics.previous_value(y_all[kept], kept_keys, gap=gap)
        previous_holdout = previous[holdout_rows]
    baselines, baseline_preds = holdout_baselines(
        task_type, y_all, training_rows, holdout_rows, previous_holdout=previous_holdout)

    # Step 8: the leak screen, on training rows only, over the selection folds.
    roles = roles.copy()
    for record in leakage.screen(features, kinds, feature_missing, y_all, training_rows,
                                 folds, scored, task_type, seed, previous=previous,
                                 previous_by_entity=by_entity):
        name = record["name"]
        if record["leak_rule"] is not None:
            at = roles.index[roles["name"] == name]
            roles.loc[at, "role"] = "excluded"
            roles.loc[at, "reason"] = "suspected_leak"
            roles.loc[at, "leak_rule"] = record["leak_rule"]
            del kinds[name]
            why = ("it is filled in for one class and empty for the other"
                   if record["leak_rule"] == "missing_by_class"
                   else f"on its own it predicts the target with skill {record['skill']:.3f}")
            warnings.append(_warning(
                "suspected_leak", name,
                f"{name!r} is excluded as a suspected leak: {why} on the training rows. "
                "Check whether it is known before the outcome."))
        elif record["warn"]:
            warnings.append(_warning(
                "strong_single_column", name,
                f"{name!r} on its own predicts the target with skill {record['skill']:.3f} "
                "on the training rows. It is kept as a feature; check that it is known "
                "before the outcome."))
    if not kinds:
        raise errors.NoFeaturesError(
            "No feature is left after the leak screen. Excluded: "
            + _excluded_text(roles) + ".")
    features = features[list(kinds)]

    # Step 8a: holdout rows with a copy among the training rows, before anything is fitted.
    # Under `duplicates="group"` the units were formed on the features of step 5; once
    # the screen has excluded a column, one holdout row that agrees with a training row
    # in every feature that is left is enough to stop.
    n_copies = copy_share = None
    if not time_ordered:
        if duplicates == "group" and (roles["reason"] == "suspected_leak").any():
            n_alike = count_holdout_copies(features, None, training_rows, holdout_rows)
            if n_alike:
                raise errors.HoldoutCopiesError(
                    _alike_after_leak_message(n_alike, len(holdout_rows), roles))
        n_copies = count_holdout_copies(features, y_all, training_rows, holdout_rows)
        copy_share = n_copies / len(holdout_rows)
        if n_copies >= COPIES_MIN_ROWS and copy_share > COPIES_MAX_SHARE:
            if duplicates != "keep":
                raise errors.HoldoutCopiesError(
                    _copies_message(n_copies, len(holdout_rows), roles))
            warnings.append(_warning(
                "holdout_copies_kept", None,
                f"{n_copies} of the {len(holdout_rows)} holdout rows "
                f"({100.0 * copy_share:.1f}%) have a copy among the training rows; they "
                "were kept (duplicates='keep'), so the headline may be inflated by rows "
                "a model can recall."))

    # Step 9: select by five folds inside the training rows.
    metric = "roc_auc" if task_type == "classification" else "mse"
    greater_is_better = metric == "roc_auc"
    configs = models.configurations(models.active_families(task_type, profile))
    model_records = select(configs, features, kinds, y_all, folds, scored, task_type,
                           metric, seed, n_jobs)

    # Steps 10 and 11: refit the chosen one on all training rows; score the holdout once.
    final = FitData(features, kinds, y_all, training_rows, holdout_rows)
    chosen_index, holdout_pred = None, None
    for index in models.ranked_configurations(model_records, greater_is_better):
        try:
            holdout_pred = fit_and_predict(configs[index], task_type, final, seed, n_jobs)
        except errors.ProfileError:
            raise
        except Exception as exc:
            # Failing after the choice keeps the fold scores it was chosen on.
            model_records[index].update(status="failed", error=_error_text(exc))
            continue
        chosen_index = index
        break
    for record in model_records:
        if record["status"] == "failed":
            warnings.append(_warning(
                "model_failed", None,
                f"Configuration {record['config']} failed: {record['error']}"))
    if chosen_index is None:
        raise errors.AllModelsFailedError(
            "Every model configuration failed: " + "; ".join(
                f"{r['config']}: {r['error']}" for r in model_records))
    chosen = model_records[chosen_index]
    chosen["chosen"] = True

    y_holdout = y_all[holdout_rows]
    score = _score(metric, y_holdout, holdout_pred)
    if not np.isfinite(score):
        # A score, not a failure: the next configuration is not tried (D38, D44).
        raise errors.NonFiniteScoreError(
            f"The holdout {metric} of the chosen configuration {chosen['config']} is not "
            f"finite ({score!r}), from predictions that are finite. Likely cause: its "
            "predictions or the target values are too large for squared error.")
    if panel_warning is not None:
        warnings.append(panel_warning)
    baseline = next(b for b in baselines if b["used"])
    headline = {
        "metric": metric,
        "score": float(score),
        "model": chosen["config"],
        "baseline": baseline["name"],
        "baseline_score": float(baseline["score"]),
        "skill": metrics.skill(metric, score, baseline["score"]),
    }
    if headline["skill"] is not None and not np.isfinite(headline["skill"]):
        raise errors.NonFiniteScoreError(
            f"headline.skill is not finite ({headline['skill']!r}) from the score "
            f"{float(score)!r} and the {baseline['name']} baseline's score "
            f"{float(baseline['score'])!r}: the baseline's error is too small beside the "
            "model's for their ratio to be held.")
    if headline["skill"] is None:
        warnings.append(_warning(
            "skill_undefined", None,
            f"Skill has no value: the {baseline['name']} baseline predicts every holdout row "
            "exactly (squared error 0), so 1 - score / baseline score is undefined. The score "
            "and every baseline are reported as measured; there is no interval on skill and "
            "no p-value."))

    # Step 12: the interval and the p-value, from the holdout predictions alone.
    spread, spread_warnings = uncertainty.headline_uncertainty(
        metric, y_holdout, holdout_pred, baseline_preds[baseline["name"]],
        headline["skill"], profile, seed,
        holdout_keys=kept_keys[split["holdout"]] if time_ordered else None,
        holdout_groups=None if units is None else units[split["holdout"]])
    headline.update(spread)
    for code, message in spread_warnings:
        warnings.append(_warning(code, None, message))
    if task_type == "classification":
        rarer = int(min(y_holdout.sum(), len(y_holdout) - y_holdout.sum()))
        small = rarer < SMALL_HOLDOUT
        detail = f"the rarer class has {rarer} holdout rows"
    else:
        small = len(holdout_rows) < SMALL_HOLDOUT
        detail = f"the holdout has {len(holdout_rows)} rows"
    if small:
        warnings.append(_warning(
            "small_holdout", None,
            f"Small holdout: {detail} (fewer than {SMALL_HOLDOUT}); the score is uncertain."))

    # Accounting.
    scored_rows = np.concatenate([training_rows, holdout_rows])
    roles["n_imputed"] = [int(feature_missing[name][scored_rows].sum()) if name in kinds else 0
                          for name in roles["name"]]
    rows = [
        {"item": "loaded", "count": n_loaded},
        {"item": "dropped_missing_target", "count": counts["dropped_missing_target"]},
        {"item": "dropped_missing_time", "count": counts["dropped_missing_time"]},
        {"item": "dropped_gap", "count": len(gap_rows)},
        {"item": "dropped_row_cap", "count": dropped_row_cap},
        {"item": "training", "count": len(training_rows)},
        {"item": "holdout", "count": len(holdout_rows)},
    ]
    if [r["item"] for r in rows] != ROWS_ITEMS or sum(r["count"] for r in rows[1:]) != n_loaded:
        raise RuntimeError(f"row accounting does not add up: {rows}")

    predictions = pd.DataFrame({
        "row": holdout_rows.astype(np.int64),
        "y_true": target_read.iloc[holdout_rows].to_numpy(),
        "y_pred": np.asarray(holdout_pred, dtype=float),
    })

    n_scored = int(sum(scored))
    validation = {
        "scheme": ("time_ordered" if time_ordered
                   else "grouped" if units is not None else "random"),
        "holdout_share": len(holdout_rows) / (len(training_rows) + len(holdout_rows)
                                              + len(gap_rows)),
        "n_folds": N_FOLDS,
        "stratified": task_type == "classification" and not time_ordered,
        "gap": gap, "time_col": time_col, "time_form": None, "train_end": None,
        "holdout_start": None, "holdout_end": None, "n_train_timestamps": None,
        "n_holdout_timestamps": None, "timestamps_unique": None,
        "folds": [{"fold": k + 1, "training_rows": len(f["training"]),
                   "validation_rows": len(f["validation"]), "gap_rows": len(f["gap"]),
                   "scored": bool(scored[k])} for k, f in enumerate(folds)],
        "caveat": (None if time_ordered
                   else GROUPED_CAVEAT if units is not None else RANDOM_CAVEAT),
        "n_holdout_copies": n_copies,
        "holdout_copy_share": copy_share,
        "group_col": group_col,
        "n_groups_training": (
            int(len(np.unique(units[split["training"]]))) if units is not None
            else None if entity is None else int(len(np.unique(entity[training_rows])))),
        "n_groups_holdout": (
            int(len(np.unique(units[split["holdout"]]))) if units is not None
            else None if entity is None else int(len(np.unique(entity[holdout_rows])))),
        "n_holdout_without_previous": n_without_previous,
        "n_rows_same_features": n_rows_same_features,
    }
    for record in warnings:
        if record["code"] == "date_column_undeclared" and validation["caveat"]:
            record["message"] = (
                f"{record['column']!r} looks like a date and is excluded (time_col=False). "
                f"Pass it as time_col for a time-ordered split. {validation['caveat']}")
    if time_ordered:
        training_keys = kept_keys[split["training"]]
        holdout_keys = kept_keys[split["holdout"]]
        validation.update({
            "time_form": time_form,
            "train_end": _stamp_value(training_keys.max(), time_form),
            "holdout_start": _stamp_value(holdout_keys.min(), time_form),
            "holdout_end": _stamp_value(holdout_keys.max(), time_form),
            "n_train_timestamps": int(len(np.unique(training_keys))),
            "n_holdout_timestamps": int(len(np.unique(holdout_keys))),
            "timestamps_unique": bool(timestamps_unique),
        })
    document = {
        "meta": _meta(profile, seed),
        "input": {
            "source": info["source"], "kind": info["kind"], "target": target,
            "time_col": False if time_order_stated else time_col,
            "task": task, "gap": gap, "n_rows": n_loaded,
            "n_columns": int(frame.shape[1]), "duplicate_rows": duplicate_rows,
            "duplicates": duplicates, "group_col": group_arg,
        },
        "rows": rows,
        "columns": _column_records(roles),
        "task": {
            "type": task_type, "decided_by": task_info["decided_by"],
            "n_distinct": int(task_info["n_distinct"]),
            "target_dtype": loaded_dtypes[target],
            "classes": (None if task_info["classes"] is None
                        else [class_value(value) for value in task_info["classes"]]),
            "positive_class": (None if task_info["positive_class"] is None
                               else class_value(task_info["positive_class"])),
        },
        "validation": validation,
        "headline": headline,
        "baselines": baselines,
        "cv": {
            "metric": metric, "greater_is_better": greater_is_better, "n_folds": N_FOLDS,
            "n_scored_folds": n_scored, "n_configurations": len(model_records),
            "chosen": chosen["config"], "chosen_mean": chosen["cv_mean"],
            "chosen_std": chosen["cv_std"],
        },
        "models": model_records,
        "warnings": warnings,
        "limitations": list(LIMITATIONS),
    }
    return document, predictions


COLUMN_TEXT_KEYS = ("name", "dtype", "role", "kind", "reason", "leak_rule")


def _column_records(roles):
    """The `columns` records of `roles`: a text field with no value is None, never NaN.

    A text column of a frame marks a missing cell as NaN; the document says
    "does not apply" with null, and holds no number that is not finite.
    """
    records = roles.to_dict(orient="records")
    for record in records:
        for key in COLUMN_TEXT_KEYS:
            value = record.get(key)
            if not isinstance(value, str) and value is not None and pd.isna(value):
                record[key] = None
    return records


def _meta(profile, seed):
    import sklearn
    import threadpoolctl

    return {
        "tool": "mlprofessor", "schema_version": 1, "profile": profile, "seed": seed,
        "n_jobs": None,
        "versions": {
            "python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit-learn": sklearn.__version__,
            "threadpoolctl": threadpoolctl.__version__,
        },
        "timing": {},
    }


__all__ = ["run", "check_arguments", "resolve_n_jobs", "select", "FitData",
           "build_features", "encode_target", "count_holdout_copies", "same_feature_sets",
           "joined_units"]
