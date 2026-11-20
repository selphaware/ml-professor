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
from mlprofessor.columns import (column_roles, feature_numbers, numeric_values, parse_numbers,
                                 stripped_text, type_family)
from mlprofessor.load import load_table, usable_rows
from mlprofessor.preprocess import MISSING_LEVEL, Preprocessor, check_feature_names
from mlprofessor.result import LIMITATIONS, RANDOM_CAVEAT, ROWS_ITEMS, ProfileResult
from mlprofessor.split import MAX_SEED, N_FOLDS, read_time, split_random, split_time
from mlprofessor.task import TASKS, decide_task, read_target_values

PROFILES = ("full", "fast")
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


def check_arguments(data, target, task, time_col, gap, profile, seed, out_dir, n_jobs):
    """Raise `InputError` for any argument value outside spec 2.1, before loading."""
    if not isinstance(data, (pd.DataFrame, str, os.PathLike)):
        raise errors.InputError(
            f"data must be a pandas DataFrame or the path of a CSV file; "
            f"got {type(data).__name__}.")
    if not isinstance(target, str):
        raise errors.InputError(f"target must be a column name (text); got {target!r}.")
    if task not in TASKS:
        raise errors.InputError(f"task must be one of {', '.join(TASKS)}; got {task!r}.")
    if time_col is not None and not isinstance(time_col, str):
        raise errors.InputError(f"time_col must be a column name or None; got {time_col!r}.")
    if not _is_whole(gap) or gap < 0:
        raise errors.InputError(f"gap must be a whole number, at least 0; got {gap!r}.")
    if gap > 0 and time_col is None:
        raise errors.InputError(
            f"gap={gap} needs a time column; pass time_col, or leave gap at 0.")
    if time_col is not None and time_col == target:
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


def _excluded_text(roles):
    """Each excluded column with its reason, as `columns[].reason` spells it."""
    return "; ".join(f"{r['name']} ({r['reason']})"
                     for r in roles.to_dict(orient="records") if r["role"] == "excluded")


def resolve_n_jobs(n_jobs):
    return int(os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)


# ------------------------------------------------------------ helpers

def _warning(code, column, message):
    return {"code": code, "column": column, "message": message}


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
    """A timestamp as written in `validation`: ISO text, or a number for the numeric form."""
    if form == "numeric":
        return float(key)
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


class _NonFiniteFoldScore(Exception):
    """A fold score that is infinite or not-a-number: its configuration fails (D44)."""


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
            try:
                values = fit_and_predict(config, task_type, data, seed, n_jobs)
                fold_score = _score(metric, data.predict_y, values)
                if not np.isfinite(fold_score):
                    raise _NonFiniteFoldScore(
                        f"its {metric} on fold {k + 1} is not finite ({fold_score!r}), from "
                        "predictions that are finite; its predictions are likely too large "
                        "for squared error")
                record["fold_scores"][k] = fold_score
            except errors.ProfileError:
                raise
            except Exception as exc:  # a configuration that raises is recorded, never dropped
                record["status"] = "failed"
                record["error"] = (str(exc) if isinstance(exc, _NonFiniteFoldScore)
                                   else _error_text(exc))
                record["fold_scores"] = [None] * N_FOLDS
    for record in records:
        if record["status"] != "fitted":
            continue
        fold_scores = [s for s in record["fold_scores"] if s is not None]
        record["cv_mean"], record["cv_std"] = _mean_and_std(fold_scores)
    return records


# ------------------------------------------------------------ the run

def run(data, target, task="auto", time_col=None, gap=0, profile="full", seed=0,
        out_dir=None, n_jobs=-1):
    """`profile()` itself: see spec 2.1 and section 3."""
    check_arguments(data, target, task, time_col, gap, profile, seed, out_dir, n_jobs)
    jobs = resolve_n_jobs(n_jobs)
    started = time.perf_counter()
    # Arithmetic that overflows is checked where it matters (spec 3, "Numbers that are
    # not finite"); numpy's own messages are held back, and the caller's settings restored.
    with threadpool_limits(limits={"blas": 1, "openmp": jobs}),             np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        document, predictions = _run_profile(data, target, task, time_col, int(gap),
                                             profile, int(seed), jobs)
    document["meta"]["n_jobs"] = jobs
    document["meta"]["timing"] = {"seconds": round(time.perf_counter() - started, 3)}
    result = ProfileResult(document, predictions)
    if out_dir is not None:
        result.write(out_dir)
    return result


def _run_profile(data, target, task, time_col, gap, profile, seed, n_jobs):
    # Step 1: load and count.
    frame, info = load_table(data)
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
    if len(usable) < MIN_USABLE_ROWS:
        with_what = "a target" if time_col is None else "a target and a time value"
        raise errors.TooFewRowsError(
            f"{len(usable)} usable rows (with {with_what}); at least {MIN_USABLE_ROWS} "
            "are needed.")

    # Step 3: under time order, read the time column of the usable rows.
    time_ordered = time_col is not None
    if time_ordered:
        usable_keys, time_form = read_time(frame[time_col].iloc[usable])
        no_key = (np.isnat(usable_keys) if np.issubdtype(usable_keys.dtype, np.datetime64)
                  else np.isnan(usable_keys))
        if no_key.any():
            raise errors.TimeColumnError(
                f"time column: {int(no_key.sum())} value(s) of {time_col!r} could not be read "
                "as a time.")

    # Step 4: the task, on the usable rows, before any cap. The target is read
    # once, text without the white space round it, and every later step uses it.
    target_read = read_target_values(frame[target]).reset_index(drop=True)
    task_info = decide_task(target_read.iloc[usable], task)
    for code in task_info["warnings"]:
        warnings.append(_warning(
            code, target,
            f"Every value of {target!r} is a whole number and the rule chose regression; "
            "declare task='classification' only if it has two values."))

    # Step 5: column roles.
    roles = column_roles(frame, target, time_col)
    warnings.extend(_column_warnings(roles, time_col))
    features, kinds, feature_missing = build_features(frame, roles)
    if not kinds:
        raise errors.NoFeaturesError(
            "No column can be used as a feature. Excluded: " + _excluded_text(roles) + ".")
    usable_features = features.iloc[usable]
    check_feature_names(kinds, {
        name: sorted(set(usable_features[name]))
        for name, kind in kinds.items() if kind == "categorical"})

    # Step 6: the fast profile's row cap.
    if time_ordered:
        keep = _time_row_cap(usable_keys, profile, gap)
        kept, kept_keys = usable[keep], usable_keys[keep]
    else:
        kept = _row_cap(usable, profile, seed)
    dropped_row_cap = len(usable) - len(kept)
    task_type = task_info["type"]
    y_all = np.zeros(n_loaded, dtype=np.int64 if task_type == "classification" else float)
    y_all[kept] = encode_target(target_read.iloc[kept].to_numpy(), task_info)
    if len(kept) < MIN_USABLE_ROWS:
        raise errors.TooFewRowsError(
            f"{len(kept)} rows are left; at least {MIN_USABLE_ROWS} are needed.")
    if task_type == "classification":
        n_positive = int(y_all[kept].sum())
        if min(n_positive, len(kept) - n_positive) < MIN_CLASS_ROWS:
            raise errors.TooFewRowsError(
                f"The rarer class has {min(n_positive, len(kept) - n_positive)} rows; "
                f"at least {MIN_CLASS_ROWS} of each class are needed.")

    # Step 7: the holdout and the folds, as positions of the kept rows.
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
        split = split_random(len(kept), classes=classes, seed=seed)
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

    # Step 11's baselines need no fit: scored on the holdout now, before any (D44).
    previous = None
    if task_type == "regression" and time_ordered and timestamps_unique:
        # The target `gap + 1` timestamps earlier among the kept rows in time order.
        previous = np.full(n_loaded, np.nan)
        previous[kept] = metrics.previous_value(y_all[kept], kept_keys, gap=gap)
    baselines, baseline_preds = holdout_baselines(
        task_type, y_all, training_rows, holdout_rows,
        previous_holdout=None if previous is None else previous[holdout_rows])

    # Step 8: the leak screen, on training rows only, over the selection folds.
    roles = roles.copy()
    for record in leakage.screen(features, kinds, feature_missing, y_all, training_rows,
                                 folds, scored, task_type, seed, previous=previous):
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
    if task_type == "regression" and time_ordered and not timestamps_unique:
        r = _panel_autocorrelation(kept_keys[split["training"]], y_all[training_rows])
        if r is not None and r > PANEL_AUTOCORRELATION:
            warnings.append(_warning(
                "autocorrelated_panel_target", target,
                f"The per-timestamp mean of {target!r} has lag-1 autocorrelation {r:.2f} "
                f"over the training timestamps (above {PANEL_AUTOCORRELATION}). With "
                "repeated timestamps the baseline is the training mean, which a "
                "persistent target beats easily; profile its changes instead."))
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
        holdout_keys=kept_keys[split["holdout"]] if time_ordered else None)
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
        "scheme": "time_ordered" if time_ordered else "random",
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
        "caveat": None if time_ordered else RANDOM_CAVEAT,
    }
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
            "time_col": time_col, "task": task, "gap": gap, "n_rows": n_loaded,
            "n_columns": int(frame.shape[1]), "duplicate_rows": duplicate_rows,
        },
        "rows": rows,
        "columns": roles.to_dict(orient="records"),
        "task": {
            "type": task_type, "decided_by": task_info["decided_by"],
            "n_distinct": int(task_info["n_distinct"]),
            "target_dtype": str(frame[target].dtype),
            "classes": task_info["classes"], "positive_class": task_info["positive_class"],
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
           "build_features", "encode_target"]
