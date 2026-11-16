# ML Professor, phase 1: the contract

This file is what the application is built from and what the tests are written
against. It restates the owner's decisions (`docs/DECISIONS.md`, cited as D1 to
D22) without changing them, pins every value they leave to this stage, and
defines the interface precisely enough to test without seeing the code.

Words used throughout: **pinned** means part of the contract; **open** means
the builder may choose and no test may depend on it (section 11).

Contents: 1 scope, 2 interface, 3 methodology in order, 4 pinned values,
5 output, 6 loading and types, 7 edges, 8 failure modes, 9 facts from Phase 0,
10 measured values, 11 what is open, 12 decision index.

---

## 1. Scope (D1, D3)

`mlprofessor` is a Python package. A data scientist passes one table and names
a target column, and gets back a measurement of how predictable that target is.

In phase 1: one table, supervised learning, binary classification and
regression, a random or a time-ordered validation scheme, one headline number
with its uncertainty, and a complete account of every row and column.

Not in phase 1, and not to be built: joins or several tables, grouped
validation, multiclass, unsupervised methods, features made from dates or text,
an HTML report, any user interface, Parquet or Excel input, packaging for
`pip install`. pandas, numpy, scikit-learn and threadpoolctl are the only
third-party imports. No network access at run time.

The package lives at `E:\CODE\ml-professor\mlprofessor` and is run from the
repo root. It is never installed into the interpreter.

---

## 2. Interface (D2, D17)

### 2.1 The function

```python
mlprofessor.profile(data, target, task="auto", time_col=None, gap=0,
                    profile="full", seed=0, out_dir=None, n_jobs=-1) -> ProfileResult
```

| Argument | Meaning |
|---|---|
| `data` | A pandas DataFrame, or a path (`str` or `os.PathLike`) to a CSV file. A DataFrame is never modified. |
| `target` | Name of the target column. |
| `task` | `"auto"`, `"classification"` or `"regression"` (D4). |
| `time_col` | Name of the time column, or `None`. Never inferred (D7). When given, the time-ordered scheme is used; when `None`, the random scheme. |
| `gap` | A whole number of distinct timestamps, at least 0 (D7). A gap above 0 without `time_col` is an `input_error`. |
| `profile` | `"full"` or `"fast"` (D12). |
| `seed` | A whole number, at least 0. It drives the row cap, the split, the folds, every model and every resample. |
| `out_dir` | A folder, or `None`. When given it is created if missing and `profile.json` and `predictions.csv` are written into it. |
| `n_jobs` | `-1` for every core (`os.cpu_count()`), or a positive whole number (D17). |

Any other value of an argument raises `InputError`. Nothing is written when an
error is raised.

### 2.2 The command line

```
python -m mlprofessor profile --data FILE.csv --target COLUMN --out FOLDER
       [--task auto|classification|regression] [--time-col COLUMN] [--gap N]
       [--profile full|fast] [--seed N] [--n-jobs N]
```

It is a thin wrapper: it calls `profile(...)` with `out_dir=FOLDER` and prints a
short plain-text summary to standard output. Defaults are those of the function.
`python -m mlprofessor --help` exits 0.

| Exit code | When |
|---|---|
| 0 | The run finished and both files were written. |
| 2 | The command line could not be parsed (argparse's own behaviour). |
| 3 | A named error (2.4). Standard error carries one line `error: <code>: <message>`. No file is written. |

Any other failure is a defect and exits with the interpreter's own non-zero code.

### 2.3 `ProfileResult`

| Member | Type | Equals |
|---|---|---|
| `headline` | a mapping (`dict`) | `profile.json["headline"]` |
| `rows` | DataFrame, columns `item`, `count` | `profile.json["rows"]` |
| `columns` | DataFrame, one row per input column | `profile.json["columns"]` |
| `models` | DataFrame, one row per configuration | `profile.json["models"]` |
| `warnings` | DataFrame, columns `code`, `column`, `message` | `profile.json["warnings"]` |
| `predictions` | DataFrame, columns `row`, `y_true`, `y_pred` | `predictions.csv` |
| `to_dict()` | a new `dict` each call | the whole of `profile.json` |
| `to_json(path)` | writes a file | the same content as `profile.json` |

"Equals" for a DataFrame means: `frame.to_dict(orient="records")`, with NaN,
`None`, `pd.NA` and `NaT` read as `null` and numpy scalars read as plain Python
values, is the JSON list. The cells of `models.params` are dicts and the cells
of `models.fold_scores` are lists.

For a user there are two entry points: the function and the command line (D2).
For tests the contract also covers `mlprofessor.ProfileResult`,
`mlprofessor.errors`, and the two halves of the seam: `mlprofessor.extensions`
(2.6) and `mlprofessor.stages` (2.7). Everything else is internal and may be
reorganised.

### 2.4 Errors

Every named error is a subclass of `mlprofessor.errors.ProfileError`, which
subclasses `Exception`. Each class has a class attribute `code`, the text the
command line prints. The message says what was found and what the caller can do.

| Class | `code` | Raised when |
|---|---|---|
| `InputError` | `input_error` | An argument has an invalid value; `data` is neither a DataFrame nor a path; the file is missing, empty or not parseable as CSV; data rows have more fields than the header; `gap` above 0 without `time_col`; `time_col` equals `target`. |
| `ColumnNotFoundError` | `column_not_found` | `target` or `time_col` is not a column. |
| `DuplicateColumnsError` | `duplicate_columns` | Two columns have the same name (D22). |
| `TargetConstantError` | `target_constant` | The target has fewer than two distinct values among the usable rows. |
| `TaskAmbiguousError` | `task_ambiguous` | `task="auto"` and the rule cannot decide (D4). |
| `TaskUnsupportedError` | `task_unsupported` | A declared task the target does not support: classification with other than two distinct values, regression on a non-numeric target. |
| `TimeColumnError` | `time_column_invalid` | The time column is not in an accepted form (D18). |
| `TooFewRowsError` | `too_few_rows` | Fewer than 100 usable rows; fewer than 10 usable rows of a class; under time order, too few distinct timestamps to form the holdout and six blocks. |
| `FastProfileError` | `fast_profile_too_few_timestamps` | Under `profile="fast"` with a time column, the rows kept by the row cap hold too few distinct timestamps to form the holdout and six blocks (D20). The message tells the caller to use `profile="full"`. |
| `GapError` | `gap_too_large` | The gap leaves fewer than six training timestamps, or no training rows before a validation block. |
| `NoFeaturesError` | `no_features` | No column has the role feature after the leak screen. |
| `OneClassError` | `one_class` | Classification, and the holdout or the training rows hold one class, or no fold can be scored. |
| `AllModelsFailedError` | `all_models_failed` | Every configuration failed. The message carries each error text. |

An invalid argument is reported before anything is loaded. Beyond that, when a
table breaks several rules at once, which of the named errors is raised is open.

### 2.5 Warning codes

A warning is a record `{code, column, message}` in `profile.json["warnings"]`
and a row of `ProfileResult.warnings`. `column` is a column name or `null`.
The wording of `message` is open. Warnings never go to the Python `warnings`
module only; they are always in the output.

| `code` | `column` | Raised when |
|---|---|---|
| `suspected_leak` | the column | One per column excluded by the leak rule (D11). |
| `small_holdout` | `null` | Classification: fewer than 50 holdout rows of the rarer class. Regression: fewer than 50 holdout rows. |
| `date_column_undeclared` | the column | One per column excluded as `datetime` when `time_col` is `None` (D7). |
| `autocorrelated_panel_target` | the target | Regression under time order with repeated timestamps, and the lag-1 autocorrelation of the per-timestamp mean of the target over the training rows is above 0.5 (D10; section 10). |
| `model_failed` | `null` | One per configuration with status `failed`. |
| `values_coerced` | the column | One per numeric column in which text cells were turned into missing (D22). |
| `whole_number_regression_target` | the target | `task="auto"` picked regression and every target value is a whole number (D21). |
| `blank_header` | pandas' name for it | One per CSV column whose header cell is blank (D22). |
| `infinite_values` | the column | One per numeric feature holding infinite values. |
| `duplicate_rows` | `null` | The loaded table holds exact duplicate rows. They are kept. |
| `fold_not_scored` | `null` | Classification, and a fold's training or validation rows hold one class. |
| `no_interval` | `null` | No interval could be computed; `headline.interval_reason` says why. |

### 2.6 Seam for tests

A test must be able to prove which rows every fitted step received. The seam is
a context manager:

```python
from mlprofessor import extensions

with extensions.use(families=[...], steps=[...]):
    result = mlprofessor.profile(...)
```

- `families`, when not `None`, **replaces** the profile's model families for
  the duration of the block, under either profile. It is a list of mappings
  `{"name": str, "grid": [params, ...], "make": callable}`. List order is the
  tie-break order; `grid[0]` is the family's default. Configurations are named
  `<name>-<index>`. `make(task, params, seed, n_jobs)` returns a new unfitted
  estimator with `fit(X, y)` and, for classification, `predict_proba(X)`
  returning two columns in class order (column 1 is the positive class), or
  for regression `predict(X)`.
- `steps`, when not `None`, is a list of zero-argument callables. Each returns
  a new object with `fit(X, y)` and `transform(X)`. The steps run first in the
  preprocessing chain, in list order, before the built-in imputation, encoding
  and scaling. `transform` must return a DataFrame with the same index and
  columns.
- A fresh step object and a fresh estimator are made for **every fit**: once
  per scored fold per configuration for estimators, once per scored fold and
  once for the final fit for steps.
- Every `X` handed to a step or to a seam-supplied estimator is a pandas
  DataFrame whose **index is the rows' zero-based positions in the loaded
  source** (the same numbers as `predictions.row`). `y` is a Series with the
  same index: 0 and 1 for classification (1 is the positive class), floats for
  regression. A step receives the feature columns as loaded and typed (numeric
  columns as floats with NaN for missing, categorical columns as text with
  `"(missing)"` for missing). An estimator receives the fully preprocessed
  matrix.
- The estimator's `fit` is called inside the tool's thread limits, so
  `threadpoolctl.threadpool_info()` called there reports them.
- When the block ends the previous state is restored, including after an
  exception. Blocks may be nested.

With this a recording fake proves: no `fit` of any kind ever receives a holdout
row; inside each fold no `fit` receives a row of that fold's validation block;
only the chosen configuration predicts on the holdout; linear algebra is on one
thread.

### 2.7 Step functions

The data-handling steps are built and verified before any model exists, so each
is also callable on its own through one module, `mlprofessor.stages`. It is a
thin facade: where the code lives is open, the names and behaviour below are
pinned, and `profile()` must use the same code (a test may compare the two).
Section 3 defines what each step does; this table defines only the call.

| Call | Returns | Raises |
|---|---|---|
| `load_table(data)` | `(frame, info)`. `frame` is the loaded table (step 1, section 6) with a default index 0..n−1, the rows' source positions, and string column names. `info` is `{"kind", "source", "warnings"}` as in `input.kind`, `input.source` and a list of warning records (only `blank_header` arises here). | `InputError`, `DuplicateColumnsError` |
| `usable_rows(frame, target, time_col=None)` | Step 2 as a count: `{"usable", "dropped_missing_target", "dropped_missing_time"}`. `usable` is an array of the positions of the usable rows in increasing order; the two counts and its length add up to the rows of `frame`. A missing time value is a missing cell (or an infinite number); the time column's form is not checked here. | `ColumnNotFoundError` |
| `column_roles(frame, target, time_col=None)` | A DataFrame with the eleven columns of `profile.json["columns"]` (5.2), one row per column of `frame` in order, as decided by step 5 alone: `leak_rule` is null and `n_imputed` is 0 throughout, since the leak screen and imputation come later. Usable rows are those with a target and, when `time_col` is given, a time value. | `ColumnNotFoundError` |
| `decide_task(values, task="auto")` | `{"type", "decided_by", "n_distinct", "classes", "positive_class", "warnings"}` for the target values given (missing ones are ignored). `warnings` is a list of warning codes: `["whole_number_regression_target"]` or `[]`. | `TargetConstantError`, `TaskAmbiguousError`, `TaskUnsupportedError`, `InputError` |
| `read_time(values)` | `(keys, form)`. `keys` is a numpy array, one per value, that sorts in time order: `datetime64[ns]` with `NaT` for a missing value under the `datetime` and `iso8601` forms (in UTC if the values carried a zone), `float64` with NaN for a missing or infinite value under the `numeric` form. `form` is as in `validation.time_form`. | `TimeColumnError` |
| `split_random(n_rows, classes=None, seed=0)` | `{"training", "holdout", "folds"}`: arrays of positions in 0..`n_rows`−1, each in increasing order; `folds` is a list of five `{"training", "validation"}`. `classes`, when given, is an array of `n_rows` labels with two distinct values and makes the split stratified (step 7). | |
| `split_time(keys, gap=0)` | `{"training", "gap", "holdout", "folds"}`: arrays of positions into `keys`, each in increasing order; `folds` is a list of five `{"training", "gap", "validation"}`, fold *k* validating on block *k*+1. `keys` is as returned by `read_time`, in any order, with no missing value. | `TooFewRowsError`, `GapError` |
| `roc_auc(y, score)` | ROC AUC of `score` for labels `y` of 0 and 1 (1 is positive); ties share their average rank. | `OneClassError` |
| `mse(y, pred)` | Mean squared error. | |
| `skill(metric, score, baseline_score)` | For `"roc_auc"`: (score − baseline) ÷ (1 − baseline), which is 2 × AUC − 1 at the baseline 0.5. For `"mse"`: 1 − score ÷ baseline. | `ValueError` for any other metric |
| `previous_value(y, keys, gap=0)` | An array aligned with `y`: for each row the target of the row `gap + 1` timestamps earlier, NaN where there is none. | `ValueError` if `keys` are not unique |
| `Preprocessor(kinds)` | The built-in preprocessing of step 8 for one fit. `kinds` maps a column name to `"numeric"` or `"categorical"`. `fit(frame)` learns from the rows given and returns the object. `transform(frame)` returns a DataFrame with the same index: one column per numeric column, under its own name, then for each categorical column one column per kept level named `<column>=<level>`, levels in sorted order, holding 0.0 and 1.0. A missing categorical cell is the level `(missing)`. | |

`split_time` raises `TooFewRowsError` when there are too few timestamps;
`profile()` turns that into `FastProfileError` when the fast profile's row cap
was what left too few (D20).

How `profile()` uses the two split functions is pinned, so that a test can
predict a run's rows. Take the usable rows that remain after the row cap, in
file order, and call them 0 to *n*−1. Under the random scheme the run's holdout
rows and folds are `split_random(n, classes, seed)` applied to them, `classes`
being their target values for classification and `None` for regression. Under
the time-ordered scheme they are `split_time(keys, gap)`, `keys` being those
rows' keys from `read_time`.

---

## 3. Methodology, in the order it happens

Every numbered step below is in the order of the brief. **From step 8 on, every
fitted step sees training rows only.** A reader should be able to decide from
this section alone whether an implementation leaks.

**1. Load the table (D3, D22); count the rows.** Section 6 gives the loading
rules. `loaded` is the number of rows of the table: for a CSV, the data lines,
not counting blank lines.

**2. Drop rows with no target, then rows with no time value; count each.**
A target is missing when the cell is missing (NaN, `None`, `pd.NA`, `NaT`) or,
for a numeric target, infinite. Under time order a row with a target and a
missing (or infinite) time value is dropped next. A row missing both counts
under the target. The rows left are the **usable rows**. Fewer than 100 usable
rows: `too_few_rows`.

**3. Under time order, read the time column in its accepted form (D18) and put
the usable rows in time order.** Accepted forms:

| Form | `validation.time_form` | Ordered |
|---|---|---|
| a datetime-typed DataFrame column, as it is (a time zone is allowed) | `datetime` | as instants |
| a numeric column (not boolean) | `numeric` | as numbers |
| text, every non-missing value matching ISO 8601 with the year first: `YYYY-MM-DD`, optionally followed by `T` or a space and `HH:MM`, `HH:MM:SS` or `HH:MM:SS.fff`, optionally followed by `Z` or `+HH:MM` / `-HH:MM` | `iso8601` | as instants |

Anything else raises `time_column_invalid`: a boolean column, text in any other
form (`25/01/2024`, `01/12/2023`, `5-Jan-2024`, `2024-01`), a text column that
mixes values with and without an offset, an impossible date (`2024-02-30`), an
object column of mixed types. The tool never guesses between day-first and
month-first. Rows with equal timestamps keep their file order.

**4. Decide the task (D4, D20, D21),** on the usable rows, before any row cap.
Let *k* be the number of distinct target values.

- `task="auto"`: *k* = 2 is classification, whatever the type. A numeric target
  (integer or float type, not boolean) with *k* > 20 is regression. Anything
  else raises `task_ambiguous`.
- `task="classification"` is honoured when *k* = 2, else `task_unsupported`.
- `task="regression"` is honoured when the target is numeric, else
  `task_unsupported`.
- *k* < 2 raises `target_constant` in every case.
- When the rule picks regression and every target value is a whole number, the
  warning `whole_number_regression_target` is added.

For classification the two classes are put in sorted order (by value; by their
text if they cannot be compared). The second is the **positive class**: `1` of
`0`/`1`, `"yes"` of `"no"`/`"yes"`, `True` of `False`/`True`. `y_pred` is the
predicted probability of the positive class.

**5. Assign the column roles that need no target (D3, D19, D20),** on the
usable rows, in file order, before any row cap. The target has role `target`
and the declared time column role `time`. For every other column, the first
rule that applies decides:

| # | Rule | Role, reason |
|---|---|---|
| 1 | every value is missing | excluded, `all_missing` |
| 2 | datetime type | excluded, `datetime` |
| 3 | boolean, integer or float type (nullable or not) | numeric: go to rule 8 |
| 4 | text, and at least 95% of the non-missing values match a date form of section 4.5 | excluded, `datetime` |
| 5 | text, and at least 90% of the non-missing values parse as numbers | numeric with coercion: go to rule 8 |
| 6 | text, one distinct value | excluded, `constant` |
| 6a | text, distinct values number more than 95% of the usable rows | excluded, `id_like` |
| 6b | text, distinct values number more than 50% of the non-missing values | excluded, `free_text` |
| 6c | any other text | feature, kind `categorical` |
| 7 | any other type (mixed objects, complex numbers, durations, containers) | excluded, `unsupported_type` |
| 8 | numeric, infinite values read as missing: no value left | excluded, `all_missing` |
| 8a | numeric, one distinct value | excluded, `constant` |
| 8b | numeric, every value a whole number, every value distinct, and the values consecutive or strictly ordered in file order (D19) | excluded, `id_like` |
| 8c | any other numeric | feature, kind `numeric` |

"Text" is a pandas string column or an object column in which every non-missing
value is a `str`; a category-typed column is treated as text through the string
form of its values. Text is compared after stripping surrounding white space.
"Parse as numbers" means `pandas.to_numeric(..., errors="coerce")` gives a
number. In rule 8b, judged on the non-missing values: "consecutive" means
largest minus smallest plus one equals their count; "strictly ordered" means
strictly increasing, or strictly decreasing, down the file.

Counts are kept per column over **every loaded row** (section 5.3): `n_missing`,
`n_coerced` (rule 5: non-missing text cells that did not parse, which become
missing, with a `values_coerced` warning), `n_infinite` (which become missing,
with an `infinite_values` warning).

**6. Apply the fast profile's row cap (D12, D20); count the rows left out.**
Only under `profile="fast"` and only when there are more than 2,000 usable rows.

- No time column: a simple random sample of 2,000 usable rows, without
  replacement, drawn from the seed.
- Time column: the most recent whole timestamps whose rows together number at
  most 2,000. If the most recent timestamp alone holds more than 2,000 rows:
  `fast_profile_too_few_timestamps`.

The rows left out are counted as `dropped_row_cap`. After the cap: fewer than
100 rows is `too_few_rows`; for classification, fewer than 10 rows of either
class is `too_few_rows`.

**7. Split off the holdout (D6, D7), applying the gap.**

*No time column (random scheme).* With *n* rows, the holdout is ⌈*n*/4⌉ rows
drawn at random from the seed; for classification it is stratified, so that
each class's holdout count is within one of a quarter of the class. The other
rows are the training rows. Five folds partition the training rows, shuffled
from the seed, stratified for classification; fold sizes differ by at most one.

*Time column (time-ordered scheme).* Let the distinct timestamps in order be
*t*₁ < … < *t*ₘ.

- The **holdout** is the latest whole timestamps that together hold at least
  25% of the rows, the smallest such set: add timestamps from the latest
  backwards until 4 × (rows held) ≥ *n*. Rows sharing a timestamp are never
  separated.
- The **gap** removes the `gap` distinct timestamps immediately before the
  holdout from the training side. Their rows are counted as `dropped_gap`.
- The remaining earlier timestamps are the **training timestamps**. Fewer than
  six before the gap is taken: `too_few_rows`, or under the fast profile's cap
  `fast_profile_too_few_timestamps`. Fewer than six after: `gap_too_large`.
- The training timestamps are cut into **six contiguous blocks** of timestamps,
  as equal as possible, the earlier blocks taking the extra ones (the cut of
  `numpy.array_split`). **Fold *k*** (1 to 5) validates on block *k*+1 and
  trains on blocks 1 to *k*, less the `gap` distinct timestamps immediately
  before block *k*+1. Those rows are reported per fold as `gap_rows`; they stay
  training rows of the final fit. No training timestamp left before a
  validation block: `gap_too_large`.

`validation.holdout_share` is holdout rows ÷ (training + holdout + gap rows).

For classification, a holdout or a set of training rows with one class raises
`one_class`. A fold whose training or validation rows hold one class is not
scored for any configuration (`fold_not_scored`); if no fold can be scored,
`one_class`.

**8. The leak screen (D11), on training rows only.** For each feature column,
in file order:

1. *Filled in for one class and empty for the other* (classification only): on
   the training rows, the column is missing for more than 95% of one class and
   less than 5% of the other. Leak rule `missing_by_class`.
2. Otherwise, *on its own it predicts the target almost perfectly*: the
   column's single-column cross-validated skill is at least 0.98. Leak rule
   `predicts_target`.

Single-column skill is measured with the **selection folds of step 7** (so
under time order it respects time, and no holdout row is involved): in each
scored fold a decision tree (`max_depth=6`, `min_samples_leaf=5`,
`random_state=seed`) is fitted on that one column of the fold's training rows
and predicts its validation rows. A numeric column is used as it is, missing
values left missing; a categorical column as whole-number codes of the sorted
levels of the fold's training rows, with −1 for an unseen level. The
predictions of all folds are pooled and skill is computed as in step 11:
2 × AUC − 1, or 1 − (squared error ÷ baseline squared error), the baseline
being each fold's training mean and, under time order with unique timestamps,
whichever of that and the previous value has the lower error.

A flagged column gets role `excluded`, reason `suspected_leak`, its
`leak_rule`, and a `suspected_leak` warning. It reaches no model. If no feature
is left: `no_features`.

**Preprocessing, fitted on the rows of each fit only (D3).** One preprocessing
is fitted for each fit, on that fit's training rows, and applied to the rows it
predicts:

- numeric features: missing values (including coerced and infinite cells) are
  replaced by the median of the fit's training rows (0 if the column is all
  missing there), then standardised with the mean and standard deviation of
  the fit's training rows (a zero deviation is read as 1);
- categorical features: a missing value is the level `"(missing)"`; one-hot
  columns for the levels seen in the fit's training rows, at most the 30 most
  frequent (ties by level text); any other level is all zeros.

The same preprocessing serves every model family. `n_imputed` of a feature is
the number of its missing cells among the training and holdout rows.

**9. Select by five folds inside the training rows (D8, D16).** For each
configuration of each family (section 4.1), in family order then grid order,
and for each scored fold: fit the preprocessing and the model on the fold's
training rows, predict its validation rows, and score them: ROC AUC for
classification, mean squared error for regression. `cv_mean` is the mean over
scored folds and `cv_std` their sample standard deviation.

A configuration whose fit or prediction raises is recorded with status
`failed`, the error text and a `model_failed` warning; it is never dropped
from the table and never chosen. All failing: `all_models_failed`.

The **chosen** configuration has the best `cv_mean`: the highest ROC AUC, the
lowest squared error. Ties go to the earlier family, then the earlier grid
position.

**10. Refit the chosen configuration on all training rows,** with its
preprocessing fitted on all training rows.

**11. Score it once on the holdout; score the baselines on the holdout (D5,
D10).** Only the chosen configuration ever predicts the holdout.

- Classification: `score` is ROC AUC of the predicted probabilities (ties share
  their average rank). The baseline is `chance`, 0.5. **skill = 2 × AUC − 1.**
- Regression: `score` is mean squared error. The baseline `training_mean`
  predicts the mean target of the training rows for every holdout row.
  **skill = 1 − score ÷ baseline score.**
- Regression under time order with **unique timestamps**: a second baseline,
  `previous_value`, predicts for each holdout row the target of the row
  `gap + 1` timestamps earlier in the time-ordered usable rows (after the row
  cap; rows inside the gap count as earlier rows and their targets are used).
  Both are scored and reported. The headline baseline is the one with the lower
  squared error on the holdout; a tie goes to `training_mean`.
- Regression under time order with **repeated timestamps**: the baseline is
  `training_mean`, with the warning `autocorrelated_panel_target` when the
  lag-1 autocorrelation of the per-timestamp mean of the target, over the
  training timestamps in order, is above 0.5.

Accuracy is never the headline. Skill may be negative and is reported as
measured. A dummy model that predicts the baseline is the reference only; it is
never a candidate and never appears in the models table.

**12. Compute the interval and the p-value from the holdout predictions (D9).**
Nothing is refitted. *R* is 1,000 under `full` and 200 under `fast`.

- *Random scheme, interval:* *R* resamples of the holdout rows with
  replacement. For each, the score and the skill (for regression the baseline's
  predictions are resampled with the rows). The 95% interval is the 2.5th and
  97.5th percentiles (linear interpolation), reported for the score and for the
  skill. Method `bootstrap_rows`.
- *Random scheme, p-value:* *R* shuffles of the holdout targets against the
  fixed predictions. **p = (1 + number of shuffles whose skill is at least the
  observed skill) ÷ (1 + *R*).**
- *Time-ordered scheme, interval:* a moving-block bootstrap over the distinct
  holdout timestamps in order. With *T* timestamps, the block length is *L* =
  the smallest whole number with *L*³ ≥ *T*. Each resample draws ⌈*T*/*L*⌉
  blocks of *L* consecutive timestamps, each starting at a position drawn
  uniformly from those that fit, joins them, keeps the first *T* timestamps,
  and takes every row of each drawn timestamp. Percentiles as above. Method
  `bootstrap_blocks`, with `block_length`. With fewer than 30 distinct holdout
  timestamps there is no interval: the fields are `null`,
  `interval_reason` says why, and the warning `no_interval` is added.
- *Time-ordered scheme, p-value:* absent (`null`), and `p_value_reason` states
  that shuffling assumes independent rows.
- A resample in which the score cannot be computed (one class) is left out and
  the number used is reported. If fewer than half can be used there is no
  interval, with a reason and the `no_interval` warning.

Also added here: `small_holdout` (2.5).

**13. Assemble the result; write the files** (section 5).

### Seeds and threads (D17)

One `seed` drives everything random. The same arguments on the same data give
the same `profile.json` and `predictions.csv`, apart from `meta.timing`, and
apart from `meta.n_jobs` when `n_jobs` differs.

For the whole run the tool uses `threadpoolctl` to hold linear algebra (BLAS)
to **one thread always** and OpenMP to `n_jobs` threads. Forests are fitted
with `n_jobs` and **predict on one thread**. Both are needed: section 9 gives
the measurements. How the generator is seeded for each purpose is open.

---

## 4. Pinned values

### 4.1 Model families and grids (D8, D16)

Family order, which is also the tie-break order: `linear`, then `forest`, then
`boosting` under `full`; `linear`, then `tree` under `fast`. Configuration
names are `<family>-<index>`; index 0 is the family's default.

**`profile="fast"`**: two configurations, no grid: `linear-0` and `tree-0`.

**`profile="full"`**: 18 configurations, 90 cross-validation fits (the limit is
8 per family, 24 and 120).

| Family | scikit-learn estimator | Fixed | Grid (index 0 first) |
|---|---|---|---|
| `linear`, classification | `LogisticRegression` | `max_iter=1000` | `C`: 1.0, 0.001, 0.01, 0.1, 10.0, 100.0 |
| `linear`, regression | `Ridge` | | `alpha`: 1.0, 0.01, 0.1, 10.0, 100.0, 1000.0 |
| `tree` (fast only) | `DecisionTreeClassifier` / `Regressor` | `random_state=seed`; depth unlimited | none |
| `forest`, classification | `RandomForestClassifier` | `n_estimators=100`, `random_state=seed`, `n_jobs=n_jobs` | (`min_samples_leaf`, `max_features`): (1, "sqrt"), (1, 1/3), (5, "sqrt"), (5, 1/3), (20, "sqrt"), (20, 1/3) |
| `forest`, regression | `RandomForestRegressor` | as above | (1, 1/3), (1, "sqrt"), (5, 1/3), (5, "sqrt"), (20, 1/3), (20, "sqrt") |
| `boosting` | `HistGradientBoostingClassifier` / `Regressor` | `early_stopping=False`, `random_state=seed` | (`max_leaf_nodes`, `learning_rate`, `max_iter`): (31, 0.1, 100), (31, 0.05, 300), (7, 0.1, 100), (7, 0.05, 300), (63, 0.1, 100), (63, 0.05, 300) |

Every argument not listed is scikit-learn's default. The forest is at full
strength: every tree draws on all training rows (no `max_samples`). The
regression forest's default considers one third of the features at each split,
the only departure from scikit-learn's defaults (D16). `params` in the models
table holds exactly the grid arguments (`{}` for `tree-0`).

### 4.2 Sizes and counts

| Value | Pinned at |
|---|---|
| holdout share | 25%: ⌈*n*/4⌉ rows (random); the latest timestamps holding at least 25% (time) |
| folds | 5; six blocks under time order |
| fast profile row cap | 2,000 |
| resamples and shuffles | 1,000 (`full`), 200 (`fast`) |
| interval level | 95%, percentile |
| too small to profile | fewer than 100 usable rows; fewer than 10 of a class |
| one-hot levels kept per column | 30 |

### 4.3 Cut-offs (measurements in section 10)

| Cut-off | Value |
|---|---|
| leak: single-column cross-validated skill | at least 0.98 |
| leak: missing by class | more than 95% of one class and less than 5% of the other |
| id-like, text column | distinct values more than 95% of the usable rows |
| id-like, whole-number column | every value distinct, and consecutive or strictly ordered in file order |
| free text | distinct values more than 50% of the non-missing values |
| numeric coercion | at least 90% of the non-missing values parse as numbers |
| date-like | datetime type, or at least 95% of the non-missing values match a form of 4.5 |
| small holdout warning | fewer than 50 holdout rows of the rarer class (classification) or in all (regression) |
| no block-bootstrap interval | fewer than 30 distinct holdout timestamps |
| autocorrelated panel target warning | lag-1 autocorrelation above 0.5 |

### 4.4 Missing-value spellings

On load, pandas' default missing-value tokens count as missing and never as
coerced (D22): the empty cell, `n/a`, `N/A`, `NA`, `NaN`, `nan`, `-NaN`,
`-nan`, `null`, `NULL`, `None`, `#N/A`, `#N/A N/A`, `#NA`, `<NA>`, `1.#IND`,
`-1.#IND`, `1.#QNAN`, `-1.#QNAN`. The tool passes no `na_values` and no
`keep_default_na`.

### 4.5 Date forms

A text value is date-like when, after stripping, the whole of it matches one of
these, case-insensitively. `Y` is four digits; `M` is 1 to 12 and `D` is 1 to
31, each with or without a leading zero; `Mon` is an English month name, full
or abbreviated (`Jan`, `January`, `Sep`, `Sept`); each may be followed by a
time: `T` or a space, `H:MM` or `HH:MM`, optional `:SS`, optional fraction,
optional `Z` or offset.

`Y-M-D`, `Y/M/D`, `D/M/Y`, `M/D/Y`, `D-M-Y`, `M-D-Y`, `D.M.Y`,
`D-Mon-Y` or `D Mon Y`, `Mon D, Y` or `Mon D Y`.

Never pandas' lenient date parser, which accepts `1,200`, `May`, `09:30`,
`2024Q1` and `1/2` as dates. This list decides only whether a column is
*excluded as a date*. It is never used to order rows: the time column has its
own, stricter rule (step 3).

---

## 5. Output

### 5.1 `predictions.csv` (D14)

Comma-separated, UTF-8, a header line `row,y_true,y_pred`, one line per holdout
row, in increasing `row`.

| Column | Meaning |
|---|---|
| `row` | Zero-based position of the row in the loaded source, counted before any row is dropped. For a CSV, blank lines are not counted. For a DataFrame it is the position, not the index label. |
| `y_true` | The target of that row, as it is in the source. |
| `y_pred` | Classification: predicted probability of the positive class. Regression: the predicted value. Always finite. |

### 5.2 `profile.json`

UTF-8 JSON. NaN and infinity never appear: a value that cannot be computed is
`null` and something nearby says why. Top-level keys, all always present:
`meta`, `input`, `rows`, `columns`, `task`, `validation`, `headline`,
`baselines`, `cv`, `models`, `warnings`, `limitations`.

**`meta`**

| Key | Type | Meaning |
|---|---|---|
| `tool` | string | `"mlprofessor"` |
| `schema_version` | integer | `1` |
| `profile` | string | `"full"` or `"fast"` |
| `seed` | integer | the seed used |
| `n_jobs` | integer | the number of cores used: the argument, or `os.cpu_count()` for `-1` |
| `versions` | object | `python`, `pandas`, `numpy`, `scikit-learn`, `threadpoolctl`: version strings |
| `timing` | object | at least `seconds` (number). **The only place a clock value may appear.** |

**`input`**

| Key | Type | Meaning |
|---|---|---|
| `source` | string | the path exactly as given, or `"<DataFrame>"` |
| `kind` | string | `"csv"` or `"dataframe"` |
| `target` | string | |
| `time_col` | string or null | |
| `task` | string | the task as requested: `auto`, `classification` or `regression` |
| `gap` | integer | |
| `n_rows` | integer | rows loaded |
| `n_columns` | integer | columns loaded |
| `duplicate_rows` | integer | rows that exactly repeat an earlier row; they are kept |

**`rows`**: a list of exactly these seven records `{item, count}`, in this
order: `loaded`, `dropped_missing_target`, `dropped_missing_time`,
`dropped_gap`, `dropped_row_cap`, `training`, `holdout`. Always:
`loaded` = the four `dropped_` counts + `training` + `holdout`.

**`columns`**: one record per input column, in file order.

| Key | Type | Meaning |
|---|---|---|
| `name` | string | |
| `position` | integer | zero-based column position |
| `dtype` | string | the pandas type as loaded (informative) |
| `role` | string | `target`, `time`, `feature` or `excluded` |
| `kind` | string or null | for a feature, `numeric` or `categorical` (pinned); for other roles informative |
| `reason` | string or null | for an excluded column: `constant`, `id_like`, `datetime`, `free_text`, `suspected_leak`, `all_missing` or `unsupported_type`; otherwise null |
| `leak_rule` | string or null | for a suspected leak: `predicts_target` or `missing_by_class`; otherwise null |
| `n_missing` | integer | missing cells among all loaded rows, as loaded |
| `n_coerced` | integer | non-missing text cells of a numeric column that did not parse, among all loaded rows |
| `n_infinite` | integer | infinite cells among all loaded rows |
| `n_imputed` | integer | for a feature, its missing cells (as loaded, coerced or infinite) among the training and holdout rows; otherwise 0 |

**`task`**

| Key | Type | Meaning |
|---|---|---|
| `type` | string | `classification` or `regression` |
| `decided_by` | string | `rule` (from `task="auto"`) or `declared` |
| `n_distinct` | integer | distinct target values among the usable rows: the evidence |
| `target_dtype` | string | pandas type of the target as loaded |
| `classes` | list or null | the two classes in sorted order; null for regression |
| `positive_class` | value or null | the second of `classes` |

**`validation`**

| Key | Type | Meaning |
|---|---|---|
| `scheme` | string | `random` or `time_ordered` |
| `holdout_share` | number | holdout ÷ (training + holdout + gap rows) |
| `n_folds` | integer | 5 |
| `stratified` | boolean | true for classification under the random scheme |
| `gap` | integer | |
| `time_col` | string or null | |
| `time_form` | string or null | `datetime`, `numeric` or `iso8601` |
| `train_end` | string, number or null | the last training timestamp |
| `holdout_start` | string, number or null | the first holdout timestamp |
| `holdout_end` | string, number or null | the last holdout timestamp |
| `n_train_timestamps` | integer or null | distinct training timestamps |
| `n_holdout_timestamps` | integer or null | distinct holdout timestamps |
| `timestamps_unique` | boolean or null | whether every timestamp holds one row |
| `folds` | list | five records `{fold, training_rows, validation_rows, gap_rows, scored}`, `fold` 1 to 5 |
| `caveat` | string or null | under the random scheme, the fixed caveat (5.4); null under time order |

The seven time fields are null under the random scheme. A timestamp is written
as `YYYY-MM-DDTHH:MM:SS` (with a fraction if it has one; in UTC if the column
carried a zone) for the `datetime` and `iso8601` forms, and as a number for the
`numeric` form.

**`headline`** (also `ProfileResult.headline`)

| Key | Type | Meaning |
|---|---|---|
| `metric` | string | `roc_auc` or `mse` |
| `score` | number | the chosen configuration's score on the holdout |
| `model` | string | the chosen configuration, as in `models.config` |
| `baseline` | string | `chance`, `training_mean` or `previous_value` |
| `baseline_score` | number | 0.5, or the baseline's squared error on the holdout |
| `skill` | number | step 11 |
| `interval_method` | string or null | `bootstrap_rows`, `bootstrap_blocks`, or null when there is no interval |
| `interval_level` | number | 0.95 |
| `interval_low`, `interval_high` | number or null | the interval on `score` |
| `skill_low`, `skill_high` | number or null | the interval on `skill` |
| `n_resamples` | integer | 1,000 or 200 |
| `n_resamples_used` | integer | resamples in which the score could be computed |
| `block_length` | integer or null | under `bootstrap_blocks` |
| `interval_reason` | string or null | why there is no interval |
| `p_value` | number or null | step 12 |
| `n_permutations` | integer or null | shuffles used; null when there is no p-value |
| `p_value_reason` | string or null | why there is no p-value |

**`baselines`**: a list of records `{name, score, used}`. Classification: one,
`chance`. Regression: `training_mean`, and under time order with unique
timestamps also `previous_value`. Exactly one has `used` true and it is the
headline baseline.

**`cv`**

| Key | Type | Meaning |
|---|---|---|
| `metric` | string | `roc_auc` or `mse` |
| `greater_is_better` | boolean | true for `roc_auc` |
| `n_folds` | integer | 5 |
| `n_scored_folds` | integer | |
| `n_configurations` | integer | rows of the models table |
| `chosen` | string | the chosen configuration |
| `chosen_mean`, `chosen_std` | number, number or null | its mean and spread across folds |

**`models`**: one record per configuration, in family order then grid order.

| Key | Type | Meaning |
|---|---|---|
| `config` | string | `<family>-<index>` |
| `family` | string | |
| `params` | object | the grid arguments of this configuration |
| `status` | string | `fitted` or `failed` |
| `error` | string or null | the error text of a failed configuration |
| `fold_scores` | list | five entries, a number or null (fold not scored, or failed) |
| `cv_mean` | number or null | mean of the scored folds |
| `cv_std` | number or null | their sample standard deviation; null with fewer than two |
| `chosen` | boolean | true for exactly one, which has status `fitted` |

**`warnings`**: a list of records `{code, column, message}` (2.5). May be empty.

**`limitations`**: the list of seven sentences of 5.4, exactly, in order.

### 5.3 Reconciliation rules

1. `rows.loaded` equals the rows of the source; it equals the four dropped
   counts plus training plus holdout.
2. Every source column appears exactly once in `columns`.
3. `predictions.csv` has one row per holdout row, and each `y_true` equals the
   target of that row in the source.
4. `headline.score` equals the metric recomputed from `predictions.csv`;
   `headline.skill` follows from it and the baseline as in step 11.
5. Every part of `ProfileResult` equals the matching part of `profile.json`.
6. The same seed gives an identical `profile.json` and `predictions.csv`, apart
   from `meta.timing` (and `meta.n_jobs` when `n_jobs` differs).

### 5.4 Fixed texts

The random-split caveat (`validation.caveat`), exactly:

> Random split: this number is optimistic if the rows are ordered in time or one entity appears on many rows.

The limitations statement (`limitations`), exactly these seven strings:

1. `Predictability here means how well these model families did on this data under this validation scheme.`
2. `Low skill is not proof that the target cannot be predicted.`
3. `High skill is not proof of cause.`
4. `A random split is optimistic when rows are ordered in time or one entity appears on many rows.`
5. `Look-ahead inside the caller's own features cannot be detected.`
6. `Classification skill and regression skill are on different scales and are not comparable.`
7. `A CSV file is read with pandas' types, so text identifiers can lose leading zeros; pass a DataFrame to control the types.`

---

## 6. Loading rules and types (D3, D22)

**A CSV path** is read with `pandas.read_csv(path, low_memory=False)` and no
other argument. Consequences, all measured on pandas 3.1.1 (section 9):

- Blank lines are not rows. A line of only commas is a row of missing values.
- The missing-value spellings of 4.4 load as missing. The tool cannot see what
  was written, so they are counted in `n_missing`, never in `n_coerced`.
- A cell such as `1,200` or `abc` in an otherwise numeric column makes the whole
  column load as text. Rule 5 of step 5 then reads it as numeric and counts
  those cells as coerced.
- **Duplicate column names stop the run** with `duplicate_columns`. pandas
  renames them silently (`a`, `a.1`), so the tool reads the header line itself
  (the first non-blank line, UTF-8, an optional byte-order mark ignored) and
  compares the names before loading.
- A blank header cell is kept under pandas' name (`Unnamed: 3`) with a
  `blank_header` warning.
- Data rows with more fields than the header are an `input_error` (pandas would
  otherwise turn the first column into an index and shift the names).

**A DataFrame** is used as it is. Its index is ignored. Column names are read
through `str()`; duplicates after that stop the run. Types a CSV cannot carry:

| DataFrame type | Treated as |
|---|---|
| nullable integer or float (`Int64`, `Float64`) | numeric; `pd.NA` is missing |
| boolean, nullable boolean | numeric 0 and 1 as a feature; a two-valued target is classification with `True` positive |
| category | text, through the string form of its values |
| datetime, with or without a zone | excluded as `datetime`; or the time column, as it is |
| duration, complex, mixed objects, containers | excluded as `unsupported_type` |

A DataFrame and a CSV path give the same result for the types a CSV can carry:
integers and floats without missing values, floats with missing values, and
text. `input.source` and `input.kind` differ.

**Stated as a limitation** (sentence 7 of 5.4): a CSV is read with pandas'
types, so a text identifier such as `000123` loads as the number 123. A caller
who needs control passes a DataFrame.

---

## 7. Edges

None is skipped silently: each raises a named error or is reported in the output.

| Edge | Outcome |
|---|---|
| the target column is absent | `column_not_found` |
| the target has one distinct value | `target_constant` |
| a declared task the target does not support | `task_unsupported` |
| an ambiguous target under `task="auto"` | `task_ambiguous` |
| no usable feature column | `no_features` |
| too few rows to split (fewer than 100 usable, or fewer than 10 of a class) | `too_few_rows` |
| a holdout, or the training rows, with one class | `one_class` |
| a fold with one class | that fold is not scored; `fold_not_scored` warning; `validation.folds[].scored` false; `fold_scores` null there. No fold scored: `one_class` |
| a time column not in an accepted form | `time_column_invalid` |
| a gap that leaves no training rows | `gap_too_large` |
| too few timestamps for the holdout and six blocks | `too_few_rows` |
| the fast profile on a panel too wide for its row cap | `fast_profile_too_few_timestamps` |
| duplicate column names | `duplicate_columns` |
| a blank header | kept under pandas' name; `blank_header` warning |
| too few holdout timestamps for the block bootstrap (fewer than 30) | interval fields null; `interval_reason`; `no_interval` warning |
| a model configuration that raises | status `failed`, error text kept, `model_failed` warning |
| every configuration failing | `all_models_failed` |
| infinite values in a feature | read as missing, counted in `n_infinite`, `infinite_values` warning |
| exact duplicate rows | kept; `input.duplicate_rows`; `duplicate_rows` warning |
| an unknown column type | excluded with reason `unsupported_type` |
| a date column present with no `time_col` | random scheme; `date_column_undeclared` warning per column |
| a whole-number target read as regression by the rule | runs; `whole_number_regression_target` warning |
| a row missing a time value | dropped and counted as `dropped_missing_time` |
| a feature all missing within one fit's training rows | imputed with 0 in that fit |
| a categorical level unseen in a fit's training rows | encoded as all zeros |

---

## 8. Failure modes (D13)

Each keeps its id through the gate, the tests and the review.

| Id | The silent failure | Judged by |
|---|---|---|
| F1 | A fitted step sees the holdout: the model refitted on all rows before scoring (large), or an imputer, encoder, scaler or the leak screen fitted before the split (small). | Gate: `model.noise_clf`, `model.noise_reg`, `model.noise_p_value` (large form). Tests: recording fakes through the seam (both forms, exactly). |
| F2 | A column that copies the target, or is only filled in after the outcome, is used as a feature. | Gate: `leak.excluded`, `leak.regression`. Tests: the leak rule on crafted columns. |
| F3 | The headline is the best cross-validation score, not the holdout score. | Gate: `model.headline_recomputed`, `model.selection_full`. Tests: a fake family whose fold scores and holdout score differ by construction. |
| F4 | Accuracy is the headline on a rare class. | Gate: `model.metric_clf`, `model.headline_recomputed`. Tests: metrics by hand. |
| F5 | The task is misread from the column's type. | Gate: `task.binary_numeric`, `task.ambiguous_stops`. Tests: the rule on every loaded shape. |
| F6 | Something is lost without being reported: rows, a column, a model family. | Gate: `data.rows_reconcile`, `data.columns_accounted`, `data.coercion_reported`, `models.all_accounted`. Tests: accounting on crafted tables; a family that raises. |
| F7 | Time order is ignored: a shuffled split, a training row at or after the holdout, the gap not applied, or the time column misread. | Gate: `time.order_respected`, `time.no_skill_on_blocks`, `time.gap_applied`, `time.dayfirst_stops`. Tests: the split as a function of keys. |
| F8 | Under time order a trending or random-walk target is scored against the training mean only. | Gate: `time.persistence_baseline`. Tests: baselines by hand. |

---

## 9. Facts from Phase 0, and the traps found

Measured on this machine on 2026-11-15. A claim here came from a command, not
from what is usually true.

**Stack.** Python 3.12.13 at `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe`.
pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0,
threadpoolctl 3.8.0, pytest 9.2.0. Threadripper PRO 5975WX, 32 cores, 64
threads, 192 GB. Default pools: OpenMP 64 threads, OpenBLAS 24.

**How `read_csv` loads the shapes that matter.**

| Shape | Loads as | Trap |
|---|---|---|
| target of 0 and 1 | `int64`, two distinct | none |
| the same with blank cells | `float64`: 0.0, 1.0, NaN | still two distinct; a rule by type would call it regression (F5) |
| integer column with blanks | `float64` | "non-float" tests miss it; judge whole numbers by value |
| numeric with `n/a`, `1,200`, `abc` | text; `n/a` already missing | 37 of 39 parse; the tool never sees `n/a` as text |
| numeric with only `NA`, `null`, `None` | `float64` with NaN | counted as missing, never as coerced |
| ISO dates; day/month/year dates | text | a CSV never yields a datetime type |
| integer ids | `int64`, all distinct | |
| text ids with leading zeros | `int64`, zeros lost | invisible to the tool: limitation 7 |
| constant; entirely blank | `int64`, one distinct; `float64`, all missing | |
| `True` / `False` | `bool` | pandas counts `bool` as numeric |
| `inf` | float infinity | reachable from a CSV |
| duplicate header names | renamed `a`, `a.1` | silent: read the header line yourself |
| blank header | `Unnamed: 3` | |
| blank lines | skipped | rows are not lines minus one |
| every data row one field longer than the header | first column becomes the index | silent shift of every name |

**Other traps.**

- Text columns have the type `str`, not `object`. `dtype == object` tests are
  wrong on this stack. Copy-on-write is always on: arrays taken from a frame
  can be read-only.
- With the default `low_memory=True`, a numeric column with one text cell in a
  100,000 × 100 file loads as `object` holding both floats and strings, with a
  `DtypeWarning`; in a small file it loads as `str`. `low_memory=False` gives
  `str` at every size. Hence D22.
- pandas' lenient date parser reads `01/12/2023` as 12 January when every value
  is ambiguous, and accepts `1,200`, `May`, `09:30`, `2024Q1` and `1/2` as
  dates. Integers passed to it become nanoseconds since 1970. Hence D18 and
  the fixed list of 4.5.
- `Int64`, `category` and datetime types do not survive a CSV round trip.

**Run time (D16).** A full run on 100,000 rows × 100 columns with the grids of
4.1 and all cores (`n_jobs=-1`, 64 threads): classification 178.6 seconds,
regression 205.6 seconds, against a budget of 600. The forest takes most of it
(105 and 138 seconds), then boosting (36), then the leak screen (27 to 29,
single-threaded in the reference). On one thread the same run would take about
an hour (classification) and several hours (regression): `n_jobs=-1` is the
default for that reason. A fast run on about 1,500 rows takes about 3 seconds
as a new process, most of it importing the libraries.

**Threads and reproducibility (D17).** Measured with the reference
implementation, full profile, 8,000 rows × 21 columns, three tasks including a
target spanning eight orders of magnitude: `profile.json` and `predictions.csv`
are identical at `n_jobs` 1, 2, 4, 16 and 64. Two things make that true, and
each was measured without it:

- A fitted forest predicting on 16 threads gave 60 different bit patterns in 60
  repeats (the trees' votes are summed in the order the threads finish). On one
  thread it is stable. Hence "forests predict on one thread".
- Logistic regression and ridge give different bits at different BLAS thread
  counts on 20,000 × 100. Hence "linear algebra on one thread always". On the
  small tables of the gate this does not show, so it is the tests, through the
  seam, that prove the limit is in force (2.6).
- Histogram gradient boosting gave identical bits at every thread count tried.
- `n_jobs` alone does not limit boosting's OpenMP threads; `threadpoolctl` does.

**The interpreter's base** is on `A:`, a drive mapped to the user's Anaconda
folder through a loopback network share (`\\localhost\C$\...`). Two
consequences, both about the environment and never about the code:

- If the mapping is absent the interpreter does not start: a harness error.
- **Interpreters started at the same instant block each other.** Measured: two,
  four or eight tool processes started together all wait exactly 35 or 70
  seconds and then finish together (every round tried). Rounds of eight starts
  that blocked, by the spacing between starts: 0.1 seconds, 2 of 2; 0.2
  seconds, 3 of 30; 0.3 seconds, 0 of 30; 0.5 seconds, 0 of 30. 35 seconds is
  the share's lock-break timeout. The gate therefore starts its runs 0.5
  seconds apart. Anything else that starts Python processes in parallel on
  this machine (a test helper, a script) must space them too, or simply run
  them one after another. Earlier the same day 17 simultaneous starts did not
  block, so the behaviour comes and goes; pointing the environment at the same
  interpreter by its local path would remove it.

---

## 10. Measured values

Set by the harness stage by measurement, for the owner's approval. Bands belong
to the gate (`verify/smoke_test.py`, where each is written beside its
measurements). Cut-offs belong to this spec.

### 10.1 Bands asserted by the gate

Method: the reference implementation (`verify/reference/`) and each broken copy
a check must catch were run over 30 data seeds, fast profile unless stated. A
band is the reference's mean plus or minus about 4.5 standard deviations,
rounded outwards, so that a correct implementation whose random split differs
from the reference's still lands inside. All bands are on `headline.skill`.

| Check | Dataset | Reference: range (mean, sd) | Band | Broken copy it must catch | Gap |
|---|---|---|---|---|---|
| `model.signal_clf` | `signal_clf` | 0.651 to 0.807 (0.727, 0.040) | 0.54 to 0.91 | none | |
| `leak.excluded` | `leak` | 0.645 to 0.791 (0.739, 0.035) | 0.54 to 0.91 (reused) | F2: 1.0000 on all 30 | 0.09 |
| `model.signal_reg` | `signal_reg` | 0.811 to 0.878 (0.844, 0.018) | 0.76 to 0.93 | none | |
| `leak.regression` | `leak_reg` | 0.810 to 0.882 (0.847, 0.016) | 0.76 to 0.93 (reused) | F2: 0.9994 to 0.9996 | 0.07 |
| `data.columns_accounted` | `messy` | 0.630 to 0.824 (0.731, 0.039) | 0.55 to 0.91 | none | |
| `model.noise_clf` | `noise_clf` | −0.099 to +0.141 (+0.010, 0.058) | −0.27 to 0.27 | F1: 0.389 to 0.559 with the linear model chosen, 1.0000 with the tree | 0.12 |
| `model.noise_reg` | `noise_reg` | −0.502 to −0.253 (−0.357, 0.062) | −0.65 to −0.05 | F1: +0.126 to +0.271 | 0.18 |
| `model.metric_clf` | `imbalanced` | −0.185 to +0.182 (−0.001, 0.092) | −0.42 to 0.42 | F4, by the metric's name and the margin below | |
| `model.selection_full` | `noise_small`, full profile | −0.223 to +0.189 (−0.008, 0.102) | −0.47 to 0.47 | F3 lands inside the band (−0.233 to +0.218) and is caught by the exact comparison | |
| `time.no_skill_on_blocks` | `time_blocks` | −0.167 to +0.270 (+0.012, 0.104) | −0.48 to 0.48 | F7: 0.994 to 1.000 | 0.51 |
| `time.persistence_baseline` | `random_walk` | −0.118 to +0.009 (−0.008, 0.024) | −0.30 to 0.05 | F8: 0.881 to 1.000 | 0.83 |

Two other measured margins:

- `model.noise_p_value`: the fast profile's p-value is a multiple of 1/201. On
  `noise_clf` the reference gave 0.010 to 0.965 and never the floor; F1 gave
  exactly 1/201 on all 30 seeds. The check requires at least 0.0075. A correct
  implementation lands on the floor by chance once in 201.
- `model.metric_clf`: the distance between `headline.score` and the majority
  share is 0.363 to 0.549 for the reference and 0.000 to 0.078 for F4. The
  check requires more than 0.20.

Three things the measurements showed, which shaped the datasets:

- **A refit on all rows (F1) is invisible on low-dimensional noise unless the
  tree is chosen.** With 10 noise columns the linear model was chosen on 7 of
  20 seeds for classification and on all 20 for regression, and its skill
  after the faulty refit (−0.01 to +0.08; +0.012 on average) is inside any
  honest band. `noise_clf` and `noise_reg` therefore have 300 columns, where
  the fault shows whichever model is chosen. The price: on `noise_reg` the
  honest result is below zero, because a linear model fitted to 300 useless
  columns on 1,125 rows is worse than the mean. The band says "no positive
  skill", not "about zero".
- **The best cross-validation score (F3) is inside the noise bands**, under
  both profiles. It is caught by recomputing the headline from
  `predictions.csv`, where it differs from the true holdout score on every
  seed (by at least 0.0016 under the full profile).
- **Blocks of 20 dates gave too few holdout blocks.** `time_blocks` uses blocks
  of 5 dates and exact feature values; with 20 and small feature noise the
  honest range (−0.40 to +0.37) came close to the fault's (0.74 to 0.95).

### 10.2 Cut-offs of the tool

| Cut-off | Starting value | Pinned | Measurement |
|---|---|---|---|
| leak: single-column skill | 0.98 | **0.98** | 30 seeds: `outcome_copy` 0.995 to 1.000 and `amount_scaled` 0.996 to 0.998, flagged 30 of 30; `strong_feature` 0.60 to 0.71, and no other column of any dataset, flagged 0 of 30. Where the edge is: a copy of a 0/1 target with noise of sd 0.25 is flagged on 14 of 30 seeds and with sd 0.30 never; a copy with 0.5% of labels flipped on 21 of 30, with 2% never; a legitimate feature with true ROC AUC 0.993 never, with 0.998 on 10 of 30. Regression: a rescaled copy is flagged while its noise is under about 12% of the target's deviation. |
| leak: missing by class | above 95% and below 5% | **the same** | 100%/0% and 97%/2%: flagged 30 of 30. 96%/4%: 26 of 30. 94%/3% and 97%/6%: 6 of 30. 90%/10%: never. |
| id-like, text | more than 95% of rows distinct (D19) | **the same** | 100% and 98% distinct: id-like. 70%: free text. 44% and 2%: features. |
| id-like, whole numbers | distinct, and consecutive or strictly ordered (D19) | **the same** | At 200, 2,000, 20,000 and 100,000 rows: salary, price in pence, volume, age and a distinct unordered measure are features at every size (the 95% rule called volume an identifier at every size, and salary and price at the smaller ones). Consecutive ids in order, in order with 1% blank, and shuffled: caught. Increasing non-consecutive ids: caught. **Not caught:** a random unordered identifier (as D19 says), and a shuffled consecutive identifier with any blank cell. **Wrongly caught:** a whole-number measure with every value distinct in a table sorted by that measure. |
| free text | not given | **more than 50% of non-missing values distinct** | Needed to tell free text from categories; see the row above. |
| numeric coercion | at least 90% parse | **90%** | Only values of 1,000 or more carry a thousands separator, so a column with 50% of its cells written that way still parses 91% and is numeric with 173 of 2,000 cells coerced; written that way throughout, 81% parse and the column is excluded as free text. |
| date-like | 95%, fixed forms | **95%, the forms of 4.5** | The fixed list matches ISO dates and date-times, d/m/y, m/d/y, d-m-y, d.m.y, y/m/d and month-name forms (4 of 4 each), and none of `1,200`, `May`, `09:30`, `2024Q1`, `1/2`, `1.2.3`, `2024-01`, or impossible dates. pandas' lenient parser accepts the first five. |
| small holdout | fewer than 50 of the rarer class | **50** (regression: 50 holdout rows) | On pure noise, 95% of honest results fall within ±0.23 of zero with 50 rows of the rarer class in a balanced holdout (±0.16 at 5% positives), ±0.36 with 20, ±0.16 with 100. |
| too small to profile | fewer than 100 usable rows | **100**, and 10 of each class | 30 seeds per size. Random scheme at 100 rows: every run finishes; skill sd 0.17 (classification) and 0.06 (regression). Time order at 100 rows: fold 1 trains on 13 rows, every run finishes, no fold unscored, skill sd 0.16 and 0.08. Below the floor under time order, folds start to hold one class: 1 run in 30 at 80 rows, 4 at 60, 13 at 40. |
| no block-bootstrap interval | fewer than 20 holdout timestamps | **30** | See 10.3. |
| autocorrelated panel target | lag-1 autocorrelation above 0.8 | **0.5** | "The same as last period" beats the training mean exactly when the autocorrelation is above 0.5 (its skill over the mean is 2r − 1). Measured on simulated panels of 200 timestamps: at a measured autocorrelation of 0.45 that naive rule has skill −0.07; at 0.74 it has +0.49, and a cut-off of 0.8 warned on only 19% of such panels; at 0.85 it has +0.75. |

### 10.3 The block bootstrap's coverage

How often the nominal 95% interval of step 12 holds the true squared error, in
400 simulated holdouts per cell, one row per timestamp, by the serial
correlation of the errors:

| Holdout timestamps | Block length | Independent | Correlation 0.5 | Correlation 0.8 |
|---|---|---|---|---|
| 8 | 2 | 75% | 70% | 46% |
| 12 | 3 | 81% | 74% | 55% |
| 16 | 3 | 84% | 76% | 56% |
| 20 | 3 | 79% | 79% | 62% |
| 25 | 3 | 88% | 85% | 70% |
| 30 | 4 | 86% | 82% | 64% |
| 40 | 4 | 90% | 85% | 71% |
| 50 | 4 | 89% | 88% | 74% |
| 100 | 5 | 94% | 90% | 80% |
| 250 | 7 | 94% | 90% | 85% |

For comparison, the row bootstrap of the random scheme on independent rows:
88% at 20 rows, 92% at 50, 95% at 100 and 250.

Reading: the interval is too narrow on short holdouts whatever the method, and
the floor of 30 is where the block interval on independent data reaches the row
bootstrap's coverage on the smallest holdout the random scheme allows (25
rows). Under strong serial correlation the interval stays too narrow even at
250 timestamps: the cube-root block length of D9 is shorter than the
dependence. That is a property of the decided method, reported to the owner;
it is not something an implementation can fix.

### 10.4 Run time and reproducibility

See section 9: full runs of 178.6 and 205.6 seconds at 100,000 × 100 on all
cores; identical files at `n_jobs` 1, 2, 4, 16 and 64.

---

## 11. What is open

The builder decides these, and no test may pin them:

- the internal module layout, function names and helper classes (only the
  surface of 2.1 to 2.7 is a contract; `stages` and `extensions` are facades
  over code that may live anywhere);
- how the one seed is turned into generator streams for the cap, the split, the
  folds and the resamples. Two correct implementations will choose different
  holdout rows from the same seed, and that is fine: the properties of step 7
  are the contract, not the row numbers. (Within one implementation,
  `profile()` and `stages.split_random` must agree: 2.7.);
- the wording of every warning `message`, every error message, the console
  summary and `interval_reason` / `p_value_reason` (their presence is pinned);
- the floating-point type of the preprocessed matrix (its columns and their
  names are pinned by `stages.Preprocessor`, 2.7);
- extra keys in `profile.json` beyond those of section 5 (allowed anywhere
  except inside `rows`; they must be deterministic or live in `meta.timing`);
- extra keys inside `meta.timing`;
- how the work is spread over `n_jobs` cores, as long as the result does not
  depend on it;
- the content of `columns[].dtype`, and `kind` for a column that is not a
  feature.

Pinned, despite looking like detail: the estimators and grids of 4.1, the
preprocessing of step 8, the single-column screen of step 8, the percentile
and block rules of step 12, the fixed texts of 5.4.

Not judged by any test or gate check, and left to a human: whether the
limitations read well, whether the console summary is clear, whether a skill
figure on real data is believable.

---

## 12. Decision index

| Decision | Where it is carried |
|---|---|
| D1 roadmap | 1 |
| D2 interface | 2.1 to 2.3 |
| D3 input and column handling | 3 (steps 1, 2, 5, 8), 6 |
| D4 task rule | 3 (step 4) |
| D5 headline and baseline | 3 (step 11) |
| D6 random validation | 3 (step 7), 5.4 |
| D7 time-ordered validation | 3 (steps 3, 7) |
| D8 models and selection | 3 (step 9), 4.1 |
| D9 uncertainty | 3 (step 12) |
| D10 regression baseline under time order | 3 (step 11) |
| D11 leak rule | 3 (step 8), 4.3 |
| D12 scale and the fast profile | 3 (step 6), 4.2, 9 |
| D13 failure modes | 8 |
| D14 predictions in the output | 5.1 |
| D15 UI direction | not in phase 1; 2.3 keeps the result object complete |
| D16 forest strength and budget | 4.1, 9 |
| D17 cores and reproducibility | 2.1, 3 (seeds and threads), 9 |
| D18 time column forms | 3 (step 3) |
| D19 id-like rule | 3 (step 5), 4.3 |
| D20 fast and full agree | 3 (steps 4, 5, 6) |
| D21 whole-number regression warning | 3 (step 4), 2.5 |
| D22 loading rules | 4.4, 4.5, 6 |
