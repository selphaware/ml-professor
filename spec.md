# ML Professor, phase 1: the contract

This file is what the application is built from and what the tests are written
against. It restates the owner's decisions (`docs/DECISIONS.md`, cited as D1 to
D43) without changing them, pins every value they leave to this stage, and
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

The repo also carries one small synthetic table, `examples/customers.csv`, for
the commands the README shows (D41). The README names no file under
`verify/` or `tests/`: those folders belong to the judge, and part of
`verify/` exists only after the gate has run.

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
| `out_dir` | A folder, or `None`. When given it is created if missing and `profile.json` and `predictions.csv` are written into it. It cannot be used when the path, or the nearest folder above it that exists, is something other than a folder (it names an existing file, for one): an `InputError` like any other invalid argument, found without creating anything (D39). |
| `n_jobs` | `-1` for every core (`os.cpu_count()`), or a positive whole number (D17). |

Any other value of an argument raises `InputError`. Nothing is written when an
error is raised.

**Process-wide settings are left as the caller had them** (D39), whether
`profile()` returns or raises; the same holds for every function of `stages`.
That covers `csv.field_size_limit()`, the thread limits `threadpoolctl`
reports, the `warnings` filters, numpy's error state, the working directory
and the environment. A test checks the first two: the others can be moved by a
library's first import, which is not the tool's doing.

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

The summary's wording and layout are open. What it shows is pinned (D39):

- the interval on the score, `interval_low` and `interval_high`, and the
  interval on skill, `skill_low` and `skill_high`, each as a number; when
  there is no interval (`interval_method` is null), the text of
  `headline.interval_reason` in their place;
- `p_value` as a number, or, when it is null, the text of
  `headline.p_value_reason`;
- one line per warning, holding that warning's `code` and, when it has one,
  its `column`. Two warnings never share a line.

A value counts as shown when some decimal number printed in the summary is
within 0.0005 × max(1, |value|) of the value in `profile.json`: four
significant digits are enough. A null `skill`, `skill_low` or `skill_high`
needs no number; the line of the `skill_undefined` or the `no_skill_interval`
warning covers it.

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
(2.6) and `mlprofessor.stages` (2.7); both are pinned and built with the
package (D26). Everything else is internal and may be reorganised.

### 2.4 Errors

Every named error is a subclass of `mlprofessor.errors.ProfileError`, which
subclasses `Exception`. Each class has a class attribute `code`, the text the
command line prints. The message says what was found and what the caller can do.

| Class | `code` | Raised when |
|---|---|---|
| `InputError` | `input_error` | An argument has an invalid value; `out_dir` cannot be used (2.1); `data` is neither a DataFrame nor a path; the file is missing, empty, not valid UTF-8 or not parseable as CSV; data rows have more fields than the header; `gap` above 0 without `time_col`; `time_col` equals `target`. |
| `ColumnNotFoundError` | `column_not_found` | `target` or `time_col` is not a column. |
| `DuplicateColumnsError` | `duplicate_columns` | Two columns have the same name (D22). A name that holds only white space is a name like any other: two header cells of one space each are two columns of the same name (D42). |
| `TargetConstantError` | `target_constant` | The target has fewer than two distinct values among the usable rows. |
| `TaskAmbiguousError` | `task_ambiguous` | `task="auto"` and the rule cannot decide (D4). |
| `TaskUnsupportedError` | `task_unsupported` | A declared task the target does not support: classification with other than two distinct values, regression on a non-numeric target. |
| `TimeColumnError` | `time_column_invalid` | The time column is not in an accepted form (D18). |
| `TooFewRowsError` | `too_few_rows` | Fewer than 100 usable rows; fewer than 10 usable rows of a class; under time order, too few distinct timestamps to form the holdout and six blocks. |
| `FastProfileError` | `fast_profile_too_few_timestamps` | Under `profile="fast"` with a time column, the rows kept by the row cap hold too few distinct timestamps to form the holdout and six blocks (D20). The message tells the caller to use `profile="full"`. |
| `GapError` | `gap_too_large` | The gap leaves fewer than six training timestamps, or no training rows before a validation block. |
| `FeatureNameCollisionError` | `feature_name_collision` | Two feature columns would give the preprocessed matrix a column of the same name (step 5; D37). The message names every column involved. |
| `NoFeaturesError` | `no_features` | No column has the role feature after the leak screen. An error returns no column accounting, so the message carries it: the name of every excluded column and, for each, its reason as `columns[].reason` spells it (D39). |
| `OneClassError` | `one_class` | Classification, and the holdout or the training rows hold one class, or no fold can be scored. |
| `AllModelsFailedError` | `all_models_failed` | Every configuration failed, during selection or after it was chosen (steps 9 and 10). The message carries each error text. |

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
| `strong_single_column` | the column | One per column that the leak screen keeps as a feature although its single-column skill is at least 0.90 (and below 0.98): step 8, rule 3 (D30, D33). |
| `small_holdout` | `null` | Classification: fewer than 50 holdout rows of the rarer class. Regression: fewer than 50 holdout rows (D23). |
| `date_column_undeclared` | the column | One per column excluded as `datetime` when `time_col` is `None` (D7). |
| `autocorrelated_panel_target` | the target | Regression under time order with repeated timestamps, and the lag-1 autocorrelation of the per-timestamp mean of the target over the training rows is above 0.5 (D10, D23; section 10). |
| `model_failed` | `null` | One per configuration with status `failed`, whether it failed during selection or after it was chosen (D38). |
| `values_coerced` | the column | One per numeric column in which text cells were turned into missing (D22). |
| `numeric_with_separators` | the column | One per column excluded with the reason `numeric_with_separators` (step 5, rule 4a; D28, D34). It has this one form: a column that is kept never carries it. |
| `whole_number_regression_target` | the target | `task="auto"` picked regression and every target value is a whole number (D21). |
| `blank_header` | pandas' name for it | One per CSV column whose header cell is blank, which means empty (D22). A cell that holds white space is a name, and gets no warning (D42). |
| `infinite_values` | the column | One per numeric feature holding infinite values. |
| `duplicate_rows` | `null` | The loaded table holds exact duplicate rows. They are kept. |
| `fold_not_scored` | `null` | Classification, and a fold's training or validation rows hold one class. |
| `no_interval` | `null` | No interval could be computed on the score; `headline.interval_reason` says why. |
| `skill_undefined` | `null` | Regression. Exactly one when `headline.skill` is null, because the headline baseline's squared error on the holdout is exactly zero (D36). Never otherwise: it is the code for a headline skill that has no value, and for nothing else (D42). The message states the reason. |
| `no_skill_interval` | `null` | Regression. Exactly one when `headline.skill` is a number and `skill_low` and `skill_high` are null although the score has its interval (`interval_method` is not null): skill could be computed in fewer than half of the resamples (step 12; D42). Never otherwise: not when `headline.skill` is null, which `skill_undefined` covers, and not when there is no interval at all, which `no_interval` covers. The message states the reason. |
| `interval_approximate` | `null` | Exactly one whenever `headline.interval_method` is `bootstrap_blocks` (D27): that interval tends to be too narrow (10.3). Never under the random scheme, and never when there is no interval. It does not depend on how long the holdout is or how correlated the errors are, which the tool cannot know (D33). |

### 2.6 Seam for tests (D26)

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
  per scored fold per configuration for estimators, and once more for each
  configuration tried after selection (step 10: one, unless a chosen
  configuration fails); once per scored fold and once for the final fit for
  steps.
- Every `X` handed to a step or to a seam-supplied estimator is a pandas
  DataFrame whose **index is the rows' zero-based positions in the loaded
  source** (the same numbers as `predictions.row`). `y` is a Series with the
  same index: 0 and 1 for classification (1 is the positive class), floats for
  regression. A step receives the feature columns as loaded and typed: numeric
  columns as floats with NaN for missing; categorical columns as text, each
  cell its level, which is the text without the white space round it (step 5;
  D42), with `"(missing)"` for missing. So a step never sees `red ` or ` red`,
  only `red`. An estimator receives the fully preprocessed matrix, in which a
  level's column is named by that same stripped text (`colour=red`).
- The estimator's `fit` is called inside the tool's thread limits, so
  `threadpoolctl.threadpool_info()` called there reports them.
- When the block ends the previous state is restored, including after an
  exception. Blocks may be nested.

With this a recording fake proves: no `fit` of any kind ever receives a holdout
row; inside each fold no `fit` receives a row of that fold's validation block;
holdout rows are handed to one configuration at a time, in the order of step
9, and to none once one has predicted them, so that when nothing fails after
selection the chosen configuration alone ever receives one (step 11); linear
algebra is on one thread.

The fake sees the steps and the models it supplies. It does not see inside the
tool's own preprocessing or its leak screen, so those are proved by effect.
The built-in imputation, scaling and encoding: the matrix handed to a
seam-supplied estimator's `fit` is compared, value for value and column for
column, between two runs on tables that differ only in rows the fit must not
have seen. The leak screen: crafted columns that only a screen fed holdout
rows would flag. Section 8 lists the faults these catch.

### 2.7 Step functions (D26)

The data-handling steps are built and verified before any model exists, so each
is also callable on its own through one module, `mlprofessor.stages`. It is a
thin facade: where the code lives is open, the names and behaviour below are
pinned, and `profile()` must use the same code (a test may compare the two).
Section 3 defines what each step does; this table defines only the call.

| Call | Returns | Raises |
|---|---|---|
| `load_table(data)` | `(frame, info)`. `frame` is the loaded table (step 1, section 6) with a default index 0..n−1, the rows' source positions, and string column names. `info` is `{"kind", "source", "warnings"}` as in `input.kind`, `input.source` and a list of warning records (only `blank_header` arises here). | `InputError`, `DuplicateColumnsError` |
| `usable_rows(frame, target, time_col=None)` | Step 2 as a count: `{"usable", "dropped_missing_target", "dropped_missing_time"}`. `usable` is an array of the positions of the usable rows in increasing order; the two counts and its length add up to the rows of `frame`. A missing time value is a missing cell (or an infinite number); the time column's form is not checked here. | `ColumnNotFoundError` |
| `column_roles(frame, target, time_col=None)` | A DataFrame with the eleven columns of `profile.json["columns"]` (5.2), one row per column of `frame` in order, as decided by step 5 alone: `leak_rule` is null and `n_imputed` is 0 throughout, since the leak screen and imputation come later. Usable rows are those with a target and, when `time_col` is given, a time value. It returns no warnings: those of step 5 (`values_coerced`, `infinite_values`, `numeric_with_separators`) are raised by `profile()`. | `ColumnNotFoundError` |
| `decide_task(values, task="auto")` | `{"type", "decided_by", "n_distinct", "classes", "positive_class", "warnings"}` for the target values given (missing ones are ignored). `warnings` is a list of warning codes: `["whole_number_regression_target"]` or `[]`. | `TargetConstantError`, `TaskAmbiguousError`, `TaskUnsupportedError`, `InputError` |
| `read_time(values)` | `(keys, form)`. `keys` is a numpy array, one per value, that sorts in time order: `datetime64[ns]` with `NaT` for a missing value under the `datetime` and `iso8601` forms (in UTC if the values carried a zone), `float64` with NaN for a missing or infinite value under the `numeric` form. `form` is as in `validation.time_form`. | `TimeColumnError` |
| `split_random(n_rows, classes=None, seed=0)` | `{"training", "holdout", "folds"}`: arrays of positions in 0..`n_rows`−1, each in increasing order; `folds` is a list of five `{"training", "validation"}`. `classes`, when given, is an array of `n_rows` labels with two distinct values and makes the split stratified (step 7). | |
| `split_time(keys, gap=0)` | `{"training", "gap", "holdout", "folds"}`: arrays of positions into `keys`, each in increasing order; `folds` is a list of five `{"training", "gap", "validation"}`, fold *k* validating on block *k*+1. `keys` is as returned by `read_time`, in any order, with no missing value. | `TooFewRowsError`, `GapError` |
| `roc_auc(y, score)` | ROC AUC of `score` for labels `y` of 0 and 1 (1 is positive); ties share their average rank. | `OneClassError` |
| `mse(y, pred)` | Mean squared error. | |
| `skill(metric, score, baseline_score)` | For `"roc_auc"`: (score − baseline) ÷ (1 − baseline), which is 2 × AUC − 1 at the baseline 0.5. For `"mse"`: 1 − score ÷ baseline; and `None`, not an error, when `baseline_score` is 0, where skill has no value (D36). | `ValueError` for any other metric |
| `previous_value(y, keys, gap=0)` | An array aligned with `y`: for each row the target of the row `gap + 1` timestamps earlier, NaN where there is none. | `ValueError` if `keys` are not unique |
| `Preprocessor(kinds)` | The built-in preprocessing of step 8 for one fit. `kinds` maps a column name to `"numeric"` or `"categorical"`. `fit(frame)` learns from the rows given and returns the object. `transform(frame)` returns a DataFrame with the same index: one column per numeric column, under its own name, then for each categorical column one column per kept level named `<column>=<level>`, levels in sorted order, holding 0.0 and 1.0. A missing categorical cell is the level `(missing)`. No two columns of the result ever share a name (D37). | `FeatureNameCollisionError` from `fit`, when two of the names it would give are the same, judged on the levels it keeps from the rows given |

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
not counting blank lines. A blank line is one that is empty or holds only
spaces and tabs (section 6).

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
| 4a | text with more than one distinct value, in which at least 90% of the non-missing values either parse as numbers or are numbers with thousands separators, and at least one is a number with thousands separators (D28, D34) | excluded, `numeric_with_separators` |
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

**A categorical level is the cell's text without the white space round it
(D42).** `red`, ` red` and `red ` are one level, `red`, everywhere a level is
used: in the counts of this step (distinct values, and so `constant`,
`id_like`, `free_text` and whether the column is a feature at all); in the
collision rule below; in the leak screen of step 8, whose codes are those of
the stripped levels; in what a seam-supplied step receives (2.6); and in the
preprocessed matrix, where the three cells share the one column
`colour=red`. A level that differs from another only by a trailing space is a
data-entry artefact, not a category. A cell that holds only white space is not
a missing cell: its level is the empty text.
"Parse as numbers" means `pandas.to_numeric(..., errors="coerce")` gives a
number. In rule 8b, judged on the non-missing values: "consecutive" means
largest minus smallest plus one equals their count; "strictly ordered" means
strictly increasing, or strictly decreasing, down the file.

**A number with thousands separators (D28)** is a text value which, after
stripping, is the whole of: an optional sign (`+` or `-`); one to three digits;
one or more groups of a comma and exactly three digits; and optionally a point
followed by one or more digits. Digits are `0` to `9`. As a pattern:
`[+-]?[0-9]{1,3}(,[0-9]{3})+(\.[0-9]+)?`.

| Matches | Does not match |
|---|---|
| `1,200`, `-3,450.75`, `+12,345,678`, `0,123` | `1200` and `532.10` (plain numbers: they parse), `12,34`, `1,2345`, `1234,567`, `1.200,50`, `1,200.`, `1,200.5.1`, `$1,200`, `(1,200)`, `1 200`, `1,200 units` |

Only this comma form is recognised (D33): `$1,200`, `1.200,50` and `1 200` are
ordinary text to every rule.

**Such a value is never parsed as a number**, because the same text means 1.2
in some locales: the tool does not strip the commas, in any column, under any
rule. **Nor is a column used with such cells left out** (D34): the cells lost
would be the values of 1,000 and more, so the column would be missing its
largest values. Rule 4a therefore comes before the coercion rule, and one such
cell is enough:

- a column written with separators throughout is excluded, however few of its
  values parse and however many distinct values it has, so it is never
  `id_like`, `free_text` or a categorical feature;
- a column of plain numbers in which a single cell is written `1,200` is
  excluded, not read as numeric with that cell coerced;
- a column of plain numbers whose other cells are text of another kind (`abc`,
  `12 units`, `$1,200`) holds no such value, so rule 4a does not apply: rule 5
  reads it as numeric and counts those cells as coerced, as before;
- a column in which fewer than 90% of the values parse or match (85% written
  with separators and 15% words) is left to the rules that follow;
- a column holding one distinct value, `1,200` in every row, is left to rule 6
  and is `constant`.

Like every rule of this step, rule 4a looks at the usable rows. A separator
cell that sits only in a row with no target does not count: the column is read
by rule 5, and that cell is one of its coerced cells.

Rule 4a adds a `numeric_with_separators` warning naming the column it excludes
(D34). That is the only form of the warning.

Counts are kept per column over **every loaded row** (section 5.3): `n_missing`,
`n_coerced` (rule 5: non-missing text cells that did not parse, which become
missing, with a `values_coerced` warning), `n_infinite` (which become missing,
with an `infinite_values` warning). `n_coerced` is 0 for a column that rule 5
did not read as numeric, including one excluded by rule 4a.

**Feature names must be distinct (D37).** The preprocessed matrix names a
numeric feature by its own name and each level of a categorical feature
`<column>=<level>` (2.7). If two of those names would be the same, the run
stops here with `feature_name_collision`, before the row cap, the split, the
leak screen and any fit, and nothing is written. Two shapes give a collision:
a numeric feature whose name is a categorical feature's name, an equals sign
and one of its levels (`colour=red` beside `colour` with the level `red`); and
two categorical features whose names and levels combine to the same text
(`site` with the level `a=b`, beside `site=a` with the level `b`). The message
names every column involved.

Which names are compared is pinned, so that the outcome depends on the table
and never on the seed, the profile or the rows a fit happens to see:

- the columns are those with the role `feature` after the rules above, before
  the leak screen: a column the screen would go on to exclude counts;
- a categorical column's levels are **every level among the usable rows**, not
  only the 30 most frequent of some fit, and `(missing)` is one of them when a
  usable row has a missing cell. A level is the stripped text, as above: a
  cell ` red ` gives the name `colour=red`;
- like every rule of this step it is decided before any row cap, so `fast` and
  `full` agree (D20): a level that sits only in rows the cap leaves out
  counts, and one that sits only in rows with no target does not.

The price is accepted (D43): a collision stops the run even when the level
sits only in rows the fast profile leaves out, or in a column the leak screen
would have excluded.

Silently renaming one column, or letting one overwrite the other, is the fault
this rule exists to prevent: the first build listed a column as a feature that
had reached no model.

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

**8. The leak screen (D11, D24, D30), on training rows only.** For each feature
column, in file order, the first rule that applies decides:

1. *Filled in for one class and empty for the other* (classification only): on
   the training rows, the column is missing for more than 95% of one class and
   less than 5% of the other. Excluded, leak rule `missing_by_class`.
2. *On its own it predicts the target almost perfectly*: the column's
   single-column cross-validated skill is at least 0.98. Excluded, leak rule
   `predicts_target`.
3. *On its own it predicts the target strongly* (D30): that same skill is at
   least 0.90, and so below 0.98. The column **stays a feature**, with a
   `strong_single_column` warning that names it, so that the caller can check
   it is known before the outcome.

A column that none of the three applies to stays a feature without comment.
Rules 2 and 3 use one number, measured once per column; a column excluded by
rule 1 is not measured and gets no `strong_single_column` warning.

Single-column skill is measured with the **selection folds of step 7** (D24; so
under time order it respects time, and no holdout row is involved): in each
scored fold a decision tree (`max_depth=6`, `min_samples_leaf=5`,
`random_state=seed`) is fitted on that one column of the fold's training rows
and predicts its validation rows. A numeric column is used as it is, missing
values left missing; a categorical column as whole-number codes of the sorted
levels of the fold's training rows (levels as step 5 defines them: the
stripped text), with −1 for an unseen level. The
predictions of all folds are pooled and skill is computed as in step 11:
2 × AUC − 1, or 1 − (squared error ÷ baseline squared error), the baseline
being each fold's training mean and, under time order with unique timestamps,
whichever of that and the previous value has the lower error. When that
baseline's squared error over the pooled rows is exactly zero, the column's
skill cannot be computed (D36): rules 2 and 3 do not apply to it, it stays a
feature and it gets no warning.

A column excluded by rule 1 or 2 gets role `excluded`, reason `suspected_leak`,
its `leak_rule`, and a `suspected_leak` warning. It reaches no model. If no
feature is left: `no_features`, its message naming every excluded column with
its reason (2.4). A column kept by rule 3 has role `feature`, a
null `reason` and a null `leak_rule`, and reaches the models like any other:
it still drives the headline, because the tool cannot tell a near-leak from a
very strong real feature (D33). The warning asks a question; it does not
accuse.

Known limit (D31): in a panel (repeated timestamps) the only baseline is the
training mean, so a lagged level of a persistent target can be excluded by rule
2 or warned about by rule 3. Profile returns or changes there.

**Preprocessing, fitted on the rows of each fit only (D3).** One preprocessing
is fitted for each fit, on that fit's training rows, and applied to the rows it
predicts:

- numeric features: missing values (including coerced and infinite cells) are
  replaced by the median of the fit's training rows (0 if the column is all
  missing there), then standardised with the mean and standard deviation of
  the fit's training rows (a zero deviation is read as 1);
- categorical features: a missing value is the level `"(missing)"`; one-hot
  columns for the levels seen in the fit's training rows, at most the 30 most
  frequent (ties by level text); any other level is all zeros. The levels are
  those of step 5, the stripped text, so `red` and `red ` fill one column.

The same preprocessing serves every model family. `n_imputed` of a feature is
the number of its missing cells among the training and holdout rows.

**9. Select by five folds inside the training rows (D8, D16).** For each
configuration of each family (section 4.1), in family order then grid order,
and for each scored fold: fit the preprocessing and the model on the fold's
training rows, predict its validation rows, and score them: ROC AUC for
classification, mean squared error for regression. `cv_mean` is the mean over
scored folds and `cv_std` their sample standard deviation.

A configuration whose fit or prediction raises on a fold is recorded with
status `failed`, the error text and a `model_failed` warning. All five of its
fold scores are null, and so are its `cv_mean` and `cv_std`: it loses the
scores of the folds it had got through. That is an accepted limit (D43): a
first-pass test requires it, and D38 covers only a failure after the choice.
It is never dropped from the table and never chosen. All failing:
`all_models_failed`.

The configurations that have a `cv_mean` are **ranked**, best first: the
highest ROC AUC, the lowest squared error. Ties go to the earlier family, then
the earlier grid position. The first in the ranking is chosen.

**10. Refit the chosen configuration on all training rows, and predict the
holdout with it (D38),** with its preprocessing fitted on all training rows.

If that fit or that prediction raises, the configuration was chosen in vain.
It gets status `failed`, its error text and a `model_failed` warning, and its
`chosen` stays false. Its `fold_scores`, `cv_mean` and `cv_std` **stay as they
were measured**: the table reports what was tried. The next configuration in
the ranking of step 9 is then chosen and tried in the same way, and so on. If
none is left: `all_models_failed`, its message carrying each error text.

The holdout plays no part in the choice. A configuration is passed over only
for raising, never for its score, so the headline is not inflated by the
fallback.

**11. Score it once on the holdout; score the baselines on the holdout (D5,
D10).** **The headline comes from exactly one configuration**, the chosen
one: the first in the ranking whose refit and holdout prediction both succeed.
`headline`, `cv.chosen`, `cv.chosen_mean`, the predictions, the interval and
the p-value come from it alone, and a failed configuration contributes no
score anywhere. Holdout rows are handed to one configuration at a time, in
ranking order, and to none once one has predicted them. So when nothing fails
after selection, the chosen configuration alone ever receives a holdout row;
and in every case each other configuration that received one has status
`failed` and stands ahead of the chosen one in the ranking.

- Classification: `score` is ROC AUC of the predicted probabilities (ties share
  their average rank). The baseline is `chance`, 0.5. **skill = 2 × AUC − 1.**
- Regression: `score` is mean squared error. The baseline `training_mean`
  predicts the mean target of the training rows for every holdout row.
  **skill = 1 − score ÷ baseline score.**
- **When the headline baseline's score is exactly zero, skill has no value
  (D36).** `headline.skill` is `null`: never an error, an infinity or a zero
  in its place. One `skill_undefined` warning says why. The score, every
  baseline with its score (the zero included) and the rest of the output are
  reported as usual, and the run completes. It happens when the baseline
  predicts every holdout row exactly: a target flat across the holdout, under
  time order, where the previous value is the value itself; and under the
  random scheme too, when the training mean is the value of every holdout row
  (D43).
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
measured, never clipped at zero (D5); the limitations statement says how to
read it (D29; 5.4, sentence 9). A dummy model that predicts the baseline is the
reference only; it is never a candidate and never appears in the models table.

**12. Compute the interval and the p-value from the holdout predictions (D9).**
Nothing is refitted. *R* is 1,000 under `full` and 200 under `fast`.

- *Random scheme, interval:* *R* resamples of the holdout rows with
  replacement. For each, the score and the skill (for regression the baseline's
  predictions are resampled with the rows). The 95% interval is the 2.5th and
  97.5th percentiles (linear interpolation), reported for the score and for the
  skill. Method `bootstrap_rows`.
- *Random scheme, p-value:* *R* shuffles of the holdout targets against the
  fixed predictions. **p = (1 + number of shuffles whose skill is at least the
  observed skill) ÷ (1 + *R*).** "At least" means a tie counts. For
  classification a tie is exact, since ROC AUC is a count of ranks, and a test
  holds an implementation to it. For regression a tie is a tie only up to
  rounding, and no tolerance is pinned (D43).
- *Time-ordered scheme, interval:* a moving-block bootstrap over the distinct
  holdout timestamps in order. With *T* timestamps, the block length is *L* =
  the smallest whole number with *L*³ ≥ *T*. Each resample draws ⌈*T*/*L*⌉
  blocks of *L* consecutive timestamps, each starting at a position drawn
  uniformly from those that fit, joins them, keeps the first *T* timestamps,
  and takes every row of each drawn timestamp. Percentiles as above. Method
  `bootstrap_blocks`, with `block_length`. Whenever the interval is computed
  this way the warning `interval_approximate` is added (D27): it tends to be
  too narrow (10.3). With fewer than 30 distinct holdout timestamps (D23) there
  is no interval: the fields are `null`, `interval_reason` says why, and the
  warning `no_interval` is added in place of `interval_approximate`.
- *Time-ordered scheme, p-value:* absent (`null`), and `p_value_reason` states
  that shuffling assumes independent rows.
- *Resamples in which the score cannot be computed* (classification: the rows
  drawn hold one class) are left out of both intervals. `n_resamples_used` is
  the number kept. If fewer than half of the *R* can be used there is no
  interval under either scheme: `interval_method` and `block_length` are
  `null`, with a reason and the `no_interval` warning (and no
  `interval_approximate`).
- *Resamples in which only the skill cannot be computed (D36, D40).* In
  regression, a resample whose baseline has a squared error of exactly zero on
  the rows drawn has a score and no skill. It **stays in the interval on the
  score** and is left out of the interval on skill only, and it is counted
  apart: `n_resamples_skill_used` is the number of resamples kept for the
  skill interval (for classification it equals `n_resamples_used`). If fewer
  than half of the *R* can be used for skill, `skill_low` and `skill_high` are
  `null` while the interval on the score stands. Which warning says so depends
  on the headline (D42):
  - `headline.skill` is null: the `skill_undefined` warning of step 11 covers
    it, and no other is added. The baseline is then exact on every holdout
    row, so every resample has a zero baseline error and
    `n_resamples_skill_used` is 0.
  - `headline.skill` is a number: one `no_skill_interval` warning is added.
    The baseline is exact on most holdout rows and not on all, so that most
    resamples miss every row on which it errs.
- *No p-value without skill.* When `headline.skill` is null there is nothing
  to compare the shuffles with: `p_value` and `n_permutations` are `null` and
  `p_value_reason` says why.

Also added here: `small_holdout` (2.5).

**13. Assemble the result; write the files** (section 5).

### Seeds and threads (D17, D25)

One `seed` drives everything random. The same arguments on the same data give
the same `profile.json` and `predictions.csv`, apart from `meta.timing`, and
apart from `meta.n_jobs` when `n_jobs` differs.

For the whole run the tool uses `threadpoolctl` to hold linear algebra (BLAS)
to **one thread always** and OpenMP to `n_jobs` threads. Forests are fitted
with `n_jobs` and **predict on one thread**. Both are needed: section 9 gives
the measurements. How the generator is seeded for each purpose is open.

---

## 4. Pinned values

### 4.1 Model families and grids (D8, D16, D25)

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
the only departure from scikit-learn's defaults (D16). Three settings are
pinned by D25: six configurations per family; histogram gradient boosting with
early stopping off, so that its behaviour does not change at 10,000 rows; and
1,000 iterations for logistic regression. `params` in the models table holds
exactly the grid arguments (`{}` for `tree-0`).

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

### 4.3 Cut-offs (D23, D28, D30, D33, D34; measurements in section 10)

| Cut-off | Value |
|---|---|
| leak: single-column cross-validated skill | at least 0.98: excluded |
| leak warning: the same skill | at least 0.90 and below 0.98: kept, with the `strong_single_column` warning |
| leak: missing by class | more than 95% of one class and less than 5% of the other |
| id-like, text column | distinct values more than 95% of the usable rows |
| id-like, whole-number column | every value distinct, and consecutive or strictly ordered in file order |
| free text | distinct values more than 50% of the non-missing values |
| numeric coercion | at least 90% of the non-missing values parse as numbers |
| numbers with separators | at least 90% of the non-missing values parse as numbers or are numbers with thousands separators, and at least one is such a number: excluded (step 5, rule 4a) |
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
| `reason` | string or null | for an excluded column: `constant`, `id_like`, `datetime`, `free_text`, `numeric_with_separators`, `suspected_leak`, `all_missing` or `unsupported_type`; otherwise null |
| `leak_rule` | string or null | for a suspected leak: `predicts_target` or `missing_by_class`; otherwise null |
| `n_missing` | integer | missing cells among all loaded rows, as loaded |
| `n_coerced` | integer | non-missing text cells that did not parse, among all loaded rows, in a column read as numeric by rule 5 of step 5; otherwise 0 |
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
| `baseline_score` | number | 0.5, or the baseline's squared error on the holdout, which may be 0 |
| `skill` | number or null | step 11; null when `baseline_score` is 0, with the `skill_undefined` warning (D36) |
| `interval_method` | string or null | `bootstrap_rows`, `bootstrap_blocks`, or null when there is no interval |
| `interval_level` | number | 0.95 |
| `interval_low`, `interval_high` | number or null | the interval on `score`; null when `interval_method` is null |
| `skill_low`, `skill_high` | number or null | the interval on `skill`; null when `interval_method` is null, and when skill could be computed in fewer than half of the resamples (step 12): always so when `skill` is null (`skill_undefined`), and otherwise with the `no_skill_interval` warning |
| `n_resamples` | integer | 1,000 or 200 |
| `n_resamples_used` | integer | resamples in which the score could be computed |
| `n_resamples_skill_used` | integer | of those, the resamples in which skill could be computed too: the ones whose baseline error was not zero. Equal to `n_resamples_used` for classification |
| `block_length` | integer or null | under `bootstrap_blocks` |
| `interval_reason` | string or null | why there is no interval on the score |
| `p_value` | number or null | step 12; null under time order, and when `skill` is null |
| `n_permutations` | integer or null | shuffles used; null when there is no p-value |
| `p_value_reason` | string or null | why there is no p-value |

**`baselines`**: a list of records `{name, score, used}`. Classification: one,
`chance`. Regression: `training_mean`, and under time order with unique
timestamps also `previous_value`. Exactly one has `used` true and it is the
headline baseline. A score of 0 is reported as it was measured.

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
| `status` | string | `fitted` or `failed`: failed when it raised during selection (step 9) or after it was chosen (step 10) |
| `error` | string or null | the error text of a failed configuration |
| `fold_scores` | list | five entries, a number or null. Null where the fold was not scored; all five null for a configuration that failed during selection. One that failed after it was chosen keeps the scores it had (D38) |
| `cv_mean` | number or null | mean of the scored folds; null for a configuration that failed during selection |
| `cv_std` | number or null | their sample standard deviation; null with fewer than two, and for a configuration that failed during selection |
| `chosen` | boolean | true for exactly one, which has status `fitted` |

**`warnings`**: a list of records `{code, column, message}` (2.5). May be empty.

**`limitations`**: the list of nine sentences of 5.4, exactly, in order.

### 5.3 Reconciliation rules

1. `rows.loaded` equals the rows of the source; it equals the four dropped
   counts plus training plus holdout.
2. Every source column appears exactly once in `columns`.
3. `predictions.csv` has one row per holdout row, and each `y_true` equals the
   target of that row in the source.
4. `headline.score` equals the metric recomputed from `predictions.csv`;
   `headline.skill` follows from it and the baseline as in step 11, and is
   null when the baseline's score is 0.
5. Every part of `ProfileResult` equals the matching part of `profile.json`.
6. The same seed gives an identical `profile.json` and `predictions.csv`, apart
   from `meta.timing` (and `meta.n_jobs` when `n_jobs` differs).

### 5.4 Fixed texts

The random-split caveat (`validation.caveat`), exactly:

> Random split: this number is optimistic if the rows are ordered in time or one entity appears on many rows.

The limitations statement (`limitations`), exactly these nine strings, in this
order, on every run whatever its scheme or task (8 is D27, 9 is D29):

1. `Predictability here means how well these model families did on this data under this validation scheme.`
2. `Low skill is not proof that the target cannot be predicted.`
3. `High skill is not proof of cause.`
4. `A random split is optimistic when rows are ordered in time or one entity appears on many rows.`
5. `Look-ahead inside the caller's own features cannot be detected.`
6. `Classification skill and regression skill are on different scales and are not comparable.`
7. `A CSV file is read with pandas' types, so text identifiers can lose leading zeros; pass a DataFrame to control the types.`
8. `Intervals are approximate: they tend to be too narrow when the holdout is short and, under time order, when errors are serially correlated.`
9. `Negative skill means the models did worse than the baseline on the holdout; read it as no evidence of predictability.`

---

## 6. Loading rules and types (D3, D22, D28)

**A CSV path** is read with `pandas.read_csv(path, low_memory=False)` and no
other argument. Consequences, all measured on pandas 3.1.1 (section 9):

- Blank lines are not rows. A blank line is one that is empty or holds only
  spaces and tabs: that is what pandas skips, between rows and before the
  header alike. A line of only commas is a row of missing values.
- **The header is the first line that is not blank (D39).** Leading blank
  lines are skipped when the header is read, as pandas skips them, so the tool
  and pandas agree on which line is the header. Any other first line is
  content, to pandas and to the tool: a line of two quotes, or of white space
  other than spaces and tabs (a form feed, a no-break space), is read as a
  header of one column. In a table of two columns or more the lines after it,
  the real header among them, then have more fields than the header:
  `input_error`, by the rule below.
- **A file that is not valid UTF-8 is an `input_error` wherever the first bad
  byte sits (D39):** in the header line, in the first data row, deep in a
  large file or in its last cell. An optional byte-order mark is ignored.
  pandas reads a file in pieces and an implementation that reads the header
  line itself meets a bad byte there first; every route ends in the same
  named error and none in a bare `UnicodeDecodeError`.
- The missing-value spellings of 4.4 load as missing. The tool cannot see what
  was written, so they are counted in `n_missing`, never in `n_coerced`.
- A cell such as `1,200` or `abc` in an otherwise numeric column makes the whole
  column load as text. When the text cells are of the `abc` kind, rule 5 of
  step 5 reads the column as numeric and counts those cells as coerced.
- A number written with thousands separators is never parsed: no `thousands`
  argument is passed, and the commas are not stripped afterwards. A column
  holding such values, throughout or in a single cell among plain numbers, is
  excluded by rule 4a of step 5 with the reason `numeric_with_separators` and
  a warning (D28, D34). A caller who wants the column used parses it and
  passes a DataFrame.
- **Duplicate column names stop the run** with `duplicate_columns`. pandas
  renames them silently (`a`, `a.1`), so the tool reads the header line itself
  (the first line that is not blank, UTF-8, an optional byte-order mark
  ignored) and compares the names before loading.
- A blank header cell is kept under pandas' name (`Unnamed: 3`) with a
  `blank_header` warning. Blank means empty. **A header cell that holds only
  white space is a name like any other (D42):** it is kept as it is written,
  with no warning, and two such cells of the same text are two columns of the
  same name, which stops the run with `duplicate_columns`. Names are compared
  as they are written: one space and two spaces are two names.
- Data rows with more fields than the header are an `input_error`. pandas does
  not report them when the first data row is the long one: it makes an index of
  the leading fields of every row and gives the header's names to the fields
  after them. A row number written in front of every row, with no name in the
  header, is lost in this way with nothing said; an extra field anywhere else
  puts each name to its left over its neighbour's values. **The index pandas
  returns cannot be used to detect this.** When the leading field is evenly
  spaced whole numbers (row numbers; 1, 4, 7; 3, 2, 1) the index is a
  `RangeIndex`, the type a well-formed file gives, and row numbers from 0 give
  the default index itself. A row longer than the first data row makes pandas
  raise `ParserError` (section 9; `docs/measurements/CP2`).

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
| no usable feature column | `no_features`; the message names each excluded column and its reason |
| two features that would give the preprocessed matrix one name (`colour=red` beside `colour` with the level `red`) | `feature_name_collision`, naming the columns, before any fit |
| an output folder that cannot be used (the path, or a folder above it, is a file) | `input_error`, before anything is loaded |
| a CSV that is not valid UTF-8, wherever the bad byte sits | `input_error` |
| blank lines (empty, or only spaces and tabs) before the header of a CSV | skipped, as pandas skips them; the header is the first line that is not blank |
| too few rows to split (fewer than 100 usable, or fewer than 10 of a class) | `too_few_rows` |
| a holdout, or the training rows, with one class | `one_class` |
| a fold with one class | that fold is not scored; `fold_not_scored` warning; `validation.folds[].scored` false; `fold_scores` null there. No fold scored: `one_class` |
| a time column not in an accepted form | `time_column_invalid` |
| a gap that leaves no training rows | `gap_too_large` |
| too few timestamps for the holdout and six blocks | `too_few_rows` |
| the fast profile on a panel too wide for its row cap | `fast_profile_too_few_timestamps` |
| duplicate column names, two header cells of the same white space among them | `duplicate_columns` |
| a blank header (an empty cell) | kept under pandas' name; `blank_header` warning |
| a header cell that holds only white space | a name like any other: kept as written, no warning |
| a categorical cell written with white space round it (`red `) | the same level as without it, in the counts, the leak screen and the matrix |
| too few holdout timestamps for the block bootstrap (fewer than 30) | interval fields null; `interval_reason`; `no_interval` warning |
| an interval from the block bootstrap (time order, 30 holdout timestamps or more) | reported, with the `interval_approximate` warning |
| negative skill | reported as measured, never clipped; sentence 9 of the limitations says how to read it |
| a regression baseline whose squared error on the holdout is exactly zero (a target flat across the holdout, under time order; a training mean that is the value of every holdout row, under the random scheme) | `headline.skill`, `skill_low` and `skill_high` null; one `skill_undefined` warning; the score, its interval and every baseline reported; no p-value, with a reason. Never an error, an infinity or a zero |
| a resample whose baseline error is zero | kept in the interval on the score, left out of the interval on skill; `n_resamples_skill_used` is the number kept for skill |
| skill that has a value and no interval (fewer than half of the resamples kept for skill, with the score's interval standing) | `skill_low` and `skill_high` null; one `no_skill_interval` warning |
| a feature whose single-column skill cannot be computed (a zero baseline error over the screen's pooled rows) | stays a feature; no `suspected_leak`, no `strong_single_column` |
| a model configuration that raises during selection | status `failed`, error text kept, all fold scores null, `model_failed` warning |
| the chosen configuration raising on the final fit or on the holdout | status `failed`, error text and fold scores kept, `model_failed` warning; the next best by `cv_mean` is chosen |
| every configuration failing, during selection or after | `all_models_failed` |
| infinite values in a feature | read as missing, counted in `n_infinite`, `infinite_values` warning |
| a text column written mostly as numbers with thousands separators | never parsed; excluded with reason `numeric_with_separators`; `numeric_with_separators` warning |
| a column of plain numbers with one cell or a few written with thousands separators | the same: excluded with reason `numeric_with_separators` and its warning, not read as numeric (D34) |
| a column of plain numbers with a few cells of other text (`abc`) | numeric; those cells read as missing and counted in `n_coerced`; `values_coerced` warning |
| a feature whose single-column skill is at least 0.90 and below 0.98 | kept as a feature; `strong_single_column` warning |
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
| F1 | A fitted step sees the holdout: the model refitted on all rows before scoring (large), or an imputer, encoder, scaler or the leak screen fitted before the split (small). | Gate: `model.noise_clf`, `model.noise_reg`, `model.noise_p_value` (large form). Tests: recording fakes through the seam, for the model and for a step the seam supplies (exactly); for the built-in preprocessing and the leak screen, which the seam cannot see inside, the guard tests of the table below. |
| F2 | A column that copies the target, or is only filled in after the outcome, is used as a feature. | Gate: `leak.excluded`, `leak.regression`, and `leak.near_leak_warned` (a build with no leak screen has no warning tier either, D30). Tests: the leak rule and its warning tier on crafted columns. |
| F3 | The headline is the best cross-validation score, not the holdout score. | Gate: `model.headline_recomputed`, `model.selection_full`. Tests: a fake family whose fold scores and holdout score differ by construction. |
| F4 | Accuracy is the headline on a rare class. | Gate: `model.metric_clf`, `model.headline_recomputed`. Tests: metrics by hand. |
| F5 | The task is misread from the column's type. | Gate: `task.binary_numeric`, `task.ambiguous_stops`. Tests: the rule on every loaded shape. |
| F6 | Something is lost without being reported: rows, a column, a model family. | Gate: `data.rows_reconcile`, `data.columns_accounted`, `data.coercion_reported`, `data.long_rows_stop`, `models.all_accounted`. Tests: accounting on crafted tables; a family that raises. |
| F7 | Time order is ignored: a shuffled split, a training row at or after the holdout, the gap not applied, or the time column misread. | Gate: `time.order_respected`, `time.no_skill_on_blocks`, `time.gap_applied`, `time.dayfirst_stops`. Tests: the split as a function of keys. |
| F8 | Under time order a trending or random-walk target is scored against the training mean only. | Gate: `time.persistence_baseline`. Tests: baselines by hand. One gate check and one test catch it; that is accepted (D43). |

### 8.1 Small faults the gate cannot see (D39)

The first review found five faults that a build could carry while passing
every gate check and all 308 tests of the first suite. Three are small forms of
F1. None moves a score on the gate's datasets by enough to leave a band, so no
gate check names them: each is caught by a test, and each test is proved by
running it against a copy of the reference implementation that carries that
one fault (`verify/mutation_check.py`; the switch is in the reference's
`_variant.py`).

| Id | The fault | Caught by (`tests/`) |
|---|---|---|
| M1 | The final fit's imputation and scaling statistics are learned from training and holdout rows together. | `test_guard_final_fit.py::test_m1_final_fit_statistics_ignore_the_holdout` |
| M2 | One-hot levels are learned from every row of the table in every fit. | `test_guard_final_fit.py::test_m2_levels_come_from_the_rows_of_each_fit` |
| M3 | The leak screen's trees are fitted on rows that include the holdout, and its missing-by-class rule counts every row. | `test_guard_leak_screen.py::test_m3_single_column_skill_is_fitted_on_fold_training_rows_only`, `::test_m3_missing_by_class_counts_training_rows_only` |
| M4 | The block bootstrap draws timestamps independently and still reports `bootstrap_blocks`. | `test_guard_uncertainty.py::test_m4_block_bootstrap_draws_whole_blocks` |
| M5 | The permutation p-value counts shuffles strictly greater than the observed skill. | `test_guard_uncertainty.py::test_m5_ties_count_in_the_p_value`, for classification only: for regression a tie is a tie only up to rounding (D43) |

The gate goes on catching the large forms; the fake goes on catching, exactly,
a model or a seam-supplied step that is handed a row it must not see.

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
| a header cell of only spaces | kept as the name `' '`; a second one is renamed `' .1'` | silent, like any duplicate name: such a cell is a name, not a blank (D42) |
| a text cell written ` red` or `red ` | text, with its spaces | pandas keeps them, so `red` and `red ` are two values until the tool strips them (D42) |
| blank lines | skipped | rows are not lines minus one |
| every data row one field longer than the header | the first field of each row becomes the index and the names go to the fields after it. The index is a `RangeIndex` when that field is evenly spaced whole numbers (1, 2, 3; 1, 4, 7; 3, 2, 1) and an `Index` otherwise (1, 4, 9; text; 0.5, 1.0, 1.5; a single row) | silent, with no error and no warning. **The index cannot be used to detect it:** a `RangeIndex` is what a well-formed file gives, and row numbers from 0 give the default index exactly. A leading row number is lost; a trailing extra field puts every name over its neighbour's values |
| a row longer than the first data row | `ParserError` | the first data row is the only long row pandas reads silently; a line of only spaces or tabs before it is skipped like a blank line |
| lines before the header (measured 2026-11-17) | an empty line and a line of only spaces and tabs are skipped. A line of a form feed, a vertical tab, a no-break space, an em space, two quotes or a quoted space is read as a header of one column; a line of only commas as a header of blank names | Python's `str.strip()` calls all of the first four white space, and pandas only spaces and tabs: a loader that strips with the default would take a different line for the header than pandas does |
| a byte that is not valid UTF-8 | `UnicodeDecodeError` from `read_csv`, when it reaches the byte | a loader that reads the header line itself meets a bad byte in the first few kilobytes before pandas does, outside any handler written around `read_csv` |

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

**Run time (D16, D25).** Measured once; no check enforces the budget (D31). A
full run on 100,000 rows × 100 columns with the grids of
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

**The interpreter's base (D32)** is `C:\Users\uahma\anaconda3\envs\py312`, a
local path: `mlprof_env\pyvenv.cfg` has it as `home`, and `sys.base_prefix`
reports it (confirmed 2026-11-16). If the interpreter does not start, that is
a harness error and never a code fault.

- **As it was on 2026-11-15.** The base was on `A:`, a drive mapped to the same
  folder through a loopback network share (`\\localhost\C$\...`), and
  interpreters started at the same instant blocked each other: two, four or
  eight tool processes started together all waited exactly 35 or 70 seconds
  and then finished together. Rounds of eight starts that blocked, by the
  spacing between starts: none, 6 of 6; 0.1 seconds, 2 of 2; 0.2 seconds, 3 of
  30; 0.3 seconds, 0 of 30; 0.5 seconds, 0 of 30. 35 seconds is the share's
  lock-break timeout. Earlier the same day 17 simultaneous starts did not
  block, so the behaviour came and went.
- **Measured again on 2026-11-16, with the local path.** Eight tool processes
  (the reference implementation, fast profile, on `signal_clf`) started at the
  same instant, within 0.2 to 1.2 milliseconds of each other, for 30 rounds,
  twice: **0 of 30 rounds blocked, both times.** Each process took 2.9 to 3.2
  seconds, except in the very first round, where all eight took 10.9 seconds.
  That is taken to be the first load of the libraries from disk: it is far
  from 35 seconds, and it did not recur in the second set of 30.
- The gate still starts its runs 0.5 seconds apart and gives a run that did
  not finish one more go. Both are kept as they are. Starting Python processes
  in parallel is no longer known to stall on this machine; running them one
  after another remains the simpler habit for a test helper.
- `pyvenv.cfg` still names the share in its `executable` and `command` lines.
  They record how the environment was made; the measurement above was taken
  with them in place.

---

## 10. Measured values

Set by the harness stage by measurement, for the owner's approval. Bands belong
to the gate (`verify/smoke_test.py`, where each is written beside its
measurements). Cut-offs belong to this spec: those measured on 2026-11-15 were
approved as D23; of the two added on 2026-11-16, the warning cut-off was
approved as D33 and the separator rule was settled by D34. Both are marked in
10.2.

The scripts, the seeds and a per-seed table behind everything measured on
2026-11-16 are in `docs/measurements/CP1/` (D35), with a README that says how
to run each measurement again. The seeds of 2026-11-15 were not recorded and
are not recoverable.

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

`leak.near_leak_warned` asserts no score: on `near_leak` the headline is the
near-copy itself (0.969 to 1.000 over 30 seeds, against 0.672 to 0.801 on
`signal_clf`), which is what its warning is for. Its margins are in 10.2.

`time.flat_holdout_reported` (D36, added 2026-11-17) asserts no band either,
since on `flat_tail` there is no skill to put in one. It recomputes both
baselines from the source (the previous value scores exactly 0, the training
mean about 300) and the score from `predictions.csv`, and requires a null
`skill`, one `skill_undefined` warning, an interval on the score that holds
the score, and 200 resamples used for the score against none for skill. The
reference as it stood before D36 failed it with a `skill` of 0.0, and the
first build with a `ZeroDivisionError`. Adding the dataset last moved nothing
else: between the two states of the judge, at the gate's seed, 18 datasets, 30
checks, 92 recorded values and every number of 23 runs are identical; what
differs in those runs is the text of the console summary and one new key,
`headline.n_resamples_skill_used` (`docs/measurements/CP3`).

**Re-measured on 2026-11-16, for D27 to D30 and again for D34.** The 30 data
seeds of the first measurement were not recorded, and 22 likely schemes failed
to reproduce its range on `messy`, so this measurement uses its own: the base
seeds 1115 + 100*k*, *k* = 0 to 29 (*k* = 0 is the gate's own data). The
reference was run on the same seeds, fast profile, in three states: as it was
on 2026-11-15; after D27 to D30; and after D34.

| Dataset | Reference, all three states: range (mean, sd) | Band (unchanged) |
|---|---|---|
| `signal_clf` | 0.672 to 0.801 (0.741, 0.033) | 0.54 to 0.91 |
| `leak` | 0.648 to 0.821 (0.751, 0.037) | 0.54 to 0.91 |
| `signal_reg` | 0.819 to 0.891 (0.848, 0.017) | 0.76 to 0.93 |
| `leak_reg` | 0.806 to 0.859 (0.840, 0.014) | 0.76 to 0.93 |
| `messy`, with `revenue` and `fee` | 0.670 to 0.826 (0.746, 0.035) | 0.55 to 0.91 |
| `noise_clf` | −0.115 to +0.159 (+0.004, 0.058) | −0.27 to 0.27 |
| `noise_reg` | −0.503 to −0.242 (−0.377, 0.076) | −0.65 to −0.05 |
| `imbalanced` | −0.150 to +0.376 (+0.017, 0.097) | −0.42 to 0.42 |
| `time_blocks` | −0.248 to +0.217 (−0.007, 0.115) | −0.48 to 0.48 |
| `random_walk` | −0.100 to +0.012 (−0.006, 0.021) | −0.30 to 0.05 |

The three states are identical on every seed: 390 runs (the ten above and
three more configurations, 30 seeds each) with no difference in the headline,
the row counts, the task, the validation record, the baselines, the models
table or the predictions, and 420 between the second state and the third,
which share `near_leak`. At the gate's own seed the same holds for all 19 runs
of the first state, the three full-profile ones included. What differs is the
new warnings and the two new limitation strings.

`messy` changed in three ways and its headline did not move. It gained
`revenue` and `fee`, which reach no model. And for D34 three text cells of
`spend` were reworded, `1,200` and `3,450` becoming `tbc` and `pending`: they
were coerced to missing before and are now. Over the 30 seeds its eleven
original columns hold 508,110 identical cells and 90 that differ as text and
read the same. The other fifteen datasets of the first state are
byte-identical, and every dataset's seed is what it was.

**No band's measurement changed, so no band was changed;** each still rests on
the first measurement, in the table above this one. Every value here is inside
its band. On this second set of seeds two bands are narrower than 4.5 sd on
one side: `noise_reg` (its lower edge is 3.6 sd below this mean) and
`imbalanced` (its upper edge is 4.2 sd above; the highest of 30 was 0.376).
D35 leaves both as they are, because a band changes only when it was measured
wrong and the gate judges one fixed seed. There the reference lands at −0.407
on `noise_reg`, 0.243 above the band's lower edge and 0.357 below its upper,
and at +0.072 on `imbalanced`, 0.492 above the lower edge and 0.348 below the
upper.

Two other measured margins:

- `model.noise_p_value`: the fast profile's p-value is a multiple of 1/201. On
  `noise_clf` the reference gave 0.010 to 0.965 and never the floor; F1 gave
  exactly 1/201 on all 30 seeds. The check requires at least 0.0075. A correct
  implementation lands on the floor by chance once in 201 (accepted, D31).
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
| leak warning: single-column skill (D30; measured 2026-11-16; approved as D33) | 0.90 | **0.90** | 30 data seeds. `outcome_noisy` of `near_leak`, the target with exactly 2% of its labels flipped: 0.938 to 0.960 (mean 0.949, sd 0.005); warned 30 of 30, excluded 0 of 30. The highest single-column skill of any legitimate column in any gate dataset: `strong_feature` of `leak`, 0.609 to 0.748 (mean 0.656, sd 0.032); the next is `f1` of `signal_reg`, 0.530 to 0.610; warned 0 of 30. The cut-off is 9.5 sd below the first mean and 7.7 sd above the second; 0.98 is 6.1 sd above the first. Where the tier's edges are, for a copy of a 0/1 target on 1,500 rows with an exact share of its labels flipped: 0.5%, 0.979 to 0.990 (excluded on 28 of 30); 1%, 0.965 to 0.980; 1.5%, 0.952 to 0.972; 2.5%, 0.925 to 0.947; 3%, 0.915 to 0.937 (each warned 30 of 30); 4%, 0.892 to 0.916 (22 of 30); 5%, 0.872 to 0.898 (never). Regression, the target rescaled with noise as a share of its deviation: 12%, 0.978 to 0.985 (excluded on 28 of 30); 15%, 0.968 to 0.977; 20%, 0.946 to 0.961; 25%, 0.920 to 0.938 (each warned 30 of 30); 30%, 0.888 to 0.918 (15 of 30); 35%, never. Recipes for the tests follow this table. |
| leak: missing by class | above 95% and below 5% | **the same** | 100%/0% and 97%/2%: flagged 30 of 30. 96%/4%: 26 of 30. 94%/3% and 97%/6%: 6 of 30. 90%/10%: never. |
| id-like, text | more than 95% of rows distinct (D19) | **the same** | 100% and 98% distinct: id-like. 70%: free text. 44% and 2%: features. |
| id-like, whole numbers | distinct, and consecutive or strictly ordered (D19) | **the same** | At 200, 2,000, 20,000 and 100,000 rows: salary, price in pence, volume, age and a distinct unordered measure are features at every size (the 95% rule called volume an identifier at every size, and salary and price at the smaller ones). Consecutive ids in order, in order with 1% blank, and shuffled: caught. Increasing non-consecutive ids: caught. **Not caught:** a random unordered identifier (as D19 says), and a shuffled consecutive identifier with any blank cell. **Wrongly caught:** a whole-number measure with every value distinct in a table sorted by that measure. Both are accepted limits (D31) and show in the column accounting. |
| free text | not given | **more than 50% of non-missing values distinct** | Needed to tell free text from categories; see the row above. |
| numeric coercion | at least 90% parse | **90%** | Only values of 1,000 or more carry a thousands separator, so a column with 50% of its cells written that way still parses 91% and passed this rule with 173 of 2,000 cells coerced; written that way throughout, 81% parse and the column was excluded as free text. That is how the rule behaved when it was measured. Since D34 neither column reaches it: both are excluded by rule 4a (next row). The rule now decides only columns whose text cells are of another kind. |
| numbers with separators (D28, D34; measured 2026-11-16) | 90%, counting the values that parse together with those that match the pattern | **the same, and at least one value that matches** | On `messy` over 30 data seeds. `revenue`, written with separators throughout: 17% to 22% of its values are below 1,000 and parse; 78% to 83% match the pattern; every value does one or the other; 99.8% to 100% are distinct. `fee`, plain numbers with six cells of 1,000 or more written with separators: 99.6% to 99.7% of its values parse, and 4 to 6 of the six cells are in rows with a target. Both are excluded with the reason `numeric_with_separators` and named by its warning on 30 of 30. The reference as it was on 2026-11-15, with no such rule, calls `revenue` `id_like` on 30 of 30; under the coercion rule alone `fee` would be numeric with its six largest values coerced. `spend`, whose eight text cells are words: numeric, with `values_coerced` and no `numeric_with_separators`, on 30 of 30. On 2,000 crafted rows, amounts mostly below 1,000: written with separators throughout (83% parse), excluded; with half the cells written that way (92% parse, 167 cells with a separator, all of them the amounts of 1,000 and more), excluded, where the coercion rule alone kept it. On 400 crafted rows, excluded with this reason: written with separators and every value distinct, 60% distinct, or five distinct values (without the rule: `id_like`, `free_text`, a categorical feature); plain numbers with one cell written `1,200`; plain numbers with 5% of the cells written with separators. Not this reason: `1,200` in every row, `constant`; the same amounts written `1.200,50` or `12,34`, `id_like` when all distinct and `free_text` at 60% distinct; 85% with separators and 15% words, `id_like`; plain numbers whose only text cells are words, numeric with `values_coerced` alone; plain numbers whose one `1,200` sits in a row with no target, numeric with that cell coerced. |
| date-like | 95%, fixed forms | **95%, the forms of 4.5** | The fixed list matches ISO dates and date-times, d/m/y, m/d/y, d-m-y, d.m.y, y/m/d and month-name forms (4 of 4 each), and none of `1,200`, `May`, `09:30`, `2024Q1`, `1/2`, `1.2.3`, `2024-01`, or impossible dates. pandas' lenient parser accepts the first five. |
| small holdout | fewer than 50 of the rarer class | **50** (regression: 50 holdout rows) | On pure noise, 95% of honest results fall within ±0.23 of zero with 50 rows of the rarer class in a balanced holdout (±0.16 at 5% positives), ±0.36 with 20, ±0.16 with 100. |
| too small to profile | fewer than 100 usable rows | **100**, and 10 of each class | 30 seeds per size. Random scheme at 100 rows: every run finishes; skill sd 0.17 (classification) and 0.06 (regression). Time order at 100 rows: fold 1 trains on 13 rows, every run finishes, no fold unscored, skill sd 0.16 and 0.08. Below the floor under time order, folds start to hold one class: 1 run in 30 at 80 rows, 4 at 60, 13 at 40. |
| no block-bootstrap interval | fewer than 20 holdout timestamps | **30** | See 10.3. |
| autocorrelated panel target | lag-1 autocorrelation above 0.8 | **0.5** | "The same as last period" beats the training mean exactly when the autocorrelation is above 0.5 (its skill over the mean is 2r − 1). Measured on simulated panels of 200 timestamps: at a measured autocorrelation of 0.45 that naive rule has skill −0.07; at 0.74 it has +0.49, and a cut-off of 0.8 warned on only 19% of such panels; at 0.85 it has +0.75. |

The rows measured on 2026-11-16 use 30 data seeds of their own (the base seeds
1115 + 100*k*, *k* = 0 to 29): the seeds of the first measurement were not
recorded. That is why `strong_feature` reads 0.60 to 0.71 in the first row and
0.609 to 0.748 in the second.

**Recipes for a column clearly inside the warning tier.** A test cannot measure
a column's single-column skill before the application exists, so these were
measured on the reference implementation, over the same 30 seeds. Each is on
**1,500 rows**, with a target of two roughly balanced classes (or, for
regression, a continuous target) and the rows to flip drawn without replacement
from a fixed seed. The tier is 0.90 to 0.98.

| Scheme | The column | Single-column skill: range (mean, sd) |
|---|---|---|
| random | a copy of the target with exactly 2% of the rows flipped (30 rows) | 0.938 to 0.960 (0.949, 0.005) |
| random | the same with exactly 2.5% (38 rows) | 0.925 to 0.947 (0.936, 0.005) |
| time order, unique timestamps | the same with exactly 2.5% | 0.927 to 0.961 (0.941, 0.008) |
| time order, unique timestamps | the same with exactly 2% | 0.939 to 0.969 (0.953, 0.008) |
| random, regression | the target plus independent normal noise whose deviation is 20% of the target's own (rescaling the result, as × 3.7 + 12, changes nothing) | 0.946 to 0.961 (0.953, 0.004) |

What does not sit clearly inside it: the 2% copy on 400 rows (0.918 to 0.967
under the random scheme, sd 0.013; up to 0.985 under time order); a copy in
which each row is flipped at a 2% chance, so that the share itself varies
(0.924 to 0.959 on 1,500 rows, sd 0.008); and any share from 4% up or from 1%
down. A column near 0.6 or below is clearly under the tier, and an exact copy
is clearly over it.

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
it is not something an implementation can fix. D27 keeps the method and has
the output say so: the `interval_approximate` warning on every block-bootstrap
interval (step 12), and sentence 8 of the limitations on every run (5.4).

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
  summary and `interval_reason` / `p_value_reason` (their presence is pinned).
  What a message must name is pinned in three places and nowhere else: the
  excluded columns and their reasons in `no_features` (2.4), the columns
  involved in `feature_name_collision` (step 5), and what the console summary
  shows (2.2);
- whether the preprocessing of the final fit is fitted again for a
  configuration tried after another one failed (step 10): each such try gets
  a fresh estimator, and that is all that is pinned;
- what `stages.skill("roc_auc", ...)` does with a baseline of 1, which no run
  can produce;
- how close two skills must be to count as a tie in the regression p-value
  (step 12; D43);
- what `stages.Preprocessor` does with white space round a level in a frame
  handed to it directly. `profile()` strips before the preprocessing sees a
  cell (step 5; D42), and that is what is pinned;
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
pattern of step 5 for a number with thousands separators, which rows' levels
decide a name collision (step 5), the preprocessing of step 8, the
single-column screen of step 8 and both of its cut-offs, the order in which
configurations are tried after selection (step 10), the percentile and block
rules of step 12 and which resamples each interval keeps, that a tie counts in
the p-value, the fixed texts of 5.4.

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
| D23 cut-offs approved as measured | 2.5, 3 (step 12), 4.3, 10.2, 10.3 |
| D24 leak rule under time order | 3 (step 8) |
| D25 pinned model settings | 3 (seeds and threads), 4.1, 9 |
| D26 step functions and the seam | 2.3, 2.6, 2.7 |
| D27 approximate interval under time order | 2.5, 3 (step 12), 5.4, 7, 10.3 |
| D28 numbers with thousands separators | 2.5, 3 (step 5), 4.3, 5.2, 6, 7, 10.2; its second warning form is replaced by D34 |
| D29 negative skill explained | 3 (step 11), 5.4, 7 |
| D30 warning tier below the leak cut-off | 2.5, 3 (step 8), 4.3, 7, 8, 10.2 |
| D31 accepted limits | 3 (step 8), 9, 10.1, 10.2 |
| D32 venv base path | 9 |
| D33 warning cut-off approved; three limits accepted | 2.5, 3 (steps 5, 8), 4.3, 10.1, 10.2 |
| D34 a partly formatted numeric column is excluded | 2.5, 3 (step 5, rule 4a), 4.3, 6, 7, 10.2 |
| D35 measurements kept in the repo | 10, 10.1; `docs/measurements/CP1/`, `CP2/`, `CP3/` |
| D36 skill when the baseline's error is zero | 2.5, 2.7 (`skill`), 3 (steps 8, 11, 12), 5.2, 5.3, 7, 10.1; the part about resamples is refined by D40 |
| D37 colliding feature names stop the run | 2.4, 2.7 (`Preprocessor`), 3 (step 5), 7, 11 |
| D38 a configuration that fails after it has been chosen | 2.4, 2.5, 2.6, 3 (steps 9, 10, 11), 5.2, 7, 11 |
| D39 first review accepted | (1) the five test holes: 2.6, 8, 8.1; `verify/mutation_check.py`. (2) the output folder: 2.1, 2.4, 7. (3) process settings: 2.1. (4) the no-features message: 2.4, 3 (step 8), 7. (5) the console summary: 2.2, 11. (6) UTF-8 and leading blank lines: 2.4, 3 (step 1), 6, 7, 9. (7) the built library timed and compared: not in this spec; `docs/measurements/CP3/` |
| D40 the score interval is kept when skill is undefined | 3 (step 12), 5.2 (`n_resamples_skill_used`), 7, 10.1 |
| D41 README examples use a table kept in the repo | 1; `examples/customers.csv` |
| D42 three small rulings | (1) `no_skill_interval`: 2.2, 2.5, 3 (step 12), 5.2, 7. (2) white space is not part of a level: 2.6, 3 (steps 5, 8), 7, 9, 11. (3) a header cell of white space is a name: 2.4, 2.5, 6, 7, 9 |
| D43 limits accepted after CP3 | (1) the p-value's tie in regression: 3 (step 12), 8.1, 11. (2) F8: 8. (3) fold scores lost by a failure during selection: 3 (step 9). (4) a collision outside the rows or columns used: 3 (step 5). (5) undefined skill under the random scheme: 3 (step 11) |
