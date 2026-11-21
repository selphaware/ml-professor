# ML Professor, phase 1: the contract

This file is what the application is built from and what the tests are written
against. It restates the owner's decisions (`docs/DECISIONS.md`, cited as D1 to
D60) without changing them, pins every value they leave to this stage, and
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
regression, a random or a time-ordered validation scheme, a declared entity
column that keeps the rows of one entity together (D57), one headline number
with its uncertainty, and a complete account of every row and column.

**The tool stops and asks where it has reason to think the headline would be
inflated (D58).** A table with a date column and no word from the caller about
time order, a panel with no word about its entities, and holdout rows that are
copies of training rows each end in a named error that says what the caller
can pass. A headline of that kind is printed only when the caller has said, in
the call, that the reason does not apply.

Not in phase 1, and not to be built: joins or several tables, multiclass,
unsupervised methods, features made from dates or text,
an HTML report, any user interface, Parquet or Excel input, packaging for
`pip install`. pandas, numpy, scikit-learn and threadpoolctl are the only
third-party imports. No network access at run time.

The package lives at `E:\CODE\ml-professor\mlprofessor` and is run from the
repo root. It is never installed into the interpreter.

The repo also carries one small synthetic table, `examples/customers.csv`, for
the commands the README shows (D41). The README names no file under
`verify/` or `tests/`: those folders belong to the judge, and part of
`verify/` exists only after the gate has run. Where the README describes the
fast profile's cap of 2,000 rows it says which rows are kept under each
scheme, as step 6 has it: drawn at random from the seed when there is no time
column, the most recent whole timestamps when there is one (D46). Its wording
is the builder's.

---

## 2. Interface (D2, D17)

### 2.1 The function

```python
mlprofessor.profile(data, target, task="auto", time_col=None, gap=0,
                    profile="full", seed=0, out_dir=None, n_jobs=-1,
                    duplicates="error", group_col=None) -> ProfileResult
```

These eleven parameters, in this order and with these defaults, are the whole
signature (D49, D54, D57). The first ten are as they were before `group_col`.

| Argument | Meaning |
|---|---|
| `data` | A pandas DataFrame, or a path (`str` or `os.PathLike`) to a CSV file. A DataFrame is never modified. |
| `target` | Name of the target column. |
| `task` | `"auto"`, `"classification"` or `"regression"` (D4). |
| `time_col` | Name of the time column, `None`, or `False`. Never inferred (D7). A name: the time-ordered scheme is used. `None`: nothing is said about time order; the rows are split at random, unless a column holds dates, in which case the run stops and asks (step 5; D58). `False`: the caller's statement that the order of the rows in time does not matter; the rows are split at random whatever the columns hold, and the statement is recorded in `input.time_col`. `True` is an `InputError`: it names no column and states nothing. |
| `gap` | A whole number of distinct timestamps, at least 0 (D7). A gap above 0 without a time column (`time_col` is `None` or `False`) is an `input_error`. |
| `profile` | `"full"` or `"fast"` (D12). |
| `seed` | A whole number from 0 to 4,294,967,295 (2³² − 1), the range scikit-learn takes as a seed (D45). It drives the row cap, the split, the folds, every model and every resample. A larger seed is never folded down, since two seeds would then give one run: it is an `InputError`, like a negative one. The largest allowed seed runs. |
| `out_dir` | A folder, or `None`. When given it is created if missing and `profile.json` and `predictions.csv` are written into it. It cannot be used when the path, or the nearest folder above it that exists, is something other than a folder (it names an existing file, for one): an `InputError` like any other invalid argument, found without creating anything (D39). |
| `n_jobs` | `-1` for every core (`os.cpu_count()`), or a positive whole number (D17). |
| `duplicates` | `"error"`, `"keep"` or `"group"` (D49, D57, D60). What happens, when there is no time column, to rows that are copies of one another (step 8a). `"error"`: the run stops once too many holdout rows have a copy among the training rows. `"keep"`: it goes ahead and the output says so. `"group"`: before the split, each set of rows that are copies of one another is kept on one side of the holdout split and of every fold (step 7), and the output says how many rows that were. Under the time-ordered scheme it changes nothing. Any other value, a value that is not text among them (`True`, `None`), is an `InputError`. |
| `group_col` | Name of the entity column, `None`, or `False` (D57, D58). An entity is whatever several rows can belong to: a customer, a patient, a device, one series of a panel. A name: with no time column, the rows of one entity stay on one side of the holdout split and of every fold (step 7); with a time column the split is by time as ever, and each entity has its own previous value (step 11). The column is not a feature. `None`: nothing is said about entities. `False`: the caller's statement that there are no entities, which a panel under time order asks for (step 11); it is recorded in `input.group_col` and changes nothing else. Any other value that is not text (`True`, a number) is an `InputError`, as is a `group_col` that equals `target` or `time_col`. |

Any other value of an argument raises `InputError`, before anything is loaded:
a table that would itself raise an error on loading (duplicate column names,
for one) still gives the `InputError` of the argument. Nothing is written when
an error is raised.

**Process-wide settings are left as the caller had them** (D39), whether
`profile()` returns or raises; the same holds for every function of `stages`.
That covers `csv.field_size_limit()`, the thread limits `threadpoolctl`
reports, the `warnings` filters, numpy's error state, the working directory
and the environment. A test checks the first two: the others can be moved by a
library's first import, which is not the tool's doing.

### 2.2 The command line

```
python -m mlprofessor profile --data FILE.csv --target COLUMN --out FOLDER
       [--task auto|classification|regression]
       [--time-col COLUMN | --no-time-order] [--gap N]
       [--profile full|fast] [--seed N] [--n-jobs N]
       [--duplicates error|keep|group]
       [--group-col COLUMN | --no-entities]
```

It is a thin wrapper: it calls `profile(...)` with `out_dir=FOLDER` and prints a
short plain-text summary to standard output. Defaults are those of the function.
`python -m mlprofessor --help` exits 0. `--duplicates` takes the three values of
the function's `duplicates` and no other: anything else is refused when the
command line is parsed, exit 2, as for `--task` and `--profile` (D49).

`--no-time-order` is `time_col=False` and `--no-entities` is `group_col=False`
(D58): each takes no value. `--time-col` and `--no-time-order` cannot be given
together, and nor can `--group-col` and `--no-entities`: either pair is
refused when the command line is parsed, exit 2, and nothing is written. With
neither of a pair, the function's `None` is passed.

| Exit code | When |
|---|---|
| 0 | The run finished and both files were written. |
| 2 | The command line could not be parsed (argparse's own behaviour). |
| 3 | A named error (2.4). Standard error carries one line `error: <code>: <message>`, and nothing else: no traceback, and no warning of a library beside it. No file is written. |

Any other failure is a defect and exits with the interpreter's own non-zero code.

An argument the function refuses is refused the same way here. `--seed
4294967296` is parsed (it is a whole number), then refused by `profile()`:
exit 3, the one line `error: input_error: ...`, no traceback (D45).

The summary's wording and layout are open. What it shows is pinned (D39, D46):

- the headline's `score` and `baseline_score`, each as a number, and its
  `skill` as a number unless it is null;
- the interval on the score, `interval_low` and `interval_high`, and the
  interval on skill, `skill_low` and `skill_high`, each as a number; when
  there is no interval (`interval_method` is null), the text of
  `headline.interval_reason` in their place;
- `p_value` as a number, or, when it is null, the text of
  `headline.p_value_reason`;
- one line per warning, holding that warning's `code` and, when it has one,
  its `column`. Two warnings never share a line.

**A value counts as shown when some number printed in the summary agrees with
it to four significant digits (D46):** the printed number, read as a decimal
number with its exponent if it has one (`9.371e-09`), is within 0.0005 × |value|
of the value in `profile.json`. The tolerance is relative, so it holds at any
scale: a squared error of 9.37e-09 printed as `0.0000` is not shown, and nor
is a p-value of 0.004975 printed as `0.0050`. A value of exactly zero is shown
by a printed number that is zero (`0`, `0.0000`). How the numbers are
formatted is open; four significant digits is the least, and more is allowed.

Until D46 the tolerance was 0.0005 × max(1, |value|), under which every value
below 0.0005 could print as zero. The rule above is the stricter of the two at
every value, so a summary that meets it meets the old one as well.

A null `skill`, `skill_low` or `skill_high` needs no number; the line of the
`skill_undefined` or the `no_skill_interval` warning covers it.

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

**The constructor is pinned (D57): `ProfileResult(document, predictions)`.**
`document` is a `dict` with the parts of `profile.json` (section 5.2), and
`predictions` a DataFrame with the columns of `predictions.csv`. It is how
`profile()` makes its result, and a test may call it. A result made from the
`to_dict()` and the `predictions` of another equals that other in every
member. **No result is ever made from a document that holds a float that is
infinite or not-a-number, in any part of it:** the constructor raises
`NonFiniteScoreError` ("Numbers that are not finite", rule 6). So such a number
cannot leave as a `null` from `task`, `columns`, `input`, `rows`, `meta` or
`warnings` any more than from the parts that hold the scores (the fourth
review, finding 4).

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
| `InputError` | `input_error` | An argument has an invalid value, a `seed` outside 0 to 4,294,967,295 among them (D45); `out_dir` cannot be used (2.1); `data` is neither a DataFrame nor a path; the file is missing, empty, not valid UTF-8 or not parseable as CSV; data rows have more fields than the header; `gap` above 0 without a time column; `time_col` equals `target`; `time_col` is `True`; `group_col` is neither a name, `None` nor `False`; `group_col` equals `target` or `time_col`. |
| `ColumnNotFoundError` | `column_not_found` | `target`, `time_col` or `group_col` is not a column. |
| `DuplicateColumnsError` | `duplicate_columns` | Two columns have the same name (D22). A name that holds only white space is a name like any other: two header cells of one space each are two columns of the same name (D42). |
| `TargetConstantError` | `target_constant` | The target has fewer than two distinct values among the usable rows. |
| `TaskAmbiguousError` | `task_ambiguous` | `task="auto"` and the rule cannot decide (D4). |
| `TaskUnsupportedError` | `task_unsupported` | A declared task the target does not support: classification with other than two distinct values, regression on a non-numeric target. A complex-number target is not numeric (step 4; D48). Also, whatever task was asked for: a target that holds whole numbers too large for 64 bits (D50), and a target whose values cannot be compared for equality, such as lists, dictionaries or sets, so that its distinct values cannot be counted (D53). |
| `TimeColumnError` | `time_column_invalid` | The time column is not in an accepted form (D18). That includes a column of whole numbers too large for 64 bits (D50), a complex-number column, and a datetime or ISO 8601 column holding a date that a nanosecond timestamp cannot hold (step 3). |
| `TooFewRowsError` | `too_few_rows` | Fewer than 100 usable rows; fewer than 10 usable rows of a class; under time order, too few distinct timestamps to form the holdout and six blocks. |
| `FastProfileRowsError` | `fast_profile_too_few_rows` | Under `profile="fast"`, the row cap left rows out and the rows it kept are fewer than 100, or hold fewer than 10 of a class, **and the same call with `profile="full"` would get past that point and form its split** (step 6; D52, D53). The message names the cap and tells the caller to use `profile="full"`. When the full profile would stop too, this error is not raised: the error the full profile would raise is. It has a code of its own because no timestamp need be involved: it arises under the random scheme as well. |
| `FastProfileError` | `fast_profile_too_few_timestamps` | Under `profile="fast"` with a time column, the rows kept by the row cap cannot form the split (the most recent timestamp alone is over the cap, or the rows kept hold too few distinct timestamps for the holdout and six blocks), **and the same call with `profile="full"` would get past the split** (D20, D46). The message tells the caller to use `profile="full"`. When the full profile would stop there too, this error is not raised: the error the full profile would raise is, `too_few_rows` or `gap_too_large` (step 7). |
| `GapError` | `gap_too_large` | The gap leaves fewer than six training timestamps, or no training rows before a validation block. |
| `FeatureNameCollisionError` | `feature_name_collision` | Two feature columns would give the preprocessed matrix a column of the same name (step 5; D37). The message names every column involved. |
| `NoFeaturesError` | `no_features` | No column has the role feature after the leak screen. An error returns no column accounting, so the message carries it: the name of every excluded column and, for each, its reason as `columns[].reason` spells it (D39). |
| `HoldoutCopiesError` | `holdout_copies_in_training` | With no time column and `duplicates` other than `"keep"`: more than 1% of the holdout rows, and at least 10 of them, have a copy among the training rows (step 8a; D49, D53). The message gives the count and the share, and says that the caller can remove the repeated rows, pass `duplicates="group"` or pass `duplicates="keep"`. When a column was excluded as `datetime` (the caller having passed `time_col=False`), the message also names that column and `time_col`. With `duplicates="group"` it is raised only when the leak screen excluded a column after the sets of copies were formed (step 8a), and the message then names the excluded column or columns. |
| `TimeOrderUndeclaredError` | `time_order_undeclared` | `time_col` is `None` and a column would be excluded with the reason `datetime` (step 5; D58). The message names every such column and the two ways on: `time_col` (`--time-col`) with a column's name, or `time_col=False` (`--no-time-order`). |
| `EntitiesUndeclaredError` | `entities_undeclared` | Regression under time order with repeated timestamps, `group_col` is `None`, and the lag-1 autocorrelation of the per-timestamp mean of the target over the training rows is above 0.5 (step 11; D58). The message names the two ways on: `group_col` (`--group-col`) with a column's name, or `group_col=False` (`--no-entities`). |
| `GroupColumnError` | `group_column_invalid` | The entity column cannot be used as it is (D57): it is empty in a row that has a target (and, under time order, a time value); it holds values that cannot be compared for equality (lists, dictionaries, sets); or, in a regression under time order, two rows of one entity have the same timestamp. |
| `TooFewGroupsError` | `too_few_groups` | With an entity column and no time column, or with `duplicates="group"` when copies were found: the rows split hold fewer than six entities, or holding a quarter of the rows out as whole entities leaves fewer than five for the folds (step 7; D57). |
| `OneClassError` | `one_class` | Classification, and the holdout or the training rows hold one class, or no fold can be scored. |
| `AllModelsFailedError` | `all_models_failed` | Every configuration failed, during selection or after it was chosen (steps 9 and 10). The message carries each error text. |
| `NonFiniteScoreError` | `non_finite_score` | A number the result rests on is infinite or not-a-number (D44; section 3, "Numbers that are not finite"): a baseline's score on the holdout; a baseline's error, or a column's own, in the leak screen; the chosen configuration's score on the holdout; the headline skill; or, when the result is made, any number in any part of the output (2.3; D57). The message names the number and the likely cause. It is not raised for a configuration's prediction or fold score: that configuration is failed (step 9). |

An invalid argument is reported before anything is loaded. Beyond that, when a
table breaks several rules at once, which of the named errors is raised is open,
with four exceptions. A regression target too large for squared error stops with
`non_finite_score` and never with `all_models_failed` ("Numbers that are not
finite", rule 1). The rule on copies is judged before any model is fitted
(step 8a), so a table that breaks it stops with `holdout_copies_in_training`
and never with `all_models_failed`. A date column with nothing said about time
order is found at step 5, before the split, so such a table stops with
`time_order_undeclared` and never with `holdout_copies_in_training`; once the
caller has passed `time_col=False`, the copies error can follow, and its
message names the date column (D58). And a panel with nothing said about
entities stops with `entities_undeclared` before the leak screen and before
any model is fitted (step 11).

**No input ends in an error that is not named (D52).** Whatever type a column
has and whatever values it holds, in any role, `profile()` returns a result or
raises a subclass of `ProfileError`. A `TypeError`, an `OverflowError` or any
other exception of Python or of a library that reaches the caller is a defect.
Section 7, "The type matrix", lists the kinds of column and the values a test
tries.

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
| `date_column_undeclared` | the column | One per column excluded as `datetime` when the caller passed `time_col=False` (D7, D58). With `time_col=None` such a column stops the run instead (step 5), so the warning is only ever seen beside the caller's own statement. |
| `autocorrelated_panel_target` | the target | Regression under time order with repeated timestamps, the caller passed `group_col=False`, and the lag-1 autocorrelation of the per-timestamp mean of the target over the training rows is above 0.5 (D10, D23, D58; section 10). With `group_col=None` that condition stops the run instead (step 11); with an entity column it does not arise. |
| `no_previous_value` | `null` | Regression under time order with an entity column: exactly one when at least one holdout row belongs to an entity that has no row at least `gap + 1` timestamps earlier (step 11; D57). For those rows the previous-value baseline predicts the training mean; the count is `validation.n_holdout_without_previous`. |
| `copies_grouped` | `null` | Exactly one, with `duplicates="group"` and no time column, when at least one row has a copy among the rows split (step 7; D57, D60). It says that each set of copies was kept on one side; the count is `validation.n_rows_with_copies`. Never when no row has a copy. |
| `model_failed` | `null` | One per configuration with status `failed`, whether it failed during selection or after it was chosen (D38), and whether it raised, predicted a value that is not finite or scored a fold with a number that is not finite (D44). |
| `values_coerced` | the column | One per numeric column in which text cells were turned into missing (D22). |
| `numeric_with_separators` | the column | One per column excluded with the reason `numeric_with_separators` (step 5, rule 4a; D28, D34). It has this one form: a column that is kept never carries it. |
| `whole_number_regression_target` | the target | `task="auto"` picked regression and every target value is a whole number (D21). |
| `blank_header` | pandas' name for it | One per CSV column whose header cell is blank, which means empty (D22). A cell that holds white space is a name, and gets no warning (D42). |
| `infinite_values` | the column | One per numeric feature holding infinite values, or finite values too large for the tree models: a magnitude above 3.4028234663852886e+38 (step 5; D48). Both are read as missing and counted in `n_infinite`. |
| `duplicate_rows` | `null` | The loaded table holds exact duplicate rows. They are kept. This warning is about the table as loaded, every column of it; whether the run may go ahead is decided by step 8a, on the split, and has the code below. |
| `holdout_copies_kept` | `null` | Exactly one, with no time column and `duplicates="keep"`, when the rule of step 8a is met: more than 1% of the holdout rows, and at least 10 of them, have a copy among the training rows (D49, D53). It tells the caller that the headline may be inflated; the count and the share are in `validation`. Never otherwise: not below the rule, where the count is reported in `validation` and nothing more is said, not with `duplicates="error"` or `"group"`, and not under time order. |
| `unsupported_type` | the column | One per column excluded with the reason `unsupported_type` (step 5, rule 7; D51), a column of whole numbers too large for 64 bits among them (D50). So that no column a caller meant as a feature is left out without a word: the code shows in the console summary like any other. |
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
  for regression `predict(X)`. What a model hands back is read as one finite
  number for each row: `predict_proba` must return an array of shape (rows, 2),
  and `predict` the rows' values in any shape that holds exactly that many (a
  column of n values is n values). Anything else, a value that is not finite
  among them, fails the configuration exactly as an exception from `fit` or
  from the prediction would (step 9).
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
  columns as floats with NaN for missing (an infinite cell and one too large
  for the tree models are missing: step 5; D48); categorical columns as text, each
  cell its level, which is the text without the white space round it (step 5;
  D42), with `"(missing)"` for missing. So a step never sees `red ` or ` red`,
  only `red`. A column of generic objects that holds only numbers is a numeric
  column like any other here: floats (step 1a; D51). An estimator receives the
  fully preprocessed matrix, in which a level's column is named by that same
  stripped text (`colour=red`).
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
| `usable_rows(frame, target, time_col=None)` | Step 2 as a count: `{"usable", "dropped_missing_target", "dropped_missing_time"}`. `usable` is an array of the positions of the usable rows in increasing order; the two counts and its length add up to the rows of `frame`. A missing target is a missing cell, an infinite number, or a text cell that holds only white space (step 2; D48). A missing time value is a missing cell (or an infinite number); the time column's form is not checked here. | `ColumnNotFoundError` |
| `column_roles(frame, target, time_col=None, group_col=None)` | A DataFrame with the eleven columns of `profile.json["columns"]` (5.2), one row per column of `frame` in order, as decided by step 5 alone: `leak_rule` is null and `n_imputed` is 0 throughout, since the leak screen and imputation come later. Usable rows are those with a target and, when `time_col` is given, a time value. `group_col`, when it names a column, gives that column the role `group` (D57). It returns no warnings and never stops for an undeclared date column: the warnings of step 5 (`values_coerced`, `infinite_values`, `numeric_with_separators`, `unsupported_type`) and the error `time_order_undeclared` are raised by `profile()`. | `ColumnNotFoundError` |
| `decide_task(values, task="auto")` | `{"type", "decided_by", "n_distinct", "classes", "positive_class", "warnings"}` for the target values given. They are read as step 4 reads them: a text value without the white space round it (D46); missing ones are ignored, a text value of only white space among them (D48). `warnings` is a list of warning codes: `["whole_number_regression_target"]` or `[]`. | `TargetConstantError`, `TaskAmbiguousError`, `TaskUnsupportedError`, `InputError` |
| `read_time(values)` | `(keys, form)`. `keys` is a numpy array, one per value, that sorts in time order: `datetime64[ns]` with `NaT` for a missing value under the `datetime` and `iso8601` forms (in UTC if the values carried a zone), `float64` with NaN for a missing or infinite value under the `numeric` form. One exception under the `numeric` form (D50, D53): for whole numbers held exactly (step 3) of which at least one has a magnitude above 2⁵³, `keys` are exact 64-bit integers, of type `int64`, or `uint64` when the values came in an unsigned type, each equal to its value; as floats two of them could become one key. What `keys` holds for such a column that also has a missing value is open: a caller passes `split_time` the keys of usable rows only. `form` is as in `validation.time_form`. | `TimeColumnError` |
| `split_random(n_rows, classes=None, seed=0, groups=None)` | `{"training", "holdout", "folds"}`: arrays of positions in 0..`n_rows`−1, each in increasing order; `folds` is a list of five `{"training", "validation"}`. `classes`, when given, is an array of `n_rows` labels with two distinct values and makes the split stratified (step 7); text labels are compared without the white space round them, so `yes` and `yes ` are one label and the split is the one their stripped forms give (D46). `seed` is in the range of 2.1. `groups`, when given, holds one value for each row, none of them empty, read as an entity column is (step 5): the rows of one value stay together, on one side of the holdout and in the validation part of one fold, by the rule of step 7 for whole entities (D57). The split depends on which rows share a value and on nothing else about the values: renaming them changes nothing. With `groups=None` the result is what it was before `groups` existed. | `InputError` for a seed outside 0 to 4,294,967,295 (D45); with `groups`: `GroupColumnError` for an empty value or a number of values other than `n_rows`, `TooFewGroupsError` (step 7) |
| `split_time(keys, gap=0)` | `{"training", "gap", "holdout", "folds"}`: arrays of positions into `keys`, each in increasing order; `folds` is a list of five `{"training", "gap", "validation"}`, fold *k* validating on block *k*+1. `keys` is as returned by `read_time`, in any order, with no missing value. | `TooFewRowsError`, `GapError` |
| `roc_auc(y, score)` | ROC AUC of `score` for labels `y` of 0 and 1 (1 is positive); ties share their average rank. | `OneClassError` |
| `mse(y, pred)` | Mean squared error. | |
| `skill(metric, score, baseline_score)` | For `"roc_auc"`: (score − baseline) ÷ (1 − baseline), which is 2 × AUC − 1 at the baseline 0.5. For `"mse"`: 1 − score ÷ baseline; and `None`, not an error, when `baseline_score` is 0, where skill has no value (D36). | `ValueError` for any other metric |
| `previous_value(y, keys, gap=0, groups=None)` | An array aligned with `y`: for each row the target of the row `gap + 1` timestamps earlier, NaN where there is none. With `groups`, one value for each row: the target of the latest row **of the same group** that lies at least `gap + 1` distinct timestamps earlier, the timestamps being those of all the rows given, whatever their group; NaN where the group has no such row (step 11; D57). | `ValueError` if `keys` are not unique; with `groups`, if a value of `groups` is empty or two rows of one group share a timestamp |
| `Preprocessor(kinds)` | The built-in preprocessing of step 8 for one fit. `kinds` maps a column name to `"numeric"` or `"categorical"`. `fit(frame)` learns from the rows given and returns the object. `transform(frame)` returns a DataFrame with the same index: one column per numeric column, under its own name, then for each categorical column one column per kept level named `<column>=<level>`, levels in sorted order, holding 0.0 and 1.0. A missing categorical cell is the level `(missing)`. A numeric cell that is infinite, or too large for the tree models (step 5), is a missing cell to it, in `fit` and in `transform` alike, so every statistic it learns and every value it returns is finite (D48). No two columns of the result ever share a name (D37). | `FeatureNameCollisionError` from `fit`, when two of the names it would give are the same, judged on the levels it keeps from the rows given |

`split_time` raises `TooFewRowsError` when there are too few timestamps;
`profile()` turns that into `FastProfileError` when the fast profile's row cap
was what left too few and the full profile would get past the split (D20,
D46; step 7).

Every function of this table that is handed a column reads it as `profile()`
does. In particular a column of generic objects that holds only numbers is read
as numbers (step 1a; D51): by `usable_rows`, `column_roles`, `decide_task` and
`read_time` alike. `column_roles` reports the type the column came with, as
`columns[].dtype` does. `split_time` and `previous_value` take the integer keys
of `read_time` as they take its float keys, and never turn them into floats.

How `profile()` uses the two split functions is pinned, so that a test can
predict a run's rows. Take the usable rows that remain after the row cap, in
file order, and call them 0 to *n*−1. Under the random scheme the run's holdout
rows and folds are `split_random(n, classes, seed)` applied to them, `classes`
being their target values, as step 4 reads them, for classification and `None`
for regression. Under
the time-ordered scheme they are `split_time(keys, gap)`, `keys` being those
rows' keys from `read_time`, with or without an entity column. Under the
grouped scheme (step 7) they are `split_random(n, classes, seed, groups)`,
`groups` holding for each of those rows the value of its entity column; under
`duplicates="group"` it holds instead a label for the set the row belongs to,
two rows having one label when they are copies of one another or of one entity,
directly or through other rows. What the labels are does not matter. For the
grouped scheme the properties of step 7 are pinned and the row numbers are
not: a test may compare `profile()` with `split_random`, and may not expect
particular rows.

---

## 3. Methodology, in the order it happens

Every numbered step below is in the order of the brief. **From step 8 on, every
fitted step sees training rows only.** A reader should be able to decide from
this section alone whether an implementation leaks.

**1. Load the table (D3, D22); count the rows.** Section 6 gives the loading
rules. `loaded` is the number of rows of the table: for a CSV, the data lines,
not counting blank lines. A blank line is one that is empty or holds only
spaces and tabs (section 6).

**1a. Read number columns of generic objects as numbers (D51).** A database
driver returns a column of numbers as Python objects: `decimal.Decimal`,
`True`/`False` with `None`, whole numbers with `None`. pandas gives such a
column the type `object`. Until D51 it was excluded as `unsupported_type` with
nothing said; on the third review's table that cost the run its signal (skill
−0.011, against 0.779 with the same values in typed columns). Now, before any
other step, each `object` column is looked at once, over all its loaded rows,
and read by what its non-missing cells are:

| Every non-missing cell is | The column is read as |
|---|---|
| a boolean (`bool`, `numpy.bool_`) | a boolean column |
| a whole number held exactly: a Python or numpy integer, or a `Decimal` with no fractional part (`Decimal("7")`, `Decimal("1E+3")`), and all of them fit in 64 bits (all within −2⁶³ to 2⁶³−1, or all within 0 to 2⁶⁴−1) | a whole-number column, every value exact |
| a whole number held exactly, as above, and they do **not** all fit in 64 bits | **whole numbers too large for 64 bits** (D50): not a number column; see below |
| a number: a whole number as above, a float (`float`, `numpy.floating`) or any `Decimal`, with at least one that is not a whole number held exactly | a column of 64-bit floats, each value converted with `float()`; a value beyond the largest double becomes infinite, with its sign |
| anything else, **including a boolean beside a number** (`True` and `3`), text beside a number, a complex number | left as it is: steps 3, 4 and 5 treat it as they treat any object column |

- **Each column is read once, here, and every later step goes by that reading
  (D57; the fourth review, finding 1).** The table above is applied to all of
  the column's loaded rows, whichever rows step 2 will drop. A cell that
  decides what the column is counts wherever its row ends up: a text cell
  among whole numbers, or a whole number too large for 64 bits, in a row that
  has no target or no time value changes the column exactly as it would in a
  usable row. So the run ends in the same named error either way, and never in
  a result that silently left the cell out. Five tables, each a test:

  | The column | The odd cell | Its row | The run ends in |
  |---|---|---|---|
  | an object target of whole numbers | a text cell | has no time value | the error the same cell gives in a usable row: `task_ambiguous` under `task="auto"`, the column being text beside numbers |
  | an object target of whole numbers | a whole number too large for 64 bits | has no time value | `task_unsupported` (step 4) |
  | an object time column of whole numbers | a text cell | has no target | `time_column_invalid` (step 3), as in a usable row |
  | a time column of Python integers | one too large for 64 bits | has no target | `time_column_invalid` |
  | an `int64` time column | its one value above 2⁵³ | has no target | a result, with exact integer keys: `train_end`, `holdout_start` and `holdout_end` are written as whole numbers (step 3 judges this over the loaded values) |

- **An empty cell stays empty.** `None`, NaN, `pd.NA`, `NaT`, and a `Decimal`
  not-a-number of either kind (`Decimal("NaN")`, `Decimal("sNaN")`) are
  missing cells, and are not looked at to decide the column. pandas itself
  raises on a signalling not-a-number, so the tool must not hand one to it.
- The column is then **the same as a typed column holding those values**, in
  every role and in every step: as a feature (step 5, rule 3), as the target
  (step 4), as the time column (step 3), in `stages` (2.7), and in what a
  seam-supplied step receives (2.6). A table read straight from a database
  gives the result its typed twin gives, number for number.
- **The output reports the type the column came with:** `columns[].dtype` and
  `task.target_dtype` say `object`. The values of the target are written as
  they are read: `y_true` and `task.classes` hold numbers or booleans.
- A `Decimal` infinity becomes an infinite float, and is then what an infinite
  float is in that role: a missing cell that `n_infinite` counts, a missing
  target, a missing time value.
- **Whole numbers too large for 64 bits** arise from a DataFrame and from a CSV
  file alike: pandas loads a CSV column holding a whole number beyond 64 bits
  as an `object` column of Python integers (section 9). Until D50 such a column
  ended in a raw `TypeError` in every role. Now: as a spare column it is
  excluded with the reason `unsupported_type` and its warning (step 5, rule 7);
  as the target it is `task_unsupported` (step 4); as the time column it is
  `time_column_invalid` (step 3).
- A float column is not looked at for whole numbers: 64-bit floats above 2⁵³
  were merged before the tool saw them. Two cases reach the tool that way and
  cannot be put right by it (section 9): a CSV whole-number column with an
  empty cell, which pandas loads as floats, and whole numbers written as text
  (rule 5 of step 5 reads text through `pandas.to_numeric`, as floats).

**2. Drop rows with no target, then rows with no time value; count each.**
A target is missing when the cell is missing (NaN, `None`, `pd.NA`, `NaT`); or,
for a numeric target, infinite; or, for a text value, **when it holds only
white space (D48)**. A cell of spaces or tabs looks blank and is treated as a
blank: its row is dropped and counted in `dropped_missing_target`, exactly as
the row of an empty cell is. It is never a class of its own (the empty text of
a DataFrame cell, `""`, is the same case). Under time order a row with a
target and a missing (or infinite) time value is dropped next. A row missing both counts
under the target. The rows left are the **usable rows**. Fewer than 100 usable
rows: `too_few_rows`.

**An infinite number is no target in an object column either (D57).** In an
object target that step 1a leaves as it is (text beside numbers, booleans
beside a number), a cell that holds an infinite float, a Python `float` or a
numpy one, positive or negative, is a missing target: its row is dropped and
counted in `dropped_missing_target`, and the cell is counted in the target's
`n_infinite` (step 5), as in a numeric target. It is never a class. So a
target of the text `a` and `b` with a few infinite floats is a classification
on `a` and `b`, and a target of `a` and infinite floats alone is
`target_constant`. A `Decimal` infinity among values that are not all numbers
is not read as a number: it is a value like any other there. (Where every
value is a number, step 1a has made the column floats, and the rule for a
numeric target applies.)

**With an entity column (D57),** every usable row must name its entity. An
entity cell is empty when it is missing, or when it is text that holds only
white space. An empty cell in a usable row stops the run with
`group_column_invalid`: the message gives the number of such rows, and says
that the caller can fill the cells in or remove the rows. An empty entity cell
in a row that is dropped for its target or its time value is of no account.

**3. Under time order, read the time column in its accepted form (D18) and put
the usable rows in time order.** Accepted forms:

| Form | `validation.time_form` | Ordered |
|---|---|---|
| a datetime-typed DataFrame column, as it is (a time zone is allowed) | `datetime` | as instants |
| a numeric column (integer or float; not boolean, not complex) | `numeric` | as numbers |
| text, every non-missing value matching ISO 8601 with the year first: `YYYY-MM-DD`, optionally followed by `T` or a space and `HH:MM`, `HH:MM:SS` or `HH:MM:SS` with a point and **one to nine digits** of a fraction of a second (D52), optionally followed by `Z` or `+HH:MM` / `-HH:MM` | `iso8601` | as instants |

Anything else raises `time_column_invalid`: a boolean column, a complex-number
column (ordering it as floats would drop its imaginary part in silence: D48),
text in any other form (`25/01/2024`, `01/12/2023`, `5-Jan-2024`, `2024-01`), a
text column that mixes values with and without an offset, an impossible date
(`2024-02-30`), an object column of mixed types. The tool never guesses between
day-first and month-first. Rows with equal timestamps keep their file order.

**Fractions of a second (D52).** One to nine digits are accepted, each count
on its own and in any mixture within a column: `09:15:00.5`, `09:15:00.123456`
(what pandas writes for a microsecond timestamp), `09:15:00.123456789`. Each is
read as the instant it names, to the nanosecond. Until D52 exactly three
digits were accepted, so a CSV that pandas itself wrote from microsecond
timestamps stopped with `time_column_invalid`. A fraction of ten digits or
more is still refused, since a nanosecond timestamp cannot hold it and pandas
would cut it short in silence.

**A refused value's message says what is wrong with it (D52).** It quotes one
refused value. When that value is ISO 8601 in every respect except that its
fraction has more than nine digits, the message says so (it holds the word
`fraction`) and does not say the value "is not ISO 8601 with the year first",
which would be false. For text in another form the message is as before.

**Whole-number timestamps are compared exactly (D50, D53).** Two distinct
whole numbers are never one timestamp. *Whole numbers held exactly* are the
values of an integer-typed column (any width, signed or unsigned, nullable or
not, sparse or not) and of an object column that step 1a reads as whole
numbers. A 64-bit float holds every whole number up to 2⁵³ in magnitude and
only some above it: at 1.7 × 10¹⁸, where nanoseconds since 1970 now stand,
only every 256th. So:

- when such a column holds at least one value of magnitude **above 2⁵³**
  (9,007,199,254,740,992), judged over its non-missing loaded values, its keys
  are the 64-bit integers themselves, compared as integers, everywhere a key is
  used: the order of the rows, the fast profile's cap, the count of distinct
  timestamps, which rows share a timestamp, the split, the gap, the previous
  value, the blocks of the bootstrap. `validation.train_end`,
  `holdout_start` and `holdout_end` are then written as those integers,
  exactly;
- otherwise its keys are 64-bit floats, as for a float column and as before
  D50: below 2⁵³ nothing is lost by it, and `stages.read_time` is pinned to
  return floats there (2.7);
- a float column is ordered as the floats it holds. If two of its values were
  one number before the tool saw them, they are one timestamp;
- whole numbers too large for 64 bits (step 1a) are `time_column_invalid`: the
  message says that they are too large.

Until D50, 600 distinct nanosecond instants 100 nanoseconds apart were read as
235 timestamps: `timestamps_unique` was false, the previous-value baseline was
gone, and a random walk with its own lag as a feature was reported at skill
0.9994 against the training mean (third review, finding 3). That is F8's
outcome with the baseline code intact, so the gate has a dataset for it
(`time.integer_keys_distinct`, section 8).

A datetime-typed or ISO 8601 column is ordered as nanosecond instants. A date
outside what a nanosecond timestamp can hold (before 1677-09-21 or after
2262-04-11: pandas 3 keeps such dates in a coarser unit) is
`time_column_invalid`, never a raw `OverflowError`.

**4. Decide the task (D4, D20, D21),** on the usable rows, before any row cap.

**A text target value is read without the white space round it (D46).** `yes`,
` yes` and `yes ` are one value, `yes`, everywhere the target is used: in the
count of distinct values and so in the task; in `task.classes` and
`task.positive_class`; in the 0 and 1 handed to the models; in the
stratification of the split (2.7); and in `y_true` (5.1), which shows `yes`
for all three. A value that is not text is as it is. White space is what it is
for a categorical level (step 5). A text value that holds nothing but white
space is not a value: it is a missing target (step 2; D48). Until D46 a yes/no
target with a few trailing spaces stopped with `task_ambiguous` and "4
distinct values".

Let *k* be the number of distinct target values, read so.

- `task="auto"`: *k* = 2 is classification, whatever the type. A numeric target
  (integer or float type, not boolean) with *k* > 20 is regression. Anything
  else raises `task_ambiguous`.
- **What the message of `task_ambiguous` tells the caller to do is pinned
  (D57).** For a target that is not numeric with three or more distinct
  values, the message says that three or more classes are not supported, and
  gives *k*. It does not suggest declaring the task: no declared task would
  run (`classification` needs two classes, `regression` a numeric target), so
  the message holds neither `task=` nor the word `declare`. It says what would
  run: the rows of two classes, or the classes merged into two. For a numeric
  target with 3 to 20 distinct values, the message says the same of three or
  more classes, and adds that a target which is a quantity can be run with
  `task='regression'`, in those words.
- **A complex-number target is not numeric (D48).** Turning it into the floats
  a regression needs would drop its imaginary part with nothing said, which
  both implementations did until D48. So `task="regression"` on it is
  `task_unsupported`, and under `task="auto"` the ordinary rule applies: two
  distinct values are classification, anything else is `task_ambiguous`.
- `task="classification"` is honoured when *k* = 2, else `task_unsupported`.
- `task="regression"` is honoured when the target is numeric, else
  `task_unsupported`.
- *k* < 2 raises `target_constant` in every case.
- When the rule picks regression and every target value is a whole number, the
  warning `whole_number_regression_target` is added.
- **A target of whole numbers too large for 64 bits** (step 1a) is
  `task_unsupported`, whatever `task` says (D50). They are not read as numbers,
  and the caller who means them as labels can pass them as text.
- **A target whose values cannot be compared for equality** (lists,
  dictionaries, sets: Python cannot hash them) is `task_unsupported`, whatever
  `task` says: its distinct values cannot be counted (D53). Both
  implementations raised a raw `TypeError` here until D53. A tuple can be
  compared, and is a value like any other.
- An object column that step 1a reads as numbers is, for all of the above, the
  typed column it stands for: `Decimal` values are a numeric target, `True` and
  `False` with `None` are two classes.

**A class value is written as a boolean, a number or text (D53).** That is
what `task.classes` and `task.positive_class` can hold in JSON. A class value
that is none of the three (bytes, a date, a duration, a period, an interval, a
complex number, a tuple) is written as its text form, `str(value)`; the exact
text is open. Until D53 a two-valued target of such a type made the
application fail while writing `profile.json`, after the whole run.

For classification the two classes are put in sorted order (by value; by their
text if they cannot be compared). The second is the **positive class**: `1` of
`0`/`1`, `"yes"` of `"no"`/`"yes"`, `True` of `False`/`True`. `y_pred` is the
predicted probability of the positive class. A target written `no` and `yes `
has the classes `no` and `yes`, and `yes` is positive.

For regression the target is of integer or float type by the rule above, so
every usable value is a finite number as it stands. Nothing is coerced and
nothing can fail to parse: an implementation has no fallback to write here.

**5. Assign the column roles that need no target (D3, D19, D20),** on the
usable rows, in file order, before any row cap. The target has role `target`,
the declared time column role `time`, and the declared entity column role
`group` (D57). The entity column is never a feature and no rule below is
applied to it: whatever it holds, it is used only to say which rows belong
together. Its record has `kind` and `reason` null. For every other column, the
first rule that applies decides:

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
| 7 | any other type (mixed objects, complex numbers, durations, containers, whole numbers too large for 64 bits) | excluded, `unsupported_type`, with an `unsupported_type` warning naming the column (D51) |
| 8 | numeric, infinite and too-large values read as missing (below): no value left | excluded, `all_missing` |
| 8a | numeric, one distinct value | excluded, `constant` |
| 8b | numeric, every value a whole number, every value distinct, and the values consecutive or strictly ordered in file order (D19) | excluded, `id_like` |
| 8c | any other numeric | feature, kind `numeric` |

An object column that step 1a reads as numbers or booleans comes under rule 3,
as the typed column it stands for.

**A date column and no word about time order stops the run (D58).** When
`time_col` is `None` and at least one column is excluded with the reason
`datetime` (rule 2 or rule 4), the run stops here with
`time_order_undeclared`, before the row cap, the split, the leak screen and
any fit; nothing is written. A table that holds dates is very often ordered in
time, and a random split of such a table reports as predictable what is not:
on the real-data run the VIX level with its date column undeclared was
reported at skill 0.939, against 0.023 with it declared. The message names
every such column, in file order, each in quotes as Python's `repr` writes a
text, and the two ways on: `time_col` (`--time-col`) with a column's name,
and `time_col=False` (`--no-time-order`). Then:

- with `time_col=False` the run goes ahead under the random scheme exactly as
  it did before D58: the column is excluded as `datetime`, one
  `date_column_undeclared` warning names it, `validation.caveat` is the
  random caveat, and `input.time_col` is `false`;
- with a time column declared, any other date column is excluded as `datetime`
  with no stop and no warning, as before;
- a table with no date column is not affected, and `time_col=False` on such a
  table changes nothing but `input.time_col`.

**How an entity column is read (D57).** Two rows are of one entity when their
cells are equal. Text is compared without the white space round it; a
category-typed column through the string form of its values; whole numbers
exactly, at any size (a column of whole numbers too large for 64 bits is a
fine entity column); 0.0 and −0.0 are one value; a boolean, a date, a
duration and a `Decimal` are compared as the values they are. A column whose
values cannot be compared for equality (lists, dictionaries, sets) is
`group_column_invalid`. Every kind of column the type matrix lists (section
7) ends, as the entity column, in a result or in a named error.

**The infinite cells of the target and of the time column are counted (D57;
the fourth review, finding 3).** `columns[].n_infinite` of a numeric target or
time column is the number of its loaded cells that are infinite, each of which
is a missing target or a missing time value (step 2); for an object target it
is the number of its infinite floats (step 2). `n_missing` does not count
them. No `infinite_values` warning is raised for the target or the time
column: the rows are in `rows`.

"Text" is a pandas string column or an object column in which every non-missing
value is a `str`; a category-typed column is treated as text through the string
form of its values. Text is compared after stripping surrounding white space.

**The distinct values of a whole-number column are counted exactly (D50).** For
whole numbers held exactly (step 3: an integer-typed column, or an object
column that step 1a reads as whole numbers), rules 8, 8a and 8b are judged on
the integers themselves, at any magnitude: how many distinct values there are,
whether every value is distinct, whether they are strictly ordered, and
whether the largest minus the smallest plus one equals their count, which is
worked out in whole numbers of any size, so that the two ends of the 64-bit
range cannot overflow it. Until D50 they were judged on 64-bit floats, where
whole numbers above 2⁵³ merge: an unsigned column holding 50 distinct values
near 2⁶³ was reported `constant`. The values handed to a step and to the
models are 64-bit floats all the same (2.6), so two such values can reach a
model as one number; the roles do not depend on that.

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
missing, with a `values_coerced` warning), `n_infinite` (infinite cells and
too-large ones, which become missing, with an `infinite_values` warning).
`n_coerced` is 0 for a column that rule 5 did not read as numeric, including
one excluded by rule 4a.

**A finite value too large for the tree models is read as an infinite one is
(D48).** scikit-learn's trees and forests read their input as 32-bit floats and
raise on a value beyond that range; the leak screen's tree reads the column as
it is, so such a cell in a row a tree was fitted on stopped the run with a raw
`ValueError` until D48, in both implementations (section 9). The rule:

- a numeric feature cell whose magnitude is **greater than
  3.4028234663852886e+38**, the largest 32-bit float
  (`numpy.finfo(numpy.float32).max`), is treated exactly as an infinite cell:
  read as missing, counted in `n_infinite`, named by the column's one
  `infinite_values` warning, imputed like any missing cell (step 8);
- a value of exactly that magnitude, or smaller, is kept as it is;
- it is judged by value, for every numeric feature, whatever its type, and
  whether the cell was a number on load or text read by rule 5 (`1e39`);
- it is everywhere a feature value is read: in the counts and rules of this
  step (a column with no other value is `all_missing`, one with a single other
  value `constant`), in the leak screen, in what a step receives (NaN: 2.6),
  in `n_imputed`, and in `stages.Preprocessor` (2.7);
- it applies to features only. A target value of that size is not missing; a
  target too large for squared error is D44's case ("Numbers that are not
  finite"). A time value is ordered as the number it is.

A sentinel such as 1.7976931348623157e308 in a feature is therefore a missing
cell that the output counts, not a crash and not a value.

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
- No time column, and an entity column (D57): **whole entities.** The entities
  are taken in an order drawn from the seed, and each is kept when all of its
  rows still fit under the cap. So at most 2,000 rows are kept; every entity
  is kept whole or left out whole; and no entity left out would have fitted
  beside those kept. Which entities are kept changes with the seed, and is
  otherwise open. An entity of more than 2,000 rows can never be kept. (With
  `duplicates="group"` and no entity column the cap is the random sample of
  rows above: the sets of copies are formed afterwards, among the rows kept.)
- Time column: the most recent whole timestamps whose rows together number at
  most 2,000. If the most recent timestamp alone holds more than 2,000 rows
  the cap can keep nothing, and the run stops as step 7 says for a cap that
  left too few timestamps: with `fast_profile_too_few_timestamps` when the
  full profile would get past the split, and with the full profile's own
  error when it would not (D46).

The rows left out are counted as `dropped_row_cap`. After the cap: fewer than
100 rows is too few; for classification, fewer than 10 rows of either class is
too few. Which error says so depends on whether the cap is the cause (D52,
D53):

- the cap left no row out (the full profile, or a table of at most 2,000
  usable rows): `too_few_rows`, as ever;
- the cap left rows out, and the same call with `profile="full"` would get
  past this point and form its split: **`fast_profile_too_few_rows`**. The
  message names the cap of 2,000 and tells the caller to use
  `profile="full"`, and that advice is true;
- the cap left rows out, and the full profile would stop too: the error the
  full profile would raise (`too_few_rows`, or `gap_too_large`), so that
  nobody is sent to a profile that stops as well.

"Would get past this point and form its split" is judged on every usable row:
for classification, at least 10 usable rows of each class; under time
order, the split of step 7 can be formed from every usable row with the
caller's gap; and, with an entity column and no time column, the split of
whole entities of step 7 can be formed from every usable row (when it cannot,
the error is `too_few_groups`, the full profile's own). Nothing further is
looked at. Two shapes arise in practice.
Under time order, a latest timestamp of 60 rows behind one of 2,500: the cap
keeps 60 rows, and the full profile would split the table (third review,
finding 9; until D52 this was `too_few_rows: 60 rows are left`, naming neither
the cap nor the full profile). Under the random scheme, a class with 20 rows in
40,000: a sample of 2,000 holds about one of them, and the full profile has
all 20. The code is separate from `fast_profile_too_few_timestamps` because it
names what is too few: rows, where the other names timestamps.

**7. Split off the holdout (D6, D7), applying the gap.**

*No time column (random scheme).* With *n* rows, the holdout is ⌈*n*/4⌉ rows
drawn at random from the seed; for classification it is stratified, so that
each class's holdout count is within one of a quarter of the class. The other
rows are the training rows. Five folds partition the training rows, shuffled
from the seed, stratified for classification; fold sizes differ by at most one.

*No time column, and rows that belong together (grouped scheme; D57, D60).*
The rows are split as whole **units**. A unit is:

- with an entity column: the rows of one entity;
- with `duplicates="group"`: a set of rows that are copies of one another, as
  step 8a defines a copy but counted **here, before the split, on the feature
  columns of step 5** (the leak screen has not run yet) and over all the rows
  that are split: the same value in every such column and the same target. A
  row with no copy is a unit of its own;
- with both: two rows are in one unit when they are of one entity or copies of
  one another, directly or through other rows. So a set of copies that spans
  two entities joins them into one unit.

`validation.n_rows_with_copies` is the number of rows that have a copy among
the rows split, and when it is above zero one `copies_grouped` warning says
so (2.5). With `duplicates="group"`, no entity column and no row that has a
copy, there is nothing to keep together: the split is the random scheme's,
row for row, and `validation.scheme` is `random`. In every other case of this
paragraph the scheme is `grouped`, however few the rows that have a copy
(D61): the interval then comes from whole units and there is no p-value (step
12). What is pinned is the following, and not which rows land where:

1. **No unit has rows on both sides of the holdout split.**
2. **Within the training rows, every unit is in the validation part of exactly
   one fold,** whole; the five validation parts are disjoint and together are
   the training rows; a fold's training rows are the other four parts. So no
   fit of a fold sees a row of a unit it then predicts.
3. **The holdout holds at least a quarter of the rows, in whole units, and no
   more than it must.** For regression: the units are taken in an order drawn
   from the seed until they hold at least a quarter of the rows, so the
   holdout holds less than a quarter of the rows plus its largest unit. For
   classification the same is done twice, once for the units of each class, a
   unit counting with the class that most of its rows hold (the positive class
   on a tie): of the units of each class, the holdout holds at least a quarter
   of their rows and less than a quarter plus the largest of them. That keeps
   the classes as close to balanced across the two sides as whole units allow;
   `validation.stratified` is true.
4. **The folds are as even as whole units allow:** the training units are
   dealt, in an order drawn from the seed, each to the fold that so far holds
   the fewest rows (for classification, of that unit's class). Fold sizes are
   not pinned beyond that.
5. The split is drawn from the seed, depends on which rows share a unit and
   on nothing else about how units are named, and equals
   `stages.split_random(n, classes, seed, groups)` (2.7).

Fewer than six units among the rows split, or fewer than five left for the
folds once a quarter of the rows is held out: `too_few_groups`. Under the fast
profile, when the cap left rows out and the split of every usable row could be
formed, the error is `fast_profile_too_few_rows` instead, as in step 6. Every
row its own entity is allowed: the units are then the rows, the split is one of
whole units of one row each, and the scheme is still reported as `grouped`,
since that is what the caller asked for. `validation.n_groups_training` and
`validation.n_groups_holdout` count the units on each side.

A class held by the rows of one unit only cannot be on both sides: the run
stops with `one_class`, by the rule at the end of this step.

*Time column (time-ordered scheme).* An entity column does not change the
split: rows are split by their timestamps, as below, and the rows of one
entity are on both sides, earlier ones in training and later ones in the
holdout. The entity column is used for the previous value (step 11). Let the
distinct timestamps in order be
*t*₁ < … < *t*ₘ.

- The **holdout** is the latest whole timestamps that together hold at least
  25% of the rows, the smallest such set: add timestamps from the latest
  backwards until 4 × (rows held) ≥ *n*. Rows sharing a timestamp are never
  separated.
- The **gap** removes the `gap` distinct timestamps immediately before the
  holdout from the training side. Their rows are counted as `dropped_gap`.
- The remaining earlier timestamps are the **training timestamps**. Fewer than
  six before the gap is taken: `too_few_rows`. Fewer than six after:
  `gap_too_large`. (Under the fast profile's cap, see below.)
- The training timestamps are cut into **six contiguous blocks** of timestamps,
  as equal as possible, the earlier blocks taking the extra ones (the cut of
  `numpy.array_split`). **Fold *k*** (1 to 5) validates on block *k*+1 and
  trains on blocks 1 to *k*, less the `gap` distinct timestamps immediately
  before block *k*+1. Those rows are reported per fold as `gap_rows`; they stay
  training rows of the final fit. No training timestamp left before a
  validation block: `gap_too_large`.

`validation.holdout_share` is holdout rows ÷ (training + holdout + gap rows).

**When the fast profile's cap is what left too few timestamps (D20, D46).**
Under `profile="fast"`, when the cap left rows out and the rows kept cannot
form the split for want of timestamps (the cap could keep nothing, step 6; or
fewer than six timestamps stand before the holdout), the tool works out what
this step would do with **every usable row**, as `profile="full"` would, with
the caller's `gap`:

- that split can be formed: `fast_profile_too_few_timestamps`. Its message
  tells the caller to use `profile="full"`, and the advice is true;
- it cannot: the error the full profile would raise there, `too_few_rows` or
  `gap_too_large`. Nobody is sent to a profile that stops too.

So five dates of 600 rows each stop with `too_few_rows` under either profile;
nine dates of 450 rows with `gap=2` stop with `gap_too_large` under either;
and eight dates of 450 rows stop with `fast_profile_too_few_timestamps` under
`fast` and run under `full`. Until D46 all three gave the fast profile's error.
One case is left as it was: when the rows kept do have six timestamps before
the holdout and the caller's gap leaves fewer than six, the error is
`gap_too_large`, whatever the full profile would do.

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
being each fold's training mean and, under time order with unique timestamps
or with an entity column, whichever of that and the previous value has the
lower error. With an entity column the previous value is each entity's own
(step 11), and a validation row whose entity has none is given its fold's
training mean in its place (D57). So the lag of a persistent panel is measured
against the baseline it must beat, and is not excluded as a leak for being as
good as the previous value. When that
baseline's squared error over the pooled rows is exactly zero, the column's
skill cannot be computed (D36): rules 2 and 3 do not apply to it, it stays a
feature and it gets no warning. When a baseline's pooled squared error, or the
column's own, is not finite, the run stops with `non_finite_score` (D44;
"Numbers that are not finite", rule 2): a number that is not finite is never
read as "not flagged".

**The previous value exists for every row it is asked for.** Under time order
with unique timestamps, the first validation block starts at least `gap + 1`
timestamps after the first training timestamp: step 7 stops with
`gap_too_large` otherwise. So every row of every validation block, and every
holdout row, has a row `gap + 1` timestamps before it, here and in step 11.
There is no rule for a row without one, because there is no such row: an
implementation that meets one has a defect in its split or its baseline and
must fail loudly. It may not drop the previous-value baseline and carry on
with the training mean. (With an entity column a row whose entity has no
earlier row is an ordinary case with a rule of its own, above and in step 11:
that is a fact about the data, not a defect.)

A column excluded by rule 1 or 2 gets role `excluded`, reason `suspected_leak`,
its `leak_rule`, and a `suspected_leak` warning. It reaches no model. If no
feature is left: `no_features`, its message naming every excluded column with
its reason (2.4). A column kept by rule 3 has role `feature`, a
null `reason` and a null `leak_rule`, and reaches the models like any other:
it still drives the headline, because the tool cannot tell a near-leak from a
very strong real feature (D33). The warning asks a question; it does not
accuse.

Known limit (D31): in a panel (repeated timestamps) with no entity column, the
only baseline is the training mean, so a lagged level of a persistent target
can be excluded by rule 2 or warned about by rule 3. Since D58 such a run is
reached only when the caller has passed `group_col=False` (step 11); with the
entity column declared the limit is gone.

**8a. Count the holdout rows that have a copy among the training rows (D49,
D53).** Whenever there is no time column: under the random scheme and under
the grouped scheme alike (D61). A declared entity column does not switch the
rule off: copies that sit in different entities on both sides of the split
are counted like any others. The headline is computed on rows no fitted step has
seen; a holdout row whose exact copy sits among the training rows has been
seen in all but name, and a model that can recall a row scores on it. On pure
noise with every row present twice the third review measured skill 0.918,
p = 0.001, under the full profile, against 0.009 without the copies.

**What this step does not see (D56, D60).** A copy is an exact copy. A row
that differs from another in any feature column at all (one added column of
unique values, a batch number, a value changed by 1e-9) is not a copy, and is
not detected in this version: on the real-data run, noise with every row twice
and such a column was reported at skill 0.35 to 0.44. A broader screen was
designed and measured in the fifth pass and not adopted, because on real
tables it stopped clean data (D60; `docs/measurements/CP6/`); it is the first
item of the next pass. Until then **the control is the caller's:** when the
caller knows which rows belong together (one customer, one patient, one
measurement written twice), `group_col` names the column that says so, and
the rows of one entity are kept on one side whatever their other columns hold
(step 7). A copy of a row altered in every column cannot be told from a
different row by any tool.

- **A copy.** A holdout row has a copy when some training row has **the same
  value in every feature column and the same target**. The feature columns are
  those with the role `feature` once the leak screen is done: an excluded
  column (an identifier, a date, a suspected leak) plays no part, so two rows
  that differ only there are copies. The rows are the training and holdout
  rows of step 7, after any row cap.
- **How values are compared.** As a seam-supplied step is handed them (2.6),
  so a test can count for itself:
  - a numeric feature as 64-bit floats. Two missing cells are equal, a missing
    cell being an empty, coerced, infinite or too-large one (step 5). `0.0`
    equals `-0.0`. `1`, `1.0` and the text `1.00` in a column read as numeric
    are equal. Whole numbers above 2⁵³ are compared as the floats a model is
    given, so two rows that differ only in such a value, by less than a float
    can hold, are copies: no model could tell them apart (step 5 counts the
    distinct values of such a column exactly; this step asks what a model
    sees);
  - a categorical feature by its level: the text without the white space
    round it, and `(missing)` for an empty cell. `red` equals ` red `; `red`
    does not equal `Red`;
  - the target as step 4 reads it: the class for classification (so `yes`
    equals `yes `), the number for regression.
- **The count and the share are always reported** when there is no time
  column, under the random and the grouped scheme:
  `validation.n_holdout_copies`, the number of holdout rows that have a copy,
  and `validation.holdout_copy_share`, that number ÷ the holdout rows. Each
  holdout row counts once, however many copies it has. Under time order both
  are `null`. They are plain numbers: a run that goes ahead always carries
  them, whatever `duplicates` was (D60).
- **The rule (D53).** It is met when **more than 1% of the holdout rows, and
  at least 10 of them, have a copy**: `n_holdout_copies ≥ 10` and
  `holdout_copy_share > 0.01`. Both are needed. Ten copies in a holdout of
  1,000 rows are exactly 1% and do not meet it; eleven do. Nine copies in a
  holdout of 100 do not meet it, though they are 9%: below ten rows a count
  cannot tell coincidence from duplication, which is the reasoning of the
  10-rows-of-a-class floor (4.2).
  - Met, `duplicates="error"` (the default): the run stops with
    `holdout_copies_in_training` (2.4). Nothing is written. The message names
    the three ways on: remove the repeated rows, `duplicates="group"`, or
    `duplicates="keep"`.
  - Met, `duplicates="keep"`: the run goes ahead exactly as it would have
    before D49, and one `holdout_copies_kept` warning says the headline may be
    inflated.
  - Met, `duplicates="group"`: see below; the run stops.
  - Not met: the run goes ahead under any of the three values, the count and
    the share are reported, and no warning is added for them. The
    `duplicate_rows` warning of the loaded table stands as it was (2.5).
  `input.duplicates` records the value the caller passed, under every scheme.
- **`duplicates="group"` (D57, D60).** Step 7 has already kept each set of
  copies on one side, so here the count is normally zero: `n_holdout_copies`
  is 0 and `holdout_copy_share` 0.0, and that is how the output shows that the
  copies were kept together. One case is left. The sets of step 7 are formed
  on the feature columns of step 5, and this count is made on the feature
  columns that are left after the leak screen. When the screen excludes a
  column, two rows that differed only in it become copies of one another, and
  they may be on opposite sides. They are counted here like any other, and
  when the rule is met the run stops with `holdout_copies_in_training`; the
  message says that these rows became copies when the leak screen excluded a
  column, names every column excluded as a suspected leak, and tells the
  caller to remove it and run again. Below the rule the run goes ahead and the
  count is reported.
- **When it is judged.** After the leak screen and the `no_features` check,
  and **before any preprocessing or model is fitted** (step 9). A run that
  stops here has fitted nothing but the screen's own trees, and a table that
  would also fail every model stops with this error (2.4). The baselines of
  step 11 are scored earlier ("Numbers that are not finite", rule 1), so a
  target too large for squared error still stops with `non_finite_score`.
- **`duplicates="keep"` changes nothing else.** The same rows, split, folds,
  models, predictions and numbers as `"error"` gives on a table that does not
  meet the rule; only `input.duplicates` differs.
- **Under time order the rule does not apply (D49),** and `duplicates` has no
  effect: rows at different times are different observations even when their
  values agree, and rows at one timestamp are on one side of the split. No
  count is made.
- **When a date column is excluded (D53, D58).** A column excluded as
  `datetime` takes no part in the comparison, so rows that differ only in
  their date are copies, and a table of repeated measurements can stop here.
  Since D58 that is reached only when the caller has passed `time_col=False`:
  with `time_col=None` the same table stops earlier, at step 5, with
  `time_order_undeclared`. The copies error's message then names each such
  column and says that under `time_col` the rule does not apply.

A table of a few categorical columns and no other feature will often meet the
rule with no row repeated by mistake: with twelve combinations of levels,
every holdout row has a training row like it. That is what `"keep"` is for
(D49).

The cut-off of 1% and the count of 10 were measured (10.6): over 30 data seeds
every dataset of the gate, and the README's example table, has no holdout row
with a copy, and a table with 5% or more of its rows repeated meets the rule
on every seed.

**Preprocessing, fitted on the rows of each fit only (D3).** One preprocessing
is fitted for each fit, on that fit's training rows, and applied to the rows it
predicts:

- numeric features: missing values (including coerced, infinite and too-large
  cells: step 5) are replaced by the median of the fit's training rows (0 if
  the column is all missing there), then standardised with the mean and
  standard deviation of the fit's training rows (a zero deviation is read as
  1). Every value left after step 5 is within ±3.4028234663852886e+38, so the
  mean and the deviation are always finite: a zero deviation is the only
  special case, and there is none for a deviation that is not finite (D48);
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

**A configuration is failed in exactly the same way when a number it gives is
not finite (D44):** a prediction for a fold's validation rows that is infinite
or not-a-number, or a fold score that is. Status `failed`, an error text that
says which, a `model_failed` warning, five null fold scores, a null `cv_mean`
and `cv_std`, never chosen. A score that is not finite is never written as
null beside the status `fitted`, and such a configuration can never be chosen.

**What can fail a configuration, and what cannot (D57; the fourth review,
finding 5).** A configuration is recorded as failed for what the model does
and for nothing else: making the estimator, its `fit`, its prediction, and
the reading of what it handed back (the shape of 2.6, and the check that every
value is finite). A fold score that is not finite fails it too, as above. The
tool's own arithmetic on a fold is not part of it: if scoring finite
predictions of the right shape raises, that is a fault of the tool, and the
run ends with that exception as any defect does. It is never caught and
written into the models table as if the model had failed, where every
configuration would fail alike and the run would end in a misleading
`all_models_failed`. The same boundary holds in step 10. This paragraph is the
contract for a builder and has no test of its own: no input reaches the tool's
scoring with predictions that are not already one finite number for each row.

The configurations that have a `cv_mean` are **ranked**, best first: the
highest ROC AUC, the lowest squared error. Ties go to the earlier family, then
the earlier grid position. The first in the ranking is chosen.

**10. Refit the chosen configuration on all training rows, and predict the
holdout with it (D38),** with its preprocessing fitted on all training rows.

If that fit or that prediction raises, or the prediction holds a value that is
not finite (D44), the configuration was chosen in vain.
It gets status `failed`, its error text and a `model_failed` warning, and its
`chosen` stays false. Its `fold_scores`, `cv_mean` and `cv_std` **stay as they
were measured**: the table reports what was tried. The next configuration in
the ranking of step 9 is then chosen and tried in the same way, and so on. If
none is left: `all_models_failed`, its message carrying each error text.

The holdout plays no part in the choice. A configuration is passed over only
for raising, never for its score, so the headline is not inflated by the
fallback. A prediction that is not finite counts as raising because it depends
on the model and the features alone, not on the holdout targets. A holdout
*score* that is not finite, from predictions that are finite, is a score: the
configuration is not passed over for it, and the run stops with
`non_finite_score` ("Numbers that are not finite", rule 4).

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
- Regression under time order **with an entity column (D57)**, whether the
  timestamps repeat across entities or not: the second baseline,
  `previous_value`, is each entity's own. For a holdout row it predicts the
  target of **the latest row of the same entity that lies at least `gap + 1`
  distinct timestamps earlier**, the timestamps being those of all the
  time-ordered usable rows after the row cap, whatever their entity (so the
  gap means what it means for the split), and rows inside the gap counting as
  earlier rows. So:
  - an entity's first holdout row is predicted from its last training row
    (with `gap` 0; with a gap, from its last row at least `gap + 1` timestamps
    back), and its later holdout rows from the holdout rows before them;
  - an entity observed at only some timestamps is predicted from its own last
    observation, however long ago;
  - **a holdout row whose entity has no such earlier row** (an entity first
    seen in the holdout, or seen only inside the gap) has no previous value.
    For such a row the previous-value baseline predicts the training mean,
    which is what is known of it. `validation.n_holdout_without_previous`
    counts those rows, and when it is above zero one `no_previous_value`
    warning says so;
  - both baselines are scored and reported, and the headline baseline is the
    one with the lower squared error on the holdout, a tie going to
    `training_mean`, exactly as with unique timestamps (D10);
  - **within one entity the timestamps must be unique.** Two rows of one
    entity at one timestamp, among the rows split, stop the run with
    `group_column_invalid`: an entity is then not one series and has no one
    previous value. The message gives the number of such rows. (In a
    classification there is no previous value, so nothing asks this of the
    entity column, and such a table runs.)
- Regression under time order with **repeated timestamps and no entity
  column**: this is a panel, several series in one table, and nothing says
  which row belongs to which. Let *a* be the lag-1 autocorrelation of the
  per-timestamp mean of the target, over the training timestamps in order.
  When those means do not vary *a* has no value, and the baseline is
  `training_mean` with nothing more said. When *a* cannot be computed because
  a value overflows, the run stops with `non_finite_score` (D44): a number
  that is not finite is never read as "not above 0.5". Otherwise:
  - *a* is at most 0.5: the baseline is `training_mean`, with no warning;
  - *a* is above 0.5 and **`group_col` is `None`: the run stops with
    `entities_undeclared` (D58).** Against the training mean, skill on such a
    table shows only that one period resembles the last: on the real-data run
    a panel of five stock prices was reported at 0.979. The message says so,
    gives *a*, and names the two ways on: `group_col` (`--group-col`) with the
    column that says which series a row belongs to, or `group_col=False`
    (`--no-entities`). It is raised as soon as the split is made, before the
    leak screen and before any fit, and nothing is written;
  - *a* is above 0.5 and the caller passed `group_col=False`: the run goes
    ahead exactly as it did before D58, with the baseline `training_mean` and
    one `autocorrelated_panel_target` warning.

A baseline needs no model, so it is scored on the holdout as soon as step 7 has
made the split, and a baseline score that is not finite stops the run before
any fit; a holdout score or a skill that is not finite stops it here ("Numbers
that are not finite", rules 1, 4 and 5; D44).

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
- *Grouped scheme, interval (D57):* *R* resamples of the **units** of the
  holdout (step 7: entities, or sets of copies), with replacement: each
  resample draws as many units as the holdout has, and takes every row of
  each unit drawn, as often as it is drawn. The score and the skill of each
  resample, and the percentiles, are as under the random scheme. Method
  `bootstrap_groups`; `block_length` is null. The rows of one entity are not
  independent, and a resample of single rows treats them as if they were: on
  the gate's `entities` table, where the true skill is zero, the interval from
  single rows left zero outside in 11 of 30 runs, and the interval from whole
  entities in none (10.7). With **fewer than 30 units in the holdout** there
  is no interval: the fields are `null`, `interval_reason` gives the number of
  units and the 30 needed, and the warning `no_interval` is added.
- *Grouped scheme, p-value:* absent (`null`), with `n_permutations` null, and
  `p_value_reason` states that shuffling the holdout targets assumes
  independent rows, which the rows of one entity are not; its wording is open,
  and it holds the words `independent rows` (10.7: the row shuffle put *p*
  below 0.05 in 5 of 30 runs on that table).
- Under time order an entity column changes nothing in this step: the interval
  is the block bootstrap over timestamps and there is no p-value.
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
  `p_value_reason` says why. **A p-value is reported only when skill is a
  number (D44).** Skill is a finite number or null by the time this step runs:
  one that is not finite has already stopped the run.

Also added here: `small_holdout` (2.5).

**13. Assemble the result; write the files** (section 5). Before anything is
returned or written, every number in every part of the output is checked to be
finite ("Numbers that are not finite", rule 6). The check is made where the
result is made, in the constructor of `ProfileResult` (2.3), so that no result
can exist that failed it.

### Numbers that are not finite (D44)

**No number that is infinite or not-a-number is ever written, and none is ever
turned into `null`.** A squared error can overflow: the square of 1e155 is
beyond the largest double (about 1.8e308). So one target cell of that size, a
sentinel such as 1.7976931348623157e308 among ordinary values, or a target on
the scale of 1e160 makes a mean squared error infinite, and what is computed
from it not-a-number. The second review found such a run finishing normally
with a null score, a null baseline, a null skill, a p-value all the same, and
no warning. `null` has two meanings in the output and no others: a value that
does not apply (the time fields under the random scheme), and a value that
cannot be computed for a reason the output states (D36, D40). The rules, in
the order they apply:

1. **The baselines first, before any fit.** A baseline needs only the target
   and the split. Every baseline of step 11 is scored on the holdout as soon
   as step 7 has made the split: before the leak screen, before any step or
   model is fitted. If a baseline's score is not finite, the run stops there
   with `non_finite_score`. Under time order with unique timestamps both are
   checked, the one that would not be the headline baseline too: its score is
   a number in `baselines`. **This is what gives the caller one clear error
   when the target itself is the cause.** The run never goes on to fail every
   configuration in turn and end in `all_models_failed` with a list of them.
   The two inputs of the second review stop here under both schemes: one cell
   of 1.7976931348623157e308, wherever the split puts it, and a target
   multiplied by 1e160.
2. **The leak screen.** In regression the screen scores baselines of its own
   over the pooled validation rows of the selection folds (step 8): each
   fold's training mean, and under time order with unique timestamps the
   previous value. If either error is not finite, or a column's own pooled
   squared error is not, the run stops with `non_finite_score`, still before
   any configuration is fitted. This catches the value that harms only the
   folds: in a series of 400 rows, one cell of 1e155 at row 120 leaves both
   holdout baselines finite (the training mean is about 3.3e152, and its
   squared error about 1.1e305), while the fold that validates on row 120
   cannot score it against anything: its own square is beyond the largest
   double.
3. **A configuration during selection (step 9).** A prediction that is not
   finite, or a fold score that is not finite, fails the configuration as
   raising would: status `failed`, error text, `model_failed` warning, null
   fold scores, never chosen. With rules 1 and 2 passed, the target is not the
   cause, the configuration is: a model whose predictions are 1e200 where the
   target is near 20. If every configuration fails so, the error is
   `all_models_failed`, and rightly: the list of them is the information.
4. **The chosen configuration (steps 10 and 11).** A prediction for the holdout
   that is not finite counts as raising: the configuration is failed, keeps
   its fold scores, and the next in the ranking takes its place (D38). If its
   predictions are finite and its score on the holdout is not, the run stops
   with `non_finite_score`, and **the next configuration is not tried**: that
   would pass a configuration over for its holdout score, which D38 forbids.
   For classification this cannot arise: ROC AUC of finite predictions is a
   count of ranks, always finite.
5. **Skill.** `headline.skill` is null in exactly one case, a baseline score of
   exactly zero (D36). If it comes out not finite from a score and a baseline
   score that are both finite (their quotient can overflow when the baseline's
   error is tiny beside the model's), the run stops with `non_finite_score`.
6. **Every other number, at the latest when the result is made.** Before
   anything is returned or written, every number of the output must be finite,
   **in every part of it (D57)**: `validation`, `headline`, `baselines`, `cv`
   and `models` (`cv_mean` and `cv_std`, both ends of both intervals, the
   p-value), and no less `meta`, `input`, `rows`, `columns`, `task` and
   `warnings`. One that is not stops the run with `non_finite_score`. This is
   the backstop: whatever arithmetic produced the number, and whatever part it
   sits in, it does not leave as a null. Until D57 the check covered the five
   scoring parts only, and a float that was not finite in another part (a
   class value, for one) was written as `null` or ended in a raw error (the
   fourth review, finding 4). The check is made in the constructor of
   `ProfileResult` (2.3), on the document it is given: a float, Python's or
   numpy's, that is infinite or not-a-number, anywhere in it.
7. **The p-value** is reported only when skill is a number (step 12).

`y_pred` needs no rule of its own: by rules 3 and 4 a prediction that is not
finite never reaches the output.

What the message of `non_finite_score` names is pinned; its wording is open.
It names the number: a baseline by its name as `baselines[].name` spells it
(`training_mean`, `previous_value`), under rule 1 and rule 2 alike; a
configuration by its name as `models[].config` spells it, under rule 4; a
field of the output by its key, under rules 5 and 6. And it gives the likely
cause, for one a target value too large for squared error. Nothing is written
when it is raised.

Where the outcome is open: at scales where every reported number can be held
but arithmetic on the way overflows (squared errors above about 1e154, whose
own squares cannot be held, as on a target of the scale of 1e80), a run may
finish with finite numbers or stop with `non_finite_score` under rule 6. It
may not write a null, and it may not end in `all_models_failed`.

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
| too few entities to split (grouped scheme) | fewer than 6 units; fewer than 5 left for the folds once the holdout is taken (D57) |
| too few holdout entities for an interval (grouped scheme) | fewer than 30 units in the holdout (D57) |
| one-hot levels kept per column | 30 |
| seed | a whole number from 0 to 4,294,967,295 (D45) |
| digits of a number in the console summary | at least four significant: within 0.0005 × \|value\| (D46) |
| digits of a fraction of a second in an ISO 8601 time value | one to nine (D52) |
| whole-number time keys kept as integers | when the column holds a magnitude above 2⁵³ = 9,007,199,254,740,992 (D50, D53) |

### 4.3 Cut-offs (D23, D28, D30, D33, D34, D48; measurements in section 10)

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
| a panel with no entity column: the stop `entities_undeclared`, or with `group_col=False` the `autocorrelated_panel_target` warning (D58) | lag-1 autocorrelation above 0.5 |
| feature value too large for the tree models | a magnitude above 3.4028234663852886e+38, the largest 32-bit float: read as missing, like an infinite value (step 5; D48) |
| copies of holdout rows among the training rows, with no time column | more than 1% of the holdout rows **and** at least 10 of them: the run stops, or with `duplicates="keep"` is warned about (step 8a; D49, D53; measured in 10.6). Unchanged by the fifth pass (D60) |

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
| `y_true` | The target of that row, as it is in the source; a text value without the white space round it (step 4; D46), so a cell written `yes ` shows `yes`, in the file and in `ProfileResult.predictions`. A target that step 1a reads as numbers or booleans shows the value read: the number `1.5` for `Decimal("1.50")` (D51). |
| `y_pred` | Classification: predicted probability of the positive class. Regression: the predicted value. Always finite: a configuration whose prediction is not finite is failed and never written (steps 9 and 10; D44). |

### 5.2 `profile.json`

UTF-8 JSON. NaN and infinity never appear: a value that cannot be computed is
`null` and something nearby says why.

**A null stands only where this section says one may, for the reason given
there (D44).** A field typed "number" below always holds a finite number. A
field typed "number or null" is null only in the cases listed beside it. A
number that comes out infinite or not-a-number is never written as null: its
configuration is failed, or the run stops with `non_finite_score` (section 3,
"Numbers that are not finite"). `ProfileResult.headline` and `to_dict()` hold
these values as they are: a number is a finite number and a null is `None`,
never NaN, so a test reads them without converting anything first. (In the
DataFrames a null cell is NaN or `None`, as 2.3 says; an infinity appears in
none of them.)

Top-level keys, all always present:
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
| `time_col` | string, `false` or null | the `time_col` argument as passed: a column's name, `false` for the caller's statement that the rows have no time order, or null when nothing was said (D58) |
| `task` | string | the task as requested: `auto`, `classification` or `regression` |
| `gap` | integer | |
| `n_rows` | integer | rows loaded |
| `n_columns` | integer | columns loaded |
| `duplicate_rows` | integer | rows that exactly repeat an earlier row; they are kept |
| `duplicates` | string | the `duplicates` argument as passed: `"error"`, `"keep"` or `"group"` (D49, D57) |
| `group_col` | string, `false` or null | the `group_col` argument as passed: a column's name, `false` for the caller's statement that there are no entities, or null when nothing was said (D57, D58) |

**`rows`**: a list of exactly these seven records `{item, count}`, in this
order: `loaded`, `dropped_missing_target`, `dropped_missing_time`,
`dropped_gap`, `dropped_row_cap`, `training`, `holdout`. Always:
`loaded` = the four `dropped_` counts + `training` + `holdout`.

**`columns`**: one record per input column, in file order.

| Key | Type | Meaning |
|---|---|---|
| `name` | string | |
| `position` | integer | zero-based column position |
| `dtype` | string | the pandas type as loaded (informative). For an object column that step 1a reads as numbers it is still `object` (D51) |
| `role` | string | `target`, `time`, `group`, `feature` or `excluded`. `group` is the declared entity column (D57): it is not a feature and no rule of step 5 is applied to it |
| `kind` | string or null | for a feature, `numeric` or `categorical` (pinned); null for the entity column; for other roles informative |
| `reason` | string or null | for an excluded column: `constant`, `id_like`, `datetime`, `free_text`, `numeric_with_separators`, `suspected_leak`, `all_missing` or `unsupported_type`; otherwise null |
| `leak_rule` | string or null | for a suspected leak: `predicts_target` or `missing_by_class`; otherwise null |
| `n_missing` | integer | missing cells among all loaded rows, as loaded. A target cell that holds only white space loaded as text and is not counted here; its row is counted in `rows.dropped_missing_target` (step 2) |
| `n_coerced` | integer | non-missing text cells that did not parse, among all loaded rows, in a column read as numeric by rule 5 of step 5; otherwise 0 |
| `n_infinite` | integer | among all loaded rows of a numeric feature column, the infinite cells and the finite cells too large for the tree models, a magnitude above 3.4028234663852886e+38 (step 5; D48). For the target and for the time column, in whatever role they hold a number: their infinite cells, each of which is a missing target or a missing time value (steps 2 and 5; D57); for an object target that is not read as numbers, its infinite floats. Otherwise 0, and always 0 for the entity column |
| `n_imputed` | integer | for a feature, its missing cells (as loaded, coerced or infinite) among the training and holdout rows; otherwise 0 |

**`task`**

| Key | Type | Meaning |
|---|---|---|
| `type` | string | `classification` or `regression` |
| `decided_by` | string | `rule` (from `task="auto"`) or `declared` |
| `n_distinct` | integer | distinct target values among the usable rows, text read without the white space round it (step 4): the evidence |
| `target_dtype` | string | pandas type of the target as loaded (`object` for one that step 1a reads as numbers) |
| `classes` | list or null | the two classes in sorted order, each a boolean, a number or text (step 4; D53), text without the white space round it; null for regression |
| `positive_class` | value or null | the second of `classes` |

**`validation`**

| Key | Type | Meaning |
|---|---|---|
| `scheme` | string | `random`, `grouped` or `time_ordered`. `grouped`: no time column, and the rows were split as whole units (step 7): an entity column was declared, or `duplicates="group"` found rows with copies (D57) |
| `holdout_share` | number | holdout ÷ (training + holdout + gap rows) |
| `n_folds` | integer | 5 |
| `stratified` | boolean | true for classification under the random and the grouped scheme |
| `gap` | integer | |
| `time_col` | string or null | the time column used, or null. Null when `input.time_col` is `false` |
| `time_form` | string or null | `datetime`, `numeric` or `iso8601` |
| `train_end` | string, number or null | the last training timestamp |
| `holdout_start` | string, number or null | the first holdout timestamp |
| `holdout_end` | string, number or null | the last holdout timestamp |
| `n_train_timestamps` | integer or null | distinct training timestamps |
| `n_holdout_timestamps` | integer or null | distinct holdout timestamps |
| `timestamps_unique` | boolean or null | whether every timestamp holds one row |
| `folds` | list | five records `{fold, training_rows, validation_rows, gap_rows, scored}`, `fold` 1 to 5 |
| `caveat` | string or null | under the random scheme, the fixed caveat; under the grouped scheme, the fixed caveat of that scheme (5.4); null under time order |
| `n_holdout_copies` | integer or null | under the random and the grouped scheme, the holdout rows that have a copy among the training rows (step 8a; D49); null under time order, where no count is made |
| `holdout_copy_share` | number or null | `n_holdout_copies` ÷ the holdout rows; null under time order |
| `n_rows_with_copies` | integer or null | with `duplicates="group"` and no time column: the rows, among those split, that have a copy among them, counted before the split on the features of step 5 (step 7; D57, D60). 0 when none has. Null with any other value of `duplicates`, and under time order |
| `group_col` | string or null | the entity column used, or null. Null when `input.group_col` is `false` or null |
| `n_groups_training` | integer or null | the units among the training rows (step 7): entities, or under `duplicates="group"` sets of copies joined with entities. Under time order with an entity column: the entities that have a training row. Null when the rows were split one by one and no entity column was declared |
| `n_groups_holdout` | integer or null | the same for the holdout rows. Under the grouped scheme no unit is counted on both sides; under time order an entity usually is |
| `n_holdout_without_previous` | integer or null | regression under time order with an entity column: the holdout rows whose entity has no row at least `gap + 1` timestamps earlier (step 11; D57). Null otherwise |

The seven time fields are null under the random and the grouped scheme. A timestamp is written
as `YYYY-MM-DDTHH:MM:SS` (with a fraction if it has one; in UTC if the column
carried a zone) for the `datetime` and `iso8601` forms, and as a number for the
`numeric` form. Where the keys are 64-bit integers (step 3; D50) it is
**exactly that whole number, written as an integer**, so that one above 2⁵³ is
not written as the nearest float.

**`headline`** (also `ProfileResult.headline`)

| Key | Type | Meaning |
|---|---|---|
| `metric` | string | `roc_auc` or `mse` |
| `score` | number | the chosen configuration's score on the holdout |
| `model` | string | the chosen configuration, as in `models.config` |
| `baseline` | string | `chance`, `training_mean` or `previous_value` |
| `baseline_score` | number | 0.5, or the baseline's squared error on the holdout, which may be 0 |
| `skill` | number or null | step 11; null when `baseline_score` is 0, with the `skill_undefined` warning (D36) |
| `interval_method` | string or null | `bootstrap_rows` (random scheme), `bootstrap_groups` (grouped scheme; D57), `bootstrap_blocks` (time order), or null when there is no interval |
| `interval_level` | number | 0.95 |
| `interval_low`, `interval_high` | number or null | the interval on `score`; null when `interval_method` is null |
| `skill_low`, `skill_high` | number or null | the interval on `skill`; null when `interval_method` is null, and when skill could be computed in fewer than half of the resamples (step 12): always so when `skill` is null (`skill_undefined`), and otherwise with the `no_skill_interval` warning |
| `n_resamples` | integer | 1,000 or 200 |
| `n_resamples_used` | integer | resamples in which the score could be computed |
| `n_resamples_skill_used` | integer | of those, the resamples in which skill could be computed too: the ones whose baseline error was not zero. Equal to `n_resamples_used` for classification |
| `block_length` | integer or null | under `bootstrap_blocks` |
| `interval_reason` | string or null | why there is no interval on the score |
| `p_value` | number or null | step 12; null under time order, under the grouped scheme (D57), and when `skill` is null. A p-value is reported only under the random scheme, and only when `skill` is a number (D44) |
| `n_permutations` | integer or null | shuffles used; null when there is no p-value |
| `p_value_reason` | string or null | why there is no p-value |

**`baselines`**: a list of records `{name, score, used}`. Classification: one,
`chance`. Regression: `training_mean`, and under time order with unique
timestamps or with an entity column also `previous_value` (step 11). Exactly
one has `used` true and it is the headline baseline. A score of 0 is reported
as it was measured.

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
| `status` | string | `fitted` or `failed`: failed when it raised, predicted a value that is not finite or scored a fold with a number that is not finite during selection (step 9), or raised or predicted a value that is not finite after it was chosen (step 10) |
| `error` | string or null | the error text of a failed configuration |
| `fold_scores` | list | five entries, a number or null. Null where the fold was not scored; all five null for a configuration that failed during selection. One that failed after it was chosen keeps the scores it had (D38) |
| `cv_mean` | number or null | mean of the scored folds; null for a configuration that failed during selection, and for no other: a configuration with status `fitted` always has a finite `cv_mean` |
| `cv_std` | number or null | their sample standard deviation; null with fewer than two, and for a configuration that failed during selection |
| `chosen` | boolean | true for exactly one, which has status `fitted` |

**`warnings`**: a list of records `{code, column, message}` (2.5). May be empty.

**`limitations`**: the list of nine sentences of 5.4, exactly, in order.

### 5.3 Reconciliation rules

1. `rows.loaded` equals the rows of the source; it equals the four dropped
   counts plus training plus holdout.
2. Every source column appears exactly once in `columns`.
3. `predictions.csv` has one row per holdout row, and each `y_true` equals the
   target of that row in the source (text compared without the white space
   round it).
4. `headline.score` equals the metric recomputed from `predictions.csv`;
   `headline.skill` follows from it and the baseline as in step 11, and is
   null when the baseline's score is 0.
5. Every part of `ProfileResult` equals the matching part of `profile.json`.
6. The same seed gives an identical `profile.json` and `predictions.csv`, apart
   from `meta.timing` (and `meta.n_jobs` when `n_jobs` differs).
7. Under the grouped scheme with an entity column, no entity of a holdout row
   (the rows of `predictions.csv`, read against the source) is the entity of a
   training row, and `validation.n_groups_training` and `n_groups_holdout` are
   the numbers of entities counted so (D57).

### 5.4 Fixed texts

The random-split caveat (`validation.caveat` under the random scheme), exactly:

> Random split: this number is optimistic if the rows are ordered in time or one entity appears on many rows.

The caveat of the grouped scheme (`validation.caveat` under it; D57), exactly:

> Random split of whole entities: this number is optimistic if the rows are ordered in time.

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
  content, to pandas and to the tool: a line of two quotes, of two quotes
  round a space, or of white space other than spaces and tabs (a form feed, a
  no-break space), is read as a header of one column. In a table of two
  columns or more the lines after it, the real header among them, then have
  more fields than the header: `input_error`, by the rule below. That holds
  before a header of two columns as before one of three, where pandas makes a
  plain index of the one leading field (measured 2026-11-20, section 9; D46).
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
| generic objects that are all booleans, all whole numbers that fit in 64 bits, or numbers of any mixture (whole, float, `Decimal`), with or without empty cells | the typed column they stand for: boolean, whole-number or float (step 1a; D51) |
| generic objects that are all whole numbers, not all fitting in 64 bits (from a DataFrame, or from a CSV: pandas loads such a column so) | excluded as `unsupported_type`, with its warning; `task_unsupported` as the target; `time_column_invalid` as the time column (D50) |
| duration, period, interval, complex, bytes, mixed objects (a boolean beside a number among them), containers | excluded as `unsupported_type`, with an `unsupported_type` warning naming the column (D51) |
| sparse | as the dense column of its values |

A DataFrame and a CSV path give the same result for the types a CSV can carry:
integers and floats without missing values, floats with missing values, and
text. `input.source` and `input.kind` differ.

**What pandas does to whole numbers in a CSV** (measured on pandas 3.1.1,
`docs/measurements/CP5`; section 9). A column of whole numbers loads as
`int64` when every value fits, as `uint64` when every value is non-negative
and fits that, and exact in both. A column with a value beyond 64 bits loads
as generic objects holding Python integers, exact, which step 1a reads as
"too large for 64 bits"; so does a column with a value below the signed
range. Two shapes lose digits before the tool is handed them, and the tool
cannot give them back: a whole-number column **with an empty cell** loads as
64-bit floats, so values above 2⁵³ arrive rounded; and a column holding a
value above the signed range but within the unsigned one, beside a negative
value, loads as text, which rule 5 of step 5 reads as floats. A caller with
such identifiers or timestamps passes a DataFrame with an integer type.

**Stated as a limitation** (sentence 7 of 5.4): a CSV is read with pandas'
types, so a text identifier such as `000123` loads as the number 123. A caller
who needs control passes a DataFrame.

---

## 7. Edges

None is skipped silently: each raises a named error or is reported in the output.

| Edge | Outcome |
|---|---|
| a `seed` outside 0 to 4,294,967,295 | `input_error`, before anything is loaded; from the command line exit 3 and one line, no traceback (D45) |
| the target column is absent | `column_not_found` |
| the target has one distinct value | `target_constant` |
| a text target written with white space round some values (`yes `) | the same class as without it; `classes`, `positive_class` and `y_true` show the value without it (D46) |
| a text target cell that holds only white space | a missing target: the row is dropped and counted in `dropped_missing_target` (D48) |
| a complex-number target | not numeric: `task_unsupported` when regression is declared; under `task="auto"` the ordinary rule (D48) |
| a regression target with a value too large for squared error (one cell of 1.7976931348623157e308; the whole column times 1e160) | `non_finite_score`, naming the baseline, before any fit; never a result with nulls in it and never `all_models_failed` (D44) |
| a baseline's error, or a column's own, that is not finite in the leak screen | `non_finite_score`, before any configuration is fitted |
| a configuration whose prediction, or whose fold score, is not finite | status `failed`, like one that raised; `model_failed` warning; never chosen |
| the chosen configuration predicting a value that is not finite on the holdout | status `failed`, fold scores kept, the next best by `cv_mean` is chosen: it counts as raising |
| the chosen configuration's holdout score not finite, from finite predictions | `non_finite_score`; no other configuration is tried |
| any other number of the output that is not finite, in any part of it | `non_finite_score`, naming it; never written as null. No `ProfileResult` can be made from a document that holds one (2.3; D57) |
| an infinite float in an object target that is not read as numbers (text classes with a few infinite cells) | a missing target: the row is dropped and counted in `dropped_missing_target`, the cell in the target's `n_infinite` (D57) |
| infinite cells in a numeric target or time column | their rows are dropped and counted; the cells are counted in that column's `n_infinite`; no `infinite_values` warning (D57) |
| a model whose `predict` returns a column of n values for n rows | read as n values; a model that returns any other number of values, or a `predict_proba` that is not (rows, 2), fails its configuration (2.6) |
| a text cell, or a whole number too large for 64 bits, in an object target or time column, in a row that is dropped for another column's empty cell | the same named error as in a usable row: the column is read once, over all loaded rows (step 1a; D57) |
| a target that is not numeric with three or more distinct values, under `task="auto"` | `task_ambiguous`; the message says that three or more classes are not supported and does not suggest declaring the task (D57) |
| a numeric feature cell that is finite and above 3.4028234663852886e+38 in magnitude | read as missing, counted in `n_infinite`, `infinite_values` warning: as an infinite cell is (D48) |
| a declared task the target does not support | `task_unsupported` |
| an ambiguous target under `task="auto"` | `task_ambiguous` |
| no usable feature column | `no_features`; the message names each excluded column and its reason |
| two features that would give the preprocessed matrix one name (`colour=red` beside `colour` with the level `red`) | `feature_name_collision`, naming the columns, before any fit |
| an output folder that cannot be used (the path, or a folder above it, is a file) | `input_error`, before anything is loaded |
| a CSV that is not valid UTF-8, wherever the bad byte sits | `input_error` |
| blank lines (empty, or only spaces and tabs) before the header of a CSV | skipped, as pandas skips them; the header is the first line that is not blank |
| a line of a form feed, of a no-break space, of two quotes or of two quotes round a space, before a header of two columns or more | `input_error`: pandas reads that line as a header of one column, and the lines after it are longer |
| too few rows to split (fewer than 100 usable, or fewer than 10 of a class) | `too_few_rows` |
| a holdout, or the training rows, with one class | `one_class` |
| a fold with one class | that fold is not scored; `fold_not_scored` warning; `validation.folds[].scored` false; `fold_scores` null there. No fold scored: `one_class` |
| a time column not in an accepted form | `time_column_invalid` |
| a gap that leaves no training rows | `gap_too_large` |
| too few timestamps for the holdout and six blocks | `too_few_rows` |
| the fast profile on a panel too wide for its row cap, where the full profile would get past the split | `fast_profile_too_few_timestamps` |
| the same, where the full profile would stop at the split too | the error the full profile would raise: `too_few_rows` or `gap_too_large` (D46) |
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
| exact duplicate rows | kept; `input.duplicate_rows`; `duplicate_rows` warning. Whether the run goes ahead is the next three rows |
| no time column: more than 1% of the holdout rows, and at least 10 of them, have a copy among the training rows | `holdout_copies_in_training`, before any model is fitted; with `duplicates="keep"` the run goes ahead with one `holdout_copies_kept` warning (D49, D53) |
| no time column: fewer copies than that (nine of 100 holdout rows; ten of 1,000) | the run goes ahead; `validation.n_holdout_copies` and `holdout_copy_share` report them; no warning for them |
| time order: rows that agree in every feature and the target | the rule does not apply; both fields null; `duplicates` has no effect |
| `duplicates="group"`, rows with copies | each set of copies on one side of the split and of every fold; scheme `grouped`; `n_rows_with_copies`; one `copies_grouped` warning; `n_holdout_copies` 0 (D57, D60) |
| `duplicates="group"`, no row with a copy | the random scheme's split, row for row; `n_rows_with_copies` 0; no warning |
| `duplicates="group"`, and the leak screen excludes the one column in which two rows differed | those rows are copies now and may be on both sides: counted at step 8a, and above the rule the run stops with `holdout_copies_in_training`, naming the excluded column |
| a copy of a row that differs in any feature column (an added column, a batch number, a value changed by 1e-9) | not a copy, and not detected in this version (D60); `group_col` is the control |
| an entity column, no time column | the grouped scheme: no entity on both sides of the holdout or in two folds; `n_groups_training`, `n_groups_holdout`; interval `bootstrap_groups`; no p-value, with a reason (D57) |
| fewer than 30 entities in the holdout (grouped scheme) | interval fields null; `interval_reason`; `no_interval` warning |
| fewer than six entities, or fewer than five left for the folds | `too_few_groups` |
| an entity cell that is empty, or only white space, in a usable row | `group_column_invalid` |
| an entity column whose values cannot be compared (lists, dictionaries, sets) | `group_column_invalid` |
| an entity that holds one class | nothing special: it counts with that class when the classes are balanced across the split. A class held by one entity alone: `one_class` |
| `group_col` equal to the target or to the time column, or not a name, `None` or `False` | `input_error`, before anything is loaded |
| every row its own entity | allowed: the grouped scheme with units of one row each |
| the fast profile's cap with an entity column and no time column | whole entities: each kept whole or left out whole, at most 2,000 rows, and no entity left out would have fitted |
| an entity column under time order | the split is by time as ever; each entity has its own previous value; `n_holdout_without_previous` |
| a holdout row whose entity has no earlier row (time order) | the previous-value baseline predicts the training mean for it; counted; one `no_previous_value` warning |
| two rows of one entity at one timestamp, in a regression under time order | `group_column_invalid` |
| a panel (repeated timestamps, target autocorrelated above 0.5) with nothing said about entities | `entities_undeclared`, before any fit; with `group_col=False` the run goes ahead with the `autocorrelated_panel_target` warning (D58) |
| an unknown column type | excluded with reason `unsupported_type`, and an `unsupported_type` warning naming it (D51) |
| an object column of `Decimal` numbers, of booleans with `None`, of whole numbers with `None`, or of whole numbers, floats and `Decimal` numbers together | read as the typed column it stands for, in every role; `dtype` still says `object` (D51) |
| an object column holding a boolean beside a number | `unsupported_type`, with its warning: not guessed |
| a `Decimal` not-a-number, quiet or signalling | a missing cell |
| whole numbers too large for 64 bits, from a CSV or a DataFrame | a spare column: excluded as `unsupported_type`, with its warning. The target: `task_unsupported`. The time column: `time_column_invalid`. Never a raw exception (D50) |
| a whole-number time column with values above 2⁵³ (nanoseconds since 1970 as 64-bit integers) | every distinct value is its own timestamp; `train_end`, `holdout_start` and `holdout_end` are written exactly (D50) |
| a whole-number feature whose distinct values are distinct only as integers (above 2⁵³) | counted exactly: not `constant`, and `id_like` only by rule 8b on the integers (D50) |
| a time value with one to nine digits in its fraction of a second | accepted; read to the nanosecond (D52) |
| a time value with ten or more digits in its fraction | `time_column_invalid`; the message names the fraction |
| a date outside what a nanosecond timestamp can hold, in a datetime or ISO 8601 time column | `time_column_invalid` |
| a complex-number time column | `time_column_invalid` |
| a target of lists, dictionaries or sets | `task_unsupported` (D53) |
| a two-valued target of bytes, durations, periods, intervals, complex numbers or tuples | classification; the classes are written as text (D53) |
| the fast profile's cap leaving fewer than 100 rows, or fewer than 10 of a class, where the full profile would run | `fast_profile_too_few_rows` (D52, D53) |
| the same, where the full profile would stop too | the error the full profile would raise |
| a date column present and `time_col` left at `None` | `time_order_undeclared`, naming the column, before the split; nothing written (D58) |
| a date column present and `time_col=False` | random scheme; `date_column_undeclared` warning per column; `input.time_col` is `false` |
| a whole-number target read as regression by the rule | runs; `whole_number_regression_target` warning |
| a row missing a time value | dropped and counted as `dropped_missing_time` |
| a feature all missing within one fit's training rows | imputed with 0 in that fit |
| a categorical level unseen in a fit's training rows | encoded as all zeros |

### The type matrix (D52)

The second review found a feature value too large for the tree models; the
third found whole numbers too large for 64 bits, whole numbers merged as
floats, and number columns held as objects. Each is one member of one class:
**a kind of column, in some role, holding some extreme value, that the code
had not been written for.** The suite therefore holds the class, as a matrix,
in one file (`tests/test_type_matrix.py`), parametrised by kind, role and
extreme so that a failure names its case.

**The contract.** For every case, `profile()` (fast profile, with an output
folder) ends in one of two ways:

1. **a named error:** an exception that is a subclass of
   `mlprofessor.errors.ProfileError`. Then nothing was written; or
2. **a result**, and then:
   - every column of the table appears exactly once in `columns`, in order;
   - the row counts add up (5.3, rule 1), and `predictions` has one row per
     holdout row;
   - `to_dict()` is strict JSON, and so is the `profile.json` that was
     written: no NaN, no infinity, and no value JSON cannot hold;
   - every field typed "number" in 5.2 holds a finite number, and every field
     typed "number or null" holds a finite number or a null with its reason
     beside it, as 5.2 lists them (`skill` with `skill_undefined`, the
     interval with `interval_reason`, the p-value with `p_value_reason`);
   - every `y_pred` is finite.

Any other exception, a `TypeError`, a `ValueError`, an `OverflowError`, a
`decimal.InvalidOperation` or one of pandas' or scikit-learn's own, is a
failure of the case. The matrix does not say *which* of the two ends a case
has, nor which named error: the tests of each rule do that
(`tests/test_whole_numbers.py`, `tests/test_object_numbers.py` and the earlier
files). It says that there is no third end.

**Roles.** The column under test is, in turn: a spare column beside two
ordinary numeric features, with a 0/1 target; the target, with those two
features; the declared time column, with those two features and a numeric
target. 200 rows.

**The entity role (D57).** A fourth role, in a file of its own
(`tests/test_type_matrix_group.py`) so that the first matrix stays as it is:
the column under test is the declared entity column, `group_col`, beside the
two features. Every kind and every extreme below is tried with a 0/1 target
and no time column; and every kind with its ordinary values is tried again
with a numeric time column and a numeric target. The contract is the same,
a result or a named error, and a result also reports the column with the role
`group`. A date column is a fine entity column: the stop of step 5 concerns
columns that would be excluded as dates, and the entity column is not one of
them.

**Kinds.** Each signed and unsigned integer width (`int8` to `int64`, `uint8`
to `uint64`); 32-bit and 64-bit floats; nullable integer (`Int8`, `Int64`,
`UInt64`), nullable float (`Float32`, `Float64`) and nullable boolean; plain
boolean; text as an `object` column, as the default string type and as the
Python-backed string type; category, ordered and not; bytes; ISO 8601 dates
as text; datetime in nanoseconds and in seconds, without a zone, in UTC and
in a zone with daylight saving; timedelta; period; interval; complex, 128-bit
and 64-bit; `Decimal` objects; object columns of Python integers, of booleans,
of floats, of numbers mixed, and of a boolean beside numbers; mixed objects
(numbers and text); lists, tuples, dictionaries and sets; an all-missing
column of each of six types; sparse float, sparse integer and sparse boolean;
and the same number columns written to a CSV file and read by the tool.

**Extremes,** where the kind allows them: ordinary values; exactly two
distinct values (as the target); one value in every row; the largest and the
smallest value of the type; whole numbers just above 2⁵³; magnitudes near
1e300 and 1e-300 (1e38 for 32-bit floats); infinities; not-a-number and the
other empty values (`None`, `pd.NA`, `NaT`, a `Decimal` not-a-number of either
kind); negative zero; whole numbers beyond 64 bits throughout the column; and
a few whole numbers beyond 64 bits among ordinary ones.

Measured on the application as it stood after the third pass
(`docs/measurements/CP5`): of 805 cases, 37 ended in a raw exception, from
four causes: whole numbers beyond 64 bits (15 cases, `TypeError`, the third
review's finding 1); a target of lists, dictionaries or sets (12,
`TypeError`); a two-valued target of a type JSON cannot hold (7: the run
finished and `profile.json` could not be written); and a `Decimal` signalling
not-a-number (3, `decimal.InvalidOperation` from pandas). The reference as it
stood failed 18: the 12 and the 3, a nullable boolean target with a missing
cell, and a date of the year 9999 in a time column, as text and as a
datetime in seconds.

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
| F6 | Something is lost without being reported: rows, a column, a model family. | Gate, for rows: `data.rows_reconcile`, `data.coercion_reported`. For its other forms, the table below. Tests: accounting on crafted tables; a family that raises. |
| F7 | Time order is ignored: a shuffled split, a training row at or after the holdout, the gap not applied, or the time column misread. | Gate, for the split: `time.order_respected`, `time.no_skill_on_blocks`, `time.gap_applied`. For a misread time column, the table below. Tests: the split as a function of keys. |
| F8 | Under time order a trending or random-walk target is scored against the training mean only. | Gate: `time.persistence_baseline`. Tests: baselines by hand. One gate check and one test catch it; that is accepted (D43). |
| F9 | Holdout rows that have copies among the training rows are scored with no error and nothing said, so recall is reported as predictability (D49). | Gate: `data.holdout_copies_stop`, on noise with every row present twice. Tests: the rule of step 8a on tables with a known number of copies. |
| F10 | An entity has rows on both sides of the split although the entity column was declared: the split is made row by row and the output says `grouped` all the same (D57). | Gate: `groups.kept_together` (no entity on both sides, counted by the gate from `predictions.csv` and the source) and `groups.no_skill_on_entities`, on 300 entities of five rows with a label per entity. Tests: recording fakes through the seam, for the holdout and for every fold. |

**The other forms of F6, F7 and F8 (D46, D57, D58).** One broken copy of the reference cannot
fail every check of its family: the copy that drops rows loses no column, and
the copy that shuffles the split parses no date. The second review found four
checks listed against F6 and F7 that had never been seen to fail. Each form
now has a broken copy of its own, under an id that shows its family, and each
check names the form it was seen to catch.

| Id | The form | Gate check | Tests |
|---|---|---|---|
| F6b | A column is lost from the accounting: a constant column is dropped with no record. | `data.columns_accounted` | every column appears once, by its rule |
| F6c | A model family is lost from the models table: only the chosen configuration's family is listed. | `models.all_accounted` | the families and configurations of each profile |
| F6d | Data rows longer than the header are read silently, with no error. | `data.long_rows_stop` | rows longer than the header; lines pandas reads as a header |
| F7b | A time column that is not ISO 8601 is parsed leniently, so a day-first date runs on the wrong order. | `time.dayfirst_stops` | the accepted forms of the time column |
| F7c | Whole-number time keys are read as 64-bit floats, so distinct instants above 2⁵³ become one timestamp: the timestamps are no longer unique and the previous-value baseline is lost (D50). | `time.integer_keys_distinct` | integer keys through `read_time`, `split_time` and `profile()` |
| F7d | A table with a date column and no word from the caller about time order is split at random, with a warning, where it should stop (D58). | `time.undeclared_stops` | the stop, its message, and the two statements that lift it |
| F8b | With the entity column declared under time order, the headline is still scored against the training mean alone: a panel of random walks looks almost perfectly predictable (D57). | `time.panel_previous_value` | each entity's previous value by hand: the first holdout row, a gap, an entity with no earlier row |

**The four gate checks added with the fifth pass (D57, D58).** Two assert a
score, each in a band measured over 30 data seeds on the reference and on the
broken copy the check must catch (10.1); two are structural.

- `time.undeclared_stops` runs `time_blocks` with no time column and nothing
  said about time order, and `--duplicates keep` so that the copies rule is
  not what stops it: exit 3, `time_order_undeclared` on standard error, no
  `profile.json`. Without the stop that run finishes under a random split at
  skill near 1.
- `time.panel_previous_value` runs `panel`, twelve random walks on the same
  125 dates, with `--time-col date --group-col series`: both baselines carry
  the scores the gate recomputes from the source, the previous value of a row
  being the level of the same series on the date before; the headline baseline
  is the previous value; skill equals the gate's own and lies in the band;
  twelve entities are reported on each side and no holdout row is without a
  previous value.
- `groups.kept_together` runs `entities` with `--group-col customer`: the gate
  reads the holdout rows from `predictions.csv` and the entity of every row
  from the source, and requires that no entity is on both sides; that the
  scheme is `grouped`; that the entities counted on each side are the gate's
  own counts; that the column has the role `group`; and that the holdout share
  is between a quarter and a quarter plus two entities' rows.
- `groups.no_skill_on_entities` asserts the band on the same run.

Two existing runs are given the statement they now need, and nothing else
about them changes: `messy`, which has a column of dates, and the run of
`time_blocks` with no time column each pass `--no-time-order`. The check
`time.undeclared_warns` reads the second and also requires `input.time_col`
to be `false`. Until the application takes that argument, those two runs fail
on it, and with them the four checks that read them: `data.rows_reconcile`,
`data.columns_accounted`, `data.coercion_reported` and
`time.undeclared_warns`. They are therefore owned by the task that builds the
statement, by the reasoning of D54.

The two gate checks added with the fourth pass assert no score.
`data.holdout_copies_stop` asserts the exit, the named error and the absence
of a headline. `time.integer_keys_distinct` asserts that the counts of
training and holdout timestamps equal the gate's own, that the timestamps are
reported unique, that the holdout is the rows the gate expects, that
`train_end` and `holdout_start` are the exact integers, and that both
baselines carry the scores the gate recomputes, the previous value being the
one used. One gate run changed with D49: the run of `time_blocks` with no time
column passes `--duplicates keep`, because with the date column excluded every
one of its holdout rows has a copy among the training rows (D53; 10.6).

**Every gate check that names a failure mode is seen to fail on that mode's own
broken copy, in every run of the gate's self-test**, which prints the table
that proves it and fails if one pair is missing. The checks that give a
written reason in place of a failure mode are listed in the same output, with
the copies they failed on, for information.

### 8.1 Small faults the gate cannot see (D39, D44, D46, D52, D57)

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
| M6 | A prediction that is not finite is scored and written as it is, where it should fail its configuration (D44). ROC AUC is a count of ranks, so the score stays finite and a NaN reaches `y_pred`. | `test_non_finite.py::test_a_prediction_that_is_not_finite_counts_as_raising` |
| M7 | The fast profile's own error is never raised: too few timestamps under the row cap is always the ordinary `too_few_rows`, also where the full profile would run (D46). | `test_fast_profile_error.py::test_fast_error_stands_when_the_full_profile_would_run`, and the first pass's `test_time_pipeline.py::test_wide_panel_under_fast_raises` |

| M8 | The interval is the 5th to the 95th percentile of the resamples, and `interval_level` still says 0.95. | `test_guard_review3.py::test_m8_the_interval_is_the_middle_95_percent` |
| M9 | The leak screen forgets the previous-value baseline, so under time order the lag of a persistent series is excluded as a leak. | `test_guard_review3.py::test_m9_the_leak_screen_uses_the_previous_value` |
| M10 | The fast profile keeps the first 2,000 usable rows of the file, not a sample drawn from the seed. | `test_guard_review3.py::test_m10_the_fast_cap_is_a_sample_from_the_seed` |
| M11 | `cv_std` is the population deviation of the fold scores, not the sample deviation. | `test_guard_review3.py::test_m11_cv_std_is_the_sample_deviation` |
| M12 | In a resampled ROC AUC, tied predictions do not share their rank. | `test_guard_review3.py::test_m12_tied_predictions_in_a_resample` |
| M13 | `n_imputed` is counted over every loaded row, not over the training and holdout rows. | `test_guard_review3.py::test_m13_n_imputed_counts_rows_that_reach_a_model` |
| M14 | An object column is read a second time over the rows that will be used only, so an odd cell in a row that is dropped for another column's empty cell is never seen (the fourth review, finding 1). | `test_read_once.py`, the first four tables of step 1a (the fifth is told apart only by how the stamps are written, which the reference does one way under either reading) |
| M15 | With an entity column the holdout is whole entities, but the folds are cut through them. | `test_groups_random.py::test_no_entity_on_both_sides_or_in_two_folds` |
| M16 | Under time order with an entity column the leak screen forgets each entity's previous value, so the lag of a persistent panel is excluded as a leak. | `test_groups_time.py::test_the_leak_screen_measures_a_lag_against_the_entity_baseline` |
| M17 | With an entity column the fast profile's cap keeps a sample of rows, whatever their entity. | `test_groups_random.py::test_the_fast_cap_keeps_whole_entities` |
| M18 | The check that no number of the output is infinite or not-a-number covers five parts of the document, not every part (finding 4). | `test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite` |
| M19 | A panel with an autocorrelated target and nothing said about entities goes ahead with a warning, where it should stop and ask (D58). | `test_panel_stop.py::test_a_persistent_panel_with_no_entity_column_stops` |
| M20 | The infinite cells of the target and of the time column are left out of `n_infinite` (finding 3). | `test_guard_review4.py::test_m20_infinite_cells_of_the_target_and_the_time_column_are_counted` |
| M21 | Under the grouped scheme the bootstrap draws rows one by one and still reports `bootstrap_groups`. | `test_groups_random.py::test_the_interval_resamples_whole_entities_and_there_is_no_p_value` |
| M22 | With `duplicates="group"` the copies are found and counted, and the split is made row by row all the same. | `test_duplicates_group.py::test_each_set_of_copies_stays_on_one_side_and_in_one_fold` |

M6 and M7 were added with the third pass (D44, D46), for two guards that pass
against the application as it stood and must go on passing. The forms F6b,
F6c, F6d, F7b, F7c, F7d and F8b above, and F9 and F10, are run through the
suite in the same way, and each must fail at least one test.

M14 to M22 were added with the fifth pass (D57, D58). M20 is a guard: the
application already counts those cells, and the reference did not. The others
are faults in behaviour the application does not have yet, each the plausible
half-built form of one new rule: said and not done (M15, M17, M21, M22), done
for the baseline and forgotten in the screen (M16), or the old behaviour left
in place (M14, M18, M19). M21 is caught by effect, as M4 is: with 40 holdout
entities of ten rows that are each wholly right or wholly wrong, the interval
on ROC AUC is 0.25 to 0.35 wide when whole entities are resampled and 0.08 to
0.10 wide when single rows are (the reference, 20 seeds), and the test asks
for one wider than 0.18.

M8 to M13 were added with the fourth pass (D52). The third review planted
each of these six faults in a copy of the application and found that all 411
tests still passed; the gate caught two of them (M12 and M13) and nothing
caught the other four. The application has none of them. Each now has a guard
that passes today and fails on its variant:

- **M8.** The percentiles are pinned (step 12) and how the resamples are drawn
  is not, so no single run can tell 2.5 and 97.5 from 5 and 95. The guard runs
  the same fixed predictions, supplied through the seam, under 60 seeds, and
  compares the mean of each end of the interval with the 2.5th and 97.5th
  percentiles of a bootstrap of its own with many more resamples, in units of
  that bootstrap's standard deviation. Measured on the reference: +0.03 at the
  lower end and −0.04 at the upper; with the fault, +0.26 and −0.35. The
  guard allows 0.18.
- **M10.** Which 2,000 rows are kept is not pinned either, only that they are
  a sample from the seed. The guard asks for what no head of the file can
  give: a holdout row beyond position 2,000, and a different sample from a
  different seed.
- **M9, M11, M12, M13** are exact: a lagged column that must stay a feature
  with no `suspected_leak`; `cv_std` against the sample deviation of
  `fold_scores` by hand; a constant prediction, whose every resample scores
  exactly 0.5, so that the interval is 0.5 to 0.5; and `n_imputed` on a table
  whose missing cells sit partly in rows with no target.

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
| a line of a form feed, a no-break space, a vertical tab, an em space, two quotes or a quoted space, before a header of **two** columns (measured 2026-11-20) | a header of one column; the real header and every row after it give their first field to an `Index` and keep one column | with two columns the leading field becomes a plain index, where three give a `MultiIndex`; both implementations stop with `input_error` on all six, as before a header of three (D46) |
| a text cell that holds only spaces or tabs | text, as written (`'  '`), never missing | in the target it looks like a blank and is a missing target (D48); in a feature it is the level of empty text (D42) |
| a text cell written `yes ` | text, with its space | in the target, as in a feature, the white space is not part of the value (D46) |

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
- **A squared error overflows long before a value does** (measured 2026-11-20).
  The largest double is about 1.8e308, so the square of anything above about
  1.34e154 is infinite, and what is computed from it is not-a-number. One
  target cell of 1.7976931348623157e308 among values near 20, or the target
  times 1e160, gave the application a null score, a null baseline and a null
  skill with a p-value of 1/201 and no warning; the reference stopped with a
  raw `ValueError` from `json.dumps`. `numpy.mean` and `numpy.std` also
  overflow on the way to an answer that can be held: `numpy.std` of five
  values near 1e160 is infinite. Hence D44.
- **scikit-learn's trees and forests read their input as 32-bit floats** and
  raise `ValueError: Input X contains infinity or a value too large for
  dtype('float32')` on a finite value above 3.4028234663852886e+38 (measured
  2026-11-20). The leak screen's tree reads a column as it is, so such a cell
  in any row a tree is fitted on stopped both implementations with that raw
  error: the application on a single cell of 1e39, the reference on a column
  of them. Where the cell fell only in the holdout the run finished, with a
  linear model's prediction for that row, and a squared error, to match.
  Hence D48.
- **A seed of 2³² or more is refused by scikit-learn** with a raw
  `InvalidParameterError` (measured 2026-11-20: from the leak screen's tree in
  the application, from `train_test_split` in the reference), which the
  command line turned into a traceback and exit 1. Hence D45.
- A complex-number column counts as numeric to `pandas.api.types.is_numeric_dtype`,
  and turning it into floats drops the imaginary part with a warning nobody
  sees. Both implementations ran a regression on such a target (measured
  2026-11-20). Hence D48. The reference went on ordering a complex time column
  that way until the fourth pass; the application refused it.
- **Whole numbers and 64-bit floats** (measured 2026-11-20 for the fourth
  pass, `docs/measurements/CP5/facts.py`). A 64-bit float holds every whole
  number up to 2⁵³ and only some above: 200 consecutive `int64` values from
  2⁵³ + 1 are 101 distinct floats; 1,500 nanosecond stamps 100 apart at
  1.7 × 10¹⁸ are 587, the spacing of floats there being 256; 50 distinct
  `uint64` values near 2⁶³, 7 apart, are one float. `numpy.unique` and a sort
  on the integers keep every one. `numpy.diff` of the two ends of the `int64`
  range overflows without a word, so a "consecutive" test must not subtract
  in 64 bits. Hence D50.
- **How `read_csv` loads whole numbers** (same measurement): up to the largest
  `int64`, as `int64`; above it and up to the largest `uint64`, with no
  negative value, as `uint64`; beyond that in either direction, as an `object`
  column of Python integers, exact (with an empty cell: Python integers and
  NaN). A value above the `int64` range and within the `uint64` range, beside
  a negative one, loads as text. A whole-number column with an empty cell
  loads as `float64`, so 2⁵³ + 1 arrives as 2⁵³.
  `numpy.isinf` raises `TypeError` on a Python integer beyond 64 bits, which
  is where the application stopped on such a column in every role.
- **Objects that are numbers** (same measurement): for an `object` column of
  integers, `pandas.api.types.is_numeric_dtype` is false and `is_string_dtype`
  is true, so a test by type calls it neither a number column nor, correctly,
  text. `pandas.isna(Decimal("NaN"))` is true; on `Decimal("sNaN")` it raises
  `decimal.InvalidOperation`, and so does `Series.isna()`. `float()` of a
  `Decimal` beyond the largest double is infinite; `float()` of a Python
  integer that large raises `OverflowError`. Hence D51 and step 1a.
- **Dates beyond the nanosecond range** (same measurement): pandas 3 parses
  `9999-12-31` and keeps it in a coarser unit, and turning that column into
  nanoseconds raises `OverflowError`. pandas parses a fraction of ten digits
  by cutting it to nine. It writes a microsecond timestamp to CSV with six
  digits. Hence the fraction rule of step 3 (D52).

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
| `groups.no_skill_on_entities` | `entities`, `--group-col customer` | −0.260 to +0.168 (−0.027, 0.119) | −0.57 to 0.52 | F10: 0.760 to 0.882 | 0.24 |
| `time.panel_previous_value` | `panel`, `--time-col date --group-col series` | −0.023 to +0.006 (−0.003, 0.007) | −0.04 to 0.03 | F8b: 0.995 to 0.999 | 0.96 |

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

### 10.5 Measured for the third pass (D44 to D48)

On 2026-11-20; scripts and tables in `docs/measurements/CP4/`.

- **No gate value moved.** Between the judge before CP4 and after it, at the
  gate's seed: 19 datasets, 31 checks, 100 recorded values and all 14,302
  values of 24 runs on the reference are identical. The only differences are
  the failure modes four checks name (F6b, F6c, F6d, F7b in place of F6 and
  F7). No band, seed or dataset was changed, and none was added: everything
  A to E decided can be caught by a test.
- **Every pair of a gate check and a failure mode it names was seen failing:**
  22 of 22, over 12 broken copies of the reference, each run twice. Before,
  four pairs had never been (section 8).
- **The changed reference fails none of the 370 cases of the first two
  passes**, so nothing decided in D44 to D48 contradicts a locked test. Each
  of F6b, F6c, F6d and F7b fails existing tests (8, 15, 3 and 6 cases).
- **The cut-off of D48** is the largest 32-bit float, 3.4028234663852886e+38.
  A value just above it rounds to it and would pass scikit-learn's own check;
  the rule is the simpler "greater than", so nothing the trees refuse is ever
  handed to them. 1e38 is kept and 1e39 is missing.
- **The previous value** exists for every validation and holdout row: 8,175
  splits, of 100 to 400 unique timestamps and every gap the split accepts,
  and 1,898,456 rows checked in each implementation; none without one.

### 10.6 Measured for the fourth pass (D49 to D54)

On 2026-11-20; scripts and tables in `docs/measurements/CP5/`.

**The rule on copies (step 8a; D49, D53).** Holdout rows with a copy among the
training rows, counted by the reference implementation with
`duplicates="keep"` and again by the measurement script through the seam; the
two counts agree on every run that has one (1,440 runs were made in all). 30
data seeds (1115, 1215, ...), each dataset as the gate runs it.

| Data | Holdout rows | Rows with a copy | Share | Runs that meet the rule |
|---|---|---|---|---|
| every gate dataset run under the random scheme but the two below (13 runs) | 100 to 500 | 0 | 0% | 0 of 390 |
| the README's example table, 30 tool seeds, both profiles; the suite's copy of it | 100; 60 | 0 | 0% | 0 of 120 |
| `int_stamps` with no time column declared (not a gate run) | 375 | 0 to 1 | 0% to 0.27% | 0 of 30 |
| `time_blocks` with no time column declared (the gate run `time_blocks_undeclared`) | 500 | 494 to 500 | 98.8% to 100% | 30 of 30 |
| `dup_noise`: noise with every row present twice | 375 | 263 to 305 | 70.1% to 81.3% | 30 of 30 |
| `signal_clf` and `signal_reg` with 0.5% of their rows written twice | 377 | 0 to 6 | 0% to 1.59% | 0 of 60 |
| the same, 1% | 379 | 1 to 10 | 0.26% to 2.64% | 1 of 60 |
| the same, 2% | 383 | 4 to 15 | 1.04% to 3.92% | 47 of 60 |
| the same, 5% | 394 | 20 to 37 | 5.08% to 9.39% | 60 of 60 |
| the same, 10% | 413 | 46 to 71 | 11.1% to 17.2% | 60 of 60 |
| the same, 25%, 50%, 100% | 469 to 500 | 121 to 285 | 25.8% to 57.0% | 180 of 180 |

So data with no repeated rows sits at 0 to 1 rows and at most 0.27%, far below
both parts of the rule, and data with 5% or more of its rows repeated sits at
20 rows and 5.08% or more, above both. Between the two, at 1% to 2% repeated
in a table of 1,500 rows, it is the count of 10 that decides, and the outcome
varies with the seed. One existing gate run meets the rule: `time_blocks` with
no time column, where the rows of a block of five dates differ in nothing but
the date. It passes `--duplicates keep` (D53), and the message of the error it
would otherwise raise names the undeclared date column.

In the locked suite, two tests profile a table with repeated rows under the
random scheme: 2 of 31 holdout rows have a copy in one, 2 of 38 in the other
(by the reference's split). Neither meets the rule, by its count; a share
alone at 1% would have stopped both, which is why D53 added the count.

**Whole-number timestamps (step 3; D50).** `int_stamps`: 1,500 rows, one
nanosecond stamp each, 100 apart from 1.7 × 10¹⁸. Read as integers: 1,125
training and 375 holdout timestamps, unique, both baselines, the previous
value used. Read as floats (the reference before the change, and its copy
F7c): 439 and 148, not unique, no previous-value baseline.

**No existing gate value moved.** Between the judge before CP5 and after it,
at the gate's seed: the 19 datasets, 31 checks, 100 recorded values and
14,302 values of 24 runs that existed before are identical but for two, both
in the run `time_blocks_undeclared`: its command line, which gained
`--duplicates keep`, and its console summary, which gained the line of the
`holdout_copies_kept` warning. Every run that writes a `profile.json` gained
the three new keys (`input.duplicates`, `validation.n_holdout_copies`,
`validation.holdout_copy_share`) and that run its new warning. Two datasets,
two runs and two checks were added, last in their lists, so no seed moved. No
band was changed and none was added.

**Each new gate check was seen failing before the reference was changed:**
`data.holdout_copies_stop` (the run exited 0 with a headline) and
`time.integer_keys_distinct` (439 and 148 timestamps against 1,125 and 375).
Both pass on the changed reference, and each fails on its broken copy in the
self-test, F9 and F7c.

**The type matrix (section 7).** 805 cases. On the changed reference 484 end
in a result and 321 in a named error; none in anything else. On the
application as it stood, 37 end in a raw exception, from the four causes
section 7 lists; on the reference as it stood, 18.

**The changed reference against the 411 cases of the first three passes:** it
fails one, `test_skeleton.py::test_profile_signature_and_defaults`, which pins
the nine-parameter signature that D49 extends. D54 has the tests stage loosen
that one test; with it loosened the reference fails none. Nothing else decided
in D49 to D53 contradicts a locked test.

### 10.7 Measured for the fifth pass (D56 to D60)

Measured on 2026-11-21, on the reference as changed for this pass, 30 data
seeds (the gate's base seed 1115, then 1215, and so on), fast profile, one
core, unless stated. The scripts behind the two bands and behind the interval
and the p-value under the grouped scheme, with every value seed by seed, are
in `docs/measurements/CP6/bands/` (D35, D61).

**The two new bands (10.1).** `entities`: 300 entities of five rows, six
measures alike within an entity, a label drawn per entity; with the entity
column declared the reference's skill is −0.260 to +0.168 (mean −0.027, sd
0.119), the holdout is 375 to 380 rows of 75 or 76 entities, and no run stops.
With the split made row by row (F10) it is 0.760 to 0.882. `panel`: twelve
random walks on 125 business days with their own lag as the one feature; with
the series declared the baseline is the previous value in all 30 runs and
skill is −0.023 to +0.006 (mean −0.003, sd 0.007); scored against the
training mean (F8b) it is 0.995 to 0.999.

**The interval and the p-value under the grouped scheme (step 12).** On
`entities` the true skill is zero, so a sound 95% interval holds zero in about
19 runs of 20 and a sound p-value is below 0.05 in about one of 20. From the
same holdout predictions, 1,000 resamples or shuffles each:

| Method | Outcome over 30 runs |
|---|---|
| interval from resampled rows | holds zero in 19 |
| interval from resampled whole entities | holds zero in 30 |
| p-value from the holdout labels shuffled row by row | below 0.05 in 5 (smallest 0.001) |
| p-value from labels shuffled entity by entity | below 0.05 in 0 (smallest 0.073) |

So the rows of an entity cannot be resampled or shuffled singly. The tool
resamples whole entities and reports no p-value: a shuffle of entities is
sound here only because each entity holds one label, which a table in general
does not, and no p-value is better than one that holds on some tables.

**Each new gate check was seen failing before the reference was changed.** On
the reference as it stood: `time.undeclared_stops`, the run exited 0 with a
headline; `time.panel_previous_value`, `groups.kept_together` and
`groups.no_skill_on_entities`, the command line refused `--group-col` (exit
2). All four pass on the changed reference, and each fails on its broken copy
in the self-test: F7d, F8b and F10.

**No other gate value moved.** Between the judge before CP6 and after it, at
the gate's seed: the 21 datasets, the 33 checks with their 100 values, and the
108 values the gate recorded are identical. Of the 14,573 values of the 26
runs that existed before, six moved, each because a change requires it: the
command lines of `messy` and `time_blocks_undeclared`, which gained
`--no-time-order`, and their `input.time_col`, which is now `false`; the text
of the copies error on `dup_noise`, which now names `duplicates='group'`; and
the text of the `task_ambiguous` error on `ambiguous_target`. Every run that
writes a `profile.json` gained the six new keys (`input.group_col` and five of
`validation`). Two datasets, three runs and four checks were added, last in
their lists, so no seed moved. No band was changed; two were added.

**The changed reference against the 1,313 cases of the first four passes**
(the README's own tests apart): it fails ten, in nine tests, each of which
pins something D57 or D58 changes: the ten-parameter signature (two tests),
a run on a table with a date column and nothing said about time order (five
tests, six cases), and a persistent panel with nothing said about entities
(two tests). D59 has the tests stage loosen exactly those; the test plan
lists each with its loosened form. With the prototypes of the loosened tests
in place the reference fails none.

**The entity role of the type matrix (section 7):** 353 cases on the changed
reference; none ends in anything but a result or a named error.

**The broader screen for repeated rows (D60)** was designed and measured in
this pass and is not part of this version. Its prototype, the first rule
proposed, D49's count, and the lead's runs of all three on eleven real public
tables are in `docs/measurements/CP6/`. In short: a rule that allows for
chance agreement stopped no clean made-up table and stopped clean diamonds,
taxis and daily prices, because real columns do not vary independently; and
D49's exact count itself stops clean tables of whole-number or few-valued
columns from about 4,000 rows up, which is what `duplicates="keep"` and
`"group"` are for.

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
  What a message must name is pinned in seven places and nowhere else: the
  excluded columns and their reasons in `no_features` (2.4), the columns
  involved in `feature_name_collision` (step 5), the number that is not finite
  in `non_finite_score` (section 3, "Numbers that are not finite"), what
  the console summary shows (2.2), and, since the fourth pass, three more.
  The message of `holdout_copies_in_training` holds the count and the number
  of holdout rows as whole numbers in digits, the share as a percentage (some
  number in it is within 0.1 of 100 × the share), the word `keep`, and, when a
  column was excluded as `datetime` with no time column declared, that
  column's name and the text `time_col` (step 8a). The message of
  `fast_profile_too_few_rows` holds the cap as `2000` or `2,000` and the word
  `full` (step 6). The message of `time_column_invalid` for a value refused
  only for the length of its fraction quotes that value and holds the word
  `fraction` (step 3). And, since the fifth pass, five more. The message of
  `time_order_undeclared` holds the name of every date column, each in quotes
  as `repr` writes it, and the texts `time_col`, `--time-col`,
  `time_col=False` and `--no-time-order` (step 5). The message of
  `entities_undeclared` holds the texts `group_col`, `--group-col`,
  `group_col=False` and `--no-entities` (step 11). The message of
  `task_ambiguous` is as step 4 pins it. The message of
  `holdout_copies_in_training` also holds the word `group`; and when it is
  raised under `duplicates="group"`, the name of every column excluded as a
  suspected leak (step 8a). The message of `group_column_invalid` holds the
  name of the entity column;
- under the grouped scheme, which units are held out and which fold each is
  in, beyond the properties of step 7; which entities the fast profile's cap
  keeps, beyond the properties of step 6; and the wording of `p_value_reason`
  there, which holds the words `independent rows` (step 12);
- what `stages.read_time` returns for a whole-number column that holds a
  magnitude above 2⁵³ and also a missing value (2.7);
- the text a class value of an unusual type is written as (step 4; D53);
- whether the preprocessing of the final fit is fitted again for a
  configuration tried after another one failed (step 10): each such try gets
  a fresh estimator, and that is all that is pinned;
- what `stages.skill("roc_auc", ...)` does with a baseline of 1, which no run
  can produce; and what `stages.skill`, `stages.mse` and `stages.roc_auc`
  return when the arithmetic overflows (`profile()` checks every number it
  reports: section 3, "Numbers that are not finite");
- how the numbers of the console summary are formatted, as long as each agrees
  with `profile.json` to four significant digits (2.2);
- at scales where every reported number can be held but arithmetic on the way
  overflows (a target near 1e80), whether the run finishes with finite numbers
  or stops with `non_finite_score`. It never writes a null there and never
  ends in `all_models_failed`;
- which of the leak screen's two baselines is named when both are not finite;
- what `stages.split_random` does with a `classes` value that is missing;
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
the p-value, the fixed texts of 5.4, the range of the seed (2.1), the
magnitude above which a feature value is read as missing (step 5), that the
baselines are scored before any fit (section 3, "Numbers that are not
finite"), which error a fast run raises when the full profile would stop
too (steps 6 and 7), how two rows are compared to be copies and both parts of
the rule on them (step 8a), which mixtures of objects are read as numbers
(step 1a), the magnitude above which whole-number time keys stay integers
(step 3), the five properties of a split of whole units (step 7), which
earlier row is an entity's previous value and what stands in for it when
there is none (step 11), that a resample under the grouped scheme draws whole
units and that fewer than 30 of them give no interval (step 12), that the
finite check covers every part of the output and is made in the constructor
(2.3), and that an object column is read once, over all loaded rows (step 1a).

**Known limits of this version, stated and not hidden (D56, D60):** a copy of
a row that differs from it in any feature column is not detected (step 8a);
the caller's control is `group_col`. A copy altered in every column cannot be
told from a different row by any tool. Under D56 phase 1 does not close until
the first of these is removed.

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
| D7 time-ordered validation | 3 (steps 3, 7); its warning for an undeclared date column is replaced by the stop of D58 |
| D8 models and selection | 3 (step 9), 4.1 |
| D9 uncertainty | 3 (step 12) |
| D10 regression baseline under time order | 3 (step 11); its warning for a panel is replaced by the stop of D58, and an entity column gives a panel its previous value (D57) |
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
| D44 no number that is not finite is written as empty | 2.4 (`non_finite_score`), 2.5 (`model_failed`), 3 (steps 8 to 13 and "Numbers that are not finite"), 5.1, 5.2, 7, 8.1 (M6), 9, 11 |
| D45 the seed's range | 2.1, 2.2, 2.4, 2.7 (`split_random`), 7, 9, 11 |
| D46 second review accepted | (1) the summary's significant digits: 2.2, 11. (2) every gate check seen to fail: 8, 8.1; `verify/smoke_test.py`, `verify/mutation_check.py`. (3) white space round a target value: 2.7, 3 (step 4), 5.1, 5.2, 5.3, 7, 9. (4) the fast profile's error: 2.4, 2.7, 3 (steps 6, 7), 7, 8.1 (M7). (5) the README: 1; `plan/test-plan.md`, task T30. (6) the self-test's own run folder: not in this spec; `verify/smoke_test.py`. (7) lines before a header: 6, 7, 9 |
| D47 when phase 1 ships | not in this spec; `docs/BACKLOG.md` |
| D48 three rulings from the CP4 session | (1) a feature value too large for the tree models: 2.5, 2.6, 2.7 (`Preprocessor`), 3 (steps 5, 8), 5.2 (`n_infinite`), 7, 9. (2) a complex-number target: 2.4, 3 (step 4), 7, 9. (3) a target cell of only white space: 2.7 (`usable_rows`, `decide_task`), 3 (steps 2, 4), 5.2 (`n_missing`), 7, 9 |
| D49 copies of a holdout row in training stop the run | 2.1 (`duplicates`), 2.2, 2.4 (`holdout_copies_in_training`), 2.5 (`holdout_copies_kept`), 3 (step 8a), 4.3, 5.2 (`input.duplicates`, `validation.n_holdout_copies`, `holdout_copy_share`), 7, 8 (F9), 10.6, 11; the cut-off is refined by D53 |
| D50 whole numbers are compared exactly | (1) integer time keys: 2.7 (`read_time`), 3 (step 3), 4.2, 5.2, 7, 8 (F7c), 9, 10.6. (2) distinct values counted exactly: 3 (step 5), 7, 9. (3) whole numbers too large for 64 bits: 2.4, 2.5, 3 (steps 1a, 3, 4, 5), 6, 7, 9; (1) is refined by D53 |
| D51 number columns as a database driver returns them | 2.5 (`unsupported_type`), 2.6, 2.7, 3 (steps 1a, 4, 5), 5.1, 5.2 (`dtype`, `target_dtype`), 6, 7, 9 |
| D52 third review accepted | (1) fractions of a second: 3 (step 3), 4.2, 7, 9, 11. (2) the six test gaps: 8.1 (M8 to M13); `verify/mutation_check.py`. (3) the fast profile's cap: 2.4 (`fast_profile_too_few_rows`), 3 (step 6), 7, 11. (4) the type matrix: 2.4, 7 ("The type matrix"), 10.6 |
| D53 rulings from the CP5 session's plan | (1) the rule needs a share and a count: 3 (step 8a), 4.3, 10.6. (2) no locked test is edited: `plan/test-plan.md`; amended by D54. (3) the message names an undeclared date column: 2.4, 3 (step 8a), 11. (4) the gate's undeclared-date run passes `--duplicates keep`: 8, 10.6; `verify/smoke_test.py`. (5) integer keys only above 2⁵³: 2.7, 3 (step 3), 4.2. (6) a target that cannot be compared; class values as written: 2.4, 3 (step 4), 5.2, 7. (7) a code of its own for the fast profile's row floors: 2.4, 3 (step 6) |
| D54 one locked test is loosened for the new argument | 2.1 (the ten parameters), 10.6; `plan/test-plan.md` ("Fourth pass"), tasks T32 and T11 in `plan/tasks.json` (the gate check `time.undeclared_warns` moves to T32) |
| D55 values approved after the CP5 session | (1) the cut-off of D49: 3 (step 8a), 4.3, 10.6. (2) a decimal that holds a whole number: 3 (step 1a). (3) limits accepted: 3 (steps 1a, 5), 9 |
| D56 phase 1 is not shipped with known defects | 1, 3 (step 8a, "What this step does not see"), 11 ("Known limits"); `docs/BACKLOG.md` |
| D57 entities and repeated rows | (1) `group_col`: 2.1, 2.2, 2.4 (`group_column_invalid`, `too_few_groups`), 2.5 (`no_previous_value`), 2.7 (`column_roles`, `split_random`, `previous_value`), 3 (steps 2, 5, 6, 7, 8, 11, 12), 4.2, 5.2, 5.4, 7 (and the entity role of the type matrix), 8 (F10, F8b), 8.1 (M15, M16, M17, M21), 10.1, 10.7. (2) the broader screen: withdrawn by D60. (3) `duplicates="group"`: 2.1, 2.2, 2.4, 2.5 (`copies_grouped`), 3 (steps 7, 8a), 5.2 (`n_rows_with_copies`), 7, 8.1 (M22). (4) the target's infinite cells: 3 (steps 2, 5), 5.2 (`n_infinite`), 7, 8.1 (M20). The fourth review's findings 1, 4, 5 and the three-class message: 2.3, 2.6, 3 (steps 1a, 4, 9, 13 and "Numbers that are not finite", rule 6), 7, 8.1 (M14, M18) |
| D58 the tool stops and asks where it used to warn | (1) an undeclared date column: 1, 2.1 (`time_col`), 2.2 (`--no-time-order`), 2.4 (`time_order_undeclared`), 2.5 (`date_column_undeclared`), 3 (steps 5, 8a), 5.2 (`input.time_col`), 7, 8 (F7d), 11. (2) the default of D49 stays `"error"`: 2.1, 3 (step 8a). (3) a panel with no entity declared: 2.1 (`group_col`), 2.2 (`--no-entities`), 2.4 (`entities_undeclared`), 2.5 (`autocorrelated_panel_target`), 3 (step 11), 5.2 (`input.group_col`), 7, 8.1 (M19), 11. D7 and D10 are replaced in those parts |
| D59 how a locked test follows a changed contract | 10.7; `plan/test-plan.md` ("Fifth pass"), tasks T39 to T47 in `plan/tasks.json` |
| D60 the broader repeat screen leaves pass 5 | 3 (step 8a: D49's rule unchanged, and what it does not see), 3 (step 7: `duplicates="group"` groups the exact copies), 5.2 (`n_holdout_copies` as a plain number), 7, 10.7, 11 ("Known limits"); `docs/measurements/CP6/` |
| D61 three rulings on the CP6 judge | (1) D49's rule whenever there is no time column: 3 (step 8a), 2.4 (`holdout_copies_in_training`), 5.2 (`n_holdout_copies`), 7. (2) `duplicates="group"` with any copy makes the scheme grouped: 3 (steps 7, 12), 5.2 (`scheme`), 7. (3) the scripts and the per-seed values of the fifth pass's bands: 10.7; `docs/measurements/CP6/bands/` |
