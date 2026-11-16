# Test plan: what the suite must prove (phase 1)

Written for the test author. You write the suite from `spec.md` alone, before
any application code exists. Section and step numbers below are those of
`spec.md`.

## Ground rules

- **Through the documented surface only** (2.1 to 2.7): `mlprofessor.profile`,
  `ProfileResult`, `mlprofessor.errors`, `mlprofessor.extensions`,
  `mlprofessor.stages` and `python -m mlprofessor`. Never import any other
  module of the package: the builder is free to reorganise them, and a test
  that names an internal module blocks a correct build.
- **Each file states its interface below, and keeps to it.** The build goes in
  order: the early tasks deliver one step function each and nothing that runs
  end to end. A test in `test_load.py` that calls `profile()` cannot pass when
  its task is judged, and blocks the build. The column "must pass with" says
  what exists when each file is first judged.
- **Pin what the spec pins, and nothing else.** Section 11 lists what is open:
  never assert the wording of a message, the row numbers a seed produces, or a
  score to more digits than the property needs. Assert properties: counts,
  disjointness, order, equality with a value the test recomputes.
- **Recompute, do not trust.** Where a test checks a count or a score, it works
  the expected value out itself from the table it built: by hand for five to
  ten rows, with plain Python or numpy otherwise. Never from another field of
  the output, and never with scikit-learn's own metric functions.
- **Tiny data.** Tables built in code from a fixed seed, or committed under
  `tests/fixtures/` as CSV, 40 to 400 rows. A run of `profile()` needs at least
  100 usable rows and 10 of each class (4.2). The whole suite runs in well
  under a minute: use `profile="fast"` and `n_jobs=1` unless the test is about
  the full profile or about `n_jobs`. One exception: the middle case of the
  leak screen (`test_leakage.py`) uses 1,500 rows, because on 400 rows no
  column sits clearly between the two cut-offs (spec 10.2).
- **No skipped or expected-to-fail tests.** The driver counts a skip as a
  failure. If something cannot be tested, leave it out and say so in a comment.
- **Name the failure mode.** A test that guards a named failure carries its id
  in its name: `test_f1_...` to `test_f8_...` (section 8). Each of F1 to F8 has
  at least one such test, in the file given below.
- **Stay away from cut-offs.** Test each rule with one case clearly on each
  side: a column 100% distinct and one 20% distinct; a copy of the target and a
  column with single-column skill near 0.6; 100%/0% missing by class and
  60%/40%; 10 and 200 holdout timestamps. The leak screen has two cut-offs,
  0.98 and 0.90, and so three cases: the copy (above both), the column near 0.6
  (below both), and one clearly between them. You cannot measure a column's
  single-column skill yourself before the application exists, so build the
  middle case by a recipe of spec 10.2, which gives its measured range.
- **Recording fakes** are the instrument for everything from `test_pipeline.py`
  on. Build them once in `tests/conftest.py`: a model and a preprocessing step
  that record `list(X.index)` on every `fit`, `transform` and `predict`,
  supplied through `extensions.use` (2.6). `X.index` holds source row
  positions, the same numbers as `result.predictions["row"]`. A fake model can
  also return predictions the test chooses, which is how a metric is checked by
  hand through a whole run.
- A helper in `conftest.py` runs the command line in a new process with the
  repo root as the working directory and the interpreter running the tests
  (`sys.executable`). **One process at a time, never in parallel.** Each start
  costs about 3 seconds, most of it importing the libraries, so keep the
  number of process starts small; most tests call the function in-process.
  (Until D32 moved the interpreter's base off a mapped drive, processes started
  at the same instant on this machine blocked each other for 35 or 70 seconds.
  Measured since: 60 rounds of eight simultaneous starts, none blocked; spec
  section 9.)

## Files

| File | Interface | Must pass with |
|---|---|---|
| `tests/test_skeleton.py` | import, `--help` | the skeleton only (T1) |
| `tests/test_load.py` | `stages.load_table`, `stages.usable_rows` | T1 to T2 |
| `tests/test_columns.py` | `stages.column_roles` (tables from `load_table` or built in code) | T1 to T3 |
| `tests/test_task.py` | `stages.decide_task` | T1 to T4 |
| `tests/test_split_random.py` | `stages.split_random` | T1 to T5 |
| `tests/test_split_time.py` | `stages.read_time`, `stages.split_time` | T1 to T6 |
| `tests/test_metrics.py` | `stages.roc_auc`, `mse`, `skill`, `previous_value` | T1 to T7 |
| `tests/test_preprocess.py` | `stages.Preprocessor` | T1 to T8 |
| `tests/test_pipeline.py` | `profile()` in memory, with `extensions.use`. Only `result.predictions`, `.models`, `.rows`, `.columns`, `.warnings` and the keys `metric`, `score`, `model`, `baseline`, `baseline_score`, `skill` of `.headline`. No `time_col`, no `to_dict`, `to_json`, `out_dir` or command line, no `profile="full"`. | T1 to T9 |
| `tests/test_result_schema.py` | `profile()` with `out_dir`, `to_dict`, `to_json`, the command line; random scheme, fast profile | T1 to T10 |
| `tests/test_time_pipeline.py` | `profile()` with `time_col` | T1 to T11 |
| `tests/test_leakage.py` | `profile()` | T1 to T12 |
| `tests/test_uncertainty.py` | `profile()`, with `extensions.use` | T1 to T13 |
| `tests/test_model_families.py` | `profile(profile="full")`, with `extensions.use` | T1 to T14 |
| `tests/test_full_profile.py` | `profile(profile="full")`, command line | T1 to T15 |
| `tests/test_edges.py` | everything | T1 to T16 |
| `tests/test_readme.py` | `README.md` | all |

### `tests/test_skeleton.py`
The package imports. `mlprofessor.profile` has the signature and defaults of
2.1 (`inspect.signature`). `mlprofessor.ProfileResult`, `mlprofessor.errors`,
`mlprofessor.extensions.use` and the module `mlprofessor.stages` exist. Every
error class of 2.4 exists, subclasses `ProfileError` and has the stated `code`.
`python -m mlprofessor --help` and `python -m mlprofessor profile --help` exit
0, and the second names every option of 2.2. `requirements.txt` pins pandas,
numpy, scikit-learn and threadpoolctl with `==`. Do not call `profile()` or any
function of `stages` here.

### `tests/test_load.py`: `stages.load_table`, `stages.usable_rows`
- A DataFrame and the same table written to CSV give equal frames for the
  types a CSV can carry: integers, floats, floats with missing cells, text
  (section 6). The index of the result is 0..n−1 in both cases, whatever the
  DataFrame's own index; the DataFrame passed in is not modified.
- Blank lines in a CSV are not rows (`test_f6_...`): a file with blank lines
  between data lines gives one row per data line, in order.
- The missing-value spellings of 4.4 load as missing; `1,200` and `abc` do not.
- Duplicate column names raise `DuplicateColumnsError`, for a CSV (where pandas
  alone would rename them) and for a DataFrame. A blank header cell is kept
  under pandas' name and reported in `info["warnings"]` as `blank_header`.
- A missing file, an empty file, rows longer than the header, and a value that
  is neither a DataFrame nor a path raise `InputError`.
- `info["kind"]` and `info["source"]` for both kinds of input.
- `usable_rows`: rows with a missing target are dropped and counted, and the
  three numbers add up to the rows loaded (`test_f6_...`); with `time_col`,
  rows with a target but no time value are counted separately and a row missing
  both counts under the target; an infinite numeric target counts as missing; a
  missing column raises `ColumnNotFoundError`.

### `tests/test_columns.py`: `stages.column_roles`
- Every input column appears exactly once, in file order, with all eleven keys
  (`test_f6_...`).
- One crafted column per rule of step 5, each asserting role, reason and, for
  features, kind: constant (numeric and text); all missing; datetime-typed;
  text dates in several forms of 4.5; free text; a text id; a column of lists
  (`unsupported_type`); a category-typed column with a few levels; boolean and
  nullable-integer columns (numeric features).
- Not dates: `1,200`-style numbers, `May`, `09:30`, `2024Q1`, `1/2` (4.5).
- The coercion rule: a numeric column with a few text cells of the `abc` kind
  (`abc`, `unknown`, `12 units`, `-`) is a numeric feature and `n_coerced`
  equals the number of such cells; `n/a` cells in it count in `n_missing`, not
  `n_coerced`; a column where half the cells are text is not numeric
  (`test_f6_...`).
- Numbers with thousands separators (step 5, rule 4a; D28, D34), through role
  and reason only: `column_roles` returns no warnings, and the warning is
  tested in `test_result_schema.py`. Two cases must both be excluded with the
  reason `numeric_with_separators` and `n_coerced` 0:
  - a column written with separators throughout: `1,200`, `-3,450.75`,
    `+12,345,678`, with its values below 1,000 written plain (`532.10`);
  - a column of plain numbers in which a few cells are written `1,200`, and
    one in which a single cell is. Neither is a numeric feature with those
    cells coerced (`test_f6_...`: the cells lost would be its largest values).
- Around that rule:
  - The pattern, not matching: the same amounts written as `1.200,50`, and as
    `12,34`, never get that reason; they fall to the rules that follow (id-like
    or free text, by their distinct values). A column of plain numbers with a
    few cells written `$1,200`, `1,200.` or `1,200 units` is a numeric feature
    with those cells in `n_coerced`.
  - The 90%: a column in which 85% of the values match or parse and the rest
    are words does not get the reason.
  - Precedence: a column written with separators in which about 60% of the
    values are distinct is `numeric_with_separators`, where the same shape in
    another form is `free_text`; one with every value distinct is
    `numeric_with_separators`, not `id_like`; one holding five distinct values
    is `numeric_with_separators`, not a categorical feature; one holding a
    single value, `1,200` in every row, is `constant`.
- Infinite values are counted in `n_infinite`.
- Counts are over every row of the frame, including rows with no target.
- The id-like rule (D19): a consecutive id in order, the same shuffled, an id
  in order with blank cells (so the column is float), an increasing id that is
  not consecutive: all `id_like`. A whole-number measure with every value
  distinct but unordered and not consecutive, and one with repeats: features.
  A float column with all values distinct: a feature.
- Roles are judged on the usable rows: a column constant on the rows that have
  a target is `constant` even if it varies on rows without one.
- `leak_rule` is null and `n_imputed` is 0 throughout (2.7).

### `tests/test_task.py`: `stages.decide_task`
- The rule on each shape as pandas loads it from a CSV (write the file, load it
  with `stages.load_table`, pass the column): 0/1 integers; 0/1 with blank
  cells, which loads as float; `yes`/`no`; `True`/`False`; two floats. All
  classification, `decided_by` rule, `n_distinct` 2 (`test_f5_...`).
- More than 20 distinct numeric values: regression. 3 to 20 distinct numeric
  values (`test_f5_...`), and more than two text values: `TaskAmbiguousError`.
- A declared task is honoured (regression on 0/1; classification on two text
  values) with `decided_by` declared, or refused with `TaskUnsupportedError`
  (classification with three values; regression on text).
- One distinct value: `TargetConstantError`. Missing values are ignored.
- `classes` is sorted and `positive_class` is the second: `1` of 0/1, `"yes"`
  of no/yes, `True` of False/True.
- `warnings` holds `whole_number_regression_target` when the rule picks
  regression on whole numbers; it is empty on fractional targets and when the
  task was declared.

### `tests/test_split_random.py`: `stages.split_random`
- Holdout and training are disjoint and together are 0..n−1.
- The holdout is ⌈n/4⌉ rows; with `classes`, each class's holdout count is
  within one of a quarter of the class, for balanced and for 90/10 classes.
- The five folds partition the training rows: each training row is in exactly
  one validation part; sizes differ by at most one; each fold's training part
  is the training rows outside its validation part; with `classes` each
  validation part holds both classes in proportion.
- The same seed gives the same split; another seed gives another.
- No fold's training part contains a holdout row (`test_f1_...`).

### `tests/test_split_time.py`: `stages.read_time`, `stages.split_time`
- `read_time` accepts a datetime column, a numeric column, ISO text with and
  without a time, and reports the form; the keys sort in time order; a missing
  value gives `NaT` or NaN.
- Day-first text, month-first text, `2024-01`, a boolean column, an impossible
  date and mixed offsets raise `TimeColumnError` (`test_f7_...`).
- Every training key precedes every holdout key (`test_f7_...`).
- No timestamp straddles the cut, on keys with several rows per timestamp.
- The holdout is the smallest set of latest timestamps holding at least 25%:
  the test computes it from the keys.
- The result depends on the keys, not on their order: shuffled keys give the
  same rows (`test_f7_...`).
- Six blocks: fold *k* validates on block *k*+1; in every fold each training
  key precedes each validation key; block sizes follow step 7.
- The gap removes exactly the stated timestamps before the holdout and before
  every validation block, and they are in no training part (`test_f7_...`).
  Rows in a fold's gap stay in the overall training part.
- Fewer than six training timestamps raises `TooFewRowsError`; a gap that
  leaves too few raises `GapError`.

### `tests/test_metrics.py`: `stages.roc_auc`, `mse`, `skill`, `previous_value`
- ROC AUC against values worked out by hand on arrays of five to ten elements:
  no ties, ties between classes, all scores equal (0.5), perfect and inverted.
- A constant prediction on a 95/5 target scores ROC AUC 0.5, skill 0, although
  its accuracy would be 0.95 (`test_f4_...`); `skill("accuracy", ...)` raises.
- One class raises `OneClassError`.
- Squared error and skill for regression by hand, including negative skill.
- `previous_value`: the target `gap + 1` timestamps earlier for `gap` 0 and 2,
  on keys given in shuffled order; NaN where there is none; repeated keys raise
  (`test_f8_...`).
- On a short random walk built in the test, the squared error of
  `previous_value` is far below that of the mean (`test_f8_...`).

### `tests/test_preprocess.py`: `stages.Preprocessor`
- The statistics come from the rows given to `fit` and from no others: fit on
  rows A, transform rows B; the median used for B's missing cells, the mean and
  deviation, and the levels are A's, by hand; changing B does not change the
  transform of A (`test_f1_...`).
- A missing numeric cell gets the median; a column all missing in `fit` gets 0;
  a zero deviation is read as 1.
- A missing categorical cell is the level `(missing)`; a level unseen in `fit`
  is all zeros; more than 30 levels keeps the 30 most frequent.
- Output index, column names and order (2.7).

### `tests/test_pipeline.py`: `profile()` in memory, random scheme, fast profile
Keep to the members listed in the table above.
- A recording fake model proves no `fit` ever receives a holdout row, and a
  recording fake step proves no preprocessing `fit` does either
  (`test_f1_...`). In each fold neither receives a row of that fold's
  validation part. One fresh step per fit: five folds and the final fit.
- The built-in preprocessing too, by effect: on a table where the holdout rows
  are found first (run once, read `predictions.row`), replacing the feature
  values of those rows with extreme values changes no `fold_scores`
  (`test_f1_...`).
- Only the chosen configuration predicts on the holdout rows.
- The headline is the chosen configuration's holdout score, not a
  cross-validation score: a fake family whose predictions are perfect on every
  row it was fitted on and random elsewhere makes `cv_mean` and the holdout
  score differ by construction; `headline.score` equals the metric the test
  recomputes from `predictions` (`test_f3_...`).
- The metric by hand through a whole run: a fake model returns predictions the
  test chose; `headline.score` and `headline.skill` equal the test's own
  arithmetic, for both tasks; the metric is `roc_auc` and the baseline `chance`
  at 0.5 for classification (`test_f4_...`), `mse` and `training_mean` for
  regression.
- Selection picks the best `cv_mean`: highest for classification, lowest for
  regression; a tie goes to the earlier family, then the earlier grid position.
- `rows` adds up; missing-target rows are counted; on a table of more than
  2,000 usable rows `dropped_row_cap` is the excess and training plus holdout
  is 2,000 (`test_f6_...`).
- The run's holdout is `stages.split_random` applied to the usable rows (2.7).
- `predictions.row` is the position in the source: each `y_true` equals the
  source target there, with missing-target rows in the table.
- `y_pred` is the probability of the positive class: with a planted signal the
  mean `y_pred` is higher on positive rows, for 0/1 and for `no`/`yes`.
- Shuffling the target of a small planted-signal table removes the skill.
- Adding a constant column changes no score.
- The same seed gives the same tables; another seed another holdout.
- Inside a fake model's `fit`, `threadpoolctl.threadpool_info()` shows every
  BLAS pool at one thread, with `n_jobs` 1 and 2.
- The default families under `fast` are `linear-0` and `tree-0`, both fitted.

### `tests/test_result_schema.py`: files, `to_dict`, command line
- `profile.json` has every key of section 5 with the stated types, for a
  classification run and a regression run; no NaN or infinity in the file
  (parse with a hook that rejects them). Fields that a later task fills
  (the interval and p-value fields of `headline`, `limitations`) are asserted
  to be present here and their content is checked in their own files.
- Each DataFrame of `ProfileResult` equals its part of the JSON (2.3);
  `to_dict()` equals the file; `to_json(path)` writes the same content.
- `predictions.csv` reconciles with the source (5.3 rule 3) and with
  `ProfileResult.predictions`; `rows` reconciles (rule 1); `columns` lists every
  column once (rule 2); the headline is recomputed from the file (rule 4).
- `meta` records the profile, seed, `n_jobs` and the library versions.
- A CSV path and the same table as a DataFrame give the same result apart from
  `input.source`, `input.kind` and `meta.timing`.
- Two runs with the same seed give identical files apart from `meta.timing`.
- The command line writes the same two files as the function, prints a
  summary, and exits 0; a named error exits 3 with `error: <code>:` on standard
  error and writes nothing; a bad option exits 2.
- Warnings that arise in steps 1 to 11 are in the output with their code and
  column: `values_coerced`, `infinite_values`, `blank_header`, `duplicate_rows`,
  `whole_number_regression_target`, `small_holdout`.
- `numeric_with_separators`, which has one form (D34): each column excluded
  with that reason is named by exactly one such warning, whether it is written
  with separators throughout or is plain numbers with a few cells written
  `1,200`, and that column carries no `values_coerced`. A numeric feature whose
  coerced cells are words (`abc`, `12 units`) carries `values_coerced` and is
  named by no `numeric_with_separators` warning.

### `tests/test_time_pipeline.py`: `profile()` with `time_col`
- The time-ordered scheme is used when and only when `time_col` is given; the
  time fields of `validation` are populated or null accordingly.
- The run's holdout, gap and folds are `stages.split_time` applied to the
  usable rows' keys; a shuffled table gives the same holdout rows
  (`test_f7_...`); a recording fake proves no fit receives a row at or after
  the first holdout timestamp, or inside its gap (`test_f7_...`).
- `dropped_gap`, `dropped_missing_time` and `validation.folds[].gap_rows`.
- The fast profile keeps the most recent whole timestamps within 2,000 rows; a
  wide panel raises `FastProfileError`; a day-first time column raises
  `TimeColumnError` (`test_f7_...`).
- `date_column_undeclared` when a date column exists and `time_col` is not
  given, with the random caveat; absent when it is given.
- Unique timestamps: both baselines are reported and the one with the lower
  error is the headline baseline; on a random walk that is `previous_value`
  and skill is near zero (`test_f8_...`). Repeated timestamps: one baseline;
  `autocorrelated_panel_target` on a strongly persistent panel, absent on an
  independent one.

### `tests/test_leakage.py`: `profile()`
- A copy of the target and a rescaled copy are excluded with `predicts_target`;
  a column filled only for one class is excluded with `missing_by_class`; each
  has a `suspected_leak` warning and reaches no model (a recording fake sees no
  such column) (`test_f2_...`).
- A merely predictive column is NOT flagged, and carries no
  `strong_single_column` warning.
- The warning tier (rule 3 of step 8, D30). Warned and kept: a column built by
  a recipe of spec 10.2 to sit clearly between 0.90 and 0.98 has the role
  `feature`, a null `reason` and a null `leak_rule`, reaches the models (a
  recording fake sees it), and is named by exactly one `strong_single_column`
  warning and by no `suspected_leak` warning (`test_f2_...`).
- Both sides of each cut-off, in one table or several: the copy of the target
  (above 0.98) is excluded and carries no `strong_single_column` warning; the
  middle column is warned and kept; the merely predictive column (near 0.6) is
  kept with neither warning.
- The rule is evaluated on training rows only: under time order, a column that
  copies the target on the holdout rows only is not flagged, and one that
  copies it on the training rows only is (`test_f1_...`). The warning tier
  uses the same measure on the same rows: a column that is the middle column
  on the training rows and noise on the holdout rows is warned; one that
  copies the target on the holdout rows and is noise on the training rows is
  neither flagged nor warned (`test_f1_...`).
- Under time order a lagged target on a random walk is not flagged (step 8).
- A table whose only feature is a leak raises `NoFeaturesError`.

### `tests/test_uncertainty.py`: `profile()`
- The interval contains the score, low is below high, the skill interval
  contains the skill; `n_resamples` is 200 under `fast` and 1,000 under `full`;
  the same seed gives the same interval.
- The p-value by hand: with a fake family predicting the holdout perfectly,
  p = 1/(1+R); it is always a multiple of 1/(1+R), above 0 and at most 1; on a
  shuffled target it is not small on most of several seeds.
- Under time order: `bootstrap_blocks`; `block_length` is the smallest L with
  L³ at least the number of distinct holdout timestamps (27 gives 3, 28 gives
  4); the p-value is null with a reason.
- Fewer than 30 distinct holdout timestamps: the interval fields are null, a
  reason is given and `no_interval` is raised; 200 give an interval.
- `interval_approximate` (D27). Present: whenever `interval_method` is
  `bootstrap_blocks` there is exactly one such warning, with a null column.
  Absent: under the random scheme, and under time order when there is no
  interval (10 holdout timestamps), where `no_interval` stands in its place.
- `small_holdout` below 50 rows of the rarer class, absent above.

### `tests/test_model_families.py`: `profile(profile="full")`
- Under `full` the families are linear, forest and boosting, in that order,
  each starting with its index-0 configuration with the `params` of 4.1, for
  both tasks. (The rest of the grids are checked in `test_full_profile.py`.)
- A fake family that raises is recorded as `failed` with its error text and a
  `model_failed` warning, is never chosen, and the run still finishes
  (`test_f6_...`).
- Every family failing raises `AllModelsFailedError`.
- `extensions.use` replaces the families only inside the block and restores
  them after an exception; blocks nest.

### `tests/test_full_profile.py`: the grids
- The models table under `full` is exactly the 18 configurations of 4.1, names,
  `params` and order, for both tasks; the regression forest's index-0
  `max_features` is one third and the classification forest's is `"sqrt"`.
- The headline equals the holdout score recomputed from `predictions`, and is
  the chosen configuration's (`test_f3_...`).
- The same seed gives identical results at `n_jobs` 1 and 2 apart from
  `meta.n_jobs` and `meta.timing`, for a regression table (the forest's fold
  scores carry every bit of its predictions).
- `meta.n_jobs` is the number given, and `os.cpu_count()` for `-1`.

### `tests/test_edges.py`
Every row of section 7 with its named error or its reported outcome; every
invalid argument of 2.1 with `InputError`; the limitations statement is the
nine fixed strings of 5.4, in order, under both schemes and both tasks,
including the two added by D27 and D29, and the caveat is the fixed text;
negative skill is reported as measured, not clipped at zero (a fake family
whose predictions are worse than the baseline's gives a `headline.skill`
below zero equal to the test's own arithmetic); `fold_not_scored` on a
time-ordered table whose early blocks hold one class; roles and the task are
the same under `fast` and `full` on a table of more than 2,000 rows (D20).

### `tests/test_readme.py`
Every command in a fenced block of `README.md` names something that exists: the
interpreter path, options that `python -m mlprofessor profile --help` lists,
files referred to. The README states the interpreter path and that the package
is not installed.

### Fixtures
`tests/conftest.py` (recording fakes, table builders, the command-line helper)
and `tests/fixtures/` (any committed CSV, 40 to 400 rows).

## Not tested here

Whether the limitations statement reads well, whether the console summary is
clear, and whether a skill figure on real data is believable. A human judges
those. The suite is for correctness; the gate's synthetic sets are for the
modelling conclusion. The 10-minute budget of D16 is not tested: the harness
stage measured it once (spec section 9). That linear algebra differs between
thread counts only shows on large tables; the suite proves the one-thread limit
is in force (through the seam) rather than its effect.
