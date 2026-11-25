# Test plan: what the suite must prove (phase 1)

> **If `mlprofessor/` already exists at the repo root, this is not the first
> pass. Go straight to the last section of this file, "Fifth pass (CP6)", and
> read its ground rules before anything else. They override your role prompt
> and the rules of the earlier passes wherever they differ.** The sections
> "Second pass (CP3)", "Third pass (CP4)" and "Fourth pass (CP5)" above it
> describe the passes before: their tests are written, built and final, apart
> from the nine tests the fifth pass's rule 3 names.

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

---

# Second pass (CP3): the first review's findings

Written on 2026-11-17, after the first build and its review
(`docs/reviews/phase1-review-1.md`; D36 to D43). Everything above describes
the first pass and stays as it is. This section is the whole of your work in
the second pass.

## Ground rules of the second pass

1. **The application already exists, and this is a second pass.** `mlprofessor/`
   is built and passes all 308 tests of the first pass. Your role prompt says
   you write the suite before the application exists and that every test must
   fail now. In this pass that is not so: some of your tests must fail today
   and some must pass today, and rule 4 says which.

2. **The first pass is final. Do not edit it in any way.** That means the 17
   existing test files, `tests/conftest.py`, `tests/mlp_helpers.py` and
   everything under `tests/fixtures/`: not a line, not a blank, not a rename,
   not a new fixture file. Your role prompt says shared fixtures go in
   `tests/conftest.py`: not in this pass. New shared helpers go in one new
   file, `tests/cp3_helpers.py`, which may import `mlp_helpers`. The driver
   does not lock `tests/` during your stage, so nothing will stop you; the
   owner compares the folder with the committed one afterwards, and any
   change to an existing file sends the whole stage back.

3. **You write ten new test files and the helper file, and nothing else.** The
   files are listed below, each with the tests it must hold, **under the names
   given**: the lists of rule 4 and the mutation check go by name. Split a
   named test into parameter cases as the text says. If you add a test of your
   own, say so in your final report, with the list of rule 4 it belongs to.

4. **Which tests fail today and which pass today is fixed, by name.** The
   lists are at the end of this section. A test on the red list fails against
   the application as it stands, in every one of its parameter cases, because
   the application does not yet do what D36 to D42 decided. A test on the
   green list passes today in every case: it guards something the application
   already gets right. Run the new files before you finish and compare with
   the lists. **Do not weaken a guard to make it fail, and do not weaken a
   red test to make it pass.** If a test on the green list fails, or one on
   the red list passes, and you believe your test is what the spec says, stop
   and report it with the output: either the application is wrong or this
   plan is, and neither is yours to settle by adjusting the test.

5. **Every guard test must fail on its mutant, and
   `verify/mutation_check.py` is how that will be judged.** The owner runs it
   after your stage. It copies `tests/` into a throwaway project beside a copy
   of the reference implementation (`verify/reference/`, a second correct
   implementation, whose random split differs from the application's), runs
   the whole suite there, and repeats that with the reference broken in one
   way at a time: F1 to F8, and the five small faults M1 to M5 of spec 8.1.
   It passes only if:
   - **the good reference fails none of your tests.** A test that passes
     against the application and fails against the reference has pinned
     something the spec leaves open (spec 11): row numbers a seed produces,
     the wording of a message, the order of warnings;
   - **each of M1 to M5 fails at least one test.** The guard named for a fault
     in spec 8.1 is the test that must fail on it.

   You may run it yourself, from the repo root, with the project interpreter:
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/mutation_check.py --jobs 4`,
   about two minutes. It writes nothing inside the repo. Do not
   read the reference implementation to tune a test to it. The tests of
   `README.md` (files named `tests/test_readme*.py`) are shown on a line of
   their own and not counted, since the README is the same file in every case:
   so the README tests of this pass go in a file of that name.

6. **The rules of the first pass still hold**: the documented surface only
   (spec 2.1 to 2.7); pin what the spec pins and nothing else; recompute, do
   not trust; tiny tables built in code; no skipped or expected-to-fail test;
   nothing written outside `tmp_path`; the application imported inside test
   functions; the command line run one process at a time through
   `mlp_helpers.run_cli`. The new files start about ten processes between
   them; the whole suite should stay under two minutes.

## Files of the second pass

Each file belongs to exactly one task of `plan/tasks.json`.

| File | Task | About | Today |
|---|---|---|---|
| `tests/test_guard_final_fit.py` | T18 | M1, M2: the built-in preprocessing of every fit; what a level is (D42) | green |
| `tests/test_guard_leak_screen.py` | T18 | M3: the leak screen sees training rows only | green |
| `tests/test_guard_uncertainty.py` | T18 | M4, M5: the block bootstrap, ties in the p-value | green |
| `tests/test_csv_edges.py` | T18 | E: a file that is not UTF-8; leading blank lines; header cells of white space (D42) | green |
| `tests/test_skill_undefined.py` | T19 | A: skill when a baseline's error is zero (D36, D40, D42) | red, one test green |
| `tests/test_feature_names.py` | T20 | B: colliding feature names (D37) | red, one test green |
| `tests/test_chosen_failure.py` | T21 | C: a configuration that fails after it is chosen (D38) | one test red, three green |
| `tests/test_run_hygiene.py` | T22 | D: the output folder, process settings, the no-features message | three tests red, two green |
| `tests/test_console_summary.py` | T22 | D: what the console summary shows | red |
| `tests/test_readme_examples.py` | T23 | the README's commands use a table kept in the repo | red |
| `tests/cp3_helpers.py` | none | shared helpers of the second pass; holds no test | |

### `tests/cp3_helpers.py`

Import `mlp_helpers as h` and build on it; do not copy from it. What the new
files need, so build it once here:

- **A fake that keeps the matrix.** The first pass's `Log` records row
  positions and column names. The guards for M1 and M2 need the values too: an
  estimator for `extensions.use` that records, for every `fit` and every
  `predict`, `list(X.index)`, `list(X.columns)` and a copy of
  `X.to_numpy(dtype=float)`, and otherwise behaves like `h.WrapEstimator`.
- **A fake that fails on command.** Like `h.LookupEstimator` (predictions the
  test chose, by source row position), with two switches: raise from `fit`
  when the rows it is given are exactly a set the test names (the training
  rows: only the final fit has all of them); raise from `predict` or
  `predict_proba` when the rows it is given include any of a set the test
  names (the holdout rows). It records the call before it raises.
- **A step that keeps what it is given.** A preprocessing step for
  `extensions.use` that changes nothing and keeps a copy of every frame it is
  fitted on, values and all: the first pass's `RecordingStep` keeps only row
  positions and column names.
- A one-line runner for `profile()` inside `extensions.use`, as each first-pass
  file has; and a check that a result is strict JSON
  (`json.dumps(..., allow_nan=False)` on `h.plain(result.to_dict())`).
- **"Is this number shown?"** for the console summary (spec 2.2): a value
  counts as shown when some decimal number in the text is within
  0.0005 × max(1, |value|) of it.

## The guards (task T18; all green today)

Each guard is built on a table crafted so that the right implementation and
the faulty one give different answers **whatever the implementation's random
stream**. The recipes below were run on the application, on the reference and
on the reference carrying each fault (`docs/measurements/CP3/`); the values
quoted are what the two gave. Keep to the recipe: a guard that is easier to
write and no longer fails on its mutant is worth nothing.

### `tests/test_guard_final_fit.py`

A table of about 240 rows under the random scheme, classification, with a
numeric feature in which about 15% of the cells are missing (so that the
imputed median is in play) and a categorical feature of three levels. One
family through the seam, the matrix-keeping fake. A run then makes six fits:
five folds and the final fit, the one whose rows are exactly the training
rows.

- `test_m1_final_fit_statistics_ignore_the_holdout`. Run once and read the
  holdout rows from `predictions.row`. Build a second table that differs only
  in the holdout rows, whose numeric features are replaced by extremes (1e6,
  −1e6, 5e5). Run again. The holdout is the same rows (the split does not
  look at features), and **every one of the six matrices handed to `fit` is
  equal to its counterpart, value for value** (`numpy.array_equal`), with the
  same rows and columns. Assert too that the training rows hold a missing
  cell, or the median is not tested. Measured: the largest change in the final
  fit's matrix is 0 for the reference and 4.28 with M1, where the mean of the
  `x1` column moves from 0 to −0.58. The first pass's test of this compares
  fold scores only, which is why it cannot see M1.
- `test_m2_levels_come_from_the_rows_of_each_fit`. Find the holdout rows and
  the folds with `stages.split_random(n, classes, seed)` (spec 2.7 pins that
  the run uses it), then give a few holdout rows the level `holdout_only` and
  a few rows of one fold's validation part the level `fold_only`. Run. Then:
  no matrix handed to any `fit` has a column `colour=holdout_only`; a matrix
  has the column `colour=fold_only` exactly when the fit's rows include a row
  holding that level, which is four folds and the final fit, five in all, and
  not the fold whose validation part holds them; no matrix has two columns of
  one name. And by effect, in the matrix handed to `predict` for the holdout:
  the rows holding `holdout_only` are zero in every `colour=` column (spec
  step 8: a level unseen in the fit is all zeros). Measured: with M2 every
  fit has both columns.
- `test_surrounding_white_space_is_not_part_of_a_level` (D42; spec step 5 and
  2.6). The application already does this; the reference did not until D42. A
  table of about 240 rows, random scheme, classification, with three text
  columns in which **every cell is written with its own amount of white space
  round it** (leading spaces, trailing spaces, some with a trailing tab, so
  that no two cells of a column are the same text as written; assert that in
  the test): `colour`, one of three words at random; `echo`, the target
  written `yes` or `no`; `one_level`, the same word in every row. Run with the
  matrix-keeping fake and the frame-keeping step. Then:
  - in counting: `colour` is a categorical feature and `one_level` is
    excluded as `constant`;
  - in the leak screen: `echo` is excluded as a suspected leak with the rule
    `predicts_target`. A screen that took the cells as written would find a
    different level in every row, know none of a fold's validation rows, and
    measure no skill;
  - what a step receives: in every frame the step was fitted on, the values
    of `colour` are among the three bare words, and `echo` is not a column;
  - in the matrix: in every matrix handed to `fit`, the columns that hold an
    equals sign are exactly `colour=blue`, `colour=green`, `colour=red`.

  Use spaces and tabs only. Measured on the reference as it was before D42:
  the test fails there, with `echo` still a feature.

### `tests/test_guard_leak_screen.py`

Both tests run under time order with unique numeric timestamps, 400 rows, so
that the test knows the rows: 300 training rows in six blocks of 50, and the
last 100 as the holdout. Assert that the holdout is rows 300 to 399. A
recording family through the seam keeps the run short.

- `test_m3_missing_by_class_counts_training_rows_only`. Classification. Class
  0 is rare in the training rows and common in the holdout: 10 training rows
  of class 0 (put two in block 1 and at least one in each later block, so
  that every fold is scored; assert that), and 90 of the 100 holdout rows.
  The column `late` is a number, never missing for class 1, missing for every
  holdout row of class 0 and for 7 of the 10 training rows of class 0; the
  three that are filled sit in blocks 2 to 6. So `late` is missing for 70% of
  class 0 among the training rows, far from the 95% of rule 1, and for 97%
  once holdout rows are counted. Assert both shares in the test. Then `late`
  has the role `feature`, a null `reason` and a null `leak_rule`, and no
  `suspected_leak` warning. Measured: its single-column skill is 0.59, so
  rule 2 and rule 3 leave it alone; with M3 it is excluded as
  `missing_by_class`. Under the random scheme this cannot be built away from
  the cut-off, because the split is stratified.
- `test_m3_single_column_skill_is_fitted_on_fold_training_rows_only`.
  Classification, the two classes in equal numbers inside every training
  block. The column `drift` takes two values in training block *b*, *b* + 0.25
  and *b* + 0.75; which of the two goes with class 1 turns round from one
  block to the next (the higher one in odd blocks, the lower one in even
  blocks). The holdout rows take the values of blocks 2 to 6 under the same
  rule, 20 rows a block, 10 of each class. A tree fitted on blocks 1 to *k*
  has never seen the values of block *k* + 1: both lie above everything it
  knows, fall in one leaf and get one prediction, so the pooled ROC AUC is
  0.5 and the skill 0. A tree that has also been fitted on the holdout rows
  knows every value. Assert `drift` has the role `feature`, null `reason` and
  `leak_rule`, and neither a `suspected_leak` nor a `strong_single_column`
  warning. Measured: single-column skill 0.000 for the reference, 1.000 with
  M3, which excludes it as `predicts_target`. The first pass's tests of the
  screen catch one that validates on holdout rows, not one that trains on
  them.

### `tests/test_guard_uncertainty.py`

- `test_m4_block_bootstrap_draws_whole_blocks`. Regression under time order,
  256 unique timestamps, so the holdout is the last 64 and the block length
  is 4. Make the target a multiple of 1/8 (so that sums are exact) and declare
  `task="regression"`. One seam family predicts chosen values: on the holdout
  rows, taken in time order, the target itself on the even positions and the
  target plus 1 on the odd ones, so the squared errors alternate 0, 1, 0, 1.
  Any four consecutive timestamps then hold two of each, 16 blocks make
  exactly 64 timestamps, and **every resample of a moving-block bootstrap has
  the same mean, exactly 0.5**, whichever blocks are drawn. Assert
  `interval_method` is `bootstrap_blocks`, `block_length` 4, `score` 0.5,
  `n_resamples_used` 200, and `interval_low` and `interval_high` both 0.5 to
  within 1e-12. Measured with M4, which draws timestamps one by one: 0.359 to
  0.625. This does not depend on the random stream, which a test built on one
  large error would.
- `test_m5_ties_count_in_the_p_value`. Classification, random scheme, one
  seam family that predicts 0.5 for every row. ROC AUC is 0.5 on the holdout
  and on every shuffle of it, so every shuffle ties with the observed skill:
  `score` 0.5, `skill` 0, `n_permutations` 200 and `p_value` exactly 1.0.
  Measured with M5: 1/201. Use classification only: for regression a tie is a
  tie only up to rounding, and the spec pins no tolerance (an accepted limit,
  D43).

### `tests/test_csv_edges.py`

The application already does both (measured 2026-11-17); the reference did
neither until this change. Through `stages.load_table` unless stated.

- `test_a_file_that_is_not_utf8_is_an_input_error_wherever_the_bad_byte_sits`,
  one case each for a byte `0xFF` in: the header line; the first data row;
  the third data row; a row past 200 kB of well-formed rows; the last cell of
  the file, with no line end after it; and a file of Latin-1 text (`café`).
  Each raises `InputError` from `stages.load_table` and from `profile()`, and
  `profile()` writes nothing.
- `test_leading_lines_of_spaces_and_tabs_are_skipped_as_pandas_skips_them`,
  one case each for the lines written before the header: three spaces; a tab;
  several lines of spaces and tabs with an empty line among them; a
  byte-order mark and then a line of spaces; two empty lines and a line of one
  space. The loaded frame has the columns `a`, `b`, `y`, the number of rows
  and the values `pandas.read_csv` gives for the same file (read it in the
  test), and `info["warnings"]` is empty: no `blank_header`.
- `test_duplicate_names_are_still_found_after_leading_blank_lines`: blank
  lines, then a header `a,b,a`: `DuplicateColumnsError`.
- `test_a_line_pandas_reads_as_the_header_is_not_skipped`, one case each for a
  first line of two quotes and of a form feed, followed by a header of three
  names and rows. pandas reads a header of one column (assert that in the
  test); the tool raises `InputError`, since the lines after it are longer
  than the header (spec 6). **Use only spaces and tabs for "blank":** Python's
  `str.strip()` also removes a form feed and a no-break space, and pandas
  does not.
- `test_profile_runs_on_a_file_with_leading_blank_lines`: a whole run on such
  a file; `columns` names the real header, `rows.loaded` is the number of
  data lines, no `blank_header` warning.
- `test_header_cells_of_only_white_space_are_names_like_any_other` (D42; spec
  6). The application already does this; the reference did not until D42. A
  header `a, , ,y`, whose second and third cells are one space each:
  `DuplicateColumnsError`, from `stages.load_table` and from `profile()`. A
  header `a, ,  ,y`, one space and two spaces: it loads, the column names are
  `a`, one space, two spaces, `y`, as `pandas.read_csv` names them (assert
  that in the test), and `info["warnings"]` is empty: neither cell is blank,
  since blank means empty.

## A: skill when a baseline's error is zero (task T19)

### `tests/test_skill_undefined.py`

Spec: step 11, step 12 (the last three bullets), step 8, 2.5, 2.7, 5.2.

- `test_flat_holdout_under_time_order_reports_null_skill`. A series of 240
  unique numeric timestamps, regression, with the target exactly 0.0 from row
  150 on: the holdout is rows 180 to 239, and the last training row is flat
  too, so the previous value is exact on every holdout row. Default families.
  The run finishes. `baselines` holds `training_mean`, with the score the
  test computes, and `previous_value`, with a score of 0 and `used` true;
  `headline.baseline_score` is 0; `headline.score` is the squared error the
  test computes from `predictions`, and above 0. `skill`, `skill_low` and
  `skill_high` are null. Exactly one `skill_undefined` warning, with a null
  column. The score keeps its interval: `interval_method` is
  `bootstrap_blocks`, the interval holds the score, `n_resamples_used` is 200
  and `n_resamples_skill_used` is 0. `p_value` is null with a reason; no
  `no_interval` warning; `interval_approximate` is there. The whole result is
  strict JSON.
- `test_zero_error_of_the_training_mean_under_the_random_scheme`. 240 rows,
  `task="regression"` declared. Find the split with
  `stages.split_random(240, None, seed)`: give half of the 180 training rows
  the target 1.0 and half −1.0, so that their mean is exactly 0, and every
  holdout row 0.0. A seam family predicts 0.25 everywhere. Then the baseline
  is `training_mean` with a score of 0; `score` is 0.0625; `skill`,
  `skill_low`, `skill_high` are null with one `skill_undefined` warning;
  `interval_method` is `bootstrap_rows` and both ends of the interval are
  0.0625; `n_resamples_used` 200, `n_resamples_skill_used` 0; `p_value` and
  `n_permutations` are null and `p_value_reason` is a non-empty string.
- `test_resample_with_a_zero_baseline_is_left_out_of_the_skill_interval_only`.
  The same table with two holdout rows changed to 2.0 and −2.0, so the
  baseline's error is 8/60 and skill is defined: assert `skill` by hand. A row
  resample misses both rows about one time in eight, and then has a score and
  no skill. Assert `n_resamples_used` is 200, `n_resamples_skill_used` is at
  least 100 and below 200, `skill_low` and `skill_high` are numbers, the
  interval holds the score, there is neither a `skill_undefined` nor a
  `no_skill_interval` warning, and `p_value` is a number. (That no resample
  of 200 misses both has a
  probability under 1e-12 for any stream.)
- `test_no_skill_interval_when_fewer_than_half_the_resamples_have_a_baseline_error`.
  Time order, 256 unique timestamps, regression. The target is about 5 over
  the training rows, exactly 5.0 on the last training row, and 0.0 on all 64
  holdout rows: the previous value is wrong on the first holdout timestamp
  only, so `baseline_score` is 25/64 and the baseline is `previous_value`. A
  seam family predicts the target plus 0.5, so `score` is 0.25 and `skill` is
  1 − 0.25 ÷ (25/64). A block resample holds the first timestamp only when a
  block starts there, about one time in four. Assert `interval_method` is
  `bootstrap_blocks` with both ends 0.25, `n_resamples_used` 200,
  `n_resamples_skill_used` above 0 and below 100, `skill_low` and
  `skill_high` null, exactly one `no_skill_interval` warning with a null
  column, and no `skill_undefined` warning: the headline skill has a value
  (D42).
- `test_skill_undefined_and_no_skill_interval_are_two_codes` (D42; spec 2.5).
  Four runs under time order, and what each carries of the two codes:
  - the flat-holdout series of the first test, where `headline.skill` is
    null: one `skill_undefined`, with a null column, and no
    `no_skill_interval`;
  - the table of the test above, where `skill` is a number, the score has its
    interval and `skill_low` is null: one `no_skill_interval`, with a null
    column, and no `skill_undefined`;
  - 40 timestamps of three rows each, so the holdout has 10 timestamps and
    there is no interval at all: `skill` is a number, `interval_method` is
    null, `no_interval` is there, and neither of the two codes;
  - an ordinary series: neither.
- `test_single_column_skill_that_cannot_be_computed_flags_nothing` (**green
  today**). Time order, 240 unique timestamps, regression. The target varies
  over the first 20 rows, is exactly 0.0 from there to the end of the
  training rows, and varies over the holdout. Over the screen's pooled
  validation rows the previous value is then exact, so no column's skill can
  be computed. A column that is the target times 2 plus 1 keeps the role
  `feature` with null `reason` and `leak_rule`, and has neither a
  `suspected_leak` nor a `strong_single_column` warning; `headline.skill` is a
  number.
- `test_every_resample_counts_for_skill_when_no_baseline_error_is_zero`, one
  case for each task, on an ordinary table under the random scheme:
  `n_resamples_skill_used` equals `n_resamples_used` (200 for regression), the
  skill interval holds the skill, and there is neither a `skill_undefined`
  nor a `no_skill_interval` warning.
- `test_stages_skill_with_a_zero_baseline_is_none`:
  `stages.skill("mse", 1.0, 0.0)` and `stages.skill("mse", 0.0, 0.0)` are
  `None`; `stages.skill("mse", 1.0, 4.0)` is still 0.75.
- `test_command_line_finishes_on_a_flat_holdout`: the first table as a CSV
  through the command line with `--time-col`: exit 0; `profile.json` parses
  with a hook that rejects NaN and infinity; `headline.skill` is null; a line
  of standard output holds `skill_undefined`.

## B: colliding feature names (task T20)

### `tests/test_feature_names.py`

Spec: step 5 ("Feature names must be distinct"), 2.4, 2.7. In every test that
expects the error, run with a recording family and a recording step through
the seam and assert that neither recorded a single call: the run stops before
any fit.

- `test_error_class_and_code`: `errors.FeatureNameCollisionError` subclasses
  `ProfileError` and its `code` is `feature_name_collision`.
- `test_numeric_column_named_like_a_level_stops_the_run`: a categorical
  `colour` with the level `red`, and a numeric column named `colour=red`. The
  error is raised; its message holds `colour=red`; with `out_dir` given,
  nothing is written.
- `test_two_categorical_columns_that_combine_to_one_name_stop_the_run`:
  `site` with the level `a=b`, beside `site=a` with the level `b`. The message
  holds `site=a`.
- `test_a_rare_level_counts`: `colour` has 40 levels and one single row holds
  `red`, so `red` is not among the 30 most frequent of any fit. The error is
  still raised.
- `test_the_missing_level_counts`: `colour` has a missing cell in a usable
  row, and a numeric column is named `colour=(missing)`.
- `test_a_level_only_in_rows_with_no_target_does_not_count` (**green
  today**): the level `purple` sits only in two rows whose target is missing,
  and a numeric column is named `colour=purple`. The run finishes; both
  columns are features; with the matrix-keeping fake, every matrix handed to
  `fit` has `colour=purple` exactly once and no name twice.
- `test_a_level_in_rows_the_fast_profile_leaves_out_still_counts`: time
  order, 2,300 unique timestamps, the fast profile. The level `red` sits only
  in the row with the earliest timestamp, which the row cap leaves out, and a
  numeric column is named `colour=red`. The error is raised: the rule is
  decided before the cap.
- `test_preprocessor_fit_raises_on_colliding_names`:
  `stages.Preprocessor({"colour": "categorical", "colour=red": "numeric"})`
  raises the error from `fit` on rows that hold the level `red`; on rows that
  do not, `fit` and `transform` work and the output has four distinct column
  names.
- `test_command_line_reports_the_collision`: exit 3, a line
  `error: feature_name_collision:` on standard error, nothing written.

A message names every column involved. The longer of two colliding column
names always contains the shorter, so asserting the longer is enough. A level
here is the stripped text (D42); the guard of that is in
`test_guard_final_fit.py`, so write the levels of these tables without white
space round them. That a collision stops the run even for a level outside the
rows used is an accepted limit (D43), and two of the tests above hold it so.

## C: a configuration that fails after it is chosen (task T21)

### `tests/test_chosen_failure.py`

Spec: steps 9, 10 and 11, 2.5, 2.6, 5.2. Random scheme, classification, about
240 rows. Four seam families of one configuration each, built on the failing
fake, whose chosen predictions are the target mixed with noise in clearly
different shares (0.8, 0.5, 0.3, 0.1 of the target), so that their `cv_mean`
ranks them `best`, `second`, `third`, `last` under any split. Find the holdout
rows with a first run. "Fails on the final fit" is the switch on `fit` given
the training rows; "fails on the holdout" is the switch on `predict` given the
holdout rows.

- `test_failure_after_the_choice_keeps_fold_scores_and_falls_back`, one case
  for each way of failing; families `best` and `second`. Run once with
  nothing failing, to have `best`'s fold scores. Then with `best` failing:
  `best-0` has status `failed`, `chosen` false, the fake's message in `error`,
  and **the same `fold_scores`, `cv_mean` and `cv_std` as in the clean run**,
  none of them null; `second-0` is `fitted`, chosen, with a null `error`;
  exactly one `model_failed` warning; `headline.model`, `cv.chosen` and
  `cv.chosen_mean` are `second-0`'s; `headline.score` is the ROC AUC the test
  computes from `second`'s values on the holdout rows, and `y_pred` is those
  values.
- `test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration`
  (**green today**). All four families, given in the order `last`, `third`,
  `best`, `second`, so that table order is not ranking order; `best` fails on
  the holdout and `second` on the final fit. Statuses, in table order:
  fitted, fitted, failed, failed. `third-0` alone is chosen; two
  `model_failed` warnings; `headline.score` and `skill` are the test's own
  arithmetic on `third`'s values. No assertion on fold scores here: that is
  the test above.
- `test_holdout_rows_reach_one_configuration_at_a_time` (**green today**).
  The same four families, sharing one log; `best` and `second` both fail on
  the holdout. The `predict` calls that include a holdout row come, in order,
  from `best`, `second`, `third`, and from no other family: `last` never
  receives one. Each but the last of them has status `failed`. No `fit` ever
  receives a holdout row. Then with nothing failing: exactly one `predict`
  call includes a holdout row, and it is `best`'s.
- `test_every_configuration_failing_after_selection_raises_all_models_failed`
  (**green today**): two families, one failing each way: `AllModelsFailedError`,
  the fake's message in it, nothing written to `out_dir`.

The first pass's `test_f6_failing_family_is_recorded_not_dropped` stays right:
a configuration that fails **during selection** has five null fold scores and
a null `cv_mean`, also when it had scored some folds before it failed (an
accepted limit, D43). Only one that fails after it was chosen keeps them.

## D: small defects (task T22)

### `tests/test_run_hygiene.py`

Spec: 2.1, 2.4.

- `test_unusable_out_dir_is_an_input_error_before_anything_is_loaded`, one
  case for `out_dir` naming an existing file and one for a path below an
  existing file. With a table whose column names are duplicated (which would
  raise `DuplicateColumnsError` as soon as it was loaded): `InputError`. With a
  good table, a recording family and a recording step: `InputError`, no call
  recorded, the file's content unchanged and nothing else created beside it.
- `test_out_dir_that_does_not_exist_yet_is_created` (**green today**): a path
  three folders deep; both files are written there.
- `test_csv_field_size_limit_is_left_as_found`, four cases: `profile()` on a
  CSV; `stages.load_table` on a CSV; and each of the two on a CSV that raises
  `DuplicateColumnsError`. **Set the limit to a value of your own first**
  (`csv.field_size_limit(100_000)`), make the call, assert the limit is still
  100,000, and put the original back in a `finally`. Comparing "before" and
  "after" without setting it proves nothing: an earlier test in the same
  process has usually raised it already.
- `test_thread_limits_are_left_as_found` (**green today**): run once first,
  so that every library is loaded; then read the user API, internal API and
  thread count of every pool from `threadpoolctl.threadpool_info()`, run with
  `n_jobs=2`, and compare; and again after a run that raises a named error.
- `test_no_features_message_names_each_excluded_column_and_its_reason`: a
  table of a constant column, a serial number and an all-distinct text column
  (give them names that are not parts of other words): the message of
  `NoFeaturesError` holds each column's name and its reason code (`constant`,
  `id_like`). And a table whose only feature copies the target, beside a
  serial number: the message holds the copy's name with `suspected_leak` and
  the serial number's with `id_like`. Assert names and reason codes only: the
  wording is open.

### `tests/test_console_summary.py`

Spec: 2.2. Three runs of the command line, fast profile, `--n-jobs 1`; read
`profile.json` from the output folder and compare with standard output, using
the "is this number shown?" helper.

- `test_summary_shows_both_intervals_the_p_value_and_each_warning`: a
  regression table of 160 rows, random scheme, built to raise three warnings:
  a text date column (`date_column_undeclared`), a numeric column with one
  cell `abc` (`values_coerced`), and the small holdout. `interval_low`,
  `interval_high`, `skill_low`, `skill_high` and `p_value` are each shown; and
  every warning has its line.
- `test_summary_under_time_order_says_why_there_is_no_p_value`: 200 unique
  dates and `--time-col`. The four interval values are shown; the text of
  `headline.p_value_reason` is in the output; every warning has its line.
- `test_summary_says_why_there_is_no_interval`: 40 dates of three rows each,
  so the holdout has 10 timestamps. The texts of `headline.interval_reason`
  and `headline.p_value_reason` are in the output; every warning has its line.

"Every warning has its line": for each warning record there is a line of the
output holding its `code` and, when it has one, its `column`, and no line
serves two records. Nothing else about the summary is pinned.

## The README's examples (task T23)

### `tests/test_readme_examples.py`

Spec: section 1. Today the README's commands read
`verify/artifacts/data/...`, files the gate writes and git ignores, so the
first pass's `test_readme.py` passes only on a machine where the gate has
run. The repo now holds `examples/customers.csv` (400 rows; target `churned`;
`signup_date` in ISO form), and task T23 makes the README use it. All three
tests are red today.

- `test_readme_names_nothing_under_verify_or_tests`: nowhere in `README.md`
  is there a path that starts with `verify/` or `tests/` (either slash, any
  case).
- `test_every_table_the_readme_names_is_kept_in_the_repo`: every value of
  `--data` in the README, and every text literal passed as the first argument
  of `profile(`, is a relative path, not under `verify` or `tests`, to a file
  that exists under the repo root. There is at least one.
- `test_every_readme_command_runs`: every command line in a fenced block that
  runs `-m mlprofessor profile` is run as it is written, with three changes:
  `--out` points into `tmp_path`, and `--profile fast` and `--n-jobs 1` are
  forced. Assert first that its `--data` is not under `verify` or `tests`
  (this is what makes the test red today on every machine). Each exits 0 and
  writes both files.

Do not edit `README.md`, and do not add files under `examples/`: both belong
to the builder.

## What must fail today, and what must pass today

Checked on 2026-11-17 with prototypes of every test below, run against the
application as it stands and against the reference implementation
(`docs/measurements/CP3/README.md`).

**Red today: 26 tests.** They fail against the application until the build
stage has done D36 to D42, and pass against the reference.

| File | Tests |
|---|---|
| `test_skill_undefined.py` | `test_flat_holdout_under_time_order_reports_null_skill`, `test_zero_error_of_the_training_mean_under_the_random_scheme`, `test_resample_with_a_zero_baseline_is_left_out_of_the_skill_interval_only`, `test_no_skill_interval_when_fewer_than_half_the_resamples_have_a_baseline_error`, `test_skill_undefined_and_no_skill_interval_are_two_codes`, `test_every_resample_counts_for_skill_when_no_baseline_error_is_zero`, `test_stages_skill_with_a_zero_baseline_is_none`, `test_command_line_finishes_on_a_flat_holdout` |
| `test_feature_names.py` | `test_error_class_and_code`, `test_numeric_column_named_like_a_level_stops_the_run`, `test_two_categorical_columns_that_combine_to_one_name_stop_the_run`, `test_a_rare_level_counts`, `test_the_missing_level_counts`, `test_a_level_in_rows_the_fast_profile_leaves_out_still_counts`, `test_preprocessor_fit_raises_on_colliding_names`, `test_command_line_reports_the_collision` |
| `test_chosen_failure.py` | `test_failure_after_the_choice_keeps_fold_scores_and_falls_back` |
| `test_run_hygiene.py` | `test_unusable_out_dir_is_an_input_error_before_anything_is_loaded`, `test_csv_field_size_limit_is_left_as_found`, `test_no_features_message_names_each_excluded_column_and_its_reason` |
| `test_console_summary.py` | `test_summary_shows_both_intervals_the_p_value_and_each_warning`, `test_summary_under_time_order_says_why_there_is_no_p_value`, `test_summary_says_why_there_is_no_interval` |
| `test_readme_examples.py` | `test_readme_names_nothing_under_verify_or_tests`, `test_every_table_the_readme_names_is_kept_in_the_repo`, `test_every_readme_command_runs` |

The three of `test_readme_examples.py` are red against the reference too,
until T23 is built: they judge `README.md`, not the package.

**Green today: 20 tests.** They pass against the application and against the
reference, and must go on passing through every task of the build.

| File | Tests |
|---|---|
| `test_guard_final_fit.py` | `test_m1_final_fit_statistics_ignore_the_holdout`, `test_m2_levels_come_from_the_rows_of_each_fit`, `test_surrounding_white_space_is_not_part_of_a_level` |
| `test_guard_leak_screen.py` | `test_m3_missing_by_class_counts_training_rows_only`, `test_m3_single_column_skill_is_fitted_on_fold_training_rows_only` |
| `test_guard_uncertainty.py` | `test_m4_block_bootstrap_draws_whole_blocks`, `test_m5_ties_count_in_the_p_value` |
| `test_csv_edges.py` | `test_a_file_that_is_not_utf8_is_an_input_error_wherever_the_bad_byte_sits`, `test_leading_lines_of_spaces_and_tabs_are_skipped_as_pandas_skips_them`, `test_duplicate_names_are_still_found_after_leading_blank_lines`, `test_a_line_pandas_reads_as_the_header_is_not_skipped`, `test_profile_runs_on_a_file_with_leading_blank_lines`, `test_header_cells_of_only_white_space_are_names_like_any_other` |
| `test_skill_undefined.py` | `test_single_column_skill_that_cannot_be_computed_flags_nothing` |
| `test_feature_names.py` | `test_a_level_only_in_rows_with_no_target_does_not_count` |
| `test_chosen_failure.py` | `test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration`, `test_holdout_rows_reach_one_configuration_at_a_time`, `test_every_configuration_failing_after_selection_raises_all_models_failed` |
| `test_run_hygiene.py` | `test_out_dir_that_does_not_exist_yet_is_created`, `test_thread_limits_are_left_as_found` |

**What the mutation check must then print**, run after your stage and before
the build: the good reference fails 0; each of F1 to F8 and M1 to M5 fails at
least one test, and each of M1 to M5 fails the guard spec 8.1 names for it;
on the README line, 3 failing.

## Not tested in the second pass

That the console summary reads well. The wording of any message. Whether the
final fit's preprocessing is fitted again for a second configuration (open:
spec 11). An output folder that passes the check of spec 2.1 and cannot be
written for another reason, such as permissions. Process settings other than
the CSV field limit and the thread limits. The run time of the built library
at 100,000 rows by 100 columns, which was measured once
(`docs/measurements/CP3/`).

---

# Third pass (CP4): the second review's findings

Written on 2026-11-20, after the second build and its review
(`docs/reviews/phase1-review-2.md`; D44 to D48). Everything above describes
the first two passes and stays as it is. This section is the whole of your
work in the third pass.

## Ground rules of the third pass

1. **The application already exists, and this is the third pass.**
   `mlprofessor/` is built and passes all 370 cases of the first two passes.
   Your role prompt says you write the suite before the application exists
   and that every test must fail now. In this pass that is not so: most of
   your tests must fail today and three must pass today, and rule 4 says
   which.

2. **The first two passes are final. Do not edit them in any way.** That means
   the 27 existing test files, `tests/conftest.py`, `tests/mlp_helpers.py`,
   `tests/cp3_helpers.py` and everything under `tests/fixtures/`: not a line,
   not a blank, not a rename, not a new fixture file. New shared helpers go in
   one new file, `tests/cp4_helpers.py`, which may import `mlp_helpers` and
   `cp3_helpers`. The driver does not lock `tests/` during your stage, so
   nothing will stop you; the owner compares the folder with the committed
   one afterwards, and any change to an existing file sends the whole stage
   back.

3. **You write seven new test files and the helper file, and nothing else.**
   The files are listed below, each with the tests it must hold, **under the
   names given**: the lists of rule 4 and the mutation check go by name. Split
   a named test into parameter cases as the text says. If you add a test of
   your own, say so in your final report, with the list of rule 4 it belongs
   to.

4. **Which tests fail today and which pass today is fixed, by name.** The
   lists are at the end of this section. A test on the red list fails against
   the application as it stands, in every one of its parameter cases, because
   the application does not yet do what D44 to D48 decided. A test on the
   green list passes today in every case: it guards something the application
   already gets right. Run the new files before you finish and compare with
   the lists. **Do not weaken a guard to make it fail, and do not weaken a red
   test to make it pass.** If a test on the green list fails, or one on the red
   list passes, and you believe your test is what the spec says, stop and
   report it with the output.

5. **Every guard must fail on its variant, and `verify/mutation_check.py` is
   how that will be judged.** The owner runs it after your stage. It runs the
   whole suite against a copy of the reference implementation
   (`verify/reference/`, a second correct implementation, whose random split
   differs from the application's), and again with the reference broken in
   one way at a time: twelve forms the gate also catches (F1 to F8, F6b, F6c,
   F6d, F7b) and seven smaller faults (M1 to M7; spec 8 and 8.1). It passes
   only if:
   - **the good reference fails none of your tests.** A test that passes
     against the application and fails against the reference has pinned
     something the spec leaves open (spec 11): row numbers a seed produces,
     the wording of a message, the order of warnings, how a number is
     formatted;
   - **every variant fails at least one test.** The three guards of this pass
     are each named for one: the guard of `test_guard_header_lines.py` must
     fail on F6d, `test_a_prediction_that_is_not_finite_counts_as_raising` on
     M6, and `test_fast_error_stands_when_the_full_profile_would_run` on M7.
     M6 is caught by no other test.

   You may run it yourself, from the repo root, with the project interpreter:
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/mutation_check.py --jobs 7`,
   about five minutes. It writes nothing inside the repo. Do not read the
   reference implementation to tune a test to it. The README test of this pass
   goes in a file named `tests/test_readme_fast_profile.py`: files named
   `tests/test_readme*.py` are shown on a line of their own and not counted.

6. **Assert on the values as the application produced them (D44).** The fault
   this pass exists for was a number that was not finite, written as null,
   and no test could see it: `h.plain`, `h.records`, `h.models_of` and
   `c.assert_strict_json` all read NaN as null before anything is asserted.
   In this pass, where a test asserts that a number is a number, or that a
   field is null, it reads `result.headline`, `result.to_dict()` or the parsed
   `profile.json` **directly**, and asserts the type: a finite `float` or
   `int` that is not a `bool`, or `None`. It never passes the result through
   a helper that turns NaN into `None` first. The earlier helpers stay in use
   for what they are good at: row positions, warning codes, column records of
   whole numbers.

7. **The rules of the earlier passes still hold**: the documented surface only
   (spec 2.1 to 2.7); pin what the spec pins and nothing else; recompute, do
   not trust; no skipped or expected-to-fail test; nothing written outside
   `tmp_path`; the application imported inside test functions; the command
   line run one process at a time through `mlp_helpers.run_cli`. Tiny tables,
   with two exceptions this pass needs: the fast profile's cap is 2,000 rows,
   so the tables of `test_fast_profile_error.py` have 3,000 to 30,000 rows
   (one numeric feature, a recording family through the seam: under two
   seconds each); and `test_non_finite.py` uses 400 rows where the text says
   so. The seven files start five processes between them and run in under
   twenty seconds. For "white space" use spaces and tabs only, as in the
   second pass.

## Files of the third pass

Each file belongs to exactly one task of `plan/tasks.json`.

| File | Task | About | Today |
|---|---|---|---|
| `tests/test_guard_header_lines.py` | T24 | G: a line pandas reads as the header, before a two-column header | green |
| `tests/test_non_finite.py` | T25 | A: no number that is not finite (D44); a feature value too large, a complex target (D48) | red, one test green |
| `tests/test_target_white_space.py` | T26 | D: white space round a target value (D46); a target cell of only white space (D48) | red |
| `tests/test_seed_range.py` | T27 | B: the seed's range (D45) | red |
| `tests/test_fast_profile_error.py` | T28 | E: the fast profile's error (D46) | one test red, one green |
| `tests/test_summary_digits.py` | T29 | C: the console summary's significant digits (D46) | red |
| `tests/test_readme_fast_profile.py` | T30 | F: which rows the fast profile keeps (D46) | red |
| `tests/cp4_helpers.py` | none | shared helpers of the third pass; holds no test | |

### `tests/cp4_helpers.py`

Import `mlp_helpers as h` and `cp3_helpers as c` and build on them; do not copy
from them. What the new files need, so build it once here:

- **`assert_raw_numbers(result)`**: every number of a finished run is finite,
  and a null stands only where spec 5.2 says one may. It reads
  `result.to_dict()` and `result.headline` as they are (rule 6). In detail:
  no `float` anywhere in `to_dict()` is NaN or infinite, and
  `json.dumps(result.to_dict(), allow_nan=False)` works without any
  conversion; `headline.score`, `baseline_score` and `interval_level` are
  numbers; `headline.skill` is a number, or it is `None` and then
  `baseline_score == 0` and there is exactly one `skill_undefined` warning;
  `interval_low` and `interval_high` are numbers when `interval_method` is
  not `None`, and otherwise both `None` with a non-empty `interval_reason`;
  `skill_low` and `skill_high` are both numbers or both `None`; `p_value` is
  a number **only if `skill` is a number**, and when it is `None` so is
  `n_permutations` and `p_value_reason` is a non-empty string; every
  `baselines[].score` is a number; every configuration with status `fitted`
  has a number for `cv_mean`, a number for each fold that
  `validation.folds[].scored` marks as scored (and `None` for the others),
  and a number for `cv_std` when two folds or more were scored; every
  configuration with status `failed` has a non-empty `error`;
  `cv.chosen_mean` and `validation.holdout_share` are numbers; every `y_pred`
  of `result.predictions` is finite. "A number" is an `int` or a `float`
  (numpy's included), not a `bool`, and finite.
- **`shown4(value, text)`**: the console summary's rule (spec 2.2). A value
  counts as shown when some number in `text` (found with
  `c.numbers_in`, which reads an exponent) is within 0.0005 × |value| of it.
  For a value of exactly 0 that asks for a printed number equal to 0.
- **Table builders**: `reg_table(n, seed)`, `h.reg_frame` with 20.0 added to
  the target and a first column `t` of `0.0, 1.0, …` (a numeric time column,
  unique, in file order); `panel(n_times, per_time, seed)`, a regression table
  of `per_time` rows on each of `n_times` numeric timestamps `t`, one numeric
  feature `x1` and a target `y = x1 + noise` (**draw `x1` and the noise from
  one generator, one after the other**: two generators from the same seed
  give the same numbers, the target becomes twice the feature, and the run
  stops with `no_features`); and the constants `LARGEST_DOUBLE =
  1.7976931348623157e308` and `LARGEST_FLOAT32 = 3.4028234663852886e38`.

## G: lines pandas reads as the header (task T24; green today)

### `tests/test_guard_header_lines.py`

Spec: section 6, section 9. The second pass tested a line of two quotes and a
line of a form feed before a header of three names. The second review asked
for the three lines below before a header of **two**, where pandas makes a
plain index of the one leading field. The application and the reference both
stop with `input_error` on all three (measured 2026-11-20).

- `test_a_line_pandas_reads_as_the_header_before_a_two_column_header_is_an_input_error`,
  one case each for a first line of: a form feed (`\f`); a no-break space
  (U+00A0, written as UTF-8); two quotes round a space (`" "`). After it, a
  header `a,y` and 160 rows of two fields. Assert first what pandas does with
  the file, so that the test says why it exists: `pandas.read_csv` returns
  one column and 161 rows. Then `stages.load_table` raises `InputError`, and
  `profile()` raises `InputError` and writes nothing into `out_dir`.

  This guard fails on F6d, the reference with its check for long rows taken
  out: there the file loads with one column, and `profile()` ends in
  `column_not_found`. To pandas these files *are* files whose rows are longer
  than the header.

## A: no number that is not finite (task T25)

### `tests/test_non_finite.py`

Spec: section 3 ("Numbers that are not finite", steps 4, 5, 8 to 13), 2.4,
2.5, 2.7, 5.1, 5.2. Regression unless stated; fast profile; a recording family
through the seam (`h.family("rec", log)`) wherever the default models are not
the point. `k` below is `cp4_helpers`.

- `test_error_class_and_code`: `errors.NonFiniteScoreError` subclasses
  `ProfileError` and its `code` is `non_finite_score`.
- `test_extreme_target_stops_with_one_named_error_before_any_fit`, six cases:
  three targets by two schemes. On `k.reg_table(400)`: the cell at row 5 set
  to `k.LARGEST_DOUBLE`; the cell at row 395 set to it; the whole target
  multiplied by 1e160. Each under the random scheme and under time order
  (`time_col="t"`). With a recording family and a recording step: the run
  raises `NonFiniteScoreError`; the message holds `training_mean` or
  `previous_value` (the baseline is named as `baselines[].name` spells it);
  **the log holds no event at all** (the baselines are scored before any
  fit); nothing is written into `out_dir`. These are the two inputs of the
  review's first finding. Wherever a split puts the cell the error is the
  same: the square of a value that size, or of the training mean it makes, is
  beyond the largest double.
- `test_extreme_target_through_the_command_line`: the row-395 table as a CSV:
  exit 3; standard error holds **exactly one line that is not blank**, and it
  starts `error: non_finite_score:`; nothing is written. (A numpy overflow
  warning on standard error fails this.)
- `test_a_target_cell_that_spoils_only_the_folds_stops_before_any_fit`:
  `k.reg_table(400)` under time order, with the cell at row 120 set to 1e155.
  The holdout is rows 300 to 399 and row 120 is in the third training block.
  Assert in the test that both holdout baselines can be scored with room to
  spare (the squared errors of the training mean add up to less than 1e308;
  the previous value's error is finite) and that the square of row 120
  against the mean of rows 0 to 99 is not finite. Then the run raises
  `NonFiniteScoreError`, its message holds `training_mean` or
  `previous_value` (which of the screen's two baselines is named is open),
  and the log holds no event. Use this cell and this row: a cell in the first
  block overflows only where squared errors are *added up*, which a careful
  implementation avoids and the spec leaves open.
- `test_a_fold_score_that_is_not_finite_fails_the_configuration`:
  `h.reg_frame(240)`, random scheme, two `h.LookupEstimator` families:
  `far`, which predicts 1e200 for every row, and `sound`, which predicts the
  target plus a little noise. Read the models from `result.to_dict()`.
  `far-0`: status `failed`, a non-empty `error`, `fold_scores` five `None`,
  `cv_mean` and `cv_std` `None`, `chosen` false. `sound-0`: `fitted` and
  chosen; `headline.model` is `sound-0`; exactly one `model_failed` warning;
  `k.assert_raw_numbers(result)`.
- `test_every_fold_score_not_finite_is_all_models_failed`: the `far` family
  alone: `AllModelsFailedError`, nothing written. Here the cause is the
  model, not the target, and the list of configurations is the right answer.
- `test_a_holdout_score_that_is_not_finite_stops_the_run_without_a_fallback`:
  `h.reg_frame(240)`. Find the holdout rows with a first run. Two
  `h.LookupEstimator` families sharing one log, given in the order `sound`,
  `wild`: `wild` predicts the target exactly on every row outside the holdout
  (so its `cv_mean` is 0, the best) and 1e200 on every holdout row; `sound`
  predicts the target plus noise. The run raises `NonFiniteScoreError`; the
  message holds `wild-0`; nothing is written; and the `predict` calls that
  include a holdout row come from `wild` alone: `sound` is never tried.
- `test_a_prediction_that_is_not_finite_counts_as_raising` (**green today**;
  fails on M6). Classification, `h.clf_frame(240)`, random scheme. Find the
  holdout rows with a first run. Three `h.LookupEstimator` families, in the
  order `folds_nan`, `sound`, `holdout_nan`: `holdout_nan` predicts
  0.8 × the target plus noise, so that it ranks first, with NaN on three
  holdout rows; `folds_nan` predicts NaN on three training rows; `sound`
  predicts 0.3 × the target plus noise. Run `holdout_nan`'s values without
  the NaN once, alone, to have its fold scores. Then: `folds_nan-0` is
  `failed` with five `None` fold scores and a `None` `cv_mean`;
  `holdout_nan-0` is `failed`, not chosen, with a non-empty `error` and **the
  same `fold_scores` and `cv_mean` as in the clean run** (D38);
  `sound-0` is chosen; two `model_failed` warnings; every `y_pred` is finite
  and equals `sound`'s value for its row; `k.assert_raw_numbers(result)`. Use
  classification: ROC AUC is a count of ranks and stays finite whatever is
  predicted, so only this rule keeps a NaN out of `y_pred`.
- `test_feature_values_too_large_for_the_tree_models_are_read_as_missing`
  (D48). `h.clf_frame(300)` with two more numeric columns: `big`, normal
  values with the cells of rows 10, 77 and 150 set to 1e39, −1e200 and
  `k.LARGEST_DOUBLE`, and row 201 set to infinity; `edge`, normal values with
  rows 11 and 78 set to 1e38 and −1e38. Run with `c.MatrixEstimator` and
  `c.keep_step_factory` on one `c.KeepLog`. The run finishes. `big` is a
  numeric feature with `n_infinite` 4 and `n_imputed` 4; `edge` is a feature
  with `n_infinite` 0; the `infinite_values` warnings name `big` and nothing
  else; in every frame a step was fitted on, `big` is NaN at those four rows
  (where the frame holds them) and `edge` holds 1e38 and −1e38 as they are;
  every matrix handed to a model's `fit` is finite; `k.assert_raw_numbers`.
  Stay away from the cut-off, 3.4028234663852886e+38: 1e38 and 1e39 are on
  either side of it.
- `test_column_roles_count_too_large_values_with_the_infinite_ones` (D48): the
  same table through `stages.column_roles`: `big` has `n_infinite` 4 and the
  role `feature`; `edge` has `n_infinite` 0.
- `test_preprocessor_reads_infinite_and_too_large_cells_as_missing` (D48):
  `stages.Preprocessor({"a": "numeric"})` fitted on and applied to a frame
  whose column `a` is 1, 2, 3, 4, 1e200, infinity, −1e39. The three last
  cells are missing cells: they take the median of the four, 2.5. The result
  equals, to 1e-12, the seven values 1, 2, 3, 4, 2.5, 2.5, 2.5 less their
  mean, divided by their deviation (`numpy.std`), and every value is finite.
- `test_a_complex_target_is_not_numeric` (D48):
  `stages.decide_task(pd.Series(np.arange(40) + 2j), task="regression")`
  raises `TaskUnsupportedError`; the same with `task="auto"` raises
  `TaskAmbiguousError`; a series of two distinct complex values is
  classification with `n_distinct` 2. And through `profile()`, on
  `h.reg_frame(160)` with its target made complex: `task="regression"` raises
  `TaskUnsupportedError`, `task="auto"` raises `TaskAmbiguousError`.
- `test_a_target_on_a_very_large_scale_gives_finite_numbers_or_the_named_error`:
  `h.reg_frame(240)` with the target multiplied by 1e80, default families.
  Squared errors are near 1e160, which a double holds, and their own squares
  are not. The spec leaves open whether such a run finishes (spec 11): the
  test accepts `NonFiniteScoreError`, **or** a result that passes
  `k.assert_raw_numbers`. Any other error fails it, `AllModelsFailedError`
  among them, and so does a result with a null where a number belongs. Write
  it without an early `return`: catch the named error, and call the helper
  only when there is a result.

## D: white space round a target value (task T26)

### `tests/test_target_white_space.py`

Spec: steps 2 and 4, 2.7, 5.1, 5.2. Classification, `no` and `yes`, random
scheme, a recording family. All red today.

- `test_a_target_written_with_stray_white_space_has_two_classes`:
  `h.clf_frame(240, labels=("no", "yes"))`, with every fourth target cell
  written with white space round it (leading spaces, a trailing space, some
  with a trailing tab), so that as written the column has more than two
  values: assert that. With `task="auto"` the run finishes: `task.type` is
  `classification`, `decided_by` `rule`, `n_distinct` 2, `classes`
  `["no", "yes"]`, `positive_class` `"yes"`. Every `y_true` equals the clean
  value of its row. The holdout is `stages.split_random(n, clean values,
  seed)["holdout"]` (spec 2.7). And the `y` every `fit` of the recording
  family received is 1 exactly where the clean value is `yes`.
- `test_every_cell_of_one_class_written_with_a_trailing_space`: every `yes`
  written `yes `, through a CSV file and `out_dir`: `classes` is
  `["no", "yes"]` and `positive_class` `"yes"`, in the result and in
  `profile.json`; the `y_true` cells of `predictions.csv`, read as text, are
  `no` and `yes` and nothing else.
- `test_a_target_cell_of_only_white_space_is_a_missing_target` (D48), one case
  for a DataFrame and one for the same table as a CSV. Five target cells hold
  only white space (two spaces, a tab, a space-tab-space, three spaces, one
  space) and two are empty (`None`). Assert that the five load as text
  (`stages.load_table`). Then `stages.usable_rows` reports
  `dropped_missing_target` 7 and none of the seven among `usable`; the run
  finishes with `rows.dropped_missing_target` 7, `classes`
  `["no", "yes"]`, and none of the seven rows in the predictions.
- `test_decide_task_reads_text_without_the_white_space_round_it`:
  `stages.decide_task` on a series of `yes`, `yes `, `no`, ` no`, two spaces,
  a tab and `yes`, and `None`, repeated: classification by the rule,
  `n_distinct` 2, `classes` `["no", "yes"]`, `positive_class` `"yes"`.
- `test_split_random_compares_text_labels_without_white_space`: 200 labels,
  about 30% `yes`; a second array with white space round every label, so that
  it has more than two distinct values. `stages.split_random(200, labels,
  seed)` gives the same holdout, the same training rows and the same five
  validation parts for both arrays, and the holdout's share of `yes` is
  within one row of a quarter of them.

A target cell of only white space is a missing target; a *feature* cell of
only white space is still the level of empty text (D42). Do not test the
feature side again: `test_guard_final_fit.py` holds it.

## B: the seed's range (task T27)

### `tests/test_seed_range.py`

Spec: 2.1, 2.2, 2.4, 2.7. All red today.

- `test_a_seed_above_the_range_is_an_input_error_before_anything_is_loaded`,
  one case each for 2³², 2⁶³ and a timestamp-style 20261119123456. With a
  table whose column names are duplicated (which would raise
  `DuplicateColumnsError` as soon as it was loaded): `InputError`. With a
  good table, a recording family and a recording step: `InputError`, no call
  recorded, nothing written into `out_dir`.
- `test_the_largest_seed_runs_and_the_next_one_is_refused`: with the seed
  4,294,967,295 the run finishes, `meta.seed` is that number, and a second
  run gives the same `to_dict()` apart from `meta.timing`; with 4,294,967,296
  it raises `InputError`. One test, both sides of the edge.
- `test_command_line_refuses_a_seed_out_of_range`: `--seed 4294967296`: exit
  3; standard error holds exactly one line that is not blank, starting
  `error: input_error:`; the word `Traceback` is nowhere in it; nothing is
  written.
- `test_split_random_refuses_a_seed_out_of_range`, one case for 2³² and one
  for −1: `stages.split_random(120, None, seed)` raises `InputError`.

## E: the fast profile's error (task T28)

### `tests/test_fast_profile_error.py`

Spec: steps 6 and 7, 2.4. Tables from `k.panel`, time order on `t`, a
recording family, `n_jobs=1`. Compare errors by their `code` and by
`type(...) is`, never with `pytest.raises` of a parent class alone.

- `test_fast_raises_the_full_profiles_error_when_full_would_stop_too`, three
  cases: 5 timestamps of 600 rows (`too_few_rows`); 3 timestamps of 2,500
  rows, where the cap can keep nothing (`too_few_rows`); 9 timestamps of 450
  rows with `gap=2` (`gap_too_large`). In each: the call with
  `profile="full"` raises that error; the same call with `profile="fast"`
  raises an error of **the same class and code**, not
  `fast_profile_too_few_timestamps`. Red today in every case: the fast call
  says "use the full profile", and the full profile stops too.
- `test_fast_error_stands_when_the_full_profile_would_run` (**green today**;
  fails on M7), two cases: 8 timestamps of 450 rows (too few timestamps after
  the cap) and 12 timestamps of 2,500 rows (the latest timestamp alone is
  over the cap). The call with `profile="fast"` raises `FastProfileError`,
  code `fast_profile_too_few_timestamps`; the same call with
  `profile="full"` returns a result under the time-ordered scheme with
  `rows.dropped_row_cap` 0.

Not tested, and left as it was: a fast run whose kept rows do hold six
training timestamps and whose gap leaves fewer raises `gap_too_large`
whatever the full profile would do (spec step 7).

## C: the console summary's significant digits (task T29)

### `tests/test_summary_digits.py`

Spec: 2.2. One test, three runs of the command line, fast profile,
`--n-jobs 1`; read `profile.json` and compare with standard output, using
`k.shown4`. Red today.

- `test_summary_numbers_agree_with_profile_json_to_four_significant_digits`.
  For each run, every one of `score`, `baseline_score`, `skill`,
  `interval_low`, `interval_high`, `skill_low`, `skill_high` and `p_value`
  that is not null is shown.
  - `h.reg_frame(240)` with the target multiplied by 1e-4: assert the score
    is below 1e-6, so the table is doing its job. This is the review's case:
    today the summary prints `mse 0.0000 … baseline 0.0000`.
  - the same with the target multiplied by 1e6: the score is above 1e9.
  - a flat holdout under time order (`h.time_frame(240)` with the target 0.0
    from row 150 on and a numeric time column): `baseline_score` is exactly 0
    and is shown as a zero; `skill` is null and needs no number.

  Keep it one test: the second and third runs alone would pass today on some
  of their numbers, and rule 4 wants a red test red in every case. (Today the
  p-value fails at every scale too: 0.004975 is printed `0.0050`.)

The second pass's `test_console_summary.py` uses the older, weaker rule and
stays as it is: a summary that meets the new rule meets the old one.

## F: the README (task T30)

### `tests/test_readme_fast_profile.py`

Spec: step 6. The README says of the fast profile's 2,000 rows that "the rows
kept are chosen from the seed". That is the random scheme only: under time
order they are the most recent whole timestamps. Red today.

- `test_readme_says_which_rows_the_fast_profile_keeps_under_each_scheme`.
  Take the blocks of `README.md` (paragraphs, and each list item as a block
  of its own) that mention the cap: `2,000` or `2000`. Assert there is one.
  In the text of those blocks, joined, look for two ideas, not for a
  sentence: the random rule, a word of `seed`, `random`, `randomly`; and the
  time-ordered rule, a word of `most recent`, `latest`, `newest`, `last`
  together with a word of `timestamp(s)`, `date(s)`, `day(s)`. Both must be
  there. Today the first is and the second is not. Match whole words, case
  ignored. Pin no wording beyond these.

Do not edit `README.md`: it belongs to the builder.

## What must fail today, and what must pass today

Checked on 2026-11-20 with prototypes of every test below, run against the
application as it stands, against the reference implementation, and against
the reference carrying F6d, M6 and M7 (`docs/measurements/CP4/README.md`).

**Red today: 24 tests, 35 cases.** They fail against the application until
the build stage has done D44 to D48, and pass against the reference.

| File | Tests |
|---|---|
| `test_non_finite.py` | `test_error_class_and_code`, `test_extreme_target_stops_with_one_named_error_before_any_fit`, `test_extreme_target_through_the_command_line`, `test_a_target_cell_that_spoils_only_the_folds_stops_before_any_fit`, `test_a_fold_score_that_is_not_finite_fails_the_configuration`, `test_every_fold_score_not_finite_is_all_models_failed`, `test_a_holdout_score_that_is_not_finite_stops_the_run_without_a_fallback`, `test_feature_values_too_large_for_the_tree_models_are_read_as_missing`, `test_column_roles_count_too_large_values_with_the_infinite_ones`, `test_preprocessor_reads_infinite_and_too_large_cells_as_missing`, `test_a_complex_target_is_not_numeric`, `test_a_target_on_a_very_large_scale_gives_finite_numbers_or_the_named_error` |
| `test_target_white_space.py` | `test_a_target_written_with_stray_white_space_has_two_classes`, `test_every_cell_of_one_class_written_with_a_trailing_space`, `test_a_target_cell_of_only_white_space_is_a_missing_target`, `test_decide_task_reads_text_without_the_white_space_round_it`, `test_split_random_compares_text_labels_without_white_space` |
| `test_seed_range.py` | `test_a_seed_above_the_range_is_an_input_error_before_anything_is_loaded`, `test_the_largest_seed_runs_and_the_next_one_is_refused`, `test_command_line_refuses_a_seed_out_of_range`, `test_split_random_refuses_a_seed_out_of_range` |
| `test_fast_profile_error.py` | `test_fast_raises_the_full_profiles_error_when_full_would_stop_too` |
| `test_summary_digits.py` | `test_summary_numbers_agree_with_profile_json_to_four_significant_digits` |
| `test_readme_fast_profile.py` | `test_readme_says_which_rows_the_fast_profile_keeps_under_each_scheme` |

The test of `test_readme_fast_profile.py` is red against the reference too,
until T30 is built: it judges `README.md`, not the package. Several tests of
`test_non_finite.py` fail today only because `errors.NonFiniteScoreError`
does not exist yet; once it does they fail for their real reason, a run that
finishes where it must stop.

**Green today: 3 tests, 6 cases.** They pass against the application and
against the reference, and must go on passing through every task of the
build.

| File | Test | Fails on |
|---|---|---|
| `test_guard_header_lines.py` | `test_a_line_pandas_reads_as_the_header_before_a_two_column_header_is_an_input_error` | F6d, in all three cases |
| `test_non_finite.py` | `test_a_prediction_that_is_not_finite_counts_as_raising` | M6 |
| `test_fast_profile_error.py` | `test_fast_error_stands_when_the_full_profile_would_run` | M7, in both cases |

**What the mutation check must then print**, run after your stage and before
the build: the good reference fails 0 of 405 (the 365 counted today and 40 of
the 41 new cases); each of F1 to F8, F6b, F6c, F6d, F7b and M1 to M7 fails at
least one test, the three guards above among them; on the README line, 1
failing of 6. Before your stage it prints `M6 … NOT CAUGHT` and ends in
`FAIL`: that is the hole this pass's guard closes.

## Not tested in the third pass

That the summary reads well, and how it formats a number beyond four
significant digits. The wording of any message. A number that is not finite
arising in the panel's autocorrelation, in a column's own error in the leak
screen, or in the skill of a run whose baseline error is tiny: each stops the
run by the spec, and each needs a table at the very edge of the double range
whose outcome depends on how an implementation adds up (spec 11). That the
previous value exists for every validation and holdout row: it follows from
the split's own rule (spec step 8) and no input reaches the other case. The
fast profile's `gap_too_large` on rows that do hold six training timestamps.
A missing value among the `classes` handed to `stages.split_random`. The
gate's self-test and its run folder, which are the gate's own business.

---

# Fourth pass (CP5): the third review's findings

Written on 2026-11-20, after the third build and its review
(`docs/reviews/phase1-review-3.md`; D49 to D54). Everything above describes
the first three passes and stays as it is. This section is the whole of your
work in the fourth pass.

## Ground rules of the fourth pass

1. **The application already exists, and this is the fourth pass.**
   `mlprofessor/` is built and passes all 411 cases of the first three passes.
   Your role prompt says you write the suite before the application exists
   and that every test must fail now. In this pass that is not so: most of
   your tests must fail today, six guards must pass today, and most cases of
   the type matrix pass today. Rule 5 says which.

2. **The first three passes are final. Do not edit them, with ONE exception,
   which rule 3 spells out.** That means the 34 existing test files,
   `tests/conftest.py`, `tests/mlp_helpers.py`, `tests/cp3_helpers.py`,
   `tests/cp4_helpers.py` and everything under `tests/fixtures/`: not a line,
   not a blank, not a rename, not a new fixture file. New shared helpers go in
   one new file, `tests/cp5_helpers.py`, which may import the three earlier
   helper files. The driver does not lock `tests/` during your stage, so
   nothing will stop you; the owner reads the diff of the folder against the
   committed one afterwards, and any change to an existing file other than the
   one of rule 3 sends the whole stage back.

3. **THE ONE EXCEPTION (D54). You change exactly one existing test, in exactly
   this way, and nothing else in any existing file.** The test is
   `tests/test_skeleton.py::test_profile_signature_and_defaults`. It pins the
   signature of `mlprofessor.profile` to exactly nine parameters, and D49 adds
   a tenth, `duplicates`. Loosen it so that it accepts both shapes:
   - the first nine parameters stay pinned **exactly as now**: the same names,
     in the same order, each with the same default (`data`, `target`,
     `task="auto"`, `time_col=None`, `gap=0`, `profile="full"`, `seed=0`,
     `out_dir=None`, `n_jobs=-1`);
   - after them there is either nothing, or exactly one more parameter named
     `duplicates` whose default is `"error"`. Any other extra parameter, a
     second extra one, or another default fails the test.

   In code: compare the list of the nine names with the first nine names of
   the signature, check the nine defaults as the test does now, then assert
   that the names after the ninth are `[]` or `["duplicates"]`, and in the
   second case that its default is `"error"`. Change only the body of that
   one test function. Do not touch the file's imports, its other four tests,
   the list `ERRORS`, or a single character outside that function; do not add
   `--duplicates` to the list of options in
   `test_help_exits_zero_and_lists_every_option` (your new file checks it).
   **The loosened test must pass against the application as it stands today**
   (nine parameters) and against the reference implementation (ten): run it
   and see. It is not on the red list. Why this is done at all: the driver
   re-runs every earlier test before it builds, and a test that fails against
   the built library reopens every finished task. Why it is safe: your new
   `tests/test_duplicates.py` requires the tenth parameter and its default,
   so the signature is pinned exactly again as soon as it is built.

4. **You write eight new test files and the helper file, and nothing else.**
   The files are listed below, each with the tests it must hold, **under the
   names given**: the lists of rule 5 and the mutation check go by name. Split
   a named test into parameter cases as the text says. If you add a test of
   your own, say so in your final report, with the list of rule 5 it belongs
   to.

5. **Which tests fail today and which pass today is fixed, by name.** The
   lists are at the end of this section. A test on the red list fails against
   the application as it stands, in every one of its parameter cases. A test
   on the green list passes today in every case. For the type matrix the red
   list names the cases, by their ids, and every other case is green. Run the
   new files before you finish and compare with the lists. **Do not weaken a
   guard to make it fail, and do not weaken a red test to make it pass.** If
   a test on the green list fails, or one on the red list passes, and you
   believe your test is what the spec says, stop and report it with the
   output. Where this section tells you to keep several cases in ONE test
   function, it is because some of them pass today and the function as a
   whole must be red: do not split them into parameter cases.

6. **Every guard must fail on its variant, and `verify/mutation_check.py` is
   how that will be judged.** The owner runs it after your stage. It runs the
   whole suite against a copy of the reference implementation
   (`verify/reference/`, a second correct implementation, whose random split
   differs from the application's), and again with the reference broken in
   one way at a time: fourteen forms the gate also catches (F1 to F9, F6b,
   F6c, F6d, F7b, F7c) and thirteen smaller faults (M1 to M13; spec 8 and
   8.1). It passes only if:
   - **the good reference fails none of your tests** (the README tests apart:
     they judge `README.md`). A test that passes against the application and
     fails against the reference has pinned something the spec leaves open
     (spec 11): row numbers a seed produces, the wording of a message beyond
     what spec 11 pins, the order of warnings, how a number is formatted;
   - **every variant fails at least one test.** Eight are new, and each has
     tests of this pass named for it: M8 to M13 the six guards of
     `test_guard_review3.py`, one each; F9 the tests of `test_duplicates.py`;
     F7c the tests of `test_whole_numbers.py`. M8, M9, M10 and M11 are caught
     by nothing else.

   You may run it yourself, from the repo root, with the project interpreter:
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/mutation_check.py --jobs 8`,
   about ten minutes. It writes nothing inside the repo. Do not read the
   reference implementation to tune a test to it. The README test of this pass
   goes in a file named `tests/test_readme_duplicates.py`: files named
   `tests/test_readme*.py` are shown on a line of their own and not counted.

7. **How the seed becomes random streams is open (spec 11), and two guards
   must live with that.** Two correct implementations draw different
   resamples and keep different rows from the same seed. So no test of this
   pass asserts a particular resample, a particular kept row or a particular
   holdout row that a seed produces, and no test compares an interval end
   with a number worked out for one stream. The guards for M8 and M10 are
   written to hold for any correct implementation; their recipes below say
   how. Where a test needs to know a run's holdout and training rows, it
   finds them out from the run itself, through the seam.

8. **The rules of the earlier passes still hold**: the documented surface only
   (spec 2.1 to 2.7); pin what the spec pins and nothing else; recompute, do
   not trust; assert on values as the application produced them (rule 6 of
   the third pass); **no skipped or expected-to-fail test** (where a
   combination does not exist, leave it out of the parameter list: a skipped
   test counts as a failure in the build); nothing written outside
   `tmp_path`; the application imported inside test functions; the command
   line run one process at a time through `mlp_helpers.run_cli`. Tiny tables,
   with the exceptions this pass needs: `test_fast_cap_error.py` has tables of
   3,260 to 40,000 rows of two numeric features (a constant family through
   the seam where a run must finish: a few seconds in all);
   `test_duplicates.py` has two tables of 4,000 rows; and the type matrix has
   about 800 tables of 200 rows, 40 seconds in all. The eight files start
   about a dozen processes between them and run in under two minutes
   (measured: 59 seconds against the application).

## Files of the fourth pass

Each file belongs to exactly one task of `plan/tasks.json`.

| File | Task | About | Today |
|---|---|---|---|
| `tests/test_guard_review3.py` | T31 | F: six pinned behaviours the third review could break unseen (D52) | green |
| `tests/test_duplicates.py` | T32 | A: copies of a holdout row among the training rows (D49, D53, D54) | red |
| `tests/test_whole_numbers.py` | T33 | B: whole numbers compared exactly, in typed columns; whole numbers beyond 64 bits (D50) | red |
| `tests/test_object_numbers.py` | T34 | C: number columns held as objects; a warning for every unsupported column (D51) | red |
| `tests/test_type_matrix.py` | T35 | H: every kind of column, role and extreme ends in a result or a named error (D52) | 37 cases red, the rest green |
| `tests/test_time_fraction.py` | T36 | D: one to nine digits in a fraction of a second (D52) | red |
| `tests/test_fast_cap_error.py` | T37 | E: the fast profile's cap leaving too few rows (D52, D53) | red |
| `tests/test_readme_duplicates.py` | T38 | the README on copies of holdout rows (D49) | red |
| `tests/cp5_helpers.py` | none | shared helpers of the fourth pass; holds no test | |
| `tests/test_skeleton.py` | (T1) | one test loosened as rule 3 says; nothing else | green before and after |

### `tests/cp5_helpers.py`

Import `mlp_helpers as h`, `cp3_helpers as c` and, where useful,
`cp4_helpers`, and build on them; do not copy from them. What the new files
need, so build it once here:

- **`flat_family(log, n_rows, value=0.5)`**: a list holding one family for the
  seam, an `h.LookupEstimator` that predicts `value` for every row. It is
  quick and never fails, so a test about rows, counts or errors does not pay
  for real models.
- **`run_kept(frame, target="y", seed=0, **kw)`**: `c.run` with that family
  and with a `c.KeepStep` through a `c.KeepLog`, so that every frame a step
  is handed is kept. Returns `(result, log)`.
- **`training_rows(log)`**: the rows of the largest `fit` a step was handed:
  the final fit's rows, which are the training rows. The holdout rows are
  `h.holdout_rows(result)`.
- **`raised(frame, target="y", **kw)`**: calls `profile()` (fast profile, one
  core, unless `kw` says otherwise) and returns the `ProfileError` it raises;
  fails if it returns a result. Any other exception passes through, which is
  what a test of this pass wants.
- **`copies_by_hand(result, log)`**: the test's own count for step 8a. From
  the largest step `fit` (the training frame and the `y` it was handed) and
  from the step `transform` whose rows are the holdout rows, build one Python
  tuple per row: every feature cell as handed, with a NaN replaced by `None`
  and `0.0` added to every float (so `-0.0` becomes `0.0`), then the target.
  The holdout rows' target is not handed to a step: take it from
  `result.predictions["y_true"]`, as 1 or 0 by comparison with
  `task.positive_class` for classification and as the number for regression.
  The count is the holdout tuples that are in the set of training tuples.
- **`plant_copies(frame, k, target="y", seed=0, **kw)`**: a copy of `frame`
  in which exactly `k` holdout rows are copies of training rows; returns it
  with the number of holdout rows. It works because the random split depends
  only on the number of rows, the seed and, for classification, the classes
  (spec step 7, 2.7): run once with `run_kept`, read the training and holdout
  rows, then give each of the first `k` holdout rows every feature value of a
  training row of its own (a different training row each time; for
  classification one of the same class, so that the classes do not change;
  for regression copy the target too). The run on the table returned has the
  same split, and with continuous features no copy arises by chance.
- **`twice(frame, seed=0)`**: every row present twice, in an order shuffled
  from `seed`; **`noise_frame(n, seed, n_columns=4, task="classification")`**:
  normal columns `n1`.. and a target `y` that depends on none of them (0/1
  drawn at random, or normal).
- **`nanosecond_walk(n=600, start=1_700_000_000_000_000_000, step=100,
  seed=0)`**: a random walk `level` (the running sum of normal steps), its
  previous value `level_prev` (0.0 for the first row), a noise column `x1`,
  and a time column `t` of `int64` values `start + step * k`; the rows in an
  order shuffled from the seed. As floats these 600 stamps are 235 distinct
  values.

## F: the guards (task T31; all green today)

### `tests/test_guard_review3.py`

Spec 8.1 (M8 to M13). The third review planted each of these six faults in a
copy of the application and all 411 tests still passed. Six tests, one per
fault. Each passes against the application and the reference today and fails
on the variant it is named for. If one fails against the application, the
application is wrong.

- `test_m8_the_interval_is_the_middle_95_percent` (fails on M8). Spec step 12.
  The draws of a bootstrap are open (rule 7), so one run cannot tell the
  2.5th and 97.5th percentiles from the 5th and 95th: with 200 resamples the
  end of one interval moves by about 0.19 of the bootstrap's standard
  deviation from seed to seed, and the two rules differ by 0.31. So average.
  A table of 400 rows, one feature `x1` and a regression target `y`, both
  standard normal from `default_rng(81)`; an `h.LookupEstimator` family whose
  values are `y` plus standard normal noise from the same generator;
  `task="regression"`, fast profile. For each `seed` from 0 to 59: run; check
  `interval_method == "bootstrap_rows"`, `interval_level == 0.95` and
  `n_resamples_used == 200`; take the 100 squared errors of the holdout from
  `result.predictions` (their mean is `headline.score`); draw 4,000 resamples
  of them with replacement from a generator of the test's own
  (`default_rng(82)`, one generator across the 60 runs) and take the 2.5th
  and 97.5th percentiles and the standard deviation of the 4,000 means;
  record `(interval_low − own 2.5th) ÷ sd` and `(interval_high − own
  97.5th) ÷ sd`. Assert that the mean of each over the 60 seeds is below 0.18
  in absolute value. Measured: the reference lands at +0.03 and −0.04 (the
  mean of 60 runs moves by about 0.02 to 0.03 with the stream), and M8 at
  +0.26 and −0.35. Do not tighten the 0.18 and do not use fewer seeds: both
  sides of it are more than four standard errors away. About five seconds.
- `test_m9_the_leak_screen_uses_the_previous_value` (fails on M9). Spec step
  8. The series must be one that the fault flags, and a plain random walk is
  not (its lag scores about 0.92 against the training mean alone: measured).
  Use a slow wave: 600 rows, `t = 0.0, 1.0, …`, `y = 30·sin(2π·t ÷ 100) +
  0.1·noise` with noise from `default_rng(91)`; features `lag` (`y` one row
  earlier; `y[0]` for the first row) and `x1` (noise from the same
  generator). `time_col="t"`, `task="regression"`, the profile's own
  families. Against the training mean alone a tree on `lag` scores 0.993,
  above the leak cut-off; against the previous value it does worse than the
  baseline (about −0.85). Assert: `lag` has role `feature` with a null
  `reason` and `leak_rule`; no `suspected_leak` and no `strong_single_column`
  warning names it; `baselines` lists `training_mean` then `previous_value`;
  `headline.baseline` is `previous_value`.
- `test_m10_the_fast_cap_is_a_sample_from_the_seed` (fails on M10). Spec step
  6. `h.clf_frame(n=3000)`, fast profile, a recording family and a recording
  step, for `seed` 0 and 1. Which rows are kept is open (rule 7); that they
  are a sample is not. For each seed: `rows.dropped_row_cap == 1000`; the
  kept rows (every row a `fit` was handed, with the holdout rows) number
  2,000; between 550 and 780 of them lie at position 2,000 or beyond (a
  sample of 2,000 from 3,000 holds about 667 from the last 1,000; the head of
  the file holds none); at least one holdout row is at position 2,000 or
  beyond. And the two seeds keep different sets of rows.
- `test_m11_cv_std_is_the_sample_deviation` (fails on M11). Spec step 9, 5.2.
  `h.clf_frame(n=300)` and `h.reg_frame(n=300)`, fast profile, the profile's
  own two families. For every configuration: five fold scores, not all
  equal; `cv_std` equals `statistics.stdev(fold_scores)` within 1e-9
  (relative), and that value differs from `statistics.pstdev` by more than
  one part in a million, so the test can tell them apart. `cv.chosen_std` is
  the chosen configuration's `cv_std`.
- `test_m12_tied_predictions_in_a_resample` (fails on M12). Spec steps 11 and
  12. 400 rows, a 0/1 target drawn at random; an `h.LookupEstimator` family
  predicting 0.5 for every row. On any rows that hold both classes, a
  constant prediction scores exactly 0.5 when ties share their rank. So:
  `score == 0.5`, `skill == 0.0`, `n_resamples_used == 200`, and
  `interval_low == interval_high == 0.5`, `skill_low == skill_high == 0.0`,
  exactly. (The second pass's `test_m5_ties_count_in_the_p_value` uses the
  same prediction and asserts the score and the p-value; nobody asserted the
  interval.) A second run in the same test, with predictions of 0.75 where
  `x1 > 0` and 0.25 elsewhere: `headline.score` equals `h.auc_by_hand` of the
  predictions within 1e-12, and the interval contains the score and reaches
  less than 0.2 to either side of it.
- `test_m13_n_imputed_counts_rows_that_reach_a_model` (fails on M13). Spec
  step 8, 5.2. `h.clf_frame(n=260)` with a text column `colour` (two levels)
  added; the target made float and set missing in rows 0 to 39; `x1` set
  missing in rows 20 to 69 (50 cells, 20 of them in rows with no target);
  `colour` set missing in rows 30 to 44 (15 cells, 10 of them in such rows).
  Fast profile, the profile's own families. Assert:
  `rows.dropped_missing_target == 40` and training plus holdout is 220;
  `x1` has `n_missing` 50 and `n_imputed` 30; `colour` is categorical with
  `n_missing` 15 and `n_imputed` 5; `z1` has 0 and 0.

## A: copies of a holdout row among the training rows (task T32)

### `tests/test_duplicates.py`

Spec: 2.1, 2.2, 2.4, 2.5, step 8a, 5.2, 11. Red today, every test: the
application has no `duplicates` argument, makes no count and never stops for
copies. Use `cp5_helpers` throughout. The error's code is
`holdout_copies_in_training`.

- `test_duplicates_is_the_tenth_parameter_and_defaults_to_error`. The
  signature of `mlprofessor.profile` is exactly the ten names of spec 2.1, in
  order; `duplicates` defaults to `"error"` and is an ordinary
  positional-or-keyword parameter. (This is the test that pins the signature
  exactly again: D54.)
- `test_error_class_and_code`. `errors.HoldoutCopiesError` subclasses
  `ProfileError` and its `code` is `holdout_copies_in_training`.
- `test_any_other_value_is_an_input_error`, one case each for `"drop"`,
  `"KEEP"`, `""`, `True`, `None`, `1`: `InputError`, on an ordinary table, and
  also on a table with duplicate column names (so it is raised before
  anything is loaded).
- `test_every_row_twice_stops_the_run`, one case for classification and one
  for regression. `twice(noise_frame(n=300, task=...))`: 600 rows. With an
  `out_dir`: `HoldoutCopiesError` and nothing written. Then the same table
  with `duplicates="keep"` through `run_kept`: 150 holdout rows; the count by
  hand is between 90 and 135 (about three quarters); `validation.
  n_holdout_copies` equals it and `holdout_copy_share` equals it ÷ 150. Then
  the error's message (spec 11): among the numbers `c.numbers_in` finds in it
  are the count and the number of holdout rows, and one within 0.1 of 100 ×
  the share; it holds the word `keep`; it does not hold `time_col`, since no
  date column was excluded.
- `test_keep_runs_and_says_so`. `twice(noise_frame(300))` with
  `duplicates="keep"`: `input.duplicates == "keep"`; the scheme is `random`;
  the count equals the count by hand and is at least 10; the warnings hold
  exactly one `("holdout_copies_kept", None)`; `duplicate_rows` is there once
  as before; the result is strict JSON.
- `test_the_rule_needs_a_share_and_a_count`, five cases, each `(rows of
  h.clf_frame, copies planted, stops?)`: `(400, 9, no)`, nine of 100, 9% and
  fewer than 10 rows; `(400, 10, yes)`; `(4000, 10, no)`, ten of 1,000,
  exactly 1% and so not more than 1%; `(4000, 11, yes)`; `(4000, 0, no)`. Full
  profile (with the constant family it is quick, and the fast profile's cap
  would cut the 4,000 rows). Build the table with `plant_copies`; assert the
  holdout is a quarter of the rows. Run with `duplicates="keep"`: the count
  by hand is exactly the number planted, `n_holdout_copies` equals it, the
  share is it ÷ the holdout rows, and there is one `holdout_copies_kept`
  warning when the rule is met and none when it is not. When it is met, the
  same call without `duplicates` raises `HoldoutCopiesError`. When it is not,
  the same call without `duplicates` returns: `input.duplicates == "error"`,
  the same count, no `holdout_copies_kept`; and its `to_dict()` equals that
  of the `keep` run once `meta.timing` and `input.duplicates` are taken out
  of both, and the two `predictions` frames are equal (`"keep"` changes
  nothing else).
- `test_the_rule_in_regression`. `plant_copies(h.reg_frame(400), 12,
  task="regression")`: stops by default; with `keep`, the count is 12 by hand
  and by the tool. Then add 1.0 to the target of six of the planted holdout
  rows: a row that agrees in every feature and differs in its target is not a
  copy, and the count is 6.
- `test_values_are_compared_as_a_step_receives_them`. Spec step 8a, "How
  values are compared". 400 rows: `x1`, `x2` normal, `y` 0/1 at random. Run
  once (full profile, `run_kept`) for the split. Then build the table under
  test: `x1` **written as text** (`f"{v:.6f}"`), so that it is numeric by
  rule 5; a text column `colour` (`red`/`blue` by row parity); a text column
  `record` of distinct ids, which is excluded; the target as `yes`/`no`. For
  twelve holdout rows, each paired with a training row of the same class,
  copy `x1`, `x2` and `colour` from the training row. Then make the pairs
  differ in ways that do not count: one pair `"0.0"` and `"-0.0"` in `x1`;
  one pair an empty cell and the text `abc` (a coerced cell: both missing);
  one pair `"1.5"` and `" 1.50 "`; one holdout row's `colour` with white
  space round it; one holdout row's target written `yes ` or `no `. And in
  one way that does count: one holdout row's `colour` in another case
  (`Red`). Run with `keep`: the holdout rows are those of the first run (the
  split has not moved); `x1` is numeric, `colour` categorical, `record`
  excluded; the count by hand is 11, and so is `n_holdout_copies`.
- `test_the_rule_is_judged_before_any_model_is_fitted`. Spec 2.4, step 8a.
  `twice(noise_frame(300))`, with a family of `h.FailingEstimator` and a
  recording step through the seam: by default `HoldoutCopiesError`, and the
  log holds no event at all (no step and no model was fitted). With `keep`,
  the same family ends in `AllModelsFailedError`.
- `test_time_order_has_no_rule`. 400 rows: `t = 0.0, 1.0, …`; `x1` four
  values repeated down the column; `x2` 0.0 and 1.0 alternating; `y` 0/1 at
  random (do not derive it from `x1`: the leak screen would exclude the
  column). With `time_col="t"`, for `duplicates` `"error"` and `"keep"`
  alike: the run returns; `n_holdout_copies` and `holdout_copy_share` are
  `None`; `input.duplicates` is the value passed; no `holdout_copies_kept`.
  The same table without `t`, random scheme: `holdout_copies_in_training`.
- `test_the_message_names_an_undeclared_date_column`. D53. 400 rows: `when`,
  400 consecutive dates as ISO text; `x1`, 50 normal values each repeated on
  eight consecutive rows; `y` 0/1 at random. With no time column the date is
  excluded and rows differ in nothing else: `holdout_copies_in_training`,
  and the message holds `when` and `time_col`. With `time_col="when"` the run
  returns and the count is `None`. With `duplicates="keep"` and no time
  column it returns with `("date_column_undeclared", "when")` and one
  `holdout_copies_kept`.
- `test_command_line_takes_duplicates`. `twice(noise_frame(300))` written to
  a CSV. Without the option: exit 3, exactly one line on standard error
  starting `error: holdout_copies_in_training: `, nothing in the output
  folder. With `--duplicates keep`: exit 0; `profile.json` has
  `input.duplicates == "keep"` and a count of at least 10; standard output
  holds `holdout_copies_kept`. With `--duplicates error`: exit 3 again. With
  `--duplicates drop`: exit 2. `profile --help` lists `--duplicates`.

## B: whole numbers are compared exactly (task T33)

### `tests/test_whole_numbers.py`

Spec: steps 1a (its last points), 3, 4, 5; 2.4; 2.7; section 6. Red today,
every test. **Typed columns only** in this file, and object columns only
where they hold whole numbers too large for 64 bits: an object column of
whole numbers that fit is the next file's. `START` is
1,700,000,000,000,000,000; the stamps are `START + 100·k` for `k` from 0 to
599, which as floats are fewer than 600 values (assert it, so the case is
one that floats merge).

- `test_read_time_keeps_integers_above_2_53`, one case per column type:
  `int64`; `uint64` (the stamps plus 2⁶³); nullable `Int64`; nullable
  `UInt64`. `stages.read_time` returns the form `numeric` and keys of type
  `int64` (`uint64` for the two unsigned types), each equal to its value,
  600 distinct. `stages.split_time` on those keys **in reverse order**: the
  holdout is positions 0 to 149, the training rows 150 to 599, and each of
  the five folds validates on 75 rows. `stages.previous_value(arange(600.0),
  keys)` is NaN for the first key and `0.0 … 598.0` after it.
- `test_the_boundary_is_a_magnitude_above_2_53`. D53. Float keys
  (`float64`) for: `int64` `[5, 3, 9]`; `[2**53, 0, -2**53]`; nullable
  `Int64` `[7, None, 2**53]` (the second key NaN); `uint8` `[1, 2]`. Integer
  keys (`int64`), each equal to its value, for `[2**53 + 1, 0, 5]` and for
  `[-2**53 - 1, 0, 5]`. Float keys for a float column `[2.0**60, 1.0, 3.0]`.
  Keep these in one test: the first half passes today.
- `test_nanosecond_stamps_are_distinct_timestamps`, one case per column type
  as above. `nanosecond_walk(600)` with `t` in that type; target `level`,
  `time_col="t"`, fast profile, the profile's own families. Assert: scheme
  `time_ordered`, form `numeric`; 450 training and 150 holdout timestamps;
  `timestamps_unique` true; `train_end`, `holdout_start` and `holdout_end`
  are Python `int`s equal to the 450th, 451st and 600th stamp in order;
  `baselines` is `training_mean` then `previous_value`, and the headline
  baseline is `previous_value`; no `suspected_leak`; `level_prev` is a
  feature. Then the twin: the same table with `t` as datetimes
  (`pd.to_datetime` of the stamps) has the same holdout rows, the same
  `headline.score` and `baseline_score`, and the same `baselines`.
- `test_missing_and_repeated_integer_stamps`. `nanosecond_walk(600)` with `t`
  as `Int64` and three cells missing: `dropped_missing_time == 3`, 597
  timestamps in all, unique. Then `t = START + k // 2` over the 600 rows
  (pairs one nanosecond apart): 300 timestamps in all, `timestamps_unique`
  false, and `baselines` is `training_mean` alone.
- `test_distinct_values_are_counted_exactly`. Spec step 5, rules 8 to 8b. 200
  rows, a 0/1 target at random, **one ordinary normal column `x1`**, and the
  columns below, judged by `stages.column_roles` and again by `profile()`,
  which must agree. (`x1` is there for a reason. The columns below reach the
  models as floats, where many of their values are one, so without it many
  rows are alike to a model and the run stops for copies: measured, 12 of 50
  holdout rows.) `wide`,
  `uint64`, 50 distinct values `2**63 + 7·k` repeated at random (a feature;
  as floats, one value); `pair`, `int64`, `2**53` and `2**53 + 1` (a
  feature; as floats, one value: assert both, so the test says why it
  exists); `same`, one value `2**60 + 1` (`constant`); `serial`, `2**53 + 1
  + k` down the file (`id_like`: distinct and rising); `shuffled_serial`,
  the same values in a random order (`id_like`: distinct and consecutive);
  `rising`, a running sum of steps of 1 to 4 plus `2**62` (`id_like`);
  `scattered`, 200 of the values `2**61 + 3·k`, `k` below 1,000, in random
  order (a feature: distinct, neither ordered nor consecutive); `ends`, the
  smallest and the largest `int64` alternating (a feature: two values, and
  nothing overflows); `nullable`, `Int64` with a missing cell wherever the
  draw is 0 and `2**53 + k` elsewhere (a feature). The run's result is
  strict JSON.
- `test_whole_numbers_beyond_64_bits_in_every_role`. Finding 1. Four shapes
  of 200 whole numbers held in an object column: 200 consecutive 26-digit
  numbers; `0 … 198` and then `2**64`; `(−1)^k · (2**63 + k)`; `−2**63 − 1`
  and then `0 … 198`. Seven cases: all four from a DataFrame, and the first,
  second and fourth from a CSV file (pandas loads the third as text, since a
  value above the signed range and within the unsigned one stands beside a
  negative one: spec section 6. Do not list it as a CSV case, and do not skip
  it). Beside two normal features: as a spare
  column with a 0/1 target, the run returns, the column is excluded with the
  reason `unsupported_type`, exactly one `("unsupported_type", column)`
  warning names it, and the result is strict JSON; as the target, under
  `task` `auto`, `classification` and `regression` alike,
  `TaskUnsupportedError`; as the time column, `TimeColumnError`. Assert the
  class with `isinstance` after catching `ProfileError`, so that a raw
  exception fails the test as itself.
- `test_command_line_on_a_26_digit_column`. The review's case: a CSV with a
  column `account` of 26-digit numbers. As a spare column: exit 0, and a line
  of standard output holds `unsupported_type` and `account`. As `--target`:
  exit 3 and one line `error: task_unsupported: …`. As `--time-col`: exit 3
  and one line `error: time_column_invalid: …`.

## C: number columns held as objects (task T34)

### `tests/test_object_numbers.py`

Spec: step 1a, 2.5, 2.6, 2.7, 5.1, 5.2, section 6. Red today, every test.
`D` is `decimal.Decimal`. **Build a typed twin from the same text** as the
object column (`D(text)` beside `float(text)`), never by rounding a float:
`numpy.round` and the decimal's own conversion can differ in the last bit,
and the twins must agree exactly.

- `test_object_number_columns_give_the_result_of_their_typed_twins`. Finding
  4. 300 rows. `amount`: numbers written with two decimals; `visits`: whole
  numbers 0 to 39; `member`: booleans; a noise column; a 0/1 target that
  depends on the three. The driver's table holds `amount` as `Decimal`
  objects, `visits` as Python integers and `member` as Python booleans, each
  in an `object` column with two or three cells `None`; the typed table holds
  the same values as `float64`, `Int64` and `boolean`. Fast profile, the
  profile's own families, both tables. In the driver's run each of the three
  is a feature of kind `numeric` with a null reason, `dtype` `object`,
  `n_missing` its count of `None`, and the `n_imputed` of its twin; there is
  no `unsupported_type` warning. The two runs have the same holdout rows, the
  same `headline` (the whole mapping), the same models table and the same
  `y_pred`.
- `test_a_step_is_handed_floats`. Spec 2.6. The same two tables through a
  keeping step: in the final fit's frame each of the four feature columns is
  `float64`, and the driver's and the twin's are equal cell for cell (NaN
  where a cell was empty); `member` holds 0.0 and 1.0.
- `test_which_mixtures_are_read_as_numbers`, one case per row of this table.
  The values are repeated down a column `c` of 240 rows, held as `object`,
  inserted first beside `h.clf_frame(240)`. `stages.column_roles` and
  `profile()` must agree; `dtype` is `object` in both.

  | Case | Values | Read as |
  |---|---|---|
  | `decimals` | `D("1.5")`, `D("2.25")`, `D("-3")` | numeric feature |
  | `whole_decimals` | `D("1")`, `D("2E+1")`, `D("30")` | numeric feature |
  | `ints_and_none` | `1`, `2`, `None`, `4` | numeric feature |
  | `numpy_ints` | `np.int64(1)`, `np.int32(2)`, `3` | numeric feature |
  | `bools_and_none` | `True`, `False`, `None` | numeric feature |
  | `numpy_bools` | `np.bool_(True)`, `False` | numeric feature |
  | `floats_and_none` | `1.5`, `None`, `2.5`, NaN | numeric feature |
  | `ints_floats_decimals` | `1`, `2.5`, `D("3.75")`, `np.float32(4.5)` | numeric feature |
  | `a_decimal_infinity` | `D("1.5")`, `D("Infinity")`, `D("2.5")`, `D("-Infinity")` | numeric feature; `n_infinite` 120, `n_missing` 0, an `infinite_values` warning |
  | `a_decimal_not_a_number` | `D("1.5")`, `D("NaN")`, `D("2.5")`, `D("sNaN")` | numeric feature; `n_missing` 120, `n_infinite` 0 |
  | `a_bool_beside_numbers` | `True`, `2`, `3.5` | excluded, `unsupported_type` |
  | `a_bool_beside_decimals` | `False`, `D("2.5")` | excluded, `unsupported_type` |
  | `text_beside_numbers` | `1`, `"two"`, `3.0` | excluded, `unsupported_type` |
  | `a_complex_number` | `1.5`, `2 + 3j` | excluded, `unsupported_type` |
  | `lists` | `[1, 2]`, `[3]` | excluded, `unsupported_type` |
  | `bytes` | `b"a"`, `b"b"` | excluded, `unsupported_type` |

  A numeric feature has no `unsupported_type` warning; an excluded column has
  exactly one, naming it. Every run is strict JSON.
- `test_every_unsupported_type_exclusion_is_warned_about`. D51.
  `h.clf_frame(240)` with five more columns: durations (`timedelta64`),
  complex numbers, monthly periods, tuples held as objects, and whole numbers
  and text mixed. The columns excluded with the reason `unsupported_type` are
  exactly those five, and the columns named by `unsupported_type` warnings
  are exactly those five, once each, each with a message that is not empty.
- `test_an_object_target_is_the_typed_target`. **One test, four kinds in a
  loop**: an object target of booleans and one of whole numbers already work
  today, one of decimal numbers does not, and all four must hold together.
  For `False`/`True` (classes `[False, True]`, positive `True`); `0`/`1`
  (`[0, 1]`, `1`); `D("1.50")`/`D("2.50")` (`[1.5, 2.5]`, `2.5`);
  `D("0")`/`D("1")` (`[0, 1]`, `1`): the labels of `h.clf_frame(240)` mapped
  onto the pair, held as `object`, with two cells `None`.
  `stages.decide_task`: classification, by the rule, two distinct values,
  those classes and that positive class. `stages.usable_rows`: two rows
  dropped for a missing target. `profile()`: `task.classes` and
  `positive_class` as above, **with those Python types** (`bool`, `int`,
  `float`); `task.target_dtype` is `object`; two rows dropped; the values of
  `y_true` are the classes.
- `test_a_decimal_target_is_a_regression_target`. The target of
  `h.reg_frame(240)` written with three decimals, as `Decimal` objects and as
  floats. The object run is a regression decided by the rule, with
  `target_dtype` `object`; its `headline` and its `y_true` equal the typed
  run's.
- `test_object_whole_numbers_are_exact`, one case for Python integers and one
  for `Decimal`s with no fractional part, each held as `object`. D50 and D51
  together. `stages.read_time` of the 600 stamps of `nanosecond_walk` gives
  `int64` keys, each equal to its value; of `[5, 3, 2**53]` it gives
  `float64` keys. `profile()` on `nanosecond_walk(600)` with `t` held so:
  450 and 150 timestamps, unique; its `validation` and its `headline` equal
  those of the `int64` table; the headline baseline is `previous_value`. And
  in `stages.column_roles`: 50 distinct values `2**62 + 7·k` repeated at
  random are a feature (as floats they are one value: assert it); `2**53 + 1
  + k` down the file is `id_like`; one value throughout is `constant`.
- `test_an_object_time_column_is_the_typed_time_column`.
  `h.time_frame(240)` with its date column replaced by `D(0) … D(239)` held
  as `object`, against the same with `int64` `0 … 239`: form `numeric`, the
  holdout is rows 180 to 239 in both, the headlines are equal, `train_end`
  is 179.

## H: the type matrix (task T35)

### `tests/test_type_matrix.py`

Spec: 2.4 ("No input ends in an error that is not named"), section 7 ("The
type matrix"). One test function, `test_a_result_or_a_named_error`,
parametrised by kind, role and extreme, with the id `kind-role-extreme` so
that a failure names its case. About 800 cases, 40 seconds.
`docs/measurements/CP5/type_matrix.py` is the script this matrix was first
run with; you may read it for the builders of each kind, and the case names
below are its names. It is a measurement script, not a test: the contract is
this section and spec section 7.

**The table of a case.** 200 rows. `x1`, `z1` and a noise series are standard
normal from `default_rng(0)`. The column under test is: as a `feature`, a
column `c` beside `x1` and `z1`, with the target `y = (x1 + 0.5·noise > 0)`
as 0/1; as the `target`, the column `y` beside `x1` and `z1`; as the `time`
column, a first column `t` with `time_col="t"`, beside `x1` and `z1`, and the
target `y = 2·x1 + noise`. Where the kind allows, the column's ordinary values
depend on `x1` as a feature or a target and are distinct and rising as a time
column. The kind `csv` writes the table to a file under `tmp_path` and passes
the path.

**Kinds** (54): `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`,
`uint32`, `uint64`; `float32`, `float64`; nullable `Int8`, `Int64`, `UInt64`,
`Float32`, `Float64`; `bool`, nullable `boolean`; text as an `object` column
(`text_object`), as the default string type (`text_str`) and as the
Python-backed string type (`text_string_python`); `category`,
`category_ordered`; `bytes`; ISO 8601 dates as text (`iso_text`);
`datetime_ns`, `datetime_s`, `datetime_utc`, `datetime_london`; `timedelta`;
`period`; `interval`; `complex128`, `complex64`; `decimal` (`Decimal`
objects); `object_int`, `object_bool`, `object_float`, `object_numbers`
(integers, floats and `Decimal`s together), `object_bool_beside_numbers`;
`mixed` (whole numbers and text); `list`, `tuple`, `dict`, `set`;
`all_missing_float`, `all_missing_none`, `all_missing_NA`, `all_missing_NaT`,
`all_missing_Int64`, `all_missing_str`; `sparse_float`, `sparse_int`,
`sparse_bool`; `csv`.

**Extremes** (11), each applied where the kind allows it and left out of the
parameter list where it does not: `ordinary`; `two` (exactly two distinct
values; the target role only); `one_value` (the column's first ordinary value
in every row, in the same type); `max_min` (the largest and the smallest
value of the type in four cells: for floats also the smallest normal and the
smallest subnormal; for nanosecond datetimes `Timestamp.max` and
`Timestamp.min`; for second datetimes and ISO text the years 9999 and 0001;
for `object_int` the two ends of `int64` and the largest `uint64` together);
`above_2p53` (whole numbers from 2⁵³ + 1: one each as a time column, 50
values repeated as a feature or target); `e300` (±1e300 and ±1e-300 in four
cells; ±1e38 for 32-bit floats; for `mixed`, 1e300, `D("1E+400")`, `10**30`
and a complex number); `inf` (both infinities); `nan` (the empty values the
kind has: NaN, `None`, `pd.NA`, `NaT`, and for `decimal` both `D("NaN")` and
`D("sNaN")`); `neg_zero` (`-0.0` and `0.0`); `beyond_64` (whole numbers from
10²⁵ throughout); `beyond_64_few` (ordinary whole numbers with `10**25`,
`-10**25`, `2**64` and `10**30` in four cells). The text kinds and `csv`
carry the number extremes as numbers written out as text.

**The assertion.** Call `profile(data, profile="fast", seed=0, n_jobs=1,
out_dir=<a folder under tmp_path>, **the role's arguments)`, inside
`warnings.catch_warnings()` with the filter `ignore` (pandas and numpy warn
while such columns are built and read; a `with` block puts the filters back).

- If it raises a `ProfileError`: its class has a `code`, its message is not
  empty, and the output folder holds no file. The case passes.
- If it raises anything else, let it propagate: that is the failure this file
  exists for. Do not catch `Exception`.
- If it returns: the names in `columns` are the table's column names, in
  order; `rows.loaded` is 200 and equals the other six counts added up;
  `predictions` has one row per holdout row; `json.dumps(result.to_dict(),
  allow_nan=False)` works; the written `profile.json` parses with a
  `parse_constant` that raises, and its `columns` and `task` equal those of
  `to_dict()`; `cp4_helpers.assert_raw_numbers(result)` holds (every number
  finite, every null with its reason: third pass); and `rows`, `columns`,
  `models`, `warnings` and `predictions` are DataFrames.

The matrix does not assert which end a case has. The files of T32 to T34 do
that for the rules they cover.

**Red today: 37 cases**, from four causes (measured 2026-11-20). They are
listed under "What must fail today" below. Every other case passes today and
must go on passing. The green cases are not guards of a named fault, and no
variant is named for them: they are the net for the class of fault, as D52
asked. What shows the net can catch something: the application fails 37
cases; the reference as it stood before this pass failed 18
(`docs/measurements/CP5/README.md`); and the variant F6b, which drops a
constant column from the accounting, fails the `one_value` cases of the
feature role.

## D: fractions of a second (task T36)

### `tests/test_time_fraction.py`

Spec: step 3, 11. Red today, every test.

- `test_one_to_nine_digits_are_read_to_the_nanosecond`, one case for each
  count of digits **1, 2, 4, 5, 6, 7, 8, 9** (three digits were accepted
  before and the first pass holds them: leave 3 out, or that case is green)
  with each of three endings: none, `Z`, `+02:00`. 24 cases. The fraction is
  the first *n* digits of `123456789`. `stages.read_time` of
  `["2024-01-05T09:15:00.<fraction><ending>", "2024-01-05 09:15:01.<fraction><ending>"]`
  (one with `T`, one with a space): form `iso8601`, keys `datetime64[ns]`;
  the first key is 09:15:00 plus the fraction as nanoseconds (the digits
  padded with zeros to nine), two hours earlier under `+02:00`; the second is
  one second later.
- `test_digit_counts_may_be_mixed_in_one_column`. One column holding a value
  with no fraction, with `.5`, `.25`, `.125000`, `.062500000`, and a missing
  cell: form `iso8601`; the five keys are 0, 500,000,000, 250,000,000,
  125,000,000 and 62,500,000 nanoseconds after the second; the sixth is
  `NaT`.
- `test_ten_digits_are_refused_and_the_message_names_the_fraction`, two
  cases: `2024-01-05 09:15:00.1234567891` and
  `2024-01-05T09:15:00.123456789012Z`. `stages.read_time` of a column holding
  a good value and that one raises `TimeColumnError`; the message holds the
  refused value, holds `fraction` (case ignored), and does not hold `not ISO
  8601`. The same through `profile()`: a time column of 200 values with one
  such value at row 17. And in the same test, so that it stays red as a
  whole: a day-first column (`25/01/2024`) raises `TimeColumnError` with a
  message that does not hold `fraction`.
- `test_a_csv_pandas_wrote_from_microsecond_timestamps_runs`. Finding 10.
  `h.time_frame(240)` with its dates replaced by timestamps 1,500
  microseconds apart from `2024-03-01 08:00:00.000001`, written with
  `to_csv`. Assert the first cell of the file is
  `2024-03-01 08:00:00.000001`, so the test says why it exists. `profile()`
  on the path with `time_col="date"`: form `iso8601`, unique timestamps, 180
  and 60, the holdout rows 180 to 239, `holdout_start` equal as a timestamp
  to the 181st date; the same holdout rows and (within 1e-9, relative) the
  same score as the DataFrame itself. The command line on that file exits 0.

## E: the fast profile's cap leaving too few rows (task T37)

### `tests/test_fast_cap_error.py`

Spec: 2.4, step 6, 11. Red today, every test. The new code is
`fast_profile_too_few_rows`. Two table builders, local to the file: a time
table with one numeric timestamp `t` per entry of a list of row counts, in
order, two normal features and a regression target; and a table of `n` rows,
two normal features, and a 0/1 target with exactly `k` rows of class 1 at
random positions. Compare errors by `code` and by `type(...) is`.

- `test_error_class_and_code`. `errors.FastProfileRowsError` subclasses
  `ProfileError`, with the code above; `errors.FastProfileError.code` is
  still `fast_profile_too_few_timestamps`.
- `test_a_small_latest_timestamp_behind_a_large_one`. Finding 9. 30
  timestamps of 100 rows, then one of 2,500, then one of 60. Under `fast`
  with `time_col="t"`: `FastProfileRowsError`; its message holds `full`, and
  `2000` or `2,000`. Under `full` (a constant family through the seam): the
  run returns with 3,000 training rows, 2,560 holdout rows and
  `dropped_row_cap` 0.
- `test_too_few_of_a_class_in_the_sample`. 40,000 rows with 20 of class 1.
  Under `fast`, for seeds 0, 1 and 2 (a sample of 2,000 holds about one such
  row, whichever stream draws it): `FastProfileRowsError`, its message
  holding `full`. Under `full` (a constant family): the run returns with
  10,000 holdout rows.
- `test_the_full_profiles_error_when_full_would_stop_too`. **One test, five
  tables**: only the first is red today, the others hold what the application
  already does, and all must hold once the new error exists. For each of:
  seven timestamps of 100 rows then 2,500 then 60, with `gap=3`
  (`gap_too_large`); three timestamps of 3,000, 2,500 and 60 rows
  (`too_few_rows`); 40,000 rows with 8 of class 1 (`too_few_rows`): the call
  under `full` raises that error; the call under `fast` raises an error of
  **the same class and code**, not the new one. Then, for a table the cap
  does not touch: 1,500 rows with 6 of class 1, and nine timestamps of 10
  rows: `too_few_rows` under `fast`.

## The README (task T38)

### `tests/test_readme_duplicates.py`

Spec: 1, 2.1, step 8a. Red today, both tests. It judges `README.md`, which
the builder owns; do not edit it. Ideas are looked for in blocks (paragraphs,
and each list item as a block of its own, as in the third pass's README
test), never a sentence. Whole words, case ignored. Pin no wording beyond
these.

- `test_readme_names_the_argument_and_the_error`. The text holds
  `--duplicates`; it holds `duplicates` followed by `=` (the Python
  argument); it holds `holdout_copies_in_training`. In the blocks that hold
  `--duplicates` or the error's code, joined: the word `keep`; `1%` (or
  `1 %`) and `10` or `ten`, the two parts of the rule; and a word starting
  `time`, for the scheme under which the rule does not apply.
- `test_the_claim_about_unseen_holdout_rows_is_true_as_worded`. D49. Every
  block that holds `no fitted step has seen` also holds a word of `copy`,
  `copies`, `duplicate…`, `repeat…`. If no block holds that phrase any more,
  some block holds one of those words together with `holdout`. Today the
  README makes the claim in a paragraph that says nothing of copies.

## What must fail today, and what must pass today

Checked on 2026-11-20 with prototypes of every test below, run against the
application as it stands, against the reference implementation, and against
the reference carrying each of F9, F7c, F6b and M8 to M13
(`docs/measurements/CP5/README.md`).

**Red today: 37 tests in 99 cases, and 37 cases of the matrix.** They fail
against the application until the build stage has done D49 to D53, and pass
against the reference.

| File | Tests |
|---|---|
| `test_duplicates.py` | `test_duplicates_is_the_tenth_parameter_and_defaults_to_error`, `test_error_class_and_code`, `test_any_other_value_is_an_input_error`, `test_every_row_twice_stops_the_run`, `test_keep_runs_and_says_so`, `test_the_rule_needs_a_share_and_a_count`, `test_the_rule_in_regression`, `test_values_are_compared_as_a_step_receives_them`, `test_the_rule_is_judged_before_any_model_is_fitted`, `test_time_order_has_no_rule`, `test_the_message_names_an_undeclared_date_column`, `test_command_line_takes_duplicates` |
| `test_whole_numbers.py` | `test_read_time_keeps_integers_above_2_53`, `test_the_boundary_is_a_magnitude_above_2_53`, `test_nanosecond_stamps_are_distinct_timestamps`, `test_missing_and_repeated_integer_stamps`, `test_distinct_values_are_counted_exactly`, `test_whole_numbers_beyond_64_bits_in_every_role`, `test_command_line_on_a_26_digit_column` |
| `test_object_numbers.py` | `test_object_number_columns_give_the_result_of_their_typed_twins`, `test_a_step_is_handed_floats`, `test_which_mixtures_are_read_as_numbers`, `test_every_unsupported_type_exclusion_is_warned_about`, `test_an_object_target_is_the_typed_target`, `test_a_decimal_target_is_a_regression_target`, `test_object_whole_numbers_are_exact`, `test_an_object_time_column_is_the_typed_time_column` |
| `test_time_fraction.py` | `test_one_to_nine_digits_are_read_to_the_nanosecond`, `test_digit_counts_may_be_mixed_in_one_column`, `test_ten_digits_are_refused_and_the_message_names_the_fraction`, `test_a_csv_pandas_wrote_from_microsecond_timestamps_runs` |
| `test_fast_cap_error.py` | `test_error_class_and_code`, `test_a_small_latest_timestamp_behind_a_large_one`, `test_too_few_of_a_class_in_the_sample`, `test_the_full_profiles_error_when_full_would_stop_too` |
| `test_readme_duplicates.py` | `test_readme_names_the_argument_and_the_error`, `test_the_claim_about_unseen_holdout_rows_is_true_as_worded` |
| `test_type_matrix.py` | `test_a_result_or_a_named_error`, in the 37 cases below |

The two tests of `test_readme_duplicates.py` are red against the reference
too, until T38 is built: they judge `README.md`, not the package.

The 37 red cases of the matrix, by cause (`kind-role-extreme`):

| Cause | Cases |
|---|---|
| Whole numbers beyond 64 bits: a raw `TypeError` (15) | `csv` and `object_int`, each in the three roles, each with `beyond_64` and `beyond_64_few` (12); `mixed` in the three roles with `e300`, whose cells include `10**30` (3) |
| A target whose values cannot be compared: a raw `TypeError` (12) | `list-target`, `dict-target` and `set-target`, each with `ordinary`, `two`, `one_value` and `nan` |
| A two-valued target of a type JSON cannot hold: the run finishes and `profile.json` cannot be written (7) | `bytes-target-two`, `complex128-target-two`, `complex64-target-two`, `decimal-target-two`, `interval-target-two`, `period-target-two`, `timedelta-target-two` |
| A `Decimal` signalling not-a-number: a raw `decimal.InvalidOperation` from pandas (3) | `decimal-feature-nan`, `decimal-target-nan`, `decimal-time-nan` |

**Green today: the six guards, 768 cases of the matrix, and the loosened
test.** They pass against the application and against the reference, and
must go on passing through every task of the build.

| File | Test | Fails on |
|---|---|---|
| `test_guard_review3.py` | `test_m8_the_interval_is_the_middle_95_percent` | M8 |
| `test_guard_review3.py` | `test_m9_the_leak_screen_uses_the_previous_value` | M9 |
| `test_guard_review3.py` | `test_m10_the_fast_cap_is_a_sample_from_the_seed` | M10 |
| `test_guard_review3.py` | `test_m11_cv_std_is_the_sample_deviation` | M11 |
| `test_guard_review3.py` | `test_m12_tied_predictions_in_a_resample` | M12 |
| `test_guard_review3.py` | `test_m13_n_imputed_counts_rows_that_reach_a_model` | M13 |
| `test_type_matrix.py` | `test_a_result_or_a_named_error`, every case but the 37 | no variant is named for them (see H); the `one_value` cases of the feature role fail on F6b |
| `test_skeleton.py` | `test_profile_signature_and_defaults`, as loosened | not a guard: it passes on nine parameters and on ten |

**What the mutation check must then print**, run after your stage and before
the build: the good reference fails 0 of 1313 (the 405 counted today and 908
of this pass: 97 red cases, 6 guards, 805 of the matrix); each of F1 to F9,
F6b, F6c, F6d, F7b, F7c and M1 to M13 fails at least one test; each of M8 to
M13 fails its own guard (measured with the prototypes: that guard and no
other test); F9 fails nine cases of `test_duplicates.py`; F7c fails ten
cases of `test_whole_numbers.py` and both of
`test_object_whole_numbers_are_exact`; on the README line, 2 failing of 8. **Before your
stage it ends in `FAIL` for two reasons, both expected:** the good reference
fails one case, `test_skeleton.py::test_profile_signature_and_defaults`
(the reference has the tenth parameter and the test is not yet loosened),
and it prints `NOT CAUGHT` for F9, F7c and M8 to M13. Your stage closes both.

## Not tested in the fourth pass

The wording of any message beyond what spec 11 pins. A particular resample,
kept row or holdout row that a seed produces (rule 7). What `stages.read_time`
returns for a whole-number column above 2⁵³ that also has a missing value
(spec 11); `profile()` on such a column is tested. Whole numbers that pandas
has already turned into floats before the tool sees them: a CSV whole-number
column with an empty cell, and whole numbers written as text (spec section
6); the tool cannot give back what was lost. That two whole numbers above
2⁵³ can reach a model as one float (spec step 5): the roles are tested, the
models' inputs stay floats. The text a class value of an unusual type is
written as (spec 11). The rule on copies at 1% to 2% of repeated rows, where
the count of 10 decides and the outcome varies with the seed (spec 10.6):
the tests plant an exact number of copies instead. Run time at 100,000 rows.

# Fifth pass (CP6): entities, the two stops, `duplicates="group"`, review 4

Decisions D56 to D60. The spec sections are named beside each file. D60
withdrew the broader screen for repeated rows from this pass: **no test of
this pass is about a copy that differs from its original**, and D49's rule on
exact copies is tested as it was.

## Ground rules of the fifth pass

1. **The application already exists, and this is the fifth pass.**
   `mlprofessor/` is built and passes all 1,321 cases of the first four passes.
   Your role prompt says you write the suite before the application exists
   and that every test must fail now. In this pass that is not so: most of
   your tests must fail today, six guards and a few other tests must pass
   today, and the loosened tests of rule 3 must pass today and after the
   build. Rule 5 says which.

2. **The first four passes are final. Do not edit them, with the exceptions
   of rule 3, which D59 allows and which are listed there by name.** That
   means the 42 existing test files, `tests/conftest.py`,
   `tests/mlp_helpers.py`, `tests/cp3_helpers.py`, `tests/cp4_helpers.py`,
   `tests/cp5_helpers.py` and everything under `tests/fixtures/`: not a line,
   not a blank, not a rename, not a new fixture file. New shared helpers go in
   one new file, `tests/cp6_helpers.py`, which may import the earlier helper
   files. The driver does not lock `tests/` during your stage, so nothing will
   stop you; the owner's lead reads the diff of every existing test file
   against the committed one afterwards, and any change other than those of
   rule 3 sends the whole stage back.

3. **THE LOOSENED TESTS (D59). Nine existing tests, in five files, each pin
   something D57 or D58 changes. You loosen each so that it passes BOTH
   against the application as it stands today AND once the change is built,
   and you change nothing else in those files.** Change only the body of the
   test function named (and add the one import it needs inside that body).
   Do not touch another test, a helper, a parameter list, a module-level
   import or a single character outside those functions. **No existing test
   is ever edited so that it fails against the library as built:** run the
   five files against the application before you finish, and see all of
   their cases pass. The new behaviour is pinned exactly by the new tests
   named in the last column.

   | Existing test | What it pins that changes | The loosened form, exactly | Pinned again by |
   |---|---|---|---|
   | `test_skeleton.py::test_profile_signature_and_defaults` | the names after the ninth parameter are `[]` or `["duplicates"]` | the first nine stay pinned as now. The names after the ninth are `[]`, `["duplicates"]` or `["duplicates", "group_col"]`; when `duplicates` is there its default is `"error"`, and when `group_col` is there its default is `None`. Anything else fails | `test_groups_random.py::test_profile_has_eleven_parameters` |
   | `test_duplicates.py::test_duplicates_is_the_tenth_parameter_and_defaults_to_error` | the list of parameters is exactly ten | the first ten names stay pinned as now; the names after the tenth are `[]` or `["group_col"]`, and in the second case its default is `None`. The assertions on `duplicates` stay as they are | the same |
   | `test_duplicates.py::test_the_message_names_an_undeclared_date_column` | a table with a date column and no time column reaches the count of copies | call `p.raised(frame, families=...)` as now. If the error's code is `time_order_undeclared`, call it again with `time_col=False` and go on with that error; and pass `time_col=False` to the later `p.run_kept(frame, duplicates="keep", ...)` of the same test as well. Every assertion stays as it is. (The run with `time_col="when"` is not touched) | `test_time_order_stated.py::test_the_date_error_comes_before_the_copies_error` |
   | `test_time_pipeline.py::test_scheme_follows_time_col` | a table with a date column and no time column runs under the random scheme | wrap the plain run in `try`; on a `ProfileError` whose code is `time_order_undeclared`, run again with `time_col=False` and use that; any other error is raised again. Every assertion stays | `test_time_order_stated.py::test_the_statement_gives_the_random_scheme_and_is_recorded` |
   | `test_time_pipeline.py::test_date_column_undeclared` | the same, and its warnings | the same fallback for its one run | the same |
   | `test_edges.py::test_limitations_and_caveat_are_the_fixed_texts` | the same, in its two `random` cases | the same fallback, **only when `scheme == "random"`**: under `time` any error is raised again | the same |
   | `test_time_pipeline.py::test_panel_uses_training_mean_and_flags_persistence` | a persistent panel with no entity column runs against the training mean, warned | wrap the run in `try`; on a `ProfileError` whose code is `entities_undeclared`, **and only in the `persistent` case**, run again with `group_col=False` and use that. Every assertion stays | `test_panel_stop.py::test_the_statement_lets_the_run_go_ahead_as_before` |
   | `test_console_summary.py::test_summary_shows_both_intervals_the_p_value_and_each_warning` | the command line runs a table with a date column and no `--time-col` | before the run the test makes now, make a probe run of the same table and arguments into a folder of its own under `tmp_path`, through `h.run_cli`. If the probe exits 3 and its standard error holds `time_order_undeclared`, pass `--no-time-order` as one more argument to the run the test makes now; otherwise pass nothing more. Every assertion stays | `test_time_order_stated.py::test_the_command_line_states_it` |
   | `test_console_summary.py::test_summary_says_why_there_is_no_interval` | the command line runs a persistent panel with `--time-col` and no entity column | the same probe, with `--time-col date`; if it exits 3 with `entities_undeclared`, pass `--no-entities` to the run the test makes now | `test_panel_stop.py::test_the_command_line_says_there_are_no_entities` |

   Why a probe, and not a fixed extra argument: today's command line does not
   know `--no-time-order` or `--no-entities` and would refuse them (exit 2),
   so a test that always passed them would fail today. Why a fallback in
   `except`, and not `time_col=False` from the start: today's `profile()`
   treats `False` as an invalid value of `time_col`. Each loosened test was
   run on 2026-11-21 as a prototype: all 96 cases of the five files pass
   against the application as it stands and against the reference
   implementation as changed.

   **One existing test is NOT loosened, by ruling (D60):**
   `test_duplicates.py::test_time_order_has_no_rule` stays exactly as it is.
   D49's rule is unchanged, and that test passes before and after.

   **One README test is NOT loosened either:**
   `test_readme_examples.py::test_every_readme_command_runs` runs the
   README's commands on `examples/customers.csv`, which has a column of
   dates. It is green today and stays green: the task that builds the stop
   (T42) adds `--no-time-order` to those commands in the same change.

4. **You write ten new test files and the helper file, and nothing else.**
   The files are listed below, each with the tests it must hold, **under the
   names given**: the lists of rule 5 and the mutation check go by name.
   Split a named test into parameter cases as the text says. If you add a
   test of your own, say so in your final report, with the list of rule 5 it
   belongs to.

5. **Which tests fail today and which pass today is fixed, by name.** The
   lists are at the end of this section. A test on the red list fails against
   the application as it stands, in every one of its parameter cases unless
   the list says otherwise. A test on the green list passes today in every
   case. Run the new files before you finish and compare with the lists.
   **Do not weaken a guard to make it fail, and do not weaken a red test to
   make it pass.** If a test on the green list fails, or one on the red list
   passes, and you believe your test is what the spec says, stop and report
   it with the output.

6. **Every new variant must fail a test, and `verify/mutation_check.py` is how
   that will be judged.** It runs the whole suite against a copy of the
   reference implementation and again with the reference broken in one way at
   a time: seventeen forms the gate also catches (F1 to F10, F6b, F6c, F6d,
   F7b, F7c, F7d, F8b) and twenty-two smaller faults (M1 to M22; spec 8 and
   8.1). It passes only if the good reference fails none of your tests (the
   README tests apart) and every variant fails at least one. Twelve are new,
   and each has a test of this pass named for it:

   | Variant | The test that must fail on it |
   |---|---|
   | F10 | `test_groups_random.py::test_no_entity_on_both_sides_or_in_two_folds` |
   | F7d | `test_time_order_stated.py::test_a_date_column_and_no_word_about_time_stops_the_run` |
   | F8b | `test_groups_time.py::test_each_entity_has_its_own_previous_value` |
   | M14 | `test_read_once.py`, its first four tests |
   | M15 | `test_groups_random.py::test_no_entity_on_both_sides_or_in_two_folds` |
   | M16 | `test_groups_time.py::test_the_leak_screen_measures_a_lag_against_the_entity_baseline` |
   | M17 | `test_groups_random.py::test_the_fast_cap_keeps_whole_entities` |
   | M18 | `test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite` |
   | M19 | `test_panel_stop.py::test_a_persistent_panel_with_no_entity_column_stops` |
   | M20 | `test_guard_review4.py::test_m20_infinite_cells_of_the_target_and_the_time_column_are_counted` |
   | M21 | `test_groups_random.py::test_the_interval_resamples_whole_entities_and_there_is_no_p_value` |
   | M22 | `test_duplicates_group.py::test_each_set_of_copies_stays_on_one_side_and_in_one_fold` |

   You may run it yourself, from the repo root, with the project interpreter:
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/mutation_check.py --jobs 8`.
   It writes nothing inside the repo. Do not read the reference
   implementation to tune a test to it. The README test of this pass goes in
   a file named `tests/test_readme_entities.py`: files named
   `tests/test_readme*.py` are shown on a line of their own and not counted.

7. **Which rows land where under the grouped scheme is open (spec 7 and 11).**
   Two correct implementations hold out different entities from the same
   seed, keep different entities under the fast profile's cap and draw
   different resamples. So no test of this pass asserts a particular holdout
   row, a particular kept entity or an interval end worked out for one
   stream. The tests assert the five properties of spec step 7, the
   properties of the cap in step 6, and, within one implementation, that
   `profile()` and `stages.split_random(..., groups=...)` agree. Where a test
   needs a run's holdout and training rows it finds them out from the run
   itself: `h.holdout_rows(result)`, and the recording fakes through the
   seam. The guard for M21 is written by effect, as the one for M4 was; its
   recipe is below.

8. **The rules of the earlier passes still hold**: the documented surface only
   (spec 2.1 to 2.7); pin what the spec pins and nothing else; recompute, do
   not trust; assert on values as the application produced them; **no
   skipped or expected-to-fail test**; nothing written outside `tmp_path`;
   the application imported inside test functions; the command line run one
   process at a time through `mlp_helpers.run_cli`. What a message must hold
   is pinned in spec 11 and nowhere else. Tiny tables, with the exceptions
   this pass needs: one table of about 3,000 rows for the cap, one of 1,600
   rows for the interval, and the entity role of the type matrix, 353 tables
   of 200 rows. Use a recording family or a constant family through the seam
   wherever a run only has to finish.

## Files of the fifth pass

Each file belongs to exactly one task of `plan/tasks.json`.

| File | Task | About | Today |
|---|---|---|---|
| `tests/test_guard_review4.py` | T39 | what the fourth review pinned and the application already does (D57) | green |
| `tests/test_read_once.py` | T40 | each object column is read once, over all loaded rows (finding 1) | red |
| `tests/test_review4_small.py` | T41 | no part of the output holds a number that is not finite (finding 4); the three-class message | red, two cases green |
| `tests/test_time_order_stated.py` | T42 | an undeclared date column stops the run; `time_col=False` (D58) | red, two tests green |
| `tests/test_groups_random.py` | T43 | `group_col` under the random scheme: the grouped split, the cap, the interval (D57) | red |
| `tests/test_groups_time.py` | T44 | entities under time order: each entity's previous value, the leak screen (D57) | red |
| `tests/test_panel_stop.py` | T44 | a panel with no entity declared stops and asks (D58) | red, one test green |
| `tests/test_duplicates_group.py` | T45 | `duplicates="group"` on the exact copies (D57, D60) | red, one test green |
| `tests/test_type_matrix_group.py` | T46 | every kind of column as the entity column (D52, D57) | red |
| `tests/test_readme_entities.py` | T47 | the README on this pass | red |
| `tests/cp6_helpers.py` | none | shared helpers of the fifth pass; holds no test | |
| `tests/test_skeleton.py`, `test_duplicates.py`, `test_time_pipeline.py`, `test_edges.py`, `test_console_summary.py` | (earlier tasks) | nine tests loosened as rule 3 says; nothing else | green before and after |

### `tests/cp6_helpers.py`

Import `mlp_helpers as h`, `cp3_helpers as c` and `cp5_helpers as p`, and
build on them; do not copy from them. What the new files need:

- **`GROUPED_CAVEAT`**: the fixed text of spec 5.4 for the grouped scheme.
- **`entity_frame(n_entities=60, sizes=5, seed=0, task="classification")`**:
  rows in entities. A text column `who` (`e0000`, ...), two measures `m1`
  and `m2` that are alike within an entity (an entity's centre plus normal
  noise of 0.05), and a target `y` that belongs to the entity: for
  classification half the entities are 0 and half 1, drawn from the seed;
  for regression a normal value per entity plus noise of 0.1. `sizes` is one
  number or one number per entity. Rows in an order shuffled from the seed.
  No two rows agree in every column.
- **`walks(n_series=6, n_times=60, seed=0, start=0)`**: a panel of random
  walks on the same numeric timestamps `t` (`start`, `start + 1`, ... as
  floats): columns `t`, `who` (`s00`, ...), `prev` (the level one step
  before, of the same series) and `y` (the level). Each series starts at a
  level of its own, ten apart. Rows in shuffled order.
- **`sides(result, frame, column="who")`**: the set of entities of the
  training rows and the set of the holdout rows, for a run in which no row
  was dropped (it asserts that).
- **`fits(log)`**: for an `h.Log`, the list of `(fit rows, predicted rows)`
  of every model the run made, in the order they were made: the five folds
  of each configuration, then the final fit.
- **`previous_by_hand(frame, gap=0, time="t", who="who", target="y")`**: for
  each row position, the target of the same entity's latest row at least
  `gap + 1` distinct timestamps earlier, or `None`; the distinct timestamps
  are those of all the rows of `frame`.
- `run`, `raised` and `count_code`, as `c.run`, `p.raised` and
  `c.count_code`.

## The guards (task T39; all green today)

### `tests/test_guard_review4.py`

Spec: steps 2 and 5 (`n_infinite`), 2.3 (the constructor), 2.6 (what a model
hands back). Six tests, all green today. A failure means the application is
wrong, not the test.

- `test_m20_infinite_cells_of_the_target_and_the_time_column_are_counted`.
  **Fails on M20.** 240 rows, two numeric features, a numeric time column
  `t` and a numeric target; two target cells are `inf`, three time cells
  `-inf`, in five different rows. With `time_col="t"`:
  `columns` reports `n_infinite` 2 for the target and 3 for the time column,
  `n_missing` 0 for both; `rows.dropped_missing_target` is 2 and
  `dropped_missing_time` 3; no `infinite_values` warning.
- `test_an_infinite_float_in_an_object_target_is_a_missing_target`. An object
  target of the text `a` and `b` in which three cells hold an infinite float
  (`float("inf")`, `numpy.float64("inf")`, `-numpy.inf`) and one holds
  `None`: the run finishes; `dropped_missing_target` is 4; the target's
  record has `n_infinite` 3, `n_missing` 1 and `dtype` `object`;
  `task.classes` is `["a", "b"]`.
- `test_an_object_target_of_numbers_with_an_infinite_one`. An object target
  of `Decimal` values, one of them `Decimal("-Infinity")`: regression; one
  row dropped; `n_infinite` 1.
- `test_a_class_beside_infinite_floats_alone_is_a_constant_target`. An object
  target of the text `a` and infinite floats: `target_constant`.
- `test_a_result_can_be_made_from_its_own_document`. From a finished run:
  `mlprofessor.ProfileResult(result.to_dict(), result.predictions)` has the
  same `to_dict()`, the same `headline`, and predictions equal to the
  other's.
- `test_a_regression_model_may_return_a_column_of_values`. Three families
  through the seam on one regression table: a lookup model; one whose
  `predict` returns its values reshaped to a column, `(n, 1)`; one whose
  `predict` returns one value too few. Their statuses are `fitted`,
  `fitted` and `failed`.

## A: each object column is read once (task T40)

### `tests/test_read_once.py`

Spec: step 1a ("Each column is read once", and its table), step 3. Five
tests, all red today: on each table the application returns a result, and on
the fifth it writes the stamps as floats. The first four fail on M14. The
fifth fails on no variant: the reference writes the stamps of a whole-number
time column as whole numbers whichever way its keys are held, and the test
pins that they are whole numbers when the keys are exact. 240 rows, two
numeric features; "whole numbers" are Python integers in an object column.

- `test_a_text_cell_of_an_object_target_in_a_row_with_no_time_value`. An
  object target of whole numbers, with a numeric time column. Put the text
  `odd` in one target cell. With every time value present the run raises a
  named error; take its code. Now empty the time value of that same row:
  the run must raise the same code, and that code is `task_ambiguous`.
- `test_a_whole_number_beyond_64_bits_in_a_target_row_with_no_time_value`.
  The same target, one cell `10**25`, in a row with no time value:
  `task_unsupported`.
- `test_a_text_cell_of_an_object_time_column_in_a_row_with_no_target`. An
  object time column of whole numbers with one text cell: with the target
  present in that row, and with the target of that row empty, the same
  code, `time_column_invalid`.
- `test_a_time_value_beyond_64_bits_in_a_row_with_no_target`. A time column
  of Python integers, one `10**25`, in a row with no target:
  `time_column_invalid`.
- `test_exact_integer_keys_are_decided_on_the_loaded_column`. An `int64`
  time column `0, 100, 200, ...` in which row 7 holds `2**53 + 1`, and row
  7 has no target. The run finishes; `validation.time_form` is `numeric`;
  `train_end`, `holdout_start` and `holdout_end` are each an `int` (not a
  `bool`, not a `float`), and `train_end` and `holdout_start` are 17900 and
  18000.

## B: findings 4 and 5, and the three-class message (task T41)

### `tests/test_review4_small.py`

Spec: 2.3, "Numbers that are not finite" rule 6, step 4. Finding 5 (spec
step 9, "What can fail a configuration, and what cannot") has no test: no
input reaches the tool's own scoring with predictions that are not already
one finite number for each row. It is pinned in the spec and in T41's brief.

- `test_no_part_of_the_output_can_hold_a_number_that_is_not_finite`, eight
  cases, one per part: `task`, `columns`, `input`, `rows`, `meta`,
  `warnings`, `headline`, `validation`. **Fails on M18** in the first six.
  Take the `to_dict()` and the predictions of a finished run. First, a
  `ProfileResult` made from a deep copy of the document has that same
  `to_dict()`. Then plant one float that is not finite in the named part
  (for example `task.classes` becomes `[0.0, inf]`; `columns[0].n_missing`
  becomes NaN; `input.gap` becomes `-inf`; `rows[0].count` NaN;
  `meta.timing.seconds` `inf`; one more warning record whose `message` is
  NaN; `headline.score` NaN; `validation.holdout_share` a numpy NaN) and
  construct: `NonFiniteScoreError`, code `non_finite_score`. Red today in
  the first six cases, where the application makes the result and writes
  null; green today in `headline` and `validation`.
- `test_the_three_class_message_does_not_suggest_declaring_the_task`. A text
  target of three values under `task="auto"`: `task_ambiguous`; the message
  holds `hree or more classes are not supported` and `3`, and holds neither
  `task=` nor `eclare`. Red today.
- `test_a_numeric_target_of_few_values_still_names_regression`. A numeric
  target of five distinct values: the message holds `task='regression'` and
  `hree or more classes are not supported`. Red today.

## C: an undeclared date column stops the run (task T42)

### `tests/test_time_order_stated.py`

Spec: 2.1 (`time_col`), 2.2, 2.4, 2.5, step 5, step 8a, 5.2, 11. Tables of
200 rows from `h.time_frame(..., task="classification")`, whose `date` column
is a datetime; "as text" means that column written `YYYY-MM-DD`.

- `test_a_date_column_and_no_word_about_time_stops_the_run`, two cases:
  datetime, and ISO text. **Fails on F7d.** `TimeOrderUndeclaredError`, code
  `time_order_undeclared`; the message holds `'date'`, `time_col`,
  `--time-col`, `time_col=False` and `--no-time-order`; nothing is written
  to the output folder.
- `test_the_message_names_every_date_column`. With a second datetime column
  `shipped`: the message holds `'date'` and `'shipped'`.
- `test_the_statement_gives_the_random_scheme_and_is_recorded`.
  `time_col=False`: `input.time_col` is `False`; `validation.time_col` is
  null; the scheme is `random`; the caveat is `h.CAVEAT`; exactly one
  `date_column_undeclared` warning names `date`; the column is excluded as
  `datetime`.
- `test_the_statement_on_a_table_with_no_dates_changes_only_its_record`. On
  `h.clf_frame(200)`: `input.time_col` is null without the statement and
  `False` with it, and the two documents are equal once `meta.timing` and
  `input.time_col` are taken out.
- `test_a_declared_time_column_lets_other_date_columns_be_excluded`. Green
  today. `time_col="date"` on a regression table with a second date column:
  that column's reason is `datetime` and there is no
  `date_column_undeclared` warning.
- `test_values_of_time_col_that_are_refused`. Green today. `time_col=True`,
  and `time_col=False` with `gap=2`: `InputError`.
- `test_the_date_error_comes_before_the_copies_error`. `p.twice` of a noise
  table of 300 rows, with a datetime column `seen` of a different date in
  every row. By default: `time_order_undeclared`. With `time_col=False`:
  `HoldoutCopiesError`, whose message holds `'seen'` and `time_col`.
- `test_the_command_line_states_it`. The table as a CSV. No argument: exit
  3, `time_order_undeclared` on standard error, nothing written.
  `--no-time-order`: exit 0, `input.time_col` is `false` and the scheme
  `random` in `profile.json`, and the summary holds
  `date_column_undeclared`. `--no-time-order` with `--time-col date`: exit
  2, nothing written.

## D: a declared entity column under the random scheme (task T43)

### `tests/test_groups_random.py`

Spec: 2.1 (`group_col`), 2.2, 2.4, 2.7, steps 2, 5, 6, 7, 8a, 12, 5.2, 5.4.
All red today. Recording family: `h.family("rec", log)`.

- `test_profile_has_eleven_parameters`. The parameters of
  `mlprofessor.profile` are exactly `data`, `target`, `task`, `time_col`,
  `gap`, `profile`, `seed`, `out_dir`, `n_jobs`, `duplicates`, `group_col`;
  the last two default to `"error"` and `None`.
- `test_no_entity_on_both_sides_or_in_two_folds`, two cases: classification
  and regression. **Fails on F10 and on M15.** `entity_frame(60, sizes=5)`
  with the recording family and a recording step, `group_col="who"`. No
  entity is among both the training rows and the holdout rows. `fits(log)`
  has six entries. For each of the first five: neither set holds a holdout
  row, and no entity is among both its fit rows and its predicted rows.
  The five sets of predicted rows are disjoint and together are every row
  that is not a holdout row. The sixth is (the training rows, the holdout
  rows). No step is fitted on a row of a holdout entity.
- `test_holdout_is_a_quarter_of_the_rows_in_whole_entities`, two cases. 90
  entities of 1 to 8 rows. For regression, over all entities; for
  classification, for the entities of each class apart (an entity counts
  with the class most of its rows hold, the positive class on a tie): four
  times the held rows are at least the rows, and the held rows are fewer
  than a quarter of the rows plus the largest of those entities.
  `validation.holdout_share` is the holdout rows over all rows.
- `test_the_output_reports_the_grouped_scheme`. `validation.scheme` is
  `grouped`; `caveat` is `GROUPED_CAVEAT`; `validation.group_col` and
  `input.group_col` are `who`; `n_groups_training` and `n_groups_holdout`
  are the counts by hand; `stratified` is true; `n_holdout_without_previous`
  is null; the record of `who` has role `group` and null `kind` and
  `reason`; `n_holdout_copies` is 0 and `holdout_copy_share` 0.0; the
  document is strict JSON.
- `test_without_an_entity_column_the_keys_are_null`. On `h.clf_frame(200)`:
  the scheme is `random`; `input.group_col`, `validation.group_col`,
  `n_groups_training`, `n_groups_holdout` and `n_rows_with_copies` are null.
- `test_group_col_false_is_recorded_and_changes_nothing_else`. The documents
  of two runs, with and without `group_col=False`, are equal once
  `meta.timing` and `input.group_col` are taken out; `input.group_col` is
  `False` in one.
- `test_the_interval_resamples_whole_entities_and_there_is_no_p_value`.
  **Fails on M21.** 160 entities of ten rows; each entity holds one class,
  alternately 0 and 1; a lookup family predicts 0.9 for the entity's class
  on the entities it is "right" about (each with probability 0.6, drawn
  once from a fixed generator) and 0.1 on the others, so every row of an
  entity is right or wrong together. `interval_method` is
  `bootstrap_groups`; `block_length` is null; `n_resamples_used` is 200;
  `p_value` and `n_permutations` are null; `p_value_reason` holds
  `independent rows`; the score lies in its interval; and
  **`interval_high - interval_low` is above 0.18.** Measured on the
  reference over 20 seeds: 0.252 to 0.346; with rows drawn singly (M21):
  0.080 to 0.103.
- `test_fewer_than_thirty_holdout_entities_no_interval`. 60 entities of five
  rows hold out about 15: `interval_method`, both ends, `skill_low` and
  `p_value` are null; `interval_reason` holds the number of holdout
  entities and `30`; a `no_interval` warning.
- `test_too_few_entities`. Regression. Five entities of 40 rows:
  `TooFewGroupsError`, code `too_few_groups`. Six entities of 40 rows: the
  same class (two are held out, four are left for five folds). Eight
  entities of 30 rows: the run finishes.
- `test_an_empty_entity_cell_in_a_usable_row_is_an_error`. One entity cell
  `None`: `GroupColumnError`, code `group_column_invalid`, the message
  holding `who`; nothing written. One entity cell of three spaces: the same
  code. A row whose entity cell and target are both empty: the run finishes
  with `dropped_missing_target` 1.
- `test_arguments_that_cannot_be_an_entity_column`. `group_col="y"` (the
  target), `True`, `3`, and `group_col="t"` with `time_col="t"`:
  `InputError`. `group_col="nobody"`: `ColumnNotFoundError`.
- `test_every_row_its_own_entity_is_allowed`. `who` is the row number: the
  scheme is `grouped`, the two counts of entities add up to the rows, and
  the role is `group` (an entity column is never excluded as an
  identifier).
- `test_a_class_held_by_one_entity_is_one_class`. Every positive row belongs
  to one entity: `OneClassError`.
- `test_the_fast_cap_keeps_whole_entities`. **Fails on M17.** 520 entities of
  1 to 11 rows, more than 2,500 rows, fast profile. The rows kept (training
  plus holdout) are at most 2,000 and `dropped_row_cap` is the rest; the
  rows a recording family saw, with the holdout rows, are whole entities
  (the sizes of the entities among them add up to the rows kept); some
  entity is left out, and the rows kept plus the smallest entity left out
  exceed 2,000; a run with another seed holds out other rows.
- `test_copies_across_entities_are_still_counted`. The entity table twice,
  the second copy under other entity names: by default
  `HoldoutCopiesError`; with `duplicates="keep"` the scheme is `grouped`,
  `n_holdout_copies` is at least 10 and there is one `holdout_copies_kept`
  warning.
- `test_the_command_line_takes_the_entity_column`. `--group-col who`: exit
  0, the scheme `grouped` and `input.group_col` `who` in `profile.json`.
  `--no-entities`: exit 0, `input.group_col` is `false`, the scheme
  `random`. Both together: exit 2, nothing written. `--group-col nobody`:
  exit 3 with `column_not_found`.

One more test belongs here if you have room for it, and the mutation check
does not need it: `profile()` against `stages.split_random(n, classes, seed,
groups=...)` on a table with a few rows that have no target, with the
entities renamed: the same holdout rows (spec 2.7).

## E: entities under time order, and the panel stop (task T44)

### `tests/test_groups_time.py`

Spec: 2.7 (`previous_value`), steps 7, 8, 11, 5.2. All red today.

- `test_each_entity_has_its_own_previous_value`, two cases: `gap` 0 and 2.
  **Fails on F8b.** `walks(6, 60)`; a lookup family predicting `prev + 0.25`;
  `time_col="t"`, `group_col="who"`. The holdout is the timestamps 45 to
  59. Both baselines are in `baselines`, with the scores computed by hand
  from `previous_by_hand(frame, gap)` and from the mean of the training
  rows (those before timestamp `45 - gap`), to a relative 1e-9; the
  previous value has the lower error, is the one used and is
  `headline.baseline`; `headline.skill` is 1 less the model's error over
  it. `validation`: scheme `time_ordered`, `group_col` `who`, six entities
  on each side, `n_holdout_without_previous` 0, `timestamps_unique` false.
  The role of `who` is `group`. No `autocorrelated_panel_target` warning.
- `test_the_split_is_the_one_time_order_gives_without_entities`. The same
  timestamps and entities with a 0/1 target that depends on a noise column:
  the holdout rows and the row counts with `group_col="who"` equal those
  without it, and `baselines` is the one record of `chance`.
- `test_an_entity_with_no_earlier_row_gets_the_training_mean`. Five walks
  over timestamps 0 to 59 and a sixth, `late`, over 50 to 59. Exactly one
  holdout row has no previous value by hand (the first row of `late`). The
  score of `previous_value` equals the error by hand with the training mean
  in place of that row's missing value; `n_holdout_without_previous` is 1;
  one `no_previous_value` warning; five entities have a training row and
  six a holdout row.
- `test_a_repeated_timestamp_inside_one_entity_is_an_error`. One row of the
  panel written twice: `GroupColumnError`, `group_column_invalid`, nothing
  written. The same table with a 0/1 target of noise: the run finishes.
- `test_the_leak_screen_measures_a_lag_against_the_entity_baseline`.
  **Fails on M16.** On `walks(6, 60)` with the recording family: `prev` has
  the role `feature` and a null reason, and there is no `suspected_leak`
  warning.
- `test_the_command_line_takes_both_columns`. `--time-col t --group-col
  who`: exit 0; `headline.baseline` is `previous_value`.

### `tests/test_panel_stop.py`

Spec: 2.4, 2.5, step 11. Tables from `h.panel_frame`.

- `test_a_persistent_panel_with_no_entity_column_stops`. **Fails on M19.**
  `EntitiesUndeclaredError`, code `entities_undeclared`; the message holds
  `group_col`, `--group-col`, `group_col=False` and `--no-entities`;
  nothing written. Red today.
- `test_the_statement_lets_the_run_go_ahead_as_before`. `group_col=False`:
  `input.group_col` is `False`, `validation.group_col` null, the one
  baseline is `training_mean`, and there is one
  `autocorrelated_panel_target` warning. Red today.
- `test_a_panel_that_does_not_persist_runs_without_a_word`.
  `persistent=False`: no warning; `input.group_col` is null. Red today, for
  the key alone, which does not exist yet.
- `test_a_classification_panel_is_not_asked`. A 0/1 target of noise on the
  persistent panel's timestamps: the run finishes under `time_ordered`.
  Green today.
- `test_the_command_line_says_there_are_no_entities`. With `--time-col
  date`: exit 3 and `entities_undeclared`, nothing written. With
  `--no-entities` as well: exit 0 and the summary holds
  `autocorrelated_panel_target`. Red today.

## F: `duplicates="group"` (task T45)

### `tests/test_duplicates_group.py`

Spec: 2.1 (`duplicates`), 2.2, 2.5, steps 7 and 8a, 5.2. A copy is an exact
copy, as in `test_duplicates.py`; this file does not test rows that differ.

- `test_each_set_of_copies_stays_on_one_side_and_in_one_fold`. **Fails on
  M22.** `p.twice(p.noise_frame(300))` with the recording family and step,
  `duplicates="group"`. By the rows' own values (the four features and the
  target as a tuple): no tuple is among both the holdout rows and the
  others, and in every entry of `fits(log)` no tuple is among both the fit
  rows and the predicted rows. `validation`: scheme `grouped`, the grouped
  caveat, `n_rows_with_copies` 600, `n_holdout_copies` 0 and
  `holdout_copy_share` 0.0, `n_groups_training` and `n_groups_holdout` the
  distinct tuples on each side, `group_col` null. `input.duplicates` is
  `group`. One `copies_grouped` warning, no `holdout_copies_kept`, one
  `duplicate_rows`. The interval is `bootstrap_groups` and there is no
  p-value.
- `test_a_table_with_no_copies_is_split_as_by_default`. On
  `h.clf_frame(200)`: the holdout rows equal those of a run without the
  argument; the scheme is `random`; `n_rows_with_copies` is 0 and there is
  no `copies_grouped` warning; without the argument the key is null.
- `test_a_few_copies_are_grouped_and_counted`. 300 rows and the first seven
  again: no copy on both sides; `n_rows_with_copies` 14; the scheme is
  `grouped`; the units on the two sides add up to 300.
- `test_copies_and_declared_entities_are_kept_together_both`. The entity
  table, with the rows of ten entities written again under ten other names:
  with `duplicates="group"` and `group_col="who"`, no entity and no tuple
  of values is on both sides; `n_rows_with_copies` is 100; the units on the
  two sides add up to 60, each of the ten pairs of entities having become
  one; `n_holdout_copies` is 0.
- `test_under_time_order_it_changes_nothing`. With a numeric time column:
  `n_rows_with_copies` is null, the scheme `time_ordered`, and the document
  equals the one without the argument once `meta.timing` and
  `input.duplicates` are taken out.
- `test_copies_that_appear_when_the_leak_screen_excludes_a_column_stop_the_run`.
  The doubled noise table with one more column, `leak`: the target plus
  0.001 times the row number, so that no two rows agree in it and it
  predicts the target on its own. With `duplicates="group"`:
  `HoldoutCopiesError`, whose message holds `'leak'` and `group`.
- `test_the_default_message_names_the_third_value`. On the doubled table by
  default: the message holds `group` and `keep`.
- `test_other_values_are_refused`, five cases: `"groups"`, `"GROUP"`, `""`,
  `None`, `True`: `InputError`. Green today.
- `test_the_command_line_takes_group`. `--duplicates group` on the doubled
  table: exit 0; the scheme `grouped` and `n_holdout_copies` 0 in
  `profile.json`; the summary holds `copies_grouped`.

## G: the type matrix, the entity role (task T46)

### `tests/test_type_matrix_group.py`

Spec: section 7, "The type matrix", "The entity role". One test,
`test_a_result_or_a_named_error_as_the_entity_column`, parametrised; 353
cases, all red today (the application has no `group_col`). Import the first
matrix, `test_type_matrix`, and use its kinds, its extremes, its `Ctx`, its
`one_value` and its `_assert_result`; add nothing to that file. For each
kind and each extreme the kind allows in the `feature` role, the column is
`who`, beside `x1` and `z1`, with the 0/1 target of the first matrix's
feature role and `group_col="who"`; the case is named
`<kind>-group-<extreme>`. For each kind with its ordinary values, once more
with a numeric time column `t` of 0.0 to 199.0 and the numeric target of the
first matrix's time role; the case is named `<kind>-group-ordinary-timed`.
Fast profile, with an output folder. A `ProfileError` with a code and a
message, and nothing written: the case passes. A result: the contract of the
first matrix (`_assert_result`), and the record of `who` has the role
`group` and `validation.group_col` is `who`. Any other exception propagates
and fails the case.

## The README (task T47)

### `tests/test_readme_entities.py`

Spec: 1. Red today, both tests. It judges `README.md`, which the builder
owns; do not edit it. Pin no wording beyond these.

- `test_the_readme_names_the_new_arguments`. The text holds `--group-col`,
  `--no-entities`, `--no-time-order`, `group_col` and `time_col=False`, and
  `duplicates` followed by `=` or a space and then `group` (quoted or not).
- `test_the_readme_says_that_a_date_column_must_be_declared_or_dismissed`.
  The text holds `time_order_undeclared` and `entities_undeclared`.

## What must fail today, and what must pass today

Checked on 2026-11-21 with prototypes of every test below, run against the
application as it stands, against the reference implementation, and against
the reference carrying each of F10, F7d, F8b and M14 to M22.

**Red today.** They fail against the application until the build stage has
done D57 and D58, and pass against the reference.

| File | Tests |
|---|---|
| `test_read_once.py` | all five |
| `test_review4_small.py` | `test_no_part_of_the_output_can_hold_a_number_that_is_not_finite` in six cases (`task`, `columns`, `input`, `rows`, `meta`, `warnings`); `test_the_three_class_message_does_not_suggest_declaring_the_task`; `test_a_numeric_target_of_few_values_still_names_regression` |
| `test_time_order_stated.py` | all but the two on the green list |
| `test_groups_random.py` | all |
| `test_groups_time.py` | all |
| `test_panel_stop.py` | all but `test_a_classification_panel_is_not_asked` |
| `test_duplicates_group.py` | all but `test_other_values_are_refused` |
| `test_type_matrix_group.py` | all 353 cases |
| `test_readme_entities.py` | both; red against the reference too, until T47 is built: they judge `README.md` |

**Green today.** They pass against the application and against the
reference, and must go on passing through every task of the build.

| File | Test | Fails on |
|---|---|---|
| `test_guard_review4.py` | `test_m20_infinite_cells_of_the_target_and_the_time_column_are_counted` | M20 |
| `test_guard_review4.py` | the other five | not guards of a variant: they hold what was ruled and the application already does |
| `test_review4_small.py` | `test_no_part_of_the_output_can_hold_a_number_that_is_not_finite[headline]` and `[validation]` | |
| `test_time_order_stated.py` | `test_a_declared_time_column_lets_other_date_columns_be_excluded`, `test_values_of_time_col_that_are_refused` | |
| `test_panel_stop.py` | `test_a_classification_panel_is_not_asked` | |
| `test_duplicates_group.py` | `test_other_values_are_refused`, five cases | |
| the five files of rule 3 | every test, the nine loosened ones among them | not guards: each passes before the change and after it |

Measured with the prototypes: 428 cases in the ten new files and 96 in the
five files of rule 3, 524 in all. Against the application as it stands, 112
pass (the 96, the six guards and ten more cases of the green list) and 412
fail. Against the reference, 522 pass and two fail, the README's.

**What the mutation check must then print**, run after your stage and before
the build: the good reference fails 0 of 1739 (the 1,313 counted today and
426 of this pass: the 428 cases of the ten new files less the two README
tests); each of F1 to F10, F6b, F6c, F6d, F7b, F7c, F7d, F8b and M1 to M22
fails at least one test (measured with the prototypes on 2026-11-21: F10
fails 11 cases, F7d 5, F8b 4, M14 4, M15 3, M16 5, M17 1, M18 6, M19 2, M20
3, M21 1, M22 4); on the README line, 3 failing of 10: the two tests of
`test_readme_entities.py` and the one named below. **Before your stage it ends in `FAIL` for two
reasons, both expected:** the good reference fails ten cases, in the nine
tests of rule 3 (the reference has the new behaviour and the tests are not
yet loosened), and it prints `NOT CAUGHT` for F10, F7d, F8b and M14 to M22
(measured on 2026-11-21: no earlier test fails on any of the twelve). Your
stage closes both. On the README line it shows one more failure until T42 is
built, in every case alike: `test_readme_examples.py::test_every_readme_command_runs`,
because the reference stops on the README's own commands, which do not yet
say `--no-time-order`.

## Not tested in the fifth pass

A copy of a row that differs from it in any column: not detected in this
version, by decision (D60). The wording of any message beyond what spec 11
pins. Which entities are held out, which fold each is in and which the cap
keeps (rule 7). The boundary of spec step 9 between a model's failure and a
fault of the tool's own scoring (finding 5): it cannot be reached from
outside, and is judged by reading. A `Decimal` infinity in an object target
that is not read as numbers: it is a value like any other there (spec step
2), and no test pins what follows from that. Run time at 100,000 rows.

# Sixth pass (CP7): both statements on every call, `"group"` on the features, the grouped holdout, dates as whole numbers

Decisions D62 to D66. The spec sections are named beside each file. **No test
of this pass is about a copy that differs from its original, or about the
gate checks never seen failing:** both are pass 7 (D60, D64).

## Ground rules of the sixth pass

1. **The application already exists, and this is the sixth pass.**
   `mlprofessor/` is built and passes all 1,757 cases of the first five
   passes. Your role prompt says you write the suite before the application
   exists and that every test must fail now. In this pass that is not so:
   most of your tests must fail today, three guards and some other tests must
   pass today, and every existing test must pass today and after the build.
   Rule 5 says which.

2. **The first five passes are final. Do not edit them, with the exceptions of
   rule 3, which D59 and D65 allow and which are listed there by name.** That
   means the 52 existing test files, `tests/conftest.py`,
   `tests/mlp_helpers.py`, `tests/cp3_helpers.py`, `tests/cp4_helpers.py`,
   `tests/cp5_helpers.py`, `tests/cp6_helpers.py` and everything under
   `tests/fixtures/`. New shared helpers go in one new file,
   `tests/cp7_helpers.py`. The owner's lead reads the diff of every existing
   file against the committed one afterwards, and any change other than
   those of rule 3 sends the whole stage back.

3. **THE LOOSENING (D59, D65). 39 replacements in 22 existing files, and no
   other change to any of them.** D63 makes every call say two things; 1,435
   of the 1,757 existing cases leave at least one unsaid, and in 1,006 of
   them (193 test functions in 39 files) the call would now stop. D65 lets
   the change be made once in a helper. The rule of every helper is the same:
   **it makes the call as the test wrote it, and makes it again with the
   missing statement only when the run stops with `time_order_undeclared` or
   `entities_undeclared` for a statement the call did not make; any other
   error is raised as it is.** Against the application as it stands a helper
   so changed does what it did, but for a table on which that application
   already asks (a date column, a persistent panel), where the helper now
   answers "no" for the test.

   **The exact text of every replacement is in
   `docs/measurements/CP7/loosening.py`.** It is the form that was proved:
   read it, and make exactly those replacements, each of which matches its
   file once. `python -B docs/measurements/CP7/loosening.py --table` prints
   them. Keep each file's line endings as they are: five of the files end
   their lines with CR LF (`test_leakage.py`, `test_pipeline.py`,
   `test_preprocess.py`, `test_result_schema.py`, `test_type_matrix.py`).
   **No existing test is ever edited so that it fails against the library as
   built:** run the whole suite against the application before you finish and
   see all 1,757 cases pass.

   **(a) `tests/mlp_helpers.py`.** `run_cli` is replaced by the first block
   below, which also adds three functions after it; the second block is
   inserted immediately before `def files_in(folder):`.

   ```python
   def run_cli(args, timeout=900, as_given=False):
       """Run `python -m mlprofessor` in a new process from the repo root.

       D59, D63: when the run stops only because nothing was said about time order or
       about entities, it is run once more with `--no-time-order` and `--no-entities`
       added for whichever was not said, and that run is returned. `as_given=True`
       returns the first run whatever it is.
       """
       env = dict(os.environ)
       env["PYTHONDONTWRITEBYTECODE"] = "1"

       def once(arguments):
           return subprocess.run(
               [PYTHON, "-m", "mlprofessor", *[str(a) for a in arguments]],
               cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout, env=env,
           )

       proc = once(args)
       more = [] if as_given else missing_flags(proc, args)
       return once([*args, *more]) if more else proc


   # ------------------------------------------------- the two statements (D63)

   def missing_statements(code, said):
       """The statements a call left unsaid when it stopped with `code`, as arguments.

       `said` maps an argument's name to what the call passed. Empty unless the run
       stopped because nothing was said about time order or about entities (D63).
       """
       if code == "time_order_undeclared" and said.get("time_col") is None:
           more = {"time_col": False}
           if said.get("group_col") is None:
               more["group_col"] = False
           return more
       if code == "entities_undeclared" and said.get("group_col") is None:
           return {"group_col": False}
       return {}


   def missing_flags(proc, args):
       """The same for a run of the command line: the flags to add, or []."""
       if proc.returncode != 3:
           return []
       words = [str(a) for a in args]

       def said(*flags):
           return any(w in flags or w.startswith(flags[0] + "=") for w in words) or None

       code = proc.stderr.strip().partition("error: ")[2].partition(":")[0]
       more = missing_statements(code, {"time_col": said("--time-col", "--no-time-order"),
                                        "group_col": said("--group-col", "--no-entities")})
       return [flag for name, flag in (("time_col", "--no-time-order"),
                                       ("group_col", "--no-entities")) if name in more]


   def stated_profile(*args, **kw):
       """`mlprofessor.profile(*args, **kw)`, for a test written before every call had to
       say whether the rows are ordered in time and whether rows belong together (D59, D63).

       The call is made as the test wrote it. Only when it stops because a statement
       was not made is it made again, with each missing statement made as "no":
       `time_col=False`, `group_col=False`. Nothing else is changed, and any other
       error is raised as it is.
       """
       import mlprofessor
       from mlprofessor import errors

       try:
           return mlprofessor.profile(*args, **kw)
       except errors.ProfileError as exc:
           more = missing_statements(exc.code, kw)
           if not more:
               raise
       return mlprofessor.profile(*args, **{**kw, **more})
   ```

   ```python
   # D66: `validation.n_rows_with_copies` became `n_rows_same_features`, and the warning
   # `copies_grouped` became `same_features_grouped`. A test written before that reads
   # whichever of the two the application gives (D59).
   SAME_FEATURES_CODES = ("same_features_grouped", "copies_grouped")


   def same_features_rows(validation):
       """`validation.n_rows_same_features`, or `n_rows_with_copies` where it is still named so."""
       if "n_rows_same_features" in validation:
           return validation["n_rows_same_features"]
       return validation["n_rows_with_copies"]


   def count_same_features_warnings(result):
       """How many warnings say that rows with the same features were kept together."""
       return sum(code in SAME_FEATURES_CODES for code, _ in warnings_of(result))
   ```

   **(b) `tests/cp5_helpers.py::raised`** becomes:

   ```python
   def raised(frame, target="y", families=None, steps=None, as_called=False, **kw):
       """The `ProfileError` that `profile()` raises (fast, one core by default).

       Fails if it returns a result; any other exception passes through.

       D59, D63: when the call stops because nothing was said about time order or
       about entities, it is made again with each missing statement made as "no".
       If that run ends in a named error, that error is returned: it is the one a
       test written before D63 is after. If it returns a result, the first error
       is. `as_called=True` returns the error of the call as written, and makes no
       second call.
       """
       import mlprofessor
       from mlprofessor import errors, extensions

       args = dict(h.FAST)
       args.update(kw)
       try:
           with extensions.use(families=families, steps=steps):
               result = mlprofessor.profile(frame, target, **args)
       except errors.ProfileError as exc:
           more = {} if as_called else h.missing_statements(exc.code, args)
           if not more:
               return exc
           asked = exc
       else:
           pytest.fail(f"profile() returned a result; a named error was expected "
                       f"(rows {h.rows_of(result)})")
       try:
           with extensions.use(families=families, steps=steps):
               mlprofessor.profile(frame, target, **{**args, **more})
       except errors.ProfileError as exc:
           return exc
       return asked
   ```

   **(c) Ten helpers and seven tests: one call each of `mlprofessor.profile(...)`
   becomes `h.stated_profile(...)` with the same arguments** (two calls in
   `full_runs`, in `test_full_profile.py` and in `test_seed_range.py`).
   Nothing else on those lines changes.

   | File | Helper or test |
   |---|---|
   | `conftest.py` | `full_runs` (both calls) |
   | `cp3_helpers.py` | `run` |
   | `test_pipeline.py`, `test_uncertainty.py`, `test_time_pipeline.py`, `test_edges.py`, `test_leakage.py` | `_run` |
   | `test_result_schema.py`, `test_model_families.py` | `_profile` |
   | `test_run_hygiene.py` | `_call_profile` |
   | `test_type_matrix.py` | `test_a_result_or_a_named_error` |
   | `test_type_matrix_group.py` | `test_a_result_or_a_named_error_as_the_entity_column` |
   | `test_csv_edges.py` | `test_profile_runs_on_a_file_with_leading_blank_lines` |
   | `test_full_profile.py` | `test_same_result_at_one_and_two_jobs`, `test_n_jobs_minus_one_records_cpu_count` |
   | `test_non_finite.py` | `test_a_target_on_a_very_large_scale_gives_finite_numbers_or_the_named_error` |
   | `test_seed_range.py` | `test_the_largest_seed_runs_and_the_next_one_is_refused` (the two runs that finish) |

   The two type-matrix tests would pass unchanged, since each accepts any
   named error. They are changed all the same: unchanged, every one of their
   751 cases that reaches step 5 would end in the new stop and prove nothing
   of what follows it.

   **(d) Five tests pin the stop itself and must not have it answered for
   them.** Each gains one keyword on one call, and nothing else.

   | Test | The call | Gains |
   |---|---|---|
   | `test_time_order_stated.py::test_a_date_column_and_no_word_about_time_stops_the_run` | `g.raised(_table(as_text), ..., out_dir=str(out))` | `as_called=True` |
   | `test_time_order_stated.py::test_the_date_error_comes_before_the_copies_error` | its first `g.raised(frame, families=...)`, the one with no `time_col` | `as_called=True` |
   | `test_time_order_stated.py::test_the_command_line_states_it` | `h.run_cli([*common, "--out", out])` into `plain` | `as_given=True` |
   | `test_panel_stop.py::test_a_persistent_panel_with_no_entity_column_stops` | `g.raised(_persistent(), ..., time_col="date", out_dir=str(out))` | `as_called=True` |
   | `test_panel_stop.py::test_the_command_line_says_there_are_no_entities` | `h.run_cli([*common, "--out", out])` into `asked` | `as_given=True` |

   **(e) Five tests pin "nothing said is recorded as null", which D63 removes:
   a finished run records a name or `false`.** The one assertion on the unsaid
   statement accepts null, as today, or `false`, the statement the helper made.

   | Test | Before | After |
   |---|---|---|
   | `test_groups_random.py::test_without_an_entity_column_the_keys_are_null` | `assert doc["input"]["group_col"] is None` | `assert doc["input"]["group_col"] is None or doc["input"]["group_col"] is False` |
   | `test_groups_random.py::test_group_col_false_is_recorded_and_changes_nothing_else` | `assert plain["input"]["group_col"] is None` | `assert plain["input"]["group_col"] is None or plain["input"]["group_col"] is False` |
   | `test_panel_stop.py::test_a_panel_that_does_not_persist_runs_without_a_word` | `assert result.to_dict()["input"]["group_col"] is None` | `said = result.to_dict()["input"]["group_col"]` and `assert said is None or said is False` |
   | `test_result_schema.py::test_profile_json_has_every_key_with_its_type` | `assert inp["target"] == "y" and inp["time_col"] is None and inp["task"] == "auto"` | `assert inp["target"] == "y" and inp["task"] == "auto"` and `assert inp["time_col"] is None or inp["time_col"] is False` |
   | `test_time_order_stated.py::test_the_statement_on_a_table_with_no_dates_changes_only_its_record` | `assert plain["input"]["time_col"] is None` | `assert plain["input"]["time_col"] is None or plain["input"]["time_col"] is False` |

   **(f) Seven tests pin the two names D66 changed.** Each reads the count
   through `h.same_features_rows(...)` and the warning through
   `h.count_same_features_warnings(...)` or `h.SAME_FEATURES_CODES`, which
   accept either name.

   | Test | Before | After |
   |---|---|---|
   | `test_duplicates_group.py::test_each_set_of_copies_stays_on_one_side_and_in_one_fold` | `v["n_rows_with_copies"] == 600`; `g.count_code(result, "copies_grouped") == 1` | `h.same_features_rows(v) == 600`; `h.count_same_features_warnings(result) == 1` |
   | `test_duplicates_group.py::test_a_table_with_no_copies_is_split_as_by_default` | `v["n_rows_with_copies"] == 0`; `g.count_code(grouped, "copies_grouped") == 0`; `plain.to_dict()["validation"]["n_rows_with_copies"] is None` | `h.same_features_rows(v) == 0`; `h.count_same_features_warnings(grouped) == 0`; `h.same_features_rows(plain.to_dict()["validation"]) is None` |
   | `test_duplicates_group.py::test_a_few_copies_are_grouped_and_counted` | `v["n_rows_with_copies"] == 14` | `h.same_features_rows(v) == 14` |
   | `test_duplicates_group.py::test_copies_and_declared_entities_are_kept_together_both` | `v["n_rows_with_copies"] == 100` | `h.same_features_rows(v) == 100` |
   | `test_duplicates_group.py::test_under_time_order_it_changes_nothing` | `grouped["validation"]["n_rows_with_copies"] is None` | `h.same_features_rows(grouped["validation"]) is None` |
   | `test_duplicates_group.py::test_the_command_line_takes_group` | `"copies_grouped" in proc.stdout` | `any(code in proc.stdout for code in h.SAME_FEATURES_CODES)` |
   | `test_groups_random.py::test_without_an_entity_column_the_keys_are_null` | the loop over four keys, the last `"n_rows_with_copies"` | the loop over the first three, then `assert h.same_features_rows(v) is None` |

   **What each loosened test no longer pins, and the new test that pins the
   new behaviour exactly:**

   | No longer pinned | By | Pinned again by |
   |---|---|---|
   | a plain call runs, and records the unsaid statement as null | the five tests of (e) | `test_statements.py::test_nothing_said_about_time_order_stops_whatever_the_table_holds`, `::test_nothing_said_about_entities_stops_whatever_the_table_holds`, `::test_the_four_combinations_and_what_is_recorded` |
   | a panel is asked only when it persists, and a classification never (the conditional stop of D58) | `test_panel_stop.py::test_a_panel_that_does_not_persist_runs_without_a_word`, `::test_a_classification_panel_is_not_asked`, which pass through `cp3_helpers.run` | `test_statements.py::test_nothing_said_about_entities_stops_whatever_the_table_holds`, in its four cases |
   | a table is asked about time order only when it holds a date column | the nine tests the fifth pass loosened, whose own fall-backs no longer fire because the helper answers first; they are left as they are | `test_statements.py::test_nothing_said_about_time_order_stops_whatever_the_table_holds` |
   | the command line with one pair missing runs | every test that calls `h.run_cli` | `test_statements.py::test_the_command_line_asks_for_each_missing_pair` |
   | the names `n_rows_with_copies` and `copies_grouped` | the seven tests of (f) | `test_duplicates_features.py::test_the_count_and_the_warning_have_their_new_names`, `::test_the_command_line_uses_the_new_names` |

   The five tests of (d) go on pinning the stop for their own tables. They
   also go on failing on the broken copies F7d and M19, which was checked
   with the loosening in place.

   **Tests that are NOT loosened, and stay as they are:**
   `test_readme_examples.py::test_every_readme_command_runs` runs the README's
   commands through `h.run_cli`. It is green today, green after T49 builds
   the stop (the helper adds the flag the README does not yet show), and
   green after T53 rewrites the commands. The new README test runs each
   command as written. And `test_duplicates.py::test_time_order_has_no_rule`
   stays locked, as D60 has it.

4. **You write six new test files and the helper file, and nothing else.** The
   files are listed below, each with the tests it must hold, **under the names
   given**: the lists of rule 5 and the mutation check go by name. Split a
   named test into parameter cases as the text says. If you add a test of
   your own, say so in your final report, with the list of rule 5 it belongs
   to.

5. **Which tests fail today and which pass today is fixed, by name.** The
   lists are at the end of this section. A test on the red list fails against
   the application as it stands, in every one of its parameter cases unless
   the list says otherwise. A test on the green list passes today in every
   case. Run the new files before you finish and compare with the lists. **Do
   not weaken a guard to make it fail, and do not weaken a red test to make
   it pass.** If a test on the green list fails, or one on the red list
   passes, and you believe your test is what the spec says, stop and report
   it with the output.

6. **Every new variant must fail a test, and `verify/mutation_check.py` is how
   that will be judged.** It runs the whole suite against a copy of the
   reference implementation and again with the reference broken in one way at
   a time: twenty-one forms the gate also catches (F1 to F14, F6b, F6c, F6d,
   F7b, F7c, F7d, F8b) and thirty smaller faults (M1 to M30; spec 8 and 8.1).
   It passes only if the good reference fails none of your tests (the README
   tests apart) and every variant fails at least one. Twelve are new, and
   each has a test of this pass named for it:

   | Variant | The test that must fail on it |
   |---|---|
   | F11 | `test_statements.py::test_nothing_said_about_time_order_stops_whatever_the_table_holds` |
   | F12 | `test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_together_whatever_their_target` |
   | F13 | `test_grouped_holdout.py::test_a_large_entity_is_never_held_out_when_smaller_ones_can_fill_a_quarter` |
   | F14 | `test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date` |
   | M23 | `test_guard_review5.py::test_m23_the_panel_warning_starts_just_above_one_half` |
   | M24 | `test_statements.py::test_one_message_asks_for_everything_that_is_missing` |
   | M25 | `test_duplicates_features.py::test_the_count_is_of_rows_that_agree_in_every_feature` |
   | M26 | `test_grouped_holdout.py::test_what_the_holdout_can_spare_goes_back` |
   | M27 | `test_grouped_holdout.py::test_a_large_unit_kept_in_training_is_named` |
   | M28 | `test_date_numbers.py::test_eight_digit_text_is_accepted_as_the_time_column` |
   | M29 | `test_date_numbers.py::test_the_date_rule_comes_before_the_id_like_rule` |
   | M30 | `test_guard_review5.py::test_m30_a_copy_at_step_8a_still_needs_the_same_target` |

   You may run it yourself, from the repo root, with the project interpreter:
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/mutation_check.py --jobs 8`.
   It writes nothing inside the repo. Do not read the reference
   implementation to tune a test to it. The README test of this pass goes in
   a file named `tests/test_readme_statements.py`: files named
   `tests/test_readme*.py` are shown on a line of their own and not counted.

7. **A test of this pass says both things in every call, or is about the stop
   itself.** The helpers of the earlier passes now answer for a test that
   says nothing (rule 3). A new test that used them to see a stop would see
   none. So the new files call through `cp7_helpers.run`, `raised` and
   `run_cli`, which add no statement, and pass `time_col` and `group_col` in
   every call that is meant to finish (`**s.SAID` for "no" to both).

8. **Which units are held out is open (spec step 7 and 11).** Two correct
   implementations hold out different entities from the same seed. So no test
   asserts a particular holdout row of a grouped split. The tests assert
   property 3 of spec step 7 through `cp7_helpers.assert_property_three`,
   which works from the result alone, and which units are named by the two
   warnings, which does not depend on the seed.

9. **The rules of the earlier passes still hold**: the documented surface only
   (spec 2.1 to 2.7); pin what the spec pins and nothing else; recompute, do
   not trust; assert on values as the application produced them; **no skipped
   or expected-to-fail test**; nothing written outside `tmp_path`; the
   application imported inside test functions; the command line run one
   process at a time. What a message must hold is pinned in spec 11 and 5.4
   and nowhere else. Tiny tables: none of this pass needs more than 700 rows.
   Use a recording family or a constant family through the seam wherever a
   run only has to finish.

## Files of the sixth pass

Each file belongs to exactly one task of `plan/tasks.json`.

| File | Task | About | Today |
|---|---|---|---|
| `tests/test_guard_review5.py` | T48 | the cut-off of the panel warning from both sides; a copy still needs the same target (D23, D49, D62) | green |
| `tests/test_statements.py` | T49 | both statements on every call (D63) | red, four tests green |
| `tests/test_duplicates_features.py` | T50 | `duplicates="group"` on the features alone; the two names; the copies advice (D62, D66) | red, one test green |
| `tests/test_grouped_holdout.py` | T51 | property 3 of step 7; the two warnings for a large unit (D64, D66) | red, one test green |
| `tests/test_date_numbers.py` | T52 | dates written as whole numbers (D66) | red, two tests green |
| `tests/test_readme_statements.py` | T53 | the README on this pass | red, one test green |
| `tests/cp7_helpers.py` | none | shared helpers of the sixth pass; holds no test | |
| the 22 files of rule 3 | (earlier tasks) | 39 replacements, as rule 3 says; nothing else | green before and after |

### `tests/cp7_helpers.py`

Import the earlier helper files and build on them; do not copy from them.

- **`SAID`**: `{"time_col": False, "group_col": False}`.
- **`TIME_TEXTS`** and **`ENTITY_TEXTS`**: the four texts each message must
  hold (spec 11): `time_col`, `--time-col`, `time_col=False`,
  `--no-time-order`; and `group_col`, `--group-col`, `group_col=False`,
  `--no-entities`.
- **`COPIES_ADVICE`**, **`LARGE_KEPT`**, **`LARGE_IS_HOLDOUT`**: the three
  fixed passages of spec 5.4, exactly.
- **`KEPT`**, **`IS_HOLDOUT`**: the two warning codes,
  `large_group_kept_in_training` and `large_group_is_holdout`.
- **`run(frame, target="y", families=None, steps=None, seed=0, **kw)`**:
  `profile()` exactly as called, fast profile and one core by default, inside
  `extensions.use`. It adds no statement.
- **`raised(frame, ...)`**: `cp5_helpers.raised(..., as_called=True)`.
- **`run_cli(args)`**: `mlp_helpers.run_cli(args, as_given=True)`.
- **`sized_entities(sizes, seed=0, labels=None)`**: one entity for each entry
  of `sizes`, named `e000`, `e001`, ...: a column `who`, two noise measures
  `m1` and `m2`, and `y`: normal noise, or with `labels` one 0 or 1 for each
  entity. Rows in an order shuffled from the seed. No two rows agree.
- **`few_valued(n=600, levels=(4, 3), seed=0)`**: text columns `c1`, `c2`,
  ... of that many levels each, drawn at random, and a 0/1 target `y` that
  depends on none of them.
- **`day_numbers(n_days=200, per=3, seed=0, start="2019-01-01")`**: `per`
  rows a day: `day` as an `int64` written `YYYYMMDD`, a noise column `x` and
  a 0/1 target `y`, rows shuffled.
- **`quarter_unit(sizes)`**: the size of the smallest unit such that the
  units no larger than it hold, together, at least a quarter of the rows.
- **`assert_property_three(holdout_rows, units, where="")`**: `units` holds
  the unit of every row split. Asserts: every unit of the holdout is whole;
  the holdout holds at least a quarter of the rows; without any one of its
  units it would hold less than a quarter; and it holds less than a quarter
  plus `quarter_unit`.
- `rec()`, `codes(result)` and `messages(result, code)`: a recording family
  on a fresh log; the warning codes of a result; the messages of one code.

## The guards (task T48; all green today)

### `tests/test_guard_review5.py`

Spec: step 8a, step 11, 4.3. Both pass against the application as it stands.

- `test_m23_the_panel_warning_starts_just_above_one_half`, two cases.
  **M23.** A panel of three rows on each of 80 numeric timestamps: the level
  of a timestamp follows the one before with weight `phi`, times 3, and each
  row adds normal noise of 0.2. The holdout is the latest 20 timestamps, so
  60 are training timestamps. The test computes the lag-1 autocorrelation of
  the per-timestamp means of those 60 **by both usual estimators** (the sum
  of products of successive deviations over the sum of squares; and the
  correlation of each mean with the one before) and asserts, as its own
  precondition, that both lie in the range given. `just_above`: `phi` 0.57
  from seed 41, both in 0.54 to 0.58: with `time_col="t", group_col=False`
  exactly one `autocorrelated_panel_target` warning, on `y`.
  `just_below`: `phi` 0.45 from seed 59, both in 0.42 to 0.46: none. In both
  the scheme is `time_ordered`, there are 60 training timestamps and the one
  baseline is `training_mean`. (If your generator gives other values from
  those seeds, choose seeds that put both estimators in the ranges; the
  ranges are what is pinned.)
- `test_m30_a_copy_at_step_8a_still_needs_the_same_target`. **M30.**
  `p.noise_frame(400)`, plus its first 120 rows again with the other class,
  plus rows 120 to 149 again as they are, shuffled. With `duplicates="keep"`
  and both statements, through `p.run_kept`:
  `validation.n_holdout_copies` equals `p.copies_by_hand(result, log)`, the
  share follows, and that count is smaller than the number of holdout rows
  that agree with a training row in the four features alone, which the test
  counts for itself.

## A: both statements on every call (task T49)

### `tests/test_statements.py`

Spec: 2.1, 2.2, 2.4, step 5, 5.2, 11.

A helper `_stopped(frame, tmp_path, **kw)` calls `s.raised` with a recording
family and a recording step and an `out_dir`, asserts that no file was
written and that the log holds no event, and returns the error.

- `test_nothing_said_about_time_order_stops_whatever_the_table_holds`, four
  cases. **F11.** No table holds a date. `classification`:
  `h.clf_frame(200)`, nothing said. `regression`: `h.reg_frame(200)`.
  `entities_named`: `g.entity_frame(60, sizes=5)` with `group_col="who"`.
  `no_entities_said`: a classification with `group_col=False`. Each: the code
  `time_order_undeclared`, an instance of `TimeOrderUndeclaredError`, the
  four time texts in the message, nothing written, nothing fitted.
- `test_nothing_said_about_entities_stops_whatever_the_table_holds`, four
  cases. **F11, M19.** `no_time_order`: a classification with
  `time_col=False`. `one_series`: `h.time_frame(200)` with
  `time_col="date"`. `classification`: `h.time_frame(200,
  task="classification")` with `time_col="date"`.
  `a_panel_that_does_not_persist`: `h.panel_frame(60, 5, seed=13,
  persistent=False)` with `time_col="date"`. Each: `entities_undeclared`, an
  instance of `EntitiesUndeclaredError`, the four entity texts, nothing
  written, nothing fitted.
- `test_one_message_asks_for_everything_that_is_missing`. **M24.** Nothing
  said on a table with no date: the code is `time_order_undeclared` and the
  message holds all eight texts.
- `test_none_passed_explicitly_is_not_said`. `time_col=None, group_col=None`
  stops with `time_order_undeclared`; `time_col=False, group_col=None` with
  `entities_undeclared`.
- `test_the_entities_message_says_that_timestamps_repeat`.
  `h.panel_frame(60, 5, persistent=False)` with four rows' targets set
  missing, `time_col="date"`: `entities_undeclared`, and the whole numbers in
  the message include 296 and 60.
- `test_an_error_of_steps_one_to_four_comes_first`, six cases. **Green
  today.** With a statement unsaid, each of these is raised and not the
  stop: `seed=-1` (`input_error`); a target that is no column
  (`column_not_found`); 60 rows (`too_few_rows`); a target of five distinct
  whole numbers (`task_ambiguous`); a day-first text time column
  (`time_column_invalid`); an empty entity cell (`group_column_invalid`).
- `test_the_stop_comes_before_every_later_error`, five cases. A numeric
  column named `colour=red` beside a `colour` with the level `red`; every row
  twice; a family that always raises; a table whose only other column is
  constant; and five timestamps of 60 rows under `time_col="t"`. With the
  statement unsaid the code is `time_order_undeclared` (the last:
  `entities_undeclared`). With both made, the error is another named error.
- `test_the_four_combinations_and_what_is_recorded`. **Green today.**
  `(False, False)`: `random`, both recorded as `False`, both `validation`
  keys null. `(False, "who")`: `grouped`. `("date", False)` on a random
  walk: `time_ordered`, baselines `training_mean` and `previous_value`,
  `n_holdout_without_previous` null. `("t", "who")` on `g.walks(6, 60)`:
  `time_ordered`, both baselines, `n_holdout_without_previous` 0.
- `test_the_signature_keeps_its_eleven_parameters_and_their_defaults`.
  **Green today.** The eleven names in order; `time_col` and `group_col`
  default to `None`.
- `test_the_step_functions_take_what_they_are_given`. **Green today.**
  `stages.column_roles(frame, "y")` on a table with a date column returns
  its records, `stages.usable_rows` and `stages.split_random` return theirs:
  none asks for a statement.
- `test_the_command_line_asks_for_each_missing_pair`. Through `s.run_cli`,
  on a classification with no date. Nothing given: exit 3, one line on
  standard error starting `error: time_order_undeclared:`, holding
  `--time-col`, `--no-time-order`, `--group-col` and `--no-entities`, no
  file. `--no-time-order` alone: exit 3, `error: entities_undeclared:` with
  `--group-col` and `--no-entities`. `--no-entities` alone: exit 3,
  `error: time_order_undeclared:` with `--time-col` and `--no-time-order`.
  Both: exit 0, and both recorded as `false`.

## B: `duplicates="group"` on the features alone (task T50)

### `tests/test_duplicates_features.py`

Spec: 2.1, 2.5, steps 7 and 8a, 5.2, 5.4, 11.

- `test_rows_that_agree_in_every_feature_stay_together_whatever_their_target`.
  **F12.** `s.few_valued(600, (4, 3))`, `duplicates="group"`, both
  statements, a recording family and step. No pair of the two feature values
  is on both sides of the holdout, nor in the fit rows and the predicted rows
  of any fit (`g.fits(log)`). The scheme is `grouped`; the units on the two
  sides are the test's own counts and add up to 12;
  `n_rows_same_features` is 600; `n_holdout_copies` is 0.
- `test_the_count_is_of_rows_that_agree_in_every_feature`. **M25.**
  `p.noise_frame(300)` plus its first 40 rows again with the other class,
  shuffled: `n_rows_same_features` is 80, the units add up to 300,
  `n_holdout_copies` is 0, exactly one `same_features_grouped` warning, and
  no four feature values are on both sides.
- `test_the_count_and_the_warning_have_their_new_names`. Every row twice:
  `n_rows_same_features` is 600, the key `n_rows_with_copies` is absent, one
  `same_features_grouped` and no `copies_grouped`. On a table where no two
  rows agree: `random`, 0, no warning. With `duplicates` `"error"` or
  `"keep"` the key is present and null. Under a time column it is null.
- `test_a_set_that_spans_two_entities_joins_them`. `g.entity_frame(60,
  sizes=5)`; for each of 20 pairs of entities, one more row for the second
  with the features of a row of the first and the other class. With
  `duplicates="group"` and `group_col="who"`: no entity on both sides, each
  pair on one side, `n_rows_same_features` 40, 40 units in all.
- `test_the_copies_stop_says_when_each_answer_fits`. `s.few_valued(600)`
  with the default `duplicates`: `HoldoutCopiesError`, the message holds
  `s.COPIES_ADVICE` exactly, and `duplicates='keep'` comes before
  `duplicates='group'` in it.
- `test_too_few_combinations_cannot_be_split_and_a_few_give_no_interval`.
  **Green today.** Four combinations: `TooFewGroupsError`. Twelve: no
  interval, a reason that holds `30`, no p-value with the words
  `independent rows`, a `no_interval` warning.
- `test_the_command_line_uses_the_new_names`. Every row twice, `--duplicates
  group` with both flags: exit 0, `n_rows_same_features` 600 in the file,
  `same_features_grouped` in the summary and `copies_grouped` not.

## C: the grouped holdout (task T51)

### `tests/test_grouped_holdout.py`

Spec: 2.5, 2.7, step 7, 5.4, 11. Property 3 is checked through
`stages.split_random(n, classes, seed, groups=...)` for seeds 0 to 9, which
fits nothing, and the warnings through `profile()`.

- `test_a_large_entity_is_never_held_out_when_smaller_ones_can_fill_a_quarter`,
  three cases. **F13.** 400 rows: one entity of 100, 120 or 240 rows and the
  rest of five each. For each seed: property 3, the large entity not in the
  holdout, and 100 to 104 holdout rows. Then one `profile()` run with
  `seed=3`: its holdout rows are those of `split_random` at that seed, and
  `holdout_share` is 0.25.
- `test_the_holdout_holds_a_quarter_and_needs_every_unit_it_has`, three
  cases. `long_tailed`: sizes drawn from a long-tailed law until they hold
  600 rows. `a_handful_of_large`: eight entities of 80, 70, 60, 50, 50, 40,
  30 and 20 rows. `two_classes`: one entity of 90 rows and ten of five with
  the label 1, twenty of five with the label 0, `classes` passed. For each
  seed: property 3, and in the third case for the units of each class apart.
- `test_what_the_holdout_can_spare_goes_back`. **M26.** Entities of 160,
  160 and five of 16 rows. For each seed the holdout is exactly 160 rows of
  one of the first two, and six entities are left for training.
- `test_a_large_unit_kept_in_training_is_named`. **M27.** One entity of 240
  rows and 32 of five: exactly one `large_group_kept_in_training`, with
  column null, whose message holds the number 240, a number within 0.1 of
  60.0 and `s.LARGE_KEPT`; no `large_group_is_holdout`; the large entity not
  in the holdout; `holdout_share` 0.25.
- `test_a_unit_too_large_for_a_sound_split_is_the_holdout_and_is_named`.
  One entity of 360 rows and eight of five: the holdout is that entity and
  no other, `holdout_share` is 0.9, 8 and 1 units on the two sides, exactly
  one `large_group_is_holdout` whose message holds 360, a number within 0.1
  of 90.0 and `s.LARGE_IS_HOLDOUT`; no `large_group_kept_in_training`; no
  interval, with the `no_interval` warning.
- `test_one_warning_for_each_large_unit_and_a_share_of_its_own_class`.
  Entities of 160, 160 and five of 16: one of each warning, each with 160
  and 40.0. Then a classification of 260 rows: one entity of 60 rows and
  eight of five with the label 1, 32 of five with the label 0: one
  `large_group_kept_in_training` with 60 and 60.0 (60 rows is under a
  quarter of the table and 60% of the rows of the units of class 1), and no
  `large_group_is_holdout`.
- `test_no_large_unit_no_warning`. **Green today.** Eighty entities of five
  rows: neither warning, and property 3 on the run's own holdout.
- `test_the_rule_holds_for_rows_that_agree_in_every_feature`. A regression
  of 400 rows with two text features: 240 rows of one combination and 16
  combinations of ten rows. With `duplicates="group"`: property 3 with the
  combinations as units, the large one not in the holdout, one
  `large_group_kept_in_training` with 240 and 60.0, `n_rows_same_features`
  400.

## D: dates written as whole numbers (task T52)

### `tests/test_date_numbers.py`

Spec: 2.7, steps 3, 5 and 8a, 4.5, 11.

- `test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date`,
  six cases. **F14.** `s.day_numbers()` with `day` as: `int64`; floats with
  every seventeenth cell empty; text, every ninth value with a space each
  side; nullable `Int64` with empty cells; Python integers with `None` in an
  object column; `Decimal` values in an object column. With both statements
  and a recording step: the role `excluded`, the reason `datetime`,
  `n_coerced`, `n_infinite` and `n_imputed` 0, exactly one
  `date_column_undeclared` warning and it names `day`, the scheme `random`,
  and no step is handed a column `day`.
- `test_the_ends_of_the_range_and_the_calendar_are_dates`, four cases,
  through `stages.column_roles` on 200 rows that repeat the values:
  19000101 and 21001231; 20240229 and 20240301; 20000229 and 20000301;
  19991231 and 20000101. Each: `excluded`, `datetime`.
- `test_values_that_are_no_dates_leave_the_column_what_it_was`, fourteen
  cases. **Green today.** One value in the column that is no date, and the
  column is a numeric feature as before: 18991231; 21010101; 20230229;
  19000229; 20231301; 20230100; 20230431; a fraction; an infinite cell;
  day-first numbers; `YYYYMM`; text of nine digits; text of seven. And one
  date throughout is `constant`.
- `test_it_is_judged_on_the_usable_rows`. A value that is no date in a row
  with no target: the column is still `datetime`.
- `test_the_date_rule_comes_before_the_id_like_rule`, two cases. **M29.** One
  row a day in date order, as `int64` and as text: `datetime`, not
  `id_like`, and named by the warning.
- `test_it_is_named_when_nothing_is_said_about_time_order`. Nothing said:
  `time_order_undeclared`, and the message holds `'day'`.
- `test_a_date_number_column_is_a_time_column_like_any_number`, two cases.
  **Green today.** `time_col="day"` on `int64` and on nullable `Int64`: the
  holdout is the rows of the latest whole days that hold a quarter, worked
  out by the test; `time_ordered`, `numeric`; `train_end`, `holdout_start`
  and `holdout_end` are the dates as numbers; the role is `time`.
- `test_eight_digit_text_is_accepted_as_the_time_column`. **M28.** The same
  for text of eight digits; `stages.read_time` returns `float64` keys equal
  to the numbers and the form `numeric`. With one value changed to
  `20231301`: `TimeColumnError`.
- `test_under_a_time_column_another_date_number_column_is_excluded_without_a_word`.
  `h.time_frame(200)` with a second column of dates as whole numbers, under
  `time_col="date"`: `excluded`, `datetime`, and no
  `date_column_undeclared`.
- `test_the_copies_stop_names_a_date_number_column`. Every row twice, with a
  column of dates as whole numbers drawn at random, and both statements:
  `HoldoutCopiesError`, whose message holds the column's name and
  `time_col`.

## The README (task T53)

### `tests/test_readme_statements.py`

Spec: 1. It judges `README.md`, which the builder owns; do not edit it. It
imports `test_readme_examples` for the three functions that read the README's
commands. Pin no wording beyond these.

- `test_every_readme_command_states_both_and_runs_as_written`. Every
  `-m mlprofessor profile` command holds `--time-col` or `--no-time-order`,
  and `--group-col` or `--no-entities`, and exits 0 through `s.run_cli`,
  forced to the fast profile and one core as the earlier test forces it.
- `test_every_python_example_states_both`. **Green today.** Every
  `profile(...)` call in a fenced block passes `time_col=` and `group_col=`.
- `test_the_readme_names_what_this_pass_added`. The text holds `YYYYMMDD`,
  `n_rows_same_features`, `same_features_grouped`,
  `large_group_kept_in_training` and `large_group_is_holdout`, and holds
  neither `n_rows_with_copies` nor `copies_grouped`.

## What must fail today, and what must pass today

Checked on 2026-11-22 with prototypes of every test below and of every
replacement of rule 3, run against the application as it stands, against the
reference implementation, and against the reference carrying each of F11 to
F14, M23 to M30, F7d and M19.

**Red today.** They fail against the application until the build stage has
done D62 to D66, and pass against the reference.

| File | Tests |
|---|---|
| `test_statements.py` | all but the four on the green list |
| `test_duplicates_features.py` | all but the one on the green list |
| `test_grouped_holdout.py` | all but the one on the green list |
| `test_date_numbers.py` | all but the two on the green list |
| `test_readme_statements.py` | `test_every_readme_command_states_both_and_runs_as_written`, `test_the_readme_names_what_this_pass_added`; red against the reference too, until T53 is built: they judge `README.md` |

**Green today.** They pass against the application and against the
reference, and must go on passing through every task of the build.

| File | Test | Fails on |
|---|---|---|
| `test_guard_review5.py` | `test_m23_the_panel_warning_starts_just_above_one_half`, two cases | M23 |
| `test_guard_review5.py` | `test_m30_a_copy_at_step_8a_still_needs_the_same_target` | M30 |
| `test_statements.py` | `test_an_error_of_steps_one_to_four_comes_first` (six cases), `test_the_four_combinations_and_what_is_recorded`, `test_the_signature_keeps_its_eleven_parameters_and_their_defaults`, `test_the_step_functions_take_what_they_are_given` | |
| `test_duplicates_features.py` | `test_too_few_combinations_cannot_be_split_and_a_few_give_no_interval` | |
| `test_grouped_holdout.py` | `test_no_large_unit_no_warning` | |
| `test_date_numbers.py` | `test_values_that_are_no_dates_leave_the_column_what_it_was` (fourteen cases), `test_a_date_number_column_is_a_time_column_like_any_number` (two cases) | |
| `test_readme_statements.py` | `test_every_python_example_states_both` | |
| the 22 files of rule 3 | every test | not guards: each passes before the change and after it |

Measured with the prototypes: 84 cases in the six new files. Against the
application as it stands, 31 pass and 53 fail. Against the reference, 82 pass
and two fail, the README's. With the 39 replacements of rule 3 in place, all
1,757 existing cases pass against the application as it stands, and all
1,757 against the reference.

**What the mutation check must then print**, run after your stage and before
the build: the good reference fails 0 of 1828 (the 1,747 counted today and
81 of this pass: the 84 cases of the six new files less the three README
tests); each of the fifty-one variants fails at least one test, and each of
the twelve new ones the test rule 6 names for it; on the README line, 2
failing of 13, the two red tests of `test_readme_statements.py`.
**Before your stage it ends in `FAIL` for two reasons, both expected:** the
good reference fails 274 cases in 36 files, every one a case that makes a
call with a statement unsaid, itself or through a fixture it shares (the
reference has the new behaviour and the tests are not yet loosened;
`docs/measurements/CP7/unloosened_failures.py` checks each); and it prints
`NOT CAUGHT` for 35 of the 51 variants (F1, F2, F4, F6c, F9, F10, F12 to
F14, M1 to M8, M10 to M13, M15, M17, M18 and M20 to M30): eleven of the
twelve new ones have no test yet (F11 already fails five existing tests),
and the others are caught only by tests that the good reference itself fails
until they are loosened. Your stage closes both. On the README line it shows
one failure until your stage,
`test_readme_examples.py::test_every_readme_command_runs`,
because the reference stops on the README's own commands and `h.run_cli` does
not yet answer for them; after your stage that test passes and the two red
tests of `test_readme_statements.py` fail, until T53.

## Not tested in the sixth pass

A copy of a row that differs from it in any column, and a broken copy for
each of the seven gate checks never seen failing: pass 7 (D60, D64). The
wording of any message beyond what spec 11 and 5.4 pin. Which units are held
out and which fold each is in (rule 8). Which of the two estimators of the
autocorrelation the panel warning uses: the guard holds either. A check that
the levels of a column are series of their own: not built (D66). Day counts
and `YYYYMM` as dates: not seen, by decision (D66). How `--help` words that a
pair is needed. Run time at 100,000 rows.

# Seventh pass (CP8): a date column with a sentinel, `"group"` after a removed leak, the fast cap and a large entity, and the judge's two other tools

Decisions D69 to D73. The spec sections are named beside each file. **No test
of this pass is about anything the judge's tools list as open:** one repeated
row under time order with no entities, a copy of a regression target that the
leak screen misses, one extreme cell, an identifier under shuffled rows, the
order of rows within a fit. Those are pass 8 (D73), and each will change a
pinned rule.

## Ground rules of the seventh pass

1. **The application already exists, and this is the seventh pass.**
   `mlprofessor/` is built and passes all 1,841 cases of the first six
   passes. Your role prompt says you write the suite before the application
   exists and that every test must fail now. In this pass that is not so:
   most of your tests must fail today, the guards and some other tests must
   pass today, and every existing test must pass today and after the build.
   Rule 5 says which.

2. **The first six passes are final. Do not edit them, with the one exception
   of rule 3, which D59 allows and which is given there exactly.** That means
   the 58 existing test files, `tests/conftest.py`, `tests/mlp_helpers.py`,
   `tests/cp3_helpers.py` to `tests/cp7_helpers.py` and everything under
   `tests/fixtures/`. New shared helpers go in one new file,
   `tests/cp8_helpers.py`. The owner's lead reads the diff of every existing
   file against the committed one afterwards, and any change other than that
   of rule 3 sends the whole stage back.

3. **THE LOOSENING (D59). One replacement in one existing file, and no other
   change to it or to any other.**

   D69 reverses what one existing test holds. Rule 2a of step 5 used to ask that
   every value of the column be a date, and
   `tests/test_date_numbers.py::test_values_that_are_no_dates_leave_the_column_what_it_was`
   holds fourteen columns to that. Eleven of them are 200 days with one value
   that is no date, which the rule of D72 calls a date column. For those eleven
   the one assertion accepts the column as built or as decided, so the test
   passes against the application as it stands and after T55 is built. The other
   three cases stay exact, and nothing else in the file changes. D65 is not
   needed: no helper changes.

   **The exact text is in `docs/measurements/CP8/loosening.py`**
   (`python -B docs/measurements/CP8/loosening.py --table` prints it). It matches
   the file once. Keep the file's line endings as they are.

   Before:

   ```python
       record = _roles(_no_date_case(case))["day"]
       if case == "one_date_throughout":
           assert record["role"] == "excluded" and record["reason"] == "constant", record
       else:
           assert record["role"] == "feature" and record["kind"] == "numeric", record
   ```

   After:

   ```python
       record = _roles(_no_date_case(case))["day"]
       if case == "one_date_throughout":
           assert record["role"] == "excluded" and record["reason"] == "constant", record
       elif case in ("day_first", "yyyymm"):
           assert record["role"] == "feature" and record["kind"] == "numeric", record
       else:
           # D59, D69: 200 days and one value that is no date. A number as built, and a
           # date column once D69 is built; `test_date_sentinels.py` holds the second.
           as_built = record["role"] == "feature" and record["kind"] == "numeric"
           as_decided = record["role"] == "excluded" and record["reason"] == "datetime"
           assert as_built or as_decided, record
   ```

   | No longer pinned | By | Pinned again by |
   |---|---|---|
   | one value that is no date leaves a column of dates a numeric feature (the cases `before_1900`, `after_2100`, `29_feb_2023`, `29_feb_1900`, `month_13`, `day_0`, `31_april`, `a_fraction`, `an_infinite_cell`, `nine_digit_text`, `seven_digit_text`) | the one test above | `test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column`, which holds the same eleven columns to the opposite, and `::test_too_many_values_that_are_no_dates_leave_the_column_a_number`, which holds where the rule now ends |

   The loosened test still fails on nothing it failed on before: F14 and M29
   leave those eleven columns numeric features, as they did.

   **No existing test is ever edited so that it fails against the library as
   built:** run the whole suite against the application before you finish and
   see all 1,841 cases pass.

4. **You write seven new test files and the helper file, and nothing else.**
   The files are listed below, each with the tests it must hold, **under the
   names given**: the lists of rule 5 and the mutation check go by name. Split
   a named test into parameter cases as the text says. If you add a test of
   your own, say so in your final report, with the list of rule 5 it belongs
   to.

5. **Which tests fail today and which pass today is fixed, by name.** The
   lists are at the end of this section. A test on the red list fails against
   the application as it stands, in every one of its parameter cases unless
   the list says otherwise. A test on the green list passes today in every
   case. Run the new files before you finish and compare with the lists. **Do
   not weaken a guard to make it fail, and do not weaken a red test to make
   it pass.** If a test on the green list fails, or one on the red list
   passes, and you believe your test is what the spec says, stop and report
   it with the output.

6. **Every new variant must fail a test, and `verify/mutation_check.py` is how
   that will be judged.** It runs the whole suite against a copy of the
   reference implementation and again with the reference broken in one way at
   a time: thirty-one forms the gate also catches (F1 to F22, F6b, F6c, F6d,
   F7b, F7c, F7d, F8b, F12b, F14b) and thirty-nine smaller faults (M1 to M39;
   spec 8 and 8.1). It passes only if the good reference fails none of your
   tests (the README tests apart) and every variant fails at least one.
   Nineteen are new. Seven of them (F16 to F22) are caught by tests that
   exist already. The other twelve each have a test of this pass named for
   them, and so does each of the eleven older variants that one test held
   alone until now:

   | Variant | The test that must fail on it | Cases that fail on it in these files |
   |---|---|---|
   | F12b | `test_group_after_leak.py::test_a_copy_of_the_target_under_group_stops_the_run` | 4 |
   | F14b | `test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column` | 37 |
   | F15 | `test_fast_cap_groups.py::test_an_entity_larger_than_the_cap_stops_the_fast_profile` | 7 |
   | M31 | `test_date_sentinels.py::test_a_sentinel_counts_once_however_many_cells_hold_it` | 19 |
   | M32 | `test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run` | 8 |
   | M33 | `test_group_after_leak.py::test_one_combination_on_both_sides_is_enough` | 1 |
   | M34 | `test_fast_cap_groups.py::test_when_the_full_profile_would_stop_too_its_error_is_raised` | 1 |
   | M35 | `test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept` | 4 |
   | M36 | `test_review6_small.py::test_the_date_warning_holds_the_caveat_of_its_scheme`, its two grouped cases | 2 |
   | M37 | `test_group_after_leak.py::test_the_grouped_warning_says_what_group_measures` | 1 |
   | M38 | `test_guard_review6.py::test_m38_one_fold_not_scored_record_for_each_fold_not_scored` | 2 |
   | M39 | `test_guard_review6.py::test_m39_rows_with_a_cell_that_cannot_be_hashed_are_counted_as_repeats` | 1 |
   | M1, M5, M6, M8, M10, M11, M12, M17, M21, M23, M28 | the test of `test_guard_review6.py` named for each (`test_m1_...` to `test_m28_...`), beside the test that held it before | one each; M12 two |

   You may run it yourself, from the repo root, with the project interpreter:
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/mutation_check.py --jobs 8`.
   It writes nothing inside the repo, and takes a little over an hour. Do not
   read the reference implementation to tune a test to it. The README test of this
   pass goes in a file named `tests/test_readme_review6.py`: files named
   `tests/test_readme*.py` are shown on a line of their own and not counted.
   **`tests/test_judge_tools.py` is not run by the mutation check at all:** it
   runs the judge's two other tools on the application, and a throwaway
   project has no `verify/`.

7. **A test of this pass says both things in every call, or is about a stop.**
   As in the sixth pass, the new files call through `cp7_helpers.run`,
   `raised` and `run_cli`, which add no statement, and pass `time_col` and
   `group_col` in every call that is meant to finish (`**s.SAID` for "no" to
   both).

8. **Which of the smaller entities the fast profile's cap keeps is open (spec
   step 6 and 11), and so is which units a grouped split holds out.** A test
   asserts the four properties of the cap and the properties of the split,
   through the rows a recording family was handed, and never a particular
   row or entity. What is pinned and does not depend on the seed: an entity of
   a quarter or more of the usable rows is kept, and the run stops when the
   cap cannot do that.

9. **Leave what pass 8 will change unpinned (D73).** Do not assert which
   baselines a regression has under time order with repeated timestamps and
   `group_col=False`; do not assert that a column which copies a regression
   target is, or is not, excluded under time order or beside a dominant
   entity; do not assert the fold scores of a forest under time order. The
   recipes below already keep clear of all three.

10. **The rules of the earlier passes still hold**: the documented surface only
    (spec 2.1 to 2.7); pin what the spec pins and nothing else; recompute, do
    not trust; assert on values as the application produced them; **no skipped
    or expected-to-fail test**; nothing written outside `tmp_path`; the
    application imported inside test functions; the command line run one
    process at a time. What a message must hold is pinned in spec 11 and 5.4
    and nowhere else. Tiny tables: the tests of the fast cap need a little
    over 2,000 rows, one guard needs 3,000 and one 1,500; nothing else needs
    more than 700. Use a recording family or a constant family through the
    seam wherever a run only has to finish.

## Files of the seventh pass

Each file belongs to exactly one task of `plan/tasks.json`.

| File | Task | About | Today |
|---|---|---|---|
| `tests/test_guard_review6.py` | T54 | a second test for eleven faults that one test each held; a date beyond the nanosecond range; the two readings D72 pins (D71, D72) | green |
| `tests/test_date_sentinels.py` | T55 | a column of dates written `YYYYMMDD` with values that are no dates, as a spare column and as the time column (D69, D72) | red, four tests green in part or in whole |
| `tests/test_group_after_leak.py` | T56 | `duplicates="group"` and a column the leak screen removes; what `"group"` measures (D69, D72) | red, three tests green |
| `tests/test_fast_cap_groups.py` | T57 | the fast profile's cap and a large entity (D70, D72) | red, three tests green |
| `tests/test_review6_small.py` | T58 | the text of the date warning (D69) | red |
| `tests/test_readme_review6.py` | T59 | the README on this pass | red |
| `tests/test_judge_tools.py` | T60 | the perturbation battery and the generated-table test, run on the application (D56, D71, D72) | red |
| `tests/cp8_helpers.py` | none | shared helpers of the seventh pass; holds no test | |
| `tests/test_date_numbers.py` | (T52) | one replacement, as rule 3 says; nothing else | green before and after |

### `tests/cp8_helpers.py`

Import the earlier helper files and build on them; do not copy from them. Like
`cp7_helpers`, nothing here makes a statement the call did not make: the
tests of this pass call through `cp7_helpers.run`, `raised` and `run_cli`.

- **`CAP`**: 2000. **`CAP_CODE`**: `fast_profile_group_too_large`.
  **`SENTINEL`**: 99991231. **`OTHERS`**: `(99991231, 0, 99999999)`, three
  values that are no dates.
- **`GROUP_MEASURES`**: the fixed passage of spec 5.4 that the
  `same_features_grouped` warning holds, exactly. **`RANDOM_CAVEAT`** and
  **`GROUPED_CAVEAT`**: the two caveats of 5.4, taken from `mlp_helpers.CAVEAT`
  and `cp6_helpers.GROUPED_CAVEAT`.
- **`roles(frame, target="y", **kw)`**: `stages.column_roles` as a mapping
  from a column's name to its record. **`is_date(record)`**: role `excluded`
  and reason `datetime`. **`is_number(record)`**: role `feature` and kind
  `numeric`.
- **`rows_reached(result, log)`**: for an `h.Log`, the holdout rows together
  with every row a model was fitted on or asked to predict: the rows a cap
  kept.
- **`feature_keys(result, frame)`**: for every row of `frame`, the tuple of
  its values in the columns the result gives the role `feature`, as step 8a
  compares them: a number as a float, text without the white space round it,
  an empty cell as `(missing)`.
- **`whole_numbers_in(text)`**: the whole numbers written in a text.
  **`holds_count_and_share(text, n_rows)`**: whether a message holds two
  whole numbers a <= b, with b at least a quarter of `n_rows` and fewer than
  `n_rows`, and some number within 0.1 of 100 x a / b.
- **`days(n_dates, start="2021-03-01")`**: that many consecutive days, each as
  the whole number `YYYYMMDD`.
- **`planted(frame, values, cells=1, share=None, seed=0, column="day")`**: a
  copy of `frame` with `values` planted in the column: `cells` cells for each
  value, or with `share` that share of the rows, dealt out among the values in
  turn. The cells are drawn from the seed and the column keeps its type.
- **`dates_with(n_dates, others=(), n=300, cells=2, seed=0)`**: a table whose
  `int64` column `day` holds exactly `n_dates` distinct dates, each many
  times, and each of `others` in `cells` cells; a noise column `x` and a 0/1
  target `y`; rows shuffled.
- **`with_copy_of_target(frame, name="status", target="y")`**: `frame` with a
  text column before the target that copies a 0/1 target: `closed` for 1 and
  `open` for 0.
- **`one_row_alike_in_every_entity(n_entities=24, per=20, seed=0)`**: `who`,
  four noise measures `n1` to `n4`, a column `leak` that is the 0/1 target
  plus (row number + 1) / (2 x rows), so that it copies the target and is
  distinct in every row, and `y`. The first row of each entity holds the
  measures 0.125, -0.5, 0.75 and -1.25, with the classes 0 and 1 in turn. So
  no two rows agree in every column, and without `leak` those rows, one in
  each entity, agree in every feature that is left. Rows shuffled.
- **`large_entities(large, small=300, per=5, seed=0, labels=None)`**:
  `s.sized_entities` with the large entities first (`e000`, `e001`, ...
  hold the rows given in `large`) and `small` more of `per` rows each.

## The guards (task T54; all green today)

### `tests/test_guard_review6.py`

Spec 8.1, with steps 3, 6, 8, 9, 10, 11 and 12, 2.2, 2.5 and 5.2. Fourteen
tests, seventeen cases. **Every one passes against the application as it
stands and against the reference, and fails on the variant it is named for;**
none of them fails on any other of the thirteen variants. If one fails
against the application, the application is wrong.

Eleven are **second tests**. `docs/BACKLOG.md` lists the small faults that one
test held alone (M1, M5, M6, M8, M10, M11, M12, M17, M21, M23, M28); each now
has a second one, by another route than the first: another scheme, another
task, another entry point. Do not write the first test again with other
numbers: the route is the point, and it is given with each test below. Two
hold what D72 pinned as the application already has it (M38, M39). One has no
variant: a date beyond the nanosecond range among ordinary ones, which ended
the process inside pandas in the reference until this pass.

The file imports the helper files that exist today (`mlp_helpers` as `h`,
`cp3_helpers` as `c`, `cp4_helpers` as `k`, `cp7_helpers` as `s`) and keeps
its own small helpers inside itself; it does not import `cp8_helpers`. Every
call says both things: `**s.SAID`, or the columns. Three helpers of its own
serve several tests:

- `_lookup(values)`: one seam family, `h.family("lookup", h.Log(),
  estimator=h.LookupEstimator, values=values)`.
- `_cli(csv_path, target, out, *more)`: `s.run_cli(["profile", "--data",
  csv_path, "--target", target, "--out", out, "--profile", "fast",
  "--n-jobs", "1", *more])`, which adds no flag.
- `_files(out)`: `h.load_json_strict(out / "profile.json")` and the rows of
  `predictions.csv` read with `csv.DictReader`.

The whole file takes about 25 seconds: six runs of the command line, and
about 110 runs through the seam.

- `test_m1_the_final_fit_is_centred_on_its_own_rows` (fails on **M1**). Spec
  step 8, 2.6. *The first test changes the holdout rows of a random split and
  asks that no fit's matrix moves. This one is under time order, for a
  regression, and reads one matrix by hand.* 240 rows, `t = 0.0, 1.0, ...`,
  from `default_rng(6101)`, drawn in this order: `rising = 0.05·t` plus
  standard normal noise; `gappy = 0.05·t` plus standard normal noise, then
  empty where a uniform draw is below 0.15; `y`, standard normal. If the
  number of filled `gappy` cells among rows 0 to 179 is even, empty the first
  filled one, so that the median of those cells is the value of one row. Both
  columns grow with time, so the holdout, the latest rows, holds their largest
  values. Run with a `c.KeepLog` and `h.family("keep", log,
  c.MatrixEstimator)`, `time_col="t"`, `group_col=False`,
  `task="regression"`. Assert: the holdout is rows 180 to 239; the model was
  fitted six times, no fit was handed a row from 180 on, and exactly one was
  handed rows 0 to 179; **in every one of the six matrices the column `rising`
  has mean 0 within 1e-9 and a standard deviation of 1 within 1e-9** (the
  population or the sample deviation: which of the two is open); in the fit
  of 180 rows, **every row whose `gappy` cell is empty holds, in the column
  `gappy`, the value of the one row whose cell is the median of the filled
  training cells**, within 1e-12; and `n_imputed` of `gappy` is the number of
  its empty cells. Measured on M1: in the fit of 180 rows `rising` has mean
  -0.424 and deviation 0.762, and the empty cells hold +0.059 where the median
  row holds -0.406; the five fold fits are as they should be.
- `test_m5_a_shuffle_that_ties_counts_where_only_some_of_them_tie` (fails on
  **M5**). Spec step 12. *The first test makes every shuffle tie, with a
  constant prediction. This one makes a known share of them tie, and counts
  it.* 400 rows from `default_rng(6501)`: a target of 200 zeros and 200 ones,
  shuffled; `x1`, standard normal; predictions of 0.9 on 16 rows and 0.1 on
  the others (0.9 where the position is a multiple of 25, then shuffled), in
  a `_lookup` family. With two predicted values the ROC AUC of a shuffle
  depends only on how many positives fall among the rows predicted 0.9, which
  has a hypergeometric law, so the share of shuffles whose skill is **at
  least** the observed one is a sum the test can make exactly. For each
  `seed` from 0 to 39, with `**s.SAID`: `n_permutations == 200`;
  `headline.score` equals `h.auc_by_hand` of the predictions within 1e-12;
  from `result.predictions` take the holdout rows *H*, its positives *P*, the
  rows predicted 0.9, *A*, and the positives among them, *k*. Twice *P*·*N*
  times the AUC with *j* positives among the *A* is the whole number
  `2·j·(N − A + j) + j·(A − j) + (P − j)·(N − A + j)`, with *N* = *H* − *P*;
  sum `comb(P, j)·comb(N, A − j) ÷ comb(H, A)` over every possible *j* for
  which that number is at least its value at *k* (use `fractions.Fraction`).
  Record `p_value − (1 + 200 × that share) ÷ 201`, and the share for which the
  number equals its value at *k*. Assert that the mean of the second over the
  40 seeds is above 0.15 (the table makes shuffles tie), and that **the mean
  of the first is below 0.05 in absolute value**. Measured over the 40 seeds:
  the reference -0.001 and the application +0.005, one run moving by 0.02 to
  0.03, so the mean by about 0.005; M5 -0.321; a third of the shuffles tie
  (0.32 to 0.33). Do not use fewer seeds and do not tighten the 0.05. A
  holdout that holds none of the 16 rows is allowed for: every shuffle then
  ties.
- `test_m6_an_infinite_holdout_prediction_in_a_regression_falls_back` (fails
  on **M6**). Spec steps 9 and 10, "Numbers that are not finite" rule 4.
  *The first test is a classification with not-a-number predictions, where
  ROC AUC stays finite. This one is a regression with infinite ones, where the
  fault ends in the wrong named error.* `h.reg_frame(n=240, seed=6601)`; from
  `default_rng(6602)`, `sharp = y + 0.3·noise` and `sound = y + 1.5·noise`.
  A first run with `sound` alone gives the holdout rows (a regression's split
  depends on the rows and the seed only), and a run with `sharp` alone its
  five fold scores. Then `wild` is `sharp` with `+inf` and `-inf` in the
  first two holdout rows; families `sharp` (values `wild`) then `sound`, with
  an output folder. Assert: the run finishes; `sharp-0` has status `failed`,
  an error text, `chosen` false, **the fold scores and the `cv_mean` of the
  clean run** (within 1e-12, relative) and a lower `cv_mean` than `sound-0`,
  so it was the first choice; `sound-0` is chosen and is `headline.model`;
  one `model_failed` warning; the holdout rows are those of the first run;
  every `y_pred` is finite and equals `sound` there; `headline.score` is the
  squared error by hand; `k.assert_raw_numbers(result)`; and every `y_pred`
  written to `predictions.csv` is finite. On M6 the infinite prediction is
  scored, the score is infinite, and the run stops with `non_finite_score`.
- `test_m8_the_interval_from_whole_entities_is_the_middle_95_percent` (fails
  on **M8**). Spec step 12. *The first test averages the row bootstrap of the
  random scheme. This one averages the bootstrap of whole entities.*
  `s.sized_entities([3] * 160, seed=6801)`, and predictions of `y` plus
  standard normal noise from `default_rng(6802)` in a `_lookup` family. For
  each `seed` from 0 to 59, with `task="regression"`, `time_col=False`,
  `group_col="who"`: `interval_method == "bootstrap_groups"`,
  `interval_level == 0.95`, `n_resamples_used == 200`; the squared errors of
  the holdout rows have `headline.score` as their mean; group them by the
  entity of each row (read from the frame): at least 30 entities, three rows
  each. Draw 4,000 resamples of those entities with replacement from a
  generator of the test's own (`default_rng(6803)`, one generator across the
  60 runs), the mean of a resample being the sum of its entities' squared
  errors over three times their number; take the 2.5th and 97.5th percentiles
  and the standard deviation of the 4,000 means; record `(interval_low − own
  2.5th) ÷ sd` and `(interval_high − own 97.5th) ÷ sd`. **Assert that the mean
  of each over the 60 seeds is below 0.22 in absolute value.** Measured over
  the 60 seeds: the reference +0.019 and -0.074, the application +0.046 and
  -0.080 (one run moves by 0.18 to 0.20, so the mean by about 0.025); M8
  +0.290 and -0.383. With 40 entities in the holdout the ends sit a little
  inside the test's own, on both implementations alike, which is why the
  allowance is 0.22 and not the 0.18 of the first test. Do not use fewer
  seeds. About four seconds.
- `test_m10_the_fast_cap_is_a_sample_whatever_the_order_of_the_file` (fails
  on **M10**). Spec step 6. *The first test counts, through the seam, the kept
  rows that lie beyond position 2,000. This one goes through the command
  line, on a file whose order makes the head of the file useless.* A CSV of
  3,000 rows written under `tmp_path`: `x1` and `z1` standard normal from
  `default_rng(6001)`, and a target `y` that is `a` in the first 2,000 rows
  and `b` in the last 1,000. For `--seed 0` and `--seed 1`, with
  `--no-time-order --no-entities`: exit code 0; `rows` has `dropped_row_cap`
  1000, `training` 1500 and `holdout` 500; `task.classes` is `["a", "b"]`;
  every row of `predictions.csv` at position 2,000 or beyond has `y_true`
  `b`, and **between 140 and 195 of the 500 holdout rows lie there** (a sample
  of 2,000 from 3,000 rows holds about 667 of the last 1,000, and a quarter of
  those are held out; the head of the file holds none). The two seeds hold
  out different rows. On M10 the 2,000 rows kept hold one class, and the run
  ends with exit code 3, `fast_profile_too_few_rows`.
- `test_m11_cv_std_of_two_scored_folds_is_their_sample_deviation` (fails on
  **M11**). Spec step 9, 5.2. *The first test has five scored folds under the
  random scheme. This one has two, under time order, where the two deviations
  differ by the square root of two.* The table `_late_classes(4)`: 400 rows
  from `default_rng(6111)`, drawn in this order: `t = 0.0, 1.0, ...`; `y`, a
  whole number 0 or 1 for each row, then set to 0 in the first `50 × (b − 1)`
  rows, where `b` is the argument, the first block that holds both classes;
  `x1`, standard normal; predictions `0.3·y` plus a uniform draw on 0 to 0.7,
  in a `_lookup` family. The holdout is the last 100 timestamps and the
  training timestamps fall into six blocks of 50, so with `b = 4` folds 1, 2
  and 3 train on rows that hold one class. With `time_col="t"`,
  `group_col=False`: the task is classification;
  `validation.folds[].scored` is false, false, false, true, true;
  `cv.n_scored_folds == 2`; the one configuration has three null fold scores
  and two numbers *a* and *b* that differ; `cv_mean` is their mean; **`cv_std`
  equals `abs(a − b) ÷ sqrt(2)`**, and `statistics.stdev([a, b])`, within 1e-9
  (relative); `cv.chosen_std` is that `cv_std`. Measured: the two scores are
  0.836 and 0.963 on both implementations, `cv_std` 0.0895, and on M11 0.0633.
- `test_m38_one_fold_not_scored_record_for_each_fold_not_scored`, two cases
  (fails on **M38**). Spec 2.5, step 7 (D72 point 6). The same table: case
  `three_folds` is `_late_classes(4)`, case `two_folds` is `_late_classes(3)`,
  in which folds 1 and 2 are not scored. With `time_col="t"`,
  `group_col=False`: the folds not scored are 1 to 3, or 1 and 2; **the
  number of `fold_not_scored` warnings equals the number of folds not
  scored**, and is at least two; each has a null `column`;
  `cv.n_scored_folds` is 5 less that number. On M38 there is one record in
  both cases. This is a guard of what D72 pinned: the application already
  writes one for each fold.
- `test_m12_tied_predictions_in_a_resample_of_blocks_and_of_entities`, two
  cases (fails on **M12**, both). Spec steps 11 and 12. *The first test
  resamples rows. These are the other two bootstraps.* A coin-flip target and
  an `h.LookupEstimator` predicting 0.5 for every row, from
  `default_rng(6121)`. Case `blocks`: 240 rows, `t = 0.0, 1.0, ...`, `x1`
  standard normal, `y` 0 or 1; `time_col="t"`, `group_col=False`; the method
  is `bootstrap_blocks`. Case `entities`: 480 rows, `who` naming 160 entities
  of three rows, `x1`, `y` 0 or 1 for each row, the rows shuffled;
  `time_col=False`, `group_col="who"`; the method is `bootstrap_groups`. In
  both: `score == 0.5`, `skill == 0.0`, `n_resamples_used == 200`,
  **`interval_low == interval_high == 0.5` and `skill_low == skill_high ==
  0.0`, exactly**, and no p-value. Measured on M12: 0.348 to 0.654 under
  blocks, 0.405 to 0.604 under entities.
- `test_m17_the_fast_cap_keeps_whole_entities_of_seven_rows` (fails on
  **M17**). Spec step 6. *The first test reads the kept rows through the seam,
  with entities of every size from 1 to 11. This one goes through the command
  line with entities of one size that does not divide the cap.* A CSV of
  `s.sized_entities([7] * 400, seed=6171)`, 2,800 rows, with
  `--no-time-order --group-col who`: exit code 0. Whole entities of seven
  rows under a cap of 2,000 are 285 entities, 1,995 rows, and never 2,000:
  **`training + holdout == 1995` and `dropped_row_cap == 805`**; the scheme is
  `grouped`; `n_groups_training + n_groups_holdout == 285`; and every entity
  named by a row of `predictions.csv` has exactly seven rows there, as many
  entities as `n_groups_holdout`. On M17 2,000 rows are kept.
- `test_m21_a_resample_of_whole_entities_has_one_score` (fails on **M21**).
  Spec step 12. *The first test asks for an interval wider than 0.18, by
  effect. This one is exact, as the block bootstrap's guard is.* From
  `default_rng(6211)`: 160 entities of two rows; `y` is a standard normal
  draw rounded to an eighth, plus 3; `x1` standard normal; the prediction is
  `y` for the first row of each entity and `y + 1.0` for the second; then the
  rows are shuffled, the predictions with them. So every squared error is
  exactly 0 or 1 (assert it), every entity holds one of each, and any resample
  of whole entities has a squared error of exactly 0.5. With
  `task="regression"`, `time_col=False`, `group_col="who"`: at least 30
  entities in the holdout; `interval_method == "bootstrap_groups"` and a null
  `block_length`; `score == 0.5`; `n_resamples_used == 200`; **`interval_low`
  and `interval_high` within 1e-12 of 0.5**; no p-value. Measured on M21, five
  seeds: 0.37 to 0.40 at the lower end and 0.59 to 0.61 at the upper.
- `test_m23_a_panel_between_the_two_cut_offs_is_warned_about_on_the_command_line`
  (fails on **M23**). Spec step 11, 4.3, 2.2. *The first test holds the
  cut-off from just above and just below, through the function. This one sits
  midway between 0.5 and the abandoned 0.8, goes through the command line with
  dates as text, and has two to four rows a date.* 80 dates from 2022-01-03,
  written `YYYY-MM-DD`, holding 2, 3, 4, 2, 3, 4, ... rows (239 rows). From
  `default_rng(6239)`, in this order: 80 shocks; the level of a date is 0.68
  times the level before plus its shock, and all levels are then multiplied
  by 3; `x`, standard normal for each row; the target `sales`, the level of the
  row's date plus 0.2 times a standard normal draw; then the rows are
  shuffled. The test
  works out the training dates itself (the holdout is the latest whole dates
  that together hold at least a quarter of the rows) and asserts, **as its own
  precondition, that the lag-1 autocorrelation of the per-date means of the
  training dates lies between 0.62 and 0.74 by both usual estimators** (those
  of `test_guard_review5.py`; from this seed both are 0.669, over 59 dates).
  If your generator gives other values, choose a seed that puts both there:
  the range is what is pinned. Then the CSV with `--time-col date
  --no-entities`: exit code 0; the scheme is `time_ordered`,
  `timestamps_unique` is false and `n_train_timestamps` is the test's own
  count; exactly one warning has the code `autocorrelated_panel_target`, with
  the column `sales`; and some line of standard output holds that code and
  `sales` (the name is one no other word of a message holds).
  It does not assert which baselines are listed: D73 changes that in pass 8.
- `test_m28_eight_digit_text_orders_a_panel_and_read_time_reads_it` (fails on
  **M28**). Spec step 3, 2.7, step 11. *The first test is one series of a
  string-typed column. This one is a panel with its entities, in an object
  column, with white space and an empty cell.* Six random walks on 60 days
  from 2023-11-01, from `default_rng(6281)`: series `s00` to `s05`, each
  starting ten apart, with `prev` (the level one step before) and `y`; the
  rows shuffled; `day` written `YYYYMMDD` as text in an `object` column, with
  a space on each side in every seventh row, and `None` in row 11. With a
  recording family, `time_col="day"`, `group_col="who"`: the holdout is the
  rows of the latest whole days that together hold a quarter of the 359 rows
  with a day, by the test's own count; the scheme is `time_ordered` and
  `time_form` is `numeric`; `dropped_missing_time == 1`; `train_end`,
  `holdout_start` and `holdout_end` are the test's own whole numbers, not
  text; `day` has the role `time`; the baselines are `training_mean` then
  `previous_value`; `n_holdout_without_previous == 0`. And
  `stages.read_time(frame["day"])` gives the form `numeric` and `float64`
  keys: not-a-number at row 11, and elsewhere the number each cell reads as.
  On M28 the run ends with `time_column_invalid`.
- `test_one_date_beyond_the_nanosecond_range_among_ordinary_ones_is_the_named_error`,
  two cases (no variant). Spec step 3, 2.2. A time column of 1,500 ordinary
  days from 2015-01-01 with `9999-12-31` in row 700, `x` and `y` standard
  normal. **Each case runs in a process of its own**, so that a build which
  dies there fails the test and does not end pytest: on a column of this
  size pandas 3.1.1 does not raise when it turns such a column into
  nanoseconds, and the process ended with an access violation in the
  reference until it asked first. Case `iso_text` (`default_rng(6901)`): the
  CSV through `_cli` with `--time-col date --no-entities`: exit code 3, one
  line on standard error and it starts with `error: time_column_invalid:`.
  Case `datetime_in_seconds` (`default_rng(6902)`): `subprocess.run([h.PYTHON,
  "-c", script, out], cwd=h.REPO_ROOT)` with bytecode off, where the script
  builds the same table with the column typed `datetime64[s]` (assert the
  type), calls `profile(..., time_col="date", group_col=False,
  profile="fast", n_jobs=1, out_dir=out)` and prints `named:` and the code of
  the `ProfileError`: exit code 0 and `named: time_column_invalid`. In both,
  nothing is written under the output folder. A table of 400 rows does not
  show the fault: keep the 1,500.
- `test_m39_rows_with_a_cell_that_cannot_be_hashed_are_counted_as_repeats`
  (fails on **M39**). Spec 5.2 (D72 point 6). 200 rows from
  `default_rng(6391)`: a column `tags` of lists of two whole numbers (the
  first from 0 to 4, the second from 5 to 8), `x1` and `z1` standard normal,
  `y` 0 or 1. Add rows 10 to 24 again, each with a new list of the same
  content, and rows 40 to 44 again with the list turned round, which is
  another text; shuffle the 220 rows. By hand, with the list compared by its
  text, 15 rows repeat an earlier row (assert it). With a constant `_lookup`
  family, `duplicates="keep"` and `**s.SAID`: **`input.duplicate_rows` is
  15**, there is exactly one `duplicate_rows` warning, and `tags` is excluded
  as `unsupported_type`. On M39 the count is 0 and there is no warning. This
  is a guard of what D72 pinned: the application already counts so.
  `duplicates="keep"` is there because the twenty added rows also agree with
  another row in both features and the target, which is step 8a's business
  and not this test's.

## A: a column of dates with values that are no dates (task T55)

### `tests/test_date_sentinels.py`

Spec: 2.4, 2.5, 2.7, steps 3 and 5 (rule 2a), 4.2, 4.3, 11. A column is
judged through `stages.column_roles` unless a run is needed. `_days(seed,
form)` is `s.day_numbers(seed=seed)`, 600 rows, three for each of 200 days,
with `day` as `int64`, as floats or as text of eight digits.

- `test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column`,
  21 cases. **F14b.** A column of 200 days with: one cell of 99991231; of 0;
  of 20240230; 99991231 in 5%, 30% and 60% of the rows; one cell each of
  99991231 and 0; one cell each of the three `OTHERS`; floats with every
  seventeenth cell empty and one infinite cell; text with one cell `0`; and
  the eleven columns of `test_date_numbers.py` that hold one value that is no
  date (18991231, 21010101, 20230229, 19000229, 20231301, 20230100, 20230431,
  a fraction, an infinite cell, text of nine digits, text of seven). Each:
  role `excluded`, reason `datetime`, and `n_coerced`, `n_infinite` and
  `n_imputed` 0.
- `test_a_sentinel_counts_once_however_many_cells_hold_it`. **M31.** 600
  rows: twenty days in 60 cells and 99991231 in the other 540: a date column.
  The same 60 cells of days with 60 distinct values that are no dates in the
  other 540: a numeric feature.
- `test_the_rule_from_both_sides`, seven cases, through
  `e.dates_with(n_dates, e.OTHERS[:n_others])`, whose column holds exactly
  that many distinct values of each kind. A date column: two dates and
  nothing else (**green today**); five dates and one other value; five and
  two; 27 and three. A numeric feature (**green today**): four dates and one
  other value; four and two; 26 and three.
- `test_a_few_codes_of_which_some_read_as_dates_are_no_date_column`, nine
  cases. **Green today.** 300 rows of two, three or five distinct eight-digit
  codes, of which one to four read as dates (20230115, 20240229, 19991231,
  20110704) and the others do not (31415926, 20231301, 12345678, 99999999):
  two codes with one date; three with one and with two; five with one, two,
  three and four, as `int64`; three with two and five with four as text. Each
  has the role `feature`.
- `test_too_many_values_that_are_no_dates_leave_the_column_a_number`.
  **Green today**; fails on M31. 200 days, with one of the three cells of 23
  of them, and then of 70, set to a value of its own that is no date
  (99991231, 99991230, ...): 200 dates beside 23 other values, then beside
  70, each a numeric feature. This is the stated limit of the rule.
- `test_the_date_rule_is_judged_on_the_usable_rows`. One cell of 99991231;
  then 60 other rows lose their target and are each given a value of their
  own that is no date (30000000 to 30000059). Over the usable rows the column holds at least 190
  days and one other value: a date column.
- `test_it_is_excluded_named_and_handed_to_no_step`. 99991231 in 30% of the
  rows, both statements, a recording family and step: the role, the reason
  and the three counts as above; exactly one `date_column_undeclared`
  warning, and it names `day`; the scheme `random`; no step is handed a
  column `day`.
- `test_it_is_named_when_nothing_is_said_and_by_the_copies_stop`. 99991231
  in 30% of the rows and nothing said: `time_order_undeclared`, and the
  message holds `'day'`. Then `p.noise_frame(300)` with a column `visit_day` of 300 days in
  which every fourth cell is 99991231, every row twice, both statements:
  `HoldoutCopiesError`, whose message holds `visit_day` and `time_col`.
- `test_under_another_time_column_it_is_excluded_without_a_word`.
  `h.time_frame(200)` with a column `closed` of 200 days in which every third
  cell is 99991231, shuffled, under `time_col="date"`: `excluded`,
  `datetime`, and no `date_column_undeclared`.
- `test_as_the_time_column_a_value_that_is_no_date_stops_the_run`, three
  cases. **M32.** `day` as `int64`, as floats (with three empty cells) and as
  text, with 99991231 in 37 cells, under `time_col="day", group_col=False`
  with an `out_dir`: exactly `TimeColumnError`, the code
  `time_column_invalid`, the message holds `99991231` and the whole number
  37, and nothing is written.
- `test_the_message_names_the_value_that_most_rows_hold`. 99991231 in 37
  cells and 0 in five: `TimeColumnError`, with `99991231` and 37.
- `test_the_time_column_is_judged_over_every_value_whatever_its_row`. The 37
  rows that hold 99991231 have no target: `TimeColumnError` all the same,
  with `99991231`.
- `test_read_time_refuses_it_too`, three cases. `stages.read_time` on the
  column of each of the three forms raises `TimeColumnError` with `99991231`
  and 37; on the clean column it returns `float64` keys and the form
  `numeric`.
- `test_a_time_column_that_is_no_date_column_is_a_number_as_before`. **Green
  today.** 200 days with 30 cells each set to a value of its own that is no
  date (30000000 to 30000029), under `time_col="day"`: `time_ordered`,
  `numeric`, `holdout_end` 30000029, the role `time`. And a column that
  counts the rows from 0 with one cell of 99991231: it runs, and
  `holdout_end` is 99991231.

## B: `duplicates="group"` and a column the leak screen removes (task T56)

### `tests/test_group_after_leak.py`

Spec: 2.4, 2.5, steps 7 and 8a, 5.4, 11. `_few_valued_with_status(seed)` is
`e.with_copy_of_target(s.few_valued(600, (4, 4), seed=seed))`.

- `test_a_copy_of_the_target_under_group_stops_the_run`. **F12b.** That
  table with `duplicates="group"`, both statements, a recording family and
  step and an `out_dir`: exactly `HoldoutCopiesError`, the code
  `holdout_copies_in_training`; the message holds `status`, `group`, `keep`
  and `s.COPIES_ADVICE`, and a count and a share that agree
  (`e.holds_count_and_share`); the log holds no event; nothing is written.
- `test_the_stop_names_a_date_column_that_was_excluded_too`. The table of
  `_few_valued_with_status(1910)` with a first column `seen on` of 600
  consecutive days from 2022-01-03 written `YYYY-MM-DD`, `duplicates="group"`
  and both statements: exactly `HoldoutCopiesError`, and the message holds
  `status`, `seen on` and `time_col`. Every message of this error names a
  date column that was excluded because no time column was declared (spec
  11), the message of this stop included.
- `test_one_combination_on_both_sides_is_enough`. **M33.**
  `e.one_row_alike_in_every_entity(24, 20)`, whose own checks are part of the
  test: no two rows agree in the four measures and `leak`; 24 rows, one in
  each entity and twelve of each class, agree in the four measures. With
  `duplicates="group"`, `time_col=False` and `group_col="who"`: exactly
  `HoldoutCopiesError`; the message holds `leak` and `group`, a whole number
  from 1 to 9 (each held-out entity gives one such holdout row, and a quarter
  of 24 entities is never ten), and a count and a share that agree.
- `test_no_combination_of_the_features_is_on_both_sides_in_a_finished_run`,
  three cases. **Green today.** `s.few_valued(600, (4, 4))`: `grouped`.
  `e.with_copy_of_target(p.noise_frame(400))`: `random`, and `status` is
  excluded as a suspected leak. The same of `p.twice(p.noise_frame(300))`:
  `grouped`, `status` excluded. Each with `duplicates="group"`, both
  statements and a recording family: every row was reached, and no tuple of
  `e.feature_keys` is held by both a holdout row and another row.
- `test_the_grouped_warning_says_what_group_measures`. **M37.**
  `s.few_valued(600, (4, 4))` with `duplicates="group"`: exactly one
  `same_features_grouped` warning, and its message holds `e.GROUP_MEASURES`.
- `test_under_keep_and_error_nothing_changes`. **Green today.** The table
  with `status` under `duplicates="keep"`: it runs, `status` is excluded as a
  suspected leak, the scheme is `random`, `n_rows_same_features` is null and
  there is exactly one `holdout_copies_kept` warning. Under the default:
  `HoldoutCopiesError` with `s.COPIES_ADVICE`.
- `test_the_command_line_stops_too`. The table as a CSV, `--duplicates group`
  with both flags, through `s.run_cli`: exit 3, one line on standard error
  starting `error: holdout_copies_in_training: ` that holds `status`, and no
  file.
- `test_without_the_leak_the_same_table_runs`. **Green today.** The table of
  the first test without `status`, `duplicates="group"`: `grouped`,
  `n_holdout_copies` 0, and the result is strict JSON.

## C: the fast profile's cap and a large entity (task T57)

### `tests/test_fast_cap_groups.py`

Spec: 2.4, steps 6 and 7, 4.2, 7, 11. Errors are compared by exact class.
Every table is `e.large_entities(...)`: noise, with the large entities named
`e000`, `e001`. `_entity_stop(frame, **kw)` is `s.raised` with a flat family,
`time_col=False` and `group_col="who"`. `_assert_too_large(err)`: exactly
`FastProfileGroupError`, the code, the cap as `2000` or `2,000` and the word
`full` in the message.

- `test_the_error_is_named`. `errors.FastProfileGroupError` is a subclass of
  `ProfileError` with the code `fast_profile_group_too_large`; the two older
  codes of the fast profile are as they were.
- `test_an_entity_larger_than_the_cap_stops_the_fast_profile`. **F15.** One
  entity of 2,500 rows and 300 of five (4,000 rows), seeds 0, 1 and 2, with
  an `out_dir`: `_assert_too_large` each time, and nothing is written.
- `test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept`. **M35.**
  One entity of 1,200 rows and 300 of five (2,700 rows), a recording family,
  seeds 0 to 5: every row of `e000` is among `e.rows_reached`; 2,000 rows are
  kept, which is `training` plus `holdout`; `dropped_row_cap` is 700.
- `test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run`.
  Two entities of 1,100 rows and 360 of five (4,000 rows), seeds 0 and 1:
  `_assert_too_large`. Two of 900 and 300 of five (3,300 rows), seeds 0 to 3:
  the run finishes, every row of both large entities is reached, and 2,000
  rows are kept.
- `test_the_cap_itself_is_the_boundary`, two cases, seeds 0 and 1. One entity
  of 2,000 rows and 300 of five: `fast_profile_too_few_rows` (the entity is
  kept and fills the cap; one entity cannot be split, and the full profile
  could). One of 2,001 rows: `fast_profile_group_too_large`. The message
  holds `full` in both.
- `test_when_the_full_profile_would_stop_too_its_error_is_raised`. **M34.**
  **Green today.** One entity of 2,500 rows and three of 300: the full
  profile stops with `too_few_groups`. One of 2,500 and 300 of five, with one
  label for each entity and the label 1 for a single entity of five rows: the
  full profile stops with `too_few_rows`. In both the fast profile raises an
  error of exactly the class the full profile raises, with the same code.
- `test_the_cap_keeps_whole_entities_every_large_one_and_no_more_than_it_can`.
  `s.sized_entities` of one entity of 700 rows and 320 of 1 to 11 rows drawn
  from seed 2008 (more than 2,500 rows, of which 700 is a quarter or more), a
  recording family, seeds 0 to 3. Each: at most 2,000 rows are kept, and they
  are `training` plus `holdout`; `dropped_row_cap` is the rest; the kept rows
  are whole entities; `e000` is kept; some entity is left out, and the
  smallest left out would not have fitted. And the holdouts of the four seeds
  are not all the same.
- `test_the_full_profile_runs_the_same_table`. **Green today.** The table of
  2,500 and 300 of five under `profile="full"`: `grouped`, no row left out by
  a cap, 4,000 rows split.
- `test_it_is_for_an_entity_column_with_no_time_column_only`. **Green
  today.** That table with a numeric time column of its own, `time_col="t"`
  and `group_col="who"`: `time_ordered`, 2,000 rows kept and 2,000 left out.
  And `s.few_valued(40_000, (4, 4))`, where each combination holds more rows
  than the cap, under `duplicates="group"` with both statements: 2,000 rows
  kept, 38,000 left out, `grouped`, sixteen units.
- `test_the_command_line_exits_three`. The table of 2,500 as a CSV with
  `--profile fast --group-col who --no-time-order`: exit 3, one line starting
  `error: fast_profile_group_too_large: `, no file.

## D: the text of the date warning (task T58)

### `tests/test_review6_small.py`

Spec: 2.5, 5.4. Every table holds a date column named `signed up`.

- `test_the_date_warning_holds_the_caveat_of_its_scheme`, three cases.
  **M36** (its two grouped cases). `random`: `p.noise_frame(300)` with a
  datetime column, both statements. `grouped_by_an_entity_column`:
  `g.entity_frame(60, sizes=5)` with the dates as ISO text, `time_col=False`
  and `group_col="who"`. `grouped_by_rows_that_agree`:
  `p.twice(p.noise_frame(300))` with the dates as whole numbers `YYYYMMDD`,
  `duplicates="group"` and both statements. Each: the scheme, and
  `validation.caveat` equal to the fixed caveat of that scheme; exactly one
  `date_column_undeclared` warning, on `signed up`; its message holds
  `'signed up'` as `repr` writes it, `time_col=False` and the run's
  `validation.caveat`, and does not hold the other scheme's caveat.

The dormant branch of D69 point 6 has no test of its own: nothing can reach
it. `test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite`
(M18) stays its guard.

## The README (task T59)

### `tests/test_readme_review6.py`

Spec: 1. It judges `README.md`, which the builder owns; do not edit it. It
imports `test_readme_examples` for the function that reads the README and for
its pattern of paths under the judge's folders. Pin no wording beyond these.

- `test_the_readme_names_what_the_seventh_pass_added`. The text holds
  `fast_profile_group_too_large`, `99991231` and `YYYYMMDDHH`, and names no
  path under `verify/` or `tests/`.

## The judge's other tools (task T60)

### `tests/test_judge_tools.py`

Spec: 8.2. Each test runs one script of `verify/` from the repo root in a new
process, with the project interpreter (`h.PYTHON`), `-B`, bytecode off in the
environment and `PYTHONPATH`, `PYTHONSTARTUP` and `PYTHONHOME` removed from
it, on the package `mlprofessor`, and asserts exit code 0; on any other code
the assertion shows the last 40 lines of what the script printed. The file
imports nothing from `verify/`. `verify/mutation_check.py` does not run this
file: a throwaway project has no `verify/`.

- `test_the_perturbation_battery_holds`: `verify/battery.py mlprofessor`,
  with a time limit of 1,500 seconds.
- `test_the_generated_tables_agree_with_the_reference`:
  `verify/generated_tables.py mlprofessor`, with a time limit of 900 seconds.

Those limits only keep a hung script from holding the suite for ever: the
loop's own limit for the whole suite, 900 seconds, is the one that binds.
Measured on 2026-11-22, on a machine doing nothing else: the battery takes
three to four minutes at its default of 16 runs side by side (193 and 226
seconds) and the generated tables a little over a minute and a half (97
seconds); the rest of the suite, with the files of this pass, about six
minutes. So the whole suite takes about eleven minutes of the loop's
fifteen. Both tools fail on the application as it stands, and each then says
which pair or which table, and how to run it again.

## What must fail today, and what must pass today

Checked on 2026-11-22 with prototypes of every test of this section and of
the one replacement of rule 3, run against the application as it stands,
against the reference implementation, and against the reference carrying each
of F12b, F14b, F15, M31 to M39 and the eleven older variants named in rule 6.
The prototypes were written outside the repo and are not kept in it: they
are the proof that the recipes work, not text to copy. What they printed is
in `docs/measurements/CP8/` (its README, section 8).

**Red today.** They fail against the application until the build stage has
done D69, D70 and D72, and pass against the reference.

| File | Tests |
|---|---|
| `test_date_sentinels.py` | all but those on the green list: 37 cases |
| `test_group_after_leak.py` | `test_a_copy_of_the_target_under_group_stops_the_run`, `test_the_stop_names_a_date_column_that_was_excluded_too`, `test_one_combination_on_both_sides_is_enough`, `test_the_grouped_warning_says_what_group_measures`, `test_the_command_line_stops_too` |
| `test_fast_cap_groups.py` | all but the three on the green list: 8 cases |
| `test_review6_small.py` | `test_the_date_warning_holds_the_caveat_of_its_scheme`, all three cases |
| `test_readme_review6.py` | its one test; red against the reference too, until T59 is built: it judges `README.md` |
| `test_judge_tools.py` | both tests, until T55 to T57 are built: each tool fails on the application until then. Not run against the reference by the mutation check |

**Green today.** They pass against the application and against the
reference, and must go on passing through every task of the build.

| File | Test | Fails on |
|---|---|---|
| `test_date_sentinels.py` | `test_the_rule_from_both_sides`, four of its seven cases: `two_dates_and_nothing_else`, `four_dates_and_one_other`, `four_dates_and_two_others`, `twenty_six_dates_and_three_others` | M31 (the last three) |
| `test_date_sentinels.py` | `test_a_few_codes_of_which_some_read_as_dates_are_no_date_column`, nine cases | |
| `test_date_sentinels.py` | `test_too_many_values_that_are_no_dates_leave_the_column_a_number` | M31 |
| `test_date_sentinels.py` | `test_a_time_column_that_is_no_date_column_is_a_number_as_before` | M31 |
| `test_group_after_leak.py` | `test_no_combination_of_the_features_is_on_both_sides_in_a_finished_run` (three cases), `test_under_keep_and_error_nothing_changes`, `test_without_the_leak_the_same_table_runs` | |
| `test_fast_cap_groups.py` | `test_when_the_full_profile_would_stop_too_its_error_is_raised` | M34 |
| `test_fast_cap_groups.py` | `test_the_full_profile_runs_the_same_table`, `test_it_is_for_an_entity_column_with_no_time_column_only` | |
| `test_date_numbers.py` | every test, the loosened one in all fourteen cases | not a guard: it passes before the change and after it |
| `test_guard_review6.py` | every test: fourteen tests, seventeen cases | each on the variant it is named for, and on no other of M1, M5, M6, M8, M10, M11, M12, M17, M21, M23, M28, M38, M39; the test of a date beyond the nanosecond range on none |

Measured with the prototypes: 96 cases in the seven new test files: 17 in
`test_guard_review6.py`, 77 in the five files of T55 to T59 (52, 10, 11, 3
and 1), and 2 in `test_judge_tools.py`. Against the application as it stands,
of the first 94, 40 pass and 54 fail. Against the reference 93 pass and one
fails, the README's. With the one replacement in place, all 33 cases of
`test_date_numbers.py` and all 1,808 other existing cases pass against the
application as it stands, and all against the reference.

**What the mutation check must then print**, run after your stage and before
the build: the good reference fails 0 of 1921 (the 1,828 counted today, 17 of
the guards and 76 of the files of T55 to T58); each of the seventy variants
fails at least one test, and each of the twelve new ones, and of the eleven
older ones of rule 6, the test named for it; on the README line, 1 failing of
14, the test of `test_readme_review6.py`; and a note that
`tests/test_judge_tools.py` is not run there.
**Before your stage it ends in `FAIL` for two reasons, both expected:** the
good reference fails eleven cases, all of
`test_date_numbers.py::test_values_that_are_no_dates_leave_the_column_what_it_was`
(the reference has the new rule and the test is not yet loosened); and it
prints `NOT CAUGHT` for twelve of the seventy variants, F12b, F14b, F15 and
M31 to M39, which have no test yet. Your stage closes both.

## Not tested in the seventh pass

Everything D73 gives to pass 8 (rule 9). A date with its hour, `YYYYMMDDHH`,
period labels, day counts and `YYYYMM` as dates: not seen, by decision (D66,
D69, D71). The wording of any message beyond what spec 11 and 5.4 pin. Which
of several values that are no dates the time-column stop names beyond the one
most rows hold. Which of the smaller entities the fast cap keeps, and which
units a grouped split holds out (rule 8). The branch that would have turned a
number that is not finite into null (D69 point 6): nothing can reach it, and
`test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite`
stays its guard. A set of rows that agree, however large, under the fast cap
with `duplicates="group"` and no entity column, beyond the one case of
`test_fast_cap_groups.py`. The time the gate takes. What the judge's two
tools find on a variant of the reference: the lead runs them by hand.

## Ground rules of the eighth pass

1. **The application already exists, and this is the eighth pass.**
   `mlprofessor/` is built and passes all 1,937 cases of the first seven
   passes. Your role prompt says you write the suite before the application
   exists and that every test must fail now. In this pass that is not so:
   most of your tests must fail today, the guards and the controls must pass
   today, and every existing test must pass today and after the build. Rule 5
   says which.

2. **The first seven passes are final. Do not edit them, with the six
   exceptions of rule 3, which D59 allows and which are given there exactly.**
   That means the 65 existing test files, `tests/conftest.py`,
   `tests/mlp_helpers.py`, `tests/cp3_helpers.py` to `tests/cp8_helpers.py` and
   everything under `tests/fixtures/`. New shared helpers go in one new file,
   `tests/cp9_helpers.py`. The owner's lead reads the diff of every existing
   file against the committed one afterwards, and any change other than those
   of rule 3 sends the whole stage back. **`tests/test_skill_undefined.py` is
   not changed:** its test of a column whose single-column skill cannot be
   computed (D36) holds after this pass as it does today, because the order
   reading of D76 is taken only when the first reading has a value.

3. **THE LOOSENINGS (D59). Six existing files; in five of them one
   replacement and no other change, and one file replaced as a whole.**

   D74 point 1 reverses what seven existing cases hold: under time order with
   repeated timestamps and `group_col=False` the training mean used to be the
   only baseline, and `stages.previous_value` used to refuse repeated keys.
   Each of the five replacements below makes its test accept the application
   as it stands and as T62 builds it. D74 point 7 does the same for
   `tests/test_judge_tools.py`: the judge of this pass asserts rules the
   application does not have yet, and that test is in the loop's ratchet. D65
   is not needed: no helper changes. Keep each file's line endings as they
   are. Each text below matches its file exactly once.

   **(a), (b), (c): `tests/test_panel_stop.py`, `tests/test_time_pipeline.py`
   and `tests/test_whole_numbers.py`**, one line in each
   (`test_the_statement_lets_the_run_go_ahead_as_before`,
   `test_panel_uses_training_mean_and_flags_persistence`,
   `test_missing_and_repeated_integer_stamps`). Before:

   ```python
       assert [b["name"] for b in doc["baselines"]] == ["training_mean"]
   ```

   After:

   ```python
       assert [b["name"] for b in doc["baselines"]] in (
           ["training_mean"], ["training_mean", "previous_value"])
   ```

   **(d): `tests/test_guard_review5.py`**, in
   `test_m23_the_panel_warning_starts_just_above_one_half`. Before:

   ```python
       assert [b["name"] for b in doc["baselines"]] == ["training_mean"]
       assert doc["headline"]["baseline"] == "training_mean"
   ```

   After:

   ```python
       names = [b["name"] for b in doc["baselines"]]
       assert names in (["training_mean"], ["training_mean", "previous_value"])
       assert doc["headline"]["baseline"] in names
   ```

   **(e): `tests/test_metrics.py`**, in
   `test_f8_previous_value_refuses_repeated_keys` (the test keeps its name).
   Before:

   ```python
       with pytest.raises(ValueError):
           stages.previous_value(np.arange(5.0), np.array([1.0, 2.0, 2.0, 3.0, 4.0]))
   ```

   After:

   ```python
       # Until D74 repeated keys were refused. Since D74 the previous value of a row is
       # the mean target of the timestamp before. Either is accepted here (D59);
       # the new behaviour is pinned by tests/test_repeated_stamps.py.
       try:
           got = np.asarray(stages.previous_value(
               np.arange(5.0), np.array([1.0, 2.0, 2.0, 3.0, 4.0])), dtype=float)
       except ValueError:
           return
       assert np.isnan(got[0])
       assert np.allclose(got[1:], [0.0, 0.0, 1.5, 3.0])
   ```

   **(f): `tests/test_judge_tools.py`, as a whole.** Until D74 each of its two
   tests asserted that its tool exits 0. Now each tool is run with
   `--json FILE`, FILE in a temporary folder of the session, and the test accepts exit
   0, or exit 1 when every pair (every table) that fails is one the tasks of
   this pass own; any other exit code fails, and so does a failure that is not
   on the list. The two fixed lists are `cp9_helpers.BATTERY_OWNED` and
   `cp9_helpers.TABLES_OWNED` (below). **A tool is run once in a session:**
   `run_tool` keeps what the run gave. The strict form is the new file
   `tests/test_judge_tools_pass8.py`, owned by T68, which asks `run_tool` for
   the same run and asserts exit code 0: the two files together cost one run of
   each tool. The new text of the file, after its docstring (which says what
   this paragraph says):

   ```python
   import json
   import os
   import subprocess
   from pathlib import Path

   import cp9_helpers as q
   import mlp_helpers as h

   BATTERY = ("verify/battery.py", 3600)
   TABLES = ("verify/generated_tables.py", 3600)
   _RUNS = {}


   def run_tool(script, timeout, tmp_path_factory):
       """`(exit code, report, last lines)` of one tool run with `--json`, on the package
       `mlprofessor`, from the repo root. The report is None when the tool wrote none. The
       tool is run once in a session, whichever test asks first."""
       if script not in _RUNS:
           report = tmp_path_factory.mktemp("judge_tools") / (Path(script).stem + ".json")
           env = {k: v for k, v in os.environ.items()
                  if k not in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME")}
           env["PYTHONDONTWRITEBYTECODE"] = "1"
           proc = subprocess.run(
               [h.PYTHON, "-B", script, "mlprofessor", "--json", str(report)],
               cwd=str(h.REPO_ROOT), env=env, capture_output=True, text=True, timeout=timeout,
           )
           printed = (proc.stdout or "") + (proc.stderr or "")
           tail = "\n".join(printed.splitlines()[-40:])
           doc = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else None
           _RUNS[script] = (proc.returncode, doc, tail)
       return _RUNS[script]


   def _owned_or_clean(script, timeout, tmp_path_factory):
       """None when the tool exits 0; its report when it exits 1; any other code fails."""
       code, doc, tail = run_tool(script, timeout, tmp_path_factory)
       assert code in (0, 1), f"{script} exited {code}; its last lines:\n{tail}"
       if code == 0:
           return None
       assert doc is not None, f"{script} exited 1 and wrote no report; its last lines:\n{tail}"
       return doc


   def test_the_perturbation_battery_holds(tmp_path_factory):
       doc = _owned_or_clean(*BATTERY, tmp_path_factory)
       if doc is None:
           return
       failing = {f"{pair['run']}:{pair['change']}" for pair in doc["pairs"]
                  if pair["class"] == "fails"}
       assert failing, "verify/battery.py exited 1 and its report names no failing pair"
       other = sorted(failing - q.BATTERY_OWNED)
       assert not other, (f"{len(other)} pair(s) fail that the eighth pass does not own "
                          f"(of {len(failing)} failing): {other[:20]}")


   def test_the_generated_tables_agree_with_the_reference(tmp_path_factory):
       doc = _owned_or_clean(*TABLES, tmp_path_factory)
       if doc is None:
           return
       failing = {int(entry["number"]) for entry in doc["tables_with_findings"]
                  if entry["class"] == "fails"}
       assert failing, "verify/generated_tables.py exited 1 and its report names no failing table"
       assert len(failing) == doc["tables_by_class"]["fails"]
       other = sorted(failing - q.TABLES_OWNED)
       assert not other, (f"{len(other)} table(s) fail that the eighth pass does not own "
                          f"(of {len(failing)} failing): {other[:20]}")
   ```

   | No longer pinned | By | Pinned again by |
   |---|---|---|
   | at repeated timestamps under `group_col=False` the training mean is the only baseline | `test_panel_stop.py::test_the_statement_lets_the_run_go_ahead_as_before`, `test_time_pipeline.py::test_panel_uses_training_mean_and_flags_persistence` (two cases), `test_whole_numbers.py::test_missing_and_repeated_integer_stamps`, `test_guard_review5.py::test_m23_the_panel_warning_starts_just_above_one_half` (two cases) | `test_repeated_stamps.py::test_both_baselines_are_scored_at_repeated_timestamps` and `::test_the_training_mean_is_the_headline_baseline_where_its_error_is_lower`, which say what the baselines are |
   | `stages.previous_value` raises `ValueError` on repeated keys | `test_metrics.py::test_f8_previous_value_refuses_repeated_keys` | `test_repeated_stamps.py::test_the_step_function_gives_the_mean_of_the_timestamp_before` |
   | each of the judge's two other tools exits 0 on the application | `test_judge_tools.py`, both tests | `test_judge_tools_pass8.py`, both tests (T68) |

   The loosened tests still fail on everything they failed on before: the
   table of rule 6 gives, for every broken copy the suite caught before this
   pass, the cases that fail on it before and after. Four gain cases, because
   the loosened cases pass against the reference of this pass again and fail
   on those copies as they did: F7 six, and F6, F7c and M23 one each. None
   loses a case.

   **No existing test is ever edited so that it fails against the library as
   built:** run the whole suite against the application before you finish and
   see all 1,937 cases pass.

4. **You write twelve new test files and the helper file, and nothing else.**
   The files are listed below, each with the tests it must hold, **under the
   names given**: the lists of rule 5 and the mutation check go by name. Split
   a named test into parameter cases as the text says, with the case names
   given. The tables are given by recipe and by seed: keep both, because for a
   few cases which colour a case has today depends on the rows a seed picks.
   If you add a test of your own, say so in your final report, with the list
   of rule 5 it belongs to.

5. **Which tests fail today and which pass today is fixed, by name.** The
   lists are at the end of this section. A test on the red list fails against
   the application as it stands, in every one of its parameter cases unless
   the list says otherwise. A test on the green list passes today in every
   case. Run the new files before you finish and compare with the lists. **Do
   not weaken a guard to make it fail, and do not weaken a red test to make
   it pass.** If a test on the green list fails, or one on the red list
   passes, and you believe your test is what the spec says, stop and report
   it with the output.

6. **Every new variant must fail a test, and `verify/mutation_check.py` is how
   that will be judged.** It runs the whole suite against a copy of the
   reference implementation and again with the reference broken in one way at
   a time: thirty-seven forms the gate also catches (F1 to F28, F6b, F6c, F6d,
   F7b, F7c, F7d, F8b, F12b, F14b) and fifty-one smaller faults (M1 to M51;
   spec 8 and 8.1). It passes only if the good reference fails none of your
   tests (the README tests apart) and every variant fails at least one.
   Eighteen are new, and each has a test of this pass named for it; so does
   each of the five older variants that one test held alone until now (M13
   has its second from the judge's two tools, spec 8.2):

   | Variant | The test that must fail on it | Cases that fail on it in the new files |
   |---|---|---|
   | F23 | `test_repeated_stamps.py::test_both_baselines_are_scored_at_repeated_timestamps` | 10 |
   | F24 | `test_screen_line.py::test_a_copy_of_the_target_is_excluded_whatever_its_scale` | 29 |
   | F25 | `test_screen_set_aside.py::test_one_wild_target_cell_does_not_hide_a_copy` | 14 |
   | F26 | `test_largest_row.py::test_the_holdout_shares_are_those_of_the_predictions_file` | 3 |
   | F27 | `test_id_dense.py::test_the_id_like_rule_for_whole_numbers` | 11 |
   | F28 | `test_no_choice.py::test_one_entity_of_nearly_all_training_rows_and_no_choice_is_made` | 9 |
   | M40 | `test_repeated_stamps.py::test_both_baselines_are_scored_at_repeated_timestamps` | 7 |
   | M41 | `test_screen_line.py::test_one_cell_far_off_in_the_copy_does_not_keep_it` | 6 |
   | M42 | `test_screen_set_aside.py::test_one_wild_target_cell_makes_no_column_a_leak` | 8 |
   | M43 | `test_order_reading.py::test_the_second_condition_keeps_the_lag_of_a_running_total_quiet` | 1 |
   | M44 | `test_order_reading.py::test_a_copy_with_some_cells_moved_is_named_under_time_order` | 11 |
   | M45 | `test_largest_row.py::test_every_configuration_has_its_share_over_the_selection_folds` | 4 |
   | M46 | `test_extreme_value.py::test_one_cell_far_outside_a_feature_is_named` | 4 |
   | M47 | `test_screen_line.py::test_a_copy_of_the_class_is_excluded_when_a_fold_holds_few_of_one_class` | 3 |
   | M48 | `test_no_choice.py::test_a_small_fold_that_trains_on_as_many_rows_as_it_validates_on_is_chosen_on` | 1 |
   | M49 | `test_fit_order.py::test_every_fit_is_handed_its_rows_in_time_order` | 5 |
   | M50 | `test_id_dense.py::test_the_id_like_rule_for_whole_numbers` | 3 |
   | M51 | `test_repeated_stamps.py::test_the_panel_warning_names_the_baseline_the_run_used` | 2 |
   | F22 | `test_guard_review7.py::test_f22_the_interval_is_the_same_at_any_number_of_cores` | 3 |
   | M33 | `test_guard_review7.py::test_m33_three_rows_alike_after_a_removed_leak_stop_a_regression` | 1 |
   | M34 | `test_guard_review7.py::test_m34_the_command_line_gives_the_full_profiles_own_error` | 1 |
   | M37 | `test_guard_review7.py::test_m37_the_file_says_what_group_measures` | 1 |
   | M39 | `test_guard_review7.py::test_m39_rows_with_a_dictionary_cell_are_counted_as_repeats` | 2 |

   And for every broken copy the suite caught before this pass, the number of
   cases that fail on it: in the suite as it was, run against the reference
   of this pass (where the seven cases of rule 3 fail on the good reference
   and so count for nothing); in the same files once rule 3 is applied; and
   with the new files of this pass:

   | Broken copy | Before | After rule 3, in the files that existed | With the new files |
   |---|---|---|---|
   | F1 | 29 | 29 | 34 |
   | F2 | 18 | 18 | 79 |
   | F3 | 24 | 24 | 28 |
   | F4 | 14 | 14 | 26 |
   | F5 | 58 | 58 | 67 |
   | F6 | 148 | 149 | 160 |
   | F7 | 212 | 218 | 283 |
   | F8 | 16 | 16 | 27 |
   | F9 | 17 | 17 | 17 |
   | F6b | 50 | 50 | 50 |
   | F6c | 19 | 19 | 28 |
   | F6d | 6 | 6 | 6 |
   | F7b | 34 | 34 | 34 |
   | F7c | 11 | 12 | 12 |
   | F10 | 40 | 40 | 52 |
   | F7d | 7 | 7 | 7 |
   | F8b | 6 | 6 | 6 |
   | F11 | 26 | 26 | 26 |
   | F12 | 6 | 6 | 7 |
   | F13 | 11 | 11 | 20 |
   | F14 | 47 | 47 | 47 |
   | F12b | 4 | 4 | 5 |
   | F14b | 37 | 37 | 37 |
   | F15 | 7 | 7 | 7 |
   | F16 | 13 | 13 | 13 |
   | F17 | 4 | 4 | 15 |
   | F18 | 17 | 17 | 17 |
   | F19 | 27 | 27 | 30 |
   | F20 | 4 | 4 | 4 |
   | F21 | 12 | 12 | 15 |
   | F22 | 1 | 1 | 4 |
   | M1 | 2 | 2 | 2 |
   | M2 | 2 | 2 | 2 |
   | M3 | 2 | 2 | 2 |
   | M4 | 3 | 3 | 3 |
   | M5 | 2 | 2 | 2 |
   | M6 | 2 | 2 | 2 |
   | M7 | 4 | 4 | 4 |
   | M8 | 2 | 2 | 2 |
   | M9 | 14 | 14 | 31 |
   | M10 | 2 | 2 | 2 |
   | M11 | 2 | 2 | 2 |
   | M12 | 3 | 3 | 3 |
   | M13 | 1 | 1 | 1 |
   | M14 | 4 | 4 | 4 |
   | M15 | 4 | 4 | 14 |
   | M16 | 7 | 7 | 8 |
   | M17 | 7 | 7 | 7 |
   | M18 | 6 | 6 | 6 |
   | M19 | 7 | 7 | 7 |
   | M20 | 3 | 3 | 3 |
   | M21 | 2 | 2 | 2 |
   | M22 | 17 | 17 | 19 |
   | M23 | 1 | 2 | 2 |
   | M24 | 2 | 2 | 2 |
   | M25 | 3 | 3 | 3 |
   | M26 | 5 | 5 | 5 |
   | M27 | 3 | 3 | 6 |
   | M28 | 4 | 4 | 4 |
   | M29 | 4 | 4 | 4 |
   | M30 | 2 | 2 | 2 |
   | M31 | 19 | 19 | 19 |
   | M32 | 8 | 8 | 8 |
   | M33 | 1 | 1 | 2 |
   | M34 | 1 | 1 | 2 |
   | M35 | 5 | 5 | 5 |
   | M36 | 2 | 2 | 2 |
   | M37 | 1 | 1 | 2 |
   | M38 | 2 | 2 | 2 |
   | M39 | 1 | 1 | 3 |

   You may run it yourself, from the repo root, with the project interpreter:
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/mutation_check.py --jobs 8`.
   It writes nothing inside the repo, and takes about an hour and a half on a
   machine doing nothing else. Do not read the reference implementation to
   tune a test to it. The README test of this pass goes in a file named
   `tests/test_readme_pass8.py`: files named `tests/test_readme*.py` are shown
   on a line of their own and not counted. **Neither
   `tests/test_judge_tools.py` nor `tests/test_judge_tools_pass8.py` is run by
   the mutation check:** each runs the judge's two other tools on the
   application, and a throwaway project has no `verify/`.

7. **A test of this pass says both things in every call, or is about a
   stop.** As in the last two passes, the new files call through
   `cp7_helpers.run`, `raised` and `run_cli`, which add no statement, and pass
   `time_col` and `group_col` in every call that is meant to finish
   (`**s.SAID` for "no" to both).

8. **Pin what the spec pins about a message, and no other wording.** Spec 5.4
   has one new fixed passage (the panel warning's) and spec 11 says what each
   new message holds: the text `holds the value X in N row` of
   `extreme_value`; for `largest_row_dominates` the word `holdout` or the
   words `selection folds`, a name, the row as a whole number and the share
   as a percentage; for `selection_not_made` the training rows of the
   smallest fold that meets its condition; for the order reading the share as
   a percentage. `cp9_helpers.holds_share` and `holds_row` read those. Which
   of the two usual estimators the panel warning's autocorrelation uses is
   open (spec 11): no test reads that number.

9. **Under the grouped scheme, which units are held out and which fold each
   is in is open beyond the properties of step 7.** The tests of T66 use
   tables on which the rule's two conditions hold, or fail, however the units
   are dealt, and read the counts from `validation.folds`. The order of rows
   handed over only to be predicted, and the order of the rows of a fit with
   no time column, are open too (spec 11): `test_fit_order.py` asserts the
   order of a fit under time order and nothing else about order.

10. **The rules of the earlier passes still hold**: the documented surface
    only (spec 2.1 to 2.7); recompute, do not trust: every rule of this pass
    that is a function of a few columns is written a second time in
    `cp9_helpers`, in plain arithmetic, and a test compares the application
    with that; assert on values as the application produced them; a share is
    compared with `math.isclose` or `pytest.approx`, never with `==`, but for
    the one case built to be exactly one half; **no skipped or
    expected-to-fail test**; nothing written outside `tmp_path`; the
    application imported inside test functions; the command line run one
    process at a time. Tiny tables: the tests of T66 need 2,000 rows, one
    guard 3,400; nothing else needs more than 700. Use a recording family or
    a lookup family through the seam wherever a run only has to finish: of
    the files of this pass only `test_no_choice.py` (three cases),
    `test_largest_row.py` (two) and the command-line tests fit a built-in
    model.

## Files of the eighth pass

Each file belongs to exactly one task of `plan/tasks.json`.

| File | Task | About | Today |
|---|---|---|---|
| `tests/test_id_dense.py` | T61 | the second clause of the id-like rule for whole numbers (D74 point 5) | red; the controls of D19 green |
| `tests/test_repeated_stamps.py` | T62 | the previous value at repeated timestamps, both baselines, `stages.previous_value`, the panel warning's passage (D74 point 1) | red, two tests green |
| `tests/test_screen_line.py` | T63 | the line beside the screen's tree, and the direct reading for a class (D74 point 2) | red under time order and for the new readings; green where the tree already excludes |
| `tests/test_screen_set_aside.py` | T63 | rows the screen sets aside, the 1% cap and the fall-back to every row (D74 point 2) | red, six cases green |
| `tests/test_order_reading.py` | T63 | the second reading of `strong_single_column`, its two conditions, and where it is quiet (D76 point 1; D36) | red where a warning is expected; the quiet cases green |
| `tests/test_largest_row.py` | T64 | the largest row's share on the holdout and over the selection folds, its null pattern, one record for each statement (D74 point 3) | red, one test green |
| `tests/test_extreme_value.py` | T64 | the `extreme_value` warning: the rule value by value, what its message holds, where it is quiet (D74 point 3, D76 point 3) | red; the quiet cases green |
| `tests/test_fit_order.py` | T65 | under time order the rows of a fit are in time order (D74 point 4) | red, one control green |
| `tests/test_no_choice.py` | T66 | no choice between configurations on too few rows (D74 point 6) | red, three controls green |
| `tests/test_readme_pass8.py` | T67 | the README on this pass | red |
| `tests/test_judge_tools_pass8.py` | T68 | both tools exit 0 on the application, strictly | red |
| `tests/test_guard_review7.py` | T68 | a second test, by another route, for F22, M33, M34, M37 and M39 | green |
| `tests/cp9_helpers.py` | none | shared helpers of the eighth pass; holds no test | |
| `tests/test_panel_stop.py`, `test_time_pipeline.py`, `test_whole_numbers.py`, `test_guard_review5.py`, `test_metrics.py` | (T44, T11, T33, T48, T7) | one replacement each, as rule 3 says; nothing else | green before and after |
| `tests/test_judge_tools.py` | (T60) | replaced as rule 3 (f) says | green before and after; red until T55 to T57 were built |

### `tests/cp9_helpers.py`

Import the earlier helper files and build on them; do not copy from them.
Nothing here makes a statement the call did not make, and nothing imports
`mlprofessor` at module level. `s` is `cp7_helpers`, `h` is `mlp_helpers`.

- **Pinned values**: `SCREEN_FAR` 10, `SCREEN_SET_ASIDE` 0.01, `ORDER_AGAINST`
  0.10, `ORDER_PAIRS` 20, `LARGEST_ROW_SHARE` 0.5, `EXTREME_GAP` 30,
  `EXTREME_SPREAD` 10, `EXTREME_OUTERMOST` 3, `SELECTION_FLOOR` 50 (spec 4.2
  and 4.3). `HEADLINE_LARGEST`: the four keys `largest_row_share`,
  `largest_row`, `baseline_largest_row_share`, `baseline_largest_row`;
  `MODEL_LARGEST`: the first two.
- **`PANEL_BASELINE`**: the second sentence of the fixed passage of spec 5.4,
  from `With no entity column` to `its own previous value.`
  **`panel_passage(baseline)`**: `Skill is measured against the baseline
  {baseline}. ` followed by `PANEL_BASELINE`.
- **`holds_share(text, share)`**: some number written in `text` is within 0.1
  of 100 x `share` (`s.holds_number`). **`holds_row(text, row)`**: `row` is
  among the whole numbers written in `text` (`cp8_helpers.whole_numbers_in`).
- **`BATTERY_OWNED`, `TABLES_OWNED`**: two frozen sets, of `"run:change"`
  texts and of table numbers: the pairs of the battery and the tables of the
  generated-table test (1,000 tables from its own seed) that fail on the
  application as the seventh pass left it. **Take them from
  `docs/measurements/CP9/owned_by_pass8.json`**, the report the lead's own
  runs of the two tools on the application wrote: the keys of
  `battery.pairs` and of `generated_tables.tables`. Do not shorten the lists
  and do not add to them.
- **Readers**: `column(result, name)`; `is_leak` and `is_feature` (role
  `feature` and reason null); `leaks(result)`, the sorted names of the
  columns excluded as `suspected_leak`; `told(result, code, name=None)`, the
  messages of the warnings with that code (about that column);
  `training_rows(result, frame)`, the row numbers not in the holdout.
- **`previous_by_hand(y, keys, gap=0)`**: for each row the mean of `y` over
  the rows at the timestamp `gap + 1` distinct timestamps before its own, or
  None; with dictionaries and plain sums.
- **`against_order(x, y)`**: the pairs `(x, y)` with neither missing, sorted
  (by `x`, then `y`); among neighbours whose `x` differ, the smaller of the
  share in which `y` falls and the share in which it rises; None with fewer
  than 20 such neighbours.
- **`far_outside(values)`**: the rule of `extreme_value` as spec step 11
  states it, value by value, over the distinct finite values: a list of
  `(value, rows that hold it, nearest value that is not far outside)`.
- **`largest_row(rows, truth, guess)`**: `(share, row)` from squared errors
  added with `math.fsum`; `(None, None)` when the sum is zero.
- **`walk(n=600, seed=0, per=1, noise=0.0, shuffle=False)`**: from
  `numpy.random.default_rng(seed)`, in this order: `level`, the cumulative
  sum of `n` normal draws; `n1` and `n2`, `n * per` normal draws each; `y`,
  `level` repeated `per` times plus `noise` times `n * per` normal draws.
  Columns `t` (0 to n - 1 as int64, each `per` times), `n1`, `n2`, `y`. With
  `shuffle`, the rows in the order of `rng.permutation`.
- **`with_column(frame, name, values, target="y")`**: a copy with the column
  put just before the target. **`rotated(values, share, seed=0, rows=None)`**:
  a float copy in which `max(2, round(share x len(rows)))` cells, drawn from
  `default_rng(seed).choice` among `rows` (all rows when None) without
  replacement, are moved round among themselves with `numpy.roll(.., 1)`.
- **`SCHEMES`** `("time", "time_repeated", "time_entities", "random",
  "grouped")` and **`scheme_table(scheme, seed)`**, a regression table with a
  noise column `n1` and no column that follows the target, with the
  statements of its call: `time`: `walk(n=600, seed=seed)`, `time_col="t"`,
  `group_col=False`; `time_repeated`: `walk(n=200, seed=seed, per=3,
  noise=0.5)`, the same statements; `time_entities`:
  `cp6_helpers.walks(n_series=6, n_times=100, seed=seed)` without its column
  `prev`, with `n1` (normal draws from `default_rng(seed)`) inserted as the
  third column, `time_col="t"`, `group_col="who"`; `random`:
  `h.reg_frame(n=600, seed=seed)` with `z1` renamed `n1`, `**s.SAID`;
  `grouped`: `s.sized_entities([5] * 120, seed=seed)` with `m1` renamed `n1`,
  `time_col=False`, `group_col="who"`.
- **`screened(frame, statements, **kw)`**: `s.run` with one recording family
  (`s.rec()`): what the screen and the warnings do is all such a test reads.
- **`dominant_entity(large=1490, small=255, per=2, seed=0)`**: `who` is
  `C0000` for `large` rows and `C0001`, ... for `small` entities of `per`
  rows; `n1` to `n4` and `y` are normal draws from `default_rng(seed)`, in
  that order; then the rows in the order of `rng.permutation`.

## A: the identifier (task T61)

### `tests/test_id_dense.py`

Spec: step 5 (rule 8b), 4.3, 7, 11. Roles are read with
`cp8_helpers.roles(frame)`, on a table of 300 rows with the column `code`,
a noise column and a noise target.

- `test_the_id_like_rule_for_whole_numbers`, eighteen cases. **F27, and M50**
  for the three cases with a repeat. Each case is a column of 300 cells and
  whether it is id-like (role `excluded`, reason `id_like`) or a numeric
  feature. *The first clause (D19), green today:* `in_file_order` (1 to 300);
  `consecutive_in_no_order` (the same, permuted); `rising_with_gaps` and
  `falling_with_gaps` (cumulative sums of whole steps of 1 to 49, up from
  1,000 and down from 90,000): all id-like. `a_measure_in_no_order` (such a
  rising column, permuted), `few_values` (whole numbers 0 to 10),
  `fractions` (halves, permuted): features. *The second clause:*
  `no_order_some_rows_gone` (300 of the 400 numbers from 100001, drawn
  without replacement), the same `_as_floats`, `no_order_beyond_2_to_the_53`
  (the same from 2**60, int64), `no_order_empty_cells` (the same as floats
  with ten empty cells), `no_order_one_row_repeated` (299 consecutive numbers
  and one of them again, permuted), `file_order_one_row_repeated` (the same,
  sorted), `286_distinct_of_300`: id-like. `285_distinct_of_300`: a feature
  (exactly 95% is not more than 95%). `fills_just_over_half_its_range` (the
  even numbers 2 to 598 and 600, permuted: 300 values in a range of 599) and
  `fills_half_its_range` (the same with 601: a range of 600): id-like;
  `fills_just_under_half_its_range` (with 602: a range of 601): a feature.
- `test_the_cases_are_what_their_names_say`: counts, from the columns alone,
  the shares the case names state.
- `test_the_rule_reads_the_rows_that_have_a_target`, cases `file_order` and
  `no_order`. **F27** on the second. An identifier 100001 to 100340 over 340
  rows of which 40 have no target; in file order and with the rows permuted;
  a run through `q.screened`: 40 rows dropped, the identifier `id_like`, the
  two noise columns features.
- `test_a_shuffled_identifier_reaches_no_model`. **F27.** A classification
  with such an identifier, a recording family and a recording step: no event
  names the column.
- `test_the_time_column_and_the_entity_column_keep_their_roles`: a dense
  whole-number time column and entity column have the roles `time` and
  `group`.

## B: the previous value at repeated timestamps (task T62)

### `tests/test_repeated_stamps.py`

Spec: 2.7, steps 8 and 11, 5.2, 5.4, 7, 11. Runs use a recording family and
`task="regression"`. `_sides(result, frame, gap, time)` gives the training
and the holdout rows from the holdout rows of the result and the timestamps
of the table; `_assert_both(result, frame, gap, time)` asserts: `baselines`
names exactly `training_mean` then `previous_value`; each score equals the
test's own (the training rows' mean; `q.previous_by_hand`) within 1e-9; the
one used is the lower, `headline.baseline` and `baseline_score` follow;
`headline.skill` is 1 - score / baseline score; `timestamps_unique` is
false. It returns the name of the baseline used.

- `test_the_step_function_gives_the_mean_of_the_timestamp_before`, gaps 0, 1
  and 2: twelve timestamps with 1 to 4 rows each, in permuted order:
  `stages.previous_value(y, keys, gap=gap)` equals `q.previous_by_hand` row
  for row, not-a-number where there is none.
- `test_the_step_function_is_unchanged_at_unique_keys`: six unique keys in
  no order, by hand, with `gap=0` and `gap=1`. Green today.
- `test_both_baselines_are_scored_at_repeated_timestamps`, cases
  `three_a_timestamp` (`q.walk(n=200, seed=8211, per=3, noise=0.5)`),
  `rows_shuffled` (the same with `shuffle=True`), `uneven` (the first with
  about 30% of its rows removed, so that timestamps hold one, two or three
  rows and some none) and `gap_2`. **F23, M40.** `_assert_both`, and the
  baseline used is `previous_value`.
- `test_one_repeated_row_leaves_the_previous_value_in_use`. **F23.**
  `q.walk(n=400, seed=8221)` with row 99 present twice: `_assert_both`, the
  previous value used, its score within 5% of the table's without the
  repeat, and the `duplicate_rows` warning.
- `test_the_training_mean_is_the_headline_baseline_where_its_error_is_lower`:
  `h.panel_frame(n_times=80, per_time=4, seed=8231, persistent=False)`:
  `_assert_both` gives `training_mean`; no panel warning.
- `test_unique_timestamps_are_scored_as_before`: `q.walk(n=300, seed=8241)`:
  the previous-value score by hand from the row before. Green today.
- `test_the_leak_screen_measures_a_lag_against_the_timestamp_before`.
  **F23.** `q.walk(n=400, seed=8251, per=3, noise=0.1)` with a column `lag`,
  the mean target of the timestamp before: a feature, with no
  `suspected_leak` and no `strong_single_column`.
- `test_the_panel_warning_names_the_baseline_the_run_used`, cases
  `persistent` (`h.panel_frame(n_times=80, per_time=4, seed=8261,
  persistent=True)`; the previous value is used) and `late_noise` (120
  timestamps of four rows: a walk for the first 90, then levels drawn afresh
  round the walk's mean with a spread of 2; the training mean is used).
  **M51** on the first. Exactly one `autocorrelated_panel_target` warning,
  about the target; its message holds `q.panel_passage(baseline)` and does
  not hold the passage with the other baseline's name.
- `test_the_command_line_reports_both_baselines`: the first table through
  the command line: both baselines in `profile.json`, one used, and the
  panel warning's passage.

## C: the leak screen (task T63)

### `tests/test_screen_line.py`

Spec: step 8, 4.3, 7. Every run is `q.screened`. `_assert_copy_is_out`: the
column has role `excluded`, reason `suspected_leak`, leak rule
`predicts_target`; one `suspected_leak` warning and no
`strong_single_column` for it; `n1` is a feature; no other column is a leak.

- `test_a_copy_of_the_target_is_excluded_whatever_its_scale`, the five
  schemes of `q.SCHEMES` times the copies `exact`, `rescaled` (3.7 y + 12),
  `negative` (-2 y + 5) and `scale_1e-9` (1e-9 y), on
  `q.scheme_table(scheme, seed=8301)`. **F24** under the three time-ordered
  schemes and at 1e-9.
- `test_one_cell_far_off_in_the_copy_does_not_keep_it`, the five schemes
  times the cells 999999 and -999, seed 8311: the rescaled copy with one
  cell replaced, in the forty-first training row of the clean run. **M41**
  under the three time-ordered schemes.
- `test_a_copy_with_empty_cells_is_excluded`, the five schemes, seed 8321:
  two cells of the copy empty.
- `test_columns_that_do_not_follow_the_target_stay_features`, the five
  schemes, seed 8331: no leak and no `strong_single_column`. Green today.
- `test_a_driver_known_before_the_outcome_is_kept_under_time_order`: a walk
  of 400 steps as the feature and the target the walk plus noise of 0.5: a
  feature, not a leak. Green today.
- `test_a_copy_of_the_class_is_excluded_when_a_fold_holds_few_of_one_class`,
  cases `number`, `number_reversed` (5 - 2 y) and `text` (`closed`, `open`).
  **M47.** 600 unique timestamps and a 0/1 target of which the first 100
  rows hold two 1s (rows 20 and 60): the copy is out.
- `test_a_column_that_drives_the_class_is_not_excluded`, cases `random` and
  `time`: `h.clf_frame(n=600, seed=8361)`: no leak. Green today.

### `tests/test_screen_set_aside.py`

Spec: step 8, 4.3, 7. `_middle(frame, statements, training, n_cells)` picks
training rows from the middle of the training rows, seven apart (under time
order the middle in time, at timestamps seven apart, so that each is among
the validation rows of a fold). `_with_wild_targets(frame, statements, n)`
adds the column `copy` (3.7 y + 12) and then sets the target of `n` such
rows to 999999: the copy keeps what the target was.

- `test_one_wild_target_cell_does_not_hide_a_copy`, the five schemes, seed
  8401. **F25.** The copy is the only leak; `n1` a feature.
- `test_one_wild_target_cell_makes_no_column_a_leak`, the five schemes, seed
  8411. **M42** under `random` and `grouped`. No copy in the table: no leak,
  no `strong_single_column`, every role as in the clean run.
- `test_a_wild_target_cell_in_the_holdout_changes_nothing_in_the_screen`.
- `test_at_most_one_percent_of_the_training_rows_is_set_aside`, cases (600
  rows, 4 cells, out), (600, 5, kept), (120, 1, out), (120, 2, kept), on
  `h.reg_frame(n=.., seed=8431)`: 450 training rows allow four and 90 allow
  one. Out: the copy is the only leak. Kept: no leak, the copy a feature,
  and one `strong_single_column` for it (the order reading). **F25** on the
  cases that are out, **M44** on those kept.
- `test_under_time_order_the_row_after_a_wild_cell_counts_too`, cases (2,
  out) and (3, kept), on `q.walk(n=600, seed=8441)`: each wild cell is in a
  validation block.
- `test_a_long_tail_of_the_target_is_scored_as_it_is`: 600 rows of which
  about 8% have their target multiplied by 500; the test counts that more
  than 1% of the training rows are far off; the copy is the only leak.

### `tests/test_order_reading.py`

Spec: step 8, 2.5, 4.3, 7, 11. `_shares(result, frame, name)` gives, by hand
over the training rows of a run at unique timestamps,
`q.against_order(column, y)` and the same for the row before.
`_assert_named_and_kept`: a feature, exactly one `strong_single_column`, no
`suspected_leak`.

- `test_a_copy_with_some_cells_moved_is_named_under_time_order`, seeds 8501
  to 8503. **M44.** `q.walk(n=600, seed)` with 3.7 y + 12 of which 4% of the
  first 450 cells are moved (`q.rotated`): the share is above 0 and at most
  0.10, the previous value's above 0.10; named and kept; the message holds
  the share as a percentage.
- `test_a_curve_of_the_target_is_named_under_time_order`, cases
  `exponential` (exp(y / 4)) and `cube`: the share is 0; the column is
  excluded, or named and kept (rule 2 may see through a mild curve).
- `test_the_first_condition_is_a_share_of_at_most_one_tenth`. **M44.** Seven
  copies with 5% to 16% of their cells moved: none excluded; each is named
  exactly when its share by hand is at most 0.10; both outcomes occur.
- `test_the_second_condition_keeps_the_lag_of_a_running_total_quiet`.
  **M43.** A running total of exponential steps, its own lag and a counter
  of the rows (0.37 k + 1.3, so no identifier): both shares are 0, for the
  column and for the previous value; neither is named. Green today.
- `test_with_no_previous_value_the_second_condition_has_nothing_to_ask`.
  **M44.** Under the random scheme, the copy with ten training targets set
  to 999999: named and kept.
- `test_fewer_than_twenty_pairs_give_no_reading`: three levels of the
  target. Green today.
- `test_a_strong_driver_known_before_the_outcome_is_not_named`, 100, 200 and
  400 rows times a true R-squared of 0.90, 0.95 and 0.99: a walk `x` and
  `y = x + noise` scaled to that share: the share is above 0.10, `x` is a
  feature and nothing is a leak; at 0.90 and 0.95 no `strong_single_column`
  (at 0.99 the first reading may name it). Green today.
- `test_a_column_warned_for_its_skill_gets_one_record`: under the random
  scheme, the target with 2.5% of its training cells moved: one record.
  Green today.
- `test_a_column_whose_skill_cannot_be_computed_is_named_by_neither_reading`
  (D36). The table of
  `test_skill_undefined.py::test_single_column_skill_that_cannot_be_computed_flags_nothing`:
  the share of `lead` against the order is 0, and it is a feature with no
  `suspected_leak` and no `strong_single_column`. Green today; no variant
  fails it (a guard of the reference's own condition).
- `test_a_classification_has_no_order_reading`, seeds 1 and 2:
  `h.clf_frame(n=600, seed, signal=5.0)`: the share of `x1` is at most 0.10
  and nothing is named. Green today.
- `test_a_categorical_column_has_no_order_reading`: twenty-four bands of the
  target as text. Green today.

## D: two statements about the result (task T64)

### `tests/test_largest_row.py`

Spec: steps 9 and 11, 2.5, 5.2, 5.3, 11. `_table()` is `h.reg_frame(n=400,
seed=8701)` with the target of its first six rows empty, so that a row's
number is not its place among the rows split. `_two_families(a, b)` makes two
lookup families on one log, `a-0` and `b-0`; `_guesses(frame, seed, spread)`
is the target plus normal noise. `_folds(log, family, n_train)` gives the
rows each selection fold was fitted on and predicted.

- `test_the_keys_are_in_every_part`, cases `regression` and
  `classification`, with the built-in families and an `out_dir`: the four
  keys in `headline` and in `cv`, the two in every record of `models` and in
  the columns of `ProfileResult.models`, in the object and in the file; all
  null for the classification and no warning; all numbers for the
  regression. **M45** on the regression.
- `test_the_holdout_shares_are_those_of_the_predictions_file`. **F26.** For
  the model from `predictions`, for the baseline from the training rows'
  mean (`q.largest_row`).
- `test_under_time_order_the_baseline_is_the_one_the_headline_used`.
  **F26.** `q.walk(n=400, seed=8721)`: against the previous value, on the
  holdout and over the selection folds (rows 50 to 299).
- `test_every_configuration_has_its_share_over_the_selection_folds`.
  **M45.** Each record of `models` by hand from the rows its folds
  predicted; `cv` holds the chosen one's and the baseline's (each fold's own
  training mean).
- `test_a_configuration_that_failed_has_no_share`.
- `test_a_sum_of_zero_has_no_largest_row`: a model that is exact; and a
  baseline that is exact on a flat holdout, where skill is null (D36).
- `test_no_warning_without_a_row_that_dominates`. Green today.
- `test_one_holdout_row_the_model_misses_is_one_record`. **F26.** One
  prediction 500 off: one record; its message holds `holdout`, `a-0`, the
  row and the share.
- `test_a_wild_holdout_target_is_a_record_for_the_model_and_one_for_the_baseline`.
- `test_a_wild_training_target_is_one_record_for_the_configurations_and_one_for_the_baseline`.
  **M45.** Two records, both holding `selection folds` and the row; one
  holds `training_mean` and its share.
- `test_exactly_one_half_is_not_above_one_half`: two holdout targets set to
  1 and 2, predicted 9 and 10, every other row exact: a share of one half
  and no record.
- `test_the_statements_leave_nothing_out_of_the_score`.

### `tests/test_extreme_value.py`

Spec: step 11, 2.5, 4.3, 7, 11. `_table()`: 300 rows, `a` about 50 with a
spread of 10, `b` about 0, a 0/1 flag and a noise target, from seed 8601.
`_named(result, name)` reads every `extreme_value` record with the pattern
`holds the value (\S+) in (\d+) row`; `_assert_as_by_hand` compares the
values named, the rows and the nearest value with `q.far_outside` over the
rows that have a target, and asserts the text `holds the value {value!r} in
{rows} row`.

- `test_one_cell_far_outside_a_feature_is_named`, `a` and `b` times -999 and
  999999. **M46** on `a` with -999.
- `test_the_target_of_a_regression_is_looked_at`.
- `test_a_value_counts_once_however_many_rows_hold_it`: four cells. **M46.**
- `test_the_gap_must_be_more_than_thirty_times_the_spread`, cases 310 (not
  named; green today) and 311 (named), above a column of the whole numbers 0
  to 10.
- `test_the_three_outermost_values_at_each_end_are_looked_at`, cases
  `two_values`, `three_values` (0, 1, 1000: named), `three_values_near` (0,
  1, 31: not), `two_beyond` and `three_beyond` (-999, -998 (, -997) below 20
  to 39: all named), `four_beyond` (none named: the fourth from an end is not
  looked at), `both_ends`. **M46** on `both_ends`. Green today where nothing
  is named.
- `test_only_rows_that_are_split_are_looked_at`: a far value in a row with
  no target is not named, and such a row is not counted.
- `test_columns_that_reach_no_model_are_not_looked_at`: an identifier, a
  column excluded as a leak (with one cell of 999999), a text column, and a
  classification target of 0 and 999999.
- `test_an_infinite_cell_is_no_extreme_value`. Green today.
- `test_a_long_tail_is_not_named`, six shapes (normal, log-normal with
  sigma 1 and 1.5, a traded volume, exponential, a count) at 200 and 2,000
  rows, from `default_rng([8641, n, len(shape)])`. Green today.
- `test_a_clean_table_has_no_such_warning_under_any_scheme`. Green today.
- `test_the_warning_changes_no_number`: the far value is in the matrix every
  model is handed, and the split is that of the table without it.
- `test_the_command_line_writes_the_warning`. **M46.**

## E: the order of the rows of a fit (task T65)

### `tests/test_fit_order.py`

Spec: 2.6, steps 9 and 10, 7, 11.

- `test_every_fit_is_handed_its_rows_in_time_order`, cases
  `unique_timestamps` (`q.walk(n=300, seed=8801, shuffle=True)`),
  `repeated_timestamps` (`q.walk(n=100, seed=8802, per=3, noise=0.5,
  shuffle=True)`) and `entities` (`cp6_helpers.walks(n_series=6, n_times=60,
  seed=8803)` with its entity column declared). **M49.** A recording family
  and a recording step: every fit of either is handed its rows by timestamp,
  rows of one timestamp by their place in the file, with the target in the
  same order; six model fits.
- `test_rows_of_one_timestamp_keep_their_file_order`. **M49.**
- `test_the_order_of_the_file_changes_no_number_under_time_order`. **M49.**
  A family that is fitted on the first third of the rows it is handed, on a
  table in time order and on the same rows permuted: the same headline
  score, the same fold scores, the same prediction for every timestamp.
- `test_with_no_time_column_the_run_is_as_it_was`. Green today.

## F: no choice on too few rows (task T66)

### `tests/test_no_choice.py`

Spec: step 9, 2.5, 4.2, 5.3, 7, 10.10, 11. `_families(frame, *spreads)` makes
one lookup family for each spread, `a`, `b`, ..., each predicting the target
plus noise of that spread; `_starved(result)` reads from `validation.folds`
the folds with fewer than 50 training rows and fewer than validation rows.

- `test_one_entity_of_nearly_all_training_rows_and_no_choice_is_made`, seeds
  0 to 2, on `q.dominant_entity(seed)` with spreads 3.0 and 0.1. **F28.** One
  such fold, validating on at least 1,490 rows; every fold scored and every
  record with five fold scores; `b-0` has by far the better `cv_mean` and
  `a-0` is the headline's and `cv.chosen`; one `selection_not_made` warning,
  whose message holds that fold's training rows.
- `test_the_first_configuration_that_fits_is_used`. **F28.** `a` raises on
  the final fit: `b-0` is used, not `c-0` with the best fold scores.
- `test_a_configuration_with_no_fold_scores_is_not_the_first`. **F28.** A
  family that fails in every fold comes first: `a-0` is used.
- `test_rows_that_agree_in_every_feature_are_units_of_the_rule_too`.
  **F28.** With `duplicates="group"` and 1,490 rows alike in every feature.
- `test_a_small_fold_that_trains_on_as_many_rows_as_it_validates_on_is_chosen_on`.
  **M48.** `s.sized_entities([30] + [1] * 74, seed=9021)`: 78 training rows;
  the fold that validates on the entity of 30 trains on 39 to 48 rows and
  validates on 30 to 39, however the units are dealt; no warning, and `b-0`
  is chosen. Green today.
- `test_an_ordinary_table_of_entities_is_chosen_on_as_before`. Green today.
- `test_the_rule_is_for_the_grouped_scheme_only`: under time order with 13
  timestamps of one row and then 77 of five, the first fold trains on fewer
  rows than it validates on, and the choice is made; and under the random
  scheme on 104 rows. Green today.
- `test_the_dominant_shape_is_reported_at_about_zero`, seeds 0 to 2, with the
  built-in families on `q.dominant_entity(seed=100 + seed)`. **F28.** One
  warning, the first configuration, and skill inside -0.03 to 0.02.

## The README (task T67)

### `tests/test_readme_pass8.py`

Spec: 1. It judges `README.md`, which the builder owns; do not edit it. It
imports `test_readme_examples`, as the README tests before it do. Pin no
wording beyond these.

- `test_the_readme_names_what_the_eighth_pass_added`: the text holds
  `largest_row_share`, `largest_row_dominates`, `extreme_value` and
  `selection_not_made`, and names no path under `verify/` or `tests/`.
- `test_the_readme_says_what_the_baseline_is_at_repeated_timestamps`: with
  white space joined, it holds `timestamp before` and no longer holds `only
  when timestamps are unique` or `when timestamps are unique, so`.
- `test_the_readme_states_the_limits_of_the_eighth_pass`: in lower case it
  holds `strong_single_column`, `noise` and `sorted`.

The fourteen existing README tests judge the README as it is and stay green
through T61 to T66, which do not touch it; T67 must keep them green.

## The judge's other tools and the guards (task T68)

### `tests/test_judge_tools_pass8.py`

Spec: 8.2. The strict form of `tests/test_judge_tools.py`. It imports that file
and holds two tests, `test_the_perturbation_battery_exits_0` and
`test_the_generated_tables_exit_0`: each asks `test_judge_tools.run_tool` for
the tool's one run of the session (`BATTERY`, `TABLES`) and asserts exit code
0, showing the last 40 lines the tool printed otherwise. Not run by the
mutation check.

Each tool has a time limit of 3,600 seconds, which only keeps a hung script
from holding the suite for ever. Measured on 2026-11-25: the battery, now
1,229 pairs, runs 32 side by side by default and takes 274 to 296 seconds on
a machine doing nothing else (at 16 side by side it took 716 seconds, and
1,313 to 1,465 beside other runs); the generated tables take about two
minutes, and
the rest of the suite with the files of this pass about eight to ten. That
is why the two files share one run of each tool. The loop's own limit for
the whole suite is the one that binds, and the suite must run with nothing
else loading the machine.

### `tests/test_guard_review7.py`

Spec: 8 and 8.1, steps 6, 7, 8a and 12, 2.2, 2.5, 5.4. All green today.

- `test_f22_the_interval_is_the_same_at_any_number_of_cores`, cases
  `bootstrap_rows` (a classification), `bootstrap_groups` (200 entities of
  three rows) and `bootstrap_blocks` (a walk of 320 timestamps), each with a
  lookup family under the fast profile at `n_jobs` 1 and 3: the interval,
  the interval on skill and the p-value are equal, and the two documents
  differ in `meta.n_jobs` and the timing only. **F22.**
- `test_m33_three_rows_alike_after_a_removed_leak_stop_a_regression`. **M33.**
  400 rows, no entity column, a numeric target and a rescaled copy of it;
  three holdout rows of `stages.split_random(400, None, 0)` are given the
  measures of three training rows: under `duplicates="group"` the run stops
  with `holdout_copies_in_training`, the message names the column and holds
  3, and no model was handed a row.
- `test_m34_the_command_line_gives_the_full_profiles_own_error`, cases `fast`
  and `full`. **M34** on `fast`. Entities of 2,500, 300, 300 and 300 rows
  through the command line: exit 3 and `too_few_groups`.
- `test_m37_the_file_says_what_group_measures`. **M37.** Through the command
  line, the `same_features_grouped` warning in `profile.json` holds
  `cp8_helpers.GROUP_MEASURES`.
- `test_m39_rows_with_a_dictionary_cell_are_counted_as_repeats`, cases
  `dictionary` and `tuple_with_a_list`. **M39.** A regression in which
  twelve rows repeat: `input.duplicate_rows` is 12.

## What must fail today, and what must pass today

Checked on 2026-11-25 with prototypes of every test of this section and of
the six loosenings of rule 3, run against the application as it stands,
against the reference implementation of this pass, against the reference as
it was (the loosened files), and against the reference carrying each of its
eighty-eight faults. The prototypes were written outside the repo and are
not kept in it: they are the proof that the recipes work, not text to copy.
What they printed is in `docs/measurements/CP9/`.

**Red today.** They fail against the application until the build stage has
done D74 and D76, and pass against the reference.

| File | Tests |
|---|---|
| `test_id_dense.py` | `test_a_shuffled_identifier_reaches_no_model`; `test_the_id_like_rule_for_whole_numbers`: `286_distinct_of_300`, `file_order_one_row_repeated`, `fills_half_its_range`, `fills_just_over_half_its_range`, `no_order_beyond_2_to_the_53`, `no_order_empty_cells`, `no_order_one_row_repeated`, `no_order_some_rows_gone`, `no_order_some_rows_gone_as_floats`; `test_the_rule_reads_the_rows_that_have_a_target`: `no_order`. |
| `test_repeated_stamps.py` | `test_both_baselines_are_scored_at_repeated_timestamps` (4 cases); `test_one_repeated_row_leaves_the_previous_value_in_use`; `test_the_command_line_reports_both_baselines`; `test_the_leak_screen_measures_a_lag_against_the_timestamp_before`; `test_the_panel_warning_names_the_baseline_the_run_used` (2 cases); `test_the_step_function_gives_the_mean_of_the_timestamp_before` (3 cases); `test_the_training_mean_is_the_headline_baseline_where_its_error_is_lower`. |
| `test_screen_line.py` | `test_a_copy_of_the_class_is_excluded_when_a_fold_holds_few_of_one_class` (3 cases); `test_a_copy_of_the_target_is_excluded_whatever_its_scale`: `grouped-scale_1e-9`, `random-scale_1e-9`, `time-exact`, `time-negative`, `time-rescaled`, `time-scale_1e-9`, `time_entities-exact`, `time_entities-negative`, `time_entities-rescaled`, `time_entities-scale_1e-9`, `time_repeated-exact`, `time_repeated-negative`, `time_repeated-rescaled`, `time_repeated-scale_1e-9`; `test_a_copy_with_empty_cells_is_excluded`: `time`, `time_entities`, `time_repeated`; `test_one_cell_far_off_in_the_copy_does_not_keep_it`: `grouped--999.0`, `random--999.0`, `time--999.0`, `time-999999.0`, `time_entities--999.0`, `time_entities-999999.0`, `time_repeated--999.0`, `time_repeated-999999.0`. |
| `test_screen_set_aside.py` | `test_a_long_tail_of_the_target_is_scored_as_it_is`; `test_at_most_one_percent_of_the_training_rows_is_set_aside` (4 cases); `test_one_wild_target_cell_does_not_hide_a_copy` (5 cases); `test_under_time_order_the_row_after_a_wild_cell_counts_too` (2 cases). |
| `test_order_reading.py` | `test_a_copy_with_some_cells_moved_is_named_under_time_order` (3 cases); `test_a_curve_of_the_target_is_named_under_time_order` (2 cases); `test_the_first_condition_is_a_share_of_at_most_one_tenth`; `test_with_no_previous_value_the_second_condition_has_nothing_to_ask`. |
| `test_largest_row.py` | `test_a_configuration_that_failed_has_no_share`; `test_a_sum_of_zero_has_no_largest_row`; `test_a_wild_holdout_target_is_a_record_for_the_model_and_one_for_the_baseline`; `test_a_wild_training_target_is_one_record_for_the_configurations_and_one_for_the_baseline`; `test_every_configuration_has_its_share_over_the_selection_folds`; `test_exactly_one_half_is_not_above_one_half`; `test_one_holdout_row_the_model_misses_is_one_record`; `test_the_holdout_shares_are_those_of_the_predictions_file`; `test_the_keys_are_in_every_part` (2 cases); `test_the_statements_leave_nothing_out_of_the_score`; `test_under_time_order_the_baseline_is_the_one_the_headline_used`. |
| `test_extreme_value.py` | `test_a_value_counts_once_however_many_rows_hold_it`; `test_columns_that_reach_no_model_are_not_looked_at`; `test_one_cell_far_outside_a_feature_is_named` (4 cases); `test_only_rows_that_are_split_are_looked_at`; `test_the_command_line_writes_the_warning`; `test_the_gap_must_be_more_than_thirty_times_the_spread`: `311.0-True`; `test_the_target_of_a_regression_is_looked_at`; `test_the_three_outermost_values_at_each_end_are_looked_at`: `both_ends`, `three_beyond`, `three_values`, `two_beyond`; `test_the_warning_changes_no_number`. |
| `test_fit_order.py` | `test_every_fit_is_handed_its_rows_in_time_order` (3 cases); `test_rows_of_one_timestamp_keep_their_file_order`; `test_the_order_of_the_file_changes_no_number_under_time_order`. |
| `test_no_choice.py` | `test_a_configuration_with_no_fold_scores_is_not_the_first`; `test_one_entity_of_nearly_all_training_rows_and_no_choice_is_made` (3 cases); `test_rows_that_agree_in_every_feature_are_units_of_the_rule_too`; `test_the_dominant_shape_is_reported_at_about_zero` (3 cases); `test_the_first_configuration_that_fits_is_used`. |
| `test_readme_pass8.py` | `test_the_readme_names_what_the_eighth_pass_added`; `test_the_readme_says_what_the_baseline_is_at_repeated_timestamps`; `test_the_readme_states_the_limits_of_the_eighth_pass`. Red against the reference too, until T67 is built: it judges `README.md` |
| `test_judge_tools_pass8.py` | both tests, until T61 to T66 are built: each tool fails on the application until then. Not run against the reference by the mutation check |

**Green today.** They pass against the application and against the
reference, and must go on passing through every task of the build.

| File | Test | Fails on |
|---|---|---|
| `test_id_dense.py` | `test_the_cases_are_what_their_names_say` |  |
| `test_id_dense.py` | `test_the_id_like_rule_for_whole_numbers`, 9 of its 18 cases: `285_distinct_of_300`, `a_measure_in_no_order`, `consecutive_in_no_order`, `falling_with_gaps`, `few_values`, `fills_just_under_half_its_range`, `fractions`, `in_file_order`, `rising_with_gaps` |  |
| `test_id_dense.py` | `test_the_rule_reads_the_rows_that_have_a_target`, 1 of its 2 cases: `file_order` | F6 |
| `test_id_dense.py` | `test_the_time_column_and_the_entity_column_keep_their_roles` |  |
| `test_repeated_stamps.py` | `test_the_step_function_is_unchanged_at_unique_keys` |  |
| `test_repeated_stamps.py` | `test_unique_timestamps_are_scored_as_before` | F7, F8 |
| `test_screen_line.py` | `test_a_column_that_drives_the_class_is_not_excluded`, 2 cases: `random`, `time` | F4, F5 |
| `test_screen_line.py` | `test_a_copy_of_the_target_is_excluded_whatever_its_scale`, 6 of its 20 cases: `grouped-exact`, `grouped-negative`, `grouped-rescaled`, `random-exact`, `random-negative`, `random-rescaled` | F2 |
| `test_screen_line.py` | `test_a_copy_with_empty_cells_is_excluded`, 2 of its 5 cases: `grouped`, `random` | F2 |
| `test_screen_line.py` | `test_a_driver_known_before_the_outcome_is_kept_under_time_order` | F7, M9 |
| `test_screen_line.py` | `test_columns_that_do_not_follow_the_target_stay_features`, 5 cases: `grouped`, `random`, `time`, `time_entities`, `time_repeated` | F7 |
| `test_screen_line.py` | `test_one_cell_far_off_in_the_copy_does_not_keep_it`, 2 of its 10 cases: `grouped-999999.0`, `random-999999.0` | F2, F25 |
| `test_screen_set_aside.py` | `test_a_wild_target_cell_in_the_holdout_changes_nothing_in_the_screen` | F2 |
| `test_screen_set_aside.py` | `test_one_wild_target_cell_makes_no_column_a_leak`, 5 cases: `grouped`, `random`, `time`, `time_entities`, `time_repeated` | F7, M42 |
| `test_order_reading.py` | `test_a_categorical_column_has_no_order_reading` | F7 |
| `test_order_reading.py` | `test_a_classification_has_no_order_reading`, 2 cases: `1`, `2` | F5 |
| `test_order_reading.py` | `test_a_column_warned_for_its_skill_gets_one_record` | F2 |
| `test_order_reading.py` | `test_a_column_whose_skill_cannot_be_computed_is_named_by_neither_reading` | F7, M9 |
| `test_order_reading.py` | `test_a_strong_driver_known_before_the_outcome_is_not_named`, 9 cases: `100-0.95`, `100-0.99`, `100-0.9`, `200-0.95`, `200-0.99`, `200-0.9`, `400-0.95`, `400-0.99`, `400-0.9` | F7, M9 |
| `test_order_reading.py` | `test_fewer_than_twenty_pairs_give_no_reading` | F7 |
| `test_order_reading.py` | `test_the_second_condition_keeps_the_lag_of_a_running_total_quiet` | F6, F7, M9, M43 |
| `test_largest_row.py` | `test_no_warning_without_a_row_that_dominates` |  |
| `test_extreme_value.py` | `test_a_clean_table_has_no_such_warning_under_any_scheme` | F7 |
| `test_extreme_value.py` | `test_a_long_tail_is_not_named`, 12 cases: `count-2000`, `count-200`, `exponential-2000`, `exponential-200`, `log_normal_1-2000`, `log_normal_1-200`, `log_normal_1.5-2000`, `log_normal_1.5-200`, `normal-2000`, `normal-200`, `volume-2000`, `volume-200` |  |
| `test_extreme_value.py` | `test_an_infinite_cell_is_no_extreme_value` |  |
| `test_extreme_value.py` | `test_the_gap_must_be_more_than_thirty_times_the_spread`, 1 of its 2 cases: `310.0-False` |  |
| `test_extreme_value.py` | `test_the_three_outermost_values_at_each_end_are_looked_at`, 3 of its 7 cases: `four_beyond`, `three_values_near`, `two_values` |  |
| `test_fit_order.py` | `test_with_no_time_column_the_run_is_as_it_was` | F1 |
| `test_no_choice.py` | `test_a_small_fold_that_trains_on_as_many_rows_as_it_validates_on_is_chosen_on` | F6c, F10, F17, M15, M48 |
| `test_no_choice.py` | `test_an_ordinary_table_of_entities_is_chosen_on_as_before` | F17 |
| `test_no_choice.py` | `test_the_rule_is_for_the_grouped_scheme_only` | F7, F17 |
| `test_guard_review7.py` | `test_f22_the_interval_is_the_same_at_any_number_of_cores`, 3 cases: `bootstrap_blocks`, `bootstrap_groups`, `bootstrap_rows` | F7, F21, F22 |
| `test_guard_review7.py` | `test_m33_three_rows_alike_after_a_removed_leak_stop_a_regression` | F2, F12b, M33 |
| `test_guard_review7.py` | `test_m34_the_command_line_gives_the_full_profiles_own_error`, 2 cases: `fast`, `full` | F10, M34 |
| `test_guard_review7.py` | `test_m37_the_file_says_what_group_measures` | F10, M22, M37 |
| `test_guard_review7.py` | `test_m39_rows_with_a_dictionary_cell_are_counted_as_repeats`, 2 cases: `dictionary`, `tuple_with_a_list` | M39 |
| `test_panel_stop.py` | every test, 5 cases, the loosened one too | not a guard where it was loosened: it passes before the change and after it |
| `test_time_pipeline.py` | every test, 15 cases, the loosened one too | not a guard where it was loosened: it passes before the change and after it |
| `test_whole_numbers.py` | every test, 19 cases, the loosened one too | not a guard where it was loosened: it passes before the change and after it |
| `test_guard_review5.py` | every test, 3 cases, the loosened one too | not a guard where it was loosened: it passes before the change and after it |
| `test_metrics.py` | every test, 15 cases, the loosened one too | not a guard where it was loosened: it passes before the change and after it |
| `test_judge_tools.py` | both tests, as loosened: each tool fails on the application on the pairs and tables of the fixed lists only | not run by the mutation check |

Measured with the prototypes: 203 cases in the twelve new test files: 23 in
`test_id_dense.py`, 15 in `test_repeated_stamps.py`, 46 in
`test_screen_line.py`, 18 in `test_screen_set_aside.py`, 23 in
`test_order_reading.py`, 13 in `test_largest_row.py`, 33 in
`test_extreme_value.py`, 6 in `test_fit_order.py`, 12 in `test_no_choice.py`, 9
in `test_guard_review7.py`, 3 in `test_readme_pass8.py`, and 2 in
`test_judge_tools_pass8.py`. Against the application as it stands, of the first
201, 86 pass and 115 fail. Against the reference 198 pass and 3 fail, the
README's. With the five replacements in place, all 57 cases of the five files
pass against the application as it stands, against the reference of this pass
and against the reference as it was.

**What the mutation check must then print**, run after your stage and before
the build: the good reference fails 0 of 2119 (the 1921 counted today and 198
of the new files); each of the eighty-eight variants fails at least one test,
and each of the twenty-three of rule 6 the test named for it; on the README
line, 3 failing of 17, the tests of `test_readme_pass8.py`; and a note that
`tests/test_judge_tools.py` and `tests/test_judge_tools_pass8.py` are not run
there. **Before your stage it ends in `FAIL` for two reasons, both expected:**
the good reference fails 7 cases, of
`test_guard_review5.py::test_m23_the_panel_warning_starts_just_above_one_half`,
`test_metrics.py::test_f8_previous_value_refuses_repeated_keys`,
`test_panel_stop.py::test_the_statement_lets_the_run_go_ahead_as_before`,
`test_time_pipeline.py::test_panel_uses_training_mean_and_flags_persistence`,
`test_whole_numbers.py::test_missing_and_repeated_integer_stamps` (the
reference has the new baseline and those tests are not yet loosened); and it
prints `NOT CAUGHT` for 18 of the eighty-eight variants, F23, F24, F25, F26,
F27, F28, M40, M41, M42, M43, M44, M45, M46, M47, M48, M49, M50, M51, which
have no test yet. Your stage closes both.

## Not tested in the eighth pass

The wording of any message beyond what spec 5.4 and 11 pin (rule 8), the
value of the autocorrelation the panel warning writes, and which of its two
usual estimators is used. Which units a grouped split holds out and which
fold each is in (rule 9). The order of the rows of a fit with no time
column, and of rows handed over only to be predicted or transformed. The
order of the `extreme_value` records of one column, and whether a value that
both ends of a column name gets one record or two. A copy of the target with
noise under time order: at 1% of the target's spread it is excluded or
warned about on the judge's own tables (the battery asserts that), and
beyond that it is a stated limit (D76; spec 11). A whole-number measure in a
table sorted by it, which D31 leaves id-like. The column-role override of
D76 point 5, which is not part of this pass, and D75. A forest under time
order by its own numbers: the seam test of `test_fit_order.py` holds the
order, and the battery holds the headline. What the judge's two tools find
on a variant of the reference: the lead runs them by hand.
