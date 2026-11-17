# Test plan: what the suite must prove (phase 1)

> **If `mlprofessor/` already exists at the repo root, this is the second
> pass. Go straight to "Second pass (CP3)" at the end of this file and read
> its ground rules before anything else. They override your role prompt and
> the first pass's rules wherever they differ.**

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
