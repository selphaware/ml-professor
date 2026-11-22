# ML Professor (phase 1)

A Python package, `mlprofessor`, that measures how predictable one target
column of one table is. The contract is `spec.md`. The decisions behind it are
`docs/DECISIONS.md` (D1 to D66). The build order is `plan/tasks.json`; what the
tests prove is `plan/test-plan.md`. The method and the table of common failures
are in the skill `mlprofessor-build`.

## Definition of done

Both of these exit 0, run from `E:\CODE\ml-professor`:

```
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/smoke_test.py mlprofessor
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -m pytest -c tests/pytest.ini --rootdir . --confcutdir tests -p no:cacheprovider tests
```

One gate check alone: add `--only <check id>`. The per-edit run adds `--fast`,
which leaves out the two slow checks; they run when you finish. A gate failure
names the dataset, the value and the likely cause; the files it read are in
`verify/artifacts/runs/<run>/` and the data in `verify/artifacts/data/`. That
folder always holds the last gate run on the build: the gate's own self-test
writes to `verify/artifacts/selftest_runs/<fixture>/` instead (D46).

## The methodology in five lines

1. Count every row and column; drop only rows with no target (or no time value), and say how many.
2. Decide the task by a stated rule; split off a 25% holdout first: at random, in whole entities when `group_col` names a column, or the latest whole timestamps when `time_col` names one. The caller says which, in every call (D63).
3. Choose a model by five folds inside the training rows only, with every preprocessing step refitted inside each fold.
4. Refit the chosen one on the training rows and score it once on the holdout, as skill over a baseline (ROC AUC against 0.5; squared error against the training mean or the previous value). If it raises there, the next best by cross-validation takes its place: the holdout never takes part in the choice.
5. Report it with an interval and a p-value from the holdout predictions, and with every exclusion, warning and limitation. Where a baseline's error is exactly zero, skill has no value and is reported as null, with the reason; the score keeps its interval.

**The headline is computed only on rows no fitted step has seen.**

**No number that is not finite is ever written, or turned into null (D44).** A
null means "does not apply" or "cannot be computed, for the reason stated
beside it", and nothing else. A squared error overflows above about 1e154, so:
the baselines are scored on the holdout before anything is fitted, and one
that is not finite stops the run with `non_finite_score`; a configuration
whose prediction or fold score is not finite is failed, like one that raised;
a holdout score that is not finite stops the run, with no fallback. Spec
section 3, "Numbers that are not finite", has the seven rules in order.

**A holdout row whose copy is among the training rows has been seen in all but
name (D49, D53).** Under the random scheme the tool counts the holdout rows
that have a copy among the training rows: the same value in every feature
column and the same target, compared as a seam step is handed them. When more
than 1% of the holdout rows, and at least 10 of them, have one, the run stops
with `holdout_copies_in_training`, before any model is fitted, unless the
caller passed `duplicates="keep"` (`--duplicates keep`); then it runs, with a
`holdout_copies_kept` warning. The count and the share are always reported.
Under time order the rule does not apply. Spec step 8a. A third value,
`duplicates="group"` (D57, D62), keeps together the rows that agree in every
feature column, whatever their target: no combination of feature values is on
both sides of the split or in two folds. The target plays no part in forming
those sets, and the count of step 8a still reads it: a copy is equal in every
feature and the target. The message of the stop says when each answer fits,
`"keep"` first (spec 5.4). **A copy is an exact copy.** One that differs in any
column (an added column, a batch number, a value changed by 1e-9) is not
detected in this version; the broader screen was measured and withdrawn (D60;
`docs/measurements/CP6/`). The caller's control is `group_col`.

**Every call says two things, and the tool assumes neither (D63).** Whether
the rows are ordered in time: `time_col` is a column's name or `False`
(`--time-col COLUMN` or `--no-time-order`). Whether rows belong together:
`group_col` is a column's name or `False` (`--group-col COLUMN` or
`--no-entities`). `None`, the default of each, means "not said", and a call
that leaves either unsaid stops at step 5 with a named error, whatever the
table holds: `time_order_undeclared` when time order is unsaid, with a message
that asks for both when both are; `entities_undeclared` when only the entities
are. On the command line that is exit code 3, never 2. It comes after an
invalid argument and after every error of steps 1 to 4, and before every later
one. **A run never goes ahead or stops because of a date form or an
autocorrelation:** those are warnings beside the caller's own statement
(`date_column_undeclared` under `time_col=False`,
`autocorrelated_panel_target` under `group_col=False`), as they were. The
step functions of `mlprofessor.stages` take what they are given.

**Dates written as whole numbers are dates (D66).** A column in which every
value is a whole number that reads as a calendar date written `YYYYMMDD`,
years 1900 to 2100, with at least two distinct values (an integer column, a
float column with gaps, or text of eight digits) is excluded with the reason
`datetime`: rule 2a of step 5, asked before the id-like rules. It is named by
`date_column_undeclared` under `time_col=False`, and accepted as the time
column, ordered as numbers. Day counts and `YYYYMM` are not seen: a stated
limit (spec 11).

**A declared entity column keeps the rows of one entity together (D57).** With
no time column: the grouped scheme, whole entities on one side of the holdout
and in one fold; under the fast profile whole entities under the cap; the
interval from resampled entities (`bootstrap_groups`, none under 30 holdout
entities) and no p-value. Under time order the split is by time as ever, and
each entity has its own previous value as a baseline, in the headline and in
the leak screen. Which rows land where is open; the five properties of spec
step 7 are the contract.

**The grouped holdout holds a quarter and no more than it must (D64, D66).** A
unit joins the holdout while the holdout stays under a quarter of the rows;
the smallest unit passed over completes it; every unit it can spare goes back
to training. So it holds at least a quarter, needs every unit it has, and
holds less than a quarter plus one unit, the smallest such that the units no
larger than it hold a quarter (property 3 of spec step 7; for classification,
for the units of each class). A unit of a quarter or more of the rows it is
drawn among is named: `large_group_kept_in_training` when smaller units fill
the quarter, so the headline never tests it; `large_group_is_holdout` when
they cannot, and that one unit is the holdout.

**No input ends in an error that is not named (D50 to D52).** Whatever type a
column has and whatever it holds, in any role, the run ends in a result or in
a subclass of `ProfileError`. Three rules carry most of it (spec steps 1a, 3
and 5): an object column that holds only numbers (`Decimal`, booleans or whole
numbers with `None`) is read as the typed column it stands for; whole numbers
are compared exactly, as time keys above 2**53 and when distinct values are
counted; whole numbers too large for 64 bits are excluded with a warning as a
spare column and are a named error as the target or the time column. Every
column excluded as `unsupported_type` gets a warning naming it. Since the
fifth pass (D57): each object column is read once, over all loaded rows, so
an odd cell in a row that is dropped for another column's empty cell gives
the same named error as in a usable row; an infinite float in an object
target is a missing target; and no `ProfileResult` can be made from a
document that holds a float that is not finite, in any part of it.

## Output

`profile(...)` returns a `ProfileResult`; with an output folder it writes
`profile.json` and `predictions.csv`. Full schema: `spec.md` section 5.

| Part of `profile.json` | Holds | On `ProfileResult` |
|---|---|---|
| `meta` | tool, schema version, profile, seed, `n_jobs`, library versions, `timing` | `to_dict()` |
| `input` | source, kind, target, `time_col` as passed (a name or `false`, never null: D63), task requested, gap, rows, columns, duplicate rows, the `duplicates` argument (D49), `group_col` as passed (a name or `false`, never null: D63) | `to_dict()` |
| `rows` | seven counts: loaded, four `dropped_` reasons, training, holdout | `.rows` |
| `columns` | one record per input column: role (`target`, `time`, `group`, `feature`, `excluded`), kind, reason, leak rule, four counts (`n_infinite` also counts feature values too large for the tree models, above 3.4028234663852886e+38: D48; and the infinite cells of the target and of the time column: D57). `dtype` is the type as loaded: `object` for an object column read as numbers (D51) | `.columns` |
| `task` | type, decided by rule or declared, distinct values, classes, positive class (a text target is read without the white space round it: D46; a class value is written as a boolean, a number or text: D53) | `to_dict()` |
| `validation` | scheme (`random`, `grouped`, `time_ordered`), holdout share, folds, gap, time form, last training and first holdout timestamp (an exact whole number for integer keys above 2**53: D50), caveat, and with no time column the holdout rows with a copy among the training rows and their share (D49); the entity column, the entities on each side, the holdout rows with no previous value, and under `duplicates="group"` the rows that agree with another in every feature, `n_rows_same_features` (D62, D66) | `to_dict()` |
| `headline` | metric, score, model, baseline and its score, skill (null when the baseline's error is zero), the interval on the score and on skill (`bootstrap_rows`, `bootstrap_groups` or `bootstrap_blocks`), the resamples used for each, p-value (random scheme only), reasons | `.headline` |
| `baselines` | each baseline scored on the holdout, and which one was used | `to_dict()` |
| `cv` | metric, folds scored, configurations, the chosen one, its mean and spread | `to_dict()` |
| `models` | one record per configuration: params, status, error, fold scores, mean, spread, chosen. One that fails after it was chosen keeps its fold scores | `.models` |
| `warnings` | records of code, column, message | `.warnings` |
| `limitations` | nine fixed sentences | `to_dict()` |
| `predictions.csv` | `row`, `y_true`, `y_pred` for every holdout row | `.predictions` |

## Failure modes and the checks that catch them

| Failure mode | Description | Caught by |
|---|---|---|
| F1 | the chosen model is refitted on all rows before the holdout is scored | model.noise_clf, model.noise_p_value, model.noise_reg |
| F2 | the leak screen is removed, so a copy of the target is a feature | leak.excluded, leak.regression, leak.near_leak_warned |
| F3 | the headline is the best cross-validation score, not the holdout score | model.headline_recomputed, model.selection_full |
| F4 | the headline metric is accuracy | model.headline_recomputed, model.metric_clf |
| F5 | the task is decided from the column's type alone | task.binary_numeric, task.ambiguous_stops |
| F6 | rows with any missing value are dropped before counting | data.rows_reconcile, data.coercion_reported |
| F7 | the split is shuffled even when a time column is declared | time.order_respected, time.no_skill_on_blocks, time.gap_applied |
| F8 | the regression baseline is always the training mean | time.persistence_baseline |
| F6b | a column is lost from the accounting (constant columns are dropped with no record) | data.columns_accounted |
| F6c | a model family is lost from the models table (only the chosen configuration's family is listed) | models.all_accounted |
| F6d | data rows longer than the header are read silently, with no error | data.long_rows_stop |
| F7b | a day-first time column is parsed leniently, so the run goes ahead on the wrong order | time.dayfirst_stops |
| F7c | whole-number time keys are read as floats, so distinct instants above 2**53 become one timestamp and the previous-value baseline is lost (D50) | time.integer_keys_distinct |
| F9 | holdout rows with copies among the training rows are scored with no error (D49) | data.holdout_copies_stop |
| F10 | an entity has rows on both sides of the split although the entity column was declared (D57) | groups.kept_together, groups.no_skill_on_entities |
| F7d | a table with a date column and nothing said about time order is split at random, where it should stop (D58) | time.undeclared_stops |
| F8b | a panel is scored against the training mean although the entity column was declared (D57) | time.panel_previous_value |
| F11 | a statement the caller did not make is assumed: nothing said about time order, or about entities, and the run goes ahead (D63) | statements.time_order_asked, statements.entities_asked |
| F12 | `duplicates="group"` forms its sets with the target, so a held-out set has only its opposite class in training (D62) | duplicates.group_on_features, duplicates.group_no_skill |
| F13 | the grouped holdout takes far more than a quarter because a large entity was drawn early (D64) | groups.holdout_bounded |
| F14 | dates written as whole numbers, `YYYYMMDD`, are a numeric feature (D66) | time.date_numbers_seen |

F6b, F6c, F6d, F7b, F7c, F7d and F8b are the other forms of F6, F7 and F8,
each with a broken copy of the reference of its own (D46, D50, D57, D58).
**Every check in this table is seen to fail on the broken copy of the mode it
names**, in every run of the gate's self-test (`verify/smoke_test.py
--selftest`, about eighteen minutes with twenty-one broken copies), which
prints the table that proves it and fails if one pair is missing.

Six checks came with the sixth pass. `statements.time_order_asked` runs
`signal_clf` with nothing said at all: exit 3, `time_order_undeclared`, all
four flags in the message, no headline. `statements.entities_asked` runs
`random_walk` with `--time-col date` and nothing said about entities: exit 3,
`entities_undeclared`. `duplicates.group_on_features` runs noise in a region
of four levels and a branch of sixteen with `--duplicates group`, and counts
from `predictions.csv` and the source that no combination of the two is on
both sides; `duplicates.group_no_skill` asserts the band on that run.
`groups.holdout_bounded` runs one entity of 1,490 rows beside 255 of two, and
holds the holdout to property 3 by the gate's own count: 500 or 501 rows.
`time.date_numbers_seen` runs a label that is random per block of five days
with the day as a whole number: the column is excluded as a date and the
warning names it. **Every run of the gate says both things:** each older run
gained the statements it lacked and nothing else (D63), but for
`time_blocks_unstated`, which still says nothing about time order.

Four checks came with the fifth pass. `time.undeclared_stops` runs
`time_blocks` with nothing said about time order: exit 3, the error's code, no
headline. `time.panel_previous_value` runs twelve random walks with
`--time-col date --group-col series`: both baselines as the gate recomputes
them, each series against its own previous value, and skill about zero.
`groups.kept_together` runs 300 entities of five rows with `--group-col
customer` and counts, from `predictions.csv` and the source, that no entity is
on both sides; `groups.no_skill_on_entities` asserts the band on that run. Two
older runs changed: `messy` and `time_blocks` with no time column each pass
`--no-time-order` (D58), so the four checks that read them belong to T42.

Two checks came with the fourth pass and assert no score.
`data.holdout_copies_stop` runs noise with every row present twice: exit 3,
the error's code, no headline. `time.integer_keys_distinct` runs a random walk
whose time column is 64-bit integer nanoseconds: the counts of timestamps equal
the gate's own, the stamps are written exactly, and the baseline is the
previous value. One older run changed: `time.undeclared_warns` passes
`--duplicates keep`, because with its date column excluded every holdout row
of that table has a copy among the training rows (D53).

Ten gate checks name no failure mode and say why they exist instead
(`--list` shows them). One is `time.flat_holdout_reported` (D36), on a target
that is flat across the holdout, where skill has no value and must be reported
as null.

The gate catches the large forms. The small forms are caught by the test
suite, each in the way that can see it:

- A model, or a step supplied through the seam, that is handed a holdout row
  or a row of its own fold's validation part: exactly, by the recording fakes,
  through `mlprofessor.extensions` (spec 2.6).
- The built-in imputation, scaling and one-hot encoding, and the leak screen.
  The fakes cannot see inside these, so they are caught by effect, by the
  guard tests below. Each guard is proved by running it against a copy of the
  reference implementation that carries that one fault
  (`verify/mutation_check.py`, which the owner runs after the tests stage).

| Small fault | Description | Caught by (`tests/`) |
|---|---|---|
| M1 | the final fit's imputation and scaling statistics are learned from training and holdout rows together | `test_guard_final_fit.py::test_m1_final_fit_statistics_ignore_the_holdout` |
| M2 | one-hot levels are learned from every row of the table in every fit | `test_guard_final_fit.py::test_m2_levels_come_from_the_rows_of_each_fit` |
| M3 | the leak screen's trees are fitted on rows that include the holdout, and the missing-by-class rule counts every row | `test_guard_leak_screen.py::test_m3_single_column_skill_is_fitted_on_fold_training_rows_only`, `::test_m3_missing_by_class_counts_training_rows_only` |
| M4 | the block bootstrap draws timestamps independently and still reports `bootstrap_blocks` | `test_guard_uncertainty.py::test_m4_block_bootstrap_draws_whole_blocks` |
| M5 | the permutation p-value counts shuffles strictly greater than the observed skill | `test_guard_uncertainty.py::test_m5_ties_count_in_the_p_value` (classification only: D43) |
| M6 | a prediction that is not finite is scored and written as it is, where it should fail its configuration | `test_non_finite.py::test_a_prediction_that_is_not_finite_counts_as_raising` |
| M7 | the fast profile's own error is never raised: too few timestamps under the row cap is always `too_few_rows` | `test_fast_profile_error.py::test_fast_error_stands_when_the_full_profile_would_run`, `test_time_pipeline.py::test_wide_panel_under_fast_raises` |
| M8 | the interval is the 5th to the 95th percentile of the resamples, under a level that still says 0.95 | `test_guard_review3.py::test_m8_the_interval_is_the_middle_95_percent` |
| M9 | the leak screen forgets the previous-value baseline, so the lag of a persistent series is excluded as a leak | `test_guard_review3.py::test_m9_the_leak_screen_uses_the_previous_value` |
| M10 | the fast profile keeps the first 2,000 usable rows, not a sample from the seed | `test_guard_review3.py::test_m10_the_fast_cap_is_a_sample_from_the_seed` |
| M11 | `cv_std` is the population deviation of the fold scores, not the sample deviation | `test_guard_review3.py::test_m11_cv_std_is_the_sample_deviation` |
| M12 | in a resampled ROC AUC, tied predictions do not share their rank | `test_guard_review3.py::test_m12_tied_predictions_in_a_resample` |
| M13 | `n_imputed` is counted over every loaded row | `test_guard_review3.py::test_m13_n_imputed_counts_rows_that_reach_a_model` |
| M14 | an object column is read a second time over the usable rows only, so an odd cell in a dropped row is never seen | `test_read_once.py`, its first four tests |
| M15 | with an entity column the holdout is whole entities but the folds are cut through them | `test_groups_random.py::test_no_entity_on_both_sides_or_in_two_folds` |
| M16 | under time order with an entity column the leak screen forgets each entity's previous value | `test_groups_time.py::test_the_leak_screen_measures_a_lag_against_the_entity_baseline` |
| M17 | with an entity column the fast cap keeps a sample of rows, whatever their entity | `test_groups_random.py::test_the_fast_cap_keeps_whole_entities` |
| M18 | the check that no number is infinite or not-a-number covers five parts of the output, not every part | `test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite` |
| M19 | a panel with nothing said about entities goes ahead with a warning, where it should stop; since D63, under a time column nothing said about entities is read as `False` | `test_panel_stop.py::test_a_persistent_panel_with_no_entity_column_stops`, `test_statements.py::test_nothing_said_about_entities_stops_whatever_the_table_holds` |
| M20 | the infinite cells of the target and of the time column are left out of `n_infinite` | `test_guard_review4.py::test_m20_infinite_cells_of_the_target_and_the_time_column_are_counted` |
| M21 | under the grouped scheme the bootstrap draws rows one by one and still reports `bootstrap_groups` | `test_groups_random.py::test_the_interval_resamples_whole_entities_and_there_is_no_p_value` |
| M22 | with `duplicates="group"` the copies are found and counted, and the split is made row by row | `test_duplicates_group.py::test_each_set_of_copies_stays_on_one_side_and_in_one_fold` |
| M23 | the cut-off of the panel warning is 0.8, not 0.5 | `test_guard_review5.py::test_m23_the_panel_warning_starts_just_above_one_half` |
| M24 | with both statements unsaid the message asks for time order only | `test_statements.py::test_one_message_asks_for_everything_that_is_missing` |
| M25 | with `duplicates="group"` the sets are formed on the features and the count still reads the target | `test_duplicates_features.py::test_the_count_is_of_rows_that_agree_in_every_feature` |
| M26 | the grouped holdout is completed by the smallest unit passed over, and nothing it can spare goes back | `test_grouped_holdout.py::test_what_the_holdout_can_spare_goes_back` |
| M27 | a unit of a quarter or more kept in training gets no warning | `test_grouped_holdout.py::test_a_large_unit_kept_in_training_is_named` |
| M28 | dates written as eight digits of text are refused as the time column | `test_date_numbers.py::test_eight_digit_text_is_accepted_as_the_time_column` |
| M29 | the id-like rule is asked before the date-number rule | `test_date_numbers.py::test_the_date_rule_comes_before_the_id_like_rule` |
| M30 | the count of step 8a is made on the features alone, whatever the target | `test_guard_review5.py::test_m30_a_copy_at_step_8a_still_needs_the_same_target` |

A guard that fails means the application is wrong, not the test. Three more
tests of that kind hold what was decided and the application already does:
surrounding white space is not part of a categorical level
(`test_guard_final_fit.py`), a header cell of only white space is a name
like any other (`test_csv_edges.py`), and a line pandas reads as the header
is never skipped, before a header of two columns as before one of three
(`test_guard_header_lines.py`, which fails on F6d).

The mutation check runs the whole suite against the reference and against
each of fifty-one variants of it: the twenty-one of the table above and M1 to
M30. The good reference must fail nothing, and every variant at least one test.

**How an existing test follows a changed contract (D59, D65).** The tests of
the first five passes were written when a call could leave both statements
unsaid. `mlp_helpers.stated_profile`, `mlp_helpers.run_cli` and
`cp5_helpers.raised` make the call as the test wrote it, and make it again
with the missing statement only when the run stops for one. A test that pins
the stop itself opts out (`as_called=True`, `as_given=True`), and the tests of
the sixth pass use `cp7_helpers`, which adds nothing. So a failure in an older
test after a change to the stop is a real one: the helper never hides another
error.

**The type matrix (D52).** `tests/test_type_matrix.py` runs `profile()` on
about 800 small tables: every kind of pandas column (each integer and float
width, nullable types, booleans, text, categories, bytes, dates, durations,
periods, intervals, complex numbers, `Decimal` and other objects, containers,
all-missing and sparse columns, and CSV), as a spare column, as the target and
as the time column, with ordinary values and with extreme ones (the ends of
the type, whole numbers above 2**53 and beyond 64 bits, 1e300, infinities,
not-a-number, negative zero, one value throughout). Each case must end in a
result that keeps the reconciliation rules or in a named error. A raw
`TypeError`, `ValueError` or `OverflowError` is a failure of the case, and the
remedy is at the place it starts, never a broad `except` round the run. Spec
section 7, "The type matrix". `tests/test_type_matrix_group.py` (D57) does the
same for a fourth role, the entity column: 353 tables.

## Environment (measured 2026-11-15; interpreter base and process starts 2026-11-16)

- Interpreter: `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe`, Python
  3.12.13. Use this path for everything. Bare `python` on this machine is the
  Microsoft Store stub and does not run. Never create another environment.
- Libraries, pinned in `requirements.txt`: pandas 3.1.1, numpy 2.5.4,
  scikit-learn 1.9.1, threadpoolctl 3.8.0 (with scipy 1.18.1, joblib 1.6.0;
  pytest 9.2.0 for the suite). No other modelling library. A score that moves
  after an upgrade is traced through `meta.versions`.
- pandas 3: text columns have the type `str`, not `object`; copy-on-write is
  always on, so arrays taken from a frame can be read-only. See spec section 9
  for how `read_csv` loads each shape.
- Machine: Threadripper PRO 5975WX, 32 cores, 64 threads, 192 GB. No GPU.
- Seeds: one `seed`, a whole number from 0 to 4,294,967,295 (D45), drives the
  cap, the split, the folds, every model and every resample. The same seed
  gives the same files apart from `meta.timing`.
- Number ranges (measured 2026-11-20, `docs/measurements/CP4`): the square of
  a value above about 1.34e154 is infinite, and `numpy.mean` and `numpy.std`
  overflow sooner than their answers do; scikit-learn's trees and forests read
  32-bit floats and raise on a value above 3.4028234663852886e+38. Hence D44
  and D48.
- Whole numbers (measured 2026-11-20, `docs/measurements/CP5`): a 64-bit
  float holds every whole number only up to 2**53. Above it, distinct
  integers become one float: 1,500 nanosecond stamps 100 apart at 1.7e18 are
  587 floats. `read_csv` loads a whole number beyond 64 bits as a Python
  integer in an `object` column, on which `numpy.isinf` raises `TypeError`; a
  whole-number column with an empty cell loads as floats, already rounded.
  `pandas.isna` raises on `Decimal("sNaN")`. Hence D50 and D51.
- Threads: `n_jobs=-1` (every core) is the default. BLAS is held to one thread
  always and OpenMP to `n_jobs`, with `threadpoolctl`; forests predict on one
  thread. The gate runs the tool with `--n-jobs 1` and the thread variables
  set to 1.
- The interpreter's base is `C:\Users\uahma\anaconda3\envs\py312`, a local
  path (`home` in `mlprof_env\pyvenv.cfg`, D32; confirmed 2026-11-16 with
  `sys.base_prefix`). If the interpreter does not start, that is a harness
  error, not a code fault.
- Until D32 the base was on `A:`, a drive mapped through a loopback network
  share, and Python processes started at the same instant blocked each other
  for exactly 35 or 70 seconds. Measured again on 2026-11-16 with the local
  path: eight processes started at the same instant, 30 rounds, twice; none
  blocked, and each run took about 3 seconds. The gate still spaces its own
  runs 0.5 seconds apart and retries once. Running Python processes one after
  another is still the simplest habit, but it is no longer a rule.
- Windows, PowerShell 7. The project is on a local disk.
- The built library, measured once on 2026-11-17 (`docs/measurements/CP3`): a
  full run on 100,000 rows by 100 columns on all cores takes 174 seconds for
  classification and 202 for regression, against a budget of 600; and a full
  run on 5,000 by 30 writes identical files at `n_jobs` 1 and 64.

## Code conventions

- Modular code: one concern per module (`load`, `columns`, `task`, `split`,
  `metrics`, `preprocess`, `models`, `leakage`, `uncertainty`, `pipeline`,
  `result`, `cli`), with `stages.py` and `extensions.py` as thin facades.
- PEP 8. Short docstrings rather than long ones. Names that say which rows a
  thing holds (`train_rows`, `holdout_rows`), never just `data`.
- Dependencies pinned in `requirements.txt`.

## Guardrails

- No secrets in code.
- Do not edit the harness: `verify/`, `plan/`, `spec.md`, `CLAUDE.md`,
  `tests/`, `loop/`, `.claude/`. If a check, a test or the spec looks wrong,
  stop and say so plainly, with the evidence. Never widen a band.
- `verify/reference/` is the judge's own fixture, with fifty-one deliberate
  faults wired into it (twenty-one the gate catches, thirty only the tests
  are relied on for). It is not a starting point: do not copy from it.
- Application code never mentions the gate, the tests or the loop.
- No git. The human owns every git operation.
- Write only inside `mlprofessor/` and `examples/`, and `requirements.txt` and
  `README.md`. `examples/` holds the small table the README's commands use;
  the README names no file under `verify/` or `tests/`.
- No network at run time.
- A new dependency must be pinned in `requirements.txt`; phase 1 needs none
  beyond the four above.
- The package is never installed into the interpreter. It runs from the repo
  root: `python -m mlprofessor ...`.

## Autonomy

Work through gate and test failures on your own: read the message, open the
files it names, fix the code, run the check again. Stop and report only if the
same failure repeats three times after three different fixes, or an external
dependency is unreachable (the interpreter does not start, a pinned library is
missing).
