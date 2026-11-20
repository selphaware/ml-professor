# ML Professor (phase 1)

A Python package, `mlprofessor`, that measures how predictable one target
column of one table is. The contract is `spec.md`. The decisions behind it are
`docs/DECISIONS.md` (D1 to D48). The build order is `plan/tasks.json`; what the
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
2. Decide the task by a stated rule; split off a 25% holdout first: at random, or the latest whole timestamps when `time_col` is given.
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

## Output

`profile(...)` returns a `ProfileResult`; with an output folder it writes
`profile.json` and `predictions.csv`. Full schema: `spec.md` section 5.

| Part of `profile.json` | Holds | On `ProfileResult` |
|---|---|---|
| `meta` | tool, schema version, profile, seed, `n_jobs`, library versions, `timing` | `to_dict()` |
| `input` | source, kind, target, time column, task requested, gap, rows, columns, duplicate rows | `to_dict()` |
| `rows` | seven counts: loaded, four `dropped_` reasons, training, holdout | `.rows` |
| `columns` | one record per input column: role, kind, reason, leak rule, four counts (`n_infinite` also counts feature values too large for the tree models, above 3.4028234663852886e+38: D48) | `.columns` |
| `task` | type, decided by rule or declared, distinct values, classes, positive class (a text target is read without the white space round it: D46) | `to_dict()` |
| `validation` | scheme, holdout share, folds, gap, time form, last training and first holdout timestamp, caveat | `to_dict()` |
| `headline` | metric, score, model, baseline and its score, skill (null when the baseline's error is zero), the interval on the score and on skill, the resamples used for each, p-value, reasons | `.headline` |
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

F6b, F6c, F6d and F7b are the other forms of F6 and F7, each with a broken
copy of the reference of its own (D46). **Every check in this table is seen to
fail on the broken copy of the mode it names**, in every run of the gate's
self-test (`verify/smoke_test.py --selftest`, eight to ten minutes), which
prints the table that proves it and fails if one pair is missing.

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

A guard that fails means the application is wrong, not the test. Three more
tests of that kind hold what was decided and the application already does:
surrounding white space is not part of a categorical level
(`test_guard_final_fit.py`), a header cell of only white space is a name
like any other (`test_csv_edges.py`), and a line pandas reads as the header
is never skipped, before a header of two columns as before one of three
(`test_guard_header_lines.py`, which fails on F6d).

The mutation check runs the whole suite against the reference and against
each of nineteen variants of it: the twelve of the table above and M1 to M7.
The good reference must fail nothing, and every variant at least one test.

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
- `verify/reference/` is the judge's own fixture, with nineteen deliberate
  faults wired into it (twelve the gate catches, seven only the tests do). It
  is not a starting point: do not copy from it.
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
