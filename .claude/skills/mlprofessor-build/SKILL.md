---
name: mlprofessor-build
description: Build or repair a tabular profiling library of the mlprofessor kind (one table, one target, a holdout headline with leak, time-order and row accounting guarantees). Use when implementing any step of spec.md, when a gate check or a test fails, or when a score looks too good.
---

# Building a tabular profiling library that cannot fool its user

The product is one number, skill on a holdout, and the accounting that makes it
believable. Every bug that matters here produces a plausible number and no
error. Build in the order below. Each step has a way to prove it before the
next one starts.

The contract is `spec.md`. The judge is `verify/smoke_test.py` and `tests/`.
This file is the method.

## Order of work

Steps 2 to 7 each deliver one function of `mlprofessor.stages` (spec 2.7) and
are judged by that step's own test file, before anything runs end to end.
`profile()` must then call those same functions, never a second copy.

1. **Skeleton.** Package folder, `__main__.py`, the command line with every
   option, `requirements.txt` with the four pinned libraries. Nothing is
   installed into the venv; the package runs from the repo root.
2. **Load and count.** One loader for a CSV path and a DataFrame. Count rows
   before anything is dropped. Detect duplicate header names on the raw header
   line, because pandas renames them silently. `row` is the position in the
   loaded table and never changes afterwards.
3. **Column roles that need no target.** Decide them on the usable rows, before
   the fast profile's row cap. Keep the counts (missing, coerced, infinite) on
   every loaded row. Every source column gets exactly one record.
4. **Task.** Two distinct values is classification whatever the type. Decide it
   before the row cap. Never guess: raise the named error.
5. **Split.** Random: 25% holdout first, then five folds inside the rest. Time
   order: sort by the time column, cut at a whole timestamp, six blocks, gap
   before the holdout and before every validation block. Write the split as a
   pure function from keys to row positions and test it on tiny tables.
6. **Metrics and baselines.** ROC AUC by ranks with ties averaged; squared
   error; skill; the previous-value baseline. Test against values worked out by
   hand.
7. **Preprocessing as an object with `fit` and `transform`.** It is created
   fresh for every fit and only ever sees that fit's training rows.
8. **Fit, select, score once.** One loop over configurations and folds. The
   holdout is touched once, by the chosen configuration, after selection.
9. **End to end.** Result object, `profile.json`, `predictions.csv`. From here
   the gate can judge. Run it after every edit.
10. **Time-ordered scheme wired in**, then the **leak screen**, then the
    **interval and p-value**, then **forest and boosting**, then the **full
    profile's grids**, then **edges and warnings**, then the **README**.

### Second pass, after the first review (tasks T18 to T23)

The package exists. Change as little as each task needs.

11. **Guards first (T18).** Tests that pass today and must go on passing: the
    built-in preprocessing, the leak screen, the block bootstrap and the
    p-value each proved against a faulty copy of the reference. Expect to
    change nothing. A guard that fails means the application is wrong.
12. **Skill where it has no value (T19)**, then **colliding feature names
    (T20)**, then **a configuration that fails after it is chosen (T21)**, then
    the **small defects (T22)**, then the **README's examples (T23)**.

## Rules that keep the number honest

- The holdout rows are chosen before anything is fitted, and nothing fitted
  ever receives one: not the model, not the imputer, scaler or encoder, not the
  leak screen. If a helper takes "the data", ask which rows.
- Selection uses cross-validation scores. The headline uses the holdout score.
  They are different variables with different names.
- A count in the output is a count somebody can redo from the source file.
- One seed drives everything. Linear algebra runs on one thread always;
  forests predict on one thread. Without both, the same seed gives different
  bits on different runs or at different `n_jobs`.
- When a rule cannot decide, stop with the named error. Never fall back.
- A number that cannot be computed is null with a reason beside it. Skill
  against a baseline whose error is exactly zero is the case in point: test
  for the zero before dividing, and never put an error, an infinity or a zero
  in its place. The score keeps its interval; count what was left out, per
  interval. Two warning codes, never together: `skill_undefined` when the
  headline skill has no value, `no_skill_interval` when it has a value and
  its interval has none.
- A categorical level is the cell's text without the white space round it,
  everywhere a level is used: the counts, the leak screen, what a step
  receives, the matrix. A header cell is blank only when it is empty; one that
  holds white space is a name like any other.
- The holdout never chooses. If the chosen configuration raises on the final
  fit or on the holdout, the next best by `cv_mean` takes its place; a holdout
  score is never a reason to pass one over, and a failed one keeps the fold
  scores it earned.
- Every feature listed in the output reaches the models under a name of its
  own. Two that would share a name stop the run.
- Leave the caller's process as you found it: put back any process-wide
  setting in a `finally`.

## Common failures and their real causes

| What you see | The real cause | Where to look |
|---|---|---|
| `model.noise_clf` or `model.noise_reg` fails with skill well above the band | The chosen configuration was refitted on training plus holdout rows before scoring (F1). | The refit after selection: which row positions does it receive? |
| A recording-fake test reports that a step's `fit` saw holdout rows | The preprocessing chain was fitted before the split, or once for all folds (F1, small form). No band can see this; the fake sees it for a model and for a step it supplied. | Where preprocessing objects are created. One per fit, inside the fold loop. |
| `test_guard_final_fit.py` fails: a matrix handed to `fit` moved when only holdout values changed, or has a column for a level its rows never held | The built-in imputer, scaler or encoder learned from rows outside the fit (M1, M2). The fake cannot see inside the built-in preprocessing, so this guard compares the matrices. The application is wrong, not the test. | Which frame the built-in `Preprocessor` is fitted on, in the fold loop and for the final fit. |
| `test_guard_leak_screen.py` fails: a column is excluded as a leak that only holdout rows make look like one | The screen's trees were fitted on rows that include the holdout, or its missing-by-class shares were counted over every row (M3). | The rows handed to the leak screen: training rows, and each fold's own training part. |
| `test_guard_uncertainty.py` fails with an interval of some width where every block sums the same | The "block" bootstrap draws timestamps one at a time (M4). Each draw is a start position; the block is that timestamp and the next L − 1. | The resampling under time order. |
| `test_guard_uncertainty.py` fails with p = 1/(R+1) for a constant prediction | The p-value counts shuffles strictly above the observed skill (M5). A tie counts: "at least". | The comparison in the p-value. |
| `time.flat_holdout_reported` fails with a `ZeroDivisionError`, or with a skill of 0 | Skill was divided by a baseline error of exactly zero, or a zero was written in its place (D36). It is null, with a `skill_undefined` warning; the score keeps its interval, and only the skill interval loses the resample. | Where skill is computed: the headline, each resample, the leak screen, `stages.skill`. |
| A categorical column gives more one-hot columns than it has categories, or the leak screen misses a copy of the target written with stray spaces | Levels were taken as written, so `red` and `red ` counted as two (D42). | Strip once, where a column is typed, so the counts, the screen, the steps and the matrix all see one level. |
| A feature is listed in `columns` and no model ever saw it | Two matrix columns had one name and the second overwrote the first: a numeric column `c=red` beside a categorical `c` with the level `red` (D37). | Stop with `feature_name_collision` at step 5, judged on every level of the usable rows. |
| A configuration with status `failed` has null fold scores although its folds were scored | It failed after it was chosen and its record was wiped (D38). Only one that fails during selection has null fold scores. | The fallback after selection. |
| `csv.field_size_limit()` is different after a run | The loader raised the limit for the whole process and did not put it back. | A `try` / `finally` round the read. |
| A bare `UnicodeDecodeError`, or the wrong line taken as the header | The header line is read outside the handler that turns a decode error into `input_error`; or leading lines were skipped with `str.strip()`, which also removes a form feed and a no-break space, where pandas skips only spaces and tabs. | The loader's own reading of the header line. |
| Fold scores look better than the holdout score, by a lot | Preprocessing was fitted once on all training rows and reused inside the folds, so each validation block leaked into its own training statistics. | The fold loop: `fit` must be called on the fold's training rows. |
| `leak.excluded` fails with skill near 1 | The leak screen is missing, runs after the model, or screens on rows that include the holdout (F2). | Step 8 of the spec: the screen runs first, on training rows, with the selection folds. |
| Every strong feature is flagged as a leak under time order | The screen uses random folds, or measures skill against the training mean only. Under time order it uses the expanding folds, and for regression with unique timestamps the previous-value baseline. | The screen's folds and baseline. |
| An id column is among the features and skill is odd | The id-like rule was applied after the row cap, or by dtype: one blank cell turns an integer id into `float64`. | Roles are decided before the cap, by value. |
| A real integer measure (price, volume) is excluded as id-like | The 95%-distinct rule was applied to a number column. It applies to text only (D19). | The whole-number rule: distinct, and consecutive or strictly ordered. |
| `task.binary_numeric` fails: 0/1 target treated as regression | The task was read from the dtype. With a blank cell a 0/1 column loads as `float64` (F5). | Count distinct values; ignore the type. |
| `data.rows_reconcile` fails by a few rows | `dropna()` somewhere, or blank lines counted as rows, or rows dropped before `loaded` was counted (F6). | The first line that changes the number of rows. |
| `spend`-like column is all missing or became categories | Coercion was done with `astype(float)`, or not at all. Use `to_numeric(errors="coerce")`, count what failed, warn (F6). | Column typing. |
| `time.no_skill_on_blocks` fails with skill near 1 | The split is shuffled although a time column was declared, or the file order was used instead of the time column (F7). | The split under time order; the sort. |
| `time.gap_applied` fails on the folds | The gap was removed before the holdout but not before each validation block (F7). | `validation.folds[].gap_rows`. |
| A day-first date column runs and gives a result | The time column went through pandas' lenient parser. Text must match ISO 8601 with the year first, or the run stops (F7). | Time column validation. |
| `time.persistence_baseline` fails with skill near 1 | The regression baseline is the training mean only (F8). | Baselines under time order with unique timestamps. |
| `model.headline_recomputed` fails | The headline is a cross-validation score (F3), or accuracy (F4), or was computed from other predictions than the ones written. | The variable assigned to `headline.score`. |
| `model.signal_reg` fails with skill near -1 | The worst configuration was chosen: squared error is lower-is-better. | The comparison in selection. |
| `output.deterministic` fails | A clock value outside `meta.timing`, an unseeded generator, a set written in iteration order, or a library thread pool left at its default. | Diff the two files. |
| `output.same_at_any_n_jobs` fails in the last digits | Forest predictions were summed across threads, or BLAS was not held to one thread. `n_jobs=-1` or unset thread variables make a score move between runs. | `threadpoolctl.threadpool_limits`; `n_jobs=1` at predict time for forests. |
| A score moves after `pip install` | A library changed version. | `requirements.txt` against `meta.versions`. |
| Several commands started together each take 35 or 70 seconds longer than they should; or a gate run reports "the tool did not finish" and passes when repeated | Not the code. Until D32 the interpreter's base was on a mapped drive, and Python processes started at the same instant blocked each other for exactly 35 or 70 seconds. The base is a local path now and the wait was not seen in 60 rounds of eight simultaneous starts. The gate still spaces its own runs and retries once. | Run the check again before changing anything. If it comes back, check `home` in `mlprof_env\pyvenv.cfg` and tell the human. |
| NaN in `profile.json` | A score that could not be computed was written as it was. Write `null` and a reason. | `json.dumps(..., allow_nan=False)` will find it. |

## How to debug a gate failure

1. Read the whole message. It names the dataset, the value and the likely cause.
2. Open the files the gate just read: `verify/artifacts/runs/<run>/profile.json`
   and `predictions.csv`. The dataset is `verify/artifacts/data/<name>.csv`.
3. Re-run the one check: `python verify/smoke_test.py mlprofessor --only <id>`.
4. Fix the code. Never the band, the dataset or the check: they are locked, and
   a band you could widen proves nothing.
