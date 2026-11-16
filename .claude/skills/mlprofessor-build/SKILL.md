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

## Common failures and their real causes

| What you see | The real cause | Where to look |
|---|---|---|
| `model.noise_clf` or `model.noise_reg` fails with skill well above the band | The chosen configuration was refitted on training plus holdout rows before scoring (F1). | The refit after selection: which row positions does it receive? |
| A recording-fake test reports that a step's `fit` saw holdout rows | A scaler, imputer or encoder was fitted before the split, or once for all folds (F1, small form). No band can see this; only the fake can. | Where preprocessing objects are created. One per fit, inside the fold loop. |
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
