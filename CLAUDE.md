# ML Professor (phase 1)

A Python package, `mlprofessor`, that measures how predictable one target
column of one table is. The contract is `spec.md`. The decisions behind it are
`docs/DECISIONS.md` (D1 to D35). The build order is `plan/tasks.json`; what the
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
`verify/artifacts/runs/<run>/` and the data in `verify/artifacts/data/`.

## The methodology in five lines

1. Count every row and column; drop only rows with no target (or no time value), and say how many.
2. Decide the task by a stated rule; split off a 25% holdout first: at random, or the latest whole timestamps when `time_col` is given.
3. Choose a model by five folds inside the training rows only, with every preprocessing step refitted inside each fold.
4. Refit the chosen one on the training rows and score it once on the holdout, as skill over a baseline (ROC AUC against 0.5; squared error against the training mean or the previous value).
5. Report it with an interval and a p-value from the holdout predictions, and with every exclusion, warning and limitation.

**The headline is computed only on rows no fitted step has seen.**

## Output

`profile(...)` returns a `ProfileResult`; with an output folder it writes
`profile.json` and `predictions.csv`. Full schema: `spec.md` section 5.

| Part of `profile.json` | Holds | On `ProfileResult` |
|---|---|---|
| `meta` | tool, schema version, profile, seed, `n_jobs`, library versions, `timing` | `to_dict()` |
| `input` | source, kind, target, time column, task requested, gap, rows, columns, duplicate rows | `to_dict()` |
| `rows` | seven counts: loaded, four `dropped_` reasons, training, holdout | `.rows` |
| `columns` | one record per input column: role, kind, reason, leak rule, four counts | `.columns` |
| `task` | type, decided by rule or declared, distinct values, classes, positive class | `to_dict()` |
| `validation` | scheme, holdout share, folds, gap, time form, last training and first holdout timestamp, caveat | `to_dict()` |
| `headline` | metric, score, model, baseline and its score, skill, interval, p-value, reasons | `.headline` |
| `baselines` | each baseline scored on the holdout, and which one was used | `to_dict()` |
| `cv` | metric, folds scored, configurations, the chosen one, its mean and spread | `to_dict()` |
| `models` | one record per configuration: params, status, error, fold scores, mean, spread, chosen | `.models` |
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
| F6 | rows with any missing value are dropped before counting | data.rows_reconcile, data.columns_accounted, data.coercion_reported, data.long_rows_stop, models.all_accounted |
| F7 | the split is shuffled even when a time column is declared | time.order_respected, time.no_skill_on_blocks, time.gap_applied, time.dayfirst_stops |
| F8 | the regression baseline is always the training mean | time.persistence_baseline |

The gate catches the large forms. The small forms (a scaler fitted before the
split, a fold that saw its own validation rows) are caught exactly by the test
suite's recording fakes, through `mlprofessor.extensions` (spec 2.6).

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
- Seeds: one `seed` drives the cap, the split, the folds, every model and every
  resample. The same seed gives the same files apart from `meta.timing`.
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
- `verify/reference/` is the judge's own fixture, with eight deliberate faults
  wired into it. It is not a starting point: do not copy from it.
- Application code never mentions the gate, the tests or the loop.
- No git. The human owns every git operation.
- Write only inside `mlprofessor/`, and `requirements.txt` and `README.md`.
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
