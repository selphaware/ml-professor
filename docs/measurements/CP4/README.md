# CP4 measurements

The scripts behind the CP4 session (2026-11-20): the second review's findings
(D44 to D46) and the three rulings that followed the same day (D48). Kept here
by D35. The conclusions are in `spec.md` and `plan/test-plan.md`; this folder
is how to get the numbers again.

| File | What it is |
|---|---|
| `findings.py` | What an implementation does on the inputs of each finding: extreme targets and features, the seed, the summary, white space round the target, the fast profile's error, lines before the header. Run against the application and against the reference. |
| `fixtures.py` | Which gate checks fail on each fixture of a gate, and whether every pair of a check and a failure mode it names was seen failing. |
| `prototypes.py` | Runs prototype tests against the application, the reference and any of its variants, and prints each test's outcome under each. |
| `previous_exists.py` | Looks for a validation or holdout row with no previous value, through the step functions alone (finding 11). |
| `runs_untouched.py` | Records the build's run folder and says whether anything in it changed (finding 6). |

Raw outputs are not kept. Every script prints its result; none writes inside
the repo, apart from the snapshot file `runs_untouched.py` is told to write.
The prototype tests themselves are not kept: the tests stage writes the real
ones from the plan.

## Running a measurement again

From the repo root, with the project interpreter, one command at a time:

```
$env:MLPROF_LAB_WORK = "C:\somewhere\outside\the\repo"
$py = "E:\CODE\ml-professor\mlprof_env\Scripts\python.exe"
$W  = "$env:MLPROF_LAB_WORK\cp4"
```

| What it backs | Command | Takes |
|---|---|---|
| The application and the reference on every finding | `& $py -B docs/measurements/CP4/findings.py app`, then `... ref` | 40 s each |
| Every `catches` of the gate proved | `& $py -B verify/smoke_test.py --selftest` | 8 to 10 min |
| The same table for an older gate | `& $py -B docs/measurements/CP4/fixtures.py <old>\verify\smoke_test.py` | 3 min |
| Every value of a full gate run on the reference | `& $py -B docs/measurements/CP2/long_rows.py values - - "$W\gate_after.json"` | 20 s |
| "No gate value moved" | `& $py -B docs/measurements/CP2/long_rows.py compare "$W\gate_before.json" "$W\gate_after.json"` | |
| The suite against the reference and its 19 variants | `& $py -B verify/mutation_check.py --jobs 7` | 5 min |
| The new tests, red or green today | `& $py -B docs/measurements/CP4/prototypes.py <folder of prototypes> app good F6d M6 M7` | 1 min |
| No row without a previous value | `& $py -B docs/measurements/CP4/previous_exists.py app`, then `... ref` | 10 s each |
| The self-test leaves the run folder alone | `& $py -B verify/smoke_test.py mlprofessor`; `& $py -B docs/measurements/CP4/runs_untouched.py snapshot "$W\runs.json"`; the self-test; `... runs_untouched.py compare "$W\runs.json"` | |

`gate_before.json` comes from the CP2 `values` command given the state before
CP4: `values <old>\verify\smoke_test.py <old>\verify\reference
"$W\gate_before.json"`, where `<old>` is a copy of the repo at commit
`deae312`, outside this folder. The human makes it, since agents do not run
git here; in the session it was a copy of `verify/` taken before the first
edit.

## Seeds

- `findings.py`: tables from `numpy.random.default_rng(0)`, 400 rows; the
  tool at seed 0, fast profile, one core.
- `previous_exists.py`: `numpy.random.default_rng(2026)`.
- The prototypes: the fixed seeds written in the test plan's recipes (21 to
  63); the tool at seed 0 unless the test says otherwise.
- No gate dataset was added or changed: every dataset's seed and bytes are
  what they were.

## What was found

pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1; 32 cores, 64 threads.

### The findings, in the application and in the reference

"Before" is the reference as it was at the start of the session.

| Input | Application today | Reference before | Reference after |
|---|---|---|---|
| A regression target with one cell of 1.7976931348623157e308, random scheme | finishes: `score`, `baseline_score`, `skill`, both intervals null; `p_value` 0.004975; no warning | a raw `ValueError` from `json.dumps` ("Out of range float values") | `non_finite_score`, naming `training_mean`, before any fit |
| The same under time order, the cell in a training row | finishes: `score` and `skill` null; `baselines` holds `training_mean` null beside `previous_value` 15.44; both configurations `fitted` with a null `cv_mean` | the same raw error | `non_finite_score` |
| The target times 1e160, either scheme | finishes with the same nulls | the same raw error | `non_finite_score` |
| One target cell of 1e155 at row 120, time order | finishes: score 1.9e305, skill −1.2e304, both configurations `fitted` with a null `cv_mean`, one of them chosen | the same raw error | `non_finite_score` from the leak screen, before any fit |
| The target on the scale of 1e80 | finishes; `cv_std` null beside the status `fitted` | the same raw error | finishes with finite numbers |
| A family predicting 1e200 everywhere, alone | finishes: `score` null, `skill` null, `p_value` 1.0, the family `fitted` and chosen | the raw error | `all_models_failed` |
| The same beside a sound family | the far family `fitted` with a null `cv_mean`; the sound one chosen | the raw error | the far family `failed`; the sound one chosen |
| A family predicting NaN on three holdout rows, classification, beside a sound one | failed after it was chosen, fold scores kept; the sound one chosen; `y_pred` finite | **`fitted` and chosen; NaN written to `y_pred`** | as the application |
| A feature column times 1e200 | a raw `ValueError` from scikit-learn: "Input X contains infinity or a value too large for dtype('float32')" | the same raw error | finishes: every cell is missing, the column is excluded as `all_missing` |
| One feature cell of 1e39, at row 7 | the same raw error | finishes, because the cell fell in its holdout and no tree was fitted on it: score 4.2e76, skill −6.7e75 | finishes: the cell is missing, `n_infinite` 1, `infinite_values` warning |
| One feature cell of 1e38 | finishes | finishes | finishes |
| `stages.Preprocessor` on 1, 2, 3, 4, 1e200, inf, −1e39 | returns values near 1e199 (a deviation that is not finite, read as 1) | returns NaN | the three cells take the median: −1.77, −0.59, 0.59, 1.77, 0, 0, 0 |
| A complex-number target, `task="regression"` or `auto` | runs as a regression; the imaginary part is dropped | the same | `task_unsupported`; `task_ambiguous` |
| `seed` of 4,294,967,295 | runs | runs | runs |
| `seed` of 2³², 2⁶³ or 20261119123456 | a raw `InvalidParameterError` from the leak screen's tree | the same from `train_test_split` | `input_error`, before anything is loaded |
| The same from the command line | exit 1, a traceback of 30 lines | exit 1, a traceback | exit 3, one line |
| `stages.split_random(120, None, 2**32)`; `-1` | returns a split; a bare `ValueError` | raw `InvalidParameterError` for both | `input_error` for both |
| A target on the scale of 1e-4, the summary | `mse 0.0000 … baseline 0.0000`, `score 0.0000 to 0.0000`, `p-value: 0.0050` | six significant digits | six significant digits |
| A yes/no target with four cells written `yes ` | `task_ambiguous`, "4 distinct values" | the same | runs; classes `no`, `yes` |
| Every `yes` written `yes ` | runs; `classes` `['no', 'yes ']`, positive class `'yes '` | the same | classes `no`, `yes` |
| Two target cells of three spaces | `task_ambiguous`, "3 distinct values"; `usable_rows` drops 0 | the same | runs; 2 rows dropped as missing target |
| 5 timestamps of 600 rows, fast | `fast_profile_too_few_timestamps` ("use profile='full'"); full gives `too_few_rows` | the same | `too_few_rows` under both |
| 3 timestamps of 2,500 rows, fast | the fast error; full gives `too_few_rows` | the same | `too_few_rows` under both |
| 9 timestamps of 450 rows, gap 2, fast | the fast error; full gives `gap_too_large` | the same | `gap_too_large` under both |
| 8 timestamps of 450; 30 of 300; 12 of 2,500, fast | the fast error; full runs | the same | the same |
| A line of a form feed, a no-break space, two quotes round a space, two quotes, a vertical tab or an em space, before a header of two columns | `input_error` on all six | `input_error` on all six | the same |

For G that is the whole answer: pandas reads each such line as a header of
one column (and 161 rows with an `Index` of the leading field), and both
implementations stop with `input_error`, before a header of two columns as
before one of three. No fault was found.

Three things the review did not have:

- **A finite feature value above the 32-bit range crashed both
  implementations wherever a tree was fitted on it.** scikit-learn's trees and
  forests read their input as 32-bit floats; the leak screen's tree reads the
  column as it is. Where the cell fell only in the holdout the run finished,
  with numbers to match (a score of 4.2e76 in the reference). The third
  quiet fallback of finding 11 (a deviation that is not finite, read as 1)
  could not be reached through `profile()` only because this crash came
  first; through `stages.Preprocessor` it could. D48 rules the value missing,
  and with that no deviation is ever anything but finite.
- **A complex-number target ran as a regression** in both, without its
  imaginary part. The first fallback of finding 11 (`errors="coerce"` on a
  regression target) coerces nothing for an integer or float target, and for
  a complex one the conversion beside it lost half the number. D48.
- **The application already fails a configuration whose prediction is not
  finite; the reference did not**, and wrote NaN into `y_pred` for
  classification, where ROC AUC stays finite whatever is predicted. The spec
  now says what the application does, and the variant M6 is the reference as
  it was.

The second fallback of finding 11 (the leak screen dropping the
previous-value baseline when a previous value is missing) is unreachable:
`previous_exists.py` tried every number of unique timestamps from 100 to 400
with every gap the split accepts, 8,175 splits and 1,898,456 validation and
holdout rows in each implementation, and found no row without a previous
value. The largest gap accepted is 10 at 100 timestamps and 42 at 400.

Two numpy facts behind the rules: `(1e155) ** 2` is infinite, and
`numpy.std` of five values near 1e160 is infinite although the answer, about
8e158, can be held.

### H: every check that names a failure mode, against its fixture

The gate as it was, by `fixtures.py` on a copy taken before the first edit:

```
good: failed nothing
F1: failed ['model.noise_clf', 'model.noise_p_value', 'model.noise_reg', 'model.selection_full', 'time.no_skill_on_blocks', 'uncertainty.reported']
F2: failed ['leak.excluded', 'leak.near_leak_warned', 'leak.regression']
F3: failed ['model.headline_recomputed', 'model.metric_clf', 'model.selection_full', 'time.flat_holdout_reported', 'time.persistence_baseline']
F4: failed ['data.columns_accounted', 'model.headline_recomputed', 'model.metric_clf', 'model.selection_full', 'task.binary_numeric', 'uncertainty.reported']
F5: failed ['task.ambiguous_stops', 'task.binary_numeric']
F6: failed ['data.coercion_reported', 'data.rows_reconcile', 'leak.control_kept', 'leak.excluded']
F7: failed ['time.flat_holdout_reported', 'time.gap_applied', 'time.no_skill_on_blocks', 'time.order_respected']
F8: failed ['time.flat_holdout_reported', 'time.persistence_baseline']

| data.columns_accounted | F6 | F4 **UNPROVED for F6** |
| data.long_rows_stop | F6 | - **UNPROVED for F6** |
| models.all_accounted | F6 | - **UNPROVED for F6** |
| time.dayfirst_stops | F7 | - **UNPROVED for F7** |
4 pair(s) of a check and a mode it names not seen failing; 8 broken fixtures
```

(The four rows of its table that carry a mark; the other 27 rows are left out
here. It exits 1.)

Those four are the review's finding 4, and the only ones: the other 17 pairs
were proved all along. After CP4 the self-test prints the table itself; see
the session's final report, or run it. 22 of 22 pairs, over 12 broken
fixtures, each fixture run twice.

What each new fixture fails: F6b `data.columns_accounted` alone; F6c
`models.all_accounted` and `model.selection_full`, whose count of families
it also breaks; F6d `data.long_rows_stop` alone; F7b `time.dayfirst_stops`
alone.

One existing fixture now fails one more check. F7, the shuffled split, fails
`time.persistence_baseline` as well: with a shuffled split a validation row
can be the first row in time, which has no previous value, and the reference
now stops there with an internal error where it used to carry on with the
training mean (finding 11). F7 is still caught by its own three checks.

The checks that give a reason in place of a failure mode, and whether a
broken fixture has ever made them fail:

| Check | Failed on a fixture |
|---|---|
| `leak.control_kept` | F6 |
| `time.flat_holdout_reported` | F3, F7, F8; and, when it was added, the reference as it stood before D36 and the first build |
| `uncertainty.reported` | F1, F4 |
| `model.signal_clf` | never |
| `model.signal_reg` | never |
| `time.undeclared_warns` | never |
| `result.matches_json` | never |
| `output.complete` | never |
| `output.deterministic` | never |
| `output.same_at_any_n_jobs` | never on a fixture of the gate; its comment records copies of the reference whose forests predict on several threads failing it in 4 of 4 trials |

Nothing was changed for these.

### The gate, before CP4 and after

Every value of a full gate run on the reference, in the two states
(`long_rows.py compare`):

```
datasets (seed, target, bytes): 19 before, 19 after
  57 values before; 0 moved, 0 gone, 0 new under an existing name
checks (verdict, message, failure modes named): 31 before, 31 after
  94 values before; 4 moved, 0 gone, 0 new under an existing name
    moved  data.columns_accounted: catches[0]: 'F6' -> 'F6b'
    moved  data.long_rows_stop: catches[0]: 'F6' -> 'F6d'
    moved  models.all_accounted: catches[0]: 'F6' -> 'F6c'
    moved  time.dayfirst_stops: catches[0]: 'F7' -> 'F7b'
values the gate recorded: 100 before, 100 after
  100 values before; 0 moved, 0 gone, 0 new under an existing name
runs (exit code, both streams, files, every field of profile.json but meta.timing, bytes of predictions.csv): 24 before, 24 after
  14302 values before; 0 moved, 0 gone, 0 new under an existing name
```

The four that moved are the four `catches` CP4 asked to change. No dataset,
no check's verdict or message, no recorded value, and no value of any run
moved: not a number in any `profile.json`, not a byte of any
`predictions.csv`, not a character of any console summary.

### The suite against the reference and its variants

`verify/mutation_check.py` on the 370 cases of the first two passes, before
CP4 and after it (cases failing; the README's five are not counted):

| Case | Before | After |
|---|---|---|
| good | 0 of 365 | 0 of 365 |
| F1 | 13 | 13 |
| F2 | 9 | 9 |
| F3 | 17 | 17 |
| F4 | 10 | 10 |
| F5 | 35 | 35 |
| F6 | 23 | 23 |
| F7 | 24 | 34 |
| F8 | 5 | 5 |
| F6b | | 8 |
| F6c | | 15 |
| F6d | | 3 |
| F7b | | 6 |
| M1 to M5 | 1, 2, 2, 3, 1 | 1, 2, 2, 3, 1 |
| M6 | | 0: not caught |
| M7 | | 1 |

**The changed reference fails none of the 365**: nothing D44 to D48 decided
contradicts a locked test. Each of the four new forms fails existing tests.
F7 fails ten more, for the reason given under H. M6 is caught by nothing in
the existing suite, which is why the test plan demands its guard; M7 is
caught by `test_time_pipeline.py::test_wide_panel_under_fast_raises`.

### The tests the plan demands, checked ahead

Every test named in the third-pass section of `plan/test-plan.md` was written
once as a prototype, outside the repo, and run with `prototypes.py` against
the application, the changed reference, F6d, M6 and M7: 27 tests, 41 cases.

- against the application 35 cases fail and 6 pass: the 24 red tests and the
  3 green ones of the plan's two lists, exactly;
- against the reference all pass but the README test, which judges
  `README.md` and passes on a sentence corrected as T30 asks;
- the guard of `test_guard_header_lines.py` fails on F6d in its three cases
  and on no other variant tried;
  `test_a_prediction_that_is_not_finite_counts_as_raising` fails on M6 alone;
  `test_fast_error_stands_when_the_full_profile_would_run` fails on M7 in
  both cases (M7 also fails one case of the red test beside it);
- the 41 cases take 11 seconds against the application and 16 against the
  reference.

One prototype had to be changed, and the plan says why: a target cell that
spoils only the folds must sit where a single squared error overflows (1e155
at row 120 of 400), not where only a sum of them does (2e155 at row 5), which
an implementation that adds carefully would survive.

### I: the self-test and the run folder

An ordinary gate run on the application (`GATE: PASS (31 checks)`), then
`runs_untouched.py snapshot`, then the self-test (through the loop's
`doctor`, and once more directly), then `runs_untouched.py compare`:

```
43 files before, 43 now: 0 gone, 0 new, 0 written again (0 with other bytes)
UNTOUCHED: every file has the bytes and the modification time it had
```

The self-test's own files are under
`verify/artifacts/selftest_runs/<fixture>/<run>/`, one folder for the good
fixture and one for each of the twelve broken ones. Before CP4 the same
sequence left a broken fixture's output in `verify/artifacts/runs/` (the
review found F8's `random_walk` there).

`doctor` on the final state: `SELFTEST: PASS`, 12.5 seconds per edit against a
budget of 20, 16.6 seconds for the full gate on the good fixture, a valid
plan of 30 tasks.

## Two things to know about these files

- The loop driver classifies anything under `docs/` as application code. Its
  build report lists application files that mention the gate; these scripts
  name the judge's folders in passing and may appear there. They are
  measurement tools for the harness stage and no part of the package.
- For the same reason a build-stage agent is not prevented from editing them.
  If a number here ever disagrees with `spec.md`, run the command again.
