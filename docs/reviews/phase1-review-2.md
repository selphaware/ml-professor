VERDICT: FIX FIRST

The method is sound and every number I recomputed from source data matched. One silent failure on a plausible input and two smaller contract breaks should be closed before shipping; nothing I found moves a headline number on ordinary data.

| # | Severity | Type | Evidence (file:line) | Proposed remedy |
|---|---|---|---|---|
| 1 | High | defect | A regression target with one sentinel-sized cell (`1.7976931348623157e308` among values near 20) finishes normally with `score`, `baseline_score`, `skill` and `cv_mean` all null, `interval_method: bootstrap_rows` with null bounds, `p_value: 0.004975` and no warning. The summary prints `mse n/a … skill none (see skill_undefined)`, a warning that is not there. Same result with the target scaled by 1e160. Cause: `mlprofessor/result.py:60` turns any non-finite number into null; nothing checks the score at `pipeline.py:488-523`; `uncertainty.py:207-213` computes a p-value because skill is NaN, not `None`. Breaks spec 5.2. | New test: both inputs, both schemes, must give either a named error or finite values in every number-typed field, and no p-value without a skill. Task: stop at assembly when the score, a baseline or a `cv_mean` is not finite. Owner decision needed first: the spec names no error for this. |
| 2 | Medium | defect | `seed=2**32` (or a timestamp-style seed such as `20261119123456`) raises sklearn's `InvalidParameterError` from the leak screen's tree. `pipeline.py:67` accepts it; `leakage.py:54-55` and `models.py:27,46,55` pass it as `random_state`. Spec 2.1 allows any whole number ≥ 0. From `cli.py:51` this would be a traceback, not exit 3 (read, not run). | New test: seeds `2**32-1`, `2**32`, `2**63` either run or raise `InputError` before loading. Task to make it pass. Owner decision: pin an upper bound in spec 2.1, or fold the seed down for sklearn. |
| 3 | Medium | spec-gap | `cli.py:58-59` prints four decimals. A target on the 1e-4 scale shows `mse 0.0000 … baseline 0.0000` and `score 0.0000 to 0.0000`; the real values are 9.37e-09, 2.24e-08 and 7.2e-09 to 1.23e-08. Spec 2.2's rule (`0.0005 × max(1, \|value\|)`) accepts this, though the same paragraph says four significant digits; `tests/cp3_helpers.py:136-139` implements the weak rule. | Owner tightens the rule to a relative tolerance. New case in `test_console_summary.py` with a small-scale target; task to print significant digits. |
| 4 | Medium | gate-gap | The self-test passes a mode when any one check naming it fails (`verify/gatekit.py:296-304`). Its output shows F6 trips only `data.rows_reconcile` and `data.coercion_reported`, and F7 does not trip `time.dayfirst_stops`. So `data.columns_accounted`, `data.long_rows_stop`, `models.all_accounted` and `time.dayfirst_stops` are in the matrix but have never been seen failing. | Add fixtures for the other forms (a lost column, a lost family, a long row read silently, a lenient date parse), or correct the matrix to list only what the self-test proves. |
| 5 | Low | test-gap | The suite's NaN assertion runs through a helper that maps NaN to `None` first (`tests/cp3_helpers.py:108`, `tests/mlp_helpers.py:75`), and the application does the same. It cannot see finding 1. No test uses a large seed, a huge target, or a small-scale summary. | Covered by the tests of findings 1 to 3; assert on the raw values before any NaN-to-null step. |
| 6 | Low | gate-gap | `--selftest` overwrites `verify/artifacts/runs/` with a broken fixture's output: after it, `random_walk/profile.json` held the F8 result (baseline `training_mean`, skill 0.999). `CLAUDE.md` sends the builder to that folder. I re-ran the gate afterwards, so it holds the build's files again. | Self-test writes its runs under its own folder. |
| 7 | Low | spec-gap | Target text is not stripped (`task.py:49-50`, `pipeline.py:173`); D42 covers features only. `yes` and `yes ` become two classes with `yes ` positive. A yes/no target with a few trailing spaces stops with `task_ambiguous … 4 distinct values` and no hint. | Owner decision on whether D42 extends to the target; one test either way. |
| 8 | Low | test-gap | `load.py:123-124` uses `str.strip()` to find the header line, which spec section 9 warns takes a different line than pandas after a form feed or no-break space. By reading, `load.py:58-67` still ends in an error. No test in `test_csv_edges.py` or `test_load.py` puts such a line before the header. | Test: form feed, no-break space and quoted-space lines before a two-column header give `input_error`. |
| 9 | Low | defect | `README.md:205-206` says the fast profile's kept rows "are chosen from the seed". Under time order they are the most recent whole timestamps (`pipeline.py:186-198`). | README test for the time-ordered cap wording; task to correct the sentence. |
| 10 | Low | judgement-call | `pipeline.py:388-392` raises `fast_profile_too_few_timestamps` whenever the cap dropped a row. On 5 timestamps × 600 rows it says "use profile='full'", and `full` then stops with `too_few_rows`. Spec leaves the choice open. | Owner's call; if changed, a test that the fast error is raised only when `full` would run. |
| 11 | Low | judgement-call | Three quiet fallbacks I could not reach with any input: `pipeline.py:174` (`errors="coerce"` on a regression target), `leakage.py:94` (drops the previous-value baseline if any value is not finite), `preprocess.py:91` (a non-finite deviation is read as 1). | Fold into the task of finding 1: raise instead of falling back. |

**Judge-awareness:** none. A grep of `mlprofessor/` for the gate, tests, fixture names, `sys.argv` and environment reads is empty. The eight warnings in the build report are all in `docs/measurements/`, the owner's lab scripts, not application code.

**Failure-mode coverage:** F1 to F8 are each caught by a gate check on a broken fixture and by at least one test. M1 to M5 each fail their guard; M1 and M5 are held by exactly one test each. The only weakness is finding 4.

**What I ran**
- Test suite: 370 passed.
- Gate: PASS, 31 checks, run twice (the second after the self-test).
- `verify/smoke_test.py --selftest`: PASS.
- `verify/mutation_check.py`: PASS; the good reference fails 0 of 365, each of 13 variants fails at least one test.
- My own in-memory runs of `profile()` on:
  - `examples/customers.csv`: rows, `y_true`, AUC and skill recomputed and equal;
  - a shuffled random walk with `gap=2`: split and previous-value baseline recomputed by hand and equal;
  - CSV against DataFrame: identical output;
  - nullable, boolean, category and zoned-datetime columns, a panel, and the full profile on separable and collinear data: no Python-only warnings.

**What I could not verify**
- The command line end to end on my own inputs, and any crafted CSV file: I may not write files. Findings 2 (exit code) and 8 rest on reading the code.
- Run time at 100,000 × 100, and identical output beyond the gate's 1-against-4-cores check.
- `docs/DECISIONS.md`, `plan/`, `loop/` and `.claude/` were not read; the tests and the gate were read selectively.
