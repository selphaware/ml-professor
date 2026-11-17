VERDICT: FIX FIRST

The build computes what the spec says on everything I ran, and I found no judge-awareness. Two things block shipping: one crash on a realistic input, and test gaps that leave the central guarantee (no fitted step sees the holdout) unprotected for the built-in preprocessing and the leak screen.

## Findings

| # | Severity | Type | Evidence (file:line) | Proposed remedy |
|---|---|---|---|---|
| 1 | High | defect + spec-gap | `mlprofessor/metrics.py:58`, called from `mlprofessor/pipeline.py:500`. Time order, unique timestamps, regression, target flat over the holdout (e.g. zero sales in the last quarter): the previous-value baseline scores exactly 0 and the run dies with `ZeroDivisionError`, no named error, no output. Reproduced twice (numeric and ISO time column). Spec 5.2 and 7 say a value that cannot be computed is `null` with a reason, or a named error; neither covers a zero baseline. | Spec: decide the outcome for a baseline error of 0. Test: `tests/test_edges.py`, flat-tail series asserts that outcome. Optionally a gate dataset `flat_tail` with a check `time.flat_holdout_reported`. Then a builder task to make them pass. |
| 2 | Medium | defect + spec-gap | `mlprofessor/preprocess.py:72-81`. Matrix columns are keyed by name, so a numeric column named `c=red` is overwritten by the one-hot column for level `red` of column `c`. Reproduced: the seam estimator received `['c=red','c=blue','c=green']`. The numeric column carrying the signal reached no model, yet was listed as a feature and drew a `strong_single_column` warning; headline skill 0.019. Spec 2.7 pins the naming but not collisions. | Spec: a collision is a named error (or pin a disambiguation). Tests: `tests/test_preprocess.py` (matrix width = numeric columns + kept levels) and a seam test that the estimator sees every feature. Builder task. |
| 3 | High | test-gap (F1 small) | `tests/test_pipeline.py:88-101` asserts fold scores only. Mutation: the final fit's imputer and scaler fitted on training + holdout rows (`pipeline.py:211` is the line protected). Suite result: only the two CLI-versus-function comparisons fail, and those fail because my in-memory patch does not reach the subprocess. A real build with this fault passes all 308 tests. | Test through the seam: record the matrix handed to `fit` for the final fit and each fold; rerun with holdout feature values replaced by extremes; assert the matrices are identical. Add the variant to the reference's fault fixtures so the test is shown to fail. No builder task: current code is correct. |
| 4 | High | test-gap (F1 small) | Same test changes numeric values only. Mutation: one-hot levels learned from every row of the table in every fit (`preprocess.py:66-67`). 0 of 308 tests fail. Spec section 8 names the encoder explicitly under F1. | Extend the test in #3: put a categorical level in holdout rows only (and one in a fold's validation rows only); assert it never appears as a matrix column. |
| 5 | Medium | test-gap (F1 small) | `tests/test_leakage.py:134-167`. Mutation: the screen's trees also fitted on holdout rows, and `missing_by_class` computed over every row (`leakage.py:73`, `:111`). 0 of 308 fail. The existing tests catch a screen that validates on holdout rows, not one that trains on them or counts them in rule 1. | Tests: a column missing-by-class only once holdout rows are counted must stay a feature; a column whose single-column skill crosses 0.98 only when holdout rows are trained on must not be excluded. |
| 6 | Medium | test-gap + gate-gap | `tests/test_uncertainty.py:84` checks `block_length` only; `verify/smoke_test.py:1564-1592` checks the label and that the interval contains the score. Mutation: timestamps drawn independently, still labelled `bootstrap_blocks` (`uncertainty.py:100-105`). 0 of 308 fail. Spec 11 lists the block rule as pinned. | Structural test: one very large error on the first holdout timestamp. Moving blocks draw it far less often than independent draws, so `interval_high` separates the two. Show it fails on the mutated copy. |
| 7 | Low | test-gap | `tests/test_uncertainty.py:54-73`. Mutation: `>` in place of `>=` in the p-value (`uncertainty.py:118`). 0 of 308 fail. With that fault a constant predictor reports p = 1/(R+1) instead of 1.0. | Test: a seam family predicting a constant must give `p_value == 1.0`. |
| 8 | Low | spec-gap | `pipeline.py:440-451`. When the best configuration raises on the final refit, the next best is promoted, the first is marked `failed` and its real fold scores are erased. If the failure is in `predict`, two configurations have predicted the holdout. Spec step 9 does not cover a failure after selection. | Spec: pin the behaviour (fall back, or `all_models_failed`-style stop; keep or erase fold scores). Test: a seam family that fails only on the final fit. |
| 9 | Low | defect | `result.py:130-135`, `pipeline.py:286-287`. `out_dir` naming an existing file raises an unnamed `FileExistsError` after the whole run has been computed. Spec 2.1 says any invalid argument is an `InputError` reported before loading. | Test in `tests/test_edges.py`: unusable `out_dir` raises `InputError` before any work. Builder task. |
| 10 | Low | defect | `load.py:87`. `csv.field_size_limit` is raised to 2³¹−1 for the caller's whole process and never restored (131072 before, 2147483647 after). | Test: the limit is unchanged after `profile()` on a CSV. Builder task. |
| 11 | Low | judgement-call | `pipeline.py:336-338`, `:425-427`. The `no_features` message says "see the column accounting", but an error returns no accounting. | Owner's call: name the excluded columns and reasons in the message. |
| 12 | Low | judgement-call | `cli.py:62-77`. The console summary prints the skill and "warnings: 2", but not the interval, the p-value or the warning codes. On `leak` the two exclusions are invisible unless the file is opened. Spec 11 leaves the summary to a human. | Owner's call: print the interval, p-value and one line per warning code. |
| 13 | Low | spec-gap | `CLAUDE.md` says the small forms of F1 "are caught exactly by the test suite's recording fakes". The seam cannot observe the built-in `Preprocessor` or the leak screen (#3 to #5). | Correct the sentence once #3 to #5 are closed. |

## Failure-mode coverage

Each row is an in-memory mutation of the built package, run against the full suite. I read the gate checks but did not run them against a broken build.

| Mode | Gate check (by reading) | Tests failing under mutation |
|---|---|---|
| F1 large (final fit on all rows) | noise bands and p-value floor, measured gaps | 9 |
| F1 small (fold trains on its validation rows) | none by design | 6 |
| F1 small (preprocessing, encoder, leak screen) | none by design | **0: findings 3, 4, 5** |
| F2 (screen removed / no warning tier / no missing-by-class) | sound | 7 / 3 / 1 |
| F3 | exact recomputation | 13 |
| F4 | sound | 6 |
| F5 | sound | 13 |
| F6 (rows dropped before counting) | sound | 20 |
| F7 (shuffled / fold gap not applied) | sound; fold gap is by reported count only | 17 / 2 |
| F8 | sound | 2 |

Forest prediction on many threads is caught by one test (`test_same_result_at_one_and_two_jobs`).

## Judge-awareness

None found. The package has no reference to the gate, tests, fixtures, dataset names or environment variables. The only broad `except` blocks are the two the spec requires for a failing configuration. The eight lines flagged in the build report are measurement scripts under `docs/measurements/` from the harness stage, not application code.

## What I ran

- Full gate: PASS, 30 checks, exit 0.
- Full suite: 308 passed in 41 s.
- Independent recomputation from the source files for `random_walk` (time order) and `messy`: row counts, holdout membership, `y_true`, both baselines, score, skill, per-column missing counts and ROC AUC all match.
- CLI: success path (exit 0, both files) and error path (exit 3, one line on stderr, nothing written).
- About 20 crafted tables in memory covering the edge cases above; CSV versus DataFrame and `n_jobs` 1 versus 8 gave identical results; the input frame is not modified.
- 19 mutation runs of the suite (15 failure-mode faults plus four targeted ones).

## What I could not verify

- The gate against a broken build: that needs a written copy of the package, which the guard forbids. The claim that the gate catches the large forms rests on reading the checks and on the measurements in spec 10.1.
- The 100,000 × 100 run time and reproducibility at 16 and 64 threads.
- The 30-seed band measurements; I ran the gate's one seed.
- Whether the limitations text and README read well to a user.

The gate run added its usual run folder under `verify/artifacts/runs/`; I changed nothing else in the repo and removed my own temporary files.
