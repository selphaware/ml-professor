VERDICT: FIX FIRST

The method is implemented as the spec describes, and every number I recomputed from source data matched. Three things block shipping: a raw crash on a plausible CSV, two routes to a confidently wrong headline that the spec currently allows, and four faults in pinned behaviour that neither the gate nor the suite can see.

## Findings

| # | Severity | Type | Evidence (file:line) | Proposed remedy |
|---|---|---|---|---|
| 1 | High | defect | `mlprofessor/load.py:167`, reached from `columns.py:257` for every column and from `load.py:179`/`:183` for the target and time column. A CSV column of integers too large for 64 bits (26 digits, such as a long account number) loads as Python integers, and `np.isinf` raises a raw `TypeError`. The command line exits 1 with a traceback and no named error, even when the column would only have been excluded. Reproduced from a CSV file and from a DataFrame, as a spare column, as the target and as the time column. Spec 2.2 and 7 allow only a named error or a reported exclusion. | New test in `tests/test_csv_edges.py`: such a column in each of the three roles ends in a named error or a column record, never a raw exception. Then a builder task to make it pass. |
| 2 | High | spec-gap | Spec 2.5 and 7 pin exact duplicate rows as "kept" (`pipeline.py:408-412`). On pure noise with every row present twice, the full profile reports skill 0.918 (interval 0.877 to 0.951), p = 0.001. The fast profile reports 0.760; a noise regression reports 0.551. Without the duplicates the same data gives 0.009. The only signal is the warning "600 row(s) exactly repeat an earlier row; they are kept." `README.md:154` says no fitted step has seen the holdout rows, which is untrue by content here. | Owner decision first: keep twins on one side of the split, or count holdout rows with an exact twin in training and say the headline is inflated. Then a gate dataset of duplicated noise with a no-skill band, a test, and a builder task. |
| 3 | Medium | spec-gap | `mlprofessor/split.py:113-117`; spec 2.7 pins numeric time keys as float64. Integers above 2^53 lose their distinctness: 600 distinct nanosecond instants 100 ns apart became 235 timestamps. `timestamps_unique` was false, the previous-value baseline vanished, and a random walk with its own lag scored skill 0.9994 against the training mean. The same instants as `datetime64` gave −0.20 against the previous value. This is F8's outcome by a route `time.persistence_baseline` cannot see. Consecutive ids above 2^53 stop with "3 distinct timestamps". The same cast at `columns.py:153` reports a `uint64` feature with 50 distinct values as `constant`. | Owner decision: compare integer-typed keys as integers, or raise `time_column_invalid` when two distinct integers map to one float. Tests in `tests/test_split_time.py` and `tests/test_time_pipeline.py` (such keys stay unique and both baselines are reported), then a builder task. |
| 4 | Medium | spec-gap | `mlprofessor/columns.py:124-128`, `:231-232`. Object columns holding `Decimal`, booleans with `None`, or integers with `None` (what a database driver returns) are excluded as `unsupported_type` with no warning and nothing in the console summary. On my table the skill was −0.011; with the same values in typed columns it was 0.779. Step 5 rule 7 allows the exclusion; no warning code exists for it. | Owner decision: a warning per `unsupported_type` exclusion (as dates and separator numbers have), or read such columns as numeric. One test either way. |
| 5 | Medium | test-gap + gate-gap | `mlprofessor/uncertainty.py:111`. Changing the percentiles to 5 and 95 while still reporting `interval_level` 0.95 passes everything. `verify/smoke_test.py:1858-1876` checks only that the interval is not degenerate and contains the score. Spec 11 pins the percentile rule. | Test through the seam with fixed predictions: compare both ends with an independent bootstrap of many resamples, within a tolerance that separates 90% from 95%. Add the fault as a variant in the reference so the mutation check proves the test. |
| 6 | Medium | test-gap + gate-gap | `mlprofessor/leakage.py:146`. Making the screen ignore the previous-value baseline passes everything; the gate has no lagged feature. With that fault every lagged feature of a persistent series is excluded as `predicts_target`. Step 8 pins the rule. | Test in `tests/test_leakage.py`: a random walk with its lag under time order keeps the lag as a feature, with no `suspected_leak`. Add the variant to the reference. |
| 7 | Low | test-gap + gate-gap | `mlprofessor/pipeline.py:185` (the fast cap taking the first 2,000 rows instead of a random sample) and `pipeline.py:324` (`cv_std` as the population deviation). Both pass everything; both are pinned (steps 6 and 9). | Tests: on a file sorted by target, the capped rows are not the head of the file; `cv_std` equals the sample deviation of `fold_scores`. Add both variants. |
| 8 | Low | test-gap | `mlprofessor/uncertainty.py:87` (tied predictions in a resampled AUC) and `pipeline.py:648` (`n_imputed` counted over every loaded row). Each fault passes all 411 tests. The gate does catch both, through `uncertainty.reported` on `time_blocks` and `data.coercion_reported` on `messy`. | Tests: a seam family with heavily tied predictions, whose interval must contain the score; `n_imputed` on a table with missing cells in rows that have no target. |
| 9 | Low | spec-gap | `mlprofessor/pipeline.py:466-468`. Under `fast` with a time column, when the cap keeps fewer than 100 rows (a 60-row latest timestamp behind a 2,500-row one), the error is `too_few_rows: 60 rows are left`. The full profile would split the same table 3,000 / 2,560. The message names neither the cap nor `profile='full'`. Step 6 says this literally; D46's intent is the opposite. | Owner decision; a case in `tests/test_fast_profile_error.py`. |
| 10 | Low | spec-gap | `mlprofessor/split.py:93-96`, `:127-130`. A fraction of six digits is refused (step 3 pins three), so a CSV that pandas itself writes from microsecond timestamps stops with `time_column_invalid`. The message says the value "is not ISO 8601 with the year first", which is false. | Owner decision: accept one to nine fractional digits, or keep the rule and have the message name the fraction. A case in `tests/test_split_time.py`. |

Findings 5 to 7 were confirmed on a scratch copy of the package outside the repo carrying all four faults: the gate passed 31 of 31 and the suite 411 of 411.

## Judge-awareness

None found. A search of `mlprofessor/` for the gate, the tests, fixture names, environment reads and `sys.argv` is empty. The only broad `except` blocks are the two the spec requires for a failing configuration (`pipeline.py:369`, `:568`), and both leave a `model_failed` warning. `result.py:93-94` still turns a non-finite number into null, but `check_finite` (`result.py:64-76`) runs first over every computed part. The warnings in the build report are all measurement scripts under `docs/measurements/`.

## Failure-mode coverage

- **Gate:** the self-test passes; each of the twelve modes is caught by a check that names it.
- **Suite:** the mutation check passes; the good reference fails 0 of 405 and each of the 19 variants fails at least one test.
- **Thin spots:** M1, M5 and M6 are each held by one test. Three of my own faults are also held by one test each: a previous value that ignores the gap, the panel autocorrelation warning removed, and the leak cut-off moved to 0.999.
- **Checks never seen to fail:** the self-test still lists `result.matches_json`, `output.complete`, `output.deterministic` and `output.same_at_any_n_jobs`. This is already in the backlog.
- **F8:** the matrix is right for the fault it names, but finding 3 reaches the same outcome with the baseline code intact.

## What I ran

- **Gate:** PASS, 31 checks. I ran it twice more against scratch copies, which overwrote `verify/artifacts/runs/`, so I re-ran it on the real build after each. The last run is the build's own and passes.
- **Suite:** 411 passed.
- **Self-test and mutation check:** both PASS.
- **My own faults:** 19, each run against the full suite in memory. Thirteen were caught and six survived; of those six, the gate caught two.
- **README example, through the command line:** row counts, `y_true` against the source, and ROC AUC recomputed with scikit-learn all match (0.838108).
- **Crafted CSV files, through the command line:** 22 files (line endings, byte-order mark, short and long rows, blank lines, index column, wrong delimiter, wrong encoding, NUL byte, quoted newline, empty files, duplicate header). All ended in exit 0 or exit 3 with one line, except finding 1.
- **Crafted tables in memory:** regression targets at 1e80, 1e150 and 1e153, odd target types, and the inputs behind findings 2, 3, 4, 9 and 10.

## What I could not verify

- Run time at 100,000 by 100, and identical output beyond the gate's own thread counts.
- Seeds other than the gate's; the surviving faults were tried at one seed.
- Whether a library warning can reach standard error beside a named error; I did not manage to provoke one.
- Spec sections 9 and 10, `docs/DECISIONS.md`, `plan/` and `loop/` were not read. The gate and the tests were read selectively; I relied on running faults against them instead.

I changed no file in the repo, and my scratch files in the temp folder are removed.
