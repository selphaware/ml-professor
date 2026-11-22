VERDICT: FIX FIRST

The build does what `spec.md` says, and the gate and tests can be trusted for what they test. Two pass-6 rules still print a confidently wrong headline on everyday tables with nothing said; both reproduce on the gate's own datasets with one cell or one column changed. The spec pins both behaviours, so each needs a decision, a check that fails today, and a task.

| # | Severity | Type | Evidence (file:line) | Proposed remedy |
|---|---|---|---|---|
| 1 | High | spec-gap, gate-gap | **One value that is no date switches off the `YYYYMMDD` rule (F14's outcome).** `mlprofessor/columns.py:262-265`, `:283-285`; spec.md:805-806 pins "one value that is no date, and the rule does not apply". On `verify/artifacts/data/day_numbers.csv` with `--no-time-order --no-entities`, clean: `day` excluded, skill +0.015 / +0.010 / +0.034 (seeds 0-2). With one cell of 1,800 set to `99991231`: `day` is a feature, skill +0.924 / +0.928 / +0.965, p 0.005, no warning. One cell of `0` gave 0.924 and `20240230` gave 0.915 (seed 0). The consequence is not among the stated limits (spec.md:3474-3480). `time.date_numbers_seen` and `tests/test_date_numbers.py` use clean columns only. | A decision on rule 2a: a share, as rule 4 has for text dates (95%), or named sentinels. Then a gate check on `day_numbers` with one sentinel cell (column excluded or named, skill in the no-skill band), a broken copy for the self-test, a test, and a task that makes them pass. |
| 2 | High | spec-gap, gate-gap, test-gap | **`duplicates="group"` forms its sets with the target when a few-valued leaked column is in the table (F12's outcome).** `mlprofessor/pipeline.py:728-739` forms units on step-5 features; `:881-908` then excludes the leak; `:913` counts only copies equal in the target too. On `few_valued.csv` plus a `status` column that copies the target, `--duplicates group`: `status` excluded as `suspected_leak`, skill −0.807 / −0.953 / −0.849 (F12's band in spec 10.1 is −1.000 to −0.648), 8 / 7 / 6 of 16 feature combinations on both sides, `n_holdout_copies` 0, no stop. The `same_features_grouped` warning (`pipeline.py:782-787`) still says "no combination of feature values is on both sides". The default `error` run of such a table stops and offers `group`, so the caller is led here. spec.md:1365-1375 covers only rows whose targets are equal. The one test, `tests/test_duplicates_group.py:122-129`, uses a leak distinct in every row, which stops. | A decision: under `"group"`, after the leak screen, any combination of the remaining features on both sides stops the run with the existing "remove the column and run again" message. Then a gate check (`few_valued` plus a two-level copy of the target: the stop, or zero combinations on both sides with the band), a broken copy, a test, and a task. |
| 3 | Low | judgement-call | **`"group"` can report negative skill where real signal exists.** A 2,000-row table whose label depends on the parity of two few-valued columns scores +0.710 under `"keep"` and −0.188 under `"group"`, with no interval (16 units). The gate's noise table gives −0.061 / +0.019 / −0.009, inside the band. This is D62's stated price, and my table is a worst case. | No code change. Consider one more sentence in the `same_features_grouped` text or the README, and a measurement of the bias on noise as the number of combinations falls. |
| 4 | Low | spec-gap | **Other everyday date forms pass unseen and unlisted.** `YYYYMMDDHH` as a number: feature, skill 0.928, no warning. A five-day period label as text: feature, skill 0.11 to 0.18. spec.md:3477-3480 lists only day counts, `YYYYMM`, day-first numbers and epoch seconds. | Add them to the stated limits when finding 1 is decided; no check needed if they stay limits. |
| 5 | Low | spec-gap | **`date_column_undeclared` says "the split is random" whatever the scheme.** `mlprofessor/pipeline.py:206-210`; under `--group-col` with `--no-time-order` the scheme is `grouped`. Read from the code, not run. No test or fixed text pins the wording. | Pin the text in spec 5.4 with a test; a one-line task. |
| 6 | Low | judgement-call | **A dormant "not finite becomes null" branch.** `mlprofessor/result.py:93-94` turns such a float into `None`. It cannot be reached today: `check_finite` (`result.py:63-76`, called at `:120`) raises first, and M18 guards that. If the check were ever narrowed, this is the D44 fault. | Leave it, or have a task replace the branch with a raise. M18 stays the guard. |

**Judge-awareness:** none found. A search of `mlprofessor/*.py` for test, gate, fixture and environment names matched only two innocent lines (the docstring of `extensions.py` and a warning text). The build report's hits are all lab scripts under `docs/measurements/`. I saw no fixture values, expected outputs or silent defaults. The broad `except Exception` at `pipeline.py:590` and `:941` is the contract's "a configuration that raises is recorded", and each adds a `model_failed` warning.

**Failure-mode coverage:** every mode in the matrix has a gate check and a test that fail on its broken copy, proved by the self-test and the mutation check I ran. The weakness is one of form: F12 and F14 are each proved against one form of the fault, and findings 1 and 2 are other forms of the same outcomes.

**What I ran**
- **Gate:** PASS, 43 checks.
- **Suite:** 1841 passed in 213 s.
- **Self-test** (`verify/smoke_test.py --selftest`): PASS, 34 of 34 pairs of a check and its failure mode seen failing over 21 broken fixtures.
- **Mutation check** (`verify/mutation_check.py --jobs 8`): PASS, the good reference fails 0 of 1828, and each of 51 variants fails at least one test.
- **Splits, by my own count:** 1,405 grouped splits through `stages.split_random` against the four statements of step 7 property 3 and the fold rules, and 556 time splits against step 7; no violation in either.
- **Panel baselines:** an irregular panel of 9 series with `gap=1` and an entity first seen in the holdout. Both baselines, the count of rows with no previous value, the row counts and the headline MSE match my recomputation to the last digit.
- **README command:** the time-ordered command on `examples/customers.csv`. Rows reconcile with the source, AUC recomputed from `predictions.csv` equals the headline, and the only rows dated inside the holdout but outside it are two with no target.
- **Code read:** every module of `mlprofessor/` except the thin facades.
- **Spec read:** section 3 steps 1 to 7, 11 and 12, the `duplicates="group"` part of step 8a, and the known limits.

**What I could not verify**
- Spec sections 2, 4 to 10 and most of step 8 were not read in full; the output schema and fixed texts are covered only by the gate and tests.
- No run at scale (100,000 rows) and none on real tables.
- Interval coverage, and whether the README prose reads well.
- `verify/reference/` was not inspected; I relied on the self-test and mutation check that use it.
- Finding 5 is from reading the code only.
- Probe runs used the fast profile, except one full-profile run for each of findings 1 and 2.
- Git history was not consulted.
