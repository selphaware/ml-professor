VERDICT: FIX FIRST

The build passes the gate (33 checks) and the suite (1,321 tests), and its numbers reconcile with the source data. It still has one defect against the spec that neither judge sees (finding 1). Separately, the spec's own copies rule (D49) is defeated by near-copies, in the reference as well, and that needs your decision (finding 2).

| # | Severity | Type | Evidence (file:line) | Proposed remedy |
|---|---|---|---|---|
| 1 | Medium | defect + test-gap | Step 1a is applied a second time, to the usable rows only: `mlprofessor/task.py:104` (called from `pipeline.py:500`), `mlprofessor/split.py:183` and `split.py:121-135` (called from `pipeline.py:489`). Spec step 1a says each object column is read once, over all loaded rows. So when the one odd cell of a column sits in a row dropped for another column's missing cell, the build returns a result where the spec and the reference stop. Five tables, run on both (evidence below). | **Test, then task.** New tests in `tests/test_object_numbers.py` and `tests/test_whole_numbers.py`: for the target and for the time column, the odd cell (text, or a whole number beyond 64 bits) in a row dropped for the other column must give the same named error as the same cell in a usable row; and `train_end` must be an integer when the column's only value above 2**53 is in a dropped row. Add a matching variant to `verify/mutation_check.py` so the guard is proved. Task: in the pipeline, steps 3 and 4 take the column as step 1a read it over the loaded rows. |
| 2 | High | spec-gap + gate-gap | `mlprofessor/pipeline.py:183-193` implements spec step 8a as written: a copy must match in every feature column. One per-row-unique column, or a 1e-9 difference in one column, makes the count zero. Measured on pure noise with every row present twice: skill 0.84 to 0.90 at p = 0.001 with no warning at all, in the build and the reference alike (evidence below). That is F9's outcome with the rule intact. `data.holdout_copies_stop` only exercises exact copies. | **Your decision first (a new D amending D49), then gate check, then task.** Options: count a copy when the target and all feature columns but one agree; or compare numbers to a tolerance; or at least warn when rows repeat on most columns. Then a gate check on "noise, every row twice, plus one unique column" (and a 1e-9 jitter variant) that must stop or warn, with its own broken copy in the self-test (an F9b). Then a build task. |
| 3 | Low | spec-gap | `mlprofessor/columns.py:295` counts the target's infinite cells in `columns[].n_infinite`; the reference writes 0. They differed on 8 of 170 generated tables (build 2 to 8, reference 0). Spec 5.2 (line 1367, "a numeric column") reads the build's way; step 5 (lines 632-637) and the `infinite_values` warning read as features only. No gate check or test pins it: the suite passed on the build, and by the mutation check's premise it passes on the reference too. | Decide which, state it in spec 5.2 for the target and time roles, add one assertion to `tests/test_result_schema.py`, and bring the other implementation into line. |
| 4 | Low | judgement-call | `mlprofessor/result.py:93` turns any non-finite float into `None`, and `check_finite` (`result.py:48`) looks at five parts only, not `meta`, `input`, `rows`, `columns` or `task`. No float there can be non-finite today, so this is unreachable now. It is exactly the silent null D44 forbids if a float is ever added to those parts. | A test that builds a `ProfileResult` with a NaN in `task` or `columns` and expects `non_finite_score`; a task that checks every part. |
| 5 | Low | judgement-call | `mlprofessor/pipeline.py:416-431`: the `except Exception` that records a failed configuration also wraps the tool's own fold scoring (`_score`), not only the model's fit and predict. A fault in the tool's metric code on one fold would be reported as `model_failed`. | A seam test in which scoring (not the model) raises, expecting the run to stop; a task that narrows the `try` to fit and predict plus the non-finite check. |

**Evidence for finding 1** (fast profile, in-memory tables, same script on both packages):

| Table | Build | Reference |
|---|---|---|
| Object target of whole numbers, one text cell in a row with no time value | runs as regression | `task_unsupported` |
| Object target, one whole number beyond 64 bits in a row with no time value | runs | `task_unsupported` |
| Object time column of whole numbers, one text cell in a row with no target | runs, `time_form` numeric | `time_column_invalid` |
| Time column as `read_csv` loads it (Python integers, one beyond 64 bits) in a row with no target | runs | `time_column_invalid` |
| int64 time column whose one value above 2**53 is in a row with no target | `train_end` 23400.0 | `train_end` 23400 |

The fourth row is my emulation of the object column of Python integers that spec section 9 says `read_csv` produces, so the defect is plausibly reachable from a plain CSV through the CLI; I did not run a real file. The type matrix cannot see any of these: it varies one column in one role, never one column's content against rows dropped by another column.

**Evidence for finding 2** (pure noise, 500 rows each present twice, 5 feature columns):

| Variant | Build | Reference | Copies counted | Warnings |
|---|---|---|---|---|
| Exact copies, default | stops, `holdout_copies_in_training` (178 of 250) | not run | 178 | n/a |
| Plus one unique noise column, classification, full | skill 0.84 and 0.90 (two tables), p 0.001, forest-0 | 0.76, p 0.001 | 0 | none |
| Same, regression, full | skill 0.42, p 0.001, boosting-1 | 0.48 | 0 | none |
| Second copy differs by 1e-9 in one column, fast | skill 0.74, p 0.005, tree-0 | 0.62 | 0 | none |

**Judge-awareness.** None found. `mlprofessor/` reads no environment variable, names no test or gate machinery, and special-cases no fixture value. Its two broad `except` clauses are the ones steps 9 and 10 require. All 24 hits in the build report are harness lab scripts under `docs/measurements/`, not application code.

**Failure-mode coverage**

- **F9:** weak. One gate check, exact copies only (finding 2).
- **F2 and F1:** the gate bands (spec 10.1) are measured and not vacuous, but the margins to the broken copies are narrow at one fixed seed: `leak.regression` 0.07, `leak.excluded` 0.09, `model.noise_clf` 0.12.
- **F3:** caught only by the exact recomputation from `predictions.csv`; the noise band itself does not catch it, as the spec says.
- **The rest:** I found nothing weak in F4 to F8 and their variants from the check list and the bands. I did not read the gate's check code or the test bodies, so this rests on those two sources and on the self-test's claim.
- **Tests:** whether the suite fails on each of the 27 faulty variants rests on `verify/mutation_check.py`, which I did not run.

**What I ran**

- **Gate:** full run, `GATE: PASS (33 checks)`, exit 0.
- **Suite:** `1321 passed, 1 warning in 149.78s`, exit 0. The one warning is a NumPy deprecation in `tests/test_split_time.py:114`.
- **Recomputation:** on a 1,203-row classification table under the full profile, I recomputed the score, the bootstrap interval and the permutation p-value independently with scikit-learn. All three matched (AUC 0.84499, interval 0.8005 to 0.8881, p 1/1001), and the row and column counts matched the source.
- **Time order:** a time-ordered run with a shuffled file, gap 2 and the fast cap. The holdout was exactly the latest 500 rows and both baselines matched my own.
- **Differential:** 170 generated tables of mixed column types, targets and time columns, through the build and the reference under the fast profile.
  - All 170 outcomes agree: 73 ran (55 time-ordered, 18 random), 97 stopped with the same named error, and no raw exception.
  - Every time-ordered score, skill, baseline, fold size and `n_imputed` agrees to 1e-7.
  - The only field differences are finding 3 and `kind` on excluded text columns, which spec 5.2 leaves open.

**What I could not verify**

- The gate's self-test (`--selftest`, nine to twelve minutes) and `verify/mutation_check.py` (27 suite runs): not run.
- The CSV and CLI path beyond the gate's own runs; my probes used DataFrames.
- A full-profile differential, run time at scale, and the README's commands beyond what the suite covers.
- Whether `tests/` and `verify/` are unchanged since the commit that says the mutation check passed: no git in this role.
- One of my differential commands was refused by the guard as a write to `verify/reference`. It wrote nothing; I re-ran it in a shape the guard accepts.
