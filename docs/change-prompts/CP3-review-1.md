# CP3 — phase 1: the first review's findings

**Session:** started from `E:\CODE\ml-professor` by
`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop\run_loop.py harness --open`,
on `fable`. You have no memory of any earlier session. Everything you need is
in the repo and in this file.

**If anything here is wrong** (a file is not as described, a change cannot be
made as written, a decided behaviour contradicts a locked test, a measurement
argues against a decision), say so plainly and stop. Do not work round it.

## Context

- Phase 1 of ML Professor is built. The application is the package
  `mlprofessor` at the repo root (17 modules), with `requirements.txt` and
  `README.md`. The build passed the full gate (`verify/smoke_test.py`, 30
  checks) and the full suite (17 files under `tests/`, 308 tests).
- **The application and the existing tests are both locked and out of bounds
  for this session.** You change the judge only: the gate, the reference
  implementation, the spec, the test plan, the task plan, `CLAUDE.md`, the
  skill. New tests are written afterwards by the tests stage; the application
  is changed afterwards by the build stage.
- The first review returned `VERDICT: FIX FIRST`. It is saved as
  `docs/reviews/phase1-review-1.md`. The owner accepted it in full and
  recorded D36 to D39 in `docs/DECISIONS.md`.
- Read first, in this order: `docs/DECISIONS.md` (D36 to D39, then the rest),
  `docs/reviews/phase1-review-1.md`, `spec.md`, `plan/test-plan.md`,
  `plan/tasks.json`, `verify/smoke_test.py`, `verify/reference/mlprofessor/`,
  `CLAUDE.md`, `loop/run_loop.py` (the `tests` and `build` commands, so you
  know how a second pass over an existing application behaves),
  `loop/prompts/test-author.md`. Read the application and the existing tests
  as far as you need to; change neither.

## Before you write anything

Report, in a short list: the files and sections you will change; the new test
files and tasks you plan; each place where a decided behaviour would
contradict an existing locked test; the choices this file leaves to you and
what you will choose. Then wait for my go.

## The changes, as a user would observe them

Finding numbers are those of the review.

### A. Skill when a baseline's error is zero (finding 1, D36)

- When the headline baseline's squared error on the holdout is exactly zero,
  `headline.skill`, `skill_low` and `skill_high` are `null`, a warning with
  the code `skill_undefined` states the reason, and the run completes with
  everything else reported. No error, no infinity, no zero in its place.
- The same rule wherever skill is computed: a bootstrap resample whose
  baseline error is zero is left out and counted; a single-column skill that
  cannot be computed flags no column and raises no `strong_single_column`
  warning.
- Pin each of these in the spec, including what `stages.skill` returns or
  raises for a zero baseline, without contradicting a locked test.

### B. Colliding feature names (finding 2, D37)

- If two columns of the preprocessed matrix would have the same name (a
  numeric column whose name is a categorical column's name, an equals sign
  and one of its levels, such as `c=red`; or two categorical columns
  whose names and levels combine to the same text), the run stops with a new
  named error, code `feature_name_collision`, naming the columns involved,
  before any model is fitted.
- Pin exactly which rows' levels are considered, so that the outcome does not
  depend on which rows a fit happens to see.

### C. A configuration that fails after it is chosen (finding 8, D38)

- If the chosen configuration raises on the final fit or while predicting the
  holdout: it is marked `failed` with its error text, keeps its fold scores
  in the models table, and gets a `model_failed` warning. The next best by
  `cv_mean` is chosen, and so on. If none succeeds: `all_models_failed`.
- The headline comes from exactly one configuration. Restate "only the chosen
  configuration predicts the holdout" so that it stays true and testable
  under this rule.

### D. Small defects (findings 9, 10, 11, 12; D39)

- An `out_dir` that cannot be used (it names an existing file, for one) is an
  `input_error`, raised before anything is loaded.
- After `profile()` returns or raises, process-wide settings are as the
  caller left them. `csv.field_size_limit` is the one found.
- The `no_features` message names each excluded column and its reason.
- The console summary also shows the interval on the score and on skill, the
  p-value or the word that there is none, and one line per warning giving its
  code and column. Pin the presence of those values; the wording stays open.

### E. Two CSV edges (D39)

- A file that is not valid UTF-8 is an `input_error` wherever the first bad
  byte sits, including in the header line or the first data row.
- Leading lines of only white space are skipped when the header is read, as
  pandas skips them, so the tool and pandas agree on which line is the header.
- The reference implementation has both faults today. Check whether the
  application has them too and say which.

### F. The five test holes (findings 3 to 7, and 13; D39)

The built code has none of these faults. Nothing in the suite would catch
them. For each, the test plan must demand a test that fails on the fault:

| Mutant | The fault | Review finding |
|---|---|---|
| M1 | the final fit's imputation and scaling statistics are learned from training and holdout rows together | 3 |
| M2 | one-hot levels are learned from every row of the table in every fit | 4 |
| M3 | the leak screen's trees are fitted on rows that include the holdout, and the missing-by-class rule counts every row | 5 |
| M4 | the block bootstrap draws timestamps independently and still reports `bootstrap_blocks` | 6 |
| M5 | the permutation p-value counts shuffles strictly greater than the observed skill | 7 |

The review gives a workable test for each; read it. Then correct the sentence
in `CLAUDE.md` that says the small forms of F1 are caught by the recording
fakes: say what catches each form once this change is in.

### G. One measurement of the built library (D39)

Report only. Run the application, writing its output to a temporary folder
outside the repo:

- a full-profile run on a synthetic table of 100,000 rows by 100 columns, for
  classification and for regression, on all cores: the seconds for each,
  against the 10-minute budget of D16;
- full-profile runs on a table of about 5,000 rows by 30 columns at
  `n_jobs` 1 and 64, both tasks: whether `profile.json` and `predictions.csv`
  are identical apart from `meta.timing` and the recorded `n_jobs`.

## The judge, updated together

- **Reference implementation:** implements A to E. The eight broken variants
  F1 to F8 stay broken in exactly one way each. Add M1 to M5 as five more
  values of the same switch. They are for the tests, not for the gate: do not
  register them as gate failure modes, since no gate check can catch them.
- **Gate:** one new dataset, added last so no seed moves: a regression series
  with unique timestamps whose target is flat across the holdout. One new
  check that the run exits 0, `skill` is `null`, the `skill_undefined`
  warning is present, and both baselines are reported. Give it a written
  `why`, unless a failure mode honestly names it. Watch it fail against the
  reference as it is now, before you change the reference, and report that.
  Add a gate check for B, C or E only if you can say what a test cannot
  catch there.
- **`verify/mutation_check.py` (new):** a script the owner runs after the
  tests stage. For the good reference and for each of F1 to F8 and M1 to M5,
  it builds a temporary project outside the repo (a copy of `tests/`, the
  reference with that switch as the folder `mlprofessor`, and copies of
  `README.md` and `requirements.txt`), runs the whole suite there, and prints
  one line per case with the number of failing tests and the names of up to
  five. It exits 0 only if the good reference fails nothing and every variant
  fails at least one test. It writes nothing inside the repo. Note the trap:
  the application now exists at the repo root and `tests/conftest.py` puts
  the repo root first on the import path, so setting `PYTHONPATH` to the
  reference no longer reaches the reference.
  Run it now, against the existing 308 tests, and report the result. The
  expected result today is that some mutants fail no test; that is the
  finding, measured.
- **Spec:** every behaviour of A to E, pinned so the test author can write
  against it without seeing code: the errors table, the warning codes, the
  headline fields (`skill` and its interval may now be `null`), steps 9 to
  12, the edges, the loading rules, the list of what is open. Bring the
  decision index up to D39.
- **Test plan:** new test files only, each belonging to one new task. State
  these ground rules at the top of the new section, in words the test author
  cannot miss:
  - the application already exists, and this is a second pass;
  - the existing 17 test files, `tests/conftest.py`, `tests/mlp_helpers.py`
    and `tests/fixtures/` are final and must not be edited in any way; new
    shared helpers go in a new file;
  - which new tests must fail today (A to E, wherever the application does
    not yet behave as decided) and which must pass today (F, the guards),
    listed by name, so the test author does not weaken a guard to make it
    fail or a red test to make it pass;
  - every guard test of F must fail on its mutant, and
    `verify/mutation_check.py` is how that will be judged.
- **Task plan:** new tasks appended after T17. Do not change the id, tests or
  gate checks of T1 to T17. Every new gate check and every new test file
  belongs to exactly one new task; every new task has at least one. A task
  that decides which rows or which configuration a number comes from is
  `complex`. The guard tests of F need a task too, whose brief says plainly
  that no code change is expected and that a failure means the application
  is wrong, not the test.
- **`CLAUDE.md`:** the corrected sentence, the new matrix, D1 to D39.

## Proof before you stop

1. The gate's self-test passes: the good fixture twice with identical values,
   each of F1 to F8 caught by a check that names it.
2. The new check failed against the unfixed reference and passes against the
   fixed one.
3. No other gate value moved at the gate seed; say how you compared.
4. `verify/mutation_check.py` run against the existing suite: the output,
   pasted. The good reference must fail nothing. If it fails any of the 308
   existing tests, a decided behaviour contradicts a locked test: stop.
5. Each of M1 to M5 changes behaviour: for each, one crafted input on which
   the mutant's output differs from the good reference's, with both values.
6. `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py doctor`
   prints `SELFTEST: PASS`, a per-edit gate time within the 20 second budget,
   and a valid plan. A warning about uncommitted changes is expected.
7. The output of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/smoke_test.py --matrix`
   and of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py build --dry-run`,
   pasted. Say what the dry run shows for T1 to T17.
8. The measurements of G, and any scripts, saved under
   `docs/measurements/CP3/` with a `README.md` (D35).
9. A final report: what changed in each file; the new test files and tasks
   with tiers; which new tests are expected red and which green today; what
   you found for E in the application; the measurements; and every place
   where the facts argue against D36 to D39.

The full gate will fail against the application until the build stage has
run, because of the new check. That is expected; say so in the report.

## Out of bounds

- The application: `mlprofessor/` at the repo root, `requirements.txt`,
  `README.md`.
- The `tests/` folder, in every form: no edits, no new files, no cache files.
- Kit files: `verify/gatekit.py`, `.claude/hooks/`, `.claude/settings.json`,
  `loop/`.
- `mlprof_env`, `BOOTSTRAP.md`, `docs/DECISIONS.md`, `docs/reviews/`.
- Git, in any form.
- Any band, seed or existing dataset.
- The methodology: flag what the facts argue against, and stop.

## Stop condition

Give the final report and stop. Do not write tests. Do not build.
