# CP4 — phase 1: the second review's findings

**Session:** started from `E:\CODE\ml-professor` by
`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop\run_loop.py harness --open`,
on `fable`. You have no memory of any earlier session. Everything you need is
in the repo and in this file.

**If anything here is wrong** (a file is not as described, a change cannot be
made as written, a decided behaviour contradicts a locked test, a measurement
argues against a decision), say so plainly and stop. Do not work round it.

## Context

- Phase 1 of ML Professor is built and has been through two passes. The
  application is the package `mlprofessor` at the repo root, with
  `requirements.txt`, `README.md` and `examples/customers.csv`. It passes the
  full gate (`verify/smoke_test.py`, 31 checks) and the full suite (27 test
  files under `tests/`, 370 cases). `verify/mutation_check.py` passes: the
  good reference fails nothing, and each of F1 to F8 and M1 to M5 fails at
  least one test.
- **The application and every existing test file are locked and out of
  bounds for this session.** You change the judge only. New tests are written
  afterwards by the tests stage; the application is changed afterwards by the
  build stage.
- The second review returned `VERDICT: FIX FIRST`. It is saved as
  `docs/reviews/phase1-review-2.md`. The owner accepted it in full and
  recorded D44 to D47 in `docs/DECISIONS.md`.
- This is the third pass. CP3 (`docs/change-prompts/CP3-review-1.md`) was the
  second; the "Second pass" section of `plan/test-plan.md` and tasks T18 to
  T23 show the shape to follow.
- Read first, in this order: `docs/DECISIONS.md` (D44 to D47, then the rest),
  `docs/reviews/phase1-review-2.md`, `spec.md`, `plan/test-plan.md`,
  `plan/tasks.json`, `verify/smoke_test.py`, `verify/mutation_check.py`,
  `verify/reference/mlprofessor/`, `verify/gatekit.py` (the self-test),
  `CLAUDE.md`. Read the application and the existing tests as far as you need
  to; change neither.

## Before you write anything

Report, in a short list: the files and sections you will change; the new test
files and tasks you plan; each place where a decided behaviour would
contradict an existing locked test; the choices this file leaves to you and
what you will choose. Then wait for my go.

## The changes, as a user would observe them

Finding numbers are those of the second review.

### A. No number that is not finite is written as empty (findings 1, 5, 11; D44)

- A configuration whose fold score is not finite is failed, exactly as one
  that raised: status, error text, warning, never chosen.
- If the chosen configuration's holdout score is not finite, or a baseline's
  score is not finite, the run stops with a new named error, code
  `non_finite_score`. The message says which number and the likely cause.
  Decide, and say in your plan, how this sits with the fallback of D38: a
  holdout score that is not finite must never become a headline, and the
  caller must get one clear error, not a list of failed configurations, when
  the cause is the target itself.
- A p-value is reported only when skill is a number.
- The three quiet fallbacks of finding 11 either raise a named error or are
  shown to be unreachable and pinned as such. None stays as a silent
  default.
- Both inputs of finding 1 (one cell near the largest double among ordinary
  values; a target scaled by 1e160), under both schemes, must give either a
  named error or finite values in every number-typed field.
- Tests assert on the raw values the application produced, before any helper
  that maps not-a-number to empty. New helpers go in a new file.

### B. The seed's range (finding 2, D45)

`seed` is a whole number from 0 to 4,294,967,295. Anything else is an
`input_error` before anything is loaded, from the function and from the
command line (exit 3, one line, no traceback). The largest allowed seed runs.

### C. The console summary prints significant digits (finding 3, D46)

Each number in the summary agrees with `profile.json` to four significant
digits, at any scale. Zero prints as zero. State the rule as a relative
tolerance, so that it is stricter than the existing one and no locked test
is contradicted.

### D. White space round a target value (finding 7, D46)

Surrounding white space is not part of a target value: `yes` and `yes ` are
one class, in the task decision, the class labels and `y_true`. Pin what
`positive_class` and `predictions.csv` then show.

### E. The fast profile's error (finding 10, D46)

`fast_profile_too_few_timestamps` is raised only when the same call with
`profile="full"` would get past that point. When the full profile would also
have too few timestamps, the ordinary `too_few_rows` is raised.

### F. The README (finding 9, D46)

The README says correctly which rows the fast profile keeps: chosen from the
seed under the random scheme, the most recent whole timestamps under time
order. A test holds it to that without pinning the wording.

### G. Lines before the header (finding 8, D46)

Tests only, unless you find a fault: a line of form feed, of no-break space,
or of two quotes round a space, before a two-column header, gives
`input_error`. Say what the application does today for each.

### H. Every gate check that names a failure mode is seen to fail (finding 4, D46)

- Four checks sit in the matrix and have never failed in the self-test:
  `data.columns_accounted`, `models.all_accounted`, `data.long_rows_stop`
  and `time.dayfirst_stops`.
- Add four broken copies of the reference, each failing the one check: a
  column lost from the accounting; a model family lost from the models
  table; a long row read silently, with no error; a day-first date parsed
  leniently, so the run goes ahead on the wrong order.
- Register each as a failure mode of its own with gatekit, under ids that
  show which family it belongs to, and change `catches` so that after this
  session **every check that names a mode fails on at least one fixture of
  that mode in the self-test**. Print a table that proves it.
- Each new variant must also fail at least one test of the suite; add them
  to `verify/mutation_check.py`. If the existing tests do not catch one, the
  test plan gains a test that does.
- Report only: list the checks that carry a `why` in place of `catches`, and
  say for each whether it has ever been seen to fail on any fixture. Change
  nothing for these.

### I. The self-test leaves the build's run folder alone (finding 6, D46)

After `--selftest`, `verify/artifacts/runs/` must still hold the output of
the last ordinary gate run, not a broken fixture's. The run folder is chosen
in `verify/smoke_test.py`, so fix it there. If it cannot be done without
editing `verify/gatekit.py`, say so and leave it.

## The judge, updated together

- **Reference implementation:** implements A to E. F1 to F8 and M1 to M5
  stay broken in exactly one way each. The four variants of H are added.
- **Gate:** the fixtures and `catches` of H; the run folder of I. Add a gate
  dataset and check for A only if you can say what a test cannot catch
  there. No band, seed or existing dataset changes.
- **Spec:** every behaviour of A to G, pinned so the test author can write
  against it without seeing code. Bring the decision index up to D47.
- **Test plan:** a "Third pass" section, with the same ground rules as the
  second: the application exists; every existing test file, helper and
  fixture is final and must not be edited; new shared helpers go in a new
  file; every new test is listed by name as red today or green today; every
  green guard must fail on a named variant, and `verify/mutation_check.py`
  is how that is judged.
- **Task plan:** new tasks appended after T23. Do not change the id, tests or
  gate checks of T1 to T23, apart from `catches` in the gate itself. Every
  new gate check and every new test file belongs to exactly one new task. A
  task that decides whether a number may be reported is `complex`.
- **`CLAUDE.md` and the skill:** the new matrix, D1 to D47, anything a
  builder must now know.

## Proof before you stop

1. The gate's self-test passes: the good fixture twice with identical values,
   every failure mode caught by a check that names it, and the table of H
   showing that every check that names a mode failed on a fixture of it.
2. No gate value moved at the gate seed apart from what these changes
   require; say how you compared and what moved.
3. `verify/mutation_check.py` against the existing 370 cases: the output,
   pasted. The good reference must fail nothing: if it does, a decided
   behaviour contradicts a locked test, so stop. Each new variant of H must
   fail at least one existing test, or the test plan names the new test that
   will catch it.
4. Each new test of the plan checked with a throwaway prototype outside the
   repo, against the application and against the fixed reference: red or
   green today, as listed.
5. After `--selftest`, the run folder holds the last ordinary run's files;
   show how you checked.
6. `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py doctor`
   prints `SELFTEST: PASS`, a per-edit gate time within the 20 second budget,
   and a valid plan. A warning about uncommitted changes is expected.
7. The output of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/smoke_test.py --matrix`
   and of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py build --dry-run`,
   pasted.
8. Any measurement script saved under `docs/measurements/CP4/` with a
   `README.md` (D35).
9. A final report: what changed in each file; the new test files and tasks
   with tiers; which new tests are red and which green today; the report-only
   list of H; what you found for G; and every place where the facts argue
   against D44 to D46.

Say whether the full gate passes against the application as it stands, and
if not, which checks fail and which task will fix each.

## Out of bounds

- The application: `mlprofessor/` at the repo root, `requirements.txt`,
  `README.md`, `examples/`.
- The `tests/` folder, in every form: no edits, no new files, no cache files.
- Kit files: `verify/gatekit.py`, `.claude/hooks/`, `.claude/settings.json`,
  `loop/`.
- `mlprof_env`, `BOOTSTRAP.md`, `docs/DECISIONS.md`, `docs/BACKLOG.md`,
  `docs/reviews/`.
- Git, in any form.
- Any band, seed or existing dataset.
- The methodology: flag what the facts argue against, and stop.

## Stop condition

Give the final report and stop. Do not write tests. Do not build.
