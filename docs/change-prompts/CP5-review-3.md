# CP5 — phase 1: the third review's findings

**Session:** started from `E:\CODE\ml-professor` by
`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop\run_loop.py harness --open`,
on `fable`. You have no memory of any earlier session. Everything you need is
in the repo and in this file.

**If anything here is wrong** (a file is not as described, a change cannot be
made as written, a decided behaviour contradicts a locked test, a measurement
argues against a decision), say so plainly and stop. Do not work round it.

## Context

- Phase 1 of ML Professor is built and has been through three passes. The
  application is the package `mlprofessor` at the repo root, with
  `requirements.txt`, `README.md` and `examples/customers.csv`. It passes the
  full gate (`verify/smoke_test.py`, 31 checks, 12 failure modes) and the
  full suite (34 test files under `tests/`, 411 cases).
  `verify/mutation_check.py` passes: the good reference fails nothing, and
  each of 19 variants fails at least one test.
- **The application and every existing test file are locked and out of
  bounds for this session.** You change the judge only. New tests are written
  afterwards by the tests stage; the application is changed afterwards by the
  build stage.
- The third review returned `VERDICT: FIX FIRST`. It is saved as
  `docs/reviews/phase1-review-3.md`. The owner accepted it in full and
  recorded D49 to D52 in `docs/DECISIONS.md`.
- This is the fourth pass. CP4 (`docs/change-prompts/CP4-review-2.md`) was the
  third; the "Third pass (CP4)" section of `plan/test-plan.md` and tasks T24
  to T30 show the shape to follow.
- Read first, in this order: `docs/DECISIONS.md` (D49 to D52, then the rest),
  `docs/reviews/phase1-review-3.md`, `spec.md`, `plan/test-plan.md`,
  `plan/tasks.json`, `verify/smoke_test.py`, `verify/mutation_check.py`,
  `verify/reference/mlprofessor/`, `CLAUDE.md`, `docs/BACKLOG.md`. Read the
  application and the existing tests as far as you need to; change neither.

## Before you write anything

Report, in a short list: the files and sections you will change; the new test
files and tasks you plan; **each place where a decided behaviour would
contradict an existing locked test** (D49 is the likeliest: look for any
locked test or gate dataset that profiles a table with repeated rows under
the random scheme); the choices this file leaves to you and what you will
choose; and the outcome of the type matrix of section H against the
application as it stands. Then wait for my go.

## The changes, as a user would observe them

Finding numbers are those of the third review.

### A. Copies of a holdout row in training (finding 2, D49)

- Under the random scheme, a holdout row has a copy in training when a
  training row has the same value in every feature column and the same
  target. Pin how values are compared (missing cells, white space, floats).
- A new argument `duplicates`, `"error"` by default, with `"keep"` as the
  other value; `--duplicates` on the command line. Any other value is an
  input error.
- With the default, when more than the cut-off share of holdout rows have a
  copy in training, the run stops with a new named error. Its message gives
  the count and the share, and says the caller can remove the copies or pass
  `duplicates="keep"`.
- With `"keep"`, the run goes ahead as today. The output records the choice,
  the count and the share, and a warning says the headline may be inflated.
- At or below the cut-off, the run goes ahead and the count is reported.
- Under time order the rule does not apply; say so in the limitations text
  or wherever the spec states the scope.
- The cut-off starts at 1% of holdout rows. Measure and pin it: over 30
  seeds, the share on every existing gate dataset and on the example table
  (all must stay below, and no existing run may stop), and on duplicated
  data (must be far above). Report both ranges.
- The README sentence that no fitted step has seen the holdout rows must
  stay true as worded, or be worded so that it is.

### B. Whole numbers are compared exactly (findings 1 and 3, D50)

- Two distinct whole-number timestamps are never merged. Prefer comparing
  integer time keys as integers. If that contradicts a locked test of
  `stages.read_time` or `stages.split_time`, the run instead stops with
  `time_column_invalid` when two distinct integers would become one number.
  Say in your plan which it is.
- Distinct values of a whole-number column are counted exactly at any
  magnitude, so a column of large distinct integers is not `constant`.
- A column of whole numbers too large for 64 bits: as a spare column it is
  excluded with a reason and a warning; as the target or the time column it
  is a named error. No raw exception in any role, from a CSV or a DataFrame.

### C. Number columns as a database driver returns them (finding 4, D51)

- A DataFrame column of generic objects holding only decimal numbers,
  booleans or whole numbers, with or without empty values, is read as
  numeric. Pin exactly which mixtures count.
- Every column still excluded as `unsupported_type` gets a warning naming it.

### D. Fractional seconds (finding 10, D52)

One to nine fractional digits are accepted in an ISO 8601 time value. A
refused value's message says what is wrong with it.

### E. The fast profile's cap (finding 9, D52)

When the capped rows are too few to profile and the same call with
`profile="full"` would get past that point, the error is the fast profile's
own, naming the cap and `profile="full"`.

### F. The six test gaps (findings 5 to 8, D52)

The built code has none of these faults. For each, the test plan demands a
test that fails on the fault, and the reference gains the fault as a
suite-only variant:

| The fault | Finding |
|---|---|
| the interval uses the 5th and 95th percentiles and still reports level 0.95 | 5 |
| the leak screen ignores the previous-value baseline, so a lag of a persistent series is excluded | 6 |
| the fast profile keeps the first rows of the file, not a sample from the seed | 7 |
| `cv_std` is the population deviation, not the sample deviation | 7 |
| tied predictions are mishandled in a resampled ROC AUC | 8 |
| `n_imputed` is counted over every loaded row | 8 |

The review gives a workable test for each; read it.

### G. Gate checks for the two routes to a wrong headline

- **Copies (A).** A new dataset: pure noise with every row present twice,
  random scheme. The right conclusion: the run stops with the new error. A
  new failure mode, with a broken copy of the reference that ignores copies,
  and a check that names it. The check is structural: it asserts the exit,
  the error and the absence of a headline, not a score.
- **Integer timestamps (B).** A new dataset: a random walk with unique
  64-bit integer nanosecond timestamps above 2 to the power 53, under time
  order. The right conclusion: the number of distinct timestamps the tool
  reports equals the gate's own count, and the headline baseline is the
  previous value. A new failure mode, with a broken copy that reads integer
  keys as floats, and a check that names it. Prefer structural assertions;
  add a band only if you measure it as CP1 did.
- Both datasets are added last, so no seed moves. Watch each check fail
  against the reference as it is now, before you change the reference.
- The proof table of the self-test must still show every pair of a check and
  a mode it names failing.

### H. The type matrix (D52)

- A new test file that holds the whole class of fault the last two reviews
  kept finding one at a time. For every kind of pandas column, in each of
  three roles (a feature, the target, the time column), with ordinary values
  and with extreme ones, `profile()` must end in a result or in a named
  error (a subclass of `ProfileError`). Never a raw exception. When it ends
  in a result: every column appears once in the accounting, and every
  number-typed field is finite or null with its reason.
- Kinds to cover, at least: each signed and unsigned integer width; 32-bit
  and 64-bit floats; nullable integer, float and boolean; plain boolean;
  text as object and as the string type; category, ordered and not; datetime
  with and without a zone; timedelta; period; interval; complex; bytes;
  decimal objects; Python integers beyond 64 bits; mixed objects; lists;
  an all-missing column; a sparse column.
- Extremes to cover, where the kind allows: the largest and smallest value
  of the type; whole numbers just above 2 to the power 53; magnitudes near
  1e300 and 1e-300; infinities; not-a-number; negative zero.
- Run the matrix now against the application, with a throwaway prototype
  outside the repo, and put the outcome in your plan: how many cases, how
  many end in a raw exception today, and what they are. Every such case is a
  red test and belongs to a task. If the list is long, group it by cause and
  propose the tasks; if a cause needs a decision the log does not hold, name
  it and wait.
- The matrix is parametrised by kind, role and extreme, so a failure names
  its case.

## The judge, updated together

- **Reference implementation:** implements A to E, and whatever H requires.
  Every existing variant stays broken in exactly one way. The two gate
  variants of G and the six suite-only variants of F are added.
- **Gate:** the two datasets, checks and failure modes of G. No band, seed
  or existing dataset changes. No existing run may stop because of A.
- **`verify/mutation_check.py`:** the new variants. Each must fail at least
  one test once the tests stage has run.
- **Spec:** every behaviour of A to E and the contract of H, pinned so the
  test author can write against it without seeing code. Bring the decision
  index up to D52.
- **Test plan:** a "Fourth pass (CP5)" section, with the same ground rules as
  the third: the application exists; every existing test file, helper and
  fixture is final and must not be edited; new shared helpers go in a new
  file; every new test is listed by name as red today or green today; every
  green guard must fail on a named variant.
- **Task plan:** new tasks appended after T30. Do not change the id, tests or
  gate checks of T1 to T30. Every new gate check and every new test file
  belongs to exactly one new task. A task that decides which rows a number is
  computed on, or whether a number may be reported, is `complex`.
- **`CLAUDE.md` and the skill:** the new matrix, D1 to D52, the new argument,
  anything a builder must now know.

## Proof before you stop

1. The gate's self-test passes: the good fixture twice with identical values,
   every failure mode caught by a check that names it, and the proof table
   complete.
2. Each new gate check failed against the unfixed reference and passes
   against the fixed one.
3. No other gate value moved at the gate seed apart from what these changes
   require; say how you compared and what moved.
4. `verify/mutation_check.py` against the existing 411 cases: the output,
   pasted. The good reference must fail nothing: if it does, a decided
   behaviour contradicts a locked test, so stop.
5. Each new test of the plan checked with a throwaway prototype outside the
   repo, against the application and against the fixed reference: red or
   green today, as listed. Each green guard fails on its variant.
6. The measurements behind the cut-off of A, with both ranges.
7. `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py doctor`
   prints `SELFTEST: PASS`, a per-edit gate time within the 20 second budget,
   and a valid plan. A warning about uncommitted changes is expected.
8. The output of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/smoke_test.py --matrix`
   and of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py build --dry-run`,
   pasted.
9. Measurement scripts saved under `docs/measurements/CP5/` with a
   `README.md` (D35).
10. A final report: what changed in each file; the new test files and tasks
    with tiers; which new tests are red and which green today; the type
    matrix's outcome; and every place where the facts argue against D49 to
    D52.

Say which gate checks fail against the application as it stands, and which
task will fix each.

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
