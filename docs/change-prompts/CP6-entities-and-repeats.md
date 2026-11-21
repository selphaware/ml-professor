# CP6 — phase 1, pass 5: entities, repeated rows, and stopping where the tool used to warn

**Session:** started from `E:\CODE\ml-professor` by
`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop\run_loop.py harness --open`,
on `fable`. You have no memory of any earlier session. Everything you need is
in the repo and in this file.

**If anything here is wrong** (a file is not as described, a change cannot be
made as written, a decided behaviour contradicts a locked test in a way D59
does not cover, a measurement argues against a decision), say so plainly and
stop. Do not work round it.

## Context

- Phase 1 of ML Professor is built and has been through four passes. The
  application is the package `mlprofessor` at the repo root, with
  `requirements.txt`, `README.md` and `examples/customers.csv`. It passes the
  full gate (`verify/smoke_test.py`, 33 checks, 14 failure modes) and the
  full suite (42 test files under `tests/`, 1,321 cases).
  `verify/mutation_check.py` passes: the good reference fails nothing and
  each of 27 variants fails at least one test.
- **The application and the existing tests are locked and out of bounds for
  this session.** You change the judge only. New tests are written afterwards
  by the tests stage; the application is changed afterwards by the build
  stage.
- The fourth review returned `VERDICT: FIX FIRST`
  (`docs/reviews/phase1-review-4.md`). The owner's standard is now D56:
  nothing known to be wrong is left for later. D57, D58 and D59 are the
  decisions this pass builds.
- The technical lead ran the built library on real public data
  (`docs/measurements/real-data-2026-11-21/README.md`). Read it: it shows in
  real numbers the three inflated headlines this pass removes.
- This is the fifth pass. CP5 (`docs/change-prompts/CP5-review-3.md`) was the
  fourth; the "Fourth pass (CP5)" section of `plan/test-plan.md` and tasks
  T31 to T38 show the shape to follow.
- Read first, in this order: `docs/DECISIONS.md` (D56 to D59, then D49, D53
  to D55, D7, D10, D31, then the rest), `docs/reviews/phase1-review-4.md`,
  the real-data write-up, `spec.md`, `plan/test-plan.md`, `plan/tasks.json`,
  `verify/smoke_test.py`, `verify/mutation_check.py`,
  `verify/reference/mlprofessor/`, `CLAUDE.md`, `docs/BACKLOG.md`,
  `loop/run_loop.py` (`reverify_done`, the tests stage). Read the application
  and the existing tests as far as you need to; change neither.

## Before you write anything

This pass changes more of the contract than any before it, so the plan matters
more. Report, in a short list, and then wait for my go:

1. The files and sections you will change, and the new test files and tasks.
2. **Every existing test and gate check that a decided behaviour
   contradicts,** by name, and for each how D59 applies: the loosened form
   that passes both before and after, and the new test that pins the new
   behaviour. Search by behaviour, not only by keyword: the signature, the
   undeclared-date warning path, the panel warning path, the README's
   commands on `examples/customers.csv` (it has a date column), the gate's
   own runs, and the type matrix.
3. Which finished tasks the next build would reopen, and why. The aim is
   none.
4. **The repeat screen you propose (section B), with its measurements.** This
   is for approval before anything is built on it.
5. The choices this file leaves to you, and what you will choose.

## The changes, as a user would observe them

### A. A declared entity column (D57)

- A new argument `group_col`, and `--group-col` on the command line. The
  column gets its own role in the column accounting and is not a feature.
- **Random scheme.** The rows of one entity stay on one side of the holdout
  split, and inside the training rows on one side of every fold. Keep the
  holdout near 25% of the rows and, for classification, the classes as close
  to balanced across the sides as whole entities allow; pin the properties,
  not the row numbers. The output says the scheme is grouped and reports the
  number of entities on each side.
- **Time order.** The split stays as it is. With unique timestamps inside
  each entity, the previous-value baseline is each entity's own previous
  value, the headline baseline is chosen as D10 chooses it, and the leak
  screen measures a lag against that baseline (D24). Pin what happens to an
  entity's first holdout row, to an entity with no earlier row, and to a
  repeated timestamp inside one entity.
- **Edges,** each a named error or a reported outcome: too few entities to
  split or to form five folds; an entity column with missing cells; an
  entity that holds one class; `group_col` equal to the target or the time
  column; every row its own entity (say what that means and whether it is
  allowed).
- The fast profile's row cap keeps whole entities where it can; pin it.

### B. A broader screen for repeated rows (D57, fourth review finding 2)

- Today a holdout row has a copy only when every feature column and the
  target agree (D49). One added column, a batch number, or a change of 1e-9
  in one column makes the count zero. The real-data run measured skill 0.35
  to 0.44 on pure noise that way, against minus 0.13 without the copies.
- Design a rule that catches copies which differ in some columns, and stops
  the run as D49 does (the same argument, the same error family, the same
  share-and-count form unless you measure a better one). Starting points, to
  test and improve on: agreement on the target and on all feature columns
  but one; agreement on the target and exactly on two or more numeric
  columns whose values are otherwise almost never repeated.
- **Measure it, and bring the measurements to the plan.** Over 30 seeds:
  - must stop: every row twice with an added unique column; with a batch
    column; with a 1e-9 change in one column; in two columns; with two added
    columns; a 5% share of rows repeated with each of those disguises;
  - must not stop: every existing gate dataset under the random scheme,
    `examples/customers.csv`, and the clean tables of the real-data run that
    you can rebuild from scikit-learn's bundled data (breast cancer,
    diabetes);
  - report, and do not hide: what it does on a table of a few categories,
    where repeats are genuine (the Titanic file has 37 of 223 holdout rows
    with an exact copy; you cannot download it, so build a table of that
    kind), and the smallest disguise that defeats it.
- State in the spec what the rule cannot see. A copy altered in every column
  cannot be told from a different row; the control for that is `group_col`.
- Run time: the screen must not change the 10-minute budget at 100,000 rows
  by 100 columns. Time it at that size.

### C. `duplicates="group"` (D57)

A third value. The repeats the screen finds are treated as entities and kept
on one side of the split and of every fold, with the count reported and a
warning that says so. Pin how it combines with a declared `group_col`.

### D. An undeclared date column stops the run (D58, part 1)

- When a column would be excluded as a date and no time column is declared,
  the run stops with a new named error. Its message names the column and the
  two ways on.
- The caller may state that row order does not matter. Choose the form that
  keeps the first nine parameters and their defaults as they are (a value of
  `time_col` other than a column name is one way; say what you choose and
  why), and give the command line the same choice. With that statement the
  run goes ahead under the random scheme, the choice is recorded, and the
  existing caveat and warning stand.
- The message of the copies error (D53) still mentions an undeclared date
  column where that applies; say how the two errors are ordered.

### E. A panel with no declared entity (D58, part 3)

Under time order, when timestamps repeat, no `group_col` is declared and the
target meets the condition that today raises `autocorrelated_panel_target`,
the run stops and asks for the entity column, or for an explicit statement
that there are no entities. With that statement the run goes ahead as today
and the warning stands.

### F. The fourth review's other findings (D57)

1. **Finding 1.** Each object column is read once, over all loaded rows, as
   the spec's step 1a says. An odd cell in a row that is dropped for another
   column's missing cell gives the same named error as in a usable row, and
   exact integer keys are decided on the loaded column. The review lists
   five tables; each becomes a test. Add the fault as a suite-only variant.
2. **Finding 3.** A target's infinite cells are counted in its column record,
   as section 5.2 reads; say the same for the time column; bring the
   reference into line.
3. **Finding 4.** The finite check covers every part of the output, so a
   float that is not finite can never be written as null from any part.
4. **Finding 5.** A fault in the tool's own scoring of a fold stops the run;
   only a model's fit and prediction, and the non-finite check on its
   output, are recorded as a failed configuration.
5. **A three-class target.** The message says that three or more classes are
   not supported, and does not suggest declaring the task.

### G. The type matrix gains a role

Every kind of column as the entity column, with the same contract: a result
or a named error, never a raw exception.

## The judge, updated together

- **Reference implementation:** implements A to G. Every existing variant
  stays broken in exactly one way.
- **Gate, new failure modes, each with a dataset, a check that names it and a
  broken copy; each new dataset added last so no seed moves:**
  - an entity on both sides of the split despite a declared entity column
    (the repeated-entities pattern: many rows per entity with near-identical
    features and a label that is random per entity; the right conclusion is
    no skill, inside a band you measure as CP1 did, and no entity on both
    sides, which the gate counts for itself from `predictions.csv`);
  - a panel scored against the training mean despite a declared entity
    column (per-entity random walks with their own lag: the baseline is the
    previous value and skill is about zero, in a measured band);
  - disguised copies scored without the caller's say (noise with every row
    twice and an added column: the run stops);
  - an undeclared date column profiled under a random split without the
    caller's say (the run stops).
  Prefer structural assertions; where you assert a score, measure the band
  over 30 seeds for the good reference and each broken copy and report the
  gap. Watch each new check fail against the reference as it is now.
- **Existing gate runs** that the new stops would end (the undeclared-time
  run is one) are given the explicit statement they need, and nothing else
  about them changes. Say which, and move a check to a new task only where
  D54's reasoning applies.
- **`verify/mutation_check.py`:** the new variants. The self-test's proof
  table stays complete.
- **Spec:** every behaviour of A to G, pinned so the test author can write
  against it without seeing code. D7 and D10 are replaced in part by D58;
  the spec must not keep the old behaviour anywhere. Bring the decision
  index up to D59.
- **Test plan:** a "Fifth pass (CP6)" section with the ground rules of the
  fourth, and D59 in place of the single exception: list every existing test
  to be loosened, the exact loosened form, and the new test that pins the
  new behaviour. Every other line of every existing file stays as it is.
- **Task plan:** new tasks appended after T38. A task that decides which
  rows a number is computed on, or whether a run may go ahead, is `complex`.
  Order them so that each can be judged when it is built.
- **`README.md` is application code.** Its commands must still run after
  this pass, so a task rewrites them; say how the README tests stay green
  from the tests stage to that task.
- **`CLAUDE.md` and the skill:** the new arguments, modes, D1 to D59.

## Proof before you stop

1. The gate's self-test passes, with the proof table complete.
2. Each new gate check failed against the unfixed reference and passes
   against the fixed one.
3. No other gate value moved at the gate seed apart from what these changes
   require; say how you compared and what moved.
4. `verify/mutation_check.py` against the existing suite, pasted. The good
   reference may fail only the tests the plan names for loosening under D59;
   name them. Anything else: stop.
5. Prototypes of every new and every loosened test, outside the repo: each
   loosened test passes against the application as it stands and against the
   fixed reference; each new test is red or green today as listed; each
   green guard fails on its variant. With the prototypes laid over the suite
   (`--overlay`), the mutation check passes.
6. The measurements of B, and the bands of the two new score checks.
7. `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py doctor`
   prints `SELFTEST: PASS`, a per-edit gate time within the 20 second budget,
   and a valid plan. A warning about uncommitted changes is expected. If the
   per-edit time is over budget, say so and propose what to mark slow.
8. The output of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/smoke_test.py --matrix`
   and of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py build --dry-run`,
   pasted.
9. Measurement scripts saved under `docs/measurements/CP6/` with a
   `README.md` (D35).
10. A final report: what changed in each file; the tests to be loosened; the
    new test files and tasks with tiers; which gate checks fail against the
    application today and which task fixes each; which finished tasks the
    next build reopens; and every place where the facts argue against D56 to
    D59.

## Out of bounds

- The application: `mlprofessor/` at the repo root, `requirements.txt`,
  `README.md`, `examples/`.
- The `tests/` folder, in every form: no edits, no new files, no cache files.
- Kit files: `verify/gatekit.py`, `.claude/hooks/`, `.claude/settings.json`,
  `loop/`.
- `mlprof_env`, `BOOTSTRAP.md`, `docs/DECISIONS.md`, `docs/BACKLOG.md`,
  `docs/reviews/`, `docs/measurements/real-data-2026-11-21/`.
- Git, in any form.
- Any band, seed or existing dataset.
- The methodology: flag what the facts argue against, and stop.

## Stop condition

Give the final report and stop. Do not write tests. Do not build.
