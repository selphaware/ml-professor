# CP7 — phase 1, pass 6: the caller always states time order and entities; `duplicates="group"` on the features alone; the fifth review

**Session:** started from `E:\CODE\ml-professor` by
`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop\run_loop.py harness --open`,
on `fable`. You have no memory of any earlier session. Everything you need is
in the repo and in this file.

**If anything here is wrong** (a file is not as described, a change cannot be
made as written, a decided behaviour contradicts a locked test in a way D59
does not cover, a measurement argues against a decision), say so plainly and
stop. Do not work round it.

## Context

- Phase 1 of ML Professor is built and has been through five passes. The
  application is the package `mlprofessor` at the repo root, with
  `requirements.txt`, `README.md` and `examples/customers.csv`. At commit
  `f3115f0` it passes the full gate (`verify/smoke_test.py`, 37 checks, 17
  failure modes) and the full suite (1,757 cases under `tests/`).
  `verify/mutation_check.py` passes: the good reference fails nothing and
  each of 39 variants fails at least one test.
- **The application and the existing tests are locked and out of bounds for
  this session.** You change the judge only. New tests are written afterwards
  by the tests stage; the application is changed afterwards by the build
  stage.
- The fifth review returned `VERDICT: FIX FIRST`
  (`docs/reviews/phase1-review-5.md`). It found no place where the build
  departs from the spec: every finding is a gap in the contract. D62, D63 and
  D64 are the decisions this pass builds. The owner's standard is D56:
  nothing known to be wrong is left for later.
- The technical lead reproduced finding 1 and tried the decided remedy
  (`docs/measurements/review-5/README.md`), and ran the fifth pass's build on
  real public tables (`docs/measurements/CP6/build-on-real-tables/README.md`).
  Read both.
- This is the sixth pass. CP6 (`docs/change-prompts/CP6-entities-and-repeats.md`)
  was the fifth; the "Fifth pass (CP6)" section of `plan/test-plan.md` and
  tasks T39 to T47 show the shape to follow.
- **Not in this pass (D60, D64):** the broader screen for repeated rows, a
  broken copy for each of the seven gate checks never seen failing, and the
  rest of `docs/BACKLOG.md`. They are pass 7. Do not design, measure or build
  any of them here.
- Read first, in this order: `docs/DECISIONS.md` (D62 to D64, then D56 to
  D61, D49, D53, D6, D7, D10, then the rest),
  `docs/reviews/phase1-review-5.md`, the two write-ups above, `spec.md`,
  `plan/test-plan.md`, `plan/tasks.json`, `verify/smoke_test.py`,
  `verify/mutation_check.py`, `verify/reference/mlprofessor/`, `CLAUDE.md`,
  `docs/BACKLOG.md`, `loop/run_loop.py` (`reverify_done`, the tests stage).
  Read the application and the existing tests as far as you need to; change
  neither.

## Before you write anything

Section A changes what nearly every existing test and every gate run does on
its first line, so the plan matters more than in any pass before. Report, in
a short list, and then wait for my go:

1. The files and sections you will change, and the new test files and tasks.
2. **How the existing suite follows D63 under D59.** Count, by file, the
   existing tests that call `profile()` or the command line with nothing said
   about time order or about entities. Then propose the smallest set of edits
   that lets every one of them pass both against the application as it
   stands and after the change: the loosened form takes its fall-back only
   when the run stops with the new named error, and changes nothing else. If
   the edits can be made once in a file's own helper and not in each test,
   propose that, and list every helper and every test touched, by name. Say
   which existing tests pin a behaviour that D63 removes outright (a plain
   call that runs; the conditional stops of D58) and how each is loosened,
   and which new test pins the new behaviour exactly.
3. Which finished tasks the next build would reopen, and why. The aim is
   none.
4. **The rule you propose for the grouped holdout (section C), with its
   measurements.** This is for approval before anything is built on it.
5. **Two candidates you are asked to measure and not to build (section D).**
6. Each open point of sections A to C that you had to settle to write the
   spec, with what you propose. Do not settle methodology yourself.

## The changes, as a user would observe them

### A. Both statements, on every call (D63; fifth review findings 2, 3, 4 and 5)

- `time_col` is a column's name or `False`. `group_col` is a column's name or
  `False`. `None`, the default of each, now means "not said", and a call that
  leaves either unsaid stops with a named error and writes nothing, whatever
  the table holds: with or without a date column, with unique or repeated
  timestamps, for classification and regression alike.
- The signature keeps its eleven parameters, their order and their defaults.
- One message tells the caller everything that is missing, so that one more
  call is enough. Propose the codes: the existing `time_order_undeclared` and
  `entities_undeclared` are the starting point. Pin which is raised when both
  are unsaid, where in the order of the steps the error is raised, and how it
  ranks against the other named errors and against `InputError`. The message
  should still name what the tool can see that bears on the answer (the
  columns that hold dates; that timestamps repeat), since that helps the
  caller choose.
- On the command line: one of `--time-col COLUMN` and `--no-time-order` is
  required, and one of `--group-col COLUMN` and `--no-entities`. Pin the exit
  code and the text for each missing pair.
- The four combinations then mean: no time order and no entities, the random
  scheme; no time order and an entity column, the grouped scheme; a time
  column and no entities, time order with one baseline for the table; a time
  column and an entity column, time order with each entity's own previous
  value.
- **The conditional stops of D58 are gone.** A run never goes ahead or stops
  because of a date form or an autocorrelation. The same conditions stay as
  warnings that question a statement, exactly as they are today:
  `date_column_undeclared` under `time_col=False`, and
  `autocorrelated_panel_target` under `group_col=False`. Finding 5: a test
  pins the 0.5 cut-off of that warning from both sides. Finding 2: measure
  how often the warning fires on short panels (8, 12, 16, 24, 36 timestamps)
  and state the result in the spec's limits; do not change the cut-off.
- The step functions of `mlprofessor.stages` take what they are given: the
  rule is for `profile()` and the command line.
- Both statements are recorded in the output (`input.time_col`,
  `input.group_col`), as `false` or the column's name. Neither is ever null
  in a finished run; say so in section 5.2.
- The fixed texts of section 5.4 stay as they are unless one of them is now
  false; if one is, report it in the plan and propose the wording.
- `README.md`: every command and every Python example states both.

### B. `duplicates="group"` keeps rows that agree in every feature together (D62; finding 1)

- The sets of step 7 are formed on the feature columns of step 5 alone. The
  target plays no part. A set that spans two entities still joins them.
- `validation.n_rows_with_copies` counts the rows that agree with another row
  in every feature column. Say in the spec and the README what `"group"` now
  means: no combination of feature values is on both sides of the split or
  in two folds.
- D49's count at step 8a is unchanged: a copy is a row equal in every feature
  and the target. Say what that count can be after step 7 has grouped on the
  features, including the case the spec already pins, where the leak screen
  excludes the one column in which two rows differed.
- The message of the copies stop says when each answer fits, `"keep"` first:
  `"keep"` when the repeated rows are different cases that look alike, as in
  a table of few-valued columns; `"group"` when they are one case recorded
  more than once; or remove the repeated rows. Pin the text.
- Everything else about the grouped scheme holds: fewer than six units is
  `too_few_groups`, fewer than 30 units in the holdout gives no interval,
  and there is no p-value (D61).

### C. The grouped holdout does not overshoot (D64; finding 6)

- Today the units are taken in an order drawn from the seed until they hold
  a quarter of the rows, so one entity of 60% of the rows, drawn early, makes
  the holdout 70% to 80% (the review's measurement).
- Design the rule and bring it to the plan with measurements: the draw leaves
  out a unit that would overshoot when another can complete the holdout, and
  a warning with its own code says so when one unit alone is too large for a
  sound split. State the bound you can guarantee on the holdout's share, for
  regression and for each class. Measure, over 30 seeds: entities of equal
  size; one entity of 30%, 60% and 90% of the rows; a handful of large
  entities; and the gate's `entities` dataset, whose band and whose split
  must not move unless the rule requires it, in which case say so and
  re-measure the band.
- The pinned properties 1, 2 and 4 of step 7 hold as they are.

### D. Two candidates: measure and report, do not build

Under D63 a caller can still say `False` where the table argues otherwise.
For each of the two forms the review found, measure a narrow warning and
report it with false-positive counts on every existing gate dataset,
`examples/customers.csv`, and tables you make for the purpose. The owner
decides after reading; nothing of section D enters the spec, the reference or
the plan in this session.

1. Finding 4: under `time_col=False`, a whole-number column in which every
   value reads as a calendar date written `YYYYMMDD` (state the range of
   years you would accept). Today such a column is a numeric feature.
2. Finding 3: under a time column and `group_col=False`, a text or
   few-valued column whose levels each form a series of their own: the
   target's previous value within a level predicts far better than the
   table's own baseline.

### E. Small things

- `CLAUDE.md` says "D1 to D60". Bring it and the spec's index up to D64.
- The backlog's thin spots of the fifth pass (M17 and M21 are each held by
  one test) are not in this pass.

## The judge, updated together

- **Reference implementation:** implements A to C. Every existing variant
  stays broken in exactly one way.
- **Gate, new failure modes, each with a dataset or run, a check that names
  it and a broken copy; each new dataset added last so no seed moves:**
  - a statement the caller did not make is assumed (a run with nothing said
    about time order, and one with nothing said about entities, each goes
    ahead): the right outcome is the stop, for both;
  - `duplicates="group"` forms its sets with the target (noise in a table of
    few-valued columns, run with `--duplicates group`: the right conclusion
    is no skill, inside a band you measure over 30 seeds for the good
    reference and for the broken copy, with a clear gap; choose the table so
    that the gap is clear, and report both ranges);
  - the grouped holdout overshoots (one dominant entity: the holdout's share
    stays within the bound of section C, counted by the gate itself from
    `predictions.csv`).
  Prefer structural assertions. Watch each new check fail against the
  reference as it is now.
- **Every existing gate run** gains the statements it needs, and nothing else
  about it changes. List them. A gate check moves to a new task only where
  D54's reasoning applies; say which and why.
- **`verify/mutation_check.py`:** the new variants. The self-test's proof
  table stays complete.
- **Spec:** every behaviour of A to C, pinned so that the test author can
  write against it without seeing code. D6, D7 and D58 are replaced in part
  by D63; the spec must not keep the old behaviour anywhere. A new section
  10.8 holds this pass's measurements.
- **Test plan:** a "Sixth pass (CP7)" section with the ground rules of the
  fifth. For D59: every helper and every existing test to be changed, by
  name, with its exact loosened form, and the new test that pins the new
  behaviour. Every other line of every existing file stays as it is. The
  lists of what is red and what is green today, by name.
- **Task plan:** new tasks appended after T47. A task that decides which
  rows a number is computed on, or whether a run may go ahead, is `complex`.
  Order them so that each can be judged when it is built.
- **`README.md` is application code.** Its commands must still run after
  this pass, so a task rewrites them; say how the README tests stay green
  from the tests stage to that task.
- **`CLAUDE.md` and the skill:** the two statements, the new meaning of
  `"group"`, D1 to D64.

## Proof before you stop

1. The gate's self-test passes, with the proof table complete.
2. Each new gate check failed against the unfixed reference and passes
   against the fixed one.
3. No other gate value moved at the gate seed apart from what these changes
   require; say how you compared and what moved. The command lines and the
   recorded statements of every run will move; no score, count or band may,
   unless section C requires it and you have said so.
4. `verify/mutation_check.py` against the existing suite, pasted. The good
   reference may fail only the tests the plan names under D59; name them.
   Anything else: stop.
5. Prototypes of every new and every loosened test, outside the repo: each
   loosened test passes against the application as it stands and against the
   fixed reference; each new test is red or green today as listed; each
   green guard fails on its variant. With the prototypes laid over the suite
   (`--overlay`), the mutation check passes.
6. The measurements of sections A (short panels), C and D, and the band of
   the new score check, each with the ranges for the good reference and the
   broken copy.
7. `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py doctor`
   prints `SELFTEST: PASS`, a per-edit gate time within the 20 second budget,
   and a valid plan. A warning about uncommitted changes is expected. The
   per-edit gate stood at 15.6 seconds before this pass: if it would go over
   budget, say so and propose what to mark slow; do not drop a check.
8. The output of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/smoke_test.py --matrix`
   and of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py build --dry-run`,
   pasted.
9. Every measurement script saved under `docs/measurements/CP7/`, with a
   `README.md` that gives the command for each with the interpreter path in
   full, the seeds, and a per-seed table (D35). Do this before the final
   report, not after it.
10. A final report: what changed in each file; the helpers and tests to be
    loosened; the new test files and tasks with tiers; which gate checks
    fail against the application today and which task fixes each; which
    finished tasks the next build reopens; and every place where the facts
    argue against D62 to D64.

## Out of bounds

- The application: `mlprofessor/` at the repo root, `requirements.txt`,
  `README.md`, `examples/`.
- The `tests/` folder, in every form: no edits, no new files, no cache files.
- Kit files: `verify/gatekit.py`, `.claude/hooks/`, `.claude/settings.json`,
  `loop/`.
- `mlprof_env`, `BOOTSTRAP.md`, `docs/DECISIONS.md`, `docs/BACKLOG.md`,
  `docs/reviews/`, and everything already under `docs/measurements/`.
- Git, in any form.
- Any existing band, seed or dataset.
- The repeat screen, the seven checks never seen failing, and the rest of
  the backlog: pass 7.
- The methodology: flag what the facts argue against, and stop.

## Stop condition

Give the plan of "Before you write anything" and wait. After my go, write the
judge, run the proofs, give the final report and stop. Do not write tests. Do
not build.
