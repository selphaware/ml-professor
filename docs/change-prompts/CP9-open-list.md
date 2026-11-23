# CP9 — phase 1, pass 8: everything the judge's two tools still list as open

**Session:** started from `E:\CODE\ml-professor` by
`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop\run_loop.py harness --open`,
on `fable`. You have no memory of any earlier session. Everything you need is
in the repo and in this file.

**If anything here is wrong** (a file is not as described, a change cannot be
made as written, a decided behaviour contradicts a locked test in a way D59
and D65 do not cover, a measurement argues against a decision), say so
plainly and stop. Do not work round it.

## Context

- Phase 1 of ML Professor is built and has been through seven passes. The
  application is the package `mlprofessor` at the repo root, with
  `requirements.txt`, `README.md` and `examples/customers.csv`. As built by
  tasks T1 to T60 it passes the full gate (`verify/smoke_test.py`, 47
  checks, 31 failure modes) and the full suite (1,937 cases under `tests/`,
  two of which run the judge's other tools). `verify/mutation_check.py`
  passes: the good reference fails nothing and each of 70 variants fails at
  least one test.
- **The application and the existing tests are locked and out of bounds for
  this session.** You change the judge only. New tests are written afterwards
  by the tests stage; the application is changed afterwards by the build
  stage.
- Pass 7 gave the judge two tools of its own: `verify/battery.py` (small
  changes to every gate dataset, the conclusion must hold) and
  `verify/generated_tables.py` (the library against the reference on tables
  made from a seed). Both are clean on the build apart from what each lists
  as **open**, with D73 and "pass 8" as owner. This pass empties both lists.
  There was no review after pass 7: a review is run when both tools list
  nothing as open (D73).
- The technical lead ran pass 7's build on real public tables
  (`docs/measurements/CP8/build-on-real-tables/README.md`). Its last table
  shows three of the open faults on real data: the VIX level under time
  order scores 0.023, 0.843 with one row repeated, and 1.000 with an exact
  rescaled copy of the target added; a public regression table falls from
  0.490 to -0.111 with one cell of 999999 in a feature.
- This is the eighth pass. CP8 (`docs/change-prompts/CP8-review-6.md`) was
  the seventh; the "Seventh pass (CP8)" section of `plan/test-plan.md` and
  tasks T54 to T60 show the shape to follow.
- **Not in this pass:** the broader screen for repeated rows (D60) and the
  backlog items not named below. **No new detection rule is added (D71):** a
  rule that exists is repaired, or the result is described more fully.
- Read first, in this order: `docs/DECISIONS.md` (D73, then D69 to D72, D56,
  D10, D11, D19, D24, D30, D31, D36, then the rest), spec section 11's open
  list, `docs/measurements/CP8/README.md`, the write-up above, `spec.md`,
  `plan/test-plan.md`, `plan/tasks.json`, `verify/smoke_test.py`,
  `verify/battery.py`, `verify/generated_tables.py`,
  `verify/mutation_check.py`, `verify/reference/mlprofessor/`, `CLAUDE.md`,
  `docs/BACKLOG.md`, `loop/run_loop.py` (`reverify_done`, the tests stage).
  Read the application and the existing tests as far as you need to; change
  neither.

## Before you write anything

Every section of A changes a pinned rule, so each needs a measured proposal
before it is built. Report, in a short list, and then wait for my go:

1. The files and sections you will change, and the new test files and tasks.
2. **For each of A1 to A6: the rule you propose, its measurements over 30
   seeds, what it does to every existing gate dataset and band, and the
   smallest change that defeats it.** Any existing gate value that would
   move is listed, with the old and the new range. Nothing is built on a
   rule before I approve it.
3. **Both tools run on a prototype of the changed reference:** every pair
   and every finding now on the open lists holds, and nothing that held
   before fails. Then extend the battery with the changes you judge most
   likely to find the next fault in the rules of A, run it, and list
   everything it finds, in the three classes. Do not fix, explain away or
   leave out any of it.
4. Every existing test, helper and gate check that a decided behaviour
   contradicts, by name, and how D59 and D65 apply to each. The M23 guard in
   `tests/test_guard_review5.py` is one.
5. Which finished tasks the next build would reopen, and why. The aim is
   none.
6. Time: what this pass adds to the per-edit gate (12.6 of 20 seconds
   today), the self-test, the suite and the two tools.
7. Each open point you had to settle to write the spec, with what you
   propose. Do not settle methodology yourself.

## The changes, as a user would observe them

### A. The open list (D73)

1. **The previous value at repeated timestamps.** Under time order with
   `group_col=False`, repeated timestamps today leave the training mean as
   the only baseline, so one repeated row turns a random walk's 0.02 into
   0.84 to 0.999. The previous-value baseline is extended: for a row, the
   mean target of the timestamp `gap + 1` distinct timestamps before. The
   headline baseline is the better of the two on the holdout, both are
   reported, and the leak screen measures a column against the stronger, as
   D10 and D24 already have it for unique timestamps. The panel warning
   stays as it is. Must hold: a random walk with one row repeated, with 1%
   and with 10% of its rows repeated, scores inside the band of the clean
   walk; the gate's panel datasets under `--no-entities` are reported, old
   value against new.
2. **A column that reproduces the target is always excluded.** Today an
   exact rescaled copy of a regression target passes the leak screen under
   time order, where the pinned tree cannot extrapolate, and beside a
   dominant entity, where one fold trains on about ten rows. Repair the
   screen and bring the repair with measurements. Must be excluded, under
   every scheme: an exact copy, a rescaled copy (`a * y + b`), a copy with a
   handful of cells that differ. Must not be: every control the gate already
   has (the strongest legitimate column; a series' own lag under time order;
   a lag beside a declared entity), and the columns of every clean gate
   dataset. The warning tier of D30 keeps its meaning.
3. **One row that carries most of the error.** One cell of -999 or 999999
   in a numeric feature or a regression target, or one empty lag cell on a
   panel, moves a regression skill far out of its band; one such cell in the
   target also switches the leak screen off. Two things: (a) the result says
   so, as a statement about the result and not a rule about the data: the
   share of the chosen model's squared error on the holdout that its single
   largest row carries, for the model and for the baseline, and a warning
   when that share is above a cut-off you measure; (b) the leak screen still
   excludes the leak on `leak_reg` with one extreme cell in the target.
   Propose how, measured against the controls of A2. Say plainly what the
   headline itself still does with such a row: this pass reports it, it does
   not change the metric.
4. **The order of rows within a fit under time order is pinned,** so that
   the library and the reference hand a model the same rows in the same
   order and forests agree. Propose the order; list any gate value that
   moves.
5. **An identifier stays an identifier when the rows are shuffled or one row
   is repeated.** Repair the id-like rule of D19 and measure it against
   D19's own controls: a whole-number measure (a price, a volume) stays a
   feature at any table size, sorted or not.
6. **Selection folds beside a dominant entity.** On the gate's
   `dominant_entity` one fold trains on about ten rows; the application
   reports skill -1.13 there and the reference -0.006. Say why the two
   differ, propose what a fold with too few training rows does (the existing
   `fold_not_scored` record is the starting point), and give the dataset a
   band once the conclusion is stable.

### B. The tools

- After this pass both open lists are empty: every pair and finding on them
  is a must-hold pair, and the code that printed an open class stays only to
  print "open: 0". The stated-limit list keeps the two it has (a copy that
  differs by an added column, D60; a whole-number time column with an empty
  cell, D55), each with its decision.
- The battery gains the changes of item 3 above that are worth keeping.
- A second test for each broken copy held by one: F22, M13, M33, M34, M37,
  M39, where the tools do not already give one; say which.

### C. Small things

- `CLAUDE.md`, the skill and the spec's index come up to the latest
  decision in `docs/DECISIONS.md`.
- The README says what changed for a caller: the baseline at repeated
  timestamps, the new statement about the largest row, and the limits.

## The judge, updated together

- **Reference implementation:** implements A. Every existing variant stays
  broken in exactly one way.
- **Gate, new failure modes, each with a dataset or run, a check that names
  it and a broken copy; each new dataset added last so no seed moves:** one
  repeated row drops the previous-value baseline (A1); a rescaled copy of
  the target is kept (A2); one extreme target cell switches the leak screen
  off, and the largest row's share is missing or wrong (A3); an identifier
  becomes a feature when rows are shuffled (A5); a fold with too few
  training rows decides the choice (A6). Prefer structural assertions; where
  you assert a score, measure the band over 30 seeds for the good reference
  and each broken copy and report the gap. Watch each new check fail against
  the reference as it is now.
- **`verify/mutation_check.py`:** the new variants. The self-test's proof
  table stays complete, with no check never seen failing.
- **Spec:** every behaviour of A, pinned so that the test author can write
  against it without seeing code. D10, D11, D19 and D24 are amended by D73;
  the spec must not keep the old behaviour anywhere. A new section 10.10
  holds this pass's measurements. Section 11's open list is empty.
- **Test plan:** an "Eighth pass (CP9)" section with the ground rules of the
  seventh, the lists of what is red and what is green today, by name, and
  every existing test or helper to be changed under D59 and D65 with its
  exact form.
- **Task plan:** new tasks appended after T60. A task that decides which
  rows a number is computed on, which columns reach a model, which baseline
  a skill is measured against, or whether a run may go ahead, is `complex`.
  The last task makes both tools clean on the build with nothing open.
- **`README.md` is application code.** A task brings it up to date; say how
  the README tests stay green until then.

## Proof before you stop

1. The gate's self-test passes, with the proof table complete and no check
   listed as never seen failing.
2. Each new gate check failed against the unfixed reference and passes
   against the fixed one.
3. No other gate value moved at the gate seed apart from what the approved
   rules require; say how you compared and list what moved, old against new.
4. `verify/mutation_check.py` against the existing suite, pasted. The good
   reference may fail only the tests the plan names under D59 and D65; name
   them. Anything else: stop.
5. Prototypes of every new and every changed test, outside the repo, and
   kept out of it: each changed test passes against the application as it
   stands and against the fixed reference; each new test is red or green
   today as listed; each green guard fails on its variant. With the
   prototypes laid over the suite (`--overlay`), the mutation check passes.
   For each broken copy the suite caught before, the number of failing cases
   before and after, in one table.
6. Both tools on the fixed reference: clean, with nothing open. On the
   application as it stands: the failures are exactly those the plan lists,
   each owned by a task.
7. The measurements of A1 to A6, each with its ranges.
8. `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py doctor`
   prints `SELFTEST: PASS`, a per-edit gate time within the 20 second budget,
   and a valid plan. A warning about uncommitted changes is expected.
9. The output of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/smoke_test.py --matrix`
   and of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py build --dry-run`,
   saved whole under `docs/measurements/CP9/output/` and the new lines
   pasted.
10. Every measurement script saved under `docs/measurements/CP9/`, with a
    `README.md` that gives the command for each with the interpreter path in
    full, the seeds, and a per-seed table (D35). Do this before the final
    report.
11. A final report: what changed in each file; the helpers and tests to be
    changed; the new test files and tasks with tiers; which gate checks fail
    against the application today and which task fixes each; which finished
    tasks the next build reopens; and every place where the facts argue
    against D73.

## Out of bounds

- The application: `mlprofessor/` at the repo root, `requirements.txt`,
  `README.md`, `examples/`.
- The `tests/` folder, in every form: no edits, no new files, no cache files.
- Kit files: `verify/gatekit.py`, `.claude/hooks/`, `.claude/settings.json`,
  `loop/`.
- `mlprof_env`, `BOOTSTRAP.md`, `docs/DECISIONS.md`, `docs/BACKLOG.md`,
  `docs/reviews/`, and everything already under `docs/measurements/`.
- Git, in any form.
- Any existing seed or dataset; any existing band unless an approved rule of
  A requires it, with its new measurement.
- The repeat screen and any new detection rule.
- The methodology: flag what the facts argue against, and stop.

## Stop condition

Give the plan of "Before you write anything" and wait. After my go, write the
judge, run the proofs, give the final report and stop. Do not write tests. Do
not build.
