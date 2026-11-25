# CP10 — phase 1, pass 9: the headline says what it rests on; the seventh review

**Session:** started from `E:\CODE\ml-professor` by
`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop\run_loop.py harness --open`,
on `fable`. You have no memory of any earlier session. Everything you need is
in the repo and in this file.

**If anything here is wrong** (a file is not as described, a change cannot be
made as written, a decided behaviour contradicts a locked test in a way D59
and D65 do not cover, a measurement argues against a decision), say so
plainly and stop. Do not work round it.

**Two things the last session ran into.** (1) If the guard refuses a write to
a judge file, the session has lost its harness-stage label: stop at once and
tell me; do not build a staged copy. (2) Helpers that edit at the same time
start overlapping gate runs that overwrite each other under
`verify/artifacts/runs/` and report false failures: every proof below is run
alone, one after another, with nothing else on the machine.

## Context

- Phase 1 of ML Professor is built and has been through eight passes. The
  application is the package `mlprofessor` at the repo root, with
  `requirements.txt`, `README.md` and `examples/customers.csv`. As built by
  tasks T1 to T68 it passes the full gate (`verify/smoke_test.py`, 53
  checks, 37 failure modes) and the full suite (2,140 cases under `tests/`,
  four of which run the judge's other tools). `verify/mutation_check.py`
  passes with 88 variants. `verify/battery.py` (1,229 pairs: 1,223 hold, 6
  are stated limits, none is open) and `verify/generated_tables.py` (1,000
  tables) are clean on the build.
- **The application and the existing tests are locked and out of bounds for
  this session.** You change the judge only. New tests are written afterwards
  by the tests stage; the application is changed afterwards by the build
  stage.
- The seventh review returned `VERDICT: FIX FIRST`
  (`docs/reviews/phase1-review-7.md`), and the technical lead's run of the
  eighth pass's build on real public tables found two more defects of the
  contract (`docs/measurements/CP9/build-on-real-tables/README.md`); the
  lead's check of the review is `docs/measurements/review-7/README.md`. Read
  all three. In none of them does the build depart from the spec.
- **D77 is the decision this pass builds, and it changes the approach.** Four
  passes each taught the leak screen one more disguise of a column that
  carries the target, and each review found the next. Two columns that give
  the target only together cannot be seen by any reading of single columns.
  So the tool now also states, on every result, which columns the headline
  rests on, and a near-perfect headline always carries a warning that names
  them. Read D77 before anything else, and D56 as it amends it.
- This is the ninth pass. CP9 (`docs/change-prompts/CP9-open-list.md`) was
  the eighth; the "Eighth pass (CP9)" section of `plan/test-plan.md` and
  tasks T61 to T68 show the shape to follow.
- **Not in this pass:** the broader screen for repeated rows (D60), the
  package (D68), `max_iter` (D75), the column-role override (D76), and the
  backlog items not named below.
- Read first, in this order: `docs/DECISIONS.md` (D77, then D71 to D76, D56,
  D11, D24, D30, D33, then the rest), the three write-ups above, `spec.md`
  (step 8 and section 11 above all), `plan/test-plan.md`, `plan/tasks.json`,
  `verify/smoke_test.py`, `verify/battery.py`, `verify/generated_tables.py`,
  `verify/mutation_check.py`, `verify/reference/mlprofessor/`, `CLAUDE.md`,
  `docs/BACKLOG.md`, `loop/run_loop.py` (`reverify_done`, the tests stage).
  Read the application and the existing tests as far as you need to; change
  neither.

## Before you write anything

Every section of A to E adds or changes a pinned rule, so each needs a
measured proposal first. Report, in a short list, and then wait for my go:

1. The files and sections you will change, and the new test files and tasks.
2. **For each of A to E: the rule you propose, its measurements over 30
   seeds, what it does to every existing gate run, and the smallest change
   that defeats it.** List every existing gate value that would move, old
   against new, and every existing gate run that gains a warning.
3. **Both tools on a prototype of the changed reference,** with the battery
   extended as section G says. List everything that does not hold, in the
   three classes. Do not fix, explain away or leave out any of it: what is
   not decided below needs a ruling.
4. Every existing test, helper and gate check that a decided behaviour
   contradicts, by name, and how D59 and D65 apply. Section A adds keys and
   a warning to finished runs, so search by behaviour: tests that pin the
   exact set of keys, the exact list of warnings, or a console summary.
5. Which finished tasks the next build would reopen, and why. The aim is
   none.
6. Time: what this pass adds to a run of the full profile at 100,000 rows by
   100 columns (the 10-minute budget of D16), to the per-edit gate (12.8 of
   20 seconds), the self-test, the suite and the two tools.
7. Each open point you had to settle to write the spec, with what you
   propose. Do not settle methodology yourself.

## The changes, as a user would observe them

### A. What the headline rests on (D77 points 1 and 2; review finding 2)

1. **On every finished run, for each feature column:** the skill of the
   chosen configuration on the holdout with that column's holdout values
   shuffled among the holdout rows, and the fall from the headline skill.
   Nothing is refitted, nothing is chosen by it, and no other number of the
   output changes. Propose and pin: how many shuffles and how they are
   seeded; where in the output it is held; what a column that reached no
   model shows; what is shown when the headline skill has no value; that the
   result is the same at any `n_jobs`. Measure its spread over shuffles, and
   its cost at 100,000 rows by 100 columns under the full profile; if that
   cost breaks the budget, say so and propose what to do. Do not drop it.
2. **A warning with its own code whenever the headline skill is at or above
   0.98.** It names the columns the headline rests on most, in order of
   their fall, and asks whether each is known before the outcome. Propose
   how many it names and by what rule. It is raised for both tasks and under
   every scheme.
3. **State what the statement cannot show,** with a measurement: two columns
   that carry the same information each show little fall, because the other
   is still there. Say what the output then shows for a table with two exact
   copies of a strong legitimate column.
4. Must hold, over 30 seeds: with `a = y + z` and `b = z` added to a gate
   table, the headline is at or above 0.98, the warning is raised, and both
   columns are named; on every noise table of the gate no such warning is
   raised. Report which existing gate runs reach 0.98 and gain the warning.

### B. The leak screen reads a column by its levels (D77 point 3; review finding 1)

Today a regression's categorical column is read by a tree on codes of its
sorted level names, so a column of 24 or 30 bands of the target with labels
in random order is kept, often in silence, with the headline at 0.95 to 1.0;
a numeric code of the target likewise, in both tasks. Add a reading by
levels: the mean target of each level (for a classification, the share of
the positive class), fitted on a fold's training rows and scored on its
validation rows, with a level not seen in training given the baseline's
prediction. It applies to text columns and to numeric columns of few
distinct values; propose "few" from measurement. Measure, over 30 seeds and
under each scheme:

- must be excluded: 8, 16, 24, 30, 40 and 60 bands of a regression target as
  text with labels in random order, as text with labels in the target's
  order, and as a numeric code in random order; for a classification, a text
  or numeric code of which every level holds one class (10, 40, 100 levels);
- must not be: an honest column of 10, 24 and 60 levels that explains 50%,
  70% and 90% of the target; a column of noise with 10, 60 and 300 levels; a
  column with two to four rows a level; every column of every clean gate
  dataset and of `examples/customers.csv`;
- report where the two meet (an honest column that explains 98% is excluded
  by the rule as written, like any column), and the smallest change that
  defeats the reading.

### C. The order reading without the tie fault (D77 point 4; the lead's finding 1)

Step 8 sorts rows of equal column value by the target, so across two values
the highest target of one group is compared with the lowest of the next and
the target nearly always falls: the warning names columns of pure noise that
hold repeated values (`age` and `s6` of scikit-learn's diabetes table; noise
rounded to one decimal place; a whole-number score from 1 to 40). Repair the
reading so that rows of equal column value are never put in the target's
order: propose the form (one point for each distinct value, holding the mean
target of its rows, is the starting point). Measure, over 30 seeds:

- must stay quiet: noise columns of 5, 10, 40, 100 and 400 distinct values,
  noise rounded to one and to two decimal places, and each of them beside a
  target with real signal in other columns; every clean gate column; the
  nine small-table cells of the CP9 measurement;
- must still warn: the altered copies and the curves of the CP9 measurement,
  and the same with the copy rounded to 40 distinct values.

### D. A copy with a handful of cells changed is excluded, whatever the size (D77 point 5; the lead's finding 2)

Under time order a copy of the target with five training cells changed by
+7 (several of the target's daily steps), by a factor of 1.5, or to another
row's value was warned about and kept on 22 of 24 real tables, with headline
skill 0.89 to 1.00; changed by +0.5 it was excluded on 8 of 8. Find out why
the size matters, repair the screen, and measure, under each scheme: 1, 5
and 15 cells of 1,500 changed by 0.1, 1, 5, 20 and 1,000 times the target's
typical step or spread, and to another row's value. Each must be excluded
up to the 1% of rows the rule already allows. The controls of CP9's A2 must
not move.

### E. A time-ordered holdout far above a quarter is named (D77 point 6; review finding 3)

When whole timestamps make the holdout far more than a quarter of the rows
(60% of the rows on the last timestamp, in the review's case), a warning
with its own code says how large the holdout is and why. Propose the
cut-off; the grouped scheme's `large_group_is_holdout` is the model.

### F. Small things

- `tests/test_judge_tools.py` is strict again (review finding 4): the test
  plan gives the tests stage its strict form, and says what becomes of
  `tests/test_judge_tools_pass8.py`.
- The README says what the new statement and the new warning mean, and what
  a near-perfect headline on an easy table should make a caller check.
- Spec section 11 states D77 point 9 as the rule for what is a limit of the
  screen.
- `CLAUDE.md`, the skill and the spec's index come up to D77.

### G. The battery

The judge missed all of this pass's defects for three reasons: every numeric
column of the gate and the battery is continuous; every altered-cell pair
uses one size of change; and no pair adds two columns at once. Add, to every
run where it makes sense:

- a noise column with repeated values (rounded to one decimal place; a score
  from 1 to 40): no new warning, the same conclusion;
- a coded copy of the target (text bands with labels in random order; a
  numeric code), in both tasks: excluded;
- two columns that give the target only together: the near-perfect warning
  names both;
- altered-cell copies at several sizes of change: excluded;
- an existing numeric feature rounded to 40 and to 10 distinct values: the
  conclusion holds.

For a pair that plants a carrier of the target, "holds" means one of three
outcomes, and the tool records which: the column is excluded; it is warned
about by name; or the headline is at or above 0.98 and the warning of A
names it. Print the count of each. A pair that ends in none of the three
fails. The stated-limit list keeps its four kinds; add a kind only with a
ruling.

## The judge, updated together

- **Reference implementation:** implements A to F. Every existing variant
  stays broken in exactly one way.
- **Gate, new failure modes, each with a dataset or run, a check that names
  it and a broken copy; each new dataset added last so no seed moves:** the
  statement of A is missing, wrong, or uses rows other than the holdout; a
  near-perfect headline carries no warning; a coded copy of the target is
  kept (B); the order reading names a noise column with repeated values (C);
  a copy with cells changed by a large amount is kept (D); a holdout far
  above a quarter is not named (E). Prefer structural assertions, and for A
  have the gate recompute one column's shuffled skill for itself where the
  seed makes that possible; where you assert a score, measure the band over
  30 seeds for the good reference and each broken copy and report the gap.
  Watch each new check fail against the reference as it is now.
- **`verify/mutation_check.py`:** the new variants, each failing a named
  test. The self-test's proof table stays complete.
- **`verify/generated_tables.py`:** properties for the new output; the
  generator gains columns with repeated values and coded columns.
- **Spec:** every behaviour of A to F, pinned so that the test author can
  write against it without seeing code. D11, D30, D33, D74 and D76 are
  amended by D77; the spec must not keep the old behaviour anywhere. A new
  section 10.11 holds this pass's measurements.
- **Test plan:** a "Ninth pass (CP10)" section with the ground rules of the
  eighth, the lists of what is red and what is green today, by name, and
  every existing test or helper to be changed under D59 and D65 with its
  exact form. Tell the test author to run the mutation check on the new
  variants only (`--only`, if the script has it, or the names): the full
  check is run by the owner after the stage.
- **Task plan:** new tasks appended after T68. A task that decides which
  rows a number is computed on, which columns reach a model, or what the
  output says a headline rests on, is `complex`. The last task makes both
  tools clean on the build.
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
7. The measurements of A to E, each with its ranges.
8. `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py doctor`
   prints `SELFTEST: PASS`, a per-edit gate time within the 20 second budget,
   and a valid plan. A warning about uncommitted changes is expected.
9. The output of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/smoke_test.py --matrix`
   and of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py build --dry-run`,
   saved whole under `docs/measurements/CP10/output/` and the new lines
   pasted.
10. Every measurement script saved under `docs/measurements/CP10/`, with a
    `README.md` that gives the command for each with the interpreter path in
    full, the seeds, and a per-seed table (D35). Do this before the final
    report.
11. A final report: what changed in each file; the helpers and tests to be
    changed; the new test files and tasks with tiers; which gate checks fail
    against the application today and which task fixes each; which finished
    tasks the next build reopens; and every place where the facts argue
    against D77.

## Out of bounds

- The application: `mlprofessor/` at the repo root, `requirements.txt`,
  `README.md`, `examples/`.
- The `tests/` folder, in every form: no edits, no new files, no cache files.
- Kit files: `verify/gatekit.py`, `.claude/hooks/`, `.claude/settings.json`,
  `loop/`.
- `mlprof_env`, `BOOTSTRAP.md`, `docs/DECISIONS.md`, `docs/BACKLOG.md`,
  `docs/reviews/`, and everything already under `docs/measurements/`.
- Any folder outside the repo other than your own scratch folder.
- Git, in any form.
- Any existing seed or dataset; any existing band unless an approved rule
  requires it, with its new measurement.
- The repeat screen, the package, `max_iter` and the role override.
- The methodology: flag what the facts argue against, and stop.

## Stop condition

Give the plan of "Before you write anything" and wait. After my go, write the
judge, run the proofs, give the final report and stop. Do not write tests. Do
not build.
