# CP8 — phase 1, pass 7: the sixth review, and a judge that finds the next form of a fault itself

**Session:** started from `E:\CODE\ml-professor` by
`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop\run_loop.py harness --open`,
on `fable`. You have no memory of any earlier session. Everything you need is
in the repo and in this file.

**If anything here is wrong** (a file is not as described, a change cannot be
made as written, a decided behaviour contradicts a locked test in a way D59
and D65 do not cover, a measurement argues against a decision), say so
plainly and stop. Do not work round it.

## Context

- Phase 1 of ML Professor is built and has been through six passes. The
  application is the package `mlprofessor` at the repo root, with
  `requirements.txt`, `README.md` and `examples/customers.csv`. At commit
  `e917ba8` it passes the full gate (`verify/smoke_test.py`, 43 checks, 21
  failure modes) and the full suite (1,841 cases under `tests/`).
  `verify/mutation_check.py` passes: the good reference fails nothing and
  each of 51 variants fails at least one test.
- **The application and the existing tests are locked and out of bounds for
  this session.** You change the judge only. New tests are written afterwards
  by the tests stage; the application is changed afterwards by the build
  stage.
- The sixth review returned `VERDICT: FIX FIRST`
  (`docs/reviews/phase1-review-6.md`). Again it found no place where the
  build departs from the spec. Its two High findings are other forms of
  faults the gate already proves in one form: F14 with one cell changed, F12
  with one column added. The last four reviews each found a fault that way.
  D69, D70 and D71 are the decisions this pass builds; the owner's standard
  is D56.
- The technical lead reproduced findings 1 and 2
  (`docs/measurements/review-6/README.md`) and ran the sixth pass's build on
  real public tables (`docs/measurements/CP7/build-on-real-tables/README.md`),
  which is where D70 comes from. Read both.
- This is the seventh pass. CP7 (`docs/change-prompts/CP7-review-5.md`) was
  the sixth; the "Sixth pass (CP7)" section of `plan/test-plan.md` and tasks
  T48 to T53 show the shape to follow.
- **Not in this pass (D60, D71):** the broader screen for repeated rows and
  the rest of `docs/BACKLOG.md` beyond what is named below. They are pass 8.
  **No new detection rule is added in this pass (D71):** a rule that exists
  is repaired or stated as a limit.
- Read first, in this order: `docs/DECISIONS.md` (D69 to D71, then D56 to
  D68, D49, D20, then the rest), `docs/reviews/phase1-review-6.md`, the two
  write-ups above, `spec.md`, `plan/test-plan.md`, `plan/tasks.json`,
  `verify/smoke_test.py`, `verify/mutation_check.py`,
  `verify/reference/mlprofessor/`, `CLAUDE.md`, `docs/BACKLOG.md`,
  `loop/run_loop.py` (`reverify_done`, the tests stage). Read the application
  and the existing tests as far as you need to; change neither.

## Before you write anything

Report, in a short list, and then wait for my go:

1. The files and sections you will change, and the new test files and tasks.
2. **The date rule you propose (section A1), with its measurements.** For
   approval before anything is built on it.
3. **The perturbation battery (section C1): its design, where it runs, how
   long it takes, and everything it finds today.** Build a prototype outside
   the repo and run it against the reference as it stands and against the
   application as it stands. List every pair of a dataset and a change whose
   conclusion does not hold, whether or not this prompt already names it. Do
   not fix, explain away or leave out any of them: each one that is not
   already decided below needs a ruling, and I will give it.
4. **The generated-table test (section C2): its design, its run time, and
   everything it finds today,** in the same way.
5. The broken copy you propose for each of the seven gate checks never seen
   failing (section C3).
6. Every existing test and gate check that a decided behaviour contradicts,
   by name, and how D59 and D65 apply to each.
7. Which finished tasks the next build would reopen, and why. The aim is
   none.
8. **Time.** The per-edit gate stands at 17.2 of its 20 seconds and the
   self-test at about 19 minutes. Say what this pass adds to each, what you
   would mark slow, and where the battery and the generated-table test run
   so that neither is in the per-edit gate. Do not drop a check.
9. Each open point you had to settle to write the spec, with what you
   propose. Do not settle methodology yourself.

## The changes, as a user would observe them

### A. The sixth review (D69)

1. **Finding 1: a date column with a sentinel is still a date column.** Rule
   2a of step 5 today asks that every value be a `YYYYMMDD` date, so one
   cell of `99991231` or `0` makes the column a numeric feature (measured:
   skill 0.97 to 0.99 where the clean column gives about 0, with no
   warning). Design the tolerant form and bring it to the plan with
   measurements: a share, a small number of distinct values that are not
   dates, or both. A share of 95% alone is not enough (measured: 0.92 to
   0.95 with 5% of cells a sentinel). Measure, over 30 seeds:
   - must be a date column: a clean one; one sentinel cell (`99991231`, `0`,
     `20240230`); 5%, 30% and 60% of rows one sentinel; two distinct
     sentinels; floats with gaps; eight-digit text;
   - must not be: every control of the CP7 measurement
     (`docs/measurements/CP7/candidates.py`), every existing gate dataset,
     `examples/customers.csv`, and code columns of two, three and five
     distinct eight-digit values of which some happen to read as dates;
   - report the smallest change that defeats the rule, and state it in the
     spec's limits.
   As the time column: a value that is not a date stops the run with a named
   error that names the value and how many rows hold it. It is never ordered
   as a number, and never dropped without the caller's say.
2. **Finding 2: `duplicates="group"` and a column the leak screen removes.**
   When the leak screen excludes a column and any combination of the
   remaining feature values is then on both sides of the split, the run
   stops, names the column, and tells the caller to remove it and run again.
   This widens the case the spec already pins (rows that became copies). The
   count the gate makes for itself: combinations of the features that
   reached the models, on both sides, is zero in every finished run under
   `"group"`.
3. **Finding 3:** the `same_features_grouped` text and the README say what
   `"group"` measures: whether a combination of feature values that the
   model has not seen can be predicted, which on a table of few-valued
   columns can give negative skill where `"keep"` finds signal. Measure the
   skill on noise under `"group"` for 8, 16, 24, 48 and 120 combinations
   over 30 seeds, and put the table in the spec's measurements.
4. **Finding 4:** `YYYYMMDDHH` and period labels join the stated limits.
5. **Finding 5:** the text of `date_column_undeclared` is pinned in section
   5.4 and names the scheme correctly under each scheme; a test holds it.
6. **Finding 6:** the branch that would turn a number that is not finite
   into an empty value is replaced by a raise. M18 stays the guard.

### B. The fast profile and an entity larger than its cap (D70)

With an entity column and no time column, when an entity holds more rows
than the fast profile's cap, the fast profile stops with a named error that
sends the caller to the full profile, as D20 does for a wide panel. The same
for a set of rows kept together under `duplicates="group"`. Propose the
code: reuse the family D20 and D53 point 7 use if it fits. Section 7's line
"each kept whole or left out whole" changes to match.

### C. The judge hardens itself (D71, D56)

1. **A perturbation battery.** For every gate dataset whose right conclusion
   is known, apply each of a fixed list of small changes and assert that the
   conclusion still holds: the same stop with the same code, or the same
   columns excluded for the same reasons, or skill inside the same band, or
   the same structural property counted by the judge from the output. The
   list has at least: one cell of a date, time or number column set to a
   sentinel; a column added that copies the target (two-valued, and distinct
   in every row); a constant column added; an identifier column added; the
   rows shuffled (where order is not the point of the dataset); one row
   repeated; a column renamed; a column of noise added; the target's two
   values respelt. Add the changes you judge likely to find a fault, and say
   for each pair of a dataset and a change what must hold and why, or why
   the pair is left out. This is the reviewer's method made permanent: its
   purpose is to find the next form of a fault before a review does.
2. **The generated-table test (D56).** A generator makes tables from a seed:
   the kinds of column the spec knows, missing values, sizes, both tasks,
   every combination of the two statements, `duplicates` in each value.
   Each table is run through the application and through the reference, and
   the two outputs must agree: the same named error, or the same
   `profile.json` and `predictions.csv` apart from the fields the spec says
   may differ. Agreement does not prove either right; a disagreement is a
   finding. Propose how many tables, with what seeds, and how it is run.
3. **A broken copy for each of the seven checks never seen failing:**
   `model.signal_clf`, `model.signal_reg`, `time.undeclared_warns`,
   `result.matches_json`, `output.complete`, `output.deterministic` and
   `output.same_at_any_n_jobs`. After this pass the self-test's list of such
   checks is empty.

Sections C1 and C2 are tools of the judge. Say where each lives (the gate's
slow profile, the suite, or a script under `verify/` with its own command),
what the loop runs and when, and what the lead runs by hand before a review.

### D. Small things

- `CLAUDE.md` and the spec's index come up to D71.
- The backlog's thin spots (broken copies held by one test: M1, M5, M6, M8,
  M10 to M13, M17, M21, M23, M28) each get a second test where the battery
  does not already give one; say which.

## The judge, updated together

- **Reference implementation:** implements A and B. Every existing variant
  stays broken in exactly one way.
- **Gate, new failure modes, each with a dataset or run, a check that names
  it and a broken copy; each new dataset added last so no seed moves:** a
  date column with a sentinel is a feature (finding 1); under `"group"` a
  removed leak leaves feature combinations on both sides (finding 2); the
  fast profile leaves out an entity too large for its cap (D70). Prefer
  structural assertions. Watch each new check fail against the reference as
  it is now.
- **`verify/mutation_check.py`:** the new variants. The self-test's proof
  table stays complete, and its list of checks never seen failing is empty.
- **Spec:** every behaviour of A and B, pinned so that the test author can
  write against it without seeing code; C described as the judge's own
  tools; a new section 10.9 for this pass's measurements; the limits of
  section 11 brought up to date.
- **Test plan:** a "Seventh pass (CP8)" section with the ground rules of the
  sixth, the lists of what is red and what is green today, by name, and
  every existing test or helper to be changed under D59 and D65 with its
  exact form.
- **Task plan:** new tasks appended after T53. A task that decides which
  rows a number is computed on, which columns reach a model, or whether a
  run may go ahead, is `complex`. Order them so that each can be judged when
  it is built.
- **`README.md` is application code.** A task brings it up to date; say how
  the README tests stay green until then.
- **`CLAUDE.md` and the skill:** D1 to D71, the battery and the
  generated-table test and how a builder runs them.

## Proof before you stop

1. The gate's self-test passes, with the proof table complete and no check
   listed as never seen failing.
2. Each new gate check failed against the unfixed reference and passes
   against the fixed one.
3. No other gate value moved at the gate seed apart from what these changes
   require; say how you compared and what moved.
4. `verify/mutation_check.py` against the existing suite, pasted. The good
   reference may fail only the tests the plan names under D59 and D65; name
   them. Anything else: stop.
5. Prototypes of every new and every changed test, outside the repo: each
   changed test passes against the application as it stands and against the
   fixed reference; each new test is red or green today as listed; each
   green guard fails on its variant. With the prototypes laid over the suite
   (`--overlay`), the mutation check passes. For each broken copy the suite
   caught before, the number of failing cases before and after, in one
   table.
6. The battery and the generated-table test, run on the fixed reference:
   both clean. Run on the application as it stands: the failures are exactly
   those the plan lists, each owned by a task.
7. The measurements of A1 and A3, each with its ranges.
8. `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py doctor`
   prints `SELFTEST: PASS`, a per-edit gate time within the 20 second budget,
   and a valid plan. A warning about uncommitted changes is expected.
9. The output of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/smoke_test.py --matrix`
   and of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py build --dry-run`,
   saved whole under `docs/measurements/CP8/output/` and the new lines
   pasted.
10. Every measurement script saved under `docs/measurements/CP8/`, with a
    `README.md` that gives the command for each with the interpreter path in
    full, the seeds, and a per-seed table (D35). Do this before the final
    report.
11. A final report: what changed in each file; the helpers and tests to be
    changed; the new test files and tasks with tiers; which gate checks fail
    against the application today and which task fixes each; which finished
    tasks the next build reopens; and every place where the facts argue
    against D69 to D71.

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
- The repeat screen and any new detection rule: pass 8, or never.
- The methodology: flag what the facts argue against, and stop.

## Stop condition

Give the plan of "Before you write anything" and wait. After my go, write the
judge, run the proofs, give the final report and stop. Do not write tests. Do
not build.
