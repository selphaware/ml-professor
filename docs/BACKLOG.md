# ML Professor — backlog

Things decided against for now, with the reason. Not a to-do list for the
build loop: each item becomes work only through a decision in
`docs/DECISIONS.md` and a change prompt. Phase 2 starts by reading this file
(D47).

**Status under D56, D71, D73, D74 and D76 (2026-11-25).** Nothing known to
be wrong is carried into phase 2. Passes 5, 6 and 7 are built. Pass 8
(CP9): the judge is written; both of its tools list nothing as open and
four kinds of stated limit (D60, D55, D31, D76). After pass 8's build: a
review. Then the broader repeat screen under D60, or the ruling that
replaces it, and every remaining weakness here. No new detection rule
during hardening (D71), with the one exception D74 records.

## After pass 8: the broader repeat screen (D60)

- **Open defect.** Copies disguised by an added column, a batch number or a
  small change pass D49. On real tables with the target shuffled, every row
  twice scored 0.17 to 0.94 where the true skill is 0.
- **What failed, and why.** The calibrated rule expects columns to vary
  independently; on real tables they do not, and it stopped clean diamonds,
  taxis and VIX prices. It also read 21 or more copies of a row as chance,
  so oversampling before the split passed. The first rule stopped clean
  taxis. See `docs/measurements/CP6/real-tables/README.md`.
- **A lead, not a rule.** Counting holdout rows whose target and every
  column but one (the same one table-wide) match a training row, numbers to
  6 significant digits: 0% to 1% on clean tables with several measured
  columns, 73% to 84% with every row twice. It fails on tables of few-valued
  columns (clean Titanic 44%).
- **To settle.** What the tool does on tables of few-valued columns, where
  no count can tell copies from look-alikes; and whether the made-up tables
  in the gate can reproduce the dependence between real columns.

## Roadmap (D67, D68, D75, D76)

1. Library core: closing. Then, as phase 1's last changes, the
   pip-installable package (D68), a small scikit-learn neural network
   family together with the caller's `max_iter` (D75), and a caller's
   override of column roles (D76).
2. Feature engineering on one table.
3. Several datasets and joins, with cross-table aggregate features.
4. Deep learning with Keras or PyTorch, including sequence models.
5. Unsupervised methods.
6. Self-contained HTML report.
7. Dash app.

## Deferred features

- **A `balanced` profile:** the full model families with a capped forest,
  for tables or machines where the full profile is too slow (D16).
- **Search on a subsample, refit on every row:** for tables far beyond
  100,000 rows (D16). Needs its own rule under time order.
- **Skill with and without the warned columns:** so a near-leak that is kept
  (D30, D33) cannot quietly be the whole headline.
- **Walk-forward validation** across the holdout, in place of a single
  cut-off (D7).
- **Keeping copies of a row on one side of the split** (`duplicates="group"`):
  the proper treatment of duplicated rows, with phase 2's grouped
  validation (D49 stops the run in the meantime).
- **Entity-aware baselines and sampling for panels:** phase 2, with grouped
  validation (D10, D20, D31).

## Methodology limits to revisit

- **Block length of the time-ordered bootstrap.** The cube-root rule is
  shorter than strong serial correlation; coverage was measured at 64% to
  85% (D27). A data-driven block length is the candidate.
- **Numbers written in other locale forms** (`1.200,50`, `$1,200`) are
  excluded as free text or id-like, not by their own reason (D33).
- **The p-value's tie in regression** has no pinned tolerance, so its guard
  covers classification only (D43).

## Found in the CP4 session (second review round)

- **Seven gate checks have never been seen to fail on any broken copy.**
  They carry a written reason in place of a failure mode:
  `model.signal_clf`, `model.signal_reg`, `time.undeclared_warns`,
  `result.matches_json`, `output.complete`, `output.deterministic` and
  `output.same_at_any_n_jobs`. Each should get a broken copy that proves it
  fires. The self-test prints this list on every run.
- **One extreme value can dominate a squared-error skill.** A feature cell
  of 1e38 in a holdout row gave a linear model a skill of about minus 7e73:
  finite, and reported as measured. Candidates: a warning for extreme
  outliers in features and target, or a robust second metric.
- **Targets on a scale near 1e80.** Every reported number can be held, but
  an intermediate step overflows. The spec allows the run either to finish
  or to stop with `non_finite_score`; it may never write an empty value.
  Worth pinning one outcome.
- **The fast profile and the gap.** When the capped rows hold six training
  timestamps and the gap leaves fewer, the fast profile raises
  `gap_too_large` whatever the full profile would do (D46 covers only the
  too-few-timestamps case).
- **A white-space-only target cell** is counted in the rows dropped for a
  missing target, but not in the target column's own missing count.
- **The `infinite_values` warning** is raised for features only in the
  library and the spec; the gate's reference raises it for any numeric
  column. No test pins it.

## Found by the third review

- **Thin spots in the suite.** M1, M5 and M6 are each held by one test, as
  are three faults the reviewer planted: a previous value that ignores the
  gap, the panel autocorrelation warning removed, and the leak cut-off moved
  to 0.999. A second test for each would make the suite less brittle.
- **Not verified by any review:** seeds other than the gate's; whether a
  library warning can reach standard error beside a named error.

## Found in the CP5 session (third review round)

- **Large whole-number features and the models.** A feature of whole
  numbers above 2 to the power 53 is counted exactly, but the models receive
  floats, so distinct values can reach them as one. No warning says so
  (D55).
- **The grey zone of the copies rule.** With 1% to 2% of rows repeated, the
  count of 10 decides and the outcome depends on the seed (D55).
- **The type matrix's passing cases have no broken copy.** It is a contract
  test over 805 cases; only its one-value cases are shown to fail, on F6b.

## Found by the fourth review and the real-data run (2026-11-21)

- **Three gate bands have narrow margins at the gate's one seed** (0.07,
  0.09 and 0.12 to the broken copies), as the reviewer reads them. Two of
  the three are bounded by the scale itself: a leak scores about 1.0 and
  skill cannot exceed 1. Pass 7 reports the broken copies' values at the
  gate seed, so the real distance is on record.
- **A three-class target** stops as ambiguous, with a message that suggests
  declaring the task, which would not help. The message should say that
  three or more classes are not supported.
- **Not checked on real data:** the owner's own tables, Python 3.12 on
  Windows, and the full profile at scale.

## Found in the CP6 session (fifth pass)

- **A Decimal infinity in an object target of mixed values** is not dropped
  as an infinite float is. The spec pins the application's reading; no test
  covers the edge.
- **Finding 1, the fifth table.** The suite variant M14 does not make its
  test fail: the only visible difference is a stamp written `17900` or
  `17900.0`. That test has no broken copy behind it.
- **Gate time.** After the CP7 judge the per-edit gate takes 17.2 of its 20
  seconds and the self-test about 19 minutes. Pass 7 adds broken copies, so
  it needs a plan for both before it adds them.
- **A p-value for grouped runs.** None is reported (D61). A shuffle of whole
  entities is sound only when each entity holds one label.
- **D49 on clean tables of few-valued columns** stops them from about 4,000
  rows (30 of 30 measured). Settled with the repeat screen (D60).

- **Thin spots of the fifth pass.** M17 and M21 are each held by one test.
- **`CLAUDE.md` says "D1 to D60".** CP7 brings it up to date.

## Found in the CP7 session (sixth pass)

- **A code column that reads as dates.** Rule 2a excludes a whole-number
  column when every value is a valid `YYYYMMDD` date and two differ. A
  random value in range is a valid date with probability 0.037, so a column
  of two or three distinct codes can be excluded by chance. It shows in the
  column accounting, but the caller has no override until the deferred
  role override is built.
- **`duplicates="group"` gives no interval on small tables.** About 120
  feature combinations are needed for 30 holdout units.
- **One entity above three quarters of the rows** is the whole holdout, and
  the model trains on the rest, with a warning (D66). Before, some such runs
  stopped with `too_few_groups`.
- **What the older tests prove about an unsaid statement:** nothing, since
  they go through the helper of D65. The five opt-out tests and
  `test_statements.py` carry it.

## Found by the lead's real-data run of the sixth pass (2026-11-22)

- **The fast profile leaves out an entity larger than its cap, without a
  warning.** With an entity column the 2,000-row cap keeps whole entities,
  so an entity of more than 2,000 rows never fits: taxis with Manhattan as
  one entity (82% of the rows) was profiled on the other 1,128 rows, with
  only `dropped_row_cap` to say so. The spec asks for exactly this.
  Candidates: a warning that gives the share left out, or the named error
  that sends the caller to the full profile, as D20 does for wide panels.
  See `docs/measurements/CP7/build-on-real-tables/README.md`.

## After the seventh pass (2026-11-23)

- **Broken copies held by one test each:** F22, M13, M33, M34, M37, M39.
  CP9 gives each a second.
- **The loop's limits were raised** for the slower suite: the suite's time
  limit to 1,800 seconds and a task's to 120 minutes. The test author of the
  seventh pass was stopped at 60 minutes after writing every file; the
  mutation check with 70 variants no longer fits in a test author's run.
- **Pass 7's build on real tables:** `docs/measurements/CP8/build-on-real-tables/README.md`.

## Found in the CP9 session (eighth pass, 2026-11-25)

- **Thin guards.** M43 and M48 each rest on one test case, and the new
  behaviour of `stages.previous_value` at repeated keys has a test but no
  broken copy.
- **One gate run is three times slower on the application than on the
  reference:** `output.same_at_any_n_jobs` took 63 s against 21 s, and timed
  out under heavy load. Find out why.
- **A single series with one row written twice** now scores in band but
  still draws the `autocorrelated_panel_target` warning, on 30 of 30 seeds.
  The warning is truthful about repeated timestamps and misleading about
  the cause.
- **The order reading** warned on 0 of 30 small tables with a legitimately
  strong feature (R-squared 0.90 to 0.99, 50 to 200 rows); it has not been
  run on real tables yet.
- **A noisy copy of a regression target under the random scheme** is
  excluded up to noise of 12% of the target's spread, warned about from 15%
  to 30%, and silent from 35% (spec 10.2, re-measured).

## Found by the lead's real-data run of the eighth pass (2026-11-25)

Both are defects in the contract, not in the build; the application and the
reference agree. See `docs/measurements/CP9/build-on-real-tables/README.md`.

- **The order reading warns about noise columns that hold repeated values**
  (clean diabetes: `age` and `s6`; pure noise rounded to one decimal place,
  or a score from 1 to 40). Step 8 sorts rows of equal column value by the
  target, which makes the target fall across nearly every pair.
- **A copy of the target with five cells changed by a large amount** is
  warned about and kept under time order (headline 0.89 to 1.00), where a
  small change is excluded.
- **Why the judge missed both:** every numeric column of the gate and the
  battery is continuous, and the altered-cell pairs use one size of change.
  The battery needs columns with repeated values and changes of several
  sizes.

## Notes for the framework (ailoop-engineering), not for this repo

- The tests stage does not lock `tests/`, so on a second pass the existing
  tests are protected only by the test plan's wording and a `git status`
  check afterwards.
- `loop/prompts/test-author.md` says every test must fail before the build.
  On a second pass that is not so, and the test plan has to say it instead.
- The self-test accepts a failure mode when any one check that names it
  fails, so a check can sit in the matrix without ever having been seen to
  fail (second review, finding 4).
- A harness session resumed with `claude --resume` loses the stage label
  (`LOOP_STAGE`), so the guard treats it as a build session; the driver has
  no way to resume a harness session with the label set.
- With several helpers editing at once, the edit hook's gate runs overwrite
  each other under `verify/artifacts/runs/` and report false failures.
- When a test the driver has already passed fails, `reverify_done` reopens
  every finished task that has tests, not only the task that owns the
  failing file. Editing one old test for a changed contract can therefore
  rebuild a whole project (D53, D54).
- The driver counts `docs/` as application code, so measurement scripts
  appear as judge-awareness warnings in every build report.
