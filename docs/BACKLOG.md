# ML Professor — backlog

Things decided against for now, with the reason. Not a to-do list for the
build loop: each item becomes work only through a decision in
`docs/DECISIONS.md` and a change prompt. Phase 2 starts by reading this file
(D47).

**Status under D56 and D64 (2026-11-21).** Nothing known to be wrong is
carried into phase 2. Three hardening passes: pass 5 (CP6: entities,
`duplicates="group"`, the two stops, the fourth review's findings; built);
pass 6 (CP7: the fifth review's findings under D62 to D64); pass 7 (the
broader repeat screen first, under D60; then the judge and every remaining
weakness here). Only the features stay on the roadmap.

## Pass 7, first item: the broader repeat screen (D60)

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

## Roadmap items awaiting a decision

- **Deep learning.** Not in any phase yet (D8 limits the tool to
  scikit-learn). Two options discussed: a small neural network family from
  scikit-learn after phase 1; and a full phase with Keras or PyTorch,
  including sequence models for time series, after joins. The owner has
  asked for this to be added to the plan; which option is not yet decided.

## Deferred features

- **A `balanced` profile:** the full model families with a capped forest,
  for tables or machines where the full profile is too slow (D16).
- **Search on a subsample, refit on every row:** for tables far beyond
  100,000 rows (D16). Needs its own rule under time order.
- **Skill with and without the warned columns:** so a near-leak that is kept
  (D30, D33) cannot quietly be the whole headline.
- **A caller's override of column roles:** for an identifier the id-like
  rule misses, or a measure it wrongly excludes (D31).
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

## Notes for the framework (ailoop-engineering), not for this repo

- The tests stage does not lock `tests/`, so on a second pass the existing
  tests are protected only by the test plan's wording and a `git status`
  check afterwards.
- `loop/prompts/test-author.md` says every test must fail before the build.
  On a second pass that is not so, and the test plan has to say it instead.
- The self-test accepts a failure mode when any one check that names it
  fails, so a check can sit in the matrix without ever having been seen to
  fail (second review, finding 4).
- When a test the driver has already passed fails, `reverify_done` reopens
  every finished task that has tests, not only the task that owns the
  failing file. Editing one old test for a changed contract can therefore
  rebuild a whole project (D53, D54).
- The driver counts `docs/` as application code, so measurement scripts
  appear as judge-awareness warnings in every build report.
