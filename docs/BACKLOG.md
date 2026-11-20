# ML Professor — backlog

Things decided against for now, with the reason. Not a to-do list for the
build loop: each item becomes work only through a decision in
`docs/DECISIONS.md` and a change prompt. Phase 2 starts by reading this file
(D47).

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

## Notes for the framework (ailoop-engineering), not for this repo

- The tests stage does not lock `tests/`, so on a second pass the existing
  tests are protected only by the test plan's wording and a `git status`
  check afterwards.
- `loop/prompts/test-author.md` says every test must fail before the build.
  On a second pass that is not so, and the test plan has to say it instead.
- The self-test accepts a failure mode when any one check that names it
  fails, so a check can sit in the matrix without ever having been seen to
  fail (second review, finding 4).
- The driver counts `docs/` as application code, so measurement scripts
  appear as judge-awareness warnings in every build report.
