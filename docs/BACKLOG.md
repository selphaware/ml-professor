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
