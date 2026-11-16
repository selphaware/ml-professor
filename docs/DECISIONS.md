# ML Professor — decision log

Each entry is a decision taken by the project owner. Briefs and change prompts
cite these by number. Entries are appended, never rewritten; a changed
decision gets a new entry that names the one it replaces.

## D1 — 2026-11-15 — Roadmap
Decision: Five phases, one harness grown phase by phase. (1) Library core: one
table, supervised, random and time-ordered validation, result object plus
`profile.json`. (2) Several datasets and joins, with grouped validation.
(3) Unsupervised methods. (4) Self-contained HTML report rendered from the
result object. (5) Dash app over the library.
Alternatives considered: time-ordered validation as its own second phase
(rejected: the first real datasets are time series, so a random-only phase 1
would give optimistic numbers on them); UI earlier (rejected: a UI over an
unverified library shows wrong numbers attractively).
Consequence: each phase is committed before the next begins. Phase 1 has no
joins, no grouped scheme, no unsupervised methods, no report and no UI.

## D2 — 2026-11-15 — Package and public interface
Decision: The package is `mlprofessor`, at `E:\CODE\ml-professor\mlprofessor`.
One public function, `mlprofessor.profile(data, target, task="auto",
time_col=None, gap=0, profile="full", seed=0, out_dir=None, n_jobs=1)`,
returns a `ProfileResult` whose parts are pandas structures (`headline`,
`models`, `columns`, `rows`, `warnings`, `predictions`) with `to_dict()` and
`to_json(path)`. A thin command line wraps the same function:
`python -m mlprofessor profile --data file.csv --target col --out dir`.
With an output folder it writes `profile.json` and `predictions.csv`.
Alternatives considered: command line only (rejected: the users are data
scientists working in Python); a class-based API (rejected for phase 1: one
function is the smallest thing that can be verified).
Consequence: the gate drives the tool only through these two entry points.
Everything a later report or UI shows must already be in the result object.

## D3 — 2026-11-15 — Phase 1 input and column handling
Decision: Input is one table, as a DataFrame or a CSV path. Numeric and
categorical columns are used. Missing values are imputed with a step fitted on
training rows only, and counted. Date, free-text, constant and id-like columns
are excluded, each listed with a reason. Rows with a missing target are
dropped and counted. Every row and every column is accounted for in the output.
Alternatives considered: Parquet and Excel input, and features built from
dates or text (deferred: each is a separate concern behind the same interface).
Consequence: nothing is dropped, coerced or skipped without appearing in the
output with a count and a reason.

## D4 — 2026-11-15 — Task types and how the tool decides
Decision: Binary classification and regression. `task="auto"` uses a stated
rule: two distinct target values means classification; a numeric target with
more than 20 distinct values means regression; anything else stops with a
named error asking the caller to declare the task. The decision and how it
was reached are recorded in the output.
Alternatives considered: binary classification only (rejected: many targets
of interest are continuous, such as returns); multiclass (deferred).
Consequence: the tool never guesses on an ambiguous target.

## D5 — 2026-11-15 — Headline measure and baseline
Decision: The headline is skill over a baseline, where 0 is no better than
the baseline and 1 is perfect. Classification: ROC AUC on the holdout,
baseline 0.5, skill = 2 x AUC - 1. Regression: skill = 1 - (model squared
error / baseline squared error) on the holdout, where the baseline predicts
the training mean. Accuracy is never the headline. Skill may be negative and
is reported as measured. The two scales are not comparable with each other
and the output says so.
Alternatives considered: accuracy (rejected: meaningless on a rare class);
the best cross-validation score (rejected: see D8).
Consequence: every headline is reported with its metric, its baseline and
its skill.

## D6 — 2026-11-15 — Validation when no time column is declared
Decision: A 25% holdout is split off first, at random from the seed,
stratified on the target for classification. The model is chosen by 5-fold
cross-validation on the other 75% only, refitted on that 75%, and scored once
on the holdout. The output records the scheme as random and carries a fixed
caveat that the number is optimistic if rows are ordered in time or one
entity appears on many rows.
Alternatives considered: cross-validation only, with no holdout (rejected:
the headline would be computed on rows used to choose the model).
Consequence: no fitted step of any kind sees a holdout row.

## D7 — 2026-11-15 — Validation when a time column is declared
Decision: The time column is declared by the caller (`time_col`), never
inferred. If a date column exists and none is declared, the tool uses the
random scheme and warns. With a time column: the holdout is the latest whole
timestamps that together hold at least 25% of the rows, so rows sharing a
timestamp stay on one side. The model is chosen on 5 expanding-window folds
inside the training period: the training timestamps are cut into 6 contiguous
blocks, and fold k trains on blocks 1 to k and validates on block k+1. `gap`
(default 0) is a number of distinct timestamps removed from the training side
immediately before the holdout and before each validation block; those rows
are counted. This serves a single series and a panel of many rows per date.
Alternatives considered: walk-forward re-fitting across the holdout
(deferred: heavier, and a single cut-off answers the phase 1 question);
inferring the time column (rejected: a silent guess about which rows are
"the future").
Consequence: under this scheme no training row is at or after the first
holdout timestamp, and none is inside the gap.

## D8 — 2026-11-15 — Models, search budget and selection
Decision: scikit-learn only. Families: a regularised linear model, a random
forest and histogram gradient boosting, each at default settings plus a fixed
grid of at most 8 combinations. A dummy baseline is fitted as a reference and
is never a candidate. The chosen model is the one with the best mean
cross-validation score; the headline is that model's score on the holdout,
never a cross-validation score. Only the chosen model is scored on the holdout.
Alternatives considered: other boosting libraries (deferred: more
dependencies to drift); a larger search (rejected for phase 1 on run time).
Consequence: at most 24 configurations and 120 cross-validation fits per run.

## D9 — 2026-11-15 — Uncertainty beside the headline
Decision: Always: the spread of scores across folds. Random scheme: a
bootstrap 95% interval on the holdout score (1,000 row resamples) and a
permutation p-value from shuffling the holdout targets against the fixed
predictions (1,000 shuffles, no refitting). Time-ordered scheme: the interval
comes from a moving-block bootstrap over contiguous holdout timestamps, with
block length = the cube root of the number of distinct holdout timestamps,
rounded up; the permutation p-value is omitted and the output states why
(shuffling assumes independent rows).
Alternatives considered: a permutation test that refits everything (rejected
for phase 1: multiplies the run time); row-level resampling under time order
(rejected: serial correlation makes it look more certain than it is).
Consequence: the headline never appears without an interval or a stated
reason for its absence.

## D10 — 2026-11-15 — Regression baseline under time order
Decision: When timestamps are unique, two naive baselines are scored on the
holdout: the training mean, and the previous value (the target `gap + 1`
timestamps earlier). The headline baseline is whichever has the lower holdout
error; both are reported. When timestamps are not unique (a panel), the
baseline is the training mean, and the tool warns when the per-timestamp mean
of the target is strongly autocorrelated.
Alternatives considered: training mean only (rejected: a price level looks
highly predictable when a model has only learned that today resembles
yesterday).
Consequence: on a random walk the reported skill is about zero.

## D11 — 2026-11-15 — Leak rule
Decision: A feature column is excluded and listed as a suspected leak when,
measured on training rows only, it alone predicts the target almost
perfectly, or it is filled in for one class and empty for the other. The
cut-offs are set by the harness stage from measurement on synthetic leak and
control data and reported for approval. A column that is merely strongly
predictive must not be flagged.
Alternatives considered: warn but keep the column (rejected: the headline
would then be the leak).
Consequence: a leak never contributes to the headline, and the output names
every excluded column with the reason.

## D12 — 2026-11-15 — Scale, run time and the fast profile
Decision: Target scale is up to 100,000 rows by 100 columns, with a full run
inside 10 minutes. `profile="fast"` runs the linear model plus one deep
decision tree, with no grid, at most 2,000 rows, and 200 resamples and
shuffles. Under time order the row cap keeps the most recent whole timestamps.
Rows left out by the cap are counted.
Alternatives considered: none; the harness stage times real fits and reports
if the budget is unrealistic.
Consequence: the gate uses the fast profile on every edit.

## D13 — 2026-11-15 — Named silent failures for phase 1
Decision: F1 a fitted step sees the holdout. F2 a column that copies the
target, or is only filled in after the outcome, is used as a feature. F3 the
headline is the best cross-validation score, not the holdout score.
F4 accuracy is the headline on a rare class. F5 the task is misread. F6
something is lost without being reported (rows, a column, a model family).
F7 time order is ignored (rows shuffled despite a declared time column, a
training row later than a holdout row, or the gap not applied). F8 under time
order a trending or random-walk target is scored against the training mean
only.
Alternatives considered: none struck.
Consequence: each has a gate check that names it, a broken reference
implementation that proves the check fires, and a regression test.

## D14 — 2026-11-15 — Holdout predictions are part of the output
Decision: The result carries the holdout predictions (`row`, `y_true`,
`y_pred`), written as `predictions.csv`.
Alternatives considered: headline only (rejected: the gate could then only
trust the tool's own account of its score).
Consequence: the gate recomputes the headline itself from the source file
and the predictions.

## D15 — 2026-11-15 — UI direction (provisional, revisit at phase 4)
Decision: A self-contained HTML report (`result.to_html(path)`) in phase 4,
then a Dash app in phase 5. Both use the same Plotly figures built from the
result object, and neither recomputes a number.
Alternatives considered: Streamlit (less control over layout, awkward with
long-running fits); Dash only (no shareable file for notebook users).
Consequence: none for phase 1 beyond D2's rule that the result object is
complete.

## D16 — 2026-11-15 — Forest strength and the run-time budget (replaces parts of D8 and D12)
Decision: Each family has at most 8 configurations including its default: at
most 24 configurations and 120 cross-validation fits. The forest stays at
full strength, with no cap on rows per tree. The regression forest considers
one third of the features at each split; that is the only departure from
scikit-learn's defaults. The budget in D12 is restated as: a full run on
100,000 rows by 100 columns inside 10 minutes on this machine using all
cores. The harness stage times a full run for both tasks before pinning the
grids and reports if either is over; it does not weaken a model to fit.
Alternatives considered: scikit-learn defaults on one thread (rejected:
Phase 0 measured about 55 minutes for classification and about 5 hours for
regression); capping each tree at 10,000 rows (rejected for now: it weakens
the forest exactly where there is most data). Deferred until a real table
proves too slow: a capped `balanced` profile, and searching the grid on a
subsample then refitting the winner on every training row.
Consequence: run time depends on the machine's cores. Tables beyond the
scale target run and take longer; nothing is capped silently.

## D17 — 2026-11-15 — Cores and reproducibility (replaces `n_jobs=1` in D2 and D12)
Decision: `n_jobs` defaults to all cores (`n_jobs=-1`); a positive whole
number uses that many. The value used is recorded in the output. The tool
uses `threadpoolctl`, already a dependency of scikit-learn, so that `n_jobs`
governs every thread pool, and holds linear algebra to one thread always. The
same seed must give the same result at any `n_jobs`; the gate proves it by
comparing runs at 1 and 4. If the harness stage finds that cannot be
achieved, it reports, and the fallback is "same seed and same `n_jobs`".
Alternatives considered: one thread by default (rejected: a plain call on a
large table would take about an hour).
Consequence: the gate runs single-threaded; real runs use the machine.

## D18 — 2026-11-15 — Accepted forms of the time column
Decision: The time column is accepted as a datetime-typed DataFrame column,
as it is; as a numeric column, ordered as numbers; or as text in ISO 8601
form, year first. Anything else stops with a named error. The tool never
guesses between day-first and month-first.
Alternatives considered: pandas' date parser (rejected: Phase 0 showed it
reads 01/12/2023 as 12 January without complaint, which mis-orders the rows
and puts the wrong rows in the holdout).
Consequence: a CSV with day-first dates must be parsed by the caller and
passed as a DataFrame. F7 gains a form (a misread time column) and a gate
check for it.

## D19 — 2026-11-15 — Id-like rule
Decision: A text column is id-like when its distinct values number more than
95% of the rows. A whole-number column, judged by value so that blanks do not
matter, is id-like only when every value is distinct and the values are
either consecutive or strictly ordered in file order. Any other whole-number
column stays a feature. The harness stage measures the rule and reports it.
Alternatives considered: the 95% rule for every non-float column (rejected:
Phase 0 showed it excludes price and volume columns on a 2,000-row table and
keeps them on a large one).
Consequence: a real integer measure is a feature at any table size. An
unordered random integer identifier is not caught and stays in as a feature.

## D20 — 2026-11-15 — Fast and full agree; wide panels under the fast profile
Decision: The task and the column roles that need no target are decided
before the fast profile's row cap, so both profiles agree on them. If the
capped rows hold too few timestamps to form the holdout and the six blocks,
the fast profile stops with a named error telling the caller to use `full`.
Alternatives considered: changing the cap rule for panels (deferred to phase
2, when entity columns allow a panel to be sampled properly).
Consequence: `profile="fast"` refuses a panel with more than roughly 250
rows per date.

## D21 — 2026-11-15 — Warning for a whole-number regression target
Decision: When `task="auto"` picks regression for a target whose values are
all whole numbers, the output carries a warning.
Alternatives considered: none; it covers the gap in D4 where an identifier or
a class with more than 20 codes is read as a continuous target.
Consequence: one more warning code.

## D22 — 2026-11-15 — Loading rules
Decision: pandas' default missing-value tokens (such as `n/a`) count as
missing, never as coerced. The loader reads with `low_memory=False`. Blank
lines are not rows. Duplicate column names stop with a named error; a blank
header is kept under pandas' name with a warning. "Date-like" is decided
against a fixed list of date formats, never pandas' lenient parser. The
floors for "too small to profile" and for the block bootstrap are set from
the harness stage's measurements and reported for approval.
Alternatives considered: forcing every column to text on load (rejected:
the tool would then own all type inference).
Consequence: a cell such as `1,200` is not read as a number; it is counted
as coerced to missing and a warning names the column. A CSV is read with
pandas' types, so text identifiers can lose leading zeros; a caller who
needs control passes a DataFrame. The output states this as a limitation.

## D23 — 2026-11-16 — Cut-offs approved as measured by the harness stage
Decision: The tool's cut-offs are approved as pinned in `spec.md` section 4.3
and measured in its section 10: leak by single-column skill at 0.98; leak by
missingness at more than 95% of one class and less than 5% of the other;
free text at more than 50% distinct; numeric coercion at 90%; date-like at
95% against fixed date forms; small-holdout warning at 50 (rows of the rarer
class, or rows in all for regression); too small to profile at 100 usable
rows or 10 of a class; no block-bootstrap interval below 30 holdout
timestamps (not 20); autocorrelated panel warning at 0.5 (not 0.8).
Alternatives considered: the starting values in the brief for the last two
(rejected on the measurements: coverage of 79% at 20 timestamps, and the
previous value beating the training mean from 0.5 upward).
Consequence: these values are part of the contract and change only by a new
decision backed by a new measurement.

## D24 — 2026-11-16 — Leak rule under time order (clarifies D11)
Decision: A column's single-column skill is measured on the scheme's own
selection folds, on training rows only. Under time order with unique
timestamps, regression skill is measured against the stronger of the training
mean and the previous value.
Alternatives considered: D11 read literally (rejected: it flags a series'
own lag as a leak).
Consequence: a lag feature of a single series is not excluded. See D31 for
the panel case, which is not covered.

## D25 — 2026-11-16 — Pinned model settings (adds to D16)
Decision: Six configurations per family, 18 in all and 90 cross-validation
fits, as listed in `spec.md` section 4.1. Histogram gradient boosting runs
with early stopping off, so its behaviour does not change at 10,000 rows.
Logistic regression is allowed 1,000 iterations. Forests predict on one
thread, which is what makes results identical at any `n_jobs`.
Alternatives considered: scikit-learn's defaults for those three settings
(rejected: a behaviour change at a row count, convergence warnings, and
results that differ between identical runs).
Consequence: a full run at 100,000 by 100 on all cores was measured at about
3 minutes for classification and 3.4 for regression. Identical output at any
`n_jobs` (D17) was achieved.

## D26 — 2026-11-16 — Step functions and the test seam are part of the contract (adds to D2)
Decision: `mlprofessor.stages` (twelve step functions) and
`mlprofessor.extensions.use` (the seam for recording fakes) are pinned in the
spec and built by the loop, so each data-handling task can be judged before
any model exists. The user-facing interface remains the `profile` function
and the command line.
Alternatives considered: two entry points only (rejected: the early tasks
could not then be judged by tests alone).
Consequence: builders have less freedom over internal structure than D2
intended.

## D27 — 2026-11-16 — The time-ordered interval is labelled as approximate (keeps D9)
Decision: The block bootstrap of D9 stays. Whenever it is used the output
carries a warning that the interval is approximate, and the limitations
statement says intervals tend to be too narrow on short holdouts and, under
time order, when errors are serially correlated.
Alternatives considered: a longer or data-driven block length (deferred: a
candidate for a later change, with its own coverage measurements); no
interval below 100 timestamps (rejected: monthly data would rarely get one).
Consequence: the nominal 95% interval was measured to cover about 86% at 30
timestamps on independent data and 64% to 85% under strong serial
correlation. The tool says so and does not hide it.

## D28 — 2026-11-16 — Numbers written with thousands separators (adds to D22)
Decision: A value such as `1,200` is never parsed as a number, because the
same text means 1.2 in some locales. A column made mostly of such values is
excluded with its own reason, `numeric_with_separators`, not as free text,
and a warning tells the caller to parse it and pass a DataFrame. A numeric
column in which some cells were coerced to missing for this reason carries
the same warning.
Alternatives considered: stripping the commas (rejected: a guess about
locale, the same class of error as guessing day-first dates).
Consequence: a financial CSV with formatted amounts loses those columns
until the caller parses them, and the output says exactly why.

## D29 — 2026-11-16 — Negative skill is explained
Decision: The limitations statement says that negative skill means the
models did worse than the baseline on the holdout and is to be read as no
evidence of predictability.
Alternatives considered: clipping skill at zero (rejected: D5 reports skill
as measured).
Consequence: one more fixed string in the output.

## D30 — 2026-11-16 — A warning tier below the leak cut-off (adds to D11)
Decision: A feature whose single-column skill is high but below the 0.98
exclusion cut-off is kept, and a warning names it and asks whether the column
is known before the outcome. The warning cut-off starts at 0.90, is measured
by the harness stage, and is reported for approval.
Alternatives considered: lowering the exclusion cut-off (rejected: it would
remove legitimately strong features from the headline).
Consequence: a near-copy of the target, such as one with 2% of labels
flipped, no longer passes without comment.

## D31 — 2026-11-16 — Limits of phase 1, accepted
Decision: These are accepted as known limits, visible in the output where
stated. (1) The id-like rule misses a shuffled identifier that has a blank
cell, and wrongly excludes a distinct whole-number measure in a table sorted
by that measure; both show in the column accounting. (2) In a panel, a lagged
level of a persistent target is flagged as a suspected leak, because the only
baseline there is the training mean; profile returns or changes until phase
2 adds entities. (3) The gate's p-value check fails a correct build with
probability 1 in 201. (4) The 10-minute budget was measured once and no check
enforces it.
Alternatives considered: an argument to override column roles (deferred: a
candidate for a later change).
Consequence: none of these blocks the build.

## D32 — 2026-11-16 — Venv base interpreter path
Decision: `mlprof_env\pyvenv.cfg` now has
`home = C:\Users\uahma\anaconda3\envs\py312`, the same interpreter by its
local path, in place of the mapped `A:` drive.
Alternatives considered: leaving it (rejected: the harness stage measured
Python processes started together stalling for 35 or 70 seconds on the
mapping's lock timeout).
Consequence: `doctor` was clean after the change, with the self-test passing
and the per-edit gate at 10.4 seconds.

## D33 — 2026-11-16 — Near-leak warning cut-off approved; objections weighed (completes D30)
Decision: The `strong_single_column` warning cut-off is 0.90, as measured in
the CP1 session: the near-copy of the target scored 0.938 to 0.960 and the
strongest legitimate column 0.609 to 0.748, over 30 seeds. Three objections
raised by that session are accepted as limits of phase 1. (1) A warned column
still drives the headline, because the tool cannot tell a near-leak from a
very strong real feature. (2) The `interval_approximate` warning fires on
every time-ordered run, including those where coverage is adequate, because
the tool cannot know how correlated the errors are. (3) Only the comma form
of a formatted number is recognised; `$1,200` and `1.200,50` are excluded as
free text or id-like.
Alternatives considered: excluding warned columns (rejected in D30).
Deferred as a later change: reporting skill with and without the warned
columns.
Consequence: any legitimate feature with single-column ROC AUC of about 0.95
or more will be warned. The warning asks a question; it does not accuse.

## D34 — 2026-11-16 — A partly formatted numeric column is excluded (replaces part of D28)
Decision: A column in which at least 90% of the non-missing values are plain
numbers or numbers with thousands separators, and at least one is a number
with thousands separators, is excluded with the reason
`numeric_with_separators` and its warning. The warning has this one form
only. A numeric column whose unparsed cells are other text is unchanged: it
is kept, with `values_coerced`.
Alternatives considered: keeping the column with those cells set to missing,
as D28 first had it (rejected: the cells lost are by construction the values
of 1,000 and above, so the headline would knowingly use a distorted column).
Consequence: a single stray `1,200` excludes a column. The output says
exactly which column and why, and the caller parses it and passes a
DataFrame.

## D35 — 2026-11-16 — Measurements are kept in the repo
Decision: The scripts, the seeds and a compact per-seed table behind every
band and cut-off measured in the CP1 session are saved under
`docs\measurements\CP1`. The two bands that session found tight on its own
seeds (`model.noise_reg` and `model.metric_clf`) are left as they are; the
session reports where the reference lands at the gate's fixed seed.
Alternatives considered: widening those bands (rejected: a band changes only
when it was measured wrong, and the gate judges one fixed seed).
Consequence: every band can be re-measured from the repo. The first harness
session's seeds were not recorded and are not recoverable.
