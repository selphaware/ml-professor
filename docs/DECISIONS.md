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
