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

## D36 — 2026-11-17 — Skill when the baseline's error is zero
Decision: When a baseline's squared error on the rows being scored is exactly
zero, skill cannot be computed and is reported as empty (`null`), never as an
error, an infinity or a zero. On the headline this comes with a warning that
states the reason; the score, both baselines and the rest of the output are
reported as usual and the run completes. The same rule holds everywhere skill
is computed: a bootstrap resample with a zero baseline is left out and
counted, and a single-column skill that cannot be computed flags no column.
Alternatives considered: stopping with a named error (rejected: it throws
away a valid row and column accounting); the crash found by the first review
(a `ZeroDivisionError` on a regression target that is flat across the
holdout under time order).
Consequence: a flat holdout gives a result that says plainly that skill could
not be measured on it.

## D37 — 2026-11-17 — Colliding feature names stop the run
Decision: The preprocessed matrix names a numeric column by its own name and
a categorical level as `<column>=<level>`. If two of those names would be the
same, the run stops with a named error that names the columns involved,
before any model is fitted.
Alternatives considered: renaming one of them silently (rejected: the first
review found the silent form, in which a numeric column named `c=red` was
overwritten, reached no model and was still listed as a feature).
Consequence: the caller renames a column. Every feature listed in the output
has reached the models.

## D38 — 2026-11-17 — A configuration that fails after it has been chosen
Decision: If the chosen configuration raises on the final fit or while
predicting the holdout, it is marked failed with its error text, keeps its
fold scores in the models table, and a warning is raised. The next best
configuration by mean cross-validation score is then chosen, and so on. If
none succeeds the run stops with the existing all-models-failed error. The
headline comes from exactly one configuration; one that failed contributes no
score.
Alternatives considered: stopping at once (rejected: one fragile
configuration would cost the whole run); erasing the failed configuration's
fold scores, as the first build did (rejected: the table would then
misreport what was tried).
Consequence: the holdout plays no part in the choice, so the headline is not
inflated by the fallback.

## D39 — 2026-11-17 — First review of phase 1: findings accepted
Decision: The first review (`docs\reviews\phase1-review-1.md`, verdict FIX
FIRST) is accepted in full. Beside D36 to D38: (1) the five test holes it
found are closed, each with a deliberately broken copy of the reference that
proves the new test fails: preprocessing statistics fitted on holdout rows,
category levels learned from every row, the leak screen using holdout rows,
a block bootstrap that draws timestamps independently, and a strict
inequality in the p-value. (2) An unusable output folder is an input error
raised before any work. (3) The tool leaves the caller's process settings as
it found them. (4) The no-features error names the excluded columns and
their reasons. (5) The console summary prints the interval, the p-value and
one line per warning. (6) A CSV that is not valid UTF-8 is an input error
wherever the bad byte sits, and leading lines of only white space are
skipped as pandas skips them. (7) The built library, not only the reference,
is timed once at 100,000 by 100 and compared at 1 and 64 cores.
Alternatives considered: shipping on the strength of the passing gate and
suite (rejected: three of the holes are small forms of F1, the failure the
tool exists to prevent, and nothing would catch a later regression).
Consequence: a third change prompt (CP3), new test files beside the existing
17, new build tasks, and a second review.

## D40 — 2026-11-17 — The score interval is kept when skill is undefined (adds to D36)
Decision: When the headline skill is undefined, the interval on the score is
still computed, from the resamples in which the score can be computed. Only
the interval on skill is empty. A resample with a zero baseline is left out
of the skill interval alone and counted separately. The p-value is empty,
with a reason. This was "CP3 ruling 1".
Alternatives considered: dropping both intervals (rejected: the model's
error on the holdout is still measurable).
Consequence: one more count in the output, for the resamples used for skill.

## D41 — 2026-11-17 — README examples use a table kept in the repo
Decision: Every command in `README.md` uses `examples\customers.csv`, a small
synthetic table committed to the repo. The README names nothing under
`verify\` or `tests\`, and a test holds it to that. This was "CP3 ruling 2".
Alternatives considered: examples that use the gate's generated data, as the
first build wrote them (rejected: those files exist only after the gate has
run, so the examples fail on a fresh clone, and so does one locked test).
Consequence: the builders' write scope gains `examples\`.

## D42 — 2026-11-17 — Three small rulings from the CP3 session
Decision: (1) When the headline skill is defined but its interval cannot be
computed, the warning has its own code, `no_skill_interval`;
`skill_undefined` is kept for an undefined headline skill. (2) Surrounding
white space is not part of a categorical level: `red` and `red ` are one
level, in counting, in the leak screen and in the preprocessed matrix.
(3) A header cell holding only white space is a name like any other, so two
such cells are duplicate column names and stop the run.
Alternatives considered: for (2) and (3), the reference's behaviour (levels
as loaded; white-space headers treated as blank). Rejected: the built
library already behaves as decided, both outcomes are visible to the caller,
and a level that differs only by a trailing space is a data-entry artefact.
Consequence: the spec and the reference change; the library does not.

## D43 — 2026-11-17 — Limits accepted after CP3
Decision: Accepted as known limits. (1) The guard on the p-value's
inequality covers classification only; for regression a tie is a tie only up
to rounding, and no tolerance is pinned. (2) F8 is caught by one test and
one gate check. (3) A configuration that fails partway through selection
loses the fold scores it had; a locked test requires it, and D38 covers only
a failure after the choice. (4) A name collision (D37) stops the run even
when the level sits only in rows the fast profile leaves out or in a column
the leak screen would exclude; that keeps the outcome independent of seed
and profile. (5) Undefined skill (D36) can arise under the random scheme
too, when the training mean predicts the holdout exactly.
Alternatives considered: none; each is recorded so it is not rediscovered.
Consequence: none of these blocks the second build.

## D44 — 2026-11-20 — No number that is not finite is ever written as empty
Decision: An infinity or a not-a-number is never turned into an empty value
in the output without a reason beside it. Three rules follow. (1) A
configuration whose fold score is not finite is treated as failed, like one
that raised. (2) If the chosen configuration's holdout score or a baseline's
score is not finite, the run stops with a new named error, code
`non_finite_score`, whose message says which number and the likely cause
(for one, a target value too large for squared error). (3) A p-value is
reported only when skill is a number. Quiet fallbacks that swallow a
non-finite value are replaced by a named error or removed.
Alternatives considered: the behaviour found by the second review, in which
a target holding one value near 1.8e308 gave an empty score, baseline and
skill, a p-value all the same, no warning, and a summary pointing at a
warning that was not there.
Consequence: an extreme target stops the run with a message the caller can
act on. Tests assert on the raw values, before any step that maps
not-a-number to empty.

## D45 — 2026-11-20 — The seed's range
Decision: `seed` is a whole number from 0 to 4,294,967,295 (2 to the power
32, less one). Any other value is an input error, raised before anything is
loaded.
Alternatives considered: folding a larger seed down (rejected: two
different seeds would give the same run, silently); no limit, as the spec
had it (the second review found that a seed of 2 to the power 32 crashes
inside scikit-learn with a raw error).
Consequence: a timestamp-style seed such as 20261119123456 is refused with a
clear message.

## D46 — 2026-11-20 — Second review of phase 1: findings accepted
Decision: The second review (`docs\reviews\phase1-review-2.md`, verdict FIX
FIRST) is accepted in full. Beside D44 and D45: (1) the console summary
prints four significant digits, so a small-scale target no longer shows as
0.0000. (2) Each gate check listed against a failure mode is proven by a
broken copy on which that check fails: a column lost from the accounting, a
model family lost from the table, a long row read silently, and a day-first
date parsed leniently. (3) Surrounding white space is not part of a target
value, as D42 already rules for features: `yes` and `yes ` are one class.
(4) The fast profile's "use the full profile" error is raised only when the
full profile would run; otherwise the ordinary too-few-rows error is raised.
(5) The README says correctly which rows the fast profile keeps under time
order. (6) The gate's self-test no longer overwrites the run folder a
builder is told to read. (7) Tests cover lines of form feed, no-break space
and quoted space before a header.
Alternatives considered: shipping with the first finding open (rejected: the
output contradicted itself and nothing flagged it).
Consequence: a fourth change prompt (CP4), a third pass of tests and build,
and a third review.

## D47 — 2026-11-20 — When phase 1 ships
Decision: After the third review, phase 1 ships unless that review finds a
High finding, or anything that misstates a number on plausible data. Every
lower finding is recorded in `docs\BACKLOG.md` and taken up at the start of
phase 2.
Alternatives considered: reviewing until a review finds nothing (rejected:
each review starts fresh and will always find something; the cost of a round
no longer matches the size of what it finds).
Consequence: the backlog file is part of the repo, and the phase 2 brief
starts from it.

## D48 — 2026-11-20 — Three rulings from the CP4 session
Decision: (1) A feature value too large for the 32-bit numbers the tree
models read (a magnitude above about 3.4e38) is treated as an infinite
value already is: missing, counted, and named in the `infinite_values`
warning. (2) A complex-number target is not numeric: a declared regression
on it is `task_unsupported`, and under `task="auto"` the ordinary rule
applies. (3) A target cell holding only white space is a missing target,
like an empty cell: the row is dropped and counted. It is not a class of
empty text.
Alternatives considered: for (1), leaving the raw scikit-learn error the
session measured in both implementations (rejected: a sentinel such as the
largest double is as plausible in a feature as in a target, which D44
already covers). For (2), the behaviour measured, in which the imaginary
part was silently lost. For (3), an empty-text class, as features have
(rejected: a yes/no target with a few blank-looking cells would stop as
ambiguous, when an empty cell in the same place is simply dropped and
counted).
Consequence: three more tests in the third pass. Nothing is dropped or
replaced without a count in the output.

## D49 — 2026-11-20 — Copies of a holdout row in training stop the run
Decision: Under the random scheme, a holdout row has a copy in training when
a training row has the same value in every feature column and the same
target. When more than a small share of the holdout rows have such a copy,
the run stops with a new named error whose message gives the count and the
share and says what the caller can do: remove the copies, or pass
`duplicates="keep"` to state that they are independent observations. With
`duplicates="keep"` the run goes ahead as before; the output records the
choice, the count and the share, and carries a warning. Below the cut-off
the run goes ahead and the count is reported. The cut-off is measured by
the harness stage and reported for approval. The argument's default is
`"error"`. Under time order the rule does not apply: rows at different
times are different observations, and rows at one timestamp fall on one
side of the split.
Alternatives considered: keeping copies on one side of the split now
(deferred to phase 2, where grouped validation belongs; value `"group"`);
warning only (rejected: the third review measured skill 0.918 and p = 0.001
on pure noise with every row present twice, against 0.009 without the
copies); dropping copies automatically (rejected: repeats can be genuine,
and dropping them changes the data).
Consequence: no headline is inflated by copies unless the caller has said
they are genuine. A table of a few categories will usually need
`duplicates="keep"`.

## D50 — 2026-11-20 — Whole numbers are compared exactly
Decision: (1) Two distinct whole-number timestamps are never merged. Integer
time keys are compared as integers, so nanosecond instants held as 64-bit
integers stay distinct. If that cannot be done without contradicting a
locked test, the run stops with the time-column error when two distinct
integers would become one number. (2) Distinct values of a whole-number
column are counted exactly, at any magnitude. (3) A column holding whole
numbers too large for 64 bits is excluded with a reason when it is a spare
column, and is a named error when it is the target or the time column.
Alternatives considered: the behaviour the third review measured: 600
distinct instants read as 235, the previous-value baseline lost, and a
random walk scored at 0.9994; a 50-valued column reported as constant; a
raw Python error on a 26-digit account number.
Consequence: nanosecond timestamps work as a numeric time column.

## D51 — 2026-11-20 — Number columns as a database driver returns them
Decision: A DataFrame column of generic objects that holds only numbers
(decimal numbers, booleans or whole numbers, with or without empty values)
is read as numeric. Any column that is still excluded as an unsupported
type gets a warning naming it, so it shows in the console summary.
Alternatives considered: excluding such columns with no warning, as built
(the third review measured skill of minus 0.011, against 0.779 with the
same values in typed columns).
Consequence: a table read straight from a database behaves like the same
table with typed columns.

## D52 — 2026-11-20 — Third review of phase 1: findings accepted
Decision: The third review (`docs\reviews\phase1-review-3.md`, verdict FIX
FIRST) is accepted in full. Under D47 its findings 1, 2 and 3 block
shipping. Beside D49 to D51: (1) one to nine fractional digits are accepted
in an ISO timestamp, and the refusal message says what is wrong. (2) The
six test gaps are closed, each with a broken copy that proves the test: the
interval's percentiles, the previous-value baseline inside the leak screen,
the fast profile's row sample, the spread across folds, tied predictions in
a resampled score, and the imputed count. (3) When the fast profile's cap
leaves too few rows and the full profile would run, the error says so.
(4) A type matrix is added to the suite: every kind of pandas column, as a
feature, the target and the time column, with ordinary and extreme values,
must end in a result or a named error, never a raw Python error, with every
column in the accounting and every reported number finite.
Alternatives considered: fixing type and magnitude faults one review at a
time (rejected: the last two reviews each found a different member of the
same class).
Consequence: a fifth change prompt (CP5), a fourth pass of tests and build,
and a fourth review, judged again under D47.

## D53 — 2026-11-20 — Rulings from the CP5 session's plan
Decision: (1) The stop of D49 needs both a share and a count: more than 1%
of the holdout rows, and at least 10 of them, have a copy in training.
Below that the run goes ahead, the count and the share are reported, and
the existing duplicate-rows warning stands; no new warning is added there.
Both values are measured by the harness stage and reported for approval.
(2) No locked test is edited. (3) When the run stops for copies and a date
column was excluded because no time column was declared, the message says
so. (4) The gate's own run of time-ordered data without a declared time
column passes `duplicates="keep"`, since every one of its holdout rows has
a copy in training. (5) Under D50, integer time keys are kept as exact
integers only when the column holds a magnitude above 2 to the power 53;
otherwise they stay 64-bit floats, as a locked test requires. (6) A target
whose values cannot be compared for equality (lists, dictionaries, sets) is
an unsupported task. A class value is written as a boolean, a number or
text. (7) When the fast profile's cap leaves too few rows or too few of a
class and the full profile would run, the error has its own code and does
not reuse the one that names timestamps.
Alternatives considered: for (1), a share alone at 1% (rejected: on a
holdout under 100 rows a single copy is above it, and two locked tests with
two or three repeated rows expect a finished run; editing them would make
them fail against the built library until the new task is built, and the
driver reopens every finished task when a test it has already passed
fails); a share alone at 10% (rejected: a table with 7% of its rows written
twice would run with only a count). The count of 10 mirrors the tool's
existing floor of 10 rows of a class: below ten rows a count cannot tell
coincidence from duplication.
Consequence: a small table with a handful of repeated rows runs, with the
count reported. A table with real duplication stops.

## D54 — 2026-11-20 — One locked test is loosened for the new argument (amends D53)
Decision: `tests\test_skeleton.py::test_profile_signature_and_defaults` pins
the public function to exactly nine parameters, so D49's new `duplicates`
argument cannot be added without changing it. The tests stage may loosen
that one test, and nothing else in any existing test file: the first nine
parameters stay pinned exactly as now, and a tenth, `duplicates` with the
default `"error"`, is accepted when present. The loosened test passes
against the library as built and after the change, so the driver reopens no
finished task. A new test, failing today, requires the tenth parameter, so
the signature is pinned exactly again once it is built. Separately, the
gate check `time.undeclared_warns` moves from task T11 to the task that
builds D49, because its run now passes `--duplicates keep`; if the driver
would still reopen T11, the reopening is accepted.
Alternatives considered: carrying the choice outside the signature, through
a context manager (rejected: the public function would no longer say what
it does); hiding the parameter from inspection (rejected outright: the test
would pass on a signature that is not the real one).
Consequence: the rule that existing tests are never edited has one recorded
exception, checked by reading the diff of that file after the tests stage.

## D55 — 2026-11-20 — Values approved after the CP5 session
Decision: (1) The cut-off of D49 is approved as measured: the run stops when
more than 1% of the holdout rows, and at least 10 of them, have a copy in
training. Over 30 seeds, clean data (13 gate runs and the example table)
had no copies and never stopped; with 5% of rows repeated every run
stopped; with every row present twice every run stopped. Between 1% and 2%
repeated the outcome depends on the seed, which is accepted: the inflation
there is small and the count is reported. (2) A decimal object that holds a
whole number is read as a whole number, not a float, so that D50 holds for
decimal columns too. (3) Accepted as limits: a whole-number feature above 2
to the power 53 is counted exactly but reaches the models as a float, so
distinct values can become one, with no warning; and three shapes arrive
already rounded before the tool sees them (digits held as text, a CSV
whole-number column with an empty cell, and a CSV value above the signed
range beside a negative one).
Alternatives considered: for (2), reading every decimal as a float, as
first agreed (rejected on measurement: a decimal column of nanosecond
instants merged, 200 read as 76).
Consequence: these values are part of the contract and change only by a new
decision backed by a new measurement.

## D56 — 2026-11-21 — Phase 1 is not shipped with known defects (replaces D47)
Decision: The owner's standard is that nothing known to be wrong is left for
later. Phase 1 closes when all three hold: a review has no finding above
Low; every Low finding is fixed, or the owner has ruled it not a defect; and
a generated-table test, run through the library and the reference, finds no
disagreement. Weaknesses that were set aside in `docs\BACKLOG.md` are fixed
in two hardening passes before that review, not carried into phase 2.
Features stay on the roadmap as features: joins, unsupervised methods, the
report, the Dash app and deep learning.
Alternatives considered: shipping under D47 with the fourth review's
near-copy finding recorded as a limit (rejected by the owner: no defect is
to be fixed in the future that is known now).
Consequence: at least two more passes through the loop. One limit cannot be
removed by any tool and is stated, not hidden: a copy of a row altered in
every column cannot be told from a different row, so the tool gives the
caller the control to declare which rows belong together.

## D57 — 2026-11-21 — Entities and repeated rows (extends D49; moves grouped validation into phase 1)
Decision: (1) A declared entity column, `group_col`. Under the random scheme
the rows of one entity stay on one side of the holdout split and of every
fold. Under time order the entity gives each one its own previous-value
baseline, and the leak screen measures a lag against it. (2) The automatic
screen for repeats is broadened beyond exact copies, to catch copies with an
added column, a batch number, or small changes to some columns. The harness
stage designs the rule, measures it on clean data and on a battery of
near-copies, and reports it for approval. (3) `duplicates="group"` keeps the
repeats the screen detects on one side of the split. The default stays
`"error"`. (4) A target's infinite cells are counted in the column
accounting, as the spec's output section already reads.
Alternatives considered: matching on all columns but one only (rejected as
the whole answer: the next variant has two differing columns); leaving
grouped validation to phase 2 with joins (rejected under D56).
Consequence: phase 2 becomes joins alone. The fourth review's findings 1, 3,
4 and 5 are fixed in the same pass.

## D58 — 2026-11-21 — The tool stops and asks where it used to warn (replaces part of D7 and D10)
Decision: (1) When the table has a date column and no time column is
declared, the run stops with a named error. The caller either names the
time column, or states explicitly that row order does not matter, and that
choice is recorded in the output with the existing caveat. (2) The default
of D49 stays `"error"`: a table whose holdout rows have copies in training
stops, as the Titanic file does, and the caller chooses. (3) The same
principle for panels: under time order, when timestamps repeat, no entity
column is declared and the target is strongly autocorrelated (the condition
that today only warns), the run stops and asks for the entity column, or
for an explicit statement that there are no entities.
Alternatives considered: a random split with a warning, as D7 had it
(rejected on real data: the VIX level with its time column undeclared was
reported at skill 0.939, against 0.023 when declared); a panel scored
against the training mean with a warning, as D10 had it (rejected on real
data: a five-stock price panel was reported at 0.979).
Consequence: a headline that the tool has reason to think is inflated is
never printed unless the caller has said, in the call, that the reason does
not apply.

## D59 — 2026-11-21 — How a locked test follows a changed contract
Decision: When a decision changes behaviour that an existing test pins, the
tests stage may loosen that one test so that it passes both before and
after the change, and nothing else in it. A new test, failing until the
change is built, pins the new behaviour exactly. Each such test is named in
the test plan, and the owner's lead reads the diff of every existing test
file after the stage. No existing test is ever edited so that it fails
against the library as built.
Alternatives considered: never editing a locked test (rejected: D54 showed
a contract change can make that impossible); editing tests to the new
behaviour directly (rejected: they would fail until the new task is built,
and the driver then reopens every finished task that has tests).
Consequence: D54 was the first use of this rule; D57 and D58 need it again.

## D60 — 2026-11-21 — The broader repeat screen leaves pass 5 (amends D57, part 2)
Decision: Neither rule proposed in the CP6 session is adopted. Pass 5 builds
the rest of D57 and D58 with D49's exact rule unchanged, and
`duplicates="group"` groups the exact copies D49 counts. The count of holdout
rows with an exact copy in training is reported in the result as a plain
number. The broader screen is the first item of pass 6, and a rule is
adopted only when (1) it has been measured on real public tables and on
made-up tables whose columns depend on each other, as well as on independent
ones; (2) it stops no clean table, other than tables of few-valued columns,
where the table alone cannot tell copies from look-alikes and asking is
right; (3) it stops every disguise of a full copy and oversampling at any
multiple; (4) every step of its count can be checked by hand.
Alternatives considered: the calibrated rule (rejected on real tables: it
stops clean diamonds, taxis and VIX prices, counting 8% to 35% of holdout
rows, because it expects columns to vary independently; and it runs when
the smaller class is copied 25 or 50 times before the split, on 4 of 5
tables); the first rule (rejected: it stops the clean taxi table);
continuing the research inside the CP6 session (rejected: the session
already carries the plan, the battery and the prototype). Measurements:
`docs/measurements/CP6/real-tables/`.
Consequence: copies disguised by an added column or a small change are not
detected in pass 5, and the spec says so. Under D56 phase 1 cannot close
until they are. `tests/test_duplicates.py::test_time_order_has_no_rule`
stays locked as it is.

## D61 — 2026-11-21 — Three rulings on the CP6 judge
Decision: (1) The copies rule of D49 applies whenever there is no time
column: under the random scheme and under the grouped scheme. A declared
entity column does not switch it off; copies that sit in different entities
on both sides of the split still stop the run unless the caller passes
`duplicates="keep"` or `"group"`. (2) `duplicates="group"` with any copy
found makes the scheme grouped, however few the copies: the interval comes
from resampling whole units and no p-value is reported, with the reason
stated. (3) The scripts and the per-seed table behind the two bands and the
grouped-uncertainty measurement of the fifth pass are kept in
`docs\measurements\CP6\bands\`, as D35 requires.
Alternatives considered: for (1), D49 under the random scheme only, as the
spec read when that was the only scheme without a time column (rejected:
declaring `group_col` would then switch the rule off silently). For (2), a
p-value when only a few rows are grouped (rejected: on data with no skill a
row-by-row p-value was below 0.05 in 5 of 30 runs, and one rule for every
grouped run is easier to trust than a rule with a size threshold).
Consequence: a caller who groups a handful of copies gives up the p-value.

## D62 — 2026-11-21 — Fifth review accepted; `duplicates="group"` forms its sets on the features alone (amends D57 part 3)
Decision: The fifth review (`docs\reviews\phase1-review-5.md`, verdict FIX
FIRST) is accepted in full. Its finding 1: under `duplicates="group"` the
rows kept together are those that agree in every feature column. The target
plays no part in forming the sets, so the split never looks at the target
beyond stratifying. What `"group"` then means is stated in the output and
the README: no combination of feature values is on both sides. The message
of the copies stop says when each answer fits: `"keep"` when the repeated
rows are different cases that look alike, as in a table of few-valued
columns; `"group"` when they are one case recorded more than once. D49's
count is unchanged: a copy is still a row equal in every feature and the
target.
Alternatives considered: sets of rows equal in every feature and the
target, as built (rejected on measurement: on noise in a table of 24
feature combinations it scored -0.28, -0.27 and -0.56 in the lead's check
and down to -0.85 in the review, because a held-out set can have only its
opposite class in training; on the features alone the same tables scored
-0.07, +0.06 and -0.04); a named error when rows that agree in every
feature differ in target (rejected: it leaves a caller with oversampled
rows no sound choice).
Consequence: on a table where different cases share feature values,
`"group"` is cautious: Titanic scores 0.645 against 0.745 with `"keep"`.
Measurements: `docs\measurements\review-5\`.

## D63 — 2026-11-21 — The caller always states time order and entities (replaces D58 parts 1 and 3; amends D6 and D7)
Decision: Every call says two things, and the tool never assumes either.
Time order: `time_col` is a column's name, or `False` for "the order of the
rows in time does not matter". Entities: `group_col` is a column's name, or
`False` for "no rows belong together". When either is left unsaid
the run stops with a named error, whatever the table holds; the caller
learns everything that is missing from one message. On the command line the
same holds for `--time-col` or `--no-time-order`, and `--group-col` or
`--no-entities`. The rule is for `profile()` and the command line; the step
functions take what they are given. The existing warnings stay as the
tool's doubt about a statement: a date column under `time_col=False`, and a
persistent panel under `group_col=False`. Both statements are recorded in
the output.
Alternatives considered: stopping only when the tool sees a reason, as D58
had it (rejected on the fifth review: a panel of 8 to 12 timestamps was
asked about in 6 runs of 20; a panel whose timestamps never coincide was
scored at 0.80 to 0.92 with nothing said; a day held as a whole number was
scored at 0.947 with no warning; each is an everyday form of financial
data, and a detector for each would leave the next form open); defaults
that assume independence with a caveat, as D6 had it.
Consequence: two more words on every call. The conditional stops of D58
and their cut-offs no longer decide whether a run goes ahead. Most existing
tests call with nothing said, so D59 is applied at scale in pass 6.

## D64 — 2026-11-21 — The grouped holdout, three hardening passes, and the loop's budget (amends D56)
Decision: (1) Fifth review, finding 6: the holdout of the grouped scheme
does not take far more than a quarter of the rows because one large entity
was drawn early. The draw leaves out an entity that would overshoot when
another can complete the holdout, and a warning says so when one entity
alone is too large for a sound split. The harness stage measures the rule
and reports it for approval. (2) Findings 5 and 7 are fixed with tests and
broken copies. (3) The hardening passes of D56 are three: pass 6 builds the
fifth review's findings (D62, D63 and this decision); pass 7 builds the
broader repeat screen under D60, the generated-table test, a broken copy
for every gate check never seen failing, and the rest of
`docs\BACKLOG.md`. (4) The loop's total budget is raised from 150 to 250
US dollars, in `loop\loop.json`; 79 were spent after five passes.
Alternatives considered: one pass for both (rejected: too large to check
properly, and the repeat screen needs its own measurement).
Consequence: phase 1 closes under D56 after pass 7's review at the
earliest.
