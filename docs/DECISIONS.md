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

## D65 — 2026-11-21 — D59 may be applied through a shared helper (extends D59)
Decision: When a changed contract touches what many existing tests do on
their first line, the tests stage may make the change once in a helper of
the test files and not in each test. The helper makes the call as the test
wrote it, and repeats it with the missing statement only when the run stops
with the new named error for a statement the call did not make; any other
error is raised as it is. Every helper and every test changed is named in
the test plan with its exact form, a test that pins the stop itself opts
out of the fall-back, and the lead reads the diff of every file after the
stage. The harness session reports, for each broken copy that the suite
caught before, how many cases fail on it before and after the loosening.
Alternatives considered: editing each test body (rejected: 27 more bodies
for the same effect, and more room for a slip).
Consequence: for D63, 30 replacements in 21 files cover 1,435 cases that
leave a statement unsaid.

## D66 — 2026-11-21 — Rulings on the plan of pass 6 (CP7)
Decision: (1) The grouped holdout, as the harness session measured it: a
unit joins the holdout while the holdout stays under a quarter of the rows;
the smallest unit passed over completes it; every unit it can spare goes
back to training. For classification the same is done for the units of each
class. (2) A warning is raised whenever a unit holds a quarter or more of
the rows it is drawn among: when it is kept in training because of its
size, the warning gives its share and says that the headline does not test
it; when the smaller units cannot fill a quarter, that unit is the holdout
and the warning says so. (3) A whole-number column in which every value
reads as a calendar date written `YYYYMMDD`, years 1900 to 2100, with at
least two distinct values, is a date form: excluded as a date, named by the
existing warning under `time_col=False`, and accepted as the time column.
(4) No check that the levels of a column are series of their own is built.
(5) The count and the warning of `duplicates="group"` are renamed so that
neither says "copies": they count rows that agree in every feature, which
is not what a copy means at step 8a. (6) Both error codes are kept; when
time order is unsaid the error is `time_order_undeclared`, and its message
also asks for the entities when both are unsaid. The error is raised at
step 5, after the errors of steps 1 to 4, so that it can name the columns
that hold dates and say that timestamps repeat. On the command line it is
exit code 3 with the function's own message. (7) The message of the copies
stop names `"keep"` first, as the session worded it.
Alternatives considered: for (2), a warning only when no sound split
exists (rejected: with one entity of 60% of the rows the holdout was
exactly a quarter and nothing said that the largest entity is never
tested); keeping a too-large unit in training and holding out a quarter of
the rest (rejected: about 50 rows of small entities). For (3), a warning
only (rejected: the same date written as text is excluded, and as a number
it is a feature that carries the row's place in time). For (4), the check
as measured: it flagged every panel, and also a segment that only sets the
level (0.89 to 0.997); the stricter form missed panels of 8 and 12
timestamps.
Consequence: day counts and `YYYYMM` are not seen as dates, and a statement
of `group_col=False` is questioned only by the panel warning, which fires
in about half of panels of 8 timestamps. Both are stated as limits.

## D67 — 2026-11-22 — Roadmap: feature engineering and deep learning (replaces the order of D1; lifts part of D8)
Decision: The phases are now: (1) the library core, closing under D56 and
D64; then, as the last changes of phase 1, a pip-installable package (D68)
and a small neural network family from scikit-learn beside the three
families of D8. (2) Feature engineering on one table. (3) Several datasets
and joins, with features aggregated across tables. (4) Deep learning with
Keras or PyTorch, including sequence models for time series. (5)
Unsupervised methods. (6) The self-contained HTML report. (7) The Dash app.
For phase 2 the owner has agreed, as starting points for its brief: the
output gives the headline on the columns as given and the headline with the
engineered features, the gain with its own interval, the table of features
with their definitions and which were kept, and the engineered columns for
the caller's rows as a DataFrame; the first families are date parts,
numeric transforms with capped pairwise ratios and differences, category
frequency and target encodings, missing-value flags and, under a time
column, lags, differences and rolling statistics for each entity; it is an
argument of `profile()`, off by default. Text features and a reusable
transformer for new data wait.
Alternatives considered: joins before feature engineering, as D1 had it
(rejected: single-table engineering needs nothing from joins, and joins
then add the cross-table aggregates on top); only one of the two neural
options (the owner chose both).
Consequence: each phase gets its own brief and its own failure modes.
Phase 2's rules are fixed before its brief: features are generated and
chosen inside the training folds only, an encoding that uses the target is
fitted on training rows only, time features use the past only and respect
the gap, and the holdout chooses nothing.

## D68 — 2026-11-22 — A pip-installable package
Decision: ML Professor is published as a Python package that `pip install`
can fetch: the distribution is named `mlprofessor`, like the import;
public, on PyPI, under the MIT licence. It is made installable from the
repo first, and the owner makes the PyPI release when he chooses; agents
hold no publishing credentials and upload nothing. The installed package
declares tested version ranges for its dependencies; the exact pins stay in
`requirements.txt` for the gate. Optional extras, `mlprofessor[dl]` and
`mlprofessor[ui]`, carry the deep-learning and Dash dependencies when those
phases exist. Python 3.12 only at first. A console command `mlprofessor`
is added beside `python -m mlprofessor`. The gate gains a check that builds
the package, installs it into a clean environment with no network, and gets
the same result as the repo copy.
Alternatives considered: exact pins in the published package (rejected: it
could then hardly be installed beside anything else).
Consequence: identical results are promised only on the pinned set, so the
output goes on recording the library versions used. Whether the name is
free on PyPI is checked first when this change is written: two web searches
found no package of that name, which does not settle it.

## D69 — 2026-11-22 — Sixth review accepted (amends D66 point 3 and D62)
Decision: The sixth review (`docs\reviews\phase1-review-6.md`, verdict FIX
FIRST) is accepted in full. (1) The rule for dates written as whole numbers
tolerates values that are not dates: a column of `YYYYMMDD` dates stays a
date column when it also holds a sentinel such as `99991231` or `0`. The
harness stage measures the exact form, a share or a small number of
distinct values that are not dates, and reports it for approval. When such
a column is the time column, a value that is not a date stops the run with
a named error that names it; it is never ordered as a number. (2) Under
`duplicates="group"`, when the leak screen excludes a column and any
combination of the remaining feature values is then on both sides of the
split, the run stops and tells the caller to remove the column and run
again. (3) The text of the grouped warning and the README say that
`"group"` asks whether a combination of feature values never seen can be
predicted, which on a table of few-valued columns can give negative skill
where `"keep"` finds signal; the bias on noise is measured as the number of
combinations falls. (4) `YYYYMMDDHH` and period labels join the stated
limits. (5) The text of `date_column_undeclared` is pinned and names the
scheme correctly. (6) The dormant branch that would turn a number that is
not finite into an empty value is replaced by a raise.
Alternatives considered: for (1), every value a date, as D66 had it
(rejected on measurement: one cell of 1,800 set to `99991231` or `0` made
the column a feature and gave skill 0.97 to 0.99 with no warning); a share
of 95% alone (rejected on the lead's check: with 5% of the cells a sentinel
the skill was still 0.92 to 0.95, and an open-ended date column can hold
far more). For (2), forming the sets again after the leak screen (rejected:
the screen was fitted on the training rows of the first split).
Consequence: measurements in `docs\measurements\review-6\`.

## D70 — 2026-11-22 — The fast profile and an entity larger than its cap
Decision: With an entity column and no time column, when an entity holds
more rows than the fast profile's cap, the fast profile stops with a named
error that sends the caller to the full profile, as D20 does for a wide
panel. The same holds for a set of rows kept together under
`duplicates="group"`.
Alternatives considered: leaving such an entity out, as built and as the
spec read (rejected on real data: with Manhattan as one entity of 82% of
the taxi rows, the fast profile reported skill 0.878 on the other 1,128
rows, with only a row count to say so); a warning (rejected: the result
would still describe a different table from the one given).
Consequence: `docs\measurements\CP7\build-on-real-tables\README.md`.

## D71 — 2026-11-22 — How the hardening passes are run (amends D64)
Decision: (1) The judge gains a perturbation battery: for every gate
dataset whose right conclusion is known, small changes are applied (one
sentinel cell, an added column that copies the target, an added constant or
identifier column, shuffled rows, a repeated row, and others the harness
proposes) and the conclusion must still hold. (2) No new detection rule is
added during the hardening passes; a rule that exists is repaired or
stated as a limit. (3) Pass 7 builds D69 and D70, the perturbation battery,
the generated-table test of D56 and a broken copy for each gate check never
seen failing. Pass 8 builds the broader repeat screen under D60, or the
ruling that replaces it, and the rest of `docs\BACKLOG.md`.
Alternatives considered: going on as before (rejected: each of the last
four reviews found, by changing one cell or adding one column, another form
of a fault the gate proved in one form only).
Consequence: one more pass than D64 planned. Phase 1 closes under D56 after
pass 8's review at the earliest.

## D72 — 2026-11-22 — Rulings on the plan of pass 7 (CP8) (amends D69 point 1 and D70)
Decision: (1) The date rule is judged by the column's distinct values, so a
sentinel counts once however many cells hold it: a column is a date column
when at least two distinct values read as `YYYYMMDD` dates and at least
nine do for each distinct value that does not. A second clause, at least
five distinct dates beside at most two values that are not, is added if the
harness measures no wrong flag with it. As the time column, a value that is
no date stops the run with `time_column_invalid`, naming the value and its
rows. (2) The stop under `duplicates="group"` after a removed leak reuses
`holdout_copies_in_training`, with no threshold: one combination of feature
values on both sides stops the run. (3) The two fixed texts are as the
harness session worded them. (4) D70's error is a new code of the fast
profile's family, `fast_profile_group_too_large`; when the full profile
would stop too, its own error is raised. (5) D70 is corrected in two ways:
an entity that holds a quarter or more of the usable rows is always kept by
the fast profile's cap, and the run stops when such entities do not fit;
and the rule is for entity columns only, since under `duplicates="group"`
the sets are formed among the rows the cap keeps. (6) Where the library and
the reference differ and the spec was silent, the library's reading is
pinned: a cell that cannot be hashed is compared by its text when repeated
rows are counted, and there is one `fold_not_scored` record for each fold
not scored. (7) The gate starts its runs 0.25 seconds apart, 16 side by
side. (8) The battery and the generated-table test are run by the loop
through one test file once a pass's last task is reached, and by the lead
by hand before a review.
Alternatives considered: for (1), a share of the cells (rejected on
measurement: wrong on every column with its sentinel in 30% or 60% of the
rows); at most two values that are no dates (rejected alone: three
sentinels of one cell each defeat it). For (5), an entity below the cap
left to the draw (rejected on measurement: one of 40% to 57% of the rows
was left out on 12 to 30 of 30 seeds).
Consequence: stated limits of the date rule: more than one distinct value
in ten that is no date, or too few distinct dates beside a sentinel.

## D73 — 2026-11-22 — What the judge's new tools found, and the order of work (amends D10, D11 and D71)
Decision: On their first run the perturbation battery and the
generated-table test found faults that no review had. Each is fixed, none
is carried as a limit: (1) Under time order with no entities, one repeated
row makes the timestamps no longer unique, the previous-value baseline is
dropped, and a random walk is reported at 0.999. The previous-value
baseline is extended to repeated timestamps: the mean target of the
timestamp before. (2) An exact rescaled copy of a regression target is not
excluded by the leak screen under time order, or beside a dominant entity
(skill 0.95 to 1.0; 27 of 123 generated tables that carried one). The
screen is repaired so that a column which reproduces the target is always
excluded; the harness measures the repair first. (3) One extreme cell can
move a regression skill out of its band, and one in the target can switch
the leak screen off. The tool reports when one row carries most of the
error, which is a statement about the result and not a new rule about the
data. (4) The order in which rows are handed to a fit under time order is
pinned, so that forests agree between the library and the reference. (5)
Pass 7 goes ahead as planned; the battery lists these, and the identifier
that becomes a feature when rows are shuffled, as open and owned. Pass 8
fixes everything the battery and the generated-table test still list.
There is no review after pass 7: a review is run when both tools are
clean. The broader repeat screen and the rest of the backlog follow pass 8.
Alternatives considered: stating (1) or (2) as limits (rejected: each is a
confidently wrong headline from one row or one column); building them in
pass 7 (rejected: each changes a pinned rule and needs its own
measurement, and pass 7 is already large).
Consequence: one more pass than D71 planned. D56's closing rule gains a
condition: the battery lists nothing as open.

## D74 — 2026-11-24 — Rulings on the plan of pass 8 (CP9) (completes D73; amends D10, D11, D19, D24 and D71)
Decision: (1) The previous-value baseline at repeated timestamps is
approved as measured; the panel warning names the baseline the run used.
(2) The leak screen's repair is approved as measured and counts as a repair
under D71: extreme training targets set aside from the screen, a robust
straight line beside the tree, the worst 1% of rows left unscored for the
exclusion cut-off only, and for classification the column read directly.
Under time order a copy of the target with more than 1% of its cells
altered, a copy with small noise, and a curve of the target must each at
least be warned about; if the repaired screen cannot, each column is also
read with the chosen model, for the warning only. (3) Extreme values: the
share of the squared error that the largest row carries is reported for the
holdout and over the selection folds, for the model and the baseline, with
a warning above 0.5. A second warning names a numeric column or the target
when it holds a value far outside all its other values. That warning is an
exception to D71's bar on new rules, made on the measurement below; it
changes no number, and is built only if the harness measures it quiet on
heavy-tailed columns. (4) Rows are handed to a fit in time order, ties in
file order. (5) The id-like rule keeps D19's clause (every value distinct,
and consecutive or ordered in file order) and gains a second: more than 95%
of the values distinct and filling at least half of their range, whatever
the row order. (6) Under the grouped scheme, when the folds are too small
to choose between configurations, as the harness measured it, no choice is
made: every fold stays scored, the first configuration is used, and a
warning says so. (7) `tests/test_judge_tools.py` is loosened under D59 to
accept exactly the failures the plan lists, and a strict new test is owned
by the pass's last task. (8) The names the harness proposed are approved.
Alternatives considered: for (3), the holdout statement alone, as CP9
asked (rejected on measurement: it named the row in 475 of 482 cases where
the cell was in the holdout and in 57 of 1,530 where it was among the
training rows; one target cell of 999999 in training gave skill 1.0000 on
noise with no warning); a cut-off of 0.25 (rejected: it fired on 11 of 30
clean tables of 100 rows). For (5), the new clause in place of D19's
(rejected: an identifier that steps by 3 or more would no longer be
caught). For (6), leaving a fold with too few training rows unscored, as
CP9 started from (rejected on measurement: it changed the choice on 0 of 30
tables; the fault is four folds that validate on two to four rows).
Consequence: a whole-number measure in a table sorted by it is still
wrongly excluded, as D31 accepted. Both open lists are empty after this
pass, or the pass does not go ahead.

## D75 — 2026-11-24 — The caller sets the iteration cap (amends D25; adds to D67)
Decision: `profile()` gains an argument `max_iter`, default 1,000, and the
command line `--max-iter`. It is a positive whole number; any other value
is an input error. It is the iteration cap of every model that iterates
until it converges: logistic regression (fixed at 1,000 until now), ridge
regression where its solver iterates, and the neural network family when it
is added. It does not replace the number of trees of gradient boosting,
which scikit-learn also calls `max_iter`: that stays on the pinned grid of
D25, and the output says so. Trees and forests have no such setting. The
value used is recorded in the output. When a configuration stops at the cap
without converging, a warning names it. It is built with the change that
adds the neural network family, after the hardening passes close.
Alternatives considered: the value also replacing boosting's tree count
(the owner chose against it: at the default it would change every
full-profile result, take boosting 3 to 10 times longer, and with early
stopping off overfit more); leaving the cap fixed, as D25 had it.
Consequence: a twelfth parameter. At the default every result is identical
to today's. The harness stage of that change first reports what the tool
does today when the cap is hit.

## D76 — 2026-11-24 — Second rulings on the plan of pass 8 (CP9) (amends D74 points 2 and 3, D30 and D67)
Decision: (1) The warning tier of D30 gains a second reading, in place of
the chosen-model reading of D74: with the training rows put in a column's
order, when the target moves the other way between at most 10% of
neighbouring rows, and the column does not follow the previous value as
closely, the existing `strong_single_column` warning names the column. It
is a warning only and excludes nothing. (2) Under time order a copy of the
target with noise of up to 1% of the target's spread must be excluded or
warned about. Larger noise is a stated limit: such a column cannot be told
from a strong feature measured at the same moment, and whether it is known
before the outcome is the caller's to say (the fixed limitation on
look-ahead). (3) The extreme-value warning, code `extreme_value`, need not
name a value that is not far outside its column and moves no number; the
harness re-places its cut-off with margin on both sides of the measured gap
and reports it. (4) A lag excluded as a suspected leak under
`time_col=False` is the rule applied as written, and stays a stated limit
(D31). (5) A caller's override of column roles is built with the changes
that follow the hardening passes, beside the package and the neural
network family. (6) The battery's stated-limit list holds exactly four,
each with its decision: a copy that differs by an added column (D60); a
whole-number time column with an empty cell (D55); a distinct whole-number
measure in a table sorted by it (D31); a copy of the target with noise
above 1% under time order (this decision). A pair that asserts an outcome
known to be wrong is never counted as holding.
Alternatives considered: for (1), reading each column with the chosen
model, as D74 had it (rejected on measurement: silent on 29 of 30 altered
copies and on 27 to 30 of 30 curves). For (2), a rule that warns on noisy
copies (rejected: the harness found none that would not warn on every
strong legitimate feature under time order).
Consequence: with a sharper screen more legitimate columns that predict
98% or more of the target are excluded, which is why the override moves
forward.

## D77 — 2026-11-25 — Seventh review accepted; the headline says what it rests on (amends D11, D30, D33, D56, D74 and D76)
Decision: The seventh review (`docs\reviews\phase1-review-7.md`, verdict FIX
FIRST) is accepted in full, with the two findings of the lead's real-data
run of the eighth pass. (1) Every result states which columns the headline
rests on: for each feature, the skill of the chosen model on the holdout
with that column's holdout values shuffled, and the fall from the headline.
It describes the result, changes no number and chooses nothing. (2) A
headline skill at or above 0.98, the level at which one column alone is
excluded as a suspected leak, always carries a warning that names the
columns it rests on most and asks whether they are known before the
outcome. (3) The leak screen also reads a column by its levels, the mean
target of each level fitted on a fold's training rows: for text columns
and for numeric columns of few distinct values, in both tasks. The harness
measures how often an honest column of many levels is wrongly excluded
before the rule is adopted. (4) The order reading of D76 no longer puts
rows of equal column value in the target's order. (5) A copy of the target
with a handful of cells changed is excluded whatever the size of the
change. (6) Under time order a holdout far above a quarter of the rows is
named by a warning. (7) `tests\test_judge_tools.py` is strict again. (8)
The battery gains columns with repeated values, coded columns, changes of
several sizes and columns that give the target only together. (9) D56's
closing rule is amended: a column, or columns, that carry the target in a
form the screen does not read count as a defect only when the output
neither excludes them, nor warns about them by name, nor names them under
the warning of point 2. What that statement catches is a stated limit of
the screen.
Alternatives considered: one more reader for the screen and nothing else
(rejected: each of the last four passes taught the screen one more
disguise and each review found the next; two columns that give the target
only together scored 1.000 with nothing said, and no reading of single
columns can see them); a warning below 0.98 (rejected: it would fire on
ordinary strong tables; the statement of point 1 is on every result).
Consequence: a legitimately easy table, such as one that scores 0.995,
carries the warning of point 2. Measurements: `docs\measurements\review-7\`
and `docs\measurements\CP9\build-on-real-tables\`. After this pass: a
review, then the repeat screen owed under D60.

## D78 — 2026-11-25 — Rulings on the plan of pass 9 (CP10) (amends D77 points 1, 3, 7 and 9)
Decision: (1) Every result names what the headline rests on, in plain
sight and with no threshold on the headline: the headline record and the
console summary list the columns whose fall is at least 0.05, at most five,
largest first, with their falls. The warning at 0.98 stays as it is. (2) A
carrier of the target that lifts the headline without reaching 0.98, and
that the screen neither excludes nor warns about, is named there and is a
stated limit of the screen: a coded copy of a regression target under time
order, and two columns that give the target only together under time order
and the fast profile, are the measured cases. D77 point 9 gains that
fourth outcome. (3) Columns that are exact copies of one another are
shuffled together and share one fall. (4) A coded copy of eight bands or
fewer is warned about, not excluded: it is the warning tier of D30. (5) In
the reading by levels, validation rows whose level was not seen in the
fold's training rows are not scored; the harness pins when too few rows
are scored for the reading to have a value. (6) A column of codes that
alone predicts the target at 0.98 is excluded as a suspected leak even
when it is an undeclared entity; the message says to declare it as
`group_col` if that is what it is. (7) The harness session's details are
approved: three keys for each column, five shuffles on a stream of their
own, the naming rule of 0.05 and five, nulls with a warning when a model
cannot predict a shuffled matrix; the order reading by pairs of rows
between neighbouring values; rule 2's reading against the fold's training
median where too many rows are marked; `holdout_far_above_quarter` at a
share of 0.5; the code names. (8) The judge-tools tests keep their
per-pass form: the older files accept exactly the failures the pass owns,
and the newest pass's file is strict.
Alternatives considered: for (1) and (2), a warning when one column
carries most of the headline (rejected: it needs a floor, it would fire on
a large share of honest tables, which often have one dominant feature, and
each threshold is the next edge a review probes; the tool cannot know
whether a column is known before the outcome, and can always say what the
number depends on). For (5), the baseline's prediction for an unseen
level, as CP10 worded it (rejected on measurement: 24 battery pairs
failed). For (8), one test that is strict for good (rejected: both files
are in the loop's ratchet, so a strict one fails mid-build and reopens
every finished task).
Consequence: a carrier below 0.98 is named, not warned about.

## D79 — 2026-11-26 — The CP10 harness session is finished on Opus (one-off)
Decision: The owner's Fable allowance ran out after the rulings of D77 and
D78 were given. The CP10 harness session is resumed on `opus` at `xhigh`
effort, with the harness-stage label, and finishes the judge there. Three
conditions: the session re-reads the change prompt, D77 and D78 and its own
measurements before it continues; it stops once more after the prototype
and both tools are brought up to the rulings, and writes the judge only on
the lead's go; and the review of pass 9 is run on Fable as the routing has
it, when the allowance is back.
Alternatives considered: waiting for the allowance (the owner chose not
to). Running the review on another model (rejected: the reviewer is the
independent reader, and with the harness on Opus it is the only stage left
on a different model).
Consequence: for this pass the judge, the tests and the build are all
written on one model. The rules were designed, measured and ruled on Fable
before the switch; what is left is their execution against proofs that a
program checks.

## D80 — 2026-11-26 — Rulings on the CP10 prototype (amends D78; extends D79)
Decision: (1) The rests-on list names a column only when its fall is at
least 0.05 and at least the half-width of the headline skill's interval;
where the headline has no interval, the harness proposes the second term
from measurement. A fall that the headline's own uncertainty could give is
not a statement about the result. (2) A battery pair that plants a carrier
of the target holds when it ends in one of the four outcomes of D78, or
when the headline with the carrier is within that run's floor of the
headline without it: a carrier that moves the headline by less than its
uncertainty is not a confident wrong number. Where the headline skill has
no value (D36), the pair is a stated limit citing D36. Of a carrier made
of two columns, either one named counts as named. (3) Section 11 states
the few-band case with the measured numbers: six to eight bands are warned
about; five sometimes; four or fewer read like an honest column of that
strength, are not warned about, and are named when they lift the headline
above the floor; under time order no band count is warned about or
excluded. (4) Copies are recognised on the matrix the models receive, as
the harness pinned it, and up to sign: a negated copy is a copy. (5) The
warning code `shuffled_score_not_computed` is approved. (6) From this
decision the lead's rulings are also made on Opus, at the owner's choice;
the review of pass 9 stays on Fable (D79).
Alternatives considered: for (1), the floor of 0.05 alone (rejected on
measurement: the list named columns on 13 to 20 of 30 seeds of the gate's
noise tables, falls of the size the headline's sampling noise gives). For
(2), the D78 carrier kinds kept as stated limits (rejected: all 11 pairs
that end in none of the four lifted the headline by at most 0.037, so the
claim can be checked instead of excused).
Consequence: the stated-limit list goes back to the four kinds of D76,
plus D36 where skill has no value.

## D81 — 2026-11-26 — How the carrier claim reads its floor (clarifies D80 point 2)
Decision: A battery pair that plants a carrier of the target, and ends in
none of the four outcomes of D78, holds when its headline skill is no more
than the unperturbed run's rests-on floor above the unperturbed run's
headline skill. The yardstick is the unperturbed run's, so a carrier cannot
excuse itself by widening the interval it is measured against; and only a
rise fails, since the claim exists to catch an inflated headline. A carrier
that lowers the headline is a sixth outcome, counted and printed apart, and
spec section 11 says when it happens: a coded copy whose levels the
holdout rows mostly do not hold, under time order, can pull the headline
far below the run without it, which then reads as no evidence of
predictability (the fixed limitation 2). The new key
`headline.rests_on_floor` and the summary line that prints it are approved.
Alternatives considered: the pair's own floor, or a fall in either
direction (rejected on measurement: six pairs failed, five of them coded
text copies under time order that lowered the headline by 1 to 900 skill
points, and the sixth a walk whose one extreme holdout cell makes both
headlines about -1e9, which `extreme_value` and `largest_row_dominates`
already name).
Consequence: with this reading every carrier pair holds and the
stated-limit list is the four kinds of D76 plus D36.
