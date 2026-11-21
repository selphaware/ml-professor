# mlprofessor

`mlprofessor` measures how predictable one target column of one table is. You
give it a table and name a target column. It holds out a quarter of the rows,
fits a few standard model families on the rest, scores the chosen one on the
held-out rows, and reports that score against a simple baseline, with an
interval, a p-value, and a complete account of every row and column it used or
set aside.

Phase 1 covers one table, supervised learning, binary classification and
regression, and a random or time-ordered validation scheme. It is a measuring
tool, not a model-building tool: it does not produce a model you should deploy.

## Setup and how to run it

The package is not installed. It is never installed into the interpreter, and
there is no `pip install` step. Run it from the repo root, with this interpreter:

```
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe
```

Dependencies are pinned in `requirements.txt` and already present in
`mlprof_env`. Nothing needs the network at run time.

Run every command from `E:\CODE\ml-professor`, so that `mlprofessor` is found
in the current folder.

The examples use `examples/customers.csv`, a synthetic table kept in the repo:
400 customers, one row per sign-up day. Its columns are `customer_id`, `signup_date`
(ISO dates, year first, so it can serve as the time column), `plan`, `region`,
`monthly_spend` (six blank cells, kept and handled inside each fold), `support_calls`,
`age`, and the target `churned` (`yes` or `no`, with three blank cells, which
are dropped and counted). To use your own table, replace the path with a CSV
path; a relative path is read from the repo root.

## Command line

```
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -m mlprofessor profile --data examples/customers.csv --target churned --no-time-order --out runs/customers
```

`signup_date` holds dates, and a table with a date column and nothing said about
time order stops with `time_order_undeclared` (exit code `3`). The tool does not
decide for you whether row order matters, so there are two ways on: name the time
column with `--time-col`, or say the order does not matter with `--no-time-order`
(the Python equivalent is `time_col=False`). Each row here is one customer, not a
step of a series in time, so the random-scheme commands pass `--no-time-order`.
The reason for the stop is a sentence long: a table with dates in it may have a
headline inflated by time order, and the tool prints that headline only once the
caller has said the reason does not apply.

Options (`python -m mlprofessor profile` takes these):

| Option | Meaning | Default |
|---|---|---|
| `--data` | the table, as a CSV file | required |
| `--target` | the target column | required |
| `--out` | folder for `profile.json` and `predictions.csv` (created if missing) | required |
| `--task` | `auto`, `classification` or `regression` | `auto` |
| `--time-col` | the time column; enables the time-ordered scheme | none (random scheme) |
| `--no-time-order` | states that the order of the rows in time does not matter, so the split is random even when a column holds dates; cannot be given with `--time-col` | not given |
| `--gap` | whole timestamps left out between training and holdout (needs `--time-col`) | 0 |
| `--profile` | `full` or `fast` (see below) | `full` |
| `--seed` | the random seed, a whole number from 0 to 4,294,967,295; the same seed gives the same output | 0 |
| `--n-jobs` | threads; `-1` uses every core | -1 |
| `--duplicates` | `error` stops the run when too many holdout rows have a copy among the training rows; `keep` goes ahead with a warning; `group` keeps each set of exact copies on one side of the split and of every fold (see "Copies of holdout rows") | `error` |
| `--group-col` | the entity column: the rows of one entity stay together (see "Entities") | none |
| `--no-entities` | states that no row belongs to one entity; cannot be given with `--group-col` | not given |

The command prints a short plain-text summary to standard output, in this order:

- the task, and the validation scheme;
- the rows loaded, the rows used for training, and the rows held out;
- the headline: the metric, the score, the chosen model, the baseline and its
  score, and the skill;
- the interval on the score and the interval on the skill, each with its method
  in brackets; where there is no interval, the reason is printed in its place;
- the p-value and the number of shuffles, or, when there is none, the reason;
- one line per warning, with its code and, when it has one, its column;
- the output folder.

Numbers in the summary are printed to at least four significant digits;
`profile.json` holds them as computed.

Exit codes: `0` finished and both files written; `2` the command line could not
be parsed; `3` a named error, with one line `error: <code>: <message>` on
standard error and no files written. The copies error has the code
`holdout_copies_in_training`; the two stops for a missing declaration have the
codes `time_order_undeclared` and `entities_undeclared`.

Time-ordered example, with `signup_date` as the time column and a one-timestamp
gap:

```
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -m mlprofessor profile --data examples/customers.csv --target churned --time-col signup_date --gap 1 --out runs/customers_time
```

Here the holdout is the latest sign-up days, and the summary gives no p-value,
with the reason: shuffling the holdout assumes independent rows, and rows in time
order are not.

Quick check, with the fast profile:

```
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -m mlprofessor profile --data examples/customers.csv --target churned --no-time-order --profile fast --out runs/customers_fast
```

Without `--time-col` the tool never guesses a time column. With neither
`--time-col` nor `--no-time-order`, a table with a date column stops with
`time_order_undeclared`; with `--no-time-order` it excludes the date column, says
so in a `date_column_undeclared` warning, and the split is random.

A panel is a table with repeated timestamps whose target is autocorrelated, such
as many series recorded over the same days. Such a table with neither
`--group-col` nor `--no-entities` stops with `entities_undeclared`. Pass
`--group-col` to name the entity column, or `--no-entities` to say no row belongs
to one entity. The reason is the same: without an entity column, rows of one
series can land on both sides of the split and inflate the headline.

## Python

```python
from mlprofessor import profile

result = profile(
    "examples/customers.csv",   # a path, or a pandas DataFrame
    "churned",
    task="auto",          # "auto", "classification" or "regression"
    time_col=False,       # a time column's name; False: time order does not matter
    gap=0,                # whole distinct timestamps between training and holdout
    profile="full",       # "full" or "fast"
    seed=0,               # a whole number from 0 to 4,294,967,295
    out_dir="runs/customers",   # optional: writes profile.json and predictions.csv
    n_jobs=-1,
    group_col=False,      # an entity column's name; False: no row belongs to one entity
    duplicates="error",   # "error", "keep" or "group": see "Copies of holdout rows"
)

print(result.headline)     # dict: metric, score, baseline, skill, interval, p-value, ...
print(result.rows)         # DataFrame: item, count
print(result.columns)      # one row per input column
print(result.models)       # one row per model configuration
print(result.warnings)     # DataFrame: code, column, message
print(result.predictions)  # DataFrame: row, y_true, y_pred (holdout rows only)
doc = result.to_dict()     # the whole of profile.json as a dict
```

A DataFrame passed as `data` is never modified. Run the snippet from the repo
root so that the relative path resolves. Errors are subclasses of
`mlprofessor.errors.ProfileError`; each has a `code` attribute.

## What the output contains

With an output folder, two files are written. Nothing is written if an error
is raised.

**`profile.json`** has these parts:

| Part | Holds |
|---|---|
| `meta` | tool, schema version, profile, seed, `n_jobs`, library versions, `timing` |
| `input` | source, kind, target, time column, task requested, gap, row and column counts, duplicate rows, the `duplicates` argument, the entity column |
| `rows` | seven counts: loaded, four reasons for dropping a row, training, holdout |
| `columns` | one record per input column: role, kind, reason, leak rule, and counts |
| `task` | type, whether it was decided by rule or declared, distinct values, classes, positive class |
| `validation` | scheme, holdout share, folds, gap, time form, last training and first holdout timestamp, caveat, holdout rows with a copy among the training rows and their share (random scheme); with an entity column, the entity column and the entities on each side |
| `headline` | metric, score, chosen model, baseline and its score, skill (null when it has no value), interval on the score and on the skill, p-value, reasons |
| `baselines` | each baseline scored on the holdout, and which one was used |
| `cv` | metric, folds scored, configurations, the chosen one, its mean and spread |
| `models` | one record per configuration: parameters, status, error, fold scores, mean, spread, chosen |
| `warnings` | records of code, column and message |
| `limitations` | the nine fixed sentences described below |

**`predictions.csv`** has a header `row,y_true,y_pred` and one line per holdout
row. `row` is the zero-based position in the source, counted before any row is
dropped. For classification `y_pred` is the predicted probability of the
positive class; for regression it is the predicted value.

The headline is computed only on holdout rows, which no fitted step has seen.
That holds for the rows themselves, but a holdout row can have an exact copy
among the training rows, and a model can recall such a row. Under the random
scheme, when more than 1% of the holdout rows, and at least 10 of them, have a
copy among the training rows, the run stops with `holdout_copies_in_training`
before any model is fitted, unless the caller passes `duplicates="keep"`
(`--duplicates keep`); then it goes ahead with a `holdout_copies_kept` warning.
The count is always reported (see "Copies of holdout rows" below). Every number
in `headline.score` can be recomputed from `predictions.csv`.

### When skill has no value

Skill is a regression quantity that can have no value. It is `null` in two cases,
and each has its own warning.

- **The baseline predicts the holdout exactly.** The headline baseline's squared
  error on the holdout is then zero, so skill would divide by zero. That happens
  when the target is flat across the holdout under time order, or when the
  training mean is the value of every holdout row under the random scheme.
  `headline.skill`, `skill_low` and `skill_high` are `null`, not an error, an
  infinity or a zero. The warning `skill_undefined` says why; it appears exactly
  once, and only in this case. The score, its interval and every baseline,
  including the zero, are reported as usual. There is no p-value, and
  `p_value_reason` says why.
- **Skill has a value, but no interval.** The baseline is exact on most holdout
  rows and not on all, so skill can be computed in fewer than half of the
  resamples. `headline.skill` is a number, `skill_low` and `skill_high` are
  `null`, and the interval on the score stands. The warning `no_skill_interval`
  says so.

## Validation schemes

Pass `--time-col` (or `time_col=`) whenever row order carries meaning: rows
are recorded over time, or the target can drift, trend or be autocorrelated.
The tool never guesses a time column.

- **Random scheme** (no `time_col`). A quarter of the rows are drawn at random
  as the holdout, stratified by class for classification. Five folds are shuffled
  from the seed. Use it for independent rows. Its result is optimistic if rows
  are ordered in time or one entity appears on many rows; `validation.caveat`
  says so. When one entity appears on many rows, name its column (see
  "Entities"); the scheme then becomes `grouped`.
- **Time-ordered scheme** (`time_col` given). The holdout is the latest whole
  timestamps that together hold at least a quarter of the rows; rows sharing a
  timestamp are never split. `--gap N` removes the N timestamps just before the
  holdout from training, to keep near-term information out. Training is cut into
  six contiguous blocks, and each fold validates on the next block. The time
  column must be a datetime column, a numeric column, or text in ISO 8601 with
  the year first (`YYYY-MM-DD`, optionally with a time). The fraction of a
  second may have one to nine digits; ten or more raise `time_column_invalid`.
  A whole-number time column with values above 2**53, such as nanoseconds since
  1970, is compared exactly, so every distinct value is its own timestamp.
  Anything else, such as
  `25/01/2024`, raises `time_column_invalid`: the tool never guesses between
  day-first and month-first, so convert the column first.

Under the time-ordered scheme the regression baseline also includes
`previous_value` when timestamps are unique, so a trending target is not scored
only against the training mean.

### Entities

An entity is the unit that a group of rows belongs to: a customer, a patient, a
device, or one series of a panel. Name its column with `--group-col` (or
`group_col=`) when one entity appears on more than one row. What happens next
depends on whether there is a time column.

- **With no time column**, the rows of one entity stay together: all of an
  entity's rows are on one side of the holdout, and all are in one fold. The
  scheme is `grouped`. The interval on the score is drawn by resampling whole
  entities, and there is no p-value, because shuffling rows assumes they are
  independent. With fewer than 30 entities in the holdout there is no interval,
  and the reason is printed. Under the fast profile the cap keeps whole entities.
- **With a time column**, the split is by time as before, and each entity has its
  own previous value as a baseline for regression, in the headline and in the
  leak screen. So a series is scored against its own past, not against the
  training mean of every series.

Pass `--no-entities` (or `group_col=False`) to say that no row belongs to one
entity. The stop for an undeclared panel is described under "Command line".

### Copies of holdout rows

A holdout row has a copy when some training row has the same value in every
feature column and the same target. Columns left out of the features (an
identifier, a date, a suspected leak) play no part, so two rows that differ only
in those are copies. Under the random scheme the tool counts the holdout rows
that have a copy, and reports the count in `validation.n_holdout_copies` and
its share of the holdout in `validation.holdout_copy_share`. The count is always
reported, even below the rule.

The rule, which applies only under the random scheme:

- when more than 1% of the holdout rows, and at least 10 of them, have a copy,
  the run stops before any model is fitted, with `holdout_copies_in_training`
  (exit code `3`, and no files written);
- with `--duplicates keep` (or `duplicates="keep"`) the run goes ahead anyway,
  and the output carries one `holdout_copies_kept` warning saying the headline
  may be inflated;
- below the rule, the run goes ahead with no warning.

The default is `--duplicates error`. The easy remedy for a table whose copies are
genuine repeats is to remove the repeated rows. A table of a few categorical
columns, such as a survey with a handful of answers, will usually have many
copies, and so will need `keep`: the warning is then the honest record that the
headline may be inflated.

`--duplicates group` (`duplicates="group"`) keeps each set of exact copies on one
side of the split and in one fold, so a copy is never in training while its twin
is held out. It applies with no time column.

**What is detected, and what is not.** A copy is detected only when a row agrees
with another in every feature column and in the target. A copy that differs in
any feature column (an added column, a batch number, a small change to a value)
is not detected in this version, and the run treats those rows as different rows.
When you know which rows belong together, name their column with `group_col`
(`--group-col`); that is the control.

Under the time-ordered scheme the rule does not apply. Both fields are `null`,
and `duplicates` has no effect.

## Profiles

- `full` (default): 18 model configurations across three families (linear,
  random forest, gradient boosting), five-fold cross-validation inside the
  training rows, 1,000 resamples for the interval and p-value.
- `fast`: two configurations (a linear model and a single decision tree), no
  grid, 200 resamples, and a cap of 2,000 usable rows. Under the random scheme
  the 2,000 rows kept are drawn at random from the seed; under the time-ordered
  scheme they are the most recent whole timestamps whose rows together number
  at most 2,000. The rows left out are counted in `rows`. Use it to look at a
  table quickly. Under the time-ordered scheme with a wide panel, the cap can
  leave too few timestamps for the split: that raises
  `fast_profile_too_few_timestamps` when `full` would get past that point, and
  otherwise the error `full` would raise. Use `full` then.

  Under either scheme, when the cap leaves fewer than 100 rows, or fewer than
  10 of a class, the error is `fast_profile_too_few_rows`, but only when `full`
  would run past that point; otherwise the error `full` would raise is raised.
  Use `full` then.

## What it measures, and what it does not

It measures how well these model families did on this table, under this
validation scheme, on rows they had not seen. It reports that against a baseline:
for classification, chance (AUC 0.5); for regression, the training mean, or the
previous value under time order. Skill is `2 × AUC − 1` for classification and
`1 − MSE ÷ baseline MSE` for regression; for regression it has no value when the
baseline's error is zero (see above).

It does not:

- prove that a target cannot be predicted when skill is low; the families may
  simply be wrong for the problem;
- prove cause when skill is high;
- detect look-ahead inside your own features (a feature built from future
  information is invisible to it);
- compare classification skill with regression skill, which are on different
  scales;
- guarantee its intervals. They are approximate: they tend to be too narrow
  when the holdout is short and, under the time-ordered scheme, when errors are
  correlated in time;
- read text identifiers faithfully when loading a CSV: pandas may drop leading
  zeros. Pass a DataFrame if the types matter;
- pick the time column or the entity column for you, or guess day-first versus
  month-first dates;
- detect a copy of a row that differs in any feature column. See "Copies of holdout
  rows": the control for rows that belong together is `group_col`.

Negative skill means the models did worse than the baseline on the holdout. Read
it as no evidence of predictability, not as a finding.

These sentences are written to `limitations` on every run, in this order:

1. Predictability here means how well these model families did on this data under this validation scheme.
2. Low skill is not proof that the target cannot be predicted.
3. High skill is not proof of cause.
4. A random split is optimistic when rows are ordered in time or one entity appears on many rows.
5. Look-ahead inside the caller's own features cannot be detected.
6. Classification skill and regression skill are on different scales and are not comparable.
7. A CSV file is read with pandas' types, so text identifiers can lose leading zeros; pass a DataFrame to control the types.
8. Intervals are approximate: they tend to be too narrow when the holdout is short and, under time order, when errors are serially correlated.
9. Negative skill means the models did worse than the baseline on the holdout; read it as no evidence of predictability.

## Rows and columns

Every row is counted. A row is dropped only if its target is missing (or, for a
numeric target, infinite), or, under the time-ordered scheme, its time value is
missing. Each reason is counted in `rows`, and `rows.loaded` equals the four
dropped counts plus training plus holdout. Rows are not dropped for missing
feature values; those are handled inside each fold, after the split.

A profile stops with a named error rather than guessing when the table is too
small (fewer than 100 usable rows, or fewer than 10 of a class), when the task
is ambiguous, or when two feature columns would give the preprocessed matrix the
same column name. The last is `feature_name_collision`: for example a numeric
column named `colour=red` beside a categorical column `colour` with a level
`red`. The run stops before any fit, names the columns involved, and writes
nothing. A column whose numbers are written with thousands separators, such
as `1,200`, is excluded rather than read as a number, and the column record says
why.

A column of objects that hold only numbers is read as the numbers it stands
for: `Decimal` values, booleans with missing cells, or whole numbers with
missing cells. Its `dtype` still says `object`. An object column that mixes a
boolean with a number is not guessed at; it is excluded as `unsupported_type`.
Every column excluded as an unsupported type gets an `unsupported_type` warning
that names it. Whole numbers are compared exactly, so the distinct values of a
whole-number column above 2**53 stay distinct. A column of whole numbers too
large for 64 bits is excluded as a feature, with a warning; as the target it
stops the run with `task_unsupported`, and as the time column with
`time_column_invalid`. The full list of error codes is in `spec.md` section 2.4.

A text target is read without the white space round its values, so `yes ` and
`yes` are the same class; `classes`, `positive_class` and `y_true` show the value
without it. A target cell that holds only white space is a missing target: its
row is dropped and counted in `dropped_missing_target`.

Three or more classes are not supported. Under `task="auto"` a target that is not
numeric with three or more distinct values stops with `task_ambiguous`, and the
message says that three or more classes are not supported. Phase 1 covers binary
classification and regression only.

An object column (one whose cells are Python objects, such as `Decimal` values) is
read once, over every loaded row. So an odd cell in a row that is later dropped
for an empty cell in another column still stops the run with its named error, as
it would in a row that is kept. The infinite cells of the target and of the time
column are counted in their column records, in `n_infinite`.

A numeric feature value that is infinite, or finite but larger than 3.4028234663852886e+38
in magnitude (too large for the tree models), is read as missing. It is counted in
`n_infinite` for its column, and an `infinite_values` warning names the column.

## Numbers that are not finite

No number that is not finite is ever written, and none is written as `null`.
A `null` means "does not apply", or "cannot be computed" with the reason beside
it, and nothing else. A regression target with a value too large for squared
error (for example one cell of 1.7976931348623157e308) stops the run with
`non_finite_score`, naming the baseline, before any model is fitted. A
configuration whose prediction or fold score is not finite fails, as one that
raises does. If the chosen configuration's holdout score is not finite, the run
stops with `non_finite_score`, with no fallback to another configuration.

## Files and reproducibility

- `spec.md` is the contract: the methodology, the output schema and the errors.
- `docs/DECISIONS.md` records the reasons behind each choice.
- The same arguments and seed give the same `profile.json` and `predictions.csv`,
  apart from `meta.timing`, and apart from `meta.n_jobs` when `--n-jobs` differs.
- If scores move after a library upgrade, compare `meta.versions` in
  `profile.json` first.
