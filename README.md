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
are dropped and counted). Each customer appears on one row, so no entity repeats
and the examples say `--no-entities`. To use your own table, replace the path with
a CSV path; a relative path is read from the repo root.

## Two statements every call must make

Every call says two things, and the tool assumes neither:

- **Whether the rows are ordered in time.** `--time-col COLUMN` (`time_col="COLUMN"`)
  says they are, and names the time column. `--no-time-order` (`time_col=False`)
  says they are not.
- **Whether rows belong together.** `--group-col COLUMN` (`group_col="COLUMN"`)
  says one entity (a customer, a device, one series of a panel) can appear on
  more than one row, and names its column. `--no-entities` (`group_col=False`)
  says no row belongs to one entity.

The default of each is "not said". A call that leaves the time order unsaid stops
with `time_order_undeclared`; one that leaves the entities unsaid stops with
`entities_undeclared`; when both are unsaid, the message asks for both. This
happens whatever the table holds: with or without date columns, with or without
repeated rows, with or without a series of repeated timestamps. On the command
line both stops exit with code `3`, not `2`. A table that fails an earlier check
(an invalid column, too few rows) reports that error first.

A run never goes ahead or stops because of a date form or an autocorrelation.
Those are warnings, read beside the statement the caller has made.

The four combinations and what each gives:

| Time order (`--time-col`) | Entities (`--group-col`) | What the run does |
|---|---|---|
| a column | a column | Time-ordered split. Each entity is scored for regression against its own previous value, in the headline and in the leak screen. |
| a column | `--no-entities` | Time-ordered split, with rows sharing a timestamp never split. For regression the baselines are the training mean and the previous value, where the previous value is the mean target of the timestamp before (see "Baselines under time order"). A repeated-timestamp panel whose per-timestamp mean persists may get the `autocorrelated_panel_target` warning (see Limits). |
| `--no-time-order` | a column | Grouped scheme: all of an entity's rows are on one side of the holdout and in one fold. No p-value. |
| `--no-time-order` | `--no-entities` | Random scheme: rows are taken as independent. A p-value is reported, and `validation.caveat` says where the scheme is optimistic. |

## Command line

Random scheme, with the time order and the entities both stated:

```
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -m mlprofessor profile --data examples/customers.csv --target churned --no-time-order --no-entities --out runs/customers
```

The date column `signup_date` is not a time order here: with `--no-time-order` it
is excluded and named by a `date_column_undeclared` warning, and the split is
random. The same call without `--no-time-order` and without `--time-col` stops
with `time_order_undeclared`, and without `--no-entities` and without `--group-col`
it stops with `entities_undeclared`, whichever columns the table has.

Options (`python -m mlprofessor profile` takes these):

| Option | Meaning | Default |
|---|---|---|
| `--data` | the table, as a CSV file | required |
| `--target` | the target column | required |
| `--out` | folder for `profile.json` and `predictions.csv` (created if missing) | required |
| `--task` | `auto`, `classification` or `regression` | `auto` |
| `--time-col` | the time column, which orders the rows and enables the time-ordered scheme; cannot be given with `--no-time-order` | not said: the run stops |
| `--no-time-order` | states that the rows are not ordered in time, so the split is random even when a column holds dates; cannot be given with `--time-col` | not said: the run stops |
| `--gap` | whole timestamps left out between training and holdout (needs `--time-col`) | 0 |
| `--profile` | `full` or `fast` (see below) | `full` |
| `--seed` | the random seed, a whole number from 0 to 4,294,967,295; the same seed gives the same output | 0 |
| `--n-jobs` | threads; `-1` uses every core | -1 |
| `--duplicates` | `error` stops the run when too many holdout rows have a copy among the training rows; `keep` goes ahead with a warning; `group` keeps together the rows that agree in every feature column (see "Copies of holdout rows") | `error` |
| `--group-col` | the entity column: the rows of one entity stay together (see "Entities"); cannot be given with `--no-entities` | not said: the run stops |
| `--no-entities` | states that no row belongs to one entity; cannot be given with `--group-col` | not said: the run stops |

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
`holdout_copies_in_training`; the two stops for a missing statement have the
codes `time_order_undeclared` and `entities_undeclared`.

Time-ordered example, with `signup_date` as the time column, a one-timestamp
gap, and the entities stated as none:

```
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -m mlprofessor profile --data examples/customers.csv --target churned --time-col signup_date --gap 1 --no-entities --out runs/customers_time
```

Here the holdout is the latest sign-up days, and the summary gives no p-value,
with the reason: shuffling the holdout assumes independent rows, and rows in time
order are not.

Quick check, with the fast profile:

```
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -m mlprofessor profile --data examples/customers.csv --target churned --no-time-order --no-entities --profile fast --out runs/customers_fast
```

## Python

```python
from mlprofessor import profile

result = profile(
    "examples/customers.csv",   # a path, or a pandas DataFrame
    "churned",
    task="auto",          # "auto", "classification" or "regression"
    time_col=False,       # required: a time column's name, or False (rows are not ordered in time)
    gap=0,                # whole distinct timestamps between training and holdout
    profile="full",       # "full" or "fast"
    seed=0,               # a whole number from 0 to 4,294,967,295
    out_dir="runs/customers",   # optional: writes profile.json and predictions.csv
    n_jobs=-1,
    group_col=False,      # required: an entity column's name, or False (no row belongs to one entity)
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
`mlprofessor.errors.ProfileError`; each has a `code` attribute. A call that
leaves `time_col` or `group_col` out stops with the same codes as on the command
line.

## What the output contains

With an output folder, two files are written. Nothing is written if an error
is raised.

**`profile.json`** has these parts:

| Part | Holds |
|---|---|
| `meta` | tool, schema version, profile, seed, `n_jobs`, library versions, `timing` |
| `input` | source, kind, target, the time column and the entity column as passed (a name or `false`), task requested, gap, row and column counts, duplicate rows, the `duplicates` argument |
| `rows` | seven counts: loaded, four reasons for dropping a row, training, holdout |
| `columns` | one record per input column: role, kind, reason, leak rule, counts, and for each feature `shuffled_score`, `shuffled_skill` and `skill_fall` (see "What the headline rests on") |
| `task` | type, whether it was decided by rule or declared, distinct values, classes, positive class |
| `validation` | scheme, holdout share, folds, gap, time form, last training and first holdout timestamp, caveat, holdout rows with a copy among the training rows and their share (random scheme); with an entity column, the entity column and the entities on each side; under `duplicates="group"`, `n_rows_same_features` |
| `headline` | metric, score, chosen model, baseline and its score, skill (null when it has no value), interval on the score and on the skill, p-value, reasons; for a regression, `largest_row_share` and `largest_row` for the model and the baseline; `rests_on` and `rests_on_floor` (see "What the headline rests on") |
| `baselines` | each baseline scored on the holdout, and which one was used; a regression under time order has both `training_mean` and `previous_value` |
| `cv` | metric, folds scored, configurations, the chosen one, its mean and spread; for a regression, the largest-row keys over the selection folds |
| `models` | one record per configuration: parameters, status, error, fold scores, mean, spread, chosen; for a regression, `largest_row_share` and `largest_row` over the selection folds |
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

Say `--time-col` (or `time_col="COLUMN"`) whenever row order carries meaning: rows
are recorded over time, or the target can drift, trend or be autocorrelated.
The tool never guesses a time column, and it never assumes that the rows are
ordered in time.

- **Random scheme** (`--no-time-order`, or `time_col=False`). A quarter of the rows are drawn at random
  as the holdout, stratified by class for classification. Five folds are shuffled
  from the seed. Use it for independent rows. Its result is optimistic if rows
  are ordered in time or one entity appears on many rows; `validation.caveat`
  says so. When one entity appears on many rows, name its column (see
  "Entities"); the scheme then becomes `grouped`.
- **Time-ordered scheme** (`--time-col` given). The holdout is the latest whole
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

### Baselines under time order

Under the time-ordered scheme a regression is always scored against two baselines,
the training mean and the previous value, so a trending target is not scored only
against the training mean. Both are reported in `baselines`. The one with the lower
error on the holdout is the headline baseline, and a tie goes to the training mean.

The previous value of a row is the mean target of the timestamp before it, that is,
the timestamp `gap + 1` distinct timestamps earlier. With one row for each timestamp
that is simply the row before. When a timestamp repeats, all of its rows share that
one previous value, so a repeated row does not change the baseline. This holds with
or without repeated timestamps.

With an entity column the previous value is each entity's own (see "Entities").
With `--no-entities` and repeated timestamps, a panel whose rows share timestamps
may still get the `autocorrelated_panel_target` warning. That warning names the
baseline the run used. If the rows are several series, such as one per customer or
per device, declare the column that names each series with `--group-col` (or
`group_col="COLUMN"`), so that each series is scored against its own previous value
and no series is split across the holdout.

### Statements a caller should read

Three parts of the output say when one row or one value decides the result.

- **`largest_row_share` and `largest_row`.** For a regression, these report the
  share of the squared error that the single largest row carries, and that row, for
  the model and for the baseline. They appear in `headline` (on the holdout), in
  `cv` (over the selection folds, for the chosen configuration and for the headline
  baseline), and for every configuration in `models`. The share is taken of the
  squared errors, not of the absolute errors. When one row carries more than half of
  the squared error, the warning `largest_row_dominates` is raised, once for each
  of the statements above one half. The score, the skill and the choice of model keep
  that row. The statement tells the caller that the result rests on one row.
- **`extreme_value`.** This warning names a value that is far outside all the other
  values of its column: a numeric feature, or a regression target, whose gap to the next
  value inward is more than 30 times the spread of the ten distinct values after it.
  Typical cases are `-999` or `999999` standing for a missing value in a column that
  otherwise runs from about 1,000 to 100,000. The value is used as it is, and the
  score, the skill and the choice keep it. The remedy is to write an empty cell where
  the value means "missing": empty cells are handled inside each fold, and the rows
  are then counted in the usual way.

### What the headline rests on

Every feature is shuffled once on the holdout, to ask how much the headline
depends on it. On each run that finishes, each feature's record in `columns`
holds three numbers:

- **`shuffled_score`**: the chosen model's score on the holdout with that
  feature's holdout values shuffled among the holdout rows. It is the mean of five
  shuffles, drawn from the seed. Nothing is refitted: the same fitted model scores
  the shuffled rows.
- **`shuffled_skill`**: the skill of that score, against the same headline baseline.
- **`skill_fall`**: `headline.skill` minus `shuffled_skill`, how far the headline
  falls when the feature's holdout values are shuffled. It can be negative, and it is
  `null` where `headline.skill` is.

`headline.rests_on` lists the features whose `skill_fall` is at least
`headline.rests_on_floor`, at most five of them, largest fall first. The floor is the
larger of 0.05 and the half-width of the skill interval (or of a bootstrap of the holdout
rows, where the skill has no interval). The console summary prints this list. An empty
list means that no feature's shuffle lowers the headline by the floor.

Copies of one another share one fall. Two columns are copies when, as the models
receive them, they are the same column within 1e-9, or one is the other's negation.
So a column that is a straight-line rescaling of another (`2 × f + 3`) is a copy of it,
and a text column whose levels are only renamed is a copy of the column. Copies are
shuffled together, so they show the same three numbers. Two columns that carry the same
information without being copies each show less than they share: shuffling one leaves
the other to carry the signal, so each can fall below the floor while together they
carry the headline.

A model that cannot predict a shuffled matrix leaves that feature's three keys `null`,
with a `shuffled_score_not_computed` warning naming it, and the run finishes.

**`near_perfect_headline`.** When `headline.skill` is 0.98 or more, the run raises
one `near_perfect_headline` warning. It changes no number and excludes no column. It
names each column of `headline.rests_on` and asks whether that column is known before
the outcome. An easy table carries it too, so a near-perfect headline on an easy table
should prompt the caller to check each named column: if one is recorded after the
outcome, or is a copy of it, the headline is not a prediction. When the list is empty,
the warning says that no column's fall reaches the floor.

**`holdout_far_above_quarter`.** Under time order, when the holdout holds half of the
rows or more, this warning is raised once. Whole timestamps made the holdout far more
than a quarter of the rows, so the score rests on a short, late stretch of the table.
Its message gives the holdout rows and the rows split. The random and grouped schemes
do not raise it.

### When the folds cannot choose: `selection_not_made`

The choice between configurations is made on the five folds inside the training rows.
With an entity column, the folds are whole entities, and one entity can hold nearly
all the training rows. When some fold then has fewer than 50 training rows and fewer
training rows than validation rows, the folds cannot make a choice. The run still
scores and reports every fold, but makes no choice: the configurations are taken in
their own order, the first one is the chosen one, and the warning `selection_not_made`
says so. The fallback for a configuration that fails after the choice follows the same
order. The headline is then the score of a configuration chosen by order rather than
by the folds, so it rests on that choice and not on the evidence.

### What the leak screen does

Before the models are fitted, each feature is read on its own against the target, on
the training rows only. A column that predicts the target almost perfectly is excluded
as a suspected leak. The reading does not depend on the column's scale: a copy of the
target multiplied by a large or small factor is excluded in the same way as an exact
copy. Under time order a copy of a trending target is excluded too, while the lag of
that target, which is a legitimate feature, stays.

The `strong_single_column` warning is the weaker tier. It names a column that predicts
the target strongly but is kept as a feature, so that the caller can check it is known
before the outcome. A column gets it in either of two ways:

- its single-column skill, with every row scored, is at least 0.90 (the exclusion line
  is 0.98: a column at or above it is excluded instead); or
- for a numeric column of a regression whose skill is below 0.90, the column follows
  the target's order: between at most 10% of neighbouring training rows, the target
  moves the other way, and the target does not follow its own previous value as
  closely. A steadily rising target, or a running total, does not raise it this way.

The warning excludes nothing. It asks the caller to read that column first.

### Dates written as whole numbers

A column in which every value is a whole number that reads as a calendar date
written `YYYYMMDD`, with years from 1900 to 2100 and at least two distinct values,
is treated as a date column. It is excluded with the reason `datetime`. Under
`--no-time-order` (`time_col=False`) that exclusion is named by a
`date_column_undeclared` warning, and the split is random. The same column can be
the time column: name it with `--time-col`, and the rows are ordered by its
numbers. The rule covers a whole-number column, a float column whose only gaps
are empty cells, and text of eight digits.

Dates written this way are read only in the `YYYYMMDD` form. A day count, a `YYYYMM`
month, and a date with its hour written `YYYYMMDDHH` are not seen as dates; see Limits.

**A value that is no date.** A date column often holds a value that is not a date,
such as `99991231` for an open end or `0` for none. Such a value does not switch the
rule off. The tool judges the column by its distinct values, not by its cells, so a
value that is no date counts once, however many rows hold it. The column is still a
date column when at least two distinct values read as dates and the values that are
no date are few beside them. A column in which more than about one distinct value in
ten is no date is read as a number again, and is then no longer treated as a date.

As the time column, a value that is no date stops the run with `time_column_invalid`.
The message names the value and the rows that hold it. The tool never reads such a
value as a number and never drops it on its own. To go ahead, either empty those
cells, in which case those rows are dropped as having no time value and counted in
`rows`, or write the date.

### Entities

An entity is the unit that a group of rows belongs to: a customer, a patient, a
device, or one series of a panel. Name its column with `--group-col` (or
`group_col="COLUMN"`) when one entity appears on more than one row. What happens
next depends on whether there is a time column.

- **With no time column** (`--no-time-order`), the rows of one entity stay
  together: all of an entity's rows are on one side of the holdout, and all are in
  one fold. The scheme is `grouped`. The interval on the score is drawn by
  resampling whole entities, and there is no p-value, because shuffling rows
  assumes they are independent. With fewer than 30 entities in the holdout there
  is no interval, and the reason is printed. Under the fast profile the cap keeps
  whole entities.
- **With a time column**, the split is by time as before, and each entity has its
  own previous value as a baseline for regression, in the headline and in the
  leak screen. So a series is scored against its own past, not against the
  training mean of every series.

Pass `--no-entities` (or `group_col=False`) to say that no row belongs to one
entity. Leaving the entities unsaid stops the run with `entities_undeclared`.

**The grouped holdout holds a quarter of the rows, and no more than it must.**
Entities join the holdout while it stays under a quarter of the rows. The smallest
entity that would complete the quarter is taken as well, and every entity the
holdout can spare goes back to training. So the holdout holds at least a quarter,
and less than a quarter plus one entity. For classification this is done for the
entities of each class.

**Large entities.** An entity that holds a quarter or more of the rows it is drawn
among is named by a warning:

- `large_group_kept_in_training` means the smaller entities fill the quarter, so
  the entity is kept in training and is never held out. The headline does not test it.
- `large_group_is_holdout` means the smaller entities cannot fill a quarter, so this
  one entity is the holdout, and the headline rests on it alone.

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

The default is `--duplicates error`. Which answer to the stop fits depends on
what the repeated rows are:

- **`keep`** when the repeated rows are different cases that happen to look
  alike, as in a table of a few categorical columns, such as a survey with a
  handful of answers, where many people give the same answers. The warning is
  then the honest record that the headline may be inflated.
- **`group`** when the repeated rows are one case recorded more than once.
- **Remove the rows** when the repeats are errors, such as a record loaded twice
  by a bad export and carrying no information of its own.

`--duplicates group` (`duplicates="group"`) keeps together the rows that agree in
every feature column, whatever their target, so that no combination of feature
values is on both sides of the split or in two folds of the training rows. The
target plays no part in forming those sets: two rows that agree in every feature
but differ in their target are kept together too. The sets then follow the
grouped holdout rules above, including the large-entity warnings. The count
`validation.n_rows_same_features` reports the rows that agree with another in
every feature column, and the warning `same_features_grouped` says how many there
are. It applies with no time column. Under the time-ordered scheme the rule does
not apply.

**What the headline measures under `group`.** Under `group` no combination of
feature values is on both sides of the split, so the held-out rows are combinations
the training rows do not contain. The headline then measures whether a combination
of feature values the model has not seen can be predicted. It does not measure how
well the model predicts a row like one it has already seen. On a table of a few
categorical columns, where many rows share one combination, this can give negative
skill where `keep` finds signal. Negative skill there is not evidence that the
target is unpredictable; run the table with `keep` as well and read the two together.

**A column the leak screen removes can leave copies.** Under `group`, when the leak
screen excludes a column, the rows are then grouped on the features that are left.
If one held-out row then agrees with a training row in every feature that is left,
the run stops with `holdout_copies_in_training` and names the column. There is no
threshold for this stop: one such row is enough. Remove that column from the table
and run again.

Under the time-ordered scheme the copies rule does not apply. Both fields are
`null`, and `duplicates` has no effect.

**What is detected, and what is not.** A copy is detected only when a row agrees
with another in every feature column and in the target. A copy that differs in
any feature column (an added column, a batch number, a small change to a value)
is not detected in this version, and the run treats those rows as different rows.
When you know which rows belong together, name their column with `group_col`
(`--group-col`); that is the control.

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

  With an entity column and no time column (see "Entities"), the fast profile's cap
  keeps whole entities, and it always keeps an entity that holds a quarter or more
  of the usable rows. The smaller entities are taken in an order drawn from the seed.
  When an entity is larger than the cap of 2,000 rows, or the entities that must be
  kept do not fit under the cap together, the run stops with
  `fast_profile_group_too_large` and names the full profile. Use `full` then. If
  `full` would stop too, its own error is raised.

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

## Limits

These limits are stated plainly. Each one is something the tool does not check.

- **Day counts, `YYYYMM` and `YYYYMMDDHH` are not seen as dates.** A column of days
  counted from an origin, of months written `YYYYMM`, or of dates with the hour
  written `YYYYMMDDHH`, is read as a number. It is not excluded as a date and is not
  warned about as one. To order rows by such a column, name it with `--time-col`,
  where it is compared as the numbers it holds.
- **Period labels are not seen as dates.** A label such as a week or a quarter
  (`2024-Q1`, `2024W05`) is text the tool does not parse. It is not excluded as a
  date and is not warned about as one. As the time column it stops the run with
  `time_column_invalid`, because only ISO 8601 with the year first is read as a date.
  Convert such a column to dates first, for example the first day of each period.
- **A column of dates with more than about one distinct value in ten that is no date
  is read as a number again.** The rule for a date column counts distinct values, so
  a value that is no date, such as `99991231`, counts once however many rows hold
  it. When more than about one distinct value in ten is no date, the column is not
  treated as a date column, and its values are compared as the numbers they are.
- **Entities are questioned only by one warning.** With `group_col=False`, the only
  check on whether rows belong together is `autocorrelated_panel_target`. It is
  raised only for regression under time order, with repeated timestamps, when the
  per-timestamp mean persists from one timestamp to the next. It names the baseline
  the run used, and it tells the caller to declare `group_col` if the rows are
  several series. On a short panel it often does not fire: about half of panels of
  8 timestamps raise no warning. A panel that is not warned of may still have one
  entity's rows on both sides of the split. Name the entity column with `group_col`
  whenever one entity has more than one row.
- **A coded copy of a target with few bands is a weaker column, and is only warned about
  or named.** A regression target cut into six to eight bands, with the bands as a text
  or a numeric code, is raised with a `strong_single_column` warning. Four or fewer bands
  read like an honest column of that strength: they are not warned about, and they appear
  only in `headline.rests_on` when they lift the headline. Sixteen or more bands are
  excluded under the random scheme. Beside entities, a coarse copy can be excluded on
  some tables and only warned about on others.
- **Under time order a coded copy of a wandering target can be neither excluded nor
  warned about.** The latest rows hold band codes the earlier rows never did, so the
  screen sees no level it has learned. On such a target a coded copy of up to eight bands
  is mostly neither excluded nor warned about. It can still show in `headline.rests_on`, or
  lower the headline when the holdout's levels differ from the training rows'. Check the
  `headline` and `rests_on` before trusting the score.
- **A noisy copy of the target is not always excluded under time order.** Under time
  order a copy of the target with noise of more than about 1% of the target's spread
  is not detected as a leak, so the column can be kept as a feature. A copy with more
  than about 1% of its cells altered is named by a warning but not excluded. Read a
  `strong_single_column` warning on such a column before trusting the score.
- **A whole-number measure in a table sorted by it is taken for an identifier.** A
  column of distinct whole numbers in a table that is sorted by that column looks
  like an identifier, so it is excluded as one. This version has no way to override a
  column's role. Check the column's record in `columns`, and if it is a measure, do not
  read the result as having used it.
- **A sentinel that is not far outside its column is not named.** `extreme_value`
  names a value only when it is far outside all the other values of its column. A
  sentinel such as `-999` in a column that runs from 1,000 to 100,000 is not far
  enough out, and no warning names it. It is used as it is. Where a value means
  "missing", write an empty cell instead.
- **No check looks for a column whose levels are series of their own.** A feature
  that identifies a series under another name (a column holding one value per
  series, for instance) is not recognised as an entity column. A model can learn
  each series from it, and a series can be on both sides of the split without
  any warning. Name the entity column with `group_col` instead.
- **Small tables under `duplicates="group"` give no interval.** The interval needs
  at least 30 units in the holdout, and under `duplicates="group"` each unit is a
  set of rows that agree in every feature. A table with about 30 distinct
  combinations of feature values, or fewer, gives no interval, and so may a larger
  one whose holdout holds fewer than 30 such sets. The reason is printed.
- **A copy that differs in any column is not detected.** Rows that differ in an
  added column, a batch number or a value changed by any amount are treated as
  different rows, and no copies warning is raised for them. Name the column that
  groups rows that belong together with `group_col`.

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

An identifier is excluded as a feature whatever order its rows are in. A
whole-number column is taken as an identifier when every value is distinct and
the values run one after another in the file, or when more than 95% of its values
are distinct and they fill at least half their range. So shuffling the rows, leaving
gaps, or repeating a few rows does not make an identifier a feature. Under time order
the result does not depend on the order of the rows in the file: the rows are put in
time order before anything is fitted, and the rows of one timestamp are taken in file
order.

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
