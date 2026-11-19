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
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -m mlprofessor profile --data examples/customers.csv --target churned --out runs/customers
```

Options (`python -m mlprofessor profile` takes these):

| Option | Meaning | Default |
|---|---|---|
| `--data` | the table, as a CSV file | required |
| `--target` | the target column | required |
| `--out` | folder for `profile.json` and `predictions.csv` (created if missing) | required |
| `--task` | `auto`, `classification` or `regression` | `auto` |
| `--time-col` | the time column; enables the time-ordered scheme | none (random scheme) |
| `--gap` | whole timestamps left out between training and holdout (needs `--time-col`) | 0 |
| `--profile` | `full` or `fast` (see below) | `full` |
| `--seed` | the random seed; the same seed gives the same output | 0 |
| `--n-jobs` | threads; `-1` uses every core | -1 |

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

Exit codes: `0` finished and both files written; `2` the command line could not
be parsed; `3` a named error, with one line `error: <code>: <message>` on
standard error and no files written.

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
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -m mlprofessor profile --data examples/customers.csv --target churned --profile fast --out runs/customers_fast
```

Without `--time-col` the tool never guesses a time column. It notices that
`signup_date` looks like a date, excludes it, and says so in a
`date_column_undeclared` warning.

## Python

```python
from mlprofessor import profile

result = profile(
    "examples/customers.csv",   # a path, or a pandas DataFrame
    "churned",
    task="auto",          # "auto", "classification" or "regression"
    time_col=None,        # name of a time column, or None for the random scheme
    gap=0,                # whole distinct timestamps between training and holdout
    profile="full",       # "full" or "fast"
    seed=0,
    out_dir="runs/customers",   # optional: writes profile.json and predictions.csv
    n_jobs=-1,
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
| `input` | source, kind, target, time column, task requested, gap, row and column counts, duplicate rows |
| `rows` | seven counts: loaded, four reasons for dropping a row, training, holdout |
| `columns` | one record per input column: role, kind, reason, leak rule, and counts |
| `task` | type, whether it was decided by rule or declared, distinct values, classes, positive class |
| `validation` | scheme, holdout share, folds, gap, time form, last training and first holdout timestamp, caveat |
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
Every number in `headline.score` can be recomputed from `predictions.csv`.

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
  says so.
- **Time-ordered scheme** (`time_col` given). The holdout is the latest whole
  timestamps that together hold at least a quarter of the rows; rows sharing a
  timestamp are never split. `--gap N` removes the N timestamps just before the
  holdout from training, to keep near-term information out. Training is cut into
  six contiguous blocks, and each fold validates on the next block. The time
  column must be a datetime column, a numeric column, or text in ISO 8601 with
  the year first (`YYYY-MM-DD`, optionally with a time). Anything else, such as
  `25/01/2024`, raises `time_column_invalid`: the tool never guesses between
  day-first and month-first, so convert the column first.

Under the time-ordered scheme the regression baseline also includes
`previous_value` when timestamps are unique, so a trending target is not scored
only against the training mean.

## Profiles

- `full` (default): 18 model configurations across three families (linear,
  random forest, gradient boosting), five-fold cross-validation inside the
  training rows, 1,000 resamples for the interval and p-value.
- `fast`: two configurations (a linear model and a single decision tree), no
  grid, 200 resamples, and a cap of 2,000 usable rows (the rows kept are
  chosen from the seed, and the rows left out are counted in `rows`). Use it
  to look at a table quickly. Under the time-ordered scheme with a wide panel,
  the cap can leave too few timestamps, which raises `fast_profile_too_few_timestamps`;
  use `full` then.

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
- pick the time column for you, or guess day-first versus month-first dates.

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
why. The full list of error codes is in `spec.md` section 2.4.

## Files and reproducibility

- `spec.md` is the contract: the methodology, the output schema and the errors.
- `docs/DECISIONS.md` records the reasons behind each choice.
- The same arguments and seed give the same `profile.json` and `predictions.csv`,
  apart from `meta.timing`, and apart from `meta.n_jobs` when `--n-jobs` differs.
- If scores move after a library upgrade, compare `meta.versions` in
  `profile.json` first.
