# CP6: three rules for holdout rows that repeat a training row

`repeat_rules.py` is the measurement prototype of CP6 made to run on its own, so
that the three rules can be tried on real tables before one of them is decided
on. It needs pandas and numpy and nothing else: it imports no part of the tool,
of its tests or of the gate. It is not the tool, and nothing in the tool uses it.

| Rule | A holdout row is counted when |
|---|---|
| `d49` | a training row has the same value in every feature column and the same target (decision D49, the rule the tool has today) |
| `first_rule` | it has such a copy, or a training row agrees with it in more than half of the columns compared, two rare values among them (the rule first proposed for D57) |
| `calibrated` | a training row agrees with it by more than chance explains in this table; the count is what is left once chance and a margin are taken off |

Each rule would stop the run when it counts more than 1% of the holdout rows and
at least 10 of them.

## The call

```python
import repeat_rules

found = repeat_rules.count_repeats(frame, "churned", "classification", seed=0)
found = repeat_rules.count_repeats(frame, "churned", "classification", holdout=positions)
```

| Argument | Is |
|---|---|
| `frame` | a pandas DataFrame, one row per observation |
| `target` | the name of the target column |
| `task` | `"classification"` (exactly two classes) or `"regression"` |
| `seed` | a whole number: the holdout is a random quarter of the rows that have a target, rounded up, in proportion to the classes |
| `holdout` | given in place of `seed`: the positions of the holdout rows, 0 for the first row of `frame`. Every other row with a target is a training row |
| `exclude` | optional: names of columns to leave out of the comparison |

It returns a dictionary:

| Key | Holds |
|---|---|
| `d49`, `first_rule`, `calibrated` | `count` (holdout rows the rule counts), `share` (of the holdout rows), `stop` (true or false) |
| `calibrated`, besides | `counted`: holdout rows that have a training row as near as the rule allows them. `chance`: how many of those chance alone gives. `spread`: the spread of that number. `level`, `levels`: the count is made at ten levels of nearness and the largest is taken; these say which, and give every level. `exact_only`: the rows judged on an exact copy alone. `facts`: the columns that stood as facts, coarsest first. `settled`: the columns left out because another column settles them |
| `rows` | rows given, without a target, training, holdout |
| `holdout` | the positions of the holdout rows |
| `columns` | the names compared, and the names left out with the reason for each |
| `classes` | the two classes in order (the second is the positive one), or `None` |

For the calibrated rule `count` is not `counted - chance`: at each level it is
the rows counted, less the chance number and four spreads of it, to the nearest
row, and never below zero.

## An example

`python repeat_rules.py` builds a table of 6,000 independent rows of the kind of
a customer file (two categories, three whole-number columns, a class) and runs
the three rules on it, then on the same table with 5% of its rows written
twice, and with every row written twice. The copies are hidden from `d49` by an
added column of unique values. On Python 3.12.13, pandas 3.1.1 and numpy 2.5.4:

```
independent rows: 1500 holdout rows
    d49         count   35   share 0.023   stop
    first_rule  count   35   share 0.023   stop
    calibrated  count    0   share 0.000   run    (35 counted, chance gives 50.6)
5% of the rows twice, a column of unique values added: 1575 holdout rows
    d49         count    0   share 0.000   run
    first_rule  count    0   share 0.000   run
    calibrated  count   15   share 0.010   run    (54 counted, chance gives 21.1)
every row twice, a column of unique values added: 3000 holdout rows
    d49         count    0   share 0.000   run
    first_rule  count    0   share 0.000   run
    calibrated  count  732   share 0.244   stop   (778 counted, chance gives 25.7)
```

The first table has no repeated row: `d49` and the first rule stop it on 35
chance copies, and the calibrated rule finds no more than chance gives. In the
second, 15 of 1,575 is 0.95%, just under the stop: where chance copies are
common, a small share of real ones is hard to tell from them. The same call in
Python:

```python
import repeat_rules

frame = repeat_rules.with_copies(repeat_rules.example_table(), share=1.0)
found = repeat_rules.count_repeats(frame, "churned", "classification", seed=0)
found["calibrated"]["count"], found["calibrated"]["stop"], found["calibrated"]["chance"]
# (732, True, 25.7...)
```

## The six constants of the calibrated rule

They are at the top of `repeat_rules.py`. The docstring of `calibrated` gives the
rule in five steps.

| Constant | Value | What it does |
|---|---|---|
| `FLOOR` | 20 | How many rows must bear a thing out before the table is believed on it: a column settles another only on 20 repeated values, and the share of rows that agree in several columns is counted directly only while 20 other rows do |
| `RARE` | 5 | A value held by 5 rows or fewer is given the chance its kind has in that column (values held by so many rows, by the count of counts), not its own share of the rows |
| `GATE` | 0.05 | A training row as near as this is improbable when fewer than 0.05 such rows are expected by chance; it sets how many columns a near-copy of each holdout row may differ in |
| `WIDE` | 0.5 | A holdout row for which even an exact copy is not that improbable is still judged, on an exact copy alone, when fewer than 0.5 are expected |
| `MARGIN` | 4.0 | The spreads of the chance number taken off, with that number, before any row counts as a repeat |
| `MAX_DIFFER` | 9 | The most columns in which a near-copy may differ, whatever chance says |

The stop (`STOP_SHARE` 0.01, `STOP_ROWS` 10) is D49's and is the same for the
three rules. The first rule has its own constants: `RARE_HOLDERS` 5 and
`RARE_AGREEMENTS` 2.

## How a table is read, and where this differs from the tool

- Rows with no target are left out before the split. A numeric target that is
  infinite is no target.
- A column is read as the tool's step 5 reads it. Numbers, booleans and text
  that parses as numbers are compared as 64-bit floats: two empty cells are
  equal, and 0.0 equals -0.0. Other text is compared without the white space
  round it, an empty cell being the level `(missing)`. Left out, with the
  reason in `columns["left_out"]`: `all_missing`, `constant`, `datetime`,
  `id_like`, `free_text`, `numeric_with_separators`, `unsupported_type`.
- The leak screen is not run here, since it fits models. A column the tool
  excludes as a suspected leak is compared here unless it is named in `exclude`.
- The fast profile's cap of 2,000 rows is not applied, and there is no time
  order and no entity column: a date column is left out, as the tool leaves it.
- With `seed`, the holdout rows are the ones the tool's random split takes at
  that seed. Checked on this interpreter: the split against scikit-learn 1.9.1
  in 3,583 cases, none differing; and the whole of it, on 147 tables, against
  the lab copy of the reference that carried the same three counts: the same
  holdout rows, the same columns and the same three counts on every one.

## What the calibrated rule is known not to see

Measured in CP6, on tables with a target of pure noise unless said:

- Few distinct rows, each with many holders. Three columns of three levels,
  every row twice: never found. Ten numeric columns, every row present 2, 5, 10
  or 20 times: the run stops; 21, 22, 25 or 40 times: it counts 0 (one table
  each). That is `FLOOR`: 20 other rows that agree in everything are taken as a
  cell of the table. `d49` and the first rule stop those tables.
- A small share of copies where chance copies are common, as in the example.
- A copy changed in nine or in all of ten numeric columns (in eight it is still
  found), and, by `MAX_DIFFER`, any copy changed in more than nine columns.
  Where the columns are ten categories of four levels: an added unique column
  and two more changed columns.

## When it gives up

The search for near-copies lists the pairs of rows that agree in the same
columns. Past 60 million pairs in one pass it raises `MemoryError` with a
message that names this section. It has not happened on a table tried so far.
If it does, the table's shape (rows, columns, distinct values in each) is worth
more than a workaround: please pass it on.

## What it was run on

Python 3.12.13, pandas 3.1.1, numpy 2.5.4, on Windows. It has not been run on
Python 3.13 or on pandas 2, and uses no call particular to one version that I
know of. Run the example first: if it prints other numbers, either the example
table differs (numpy does not promise the same random table in every version)
or something here does, and that is worth knowing before a real table is read.

Time, the three rules together: 0.1 seconds on the example's first table; 3
seconds on 48,000 rows by 14 columns; 10 to 23 seconds on 100,000 rows by 16 to
100 columns.
