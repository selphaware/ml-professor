# Pass 8 (CP9) build on real public tables

Run by the technical lead (the claude.ai Project) on 25 November 2026, on the
application as built by tasks T61 to T68, before it was committed. Python
3.13 on Linux with the pinned libraries, fast profile, two cores. Nothing
here is part of the gate and no score here is asserted anywhere.
`cp9_cases.py` states, before each run, what a correct tool should conclude;
`output.txt` is its output and `results.json` the same as data. 23 runs, all
finished; none raised a raw Python error or let a library warning through.

## What was open after pass 7 and now holds

| What was tried (VIX level under time order unless said) | Pass 7 | Pass 8 |
|---|---|---|
| Clean | 0.023 | 0.023 |
| One row repeated | 0.843 | 0.024, previous value still the baseline |
| 10% of the rows repeated | | 0.018 |
| An exact rescaled copy of the target added | 1.000, silent | the copy is excluded; 0.023 |
| The square of the target added | | warned about by the order reading (kept, as ruled) |
| Diabetes, one cell of 999999 in a feature | -0.111, silent | -0.111; `extreme_value` names the cell, `largest_row_dominates` names the row |
| Titanic with a consecutive id, rows shuffled | the id was a feature | excluded as `id_like` |
| mpg with one entity of 62.6% | 0.822 | 0.776, `selection_not_made` says no choice was made |

Eight tables with nothing new to say give the same skill and interval as
the previous build, to every digit.

## Two findings (`review_cp9.py`; the application and the reference agree on both)

**1. The order reading warns about columns of pure noise when they hold
repeated values.** On clean diabetes it names `age` and `s6` (correlation
with the target 0.19 and 0.38). On tables of pure noise it names a column
rounded to one decimal place and a whole-number score from 1 to 40, on three
seeds of three. The spec sorts rows of equal column value by the target
(step 8, "among equal values, by the target"), so a pair across two values
compares the highest target of one group with the lowest of the next, and
the target nearly always falls. For `age`: 57 pairs, share against the order
0.035 as specified, 0.474 with ties in random order. The gate's and the
battery's numeric columns are continuous, so neither saw it.

**2. A copy of the target with five training cells changed by a large amount
is warned about and kept, not excluded.** Under time order, five cells
changed by +7, by a factor of 1.5, or to another row's value: excluded on 2
of 24 tables, warned on 22, headline skill 0.89 to 1.00. Changed by +0.5:
excluded on 8 of 8. The harness measured "5 cells differ" as excluded on 30
of 30; the size of the change matters and was not varied.
