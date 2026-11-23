# Pass 7 (CP8) build on real public tables

Run by the technical lead (the claude.ai Project) on 23 November 2026, on the
application as built by tasks T54 to T60, before it was committed. Python
3.13 on Linux with the pinned libraries, fast profile unless stated, two
cores. Nothing here is part of the gate and no score here is asserted
anywhere. `cp8_cases.py` states, before each run, what a correct tool should
conclude; `output.txt` is its output and `results.json` the same as data.

25 runs: 22 finished, 3 stopped with a named error, none raised a raw Python
error, none let a library warning through.

| What was tried | Result |
|---|---|
| Seattle weather with the day written `YYYYMMDD`: clean; one cell `99991231`; one cell `0`; 30% of rows `99991231` | the column is excluded as a date every time, and the skill is the same to every digit (0.506) |
| The same column with one `99991231`, as the time column | stops: `time_column_invalid`, naming the value and its one row |
| Eight quarter-ends with one sentinel | excluded as a date (the second clause of D72) |
| Titanic with `alive`, a copy of the target, `duplicates="group"` | stops, names `alive`, says to remove it and run again |
| Taxis with Manhattan as one entity of 5,236 rows, fast profile | stops: `fast_profile_group_too_large` (before this pass: skill 0.878 on the other 1,128 rows, unsaid) |
| The same, full profile | as before: Manhattan is the holdout, and the warning says so |
| mpg with every US car one entity (62.6%), fast profile | kept, named by `large_group_kept_in_training`; 0.822 as before |
| Eight tables with nothing new to say; Titanic under `group` | identical to the previous build, to every digit |

## What is still open for pass 8 (D73), on real data

| What was tried | Today's build |
|---|---|
| VIX level under time order, clean | previous-value baseline, skill 0.023 |
| The same with **one row repeated** | the timestamps are no longer unique, the baseline becomes the training mean, skill **0.843** |
| The same with **an exact rescaled copy of the target** added (`close * 1.07 + 3`) | not excluded, not warned about: skill **1.000**, interval 1.000 to 1.000 |
| Diabetes with **one cell of 999999** in a feature | skill **-0.111**, against 0.490 on the clean table |
