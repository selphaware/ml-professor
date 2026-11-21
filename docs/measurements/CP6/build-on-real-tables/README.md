# Pass 5 (CP6) build on real public tables

Run by the technical lead (the claude.ai Project) on 21 November 2026, on the
application as built by tasks T39 to T47, before it was committed. Python
3.13 on Linux with the pinned libraries, fast profile, two cores. Nothing
here is part of the gate and no score here is asserted anywhere. The tables
are the ones listed in `..\real-tables\README.md`, plus `stocks.csv` from
vega/vega-datasets. `cp6_cases.py` states, before each run, what a correct
tool should conclude; `output.txt` is its output and `results.json` the same
as data. It imports `lagged` from `..\..\real-data-2026-11-21\cases.py`.

31 runs: 19 finished, 12 stopped with a named error, none raised a raw
Python error, and none let a library warning through.

| What was tried | Before this pass | Now |
|---|---|---|
| VIX level with a date column and nothing said | skill 0.939, a warning | stops: `time_order_undeclared` |
| The same with `time_col="date"` | 0.023 | 0.023 |
| The same with `time_col=False` | | 0.939, on the caller's word, with the warning |
| Five-stock price panel, `time_col` only | 0.979, a warning | stops: `entities_undeclared` |
| The same with `group_col="symbol"` | | each stock's previous value is the baseline; skill -0.139 |
| The same with `group_col=False` | | 0.979, on the caller's word, with the warning |
| Stock returns as a panel | about 0 | about 0, with or without the entity column; no stop |
| mpg with `group_col="name"` | | grouped scheme, 230 and 75 entities, interval from whole entities, no p-value, skill 0.783 (0.822 row by row) |
| Five stocks, or three islands, as entities under the random scheme | | `too_few_groups` |
| An entity column with 25 empty cells (taxis, pickup zone) | | `group_column_invalid`, with the count |
| Titanic, default | stops | stops; the message now names `duplicates="group"` |
| Titanic, `duplicates="group"` | | grouped; 160 rows with a copy; skill 0.741 (0.745 with `keep`) |
| Titanic, target shuffled, every row twice, `keep` | | 0.733: inflated, as `keep` permits |
| The same with `duplicates="group"` | | -0.154, interval holds 0 |
| Breast cancer, target shuffled, every row twice, `group` | | -0.033, interval holds 0 |
| Breast cancer, target shuffled, every row twice and a batch column | 0.35 to 0.44 on the tables of the earlier real-data run | **0.741, no stop and no warning: the open defect of D60** |
| Breast cancer, diabetes, penguins, diamonds, Titanic with `keep` | | skill and interval identical to the previous build, to every digit |

One thing to know: `examples\customers.csv` has a date column, so a plain
call on it now stops and asks; the README's commands pass `--no-time-order`.
