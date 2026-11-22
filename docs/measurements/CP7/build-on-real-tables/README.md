# Pass 6 (CP7) build on real public tables

Run by the technical lead (the claude.ai Project) on 22 November 2026, on the
application as built by tasks T48 to T53, before it was committed. Python
3.13 on Linux with the pinned libraries, fast profile unless stated, two
cores. Nothing here is part of the gate and no score here is asserted
anywhere. The tables are those of `..\..\CP6\real-tables\README.md` plus
`stocks.csv`. `cp7_cases.py` states, before each run, what a correct tool
should conclude; `output.txt` is its output and `results.json` the same as
data.

28 runs: 22 finished, 6 stopped with a named error, none raised a raw
Python error, none let a library warning through.

| What was tried | Result |
|---|---|
| Nothing said (breast cancer) | stops: `time_order_undeclared`, and the same message asks for the entities |
| Only one of the two said | stops, asking for the other |
| One series with a time column, entities unsaid (VIX) | stops: `entities_undeclared`; with `group_col=False`, skill 0.023 as before |
| Command line with neither flag, or one | exit 3, nothing written; with both, it runs |
| Eight tables with both said | skill and interval identical to the previous build, to every digit |
| Titanic, `duplicates="group"` | 0.645 (0.741 before this pass, 0.745 with `keep`); 179 rows agree with another in every feature |
| Titanic, target shuffled, every row twice, `group` | -0.035, interval holds 0 |
| Noise, 24 feature combinations, `group`, three seeds | -0.04, -0.13, -0.05 (before: -0.28, -0.27, -0.56); no interval, as stated |
| mpg, every US car as one entity (62.6% of rows) | holdout a quarter; `large_group_kept_in_training` names the 62.6% |
| Taxis, Manhattan as one entity (82.3%), full profile | Manhattan is the holdout; `large_group_is_holdout` says the headline rests on it alone; no interval |
| VIX with the day as `20240105`, nothing said | stops, naming `day` as a column that holds dates |
| The same, both `False` | `day` excluded as a date, `date_column_undeclared`; 0.939 on the caller's word |
| The same, `time_col="day"` | 0.023, identical to the date-typed column |
| A random 8-digit account number; a `YYYYMM` column | not taken for dates |

## One finding

**The fast profile leaves out an entity larger than its cap, and only the
row count says so.** Taxis with Manhattan as one entity of 5,236 rows: the
fast profile's 2,000-row cap keeps whole entities, so Manhattan can never
fit. The run reports skill 0.878 on the other 1,128 rows, with
`dropped_row_cap: 5236` in the row accounting and no warning. The full
profile handles the same table as D66 says. The build follows the spec
here (section 7: "each kept whole or left out whole"); the gap is in the
contract. It is in `docs\BACKLOG.md` for pass 7.
