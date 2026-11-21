# CP6: the three repeat rules on real public tables

Run by the technical lead (the claude.ai Project), 21 November 2026, on
`docs\measurements\CP6\repeat_rules.py` exactly as the CP6 harness session
saved it. Nothing here is part of the gate, and no score here is asserted
anywhere. Python 3.13 on Linux with the pinned libraries (the repo uses 3.12
on Windows); the module's own example gave the same output as its README.

## What was run

Eleven tables, each split 25% holdout with seed 0 by `repeat_rules.count_repeats`.

| Table | Rows | Target | Source |
|---|---|---|---|
| titanic (`alive` left out) | 891 | survived | mwaskom/seaborn-data |
| diamonds | 53,940 | price | mwaskom/seaborn-data |
| penguins (rows with sex) | 333 | sex | mwaskom/seaborn-data |
| taxis (pickup and dropoff times left out) | 6,389 | payment | mwaskom/seaborn-data |
| tips | 244 | smoker | mwaskom/seaborn-data |
| mpg (`name` left out) | 398 | mpg | mwaskom/seaborn-data |
| Seattle weather (date left out) | 1,460 | rain tomorrow | vega/vega-datasets |
| VIX daily open, high, low, close (date left out) | 9,277 | next day's change | datasets/finance-vix |
| breast cancer | 569 | target | scikit-learn |
| diabetes | 442 | target | scikit-learn |
| customers | 400 | churned | `examples\customers.csv` in this repo |

To run again: put the CSV files in one folder, set `REAL_DATA` to it, and
run `python real_tables.py clean`, `copies`, `over`; `python detail.py`;
`python consequence.py clean`, `copies`. The scripts import `repeat_rules`,
so run them with `docs\measurements\CP6` on the path.

## Findings

1. **The calibrated rule stops clean real tables.** Diamonds: 4,780 of
   13,485 holdout rows counted as repeats (35%). Taxis: 307 of 1,598 (19%).
   VIX: 197 of 2,320 (8%). Titanic: 12 of 223. On the session's own 1,590
   clean tables it stopped none. `out\calibrated_levels.txt` shows where:
   the count comes from rows that agree with a training row in all but 3 to
   5 columns, and the rule expects far fewer such rows than there are
   (diamonds: 134 expected, 4,959 found). Its expectation treats the columns
   as varying independently. In these tables they do not: a diamond's
   length, width, depth and weight move together, as do a day's open, high,
   low and close, and a taxi ride's distance, fare and total.
2. **The calibrated rule passes heavy oversampling.** With the smaller class
   copied 25 or 50 times before the split it runs on 4 of 5 tables; D49 and
   the first rule stop every one. A whole table repeated 21 or 30 times also
   runs. This is the constant of 20 the session reported.
3. **The calibrated rule does catch the disguises.** Every row twice with a
   column of unique values or a batch column: it stops 22 of 22. D49 stops
   1 of 22, the first rule 17 of 22.
4. **The first rule stops clean taxis** (68 of 1,598) and Titanic.
5. **D49 stops Titanic only** (31 of 223), which was known.

Outputs: `out\rules_clean.txt`, `out\rules_copies.txt`,
`out\rules_oversampled.txt`, `out\calibrated_levels.txt`.

## A simpler count, tried once (`consequence.py`)

Not a proposal for the spec; a first look at whether a simpler rule could
do the job. A holdout row is a "twin" if a training row has the same target
and the same value in every column but at most one, that one column being
the same for the whole table, with numbers compared to 6 significant
digits. The script also refits a random forest with the matched training
rows removed and reports the fall in skill.

- Clean tables with several measured columns: twins are 0% to 1% of the
  holdout (diamonds 0.8%, taxis 0.9%, VIX 0%), and the skill does not move.
- Every row twice, under each disguise including numbers moved by 1e-9, with
  the target shuffled so the true skill is 0: twins are 73% to 84%, and the
  skill falls by 0.30 to 0.98 when the matched rows are removed.
- 5% of rows twice: twins are 6% to 16%.
- **It fails on tables of few-valued columns.** Titanic 44% twins, tips 15%,
  weather 17%, a made-up table of eight three-level columns 81%, all clean;
  and removing the matched rows lowers the skill there with no copy present
  (Titanic by 0.13). On such tables the table alone does not say whether
  matching rows are copies or different cases that look alike.

Outputs: `out\twins_clean.txt`, `out\twins_copies.txt`. Two constants, eleven
tables, one seed, one model: treat it as a lead, not a result.
