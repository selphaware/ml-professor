# Pass 9 (CP10) build on real public tables

Run by the technical lead (the claude.ai Project) on 27 November 2026, on the
application as built by tasks T69 to T76, before it was committed. Python 3.13.18
on Linux with the pinned libraries (pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1,
threadpoolctl 3.8.0), two cores, the fast profile unless a line says full. Nothing
here is part of the gate and no score here is asserted anywhere. Each script
states, before each run, what a correct tool should conclude. `lagged` comes from
`docs/measurements/real-data-2026-11-21/cases.py`. Every run finished; none
raised a raw Python error or let a library warning through.

| Script | Output | What it asks |
|---|---|---|
| `cp10_cases.py` | `cp10_cases.txt`, `results_cp10.json` | the 23 cases of pass 8, with the new statements printed |
| `review_cp10.py` | `review_cp10_app10.json`, `review_cp10_ref10.json` | the lead's two findings on pass 8, application and reference |
| `review7_on_cp10.py` | `review7_on_cp10.txt` | review 7's findings 1 and 2, full profile |
| `cp10_levels.py` | `cp10_levels_app10.txt`, `cp10_levels_ref10.txt` | the reading by levels on honest columns, under the rule T76 adopted and the rule T71 first built |
| `cp10_extra.py` | `cp10_extra_app10.txt` | the near-perfect warning on the two-column carrier, every fall on diabetes, a heavy-tailed target |

## What was open and now holds

| What was tried | Pass 8 | Pass 9 |
|---|---|---|
| Diabetes, clean: the order reading on columns with repeated values | `age` and `s6` warned | nothing warned; skill 0.490034, unchanged |
| Pure noise with a rounded column and whole-number scores, 3 seeds | warned on 3 of 3 | nothing warned on 3 of 3 |
| VIX level + a copy of the target with 5 training cells changed by +7, by ×1.5, or to another row's value, 8 seeds each | excluded on 2 of 24, warned on 22, skill 0.89 to 1.00 | excluded on 24 of 24 |
| The same with +0.5 | excluded on 8 of 8 | excluded on 8 of 8 |
| Diabetes and mpg + the target in 24 bands, labels in random order, 3 seeds each (review 7, finding 1) | kept, headline 0.95 to 1.0 | excluded on 6 of 6, with the sentence on `group_col`; headline equal to the clean table's (0.490, 0.856) |
| Diabetes + a = y + z and b = z (review 7, finding 2) | 1.000, nothing said | 1.000; `near_perfect_headline` names `a` (fall 4.71) and `b` (fall 2.28), full and fast profile; on mpg the skill is 1.000 too, so the warning fires by its rule (its text was not printed) |

Of the 23 cases of pass 8, 22 give the same skill to every digit. The one that
moved is the copy with five cells changed, now excluded (0.975 to 0.023). The
warnings differ only where they should: `strong_single_column` is gone from the
three diabetes runs, and `near_perfect_headline` is new on the four runs at 0.98
or more.

## The new statements

- **`headline.rests_on` and its floor** are on every result, and the console
  summary prints them (`rests on (columns whose skill falls by at least 0.129688
  shuffled): 's5' 0.427562, 's1' 0.423895, 'bmi' 0.191774` on diabetes).
- **`near_perfect_headline`** fired on exactly the runs at 0.98 or more:
  - VIX + the square of the target (0.9986, names it);
  - diabetes with a target cell of 999999 (0.9992, names no column; `extreme_value`
    and `largest_row_dominates` name the cell and the row, as in pass 8);
  - breast cancer (0.9945, "rests on no one column");
  - taxis fare (0.9960, names `total` and `tip`);
  - diamonds price on the full profile (0.9813, names `carat`, `y`, `clarity`, `color`).

  The last two are honest easy tables, which D77 expected to carry the warning.
- **`holdout_far_above_quarter`** fired on none of the ten time-ordered runs of `cp10_cases.py`.
- **`shuffled_score_not_computed`** fired on none.

## The reading by levels after T76

T76 changed which numeric columns are read by their levels, to agree with the
reference and with step 5's convention: training rows at least 5 × the distinct
values, where T71 had built at least five rows of every value. On ten real tables
the new rule reads 21 more columns by their levels than the first rule. Among them:
- diamonds `carat`, `x`, `y`, `z`, `depth` and `table`;
- titanic `age` and `parch`;
- diabetes `age`, `s3`, `s4` and `s6`;
- mpg `cylinders`, tips `size`, planets `year`, and taxis `fare` and `tolls`.

On every table the two rules give the same headline, the same exclusions (none),
the same warnings (none from the screen) and the same `rests_on`. The highest
screen skill among the columns read so is 0.877 (diamonds `y`, full profile),
below the warning tier of 0.90.

## Two observations for the review (not defects of the build)

1. **A fall can be larger than what the column carries alone.** The fall is the
   chosen model's dependence on the column. A linear model can offset correlated
   columns against each other, so shuffling one breaks the balance: on diabetes,
   `s1` falls by 0.424, `s2` by 0.062 and `s5` by 0.428. The reference, on its own
   split, gives 0.175 for `s1`. The README explains the converse case (columns that
   share information each fall less) but not this one. One README sentence would
   cover it.
2. **A heavy-tailed target makes the floor large.** Planets orbital period: skill
   0.478 with an interval of -21.0 to 0.88, so the floor is 10.96 and no column is
   named. Four `largest_row_dominates` warnings say why. This is D80 point 1 as
   designed.

As in pass 8, the application and the reference give different skills on real
tables (diabetes 0.490 and 0.358, diamonds 0.933 and 0.862). The rows of the split
are open (spec 11). The generated-table test compares the two, and it agrees on
1,200 of 1,200.
