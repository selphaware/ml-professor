r"""spec.md for the ninth pass, part 1 of 3: sections 1 to 7 (D77, D78, D80, D81).

    python -B edit_spec10a.py <spec.md in> <spec.md out>
"""
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
t = SRC.read_bytes().decode("utf-8")
assert "\r" not in t


def sub(old, new, count=1):
    global t
    found = t.count(old)
    assert found == count, f"{found} matches, expected {count}, for:\n{old[:160]}"
    t = t.replace(old, new)


# ------------------------------------------------------------ preamble --
sub("D76) without changing them,", "D81) without changing them,")

# ---------------------------------------------------- 2.2 the summary --
sub('''- one line per warning, holding that warning's `code` and, when it has one,
  its `column`. Two warnings never share a line.''',
    '''- one line per warning, holding that warning's `code` and, when it has one,
  its `column`. Two warnings never share a line;
- **one line for what the headline rests on (step 12a; D78 point 1, D80,
  D81),** on every finished run. It begins with the words `rests on`; it
  holds `headline.rests_on_floor` as a number when that is not null; and it
  names each column of `headline.rests_on`, in the list's order, in quotes
  as Python's `repr` writes a text, each with its `skill_fall` as a number.
  When the list is empty it holds the word `none` in their place.''')

# --------------------------------------------- 2.7 the column-roles step --
sub('''A DataFrame with the eleven columns of `profile.json["columns"]` (5.2), one row per column of `frame` in order, as decided by step 5 alone: `leak_rule` is null and `n_imputed` is 0 throughout, since the leak screen and imputation come later.''',
    '''A DataFrame with the fourteen columns of `profile.json["columns"]` (5.2), one row per column of `frame` in order, as decided by step 5 alone: `leak_rule`, `shuffled_score`, `shuffled_skill` and `skill_fall` are null and `n_imputed` is 0 throughout, since the leak screen, imputation and the statement of step 12a come later.''')

# ------------------------------------------------------ 2.5 the codes --
sub('''| `suspected_leak` | the column | One per column excluded by the leak rule (D11). |''',
    '''| `suspected_leak` | the column | One per column excluded by the leak rule (D11). For a column read by its levels (step 8: every categorical column, and a numeric column of few distinct values) and excluded by rule 2, the message also holds the fixed sentence of 5.4 that says to declare the column as `group_col` if it names the entity each row belongs to (D78 point 6). |''')
sub('''with the training rows put in the column's order, the target moves the other way between at most 10% of neighbouring rows, and it does not follow its own previous value as closely.''',
    '''the target moves against the column's order in at most 10% of the comparisons between training rows of neighbouring distinct values, rows of equal value never being put in the target's order (D77 point 4), and it does not follow its own previous value as closely.''')
sub('''| `extreme_value` | the column |''', '''| `near_perfect_headline` | `null` | Exactly one when `headline.skill` is 0.98 or more (step 12a; D77 point 2): the level at which one column alone is excluded as a suspected leak. In both tasks and under every scheme. Its message names each column of `headline.rests_on` and asks whether each is known before the outcome; when the list is empty it says that no column's fall reaches the floor. Never otherwise, and never when `skill` is null. It changes nothing: no column is excluded for it. A legitimately easy table carries it too (D77, "Consequence"). |
| `shuffled_score_not_computed` | the column | One for each feature whose statement could not be made: the chosen configuration raised, or predicted a value that is not finite, when it scored the holdout with that feature's holdout values shuffled (step 12a; D78 point 7). The three keys of its record are null (5.2) and the run goes on. Never otherwise. |
| `holdout_far_above_quarter` | `null` | Exactly one, under time order, when the holdout holds half of the rows split or more: the training, gap and holdout rows (step 7; D77 point 6, D78 point 7). Whole timestamps made it far more than a quarter. The message holds the holdout rows and the rows split, and the fixed passage of 5.4. Never under the random or the grouped scheme, where `large_group_is_holdout` says the like of one unit. |
| `extreme_value` | the column |''')

# ---------------------------------------------------- step 7: the E rule --
sub('''For classification, a holdout or a set of training rows with one class raises
`one_class`. A fold whose training or validation rows hold one class is not
scored for any configuration, and one `fold_not_scored` warning is raised for
each such fold (D72); if no fold can be scored, `one_class`.
''', '''For classification, a holdout or a set of training rows with one class raises
`one_class`. A fold whose training or validation rows hold one class is not
scored for any configuration, and one `fold_not_scored` warning is raised for
each such fold (D72); if no fold can be scored, `one_class`.

**A time-ordered holdout far above a quarter is named (D77 point 6, D78 point
7).** The holdout is the latest whole timestamps that make a quarter, so a
timestamp that holds many rows can make it far more: when the last date of a
table holds 60% of the rows, that date alone is the holdout, and the models
learn from the 40% before it (the seventh review, finding 3). **When the
holdout holds half of the rows split or more** (the training, gap and holdout
rows: `rows.training` + `rows.dropped_gap` + `rows.holdout`), one
`holdout_far_above_quarter` warning says so. Its message holds the holdout
rows and the rows split as `H of the N rows`, both whole numbers in digits,
the share as a percentage, the rows of the holdout's earliest timestamp, and
the fixed passage of 5.4. Nothing else changes: the split is the split, and
the warning is a statement about it. Under the grouped scheme
`large_group_is_holdout` says the like of one unit, and was the model. The
cut-off of one half was measured (10.11): on tables whose last timestamp
holds from 0.2 to 0.7 of the rows the warning is raised exactly when that
share is one half or more; on tables of 8 to 40 equal timestamps, whose
holdouts hold 0.25 to 0.33 of the rows, never; and on no run of the gate
before this pass.
''')

# ------------------------------------------------- step 8: the leak screen --
sub('''**8. The leak screen (D11, D24, D30, D74, D76), on training rows only.**''',
    '''**8. The leak screen (D11, D24, D30, D74, D76, D77, D78), on training rows only.**''')
sub('''*The readers (D74).* A column is read by a tree and, beside it, by a second
reader. The column's skill is that of the better of the two: the higher
skill.''', '''*The readers (D74, D77).* A column is read by a tree and, beside it, by one or
two other readers. The column's skill is that of the best of them: the
highest skill.''')
sub('''- **The column read directly,** in a classification. A leaf of at least five
  rows cannot hold a class that has fewer than five rows among a fold's
  training rows, so the tree could not follow an exact copy of such a target
  (section 9). For a numeric column the direct reader is a tree of one split
  with no minimum leaf, `DecisionTreeClassifier(max_depth=1,
  min_samples_leaf=1, random_state=seed)`, fitted on the column as it is, and
  its probability of the positive class. For a categorical column it is, for
  each level, the share of the positive class among the fold's training rows
  of that level: a validation row is given the share of its level, and a
  level that the fold's training rows do not hold is given the share of the
  positive class among all of the fold's training rows.
- A categorical column of a regression is read by the tree alone.''',
    '''- **The column read directly,** for a numeric column of a classification. A
  leaf of at least five rows cannot hold a class that has fewer than five
  rows among a fold's training rows, so the tree could not follow an exact
  copy of such a target (section 9). The direct reader is a tree of one split
  with no minimum leaf, `DecisionTreeClassifier(max_depth=1,
  min_samples_leaf=1, random_state=seed)`, fitted on the column as it is, and
  its probability of the positive class.
- **The column read by its levels (D77 point 3, D78 point 5),** in both
  tasks, for every categorical column and for a numeric column of few
  distinct values: one whose training rows of step 7 in which it is not
  missing number at least five for each distinct value among them. For each
  validation row, the reader predicts **the mean target of the fold's
  training rows that hold the row's level**: in a classification the share of
  the positive class among them. A level is, for a categorical column, the
  stripped text of step 5, `(missing)` being a level like any other; for a
  numeric column, the value, a missing cell being no level. **A validation
  row whose level the fold's training rows do not hold is not scored by this
  reader:** its skill is measured over the pooled rows it scores, against the
  same baselines as every reader, the lower counting, and rules 2 and 3 read
  it as below, over those rows. **The reading has a value only when at least
  20 pooled validation rows are scored and, in a classification, at least 10
  of each class are, or every pooled validation row of that class is;** with
  fewer, or where a baseline's squared error over the rows it scores is zero,
  it is no reader of that column, and the others decide. A categorical column
  of a classification was read directly until D77, an unseen level being
  given its fold's share of the positive class; this reader takes the place
  of that reading. A categorical column of a regression was read by the tree
  alone, on codes of its sorted level names.

*Why levels, and why unseen levels are not scored.* A column that is the
target cut into 24 or 30 bands, with labels in random order, is noise to a
tree on codes of its sorted labels: it was kept, often in silence, with the
headline at 0.95 to 1.0 (the seventh review, finding 1), and a numeric code of
the target likewise, in both tasks. Read by its levels it is excluded on
every table from 16 bands up under the random scheme, and on most beside
entities (10.11); coarser bands are weaker columns, and under time order the
latest rows' levels are not the earlier rows': section 11 says what becomes
of them (D78 point 4, D80 point 3). An unseen level given the baseline's prediction, as CP10 first
worded it, scores rows that the reading knows nothing of: 24 battery pairs
failed so (D78). Leaving them out needs a floor, because a classification
then reads noise perfectly on a few rows: with one to four scored rows of
the rarer class a noise column read 1.000; with the floor, none of 16,836
noise columns was excluded or warned about (10.11).

**A column of codes is judged like any other (D78 point 6).** A column that
names the entity each row belongs to, where each entity holds one value of
the target, predicts the target from its levels as a coded copy does, and
the screen cannot tell the two apart. It is excluded by rule 2 like any
column that predicts the target at 0.98, and the `suspected_leak` message of
a column read by its levels holds the fixed sentence of 5.4 that tells the
caller to declare it as `group_col` if that is what it is.''')
sub('''**The rows marked by either are
set aside when they are at most 1% of the training rows:** at most
⌊0.01 × the training rows⌋ of them, and never fewer than one row is allowed.
A row set aside is in no fold's training rows and in no fold's validation
rows for the screen: in no mean, no tree, no line, no baseline and no sum. A
fold that is left with no training row or no validation row by it takes no
part in the screen. **When more rows are marked than that, none is set aside
and rule 2 scores every row, as rule 3 does:** that many rows far off are
the shape of the target and not a stray cell, and the screen is then what it
was before D74 with the second reader added.''',
    '''**The rows marked by either are
set aside when they are at most 1% of the training rows:** at most
⌊0.01 × the training rows⌋ of them, and never fewer than one row is allowed.
A row set aside is in no fold's training rows and in no fold's validation
rows for the screen: in no mean, no tree, no line, no reading by levels, no
baseline and no sum. A fold that is left with no training row or no
validation row by it takes no part in the screen. **When more rows are
marked than that, none is set aside (D77 point 5, D78 point 7):** that many
rows far off are the shape of the target, a long tail, and not a stray cell.
Rule 2 still leaves its rows unscored, as above, and **its reading is also
taken against each fold's training median,** a further baseline that gives
every validation row the median target of its fold's training rows, the
lower value counting as against every baseline. Rows far off cannot move a
median: so a column of noise that fits a few of those rows is not excluded
for beating a mean that they drag away from every other row, as it was
beside a target with 1.2% to 1.5% of its cells set to −999 when the median
was left out (10.11). Rule 3 and the order reading score every row, as
always. Until D77 rule 2 then scored every row as well: a few changed cells
of a copy of a long-tailed target were most of the copy's error, and a copy
with five cells changed by several of the target's daily steps, by a factor
of 1.5 or to another row's value was warned about and kept on 22 of 24 real
tables, with headline skill 0.89 to 1.00 (D77 point 5). Such a copy is now
excluded at every size of change, up to the 1% of rows the rule leaves
unscored (10.11).''')
sub('''Take the training rows in which the column is not missing, sorted by the
column's value and, among equal values, by the target. A *pair* is two rows
next to each other in that order whose values of the column differ. The
**share against the order** is the smaller of two shares of the pairs: those
in which the target is higher in the second row, and those in which it is
lower. It is 0 when the target rises with the column without exception, or
falls with it without exception. With fewer than 20 pairs it has no value.''',
    '''Take the training rows in which the column is not missing. **Rows of equal
column value are never put in the target's order (D77 point 4, D78 point
7).** For each two neighbouring distinct values of the column, *v* and the
next larger *w*, every row that holds *v* is compared with every row that
holds *w*: the share of those comparisons in which the target is higher in
the row of *w*, and the share in which it is lower (an equal target counts in
neither). Each of the two is averaged over the neighbouring pairs of values,
each pair counting once however many rows it holds, and the **share against
the order** is the smaller of the two averages. It is 0 when the target rises
with the column without exception, or falls with it without exception. With
fewer than 20 neighbouring pairs of values (21 distinct values) it has no
value. On a column whose values are all distinct it is the reading D76
built. Until D77 the rows were sorted by the column and, among equal values,
by the target, and neighbouring rows were compared: inside each run of equal
values the target then rose, and a column of noise with repeated values
(`age` and `s6` of scikit-learn's diabetes table; noise rounded to one
decimal place; a whole number from 1 to 40) read as one the target follows,
and was warned about (the lead's real-data run of the eighth pass, finding
1). One point for each distinct value, the mean target of its rows, was
measured and not adopted: on large tables the means of an honest weak column
rise with it almost without exception (10.11).''')
sub('''   with each row's previous value in the column's place, over the training
   rows that have one:''', '''   with each row's previous value in the column's place, over the training
   rows that have one, and read in the same way:''')

# ------------------------------------------------------- step 12a, the statement --
sub('''**13. Assemble the result; write the files** (section 5).''',
    '''**12a. State what the headline rests on (D77 points 1 and 2, D78, D80, D81).**
On every finished run, for each feature, the chosen configuration's score on
the holdout with that feature's holdout values shuffled among the holdout
rows. **Nothing is refitted, nothing is chosen by it, and no other number of
the output changes.** It describes the result: the tool cannot know whether a
column is known before the outcome, and it can always say what the number
depends on (D78, "Alternatives considered"). Two columns that give the target
only together (`a` = the target plus a noise, `b` = that noise) cannot be
seen by any reading of single columns: such a table scored 1.000 with nothing
said (the seventh review, finding 2). Now the headline names both.

- **The shuffle.** The holdout matrix of step 10, as the chosen
  configuration was handed it (imputed, scaled, one-hot encoded), with that
  feature's part of it (its column, or its block of level columns) moved
  among the holdout rows by one permutation of the holdout rows; every other
  part stays where it is. The configuration predicts the shuffled matrix,
  and the prediction is scored with the headline metric against the holdout
  targets. **Five shuffles,** each with a permutation of its own; the
  feature's **`shuffled_score`** is the mean of the five scores. The
  permutations come from a stream of their own, drawn from the seed and from
  the position of the feature, that no other step draws from: the result is
  the same at any `n_jobs` (D17), and which permutations are drawn is open
  (11). **Each shuffle is one call of the fitted configuration's prediction on
  every holdout row,** in the holdout's order, handed as the holdout was for
  the headline: so a model supplied through the seam (2.6) is asked, after
  the headline's own call, five times for each set of copies, each time on
  the holdout rows. In which order the sets are taken is open.
- **Copies are shuffled together (D78 point 3, D80 point 4).** Features
  whose parts of the final fit's matrix are the same, in every training row
  and every holdout row, are copies of one another as the models receive
  them. Their parts are moved by one permutation, and they share one
  statement: the same three numbers in each record. Two numeric features are
  copies when their columns agree, or one agrees with the other's negation,
  within 1e-9; two categorical features when their blocks of level columns
  hold the same columns, in any order. Features of different kinds are never
  copies. So `2 × f + 3` and `−f` are copies of `f`, and a text column with
  its levels renamed is a copy of the column. A set of copies uses the stream
  of its first member in file order. Shuffled one at a time, each copy would
  show a part of the fall they share, because the other is still there: a
  rescaled copy beside its column showed a quarter of the column's fall each
  (10.11).
- **The three keys of a column's record (5.2).** `shuffled_score`;
  `shuffled_skill`, the skill of that score against the headline baseline,
  as step 11 computes skill (2 × AUC − 1, or 1 − score ÷ `baseline_score`);
  and `skill_fall`, `headline.skill` − `shuffled_skill`, which may be
  negative. All three are null for a column that reached no model. When
  `headline.skill` is null (D36), `shuffled_score` is a number and the other
  two are null. **When the configuration raises, or predicts a value that is
  not finite, for a shuffled matrix,** the three keys of that feature (and of
  its copies) are null, one `shuffled_score_not_computed` warning names each
  such feature, and the run goes on (D78 point 7): the headline was scored
  before, and nothing about it is in doubt.
- **The list, `headline.rests_on` (D78 point 1, D80 point 1).** The features
  whose `skill_fall` is at least the **floor**, at most five, the largest fall
  first, ties in file order, each as `{column, skill_fall}`. **The floor,
  `headline.rests_on_floor`, is the larger of 0.05 and the half-width of the
  headline skill's interval,** (`skill_high` − `skill_low`) ÷ 2: a fall that
  the headline's own uncertainty could give is not a statement about the
  result (D80). Where the skill has no interval (no interval at all, or
  `no_skill_interval`), the half-width is that of a stand-in: the middle 95%
  of the skill over a bootstrap of the holdout rows taken as if they were
  independent, with the resamples of 4.2, on a stream of its own, a resample
  in which skill cannot be computed being left out; with fewer than half of
  the resamples left, there is no stand-in and the floor is 0.05. The floor
  is null when `skill` is null, and the list is then empty. **The list has no
  threshold on the headline:** it is on every result, and the console
  summary prints it (2.2). With the floor of 0.05 alone it named columns on
  up to 21 of 30 seeds of a gate table of noise, and on 10 or more of 30 on
  seven of them; with the half-width, on at most 2 of 30 (D80; 10.11).
- **The near-perfect warning (D77 point 2).** When `headline.skill` is 0.98 or
  more, the level at which one column alone is excluded as a suspected leak,
  one `near_perfect_headline` warning names each column of the list, with its
  shuffled skill and its fall, and asks whether each is known before the
  outcome; with an empty list it says that no column's fall reaches the
  floor. In both tasks and under every scheme. It excludes nothing and
  changes no number: a legitimately easy table carries it too, and the
  caller is the one who knows whether each named column comes before the
  outcome.

What the statement cannot show, and what it then shows instead, is in
section 11 ("Known limits"). Its cost was measured at 100,000 rows by 100
columns under the full profile on all cores: 22 seconds of a run of 218
(classification) and of 244 (regression), against the budget of 600 (D16;
10.11).

**13. Assemble the result; write the files** (section 5).''')

# -------------------------------------------------------------- 4.2 sizes --
sub('''| resamples and shuffles | 1,000 (`full`), 200 (`fast`) |''',
    '''| resamples and shuffles | 1,000 (`full`), 200 (`fast`): the interval, the p-value, and the stand-in of step 12a's floor |
| shuffles of each feature, or set of copies, for the statement of step 12a | 5, on a stream of their own (D77, D78 point 7) |
| columns named in `headline.rests_on` | at most 5 (step 12a; D78 point 1) |''')
sub('''| the leak screen: pairs needed for a share against the order | at least 20 neighbouring pairs whose values of the column differ (step 8; D76) |''',
    '''| the leak screen: pairs needed for a share against the order | at least 20 pairs of neighbouring distinct values of the column, so at least 21 distinct values (step 8; D76, D77 point 4) |
| the leak screen: a numeric column read by its levels | at least 5 training rows, the column not missing, for each distinct value among them (step 8; D77 point 3) |
| the leak screen: rows the reading by levels must score to have a value | at least 20 pooled validation rows, and in a classification at least 10 of each class, or every one of that class (step 8; D78 point 5) |''')

# ------------------------------------------------------------ 4.3 cut-offs --
sub('''### 4.3 Cut-offs (D23, D28, D30, D33, D34, D48, D74, D76; measurements in section 10)''',
    '''### 4.3 Cut-offs (D23, D28, D30, D33, D34, D48, D74, D76, D77, D78, D80; measurements in section 10)''')
sub('''| leak warning, second reading: the share against the column's order, for a numeric column of a regression whose skill has a value below 0.90 | at most 0.10, and the same share for the previous value above 0.10 or without a value: kept, with the same warning (step 8; D76) |''',
    '''| leak warning, second reading: the share against the column's order, between rows of neighbouring distinct values, for a numeric column of a regression whose skill has a value below 0.90 | at most 0.10, and the same share for the previous value above 0.10 or without a value: kept, with the same warning (step 8; D76, D77 point 4) |
| the near-perfect warning | a headline skill of 0.98 or more (step 12a; D77 point 2) |
| a column named in `headline.rests_on` | a fall of at least 0.05 and at least the half-width of the headline skill's interval, or of its stand-in (step 12a; D80 point 1) |
| two numeric columns of the final fit's matrix that are copies | they agree, or one agrees with the other's negation, within 1e-9 in every training and holdout row (step 12a; D78 point 3, D80 point 4) |
| a time-ordered holdout that is named | half of the rows split or more: 2 × holdout rows ≥ training + gap + holdout rows (step 7; D77 point 6, D78 point 7) |''')

# ----------------------------------------------------------- 5.2 columns --
sub('''| `n_imputed` | integer | for a feature, its missing cells (as loaded, coerced or infinite) among the training and holdout rows; otherwise 0 |
''', '''| `n_imputed` | integer | for a feature, its missing cells (as loaded, coerced or infinite) among the training and holdout rows; otherwise 0 |
| `shuffled_score` | number or null | for a feature, the chosen configuration's score on the holdout with the feature's holdout values shuffled among the holdout rows, the mean of five shuffles; copies of one another share it (step 12a; D77, D78). Null for a column that reached no model, and for a feature that a `shuffled_score_not_computed` warning names |
| `shuffled_skill` | number or null | the skill of `shuffled_score` against the headline baseline, as step 11 computes skill. Null where `shuffled_score` is, and when `headline.skill` is null |
| `skill_fall` | number or null | `headline.skill` − `shuffled_skill`: how far the headline falls with the feature shuffled. It may be negative. Null where `shuffled_skill` is |
''')

# ---------------------------------------------------------- 5.2 headline --
sub('''| `baseline_largest_row` | integer or null | that row; null when the share is |
''', '''| `baseline_largest_row` | integer or null | that row; null when the share is |
| `rests_on` | list | what the headline rests on (step 12a; D78 point 1, D80): records `{column, skill_fall}` for the features whose `skill_fall` is at least `rests_on_floor`, at most five, largest first, ties in file order. Empty when no fall reaches the floor, and when `skill` is null |
| `rests_on_floor` | number or null | the larger of 0.05 and the half-width of the skill interval, or of its stand-in where the skill has no interval (step 12a; D80 point 1). Null exactly when `skill` is null |
''')

# ----------------------------------------------------------- 5.3 the rules --
sub('''16. In a regression under time order `baselines` holds both `training_mean`
    and `previous_value`, whatever `validation.timestamps_unique` and
    `validation.group_col` say (D74).
''', '''16. In a regression under time order `baselines` holds both `training_mean`
    and `previous_value`, whatever `validation.timestamps_unique` and
    `validation.group_col` say (D74).
17. In every record of `columns` whose `role` is not `feature`,
    `shuffled_score`, `shuffled_skill` and `skill_fall` are null. For a
    feature that no `shuffled_score_not_computed` warning names,
    `shuffled_score` is a number; `shuffled_skill` equals 2 ×
    `shuffled_score` − 1 (classification) or 1 − `shuffled_score` ÷
    `headline.baseline_score` (regression), and `skill_fall` equals
    `headline.skill` − `shuffled_skill`, both null when `headline.skill` is.
    A feature that such a warning names has all three null (D77, D78).
18. `headline.rests_on_floor` is null exactly when `headline.skill` is. It is
    otherwise at least 0.05, and equals the larger of 0.05 and (`skill_high`
    − `skill_low`) ÷ 2 when those two are numbers (D80).
19. `headline.rests_on` lists, as `{column, skill_fall}`, exactly the features
    whose `skill_fall` is at least `headline.rests_on_floor`, at most five, by
    `skill_fall` from the largest, ties by `position`; each `skill_fall`
    equals that of the column's record (D78, D80).
20. `near_perfect_headline` is among the warnings, once, exactly when
    `headline.skill` is 0.98 or more (D77).
21. `holdout_far_above_quarter` is among the warnings, once, exactly when
    `validation.scheme` is `time_ordered` and 2 × `rows.holdout` ≥
    `rows.training` + `rows.dropped_gap` + `rows.holdout` (D77, D78).
''')

# --------------------------------------------------------- 5.4 the texts --
sub('''> Skill is measured against the baseline NAME. With no entity column the previous value of a row is the mean target of the timestamp before, over every series at once: if the rows are several series, declare group_col, and each series is measured against its own previous value.
''', '''> Skill is measured against the baseline NAME. With no entity column the previous value of a row is the mean target of the timestamp before, over every series at once: if the rows are several series, declare group_col, and each series is measured against its own previous value.

Two more were added by the ninth pass (D77 point 6, D78 point 6), held in the
same way.

In the message of the `holdout_far_above_quarter` warning:

> The models were fitted on the rows before it and the headline is measured on it: fewer rows were learned from than the split usually leaves.

In the message of a `suspected_leak` warning for a column read by its levels
(step 8) and excluded by rule 2, after the reason:

> If it names the entity each row belongs to, declare it as group_col (--group-col): the rows of one entity are then kept together.
''')

# ------------------------------------------------------- section 7: edges --
sub('''| more than 1% of the training rows with a target wildly off | none is set aside and every row is scored: the screen as it was before D74, with the second reader |''',
    '''| more than 1% of the training rows with a target wildly off | none is set aside; rule 2 still leaves its rows unscored and is read against the fold's training median as well (step 8; D77 point 5, D78 point 7); rule 3 scores every row |
| the target cut into 16 or more bands, as text with labels in any order or as a numeric code (random scheme) | excluded as a suspected leak: read by its levels (step 8; D77 point 3); beside entities, under time order and with coarser bands, as section 11 says (D80 point 3) |
| a column of noise with repeated values (rounded to one decimal place; whole numbers from 1 to 40) | a feature with no warning: the order reading never puts rows of equal value in the target's order (step 8; D77 point 4) |
| a table whose headline skill is 0.98 or more | one `near_perfect_headline` warning naming the columns of `headline.rests_on` (step 12a; D77 point 2) |
| two features that are the same column as the models receive it (an exact, rescaled or negated copy; a text column with its levels renamed) | one statement for both: the same three keys, one fall (step 12a; D78 point 3, D80 point 4) |
| the chosen configuration raises on a shuffled matrix | `shuffled_score_not_computed` for each feature concerned, its three keys null; the run finishes (step 12a; D78 point 7) |
| under time order, a holdout of half of the rows split or more | `holdout_far_above_quarter` (step 7; D77 point 6) |''')

DST.write_bytes(t.encode("utf-8"))
print("wrote", DST)
