"""Which named fault this copy carries. None is the known-good reference.

A fixture builder overwrites this file in its own copy with one value; every
other line of the package stays the same.

"F1" .. "F10" the gate's failure modes (spec.md section 8). The gate's
              self-test proves a check catches each.
    F9   the rule on copies is never applied: a table whose holdout rows have
         copies among the training rows runs with no error (D49)
    F10  the entity column is declared and reported, and the rows are split
         one by one all the same: an entity has rows on both sides (D57)
"F6b" "F6c" "F6d" "F7b" "F7c" "F7d" "F8b"
              the other forms of F6, F7 and F8 (D46, D50, D57, D58), each with a
              gate fixture of its own, so that every gate check that names a
              failure mode is seen to fail on one. The test suite must catch
              them too.
    F6b  a column is lost from the accounting: a constant column is dropped
         with no record
    F6c  a model family is lost from the models table: only the family of the
         chosen configuration is listed
    F6d  data rows longer than the header are read silently, with no error
    F7b  a time column that is not ISO 8601 is parsed leniently, so a
         day-first date runs on the wrong order
    F7c  whole-number time keys are always read as 64-bit floats, so distinct
         instants above 2**53 become one timestamp (D50)
    F7d  a table with a date column and no word from the caller about time
         order is split at random, with a warning, where it should stop (D58).
         Since D63 every table stops there; this copy still lets through the
         one that holds dates
    F8b  with the entity column declared under time order, the headline is
         still scored against the training mean alone (D57)
"F11" .. "F14" the failure modes of the sixth pass (D62 to D66), each with a gate
              dataset or run and a check that names it.
    F11  a statement the caller did not make is assumed: with nothing said
         about time order, or about entities, the run goes ahead as if
         `False` had been passed (D63)
    F12  `duplicates="group"` forms its sets with the target: rows that agree
         in every feature and differ in class are kept apart, so a held-out
         set has only its opposite class among the training rows (D62)
    F13  the grouped holdout takes the units in seed order until they hold a
         quarter, so one large unit drawn early makes it far more (D64)
    F14  a column of dates written as whole numbers, YYYYMMDD, is a numeric
         feature and nothing says that the table holds dates (D66)
"M1" .. "M30" small faults that no gate check is relied on to catch (D39, D46,
              D52, D57, D62 to D66). They exist for the test suite:
              verify/mutation_check.py proves a test fails on each.
    M1  the final fit's imputation and scaling statistics are learned from
        training and holdout rows together
    M2  one-hot levels are learned from every row of the table in every fit
    M3  the leak screen's trees are fitted on rows that include the holdout,
        and the missing-by-class rule counts every row
    M4  the block bootstrap draws timestamps independently and still reports
        `bootstrap_blocks`
    M5  the permutation p-value counts shuffles strictly greater than the
        observed skill
    M6  a prediction that is not finite is scored and written as it is, where
        it should fail its configuration (D44)
    M7  the fast profile's own error is never raised: too few timestamps under
        the row cap is always the ordinary `too_few_rows` (D46)
    M8  the interval is the 5th to the 95th percentile of the resamples, and
        `interval_level` still says 0.95
    M9  the leak screen forgets the previous-value baseline, so the lag of a
        persistent series is excluded as a leak
    M10 the fast profile keeps the first 2,000 usable rows of the file, not a
        sample drawn from the seed
    M11 `cv_std` is the population deviation of the fold scores, not the
        sample deviation
    M12 in a resampled ROC AUC, tied predictions do not share their rank
    M13 `n_imputed` is counted over every loaded row, not over the rows that
        reach a model
    M14 an object column is read a second time, over the rows that will be
        used only: an odd cell in a row that is dropped for another column's
        empty cell is never seen (the fourth review, finding 1)
    M15 with an entity column, the holdout is whole entities but the folds are
        cut through them
    M16 under time order with an entity column, the leak screen forgets each
        entity's previous value, so the lag of a persistent panel is excluded
        as a leak
    M17 with an entity column, the fast profile's cap keeps a sample of rows,
        whatever their entity
    M18 the check that no number of the output is infinite or not-a-number
        covers five parts of the document, not every part (finding 4)
    M19 a panel with no declared entity column and an autocorrelated target
        goes ahead with a warning, where it should stop and ask (D58). Since
        D63: under a time column, nothing said about entities is read as
        `group_col=False`, whatever the table holds
    M20 the infinite cells of the target and of the time column are left out
        of `n_infinite` in their column records (finding 3)
    M21 under the grouped scheme the bootstrap draws rows one by one and still
        reports `bootstrap_groups`
    M22 with `duplicates="group"` the copies are found and counted, and the
        split is made row by row all the same
    M23 the panel warning's cut-off is 0.8, the value abandoned in D23, not 0.5
    M24 with both statements unsaid, the message asks for time order only, so
        the caller needs a second failed call to learn of the other (D63)
    M25 with `duplicates="group"` the sets are formed on the features, and the
        count of rows still reads the target: it counts exact copies (D62)
    M26 the grouped holdout is completed by the smallest unit passed over, and
        nothing it can spare goes back to the training rows (D64)
    M27 a unit of a quarter or more that is kept among the training rows gets
        no warning; only the one that is the holdout is named (D66)
    M28 dates written as eight digits of text are refused as the time column,
        though the same column is excluded as a date (D66)
    M29 the id-like rule is asked before the date-number rule: one row a day
        in date order is `id_like`, and nothing says the table holds dates
    M30 the count of step 8a is made on the features alone, like the sets of
        `duplicates="group"`: a holdout row counts as having a copy when a
        training row agrees in every feature, whatever its target (D62)
"""
BROKEN = None
