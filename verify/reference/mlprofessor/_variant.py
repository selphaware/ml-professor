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
         order is split at random, with a warning, where it should stop (D58)
    F8b  with the entity column declared under time order, the headline is
         still scored against the training mean alone (D57)
"M1" .. "M22" small faults that no gate check is relied on to catch (D39, D46,
              D52, D57). They exist for the test suite:
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
        goes ahead with a warning, where it should stop and ask (D58)
    M20 the infinite cells of the target and of the time column are left out
        of `n_infinite` in their column records (finding 3)
    M21 under the grouped scheme the bootstrap draws rows one by one and still
        reports `bootstrap_groups`
    M22 with `duplicates="group"` the copies are found and counted, and the
        split is made row by row all the same
"""
BROKEN = None
