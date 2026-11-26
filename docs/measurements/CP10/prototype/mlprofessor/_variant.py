"""Which named fault this copy carries. None is the known-good reference.

A fixture builder overwrites this file in its own copy with one value; every
other line of the package stays the same.

Eighty-eight faults in all: thirty-seven the gate's self-test proves a check
catches (F1 to F28, with the forms F6b, F6c, F6d, F7b, F7c, F7d, F8b, F12b and
F14b), and fifty-one that only the test suite is relied on for (M1 to M51).

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
"F12b" "F14b" "F15"
              the failure modes of the seventh pass (D69, D70, D72): the other
              forms of F12 and F14 that the sixth review found by adding one
              column and changing one cell, and the fast profile's cap.
    F12b with `duplicates="group"`, when the leak screen has excluded a
         column, only the rows that are copies, target and all, are counted:
         combinations of the features that are left are on both sides of
         the split and the run goes ahead (D69)
    F14b every value of a column of dates written YYYYMMDD must read as a
         date, as built in the sixth pass: one sentinel cell, and the column
         is a numeric feature again (D69, D72)
    F15  with an entity column the fast profile's cap takes the entities in
         an order drawn from the seed, whatever their size: one larger than
         the cap, or one of a quarter or more of the rows that is drawn
         late, is left out, and only the row count says so (D70, D72)
"F16" .. "F22" one broken copy for each of the seven gate checks that gave a
              reason in place of a failure mode and had never been seen to
              fail (D64, D71). Each is the fault its own check's message
              tells a builder to look for.
    F16  the positive class is inverted: the probability handed on, scored
         and written is the other class's, so skill that is there is
         reported as far below zero
    F17  squared error is read as higher-is-better, so the configuration
         with the worst cross-validation score is chosen
    F18  a column of dates under `time_col=False` gets no
         `date_column_undeclared` warning
    F19  the result object is not the file: `ProfileResult.headline` holds
         rounded numbers
    F20  a part of the output is missing: the limitations statement is empty
    F21  a clock value is written outside `meta.timing`, so two identical
         runs differ
    F22  the resamples are drawn from a stream that depends on the number of
         cores, so the interval differs between `n_jobs` 1 and 4
"F23" .. "F28" the failure modes of the eighth pass (D73, D74, D76): what the
              judge's two other tools found on their first run, each with a gate
              dataset or run and a check that names it.
    F23  under time order with no entity column, the previous-value baseline is
         scored only when the timestamps are unique, as it was: one repeated
         row leaves the training mean as the only baseline, and a random walk
         looks almost perfectly predictable
    F24  the leak screen reads a column with its tree alone: a rescaled copy
         of a trending target is kept under time order, where a tree cannot
         follow the target beyond the range it was fitted on
    F25  no training row is set aside from the leak screen and every row is
         scored: one extreme target cell switches the screen off, and a copy of
         the target stays a feature
    F26  the largest row's share is taken of the absolute errors, not of the
         squared errors: the number reported is not the share of the score
    F27  the id-like rule for whole numbers is D19's clause alone: an
         identifier becomes a feature when the rows are shuffled
    F28  under the grouped scheme the choice between configurations is made
         however few rows the folds outside one large unit validate on
"M1" .. "M51" small faults that no gate check is relied on to catch (D39, D46,
              D52, D57, D62 to D66, D69 to D76). They exist for the test
              suite: verify/mutation_check.py proves a test fails on each.
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
    M31 the date rule counts cells, not distinct values: 95% of the cells
        must read as dates, so a sentinel in a third of the rows makes the
        column a number again (D69, D72)
    M32 as the time column, a value that is no date among dates written
        YYYYMMDD is ordered as the number it is: no stop (D69)
    M33 the stop under `duplicates="group"` after a removed leak keeps the
        threshold of the copies rule, more than 1% and at least ten rows:
        a few combinations on both sides pass (D72)
    M34 the fast profile's stop for a large entity is raised where the full
        profile would stop too, in place of the full profile's own error
    M35 only an entity of more rows than the cap stops the fast profile; one
        of a quarter or more of the rows that fits is left to the draw
        (D70 as first worded; D72)
    M36 the `date_column_undeclared` warning carries the caveat of the
        random scheme under the grouped scheme too (the sixth review,
        finding 5)
    M37 the `same_features_grouped` warning does not say what
        `duplicates="group"` measures (D69)
    M38 one `fold_not_scored` record for the run, however many folds were
        not scored (D72)
    M39 a row with a cell that cannot be hashed is never counted as a repeat:
        `input.duplicate_rows` is 0 on such a table (D72)
    M40 at repeated timestamps the previous value is the target of the last
        row of the timestamp before, not the mean of its rows (D74)
    M41 the screen's line is fitted by least squares: one cell far off in a
        copy of the target turns it, and the copy is kept under time order
    M42 the rows the screen sets aside are left out of its sums only: they
        stay among each fold's training rows, so one target cell of 999999
        still shifts every fold's mean and every column is excluded
    M43 the order reading of the warning tier has no second condition: the
        lag of a series that only rises is warned about (D76)
    M44 the order reading excludes the column, where it may only warn (D76)
    M45 over the selection folds the largest row's share is reported for the
        chosen configuration only: a cell that ruins another configuration,
        and with it the choice, is not named (D74)
    M46 the extreme-value warning compares the gap with the whole range of
        the other values, not with the spread of the ten values after it: -999
        in a column of about 50, spread 10, is not named
    M47 in a classification the screen reads a column with its tree alone: an
        exact copy of the class is kept when a fold's training rows hold fewer
        than five rows of one class
    M48 the no-choice rule of the grouped scheme asks only for a fold of fewer
        than 50 training rows, whatever it validates on: a small table with
        even folds gets no choice either (D74)
    M49 under time order the rows of a fit are handed over in file order, not
        in time order (D74 point 4)
    M50 the identifier's second clause asks for every value to be distinct:
        one repeated row makes a shuffled identifier a feature (D74)
    M51 the panel warning names the training mean whatever baseline the run
        used (D74)
"""
BROKEN = None
