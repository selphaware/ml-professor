"""Which named fault this copy carries. None is the known-good reference.

A fixture builder overwrites this file in its own copy with one value; every
other line of the package stays the same.

"F1" .. "F8"  the gate's failure modes (spec.md section 8). The gate's
              self-test proves a check catches each.
"F6b" "F6c" "F6d" "F7b"
              the other forms of F6 and F7 (D46), each with a gate fixture of
              its own, so that every gate check that names a failure mode is
              seen to fail on one. The test suite must catch them too.
    F6b  a column is lost from the accounting: a constant column is dropped
         with no record
    F6c  a model family is lost from the models table: only the family of the
         chosen configuration is listed
    F6d  data rows longer than the header are read silently, with no error
    F7b  a time column that is not ISO 8601 is parsed leniently, so a
         day-first date runs on the wrong order
"M1" .. "M7"  small faults that no gate check can catch (D39, D46). They exist
              for the test suite: verify/mutation_check.py proves a test fails
              on each.
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
"""
BROKEN = None
