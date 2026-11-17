"""Which named fault this copy carries. None is the known-good reference.

A fixture builder overwrites this file in its own copy with one value; every
other line of the package stays the same.

"F1" .. "F8"  the gate's failure modes (spec.md section 8). The gate's
              self-test proves a check catches each.
"M1" .. "M5"  small faults that no gate check can catch (D39). They exist for
              the test suite: verify/mutation_check.py proves a test fails on
              each.
    M1  the final fit's imputation and scaling statistics are learned from
        training and holdout rows together
    M2  one-hot levels are learned from every row of the table in every fit
    M3  the leak screen's trees are fitted on rows that include the holdout,
        and the missing-by-class rule counts every row
    M4  the block bootstrap draws timestamps independently and still reports
        `bootstrap_blocks`
    M5  the permutation p-value counts shuffles strictly greater than the
        observed skill
"""
BROKEN = None
