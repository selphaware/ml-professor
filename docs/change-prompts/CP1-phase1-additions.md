# CP1 — phase 1 harness: four additions before the tests are written

**Session:** started from `E:\CODE\ml-professor` by
`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop\run_loop.py harness --open`,
on `fable`. You have no memory of any earlier session. Everything you need is
in the repo and in this file.

**If anything here is wrong** (a file is not as described, a change cannot be
made as written, a measurement argues against a decision), say so plainly and
stop. Do not work round it.

## Context

- The phase 1 harness for ML Professor exists and its self-test passes: the
  gate `verify/smoke_test.py` (28 checks, 16 synthetic datasets), the
  reference implementation `verify/reference/mlprofessor/` with eight broken
  variants, `spec.md`, `plan/test-plan.md`, `plan/tasks.json` (17 tasks),
  `CLAUDE.md` and `.claude/skills/mlprofessor-build/SKILL.md`. There are no
  tests and no application yet.
- The owner reviewed the harness report and recorded D23 to D32 in
  `docs/DECISIONS.md`. D23 to D26 and D31 confirm what the harness already
  pins. **D27 to D30 each need a change. D32 is an environment change.**
- Read first, in this order: `docs/DECISIONS.md` (all of it), `spec.md`,
  `plan/tasks.json`, `plan/test-plan.md`, `verify/smoke_test.py`,
  `verify/reference/mlprofessor/`, `CLAUDE.md`. `BOOTSTRAP.md` is the
  original brief: background only. Where it differs from D23 onward, the
  decisions hold.

## Before you write anything

Report, in a short list: the files and sections you will change; any conflict
between this file and the repo; any change below you cannot make as
described. Then wait for my go.

## The changes, as a user would observe them

### 1. The time-ordered interval is labelled approximate (D27)

- Whenever `headline.interval_method` is `bootstrap_blocks`, `warnings` holds
  one record with the code `interval_approximate` and a null column. It is
  absent under the random scheme.
- The limitations statement gains this string, exactly:
  `Intervals are approximate: they tend to be too narrow when the holdout is short and, under time order, when errors are serially correlated.`
- The method of D9 does not change.

### 2. Numbers written with thousands separators (D28)

A *number with thousands separators* is: an optional sign, one to three
digits, one or more groups of a comma and exactly three digits, and an
optional decimal part after a point.

- Such a value is never parsed as a number.
- A text column that fails the numeric-coercion rule, but in which at least
  90% of the non-missing values are plain numbers or numbers with thousands
  separators, is excluded with the new reason `numeric_with_separators`. That
  reason takes precedence over `free_text` and over the id-like rule for text.
  A warning with the code `numeric_with_separators` names the column.
- A column that passes the coercion rule and is used as numeric, in which one
  or more of the cells coerced to missing are numbers with thousands
  separators, carries the same warning as well as `values_coerced`.

### 3. Negative skill is explained (D29)

The limitations statement gains this string, exactly:
`Negative skill means the models did worse than the baseline on the holdout; read it as no evidence of predictability.`

### 4. A warning tier below the leak cut-off (D30)

- After the exclusion rules of step 8, a feature column whose single-column
  cross-validated skill (the same measure, on the same folds, D24) is at
  least the warning cut-off and below 0.98 stays a feature, and a warning
  with the code `strong_single_column` names it.
- The warning cut-off starts at 0.90. Measure it and pin it (see below).

### 5. The venv base path (D32)

`mlprof_env\pyvenv.cfg` now has `home = C:\Users\uahma\anaconda3\envs\py312`.
Confirm with
`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -c "import sys; print(sys.base_prefix)"`.
Correct `CLAUDE.md` and the Phase 0 facts in `spec.md` wherever they describe
the `A:` drive. Then start eight Python processes at the same instant, for 30
rounds, and report how many rounds blocked. Leave the gate's 0.5 second
spacing and its single retry exactly as they are, whatever you find.

### 6. The record

Bring the decision index in `spec.md` up to D32, and cite D23 to D32 where
the spec restates them.

## The judge, updated together

Nothing above is done until the gate, the reference implementation, the spec,
the test plan and the task plan all agree.

- **Reference implementation:** implements changes 1 to 4. The eight broken
  variants stay broken in exactly one way each.
- **Change 1:** `uncertainty.reported` also asserts `interval_approximate` is
  present on its time-ordered run and absent on its random-scheme run. Any
  check that asserts the limitations strings is updated for the two new ones.
  Test plan: `tests/test_uncertainty.py` covers present and absent.
- **Change 2:** `messy` gains one column written entirely with thousands
  separators, built so that fewer than 90% of its values parse as plain
  numbers. Generate it from its own seeded stream, so that no existing column
  of `messy` changes. `data.columns_accounted` asserts its role, its reason
  and its warning. `data.coercion_reported` also asserts the
  `numeric_with_separators` warning on the existing mostly-numeric column.
  Test plan: `tests/test_columns.py` covers the pattern (matching and
  non-matching forms, including `1.200,50` and `12,34`, which do not match),
  the precedence over `free_text`, and both forms of the warning.
- **Change 3:** covered by the limitations assertion and
  `tests/test_edges.py`.
- **Change 4:**
  - A new dataset `near_leak`: as `signal_clf`, plus `outcome_noisy`, a copy
    of the target with 2% of the labels flipped. The right conclusion: the
    column has the role feature and the `strong_single_column` warning names
    it. Assert no score on this dataset.
  - A new check `leak.near_leak_warned`, `catches=["F2"]`. The F2 broken
    variant, which has no leak screen, must fail it. If it does not, say so
    and stop.
  - `leak.control_kept` also asserts that `strong_feature` carries no
    `strong_single_column` warning.
  - Measure over 30 data seeds, as before: the single-column skill of
    `outcome_noisy`, and the highest single-column skill of any legitimate
    column in any gate dataset. Pin the warning cut-off between the two
    ranges and report both ranges. If they overlap or nearly touch, change
    the dataset, not the cut-off, and say what you changed.
  - Test plan: `tests/test_leakage.py` covers warned-and-kept, both sides of
    each cut-off, and that the measure uses training rows only.
  - Task plan: `T12` gains `leak.near_leak_warned`. Every check and every
    planned test file still belongs to exactly one task.
- **Spec:** the warning codes, the reason values, step 8, the cut-off table,
  the fixed texts, the edges and the measured values are all updated, so the
  test author can write against them without seeing any code.
- **`CLAUDE.md`:** paste the new output of `--matrix`.

## Proof before you stop

1. The gate's self-test passes: the good fixture twice with identical values,
   each of the eight failure modes caught by a check that names it, and the
   F2 variant failing `leak.near_leak_warned`.
2. Measured ranges reported for: the warning cut-off of change 4; the skill
   band on `messy`, re-measured over 30 seeds; and any other band whose
   dataset or reference behaviour changed. A band is changed only if its
   measurement changed.
3. `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py doctor`
   prints `SELFTEST: PASS`, a per-edit gate time within the 20 second budget,
   and a valid plan of 17 tasks. A warning about uncommitted changes is
   expected.
4. The output of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/smoke_test.py --matrix`
   and of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py build --dry-run`,
   pasted.
5. A final report: what changed in each file; every measured value; the
   result of the simultaneous-start measurement; and every place where the
   facts argue against D27 to D30.

## Out of bounds

- Kit files: `verify/gatekit.py`, `.claude/hooks/`, `.claude/settings.json`,
  `loop/`, `tests/pytest.ini`.
- The `tests/` folder: write no tests.
- Application code: do not create the `mlprofessor` folder at the repo root.
- `mlprof_env`, `BOOTSTRAP.md`, `docs/DECISIONS.md`.
- Git, in any form.
- The methodology: flag what the facts argue against, and stop.
- Any band, seed or fixture, unless its own measurement changed.

## Stop condition

Give the final report and stop. Do not write tests. Do not build.
