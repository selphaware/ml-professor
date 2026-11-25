VERDICT: FIX FIRST

The build does what `spec.md` says, and every number I recomputed from source data matched. One finding blocks shipping: the leak screen cannot read a column whose values are unordered codes of the target, and such a column often stays a feature with no warning while the headline goes to 0.95–1.0. The code follows the spec here (`spec.md:1426`), so the fix starts with a decision, not a patch.

| # | Severity | Type | Evidence (file:line) | Proposed remedy |
|---|---|---|---|---|
| 1 | High | spec-gap, gate-gap | `spec.md:1403-1426` (a regression's categorical column is read by the tree alone); `mlprofessor/leakage.py:25`, `:56-68`, `:357-364`; `mlprofessor/preprocess.py:18`; `verify/battery.py:1087-1091` (every copy of a regression target is numeric); not among the limits in `spec.md:5095-5219` | A decision first: a reader that can read codes, or a stated limit with these numbers. Then a battery change (banded copy, labels in random order, text and numeric, both tasks), a gate check on `leak_reg` with its own broken copy, a guard test with a mutation variant, and a task that makes them pass. |
| 2 | Medium | judgement-call | `mlprofessor/pipeline.py:1084-1114` (one column at a time); `mlprofessor/result.py:38` (limitation 5) | Decide whether a headline near 1 should carry a warning. If yes: a warning code, a gate check on a two-column table, and a task. If no: add it to section 11 with the measurement. |
| 3 | Low | judgement-call | `spec.md:1328`; `mlprofessor/pipeline.py:968-978` | Decide whether a time-ordered holdout far above a quarter is named, as `large_group_is_holdout` names it under the grouped scheme. If yes: a row in section 7, a test, and a task. |
| 4 | Low | test-gap | `tests/test_judge_tools.py:48-67` | In the next harness session, make this file strict again. It still accepts exit 1 for pairs the eighth pass owned; only `tests/test_judge_tools_pass8.py` holds the line today. |

## Finding 1: unordered codes of the target

The screen reads a column with a tree of depth 6 and a minimum leaf of 5, on the column's numbers or on codes of its sorted level names. When the values are codes in no order (a band, a tier, a status code), that tree cannot separate more than about 16 of them. The models can: one-hot encoding keeps up to 30 levels, and the forest and boosting read numeric codes.

Each row below is a column that bands or encodes the target, on tables of 1,000 rows under the full profile:

| Column added | Tables | Excluded | Warned | Silent | Headline skill where kept | Same tables without it |
|---|---|---|---|---|---|---|
| Regression, text, 24 bands, labels in random order | 10 | 1 | 5 | 4 | 0.981 to 0.993 | 0.78 to 0.814 |
| Regression, text, 30 bands, labels in random order | 10 | 0 | 4 | 6 | 0.985 to 0.995 | 0.778 to 0.814 |
| Regression, text, 24 or 30 bands, labels that sort in the target's order | 20 | 20 | 0 | 0 | none kept | |
| Regression, numeric code, 24 bands | 6 | 1 | 2 | 3 | 0.947 to 0.973 | 0.788 to 0.822 |
| Classification, numeric code, 40 codes each of one class | 6 | 1 | 2 | 3 | 0.999 to 1.0 | 0.579 to 0.761 |

- **It depends on label spelling.** The same banded column is excluded on every table when its labels sort in the target's order, and on 1 of 20 when they do not.
- **On the gate's own table.** `leak_reg` (1,500 rows) with `amount_scaled` removed and a 24-band text column added goes from 0.8218 to 0.9887 under the fast profile and 0.9912 under the full, with no exclusion and no warning.
- **Not only small tables.** At 2,000 rows a 30-band text column was kept in silence at 0.9907, and a 40-band one at 0.9114 against 0.7864.
- **Never measured.** Nothing under `docs/measurements/` covers a categorical copy of a regression target, and `spec.md` section 11 says nothing is open.
- **One indication for the planner.** A per-level training mean read the text bands at 0.98 to 0.996 on five tables, and an honest 24-segment column at 0.71 on one. That is one table for the honest case; wrongly excluded columns need counting before any rule is proposed.

This is F2 in a form no gate check, test or battery pair can see. Classification with a text column is already covered by the direct reader (excluded on the one table I tried).

## Findings 2 to 4

- **Two columns that give the target together.** With `a = y + z` and `b = z`, the run reports skill 1.0000 with `linear-1` and no warning of any kind. The screen is single-column by design, and limitation 5 covers it only loosely.
- **A time-ordered holdout of 60%.** With 60% of the rows on the last timestamp, the holdout is those 600 rows and training is 400. That conforms to the spec; the only warning is `no_interval`.
- **The loosened judge-tools test.** Both tools pass with nothing open, so the loosening hides nothing today.

## What held up

- **Judge-awareness: none found.** Nothing in `mlprofessor/` mentions tests, the gate or fixture values, or reads the environment. The two `except Exception` blocks (`pipeline.py:685`, `:1170`) record a `model_failed` warning and let every `ProfileError` through, as the spec has it. Every flagged line in the build report is a lab script under `docs/measurements/`, not application code.
- **Numbers reconcile.** Recomputed from the source and matched:
  - ROC AUC and skill;
  - squared error and the training-mean baseline;
  - the previous value at unique timestamps on a shuffled file;
  - the mean of the timestamp four before, under `gap=3` with three rows a timestamp, and its 9 gap rows;
  - each entity's own previous value in a panel;
  - the row counts and `n_imputed`.
- **Index labels play no part.** Permuted and duplicate index labels give the same headline and predictions as the default index.
- **Odd inputs end in named errors.** Sixteen cases (gap without a time column, duplicate column names, non-text target name, empty table, constant target, three classes, empty entity cells and others) each ended in a result or a named error.
- **Stated limits behave as stated.** Under time order a curve of the target and a copy with 0.5% noise were warned about, and a negative-scale copy was excluded.

## What I ran

- Full gate: PASS, 53 checks.
- Full suite: 2140 passed in 11:36, including both judge-tools tests.
- Gate self-test: PASS, 51 of 51 pairs seen failing over 37 broken fixtures.
- Mutation check on `F2,F24,M47` only: the good reference fails 0 of 2119; F2 fails 79 tests, F24 29, M47 3.
- About 150 in-memory `profile()` runs on my own tables, and the gate's `leak_reg` data with a column added.

## What I could not verify

- The other 85 mutation variants. So for most failure modes the test side rests on the owner's last full run, not on mine; the gate side is proved by the self-test.
- The battery and the generated tables as separate runs; I saw them pass only through the suite.
- The CLI end to end. `--out` is required and I may not write files, so I read `cli.py` and relied on the gate's runs; the README commands were not run.
- `columns.py`, `load.py`, `split.py`, `task.py` and `models.py` were not read line by line.
- Run time at 100,000 rows.
