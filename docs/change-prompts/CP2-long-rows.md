# CP2 — phase 1 harness: rows longer than the header

**Session:** started from `E:\CODE\ml-professor` by
`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop\run_loop.py harness --open`,
on `fable`. You have no memory of any earlier session. Everything you need is
in the repo and in this file.

**If anything here is wrong** (a file is not as described, a change cannot be
made as written, a measurement disagrees with this file), say so plainly and
stop. Do not work round it.

## Context

- The phase 1 harness for ML Professor exists and its self-test passes: the
  gate `verify/smoke_test.py` (29 checks, 17 synthetic datasets), the
  reference implementation `verify/reference/mlprofessor/` with eight broken
  variants, `spec.md`, `plan/test-plan.md`, `plan/tasks.json` (17 tasks),
  `CLAUDE.md`. Decisions D1 to D35 are in `docs/DECISIONS.md`.
- The test suite exists and is committed: 17 files under `tests/`, 308 tests,
  all failing before the build, as they should. **The tests are locked and
  out of bounds for this session.** There is no application yet.
- Read first: `docs/DECISIONS.md` (D13, D22, D34, D35 matter most here),
  `spec.md` sections 6, 7 and 9, `verify/smoke_test.py`,
  `verify/reference/mlprofessor/core.py`, `plan/tasks.json`,
  `tests/test_load.py`.

## The finding

The owner ran the committed suite against the reference implementation, with
`PYTHONPATH` set to `E:\CODE\ml-professor\verify\reference`. Result: 304
passed, 4 failed. Three failures are expected, because the reference has no
`README.md` and no `requirements.txt`. The fourth is a defect in the
reference:

- `tests/test_load.py::test_bad_files_raise_input_error[long_rows]` gives the
  tool `a,b` as the header and `1,2,3` and `4,5,6` as the rows. The spec
  (section 6, D22) says data rows with more fields than the header are an
  input error. The test is right. The reference does not raise.
- Cause: the reference decides by the type of index pandas returns. Measured
  on pandas 3.1.1, `read_csv(path, low_memory=False)` on a file whose every
  row is one field longer than the header returns:

  | First column of the file | Index returned | Reference raises |
  |---|---|---|
  | uneven numbers (1, 4, 9) | `Index` | yes |
  | text (x, y) | `Index` | yes |
  | evenly spaced numbers (1, 4) | `RangeIndex` | **no** |
  | row numbers (1, 2, 3, 4) | `RangeIndex` | **no** |

- Consequence: a file exported with a row-number column that its header does
  not name is read with every column under its neighbour's name, and nothing
  is reported. That is the commonest real form of this fault.

Confirm the table on this machine before you rely on it.

## Before you write anything

Report, in a short list: that you reproduced the finding; the files and
sections you will change; how you will detect long rows; any conflict between
this file and the repo. Then wait for my go.

## The changes

No behaviour in the spec changes. The spec already says what must happen.

1. **The gate first.** Add a dataset: as `signal_clf`, with a row number from
   1 upward written as a leading extra field on every data row and no name
   for it in the header. The right conclusion: the tool exits non-zero naming
   the input error, and writes no headline. Add one check for it. Name F6 in
   `catches` if gatekit accepts a check the F6 variant does not fail, as
   `time.dayfirst_stops` does for F7; otherwise give it a written `why`.
   Add the dataset last, so that no existing dataset or seed moves.
2. **Watch the check fail.** Run the new check against the reference as it is
   now, before you fix it, and report that it fails. A check you have not
   watched fail is not a check.
3. **Then fix the reference.** It must raise the input error whenever a data
   row has more fields than the header, for all four forms in the table
   above. It must not depend on the type of index pandas returns. Report how
   your method scales: time it on a file of 100,000 rows by 100 columns.
4. **The spec.** In section 6 and in the Phase 0 facts, record that pandas
   returns a `RangeIndex` when the first column is evenly spaced, so the
   index cannot be used to detect this fault. This warns the builders; it
   pins no method.
5. **The task plan.** The new check belongs to the first task at which it can
   pass. Every check still belongs to exactly one task.
6. **`CLAUDE.md`:** paste the new output of `--matrix`.
7. **A small tidy-up in the reference.** The line that builds an array of
   `NaT` values raises a NumPy `DeprecationWarning` about the generic
   timedelta unit. Replace it with a form that does not warn. No value may
   change.

## Proof before you stop

1. The gate's self-test passes: the good fixture twice with identical values,
   and each of the eight failure modes caught by a check that names it.
2. No other gate value moved: compare every recorded field of every existing
   run at the gate seed before and after, and say so.
3. The committed suite, run against the fixed reference, fails exactly three
   tests: the two in `tests/test_readme.py` and
   `tests/test_skeleton.py::test_requirements_pin_the_four_libraries`. Run it
   without writing anything under `tests/` or `verify/`:
   set `PYTHONPATH` to `E:\CODE\ml-professor\verify\reference` and
   `PYTHONDONTWRITEBYTECODE` to `1` for that command only, then
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -m pytest -c tests/pytest.ini --rootdir . -q -p no:cacheprovider`.
   Paste its summary lines.
4. `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py doctor`
   prints `SELFTEST: PASS`, a per-edit gate time within the 20 second budget,
   and a valid plan of 17 tasks. A warning about uncommitted changes is
   expected.
5. The output of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe verify/smoke_test.py --matrix`
   and of
   `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe loop/run_loop.py build --dry-run`,
   pasted.
6. Any measurement script you wrote is saved under `docs/measurements/CP2/`
   with a short `README.md` (D35).
7. A final report: what changed in each file, and the timing from change 3.

## Out of bounds

- The `tests/` folder, in every form: no edits, no new files, no cache files.
- Kit files: `verify/gatekit.py`, `.claude/hooks/`, `.claude/settings.json`,
  `loop/`.
- Application code: do not create the `mlprofessor` folder at the repo root.
- `mlprof_env`, `BOOTSTRAP.md`, `docs/DECISIONS.md`.
- Git, in any form.
- Any band, seed or existing dataset.

## Stop condition

Give the final report and stop. Do not build.
