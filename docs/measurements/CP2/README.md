# CP2 measurements

The script behind the CP2 session (2026-11-16): data rows longer than the
header. Kept here by D35. The conclusions are in `spec.md` sections 6 and 9;
this folder is how to get the numbers again.

| File | What it is |
|---|---|
| `long_rows.py` | What `pandas.read_csv` and the reference's loader do with 39 crafted files; what two ways of finding a long row cost on files of up to 100,000 rows × 100 columns; every value of a full gate run, kept and compared between two states of the judge; four ways to build an array of `NaT`. |

Raw outputs are not kept. Every command prints its result and writes its
files outside the repo.

## Running a measurement again

From the repo root, with the project interpreter, one command at a time:

```
$env:MLPROF_LAB_WORK = "C:\somewhere\outside\the\repo"
$py = "E:\CODE\ml-professor\mlprof_env\Scripts\python.exe"
$W  = "$env:MLPROF_LAB_WORK\cp2"
```

`MLPROF_LAB_WORK` is where the generated files and results go; without it they
go to `mlprof-lab` in the system's temporary folder. In every command `-`
stands for the repo's own reference implementation or gate script; give a path
instead to measure another state of the judge.

| What it backs | Command | Takes |
|---|---|---|
| The table of the finding, and 35 more files | `& $py -B docs/measurements/CP2/long_rows.py forms -` | |
| The cost of finding a long row | `& $py -B docs/measurements/CP2/long_rows.py timing -` | 1 min; writes six files, 160 MB |
| The `NaT` line of the reference | `& $py -B docs/measurements/CP2/long_rows.py nat` | |
| Every value of a full gate run on a reference | `& $py -B docs/measurements/CP2/long_rows.py values - - "$W\after.json"` | 20 s |
| "No other gate value moved" | `& $py -B docs/measurements/CP2/long_rows.py compare "$W\before.json" "$W\after.json"` | |

`before.json` comes from the same `values` command given an earlier state:
`values <old>\verify\smoke_test.py <old>\verify\reference "$W\before.json"`.
`<old>` is a copy of the repo at an earlier commit, outside this folder; the
human makes it, since agents do not run git here. A copy of the gate script
needs no `gatekit.py` beside it: the repo's own is used.

`values` runs every check of the gate, with its artifacts under
`MLPROF_LAB_WORK` and not under `verify/artifacts`. For each run it keeps the
exit code, both output streams, the names of the files written, every field of
`profile.json` except `meta.timing`, and a hash of `predictions.csv`; for the
gate, each check's verdict and message, each value it recorded, and each
dataset's seed and a hash of its bytes. `compare` exits 1 if any value of the
earlier state moved or is gone.

## Seeds

- The gate's dataset `long_rows` is `signal_clf`'s generator at the base seed
  plus 17, its position in the gate's `DATASETS` (after `near_leak`, +16;
  the others are listed in `docs/measurements/CP1`). At the gate's base seed,
  1115, that is 1132. Each row is led by its number, from 1.
- The timing files come from `numpy.random.default_rng(7)`: 89 numeric
  columns, 10 text columns of six levels (one with a comma in it, so it is
  quoted), one yes/no column. The long-row form is the same table written with
  its row numbers from 1 and no name for them.

## What was found

pandas 3.1.1, numpy 2.5.4.

- **The finding.** Of 39 crafted files, 33 hold a data row longer than the
  header. The reference before CP2 loaded 18 of those 33 without a word; after
  it, none. All six well-formed files load in both.
- **What pandas does.** When the first data row is longer than the header,
  pandas makes an index of its leading fields and gives the names to the
  fields after them, with no error and no warning. The index is a `RangeIndex`
  when the leading field is evenly spaced whole numbers (1, 2, 3, 4; 1, 4;
  1, 4, 7; 3, 2, 1; 10, 20, 30) and an `Index` otherwise (1, 4, 9; 1, 2, 4;
  5, 5, 5; text; 0.5, 1.0, 1.5; a single row); two extra fields give a
  `MultiIndex`. Row numbers from 0 give the default index itself.
- **What is lost.** A leading extra field leaves every named column with its
  own values and is itself lost. A trailing one puts every name over its
  neighbour's values (`a,b` over `1,2,` gives a = 2, b = missing). One in the
  middle moves the names to its left only.
- **Later rows.** A row longer than the first data row makes pandas raise
  `ParserError`, whatever the first row's length. So the first data row is the
  only long row pandas reads silently. Before it, pandas skips an empty line
  and a line of only spaces or tabs; a line of two quotes or of only commas is
  a row.
- **The cost of looking** (seconds, best of 5):

  | File of 100 columns | `read_csv` | A: header and first data row | B: every row's fields | Loader, well formed | Loader, long rows |
  |---|---|---|---|---|---|
  | 1,000 rows, 0.7 MB | 0.005 | 0.0001 | 0.007 | 0.005 | 0.0003 |
  | 10,000 rows, 7.2 MB | 0.044 | 0.0001 | 0.072 | 0.045 | 0.0003 |
  | 100,000 rows, 72 MB | 0.46 | 0.0001 | 0.75 | 0.46 | 0.0003 |

  The reference uses A. It does not grow with the file. Before CP2 the
  loader took 0.46 seconds on both 100,000-row files and returned the
  long-row one as a table.
- **No other gate value moved.** Between the state before CP2 and after it, at
  the gate's seed: 17 datasets (seed, target, bytes), 29 checks (verdict,
  message), 91 recorded values and 22 runs (14,067 values) are identical.
  Only the dataset `long_rows`, its run, the check `data.long_rows_stop` and
  its recorded exit code are new.
- **`NaT`.** `np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")` raises
  a `DeprecationWarning` about the generic unit. With `np.datetime64("NaT",
  "ns")` the array has the same type and the same bytes, and nothing warns.

## Two things to know about these files

- The loop driver classifies anything under `docs/` as application code. Its
  build report lists application files that mention the gate, and this script
  does, so it will appear in that list. It is a measurement tool for the
  harness stage and no part of the package.
- For the same reason a build-stage agent is not prevented from editing it.
  If a number here ever disagrees with `spec.md`, run the command again.
