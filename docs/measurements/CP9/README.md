# CP9: the measurements behind the eighth pass (D73, D74, D76)

The scripts and the values behind spec section 10.10, sections 8.2 and 11 and
the plan of the pass (D35). Measured from 2026-11-23 to 2026-11-25 in the
harness session of CP9, on this machine (Threadripper PRO 5975WX, Python
3.12.13, pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1). Nothing here is part
of the tool, the gate or the suite. No script writes inside the repo: lab
copies and work files go to `MLPROF_LAB_WORK`, or to `mlprof-lab` in the
system's temporary folder, and the JSON a script writes goes to the path its
command names, which is outside the repo (`<out>` below). Every table below is
the text a script printed; the files are kept in `output\`, `battery\` and
`generated_tables\`.

Four packages are measured:

- **the application**: `mlprofessor` at the repo root, as built by the seventh
  pass (commit `388ea71`);
- **the reference**: `verify\reference\mlprofessor`, as changed by this pass;
- **the reference as it stood**: the same folder as committed at `388ea71`,
  from a copy of `verify\` taken before the first edit of the pass and kept
  outside the repo (`<old reference>` below);
- **the prototype**: `prototype\mlprofessor` in this folder, a copy of the
  reference as it stood with one environment switch for each rule of the pass
  (`_proto.py` lists them), on which every rule was measured before it was
  written. Its defaults are the rules adopted. It is the judge's material,
  kept so that the measurements can be made again: the guardrail on
  `verify\reference\` applies to it, and it is not a starting point for
  anything. Section 0 shows that it and the reference give the same output.

Run each command from `E:\CODE\ml-professor`.

## Seeds

- **Data seed *k*** of a gate dataset: the gate's base seed 1115 + 100 x *k*,
  *k* = 0 to 29 (*k* = 0 is the gate's own data), each dataset at its own
  position, as `verify\smoke_test.py` makes them. The measurements are of the
  runs the gate had before this pass: the five runs the pass adds each hold
  something planted, and `lab9.ADDED` leaves them out.
- **Tool seed**: 0.
- **A change to a gate table** (a repeated row, an added copy, one cell set to
  -999 or 999999) is drawn from `random.Random("cp9.<run>.<change>.<k>.<draw>")`:
  the same table on every run. Where a cell is planted in three different rows
  of one table, `draw` is 0, 1 and 2.
- **Tables made by a script**: from `numpy.random.default_rng(...)` with the
  seed its docstring gives: `[9700, rows, seed]` for the long tails of
  `extreme.py`, `[9900, rows, 1000 x R-squared, seed, shape]` for
  `order_strong.py`, `[9800, seed]` for the running total of
  `order_reading.py`.
- **The battery** changes each gate table from `random.Random` seeded with the
  names of the run and the change. **The generated tables**: table *i* of seed
  20261122 (the default), and of seeds 1 and 2.

## The scripts

`P` marks a script that reads the prototype's switches and so runs on the
prototype only; the others run on any of the packages.

| Script | What it measures | Command (after `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP9\`) |
|---|---|---|
| `lab9.py` | the lab: a pool of processes that import one package and run it on the gate's tables, clean or with one change | (imported by the others) |
| `measure.py` | the runs of sections 1, 2, 3, 5 and 6, written as JSON | `measure.py <a1, a2, a3, a6 or gate_new> <package folder> <out>\<name>.json [--seeds 30] [--workers 28]` |
| `per_seed.py` | the per-seed tables of those runs, and the comparison of two packages field by field | `per_seed.py <a1, a2, a6 or gate> <new.json> <before.json>`; `per_seed.py a3 <new.json>`; `per_seed.py same <one.json> <other.json>` |
| `copy_summary.py` | section 2: what became of the added copy, counted over the seeds | `copy_summary.py <a2.json>` |
| `noise_scale.py` P | section 2: how small the noise on a copy must be under time order | `noise_scale.py docs\measurements\CP9\prototype\mlprofessor` |
| `order_reading.py` P | section 2: the order reading of D76, with and without it, and its controls | `order_reading.py docs\measurements\CP9\prototype\mlprofessor` |
| `order_strong.py` P | section 2: the order reading on a legitimately strong feature | `order_strong.py docs\measurements\CP9\prototype\mlprofessor --per-seed` |
| `a3b_defeat.py` P | section 2: how many extreme target cells the rows set aside can absorb | `a3b_defeat.py docs\measurements\CP9\prototype\mlprofessor` |
| `a3_share.py` P | section 3: the largest row's share on clean tables, by size and by tail | `a3_share.py docs\measurements\CP9\prototype\mlprofessor` |
| `extreme.py` P | section 3: the cut-off of `extreme_value`, at 20, 25, 30, 40 and 50 | `extreme.py docs\measurements\CP9\prototype\mlprofessor` |
| `visits.py` P | section 3: the one planted cell that is not named | `visits.py docs\measurements\CP9\prototype\mlprofessor` |
| `a5.py` P | section 4: the id-like rule on measures and on identifiers | `a5.py docs\measurements\CP9\prototype\mlprofessor` |
| `a5_messy.py` P | section 4: `messy` shuffled and with rows repeated | `a5_messy.py docs\measurements\CP9\prototype\mlprofessor` |
| `a6_cut.py` P | section 5: on how few rows the choice of model fails | `a6_cut.py docs\measurements\CP9\prototype\mlprofessor` |
| `a6_screen.py` P | section 5: what the leak screen rests on beside a dominant entity | `a6_screen.py docs\measurements\CP9\prototype\mlprofessor` |
| `edges.py` | section 2: where the edges of the two leak cut-offs are (spec 10.2), on the tables of the first pass | `edges.py verify\reference\mlprofessor --per-seed` |
| `same_verdicts.py` | section 9: whether runs of the battery give the same verdict on every pair | `same_verdicts.py <label>=<a --json file of the battery> ...` |

`prototype_tools\` holds the scripts that made the prototypes of the two
judge's tools from the tools as they stood (`patch_battery.py` then
`patch_battery2.py`; `patch_judge.py` after `update_patch_judge.py`). The
tools in `verify\` are what they became; the prototypes' last runs are in
`battery\` and `generated_tables\` (sections 9 and 10).

The JSON files behind sections 1 to 6 were made with:

```text
measure.py a1       verify\reference\mlprofessor <out>\a1_ref.json
measure.py a1       <old reference>              <out>\a1_before.json
measure.py a6       verify\reference\mlprofessor <out>\a6_ref.json
measure.py a6       <old reference>              <out>\a6_before.json
measure.py a2       verify\reference\mlprofessor <out>\a2_ref.json
measure.py a2       <old reference>              <out>\a2_before.json
measure.py a3       docs\measurements\CP9\prototype\mlprofessor <out>\a3.json
measure.py a3       verify\reference\mlprofessor <out>\a3_ref.json
measure.py gate_new verify\reference\mlprofessor <out>\gate_new_ref.json
measure.py gate_new <old reference>              <out>\gate_new_before.json
```

`a3.json` is made on the prototype because the table of section 3 needs to
know which rows were held out, which only the prototype records.

## 0. The prototype and the reference give the same output

`measure.py` was run on the prototype and on the reference for all five sets of
runs (for `a2` the prototype with `--env MLP_VARIANTS=0`, see below), and
`per_seed.py same <prototype's json> <reference's json>` compared eighteen
fields of every record: the error, skill, score, model, baseline and its score,
scheme, every baseline, every column's role, reason and leak rule, every
warning, the row counts, the folds, every configuration's mean, and the four
largest-row values of the headline.

```text
## a1: the prototype against the reference of this pass
records compared: 510 (the other file holds 510); fields compared in each: 18; fields that differ: 0
## a6: the prototype against the reference of this pass
records compared: 150 (the other file holds 150); fields compared in each: 18; fields that differ: 0
## a2: the prototype against the reference of this pass
records compared: 4380 (the other file holds 4380); fields compared in each: 18; fields that differ: 0
## a3: the prototype against the reference of this pass
records compared: 4230 (the other file holds 4230); fields compared in each: 18; fields that differ: 0
## gate_new: the prototype against the reference of this pass
records compared: 1110 (the other file holds 1110); fields compared in each: 18; fields that differ: 0
```

10,380 runs, no field differs. Three things to know when reading it:

- **A measuring switch of the prototype changes what it measures.** With
  `MLP_VARIANTS=1` the prototype reads each column under every variant of the
  screen in one run, and in doing so it overwrote the value the warning tier
  reads with the tree's alone. The exclusions were not affected; the count of
  `strong_single_column` warnings was, on 34 of the 4,380 runs of `a2`
  (`dominant_entity` with noise of 10% and 20% on the copy, one seed of
  `random_walk` with 5%): the rule warns there and the switch hid it. So every
  table of sections 1 to 6 that counts warnings is made on the reference
  itself, and the comparison above uses the prototype without the switch.
- **After the comparison the reference changed in three places, none of which
  the prototype has.** The second reading of the warning tier is taken only of
  a column whose first reading has a value (D36: found when the existing suite
  was run against the reference, where
  `tests\test_skill_undefined.py::test_single_column_skill_that_cannot_be_computed_flags_nothing`
  failed); the runs of `a2` and `a3` on the reference were made after that
  change, those of `a1`, `a6` and the gate were made again after it. The
  largest row's share is computed with the errors divided by the largest
  first, so that it cannot overflow (D44): a share moves by a unit in the last
  place at most. The message of `selection_not_made` lost the words "the first
  configuration is used". The last two came after the comparison.
- **Checksums.** The prototype's `core.py` has SHA-256 `96e15abe...6cdb12` and
  its `_proto.py` `dc1f60b4...`.

## 1. The previous value at repeated timestamps (A1; D74 point 1)

`random_walk`, `int_stamps` and `flat_tail` as they are, with one row written
twice, with 1% and with 10% of the rows written twice; `panel` under
`--no-entities` as it is, with one row repeated and with a rescaled copy of the
target added; `time_blocks` (a classification) as a control. For each: the
skill seed by seed, before and now; which baseline the headline used, of those
listed; and on how many seeds the panel warning is raised.

```text
random_walk, (as it is)
  before  -0.0998 to +0.0118
    +0.001 -0.001 -0.011 -0.004 +0.005 +0.001 -0.003 +0.012 -0.027 +0.002 +0.002 +0.002 +0.002 -0.036 +0.004 +0.001 -0.018 -0.100 +0.003 +0.002 +0.004 +0.003 +0.001 +0.003 -0.005 -0.021 -0.001 -0.003 +0.004 -0.011
  new     -0.0998 to +0.0118
    +0.001 -0.001 -0.011 -0.004 +0.005 +0.001 -0.003 +0.012 -0.027 +0.002 +0.002 +0.002 +0.002 -0.036 +0.004 +0.001 -0.018 -0.100 +0.003 +0.002 +0.004 +0.003 +0.001 +0.003 -0.005 -0.021 -0.001 -0.003 +0.004 -0.011
  before: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 0 of 30

random_walk, row_repeated
  before  +0.9393 to +0.9996; 1 stopped
    +0.999 +0.998 +0.999 +0.998 +0.977 +0.999 +0.990 +0.939 +1.000 +0.996 no_features +0.996 +0.977 +0.997 +0.961 +0.998 +0.997 +0.999 +0.978 +0.999 +0.985 +0.989 +0.995 +0.997 +0.995 +0.995 +0.998 +0.999 +0.968 +0.994
  new     -0.1012 to +0.0118
    +0.001 +0.001 -0.014 -0.004 +0.004 +0.001 -0.004 +0.012 -0.026 +0.002 +0.002 +0.003 +0.002 -0.040 +0.004 +0.001 -0.018 -0.101 +0.003 +0.002 +0.004 +0.002 +0.002 +0.002 -0.005 -0.020 -0.001 -0.003 +0.004 -0.011
  before: headline baseline, of those listed: training_mean of training_mean on 29
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 30 of 30

random_walk, rows_repeated_1pct
  before  +0.9399 to +0.9996; 1 stopped
    +0.999 +0.998 +0.999 +0.998 +0.977 +0.999 +0.990 +0.940 +1.000 +0.996 no_features +0.996 +0.977 +0.997 +0.962 +0.998 +0.997 +0.999 +0.978 +0.999 +0.985 +0.990 +0.995 +0.997 +0.995 +0.995 +0.998 +0.999 +0.968 +0.994
  new     -0.0845 to +0.0120
    +0.001 -0.001 -0.013 -0.005 +0.004 +0.001 -0.002 +0.012 -0.017 +0.002 +0.002 +0.003 +0.002 -0.035 +0.004 +0.001 -0.019 -0.085 +0.004 +0.002 +0.004 +0.002 -0.000 +0.002 -0.006 -0.017 -0.001 -0.002 +0.004 -0.011
  before: headline baseline, of those listed: training_mean of training_mean on 29
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 30 of 30

random_walk, rows_repeated_10pct
  before  +0.9402 to +0.9996; 1 stopped
    +0.999 +0.998 +0.999 +0.998 +0.978 +0.999 +0.990 +0.940 +1.000 +0.996 no_features +0.996 +0.978 +0.997 +0.961 +0.998 +0.997 +0.999 +0.975 +0.999 +0.985 +0.989 +0.995 +0.997 +0.995 +0.995 +0.998 +0.999 +0.968 +0.994
  new     -0.0956 to +0.0094
    +0.001 -0.003 -0.010 -0.000 +0.006 -0.001 -0.009 +0.009 -0.021 +0.003 +0.001 +0.003 +0.001 -0.036 +0.005 -0.002 -0.033 -0.096 +0.005 +0.003 +0.005 +0.001 +0.005 +0.005 -0.002 -0.017 -0.001 -0.004 +0.003 -0.008
  before: headline baseline, of those listed: training_mean of training_mean on 29
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 30 of 30

int_stamps, (as it is)
  before  -0.1823 to +0.0052
    +0.002 -0.016 +0.005 +0.002 -0.182 +0.001 +0.002 +0.003 +0.003 -0.001 -0.001 -0.081 -0.038 -0.030 -0.011 -0.004 -0.005 -0.008 -0.016 -0.014 -0.176 +0.001 -0.038 -0.014 -0.000 -0.000 -0.001 +0.001 -0.003 -0.020
  new     -0.1823 to +0.0052
    +0.002 -0.016 +0.005 +0.002 -0.182 +0.001 +0.002 +0.003 +0.003 -0.001 -0.001 -0.081 -0.038 -0.030 -0.011 -0.004 -0.005 -0.008 -0.016 -0.014 -0.176 +0.001 -0.038 -0.014 -0.000 -0.000 -0.001 +0.001 -0.003 -0.020
  before: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 0 of 30

int_stamps, row_repeated
  before  +0.9599 to +0.9996
    +0.995 +1.000 +0.960 +0.995 +0.999 +0.995 +0.998 +0.998 +0.994 +0.992 +0.985 +0.999 +0.999 +0.997 +0.996 +0.991 +0.999 +0.999 +0.996 +0.999 +0.998 +0.996 +0.996 +0.995 +0.996 +0.993 +0.999 +0.986 +0.994 +0.999
  new     -0.1899 to +0.0052
    +0.003 -0.016 +0.005 +0.002 -0.190 +0.001 +0.003 +0.002 +0.003 -0.001 -0.001 -0.081 -0.040 -0.030 -0.012 -0.004 -0.005 -0.008 -0.016 -0.012 -0.173 +0.001 -0.038 -0.021 -0.000 +0.000 -0.001 +0.001 -0.003 -0.019
  before: headline baseline, of those listed: training_mean of training_mean on 30
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 30 of 30

int_stamps, rows_repeated_1pct
  before  +0.9602 to +0.9996
    +0.995 +1.000 +0.960 +0.995 +0.999 +0.995 +0.998 +0.998 +0.994 +0.992 +0.985 +0.999 +0.999 +0.997 +0.996 +0.991 +0.999 +0.999 +0.996 +0.999 +0.998 +0.996 +0.996 +0.995 +0.996 +0.993 +0.999 +0.986 +0.994 +0.999
  new     -0.1926 to +0.0048
    +0.004 -0.017 +0.005 +0.002 -0.193 +0.001 +0.003 +0.002 +0.004 -0.001 +0.001 -0.080 -0.040 -0.023 -0.010 -0.004 -0.005 -0.007 -0.015 -0.016 -0.172 +0.000 -0.038 -0.013 +0.000 -0.000 -0.002 +0.001 -0.001 -0.019
  before: headline baseline, of those listed: training_mean of training_mean on 30
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 30 of 30

int_stamps, rows_repeated_10pct
  before  +0.9608 to +0.9996
    +0.995 +1.000 +0.961 +0.995 +0.999 +0.995 +0.998 +0.998 +0.994 +0.992 +0.985 +0.999 +0.999 +0.997 +0.996 +0.991 +0.999 +0.999 +0.996 +0.999 +0.997 +0.996 +0.996 +0.995 +0.996 +0.993 +0.999 +0.986 +0.993 +0.999
  new     -0.2509 to +0.0062
    +0.003 -0.008 +0.006 +0.003 -0.251 +0.002 +0.002 +0.000 +0.002 -0.003 -0.003 -0.069 -0.011 -0.025 -0.009 -0.013 -0.010 -0.014 -0.019 -0.020 -0.213 -0.002 -0.051 -0.006 -0.003 -0.015 -0.000 +0.003 -0.006 -0.009
  before: headline baseline, of those listed: training_mean of training_mean on 30
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 30 of 30

flat_tail, (as it is)
  before  no value
    null null null null null null null null null null null null null null null null null null null null null null null null null null null null null null
  new     no value
    null null null null null null null null null null null null null null null null null null null null null null null null null null null null null null
  before: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 0 of 30

flat_tail, row_repeated
  before  -0.2228 to -0.0327
    -0.171 -0.118 -0.223 -0.148 -0.111 -0.101 -0.055 -0.033 -0.142 -0.132 -0.133 -0.102 -0.081 -0.193 -0.153 -0.061 -0.125 -0.141 -0.097 -0.041 -0.088 -0.138 -0.082 -0.149 -0.095 -0.101 -0.086 -0.219 -0.091 -0.197
  new     no value
    null null null null null null null null null null null null null null null null null null null null null null null null null null null null null null
  before: headline baseline, of those listed: training_mean of training_mean on 30
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 19 of 30

flat_tail, rows_repeated_1pct
  before  -0.2251 to -0.0284
    -0.169 -0.127 -0.225 -0.142 -0.122 -0.106 -0.049 -0.028 -0.150 -0.138 -0.127 -0.112 -0.077 -0.189 -0.147 -0.058 -0.120 -0.150 -0.109 -0.042 -0.082 -0.138 -0.087 -0.155 -0.089 -0.107 -0.084 -0.221 -0.089 -0.192
  new     no value
    null null null null null null null null null null null null null null null null null null null null null null null null null null null null null null
  before: headline baseline, of those listed: training_mean of training_mean on 30
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 19 of 30

flat_tail, rows_repeated_10pct
  before  -0.2295 to -0.0243
    -0.170 -0.131 -0.229 -0.171 -0.107 -0.137 -0.041 -0.024 -0.159 -0.118 -0.141 -0.105 -0.092 -0.200 -0.154 -0.063 -0.120 -0.133 -0.077 -0.031 -0.109 -0.158 -0.105 -0.151 -0.073 -0.103 -0.091 -0.220 -0.105 -0.206
  new     no value
    null null null null null null null null null null null null null null null null null null null null null null null null null null null null null null
  before: headline baseline, of those listed: training_mean of training_mean on 30
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 19 of 30

panel --no-entities, (as it is)
  before  +0.7717 to +0.9581
    +0.916 +0.911 +0.889 +0.919 +0.870 +0.877 +0.855 +0.779 +0.932 +0.952 +0.873 +0.941 +0.772 +0.926 +0.938 +0.946 +0.916 +0.955 +0.837 +0.867 +0.779 +0.954 +0.940 +0.958 +0.913 +0.942 +0.794 +0.827 +0.874 +0.924
  new     +0.7585 to +0.9580
    +0.916 +0.910 +0.889 +0.919 +0.870 +0.870 +0.852 +0.778 +0.932 +0.952 +0.872 +0.941 +0.758 +0.926 +0.938 +0.946 +0.916 +0.955 +0.836 +0.862 +0.777 +0.954 +0.939 +0.958 +0.911 +0.941 +0.791 +0.825 +0.873 +0.924
  before: headline baseline, of those listed: training_mean of training_mean on 30
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 30 of 30

panel --no-entities, row_repeated
  before  +0.7716 to +0.9579
    +0.916 +0.910 +0.889 +0.920 +0.870 +0.877 +0.855 +0.779 +0.932 +0.952 +0.873 +0.941 +0.772 +0.926 +0.938 +0.946 +0.916 +0.955 +0.837 +0.867 +0.779 +0.954 +0.940 +0.958 +0.913 +0.941 +0.794 +0.828 +0.874 +0.924
  new     +0.7585 to +0.9579
    +0.916 +0.909 +0.889 +0.919 +0.870 +0.870 +0.852 +0.778 +0.932 +0.952 +0.872 +0.941 +0.758 +0.926 +0.938 +0.946 +0.916 +0.955 +0.836 +0.862 +0.777 +0.954 +0.939 +0.958 +0.911 +0.941 +0.792 +0.825 +0.874 +0.924
  before: headline baseline, of those listed: training_mean of training_mean on 30
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 30 of 30

panel --no-entities, copy_rescaled
  before  +0.7717 to +0.9581
    +0.916 +0.911 +0.889 +0.919 +0.870 +0.877 +0.855 +0.779 +0.932 +0.952 +0.873 +0.941 +0.772 +0.926 +0.938 +0.946 +0.916 +0.955 +0.837 +0.867 +0.779 +0.954 +0.940 +0.958 +0.913 +0.942 +0.794 +0.827 +0.874 +0.924
  new     +0.7585 to +0.9580
    +0.916 +0.910 +0.889 +0.919 +0.870 +0.870 +0.852 +0.778 +0.932 +0.952 +0.872 +0.941 +0.758 +0.926 +0.938 +0.946 +0.916 +0.955 +0.836 +0.862 +0.777 +0.954 +0.939 +0.958 +0.911 +0.941 +0.791 +0.825 +0.873 +0.924
  before: headline baseline, of those listed: training_mean of training_mean on 30
  new: headline baseline, of those listed: previous_value of previous_value+training_mean on 30
  new: `autocorrelated_panel_target` on 30 of 30

time_blocks, (as it is)
  before  -0.2483 to +0.2170
    +0.122 +0.118 +0.217 -0.009 +0.031 +0.080 -0.104 +0.040 +0.028 +0.017 +0.014 -0.082 -0.155 +0.016 +0.050 +0.138 -0.068 -0.167 -0.144 +0.069 +0.217 -0.157 +0.085 -0.053 -0.173 -0.037 -0.007 -0.038 -0.023 -0.248
  new     -0.2483 to +0.2170
    +0.122 +0.118 +0.217 -0.009 +0.031 +0.080 -0.104 +0.040 +0.028 +0.017 +0.014 -0.082 -0.155 +0.016 +0.050 +0.138 -0.068 -0.167 -0.144 +0.069 +0.217 -0.157 +0.085 -0.053 -0.173 -0.037 -0.007 -0.038 -0.023 -0.248
  before: headline baseline, of those listed: chance of chance on 30
  new: headline baseline, of those listed: chance of chance on 30
  new: `autocorrelated_panel_target` on 0 of 30

time_blocks, row_repeated
  before  -0.2296 to +0.2170
    +0.180 +0.136 +0.217 -0.008 +0.036 +0.188 -0.098 -0.008 +0.028 +0.020 +0.008 -0.082 -0.127 +0.020 +0.051 +0.141 -0.068 -0.167 +0.002 +0.041 +0.084 -0.157 +0.088 -0.048 -0.187 -0.070 +0.001 -0.044 -0.018 -0.230
  new     -0.2296 to +0.2170
    +0.180 +0.136 +0.217 -0.008 +0.036 +0.188 -0.098 -0.008 +0.028 +0.020 +0.008 -0.082 -0.127 +0.020 +0.051 +0.141 -0.068 -0.167 +0.002 +0.041 +0.084 -0.157 +0.088 -0.048 -0.187 -0.070 +0.001 -0.044 -0.018 -0.230
  before: headline baseline, of those listed: chance of chance on 30
  new: headline baseline, of those listed: chance of chance on 30
  new: `autocorrelated_panel_target` on 0 of 30
```

What it shows. One repeated row made a random walk look almost perfectly
predictable (0.939 to 0.9996; one seed stopped with `no_features`, every
column having been excluded as a leak against the training mean). With the
mean target of the timestamp before as the previous value, the same tables
give -0.101 to +0.012, the band of `random_walk`. `flat_tail` gets its null
skill back. `panel` under `--no-entities` is measured against the previous
value and still shows 0.76 to 0.96, with the warning on every seed: the mean
of twelve series is not any one series' previous value, which is what the
warning's new text says.

## 2. The leak screen (A2, A3b; D74 point 2, D76 points 1 and 2)

### 2.1 Copies of the target, under every scheme

Nine regression runs (`signal_reg`, `noise_reg`, `leak_reg` under the random
scheme; `random_walk`, `int_stamps`, `flat_tail` under time order; `panel`
under time order with entities, and under `--no-entities`; `dominant_entity`
and `quarter_entity` under the grouped scheme), each with a column
`target_copy` added: an exact copy; rescaled; negative; scaled by 1e-9; a
rescaled copy with 5 cells, 15 cells or 2% of the cells altered; with one cell
of the copy set to 999999 or -999; with noise of 5%, 10% and 20% of the
target's spread; the square and the exponential of the target; and a rescaled
copy beside one target cell of -999. Counted over the 30 seeds, on the
reference:

| run | change | copy excluded | copy warned | copy kept in silence | other column excluded | errors | skill |
| signal_reg | copy_exact | 30 | 0 | 0 | 0 |  | +0.819 to +0.891 |
| signal_reg | copy_rescaled | 30 | 0 | 0 | 0 |  | +0.819 to +0.891 |
| signal_reg | copy_negative | 30 | 0 | 0 | 0 |  | +0.819 to +0.891 |
| signal_reg | copy_tiny_scale | 30 | 0 | 0 | 0 |  | +0.819 to +0.891 |
| signal_reg | copy_5_cells_differ | 30 | 0 | 0 | 0 |  | +0.819 to +0.891 |
| signal_reg | copy_15_cells_differ | 30 | 0 | 0 | 0 |  | +0.819 to +0.891 |
| signal_reg | copy_2pct_cells_differ | 30 | 0 | 0 | 0 |  | +0.819 to +0.891 |
| signal_reg | copy_sentinel_999999 | 30 | 0 | 0 | 0 |  | +0.819 to +0.891 |
| signal_reg | copy_sentinel_-999 | 30 | 0 | 0 | 0 |  | +0.819 to +0.891 |
| signal_reg | copy_noise_5pct | 30 | 0 | 0 | 0 |  | +0.819 to +0.891 |
| signal_reg | copy_noise_10pct | 30 | 0 | 0 | 0 |  | +0.819 to +0.891 |
| signal_reg | copy_noise_20pct | 0 | 30 | 0 | 0 |  | +0.965 to +0.976 |
| signal_reg | copy_squared | 30 | 0 | 0 | 0 |  | +0.819 to +0.891 |
| signal_reg | copy_exp | 30 | 0 | 0 | 0 |  | +0.819 to +0.891 |
| signal_reg | target_-999_and_copy | 30 | 0 | 0 | 0 |  | -646.300 to +0.720 |
| noise_reg | copy_exact | 30 | 0 | 0 | 0 |  | -0.503 to -0.242 |
| noise_reg | copy_rescaled | 30 | 0 | 0 | 0 |  | -0.503 to -0.242 |
| noise_reg | copy_negative | 30 | 0 | 0 | 0 |  | -0.503 to -0.242 |
| noise_reg | copy_tiny_scale | 30 | 0 | 0 | 0 |  | -0.503 to -0.242 |
| noise_reg | copy_5_cells_differ | 30 | 0 | 0 | 0 |  | -0.503 to -0.242 |
| noise_reg | copy_15_cells_differ | 30 | 0 | 0 | 0 |  | -0.503 to -0.242 |
| noise_reg | copy_2pct_cells_differ | 30 | 0 | 0 | 0 |  | -0.503 to -0.242 |
| noise_reg | copy_sentinel_999999 | 30 | 0 | 0 | 0 |  | -0.503 to -0.242 |
| noise_reg | copy_sentinel_-999 | 30 | 0 | 0 | 0 |  | -0.503 to -0.242 |
| noise_reg | copy_noise_5pct | 30 | 0 | 0 | 0 |  | -0.503 to -0.242 |
| noise_reg | copy_noise_10pct | 30 | 0 | 0 | 0 |  | -0.503 to -0.242 |
| noise_reg | copy_noise_20pct | 0 | 30 | 0 | 0 |  | +0.946 to +0.962 |
| noise_reg | copy_squared | 30 | 0 | 0 | 0 |  | -0.503 to -0.242 |
| noise_reg | copy_exp | 30 | 0 | 0 | 0 |  | -0.503 to -0.242 |
| noise_reg | target_-999_and_copy | 30 | 0 | 0 | 0 |  | -1603.695 to +0.002 |
| leak_reg | copy_exact | 30 | 0 | 0 | 0 |  | +0.806 to +0.859 |
| leak_reg | copy_rescaled | 30 | 0 | 0 | 0 |  | +0.806 to +0.859 |
| leak_reg | copy_negative | 30 | 0 | 0 | 0 |  | +0.806 to +0.859 |
| leak_reg | copy_tiny_scale | 30 | 0 | 0 | 0 |  | +0.806 to +0.859 |
| leak_reg | copy_5_cells_differ | 30 | 0 | 0 | 0 |  | +0.806 to +0.859 |
| leak_reg | copy_15_cells_differ | 30 | 0 | 0 | 0 |  | +0.806 to +0.859 |
| leak_reg | copy_2pct_cells_differ | 30 | 0 | 0 | 0 |  | +0.806 to +0.859 |
| leak_reg | copy_sentinel_999999 | 30 | 0 | 0 | 0 |  | +0.806 to +0.859 |
| leak_reg | copy_sentinel_-999 | 30 | 0 | 0 | 0 |  | +0.806 to +0.859 |
| leak_reg | copy_noise_5pct | 30 | 0 | 0 | 0 |  | +0.806 to +0.859 |
| leak_reg | copy_noise_10pct | 30 | 0 | 0 | 0 |  | +0.806 to +0.859 |
| leak_reg | copy_noise_20pct | 0 | 30 | 0 | 0 |  | +0.964 to +0.976 |
| leak_reg | copy_squared | 30 | 0 | 0 | 0 |  | +0.806 to +0.859 |
| leak_reg | copy_exp | 30 | 0 | 0 | 0 |  | +0.806 to +0.859 |
| leak_reg | target_-999_and_copy | 30 | 0 | 0 | 0 |  | -341.299 to +0.711 |
| random_walk | copy_exact | 30 | 0 | 0 | 0 |  | -0.100 to +0.012 |
| random_walk | copy_rescaled | 30 | 0 | 0 | 0 |  | -0.100 to +0.012 |
| random_walk | copy_negative | 30 | 0 | 0 | 0 |  | -0.100 to +0.012 |
| random_walk | copy_tiny_scale | 30 | 0 | 0 | 0 |  | -0.100 to +0.012 |
| random_walk | copy_5_cells_differ | 30 | 0 | 0 | 0 |  | -0.100 to +0.012 |
| random_walk | copy_15_cells_differ | 29 | 1 | 0 | 0 |  | -0.100 to +0.124 |
| random_walk | copy_2pct_cells_differ | 1 | 29 | 0 | 0 |  | -47.660 to +0.590 |
| random_walk | copy_sentinel_999999 | 30 | 0 | 0 | 0 |  | -0.100 to +0.012 |
| random_walk | copy_sentinel_-999 | 30 | 0 | 0 | 0 |  | -0.100 to +0.012 |
| random_walk | copy_noise_5pct | 0 | 1 | 29 | 0 |  | +0.431 to +0.932 |
| random_walk | copy_noise_10pct | 0 | 0 | 30 | 0 |  | +0.098 to +0.772 |
| random_walk | copy_noise_20pct | 0 | 0 | 30 | 0 |  | -0.015 to +0.456 |
| random_walk | copy_squared | 0 | 30 | 0 | 0 |  | -36.407 to +0.982 |
| random_walk | copy_exp | 0 | 30 | 0 | 0 |  | -83.623 to +1.000 |
| random_walk | target_-999_and_copy | 30 | 0 | 0 | 0 |  | -29.544 to +0.388 |
| int_stamps | copy_exact | 30 | 0 | 0 | 0 |  | -0.182 to +0.005 |
| int_stamps | copy_rescaled | 30 | 0 | 0 | 0 |  | -0.182 to +0.005 |
| int_stamps | copy_negative | 30 | 0 | 0 | 0 |  | -0.182 to +0.005 |
| int_stamps | copy_tiny_scale | 30 | 0 | 0 | 0 |  | -0.182 to +0.005 |
| int_stamps | copy_5_cells_differ | 30 | 0 | 0 | 0 |  | -0.182 to +0.005 |
| int_stamps | copy_15_cells_differ | 28 | 2 | 0 | 0 |  | -0.182 to +0.395 |
| int_stamps | copy_2pct_cells_differ | 1 | 29 | 0 | 0 |  | -121.013 to +0.285 |
| int_stamps | copy_sentinel_999999 | 30 | 0 | 0 | 0 |  | -0.182 to +0.005 |
| int_stamps | copy_sentinel_-999 | 30 | 0 | 0 | 0 |  | -0.182 to +0.005 |
| int_stamps | copy_noise_5pct | 0 | 0 | 30 | 0 |  | +0.456 to +0.858 |
| int_stamps | copy_noise_10pct | 0 | 0 | 30 | 0 |  | +0.148 to +0.679 |
| int_stamps | copy_noise_20pct | 0 | 0 | 30 | 0 |  | -0.085 to +0.284 |
| int_stamps | copy_squared | 0 | 30 | 0 | 0 |  | -62.342 to +0.807 |
| int_stamps | copy_exp | 0 | 30 | 0 | 0 |  | -638.596 to +0.997 |
| int_stamps | target_-999_and_copy | 30 | 0 | 0 | 0 |  | -38.398 to +0.345 |
| flat_tail | copy_exact | 30 | 0 | 0 | 0 |  |  |
| flat_tail | copy_rescaled | 30 | 0 | 0 | 0 |  |  |
| flat_tail | copy_negative | 30 | 0 | 0 | 0 |  |  |
| flat_tail | copy_tiny_scale | 30 | 0 | 0 | 0 |  |  |
| flat_tail | copy_5_cells_differ | 30 | 0 | 0 | 0 |  |  |
| flat_tail | copy_15_cells_differ | 30 | 0 | 0 | 0 |  |  |
| flat_tail | copy_2pct_cells_differ | 30 | 0 | 0 | 0 |  |  |
| flat_tail | copy_sentinel_999999 | 30 | 0 | 0 | 0 |  |  |
| flat_tail | copy_sentinel_-999 | 30 | 0 | 0 | 0 |  |  |
| flat_tail | copy_noise_5pct | 30 | 0 | 0 | 0 |  |  |
| flat_tail | copy_noise_10pct | 30 | 0 | 0 | 0 |  |  |
| flat_tail | copy_noise_20pct | 0 | 30 | 0 | 0 |  |  |
| flat_tail | copy_squared | 30 | 0 | 0 | 0 |  |  |
| flat_tail | copy_exp | 30 | 0 | 0 | 0 |  |  |
| flat_tail | target_-999_and_copy | 30 | 0 | 0 | 0 |  | -0.009 to +0.028 |
| panel | copy_exact | 30 | 0 | 0 | 0 |  | -0.023 to +0.006 |
| panel | copy_rescaled | 30 | 0 | 0 | 0 |  | -0.023 to +0.006 |
| panel | copy_negative | 30 | 0 | 0 | 0 |  | -0.023 to +0.006 |
| panel | copy_tiny_scale | 30 | 0 | 0 | 0 |  | -0.023 to +0.006 |
| panel | copy_5_cells_differ | 30 | 0 | 0 | 0 |  | -0.023 to +0.006 |
| panel | copy_15_cells_differ | 26 | 4 | 0 | 0 |  | -3.285 to +0.572 |
| panel | copy_2pct_cells_differ | 1 | 29 | 0 | 0 |  | -4.679 to +0.745 |
| panel | copy_sentinel_999999 | 30 | 0 | 0 | 0 |  | -0.023 to +0.006 |
| panel | copy_sentinel_-999 | 30 | 0 | 0 | 0 |  | -0.023 to +0.006 |
| panel | copy_noise_5pct | 0 | 0 | 30 | 0 |  | +0.277 to +0.714 |
| panel | copy_noise_10pct | 0 | 0 | 30 | 0 |  | -0.020 to +0.451 |
| panel | copy_noise_20pct | 0 | 0 | 30 | 0 |  | -0.037 to +0.177 |
| panel | copy_squared | 0 | 30 | 0 | 0 |  | -4.574 to +0.978 |
| panel | copy_exp | 0 | 30 | 0 | 0 |  | -4.574 to +0.977 |
| panel | target_-999_and_copy | 30 | 0 | 0 | 0 |  | -2201.992 to +0.201 |
| dominant_entity | copy_exact | 30 | 0 | 0 | 0 |  | -0.012 to +0.003 |
| dominant_entity | copy_rescaled | 30 | 0 | 0 | 0 |  | -0.012 to +0.003 |
| dominant_entity | copy_negative | 30 | 0 | 0 | 0 |  | -0.012 to +0.003 |
| dominant_entity | copy_tiny_scale | 30 | 0 | 0 | 0 |  | -0.012 to +0.003 |
| dominant_entity | copy_5_cells_differ | 30 | 0 | 0 | 0 |  | -0.012 to +0.003 |
| dominant_entity | copy_15_cells_differ | 30 | 0 | 0 | 0 |  | -0.012 to +0.003 |
| dominant_entity | copy_2pct_cells_differ | 30 | 0 | 0 | 0 |  | -0.012 to +0.003 |
| dominant_entity | copy_sentinel_999999 | 30 | 0 | 0 | 0 |  | -0.012 to +0.003 |
| dominant_entity | copy_sentinel_-999 | 30 | 0 | 0 | 0 |  | -0.012 to +0.003 |
| dominant_entity | copy_noise_5pct | 30 | 0 | 0 | 0 |  | -0.012 to +0.003 |
| dominant_entity | copy_noise_10pct | 27 | 3 | 0 | 0 |  | -0.012 to +0.992 |
| dominant_entity | copy_noise_20pct | 0 | 30 | 0 | 0 |  | +0.959 to +0.974 |
| dominant_entity | copy_squared | 30 | 0 | 0 | 0 |  | -0.012 to +0.003 |
| dominant_entity | copy_exp | 30 | 0 | 0 | 0 |  | -0.012 to +0.003 |
| dominant_entity | target_-999_and_copy | 30 | 0 | 0 | 0 |  | -3.464 to +0.000 |
| quarter_entity | copy_exact | 30 | 0 | 0 | 0 |  | -0.011 to +0.006 |
| quarter_entity | copy_rescaled | 30 | 0 | 0 | 0 |  | -0.011 to +0.006 |
| quarter_entity | copy_negative | 30 | 0 | 0 | 0 |  | -0.011 to +0.006 |
| quarter_entity | copy_tiny_scale | 30 | 0 | 0 | 0 |  | -0.011 to +0.006 |
| quarter_entity | copy_5_cells_differ | 30 | 0 | 0 | 0 |  | -0.011 to +0.006 |
| quarter_entity | copy_15_cells_differ | 30 | 0 | 0 | 0 |  | -0.011 to +0.006 |
| quarter_entity | copy_2pct_cells_differ | 30 | 0 | 0 | 0 |  | -0.011 to +0.006 |
| quarter_entity | copy_sentinel_999999 | 30 | 0 | 0 | 0 |  | -0.011 to +0.006 |
| quarter_entity | copy_sentinel_-999 | 30 | 0 | 0 | 0 |  | -0.011 to +0.006 |
| quarter_entity | copy_noise_5pct | 30 | 0 | 0 | 0 |  | -0.011 to +0.006 |
| quarter_entity | copy_noise_10pct | 30 | 0 | 0 | 0 |  | -0.011 to +0.006 |
| quarter_entity | copy_noise_20pct | 0 | 30 | 0 | 0 |  | +0.958 to +0.970 |
| quarter_entity | copy_squared | 30 | 0 | 0 | 0 |  | -0.011 to +0.006 |
| quarter_entity | copy_exp | 30 | 0 | 0 | 0 |  | -0.011 to +0.006 |
| quarter_entity | target_-999_and_copy | 30 | 0 | 0 | 0 |  | -1405.119 to +0.003 |
| panel --no-entities | copy_rescaled | 30 | 0 | 0 | 30 |  | +0.758 to +0.958 |
| panel --no-entities | copy_5_cells_differ | 30 | 0 | 0 | 30 |  | +0.758 to +0.958 |

Seed by seed, before and now (X excluded, w warned, - kept in silence, E the
run stopped):

| run | change | before, seeds 0 to 29 | new, seeds 0 to 29 | new: X / w / - / E |
|---|---|---|---|---|
| signal_reg | copy_exact | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| signal_reg | copy_rescaled | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| signal_reg | copy_negative | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| signal_reg | copy_tiny_scale | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| signal_reg | copy_5_cells_differ | `XXXXXXwXXXXXXXXXwXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| signal_reg | copy_15_cells_differ | `XXXXwXwXwXwXwXwwwwwwXwwXwwwXww` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| signal_reg | copy_2pct_cells_differ | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| signal_reg | copy_sentinel_999999 | `XXXXwXXXXXXXXXXXXXXXwwXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| signal_reg | copy_sentinel_-999 | `XXXXXwXXXXXXXXXXXXXXXXXXXXwXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| signal_reg | copy_noise_5pct | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| signal_reg | copy_noise_10pct | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| signal_reg | copy_noise_20pct | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | 0 / 30 / 0 / 0 |
| signal_reg | copy_squared | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| signal_reg | copy_exp | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| signal_reg | target_-999_and_copy | `-------XX-X------X----XX---X--` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| noise_reg | copy_exact | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| noise_reg | copy_rescaled | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| noise_reg | copy_negative | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| noise_reg | copy_tiny_scale | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| noise_reg | copy_5_cells_differ | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| noise_reg | copy_15_cells_differ | `XXXwwwXXwwwwXXwXwwwwXXwXwwwXXw` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| noise_reg | copy_2pct_cells_differ | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| noise_reg | copy_sentinel_999999 | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| noise_reg | copy_sentinel_-999 | `XXXXXwXXXXXXXXXXXXXXXXXwXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| noise_reg | copy_noise_5pct | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| noise_reg | copy_noise_10pct | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| noise_reg | copy_noise_20pct | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | 0 / 30 / 0 / 0 |
| noise_reg | copy_squared | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| noise_reg | copy_exp | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| noise_reg | target_-999_and_copy | `---X---XX----X------X---------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| leak_reg | copy_exact | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| leak_reg | copy_rescaled | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| leak_reg | copy_negative | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| leak_reg | copy_tiny_scale | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| leak_reg | copy_5_cells_differ | `XXXXXXXXXwXXXXXwXXXXXXXwwXXXwX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| leak_reg | copy_15_cells_differ | `XXwwwXXwXwwXXXXwXwwXXwXXXwwwXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| leak_reg | copy_2pct_cells_differ | `wwwwwwwwwwwwwwwwwwXwwwwwwwwwww` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| leak_reg | copy_sentinel_999999 | `XXXwXXXXXXXXXXXXXXwXXXXXwXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| leak_reg | copy_sentinel_-999 | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| leak_reg | copy_noise_5pct | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| leak_reg | copy_noise_10pct | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| leak_reg | copy_noise_20pct | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | 0 / 30 / 0 / 0 |
| leak_reg | copy_squared | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| leak_reg | copy_exp | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| leak_reg | target_-999_and_copy | `-X---X--X-X--XX---XX---X------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| random_walk | copy_exact | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| random_walk | copy_rescaled | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| random_walk | copy_negative | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| random_walk | copy_tiny_scale | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| random_walk | copy_5_cells_differ | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| random_walk | copy_15_cells_differ | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXwXX` | 29 / 1 / 0 / 0 |
| random_walk | copy_2pct_cells_differ | `------------------------------` | `wwwwwwwwwwwwwwwwwwwwwwXwwwwwww` | 1 / 29 / 0 / 0 |
| random_walk | copy_sentinel_999999 | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| random_walk | copy_sentinel_-999 | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| random_walk | copy_noise_5pct | `------------------------------` | `-------w----------------------` | 0 / 1 / 29 / 0 |
| random_walk | copy_noise_10pct | `------------------------------` | `------------------------------` | 0 / 0 / 30 / 0 |
| random_walk | copy_noise_20pct | `------------------------------` | `------------------------------` | 0 / 0 / 30 / 0 |
| random_walk | copy_squared | `------------------------------` | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | 0 / 30 / 0 / 0 |
| random_walk | copy_exp | `------------------------------` | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | 0 / 30 / 0 / 0 |
| random_walk | target_-999_and_copy | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| int_stamps | copy_exact | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| int_stamps | copy_rescaled | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| int_stamps | copy_negative | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| int_stamps | copy_tiny_scale | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| int_stamps | copy_5_cells_differ | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| int_stamps | copy_15_cells_differ | `------------------------------` | `XXXXXXXXXXXXXXwXXXXXXwXXXXXXXX` | 28 / 2 / 0 / 0 |
| int_stamps | copy_2pct_cells_differ | `------------------------------` | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwwX` | 1 / 29 / 0 / 0 |
| int_stamps | copy_sentinel_999999 | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| int_stamps | copy_sentinel_-999 | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| int_stamps | copy_noise_5pct | `------------------------------` | `------------------------------` | 0 / 0 / 30 / 0 |
| int_stamps | copy_noise_10pct | `------------------------------` | `------------------------------` | 0 / 0 / 30 / 0 |
| int_stamps | copy_noise_20pct | `------------------------------` | `------------------------------` | 0 / 0 / 30 / 0 |
| int_stamps | copy_squared | `------------------------------` | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | 0 / 30 / 0 / 0 |
| int_stamps | copy_exp | `------------------------------` | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | 0 / 30 / 0 / 0 |
| int_stamps | target_-999_and_copy | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| flat_tail | copy_exact | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| flat_tail | copy_rescaled | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| flat_tail | copy_negative | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| flat_tail | copy_tiny_scale | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| flat_tail | copy_5_cells_differ | `wX--ww-wwXww--www-ww-wwwww-XX-` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| flat_tail | copy_15_cells_differ | `w----w------w-w---------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| flat_tail | copy_2pct_cells_differ | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| flat_tail | copy_sentinel_999999 | `XXXXXXwXXXXXXXXXXXXXXXXXwXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| flat_tail | copy_sentinel_-999 | `wwwwXXwwX-XXwwXXwwXwww-wXwXwww` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| flat_tail | copy_noise_5pct | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| flat_tail | copy_noise_10pct | `XwXwXXwwwXXXwwXwXwwXwXXXwXwXww` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| flat_tail | copy_noise_20pct | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | 0 / 30 / 0 / 0 |
| flat_tail | copy_squared | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| flat_tail | copy_exp | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| flat_tail | target_-999_and_copy | `---XX--XX--X-------X----X-----` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| panel | copy_exact | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| panel | copy_rescaled | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| panel | copy_negative | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| panel | copy_tiny_scale | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| panel | copy_5_cells_differ | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| panel | copy_15_cells_differ | `------------------------------` | `XXXXXXXwwXXXwXXwXXXXXXXXXXXXXX` | 26 / 4 / 0 / 0 |
| panel | copy_2pct_cells_differ | `------------------------------` | `wXwwwwwwwwwwwwwwwwwwwwwwwwwwww` | 1 / 29 / 0 / 0 |
| panel | copy_sentinel_999999 | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| panel | copy_sentinel_-999 | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| panel | copy_noise_5pct | `------------------------------` | `------------------------------` | 0 / 0 / 30 / 0 |
| panel | copy_noise_10pct | `------------------------------` | `------------------------------` | 0 / 0 / 30 / 0 |
| panel | copy_noise_20pct | `------------------------------` | `------------------------------` | 0 / 0 / 30 / 0 |
| panel | copy_squared | `------------------------------` | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | 0 / 30 / 0 / 0 |
| panel | copy_exp | `------------------------------` | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | 0 / 30 / 0 / 0 |
| panel | target_-999_and_copy | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| dominant_entity | copy_exact | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| dominant_entity | copy_rescaled | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| dominant_entity | copy_negative | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| dominant_entity | copy_tiny_scale | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| dominant_entity | copy_5_cells_differ | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| dominant_entity | copy_15_cells_differ | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| dominant_entity | copy_2pct_cells_differ | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| dominant_entity | copy_sentinel_999999 | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| dominant_entity | copy_sentinel_-999 | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| dominant_entity | copy_noise_5pct | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| dominant_entity | copy_noise_10pct | `------------------------------` | `XXXXXXXXXXXXXXwXXXXwXXXXXXXXXw` | 27 / 3 / 0 / 0 |
| dominant_entity | copy_noise_20pct | `------------------------------` | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | 0 / 30 / 0 / 0 |
| dominant_entity | copy_squared | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| dominant_entity | copy_exp | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| dominant_entity | target_-999_and_copy | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| quarter_entity | copy_exact | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| quarter_entity | copy_rescaled | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| quarter_entity | copy_negative | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| quarter_entity | copy_tiny_scale | `------------------------------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| quarter_entity | copy_5_cells_differ | `XXXXXXXXXXXXXwXXwXXXXXwXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| quarter_entity | copy_15_cells_differ | `XXwwwXXwXwXwwXXwwXXwXXXwXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| quarter_entity | copy_2pct_cells_differ | `wwwwwwwwXwwwwwwwwXwwwwwwwwwwww` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| quarter_entity | copy_sentinel_999999 | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| quarter_entity | copy_sentinel_-999 | `XXXXXXXXXXXXXXXXXXXXXXXXwXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| quarter_entity | copy_noise_5pct | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| quarter_entity | copy_noise_10pct | `XXXwXXXXXwXXXXXXXXXXXXwXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| quarter_entity | copy_noise_20pct | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | `wwwwwwwwwwwwwwwwwwwwwwwwwwwwww` | 0 / 30 / 0 / 0 |
| quarter_entity | copy_squared | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| quarter_entity | copy_exp | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| quarter_entity | target_-999_and_copy | `--XXX-XXXX----XX-X-X--XX------` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| panel --no-entities | copy_rescaled | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |
| panel --no-entities | copy_5_cells_differ | `XXwXXXXXwXXXwXwXXXXXXXXXXXXXww` | `XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` | 30 / 0 / 0 / 0 |

A column other than `target_copy` (and other than the gate's own leak `amount_scaled`) excluded as a suspected leak:
  panel --no-entities, copy_5_cells_differ: `level_prev` on 30 seeds before, 30 now
  panel --no-entities, copy_rescaled: `level_prev` on 30 seeds before, 30 now

### 2.2 How small the noise must be under time order (D76 point 2)

| run | noise, share of the target's spread | excluded / warned / silent | the column's skill against the previous value | headline skill |
|---|---|---|---|---|
| random_walk | 0.5% | 30 / 0 / 0 | +0.986 to +0.999 | -0.100 to +0.012 |
| random_walk | 1% | 20 / 10 / 0 | +0.948 to +0.997 | -0.100 to +0.969 |
| random_walk | 2% | 1 / 20 / 9 | +0.775 to +0.989 | +0.012 to +0.981 |
| random_walk | 5% | 0 / 1 / 29 | -0.651 to +0.928 | +0.387 to +0.938 |
| int_stamps | 0.5% | 30 / 0 / 0 | +0.987 to +0.998 | -0.182 to +0.005 |
| int_stamps | 1% | 19 / 11 / 0 | +0.948 to +0.994 | -0.176 to +0.971 |
| int_stamps | 2% | 0 / 23 / 7 | +0.773 to +0.976 | +0.824 to +0.977 |
| int_stamps | 5% | 0 / 0 / 30 | -0.480 to +0.841 | +0.441 to +0.863 |
| panel | 0.5% | 30 / 0 / 0 | +0.980 to +0.997 | -0.023 to +0.006 |
| panel | 1% | 3 / 27 / 0 | +0.919 to +0.985 | -0.002 to +0.953 |
| panel | 2% | 0 / 5 / 25 | +0.706 to +0.944 | +0.729 to +0.939 |
| panel | 5% | 0 / 0 / 30 | -0.967 to +0.657 | +0.210 to +0.734 |

At 0.5% of the target's spread the copy is excluded on every seed; at 1% it is
excluded or warned about on every seed; at 2% it is kept in silence on 9, 7
and 25 seeds of 30; at 5% on 29, 30 and 30. Hence the limit of spec section
11, and the three battery pairs listed under D76.

### 2.3 The order reading (D76 point 1)

Without it and with it, on the cases D74 said must at least draw a warning,
and its controls: every clean fast run of the gate, and a series that only
rises beside its own lag.

| run | column added | as ruled: excluded / warned / silent | with the order reading: excluded / warned / silent |
|---|---|---|---|
| random_walk | copy_2pct_cells_differ | 1 / 0 / 29 | 1 / 29 / 0 |
| random_walk | copy_noise_5pct | 0 / 1 / 29 | 0 / 1 / 29 |
| random_walk | copy_squared | 0 / 1 / 29 | 0 / 30 / 0 |
| random_walk | copy_exp | 0 / 0 / 30 | 0 / 30 / 0 |
| int_stamps | copy_2pct_cells_differ | 1 / 0 / 29 | 1 / 29 / 0 |
| int_stamps | copy_noise_5pct | 0 / 0 / 30 | 0 / 0 / 30 |
| int_stamps | copy_squared | 0 / 1 / 29 | 0 / 30 / 0 |
| int_stamps | copy_exp | 0 / 0 / 30 | 0 / 30 / 0 |
| panel | copy_2pct_cells_differ | 1 / 0 / 29 | 1 / 29 / 0 |
| panel | copy_noise_5pct | 0 / 0 / 30 | 0 / 0 / 30 |
| panel | copy_squared | 0 / 0 / 30 | 0 / 30 / 0 |
| panel | copy_exp | 0 / 0 / 30 | 0 / 30 / 0 |

gate: 960 clean fast runs; runs that gain a warning or move with the order reading: 0 []
a running total with its own lag, under time order: the lag warned about on 0 of 30 seeds

On a legitimately strong feature under time order, at 50, 100 and 200 rows and
a true R-squared of 0.90, 0.95 and 0.99, independent and persistent:

```text
cut-off 0.1

| shape | rows | true R-squared | share for x (range) | order reading holds, of seeds | run: excluded / warned by skill / warned by order alone / neither |
|---|---|---|---|---|---|
| independent | 50 | 0.90 | 0.278 to 0.500 | 0 of 30 | stops: too_few_rows |
| independent | 50 | 0.95 | 0.250 to 0.472 | 0 of 30 | stops: too_few_rows |
| independent | 50 | 0.99 | 0.222 to 0.389 | 0 of 30 | stops: too_few_rows |
| independent | 100 | 0.90 | 0.392 to 0.500 | 0 of 30 | 0 / 4 / 0 / 26 |
| independent | 100 | 0.95 | 0.351 to 0.486 | 0 of 30 | 0 / 30 / 0 / 0 |
| independent | 100 | 0.99 | 0.311 to 0.432 | 0 of 30 | 30 / 0 / 0 / 0 |
| independent | 200 | 0.90 | 0.430 to 0.497 | 0 of 30 | 0 / 6 / 0 / 24 |
| independent | 200 | 0.95 | 0.396 to 0.490 | 0 of 30 | 0 / 30 / 0 / 0 |
| independent | 200 | 0.99 | 0.356 to 0.463 | 0 of 30 | 30 / 0 / 0 / 0 |
| persistent | 50 | 0.90 | 0.361 to 0.500 | 0 of 30 | stops: too_few_rows |
| persistent | 50 | 0.95 | 0.278 to 0.500 | 0 of 30 | stops: too_few_rows |
| persistent | 50 | 0.99 | 0.194 to 0.472 | 0 of 30 | stops: too_few_rows |
| persistent | 100 | 0.90 | 0.405 to 0.500 | 0 of 30 | 0 / 0 / 0 / 30 |
| persistent | 100 | 0.95 | 0.392 to 0.500 | 0 of 30 | 0 / 0 / 0 / 30 |
| persistent | 100 | 0.99 | 0.324 to 0.432 | 0 of 30 | 0 / 14 / 0 / 16 |
| persistent | 200 | 0.90 | 0.450 to 0.497 | 0 of 30 | 0 / 0 / 0 / 30 |
| persistent | 200 | 0.95 | 0.423 to 0.497 | 0 of 30 | 0 / 0 / 0 / 30 |
| persistent | 200 | 0.99 | 0.376 to 0.483 | 0 of 30 | 0 / 7 / 0 / 23 |

per seed, the share for x (seeds 0 to 29):
independent 50 0.90: 0.361 0.389 0.444 0.417 0.389 0.417 0.472 0.444 0.417 0.417 0.472 0.444 0.333 0.278 0.417 0.500 0.444 0.389 0.361 0.417 0.444 0.361 0.444 0.361 0.361 0.417 0.444 0.444 0.389 0.389
independent 50 0.95: 0.361 0.389 0.389 0.361 0.389 0.417 0.417 0.389 0.472 0.417 0.417 0.389 0.444 0.389 0.250 0.389 0.333 0.361 0.417 0.361 0.444 0.444 0.361 0.417 0.389 0.361 0.333 0.389 0.444 0.389
independent 50 0.99: 0.389 0.333 0.250 0.278 0.306 0.306 0.333 0.250 0.333 0.250 0.250 0.250 0.278 0.250 0.333 0.306 0.389 0.222 0.306 0.278 0.306 0.278 0.361 0.278 0.250 0.250 0.278 0.222 0.278 0.250
independent 100 0.90: 0.473 0.419 0.432 0.473 0.459 0.459 0.459 0.432 0.486 0.446 0.405 0.392 0.459 0.459 0.446 0.486 0.446 0.419 0.432 0.446 0.500 0.459 0.419 0.500 0.405 0.419 0.459 0.500 0.446 0.419
independent 100 0.95: 0.419 0.392 0.446 0.392 0.432 0.392 0.473 0.419 0.419 0.446 0.446 0.432 0.459 0.432 0.446 0.392 0.419 0.446 0.419 0.432 0.486 0.473 0.378 0.419 0.432 0.419 0.432 0.351 0.459 0.446
independent 100 0.99: 0.351 0.365 0.378 0.338 0.351 0.351 0.351 0.378 0.365 0.392 0.351 0.392 0.351 0.338 0.351 0.378 0.338 0.365 0.419 0.311 0.338 0.351 0.405 0.405 0.392 0.419 0.378 0.324 0.432 0.392
independent 200 0.90: 0.490 0.490 0.470 0.463 0.443 0.456 0.470 0.456 0.490 0.490 0.430 0.490 0.470 0.497 0.490 0.497 0.490 0.497 0.450 0.497 0.483 0.430 0.450 0.497 0.470 0.463 0.490 0.490 0.470 0.470
independent 200 0.95: 0.470 0.436 0.470 0.396 0.443 0.477 0.443 0.456 0.456 0.456 0.463 0.450 0.450 0.470 0.490 0.456 0.443 0.430 0.463 0.443 0.477 0.470 0.443 0.450 0.430 0.463 0.423 0.409 0.450 0.483
independent 200 0.99: 0.396 0.463 0.443 0.396 0.430 0.430 0.409 0.389 0.443 0.376 0.383 0.430 0.443 0.450 0.450 0.396 0.396 0.463 0.383 0.423 0.436 0.443 0.383 0.369 0.436 0.356 0.430 0.423 0.430 0.376
persistent 50 0.90: 0.389 0.389 0.417 0.444 0.417 0.417 0.389 0.417 0.417 0.444 0.472 0.472 0.417 0.361 0.472 0.472 0.444 0.389 0.444 0.389 0.472 0.444 0.500 0.417 0.444 0.444 0.417 0.361 0.417 0.417
persistent 50 0.95: 0.389 0.361 0.444 0.417 0.361 0.500 0.389 0.278 0.389 0.389 0.361 0.306 0.444 0.361 0.389 0.389 0.444 0.333 0.417 0.306 0.472 0.361 0.333 0.333 0.417 0.417 0.389 0.444 0.389 0.500
persistent 50 0.99: 0.444 0.361 0.361 0.278 0.278 0.417 0.306 0.194 0.333 0.194 0.333 0.333 0.417 0.250 0.306 0.222 0.306 0.306 0.417 0.333 0.250 0.306 0.389 0.306 0.333 0.361 0.472 0.306 0.278 0.250
persistent 100 0.90: 0.500 0.419 0.486 0.473 0.446 0.486 0.473 0.432 0.486 0.459 0.473 0.419 0.405 0.473 0.486 0.419 0.486 0.419 0.500 0.486 0.473 0.446 0.419 0.500 0.419 0.446 0.486 0.473 0.486 0.459
persistent 100 0.95: 0.459 0.446 0.432 0.392 0.486 0.405 0.432 0.432 0.392 0.473 0.486 0.486 0.459 0.432 0.459 0.473 0.405 0.405 0.500 0.405 0.486 0.473 0.446 0.446 0.432 0.405 0.432 0.392 0.432 0.432
persistent 100 0.99: 0.351 0.338 0.365 0.419 0.378 0.392 0.405 0.392 0.405 0.378 0.338 0.405 0.351 0.432 0.392 0.419 0.365 0.405 0.351 0.392 0.324 0.378 0.432 0.365 0.419 0.392 0.378 0.392 0.405 0.365
persistent 200 0.90: 0.490 0.463 0.497 0.483 0.450 0.463 0.470 0.483 0.463 0.463 0.483 0.456 0.497 0.456 0.497 0.477 0.450 0.456 0.483 0.490 0.470 0.463 0.470 0.490 0.477 0.477 0.477 0.490 0.483 0.483
persistent 200 0.95: 0.497 0.490 0.443 0.477 0.436 0.436 0.456 0.450 0.490 0.483 0.450 0.430 0.483 0.477 0.477 0.423 0.443 0.456 0.463 0.477 0.483 0.490 0.470 0.497 0.483 0.456 0.497 0.470 0.450 0.456
persistent 200 0.99: 0.450 0.463 0.430 0.470 0.470 0.450 0.423 0.443 0.416 0.477 0.383 0.416 0.389 0.376 0.436 0.403 0.430 0.450 0.416 0.416 0.416 0.456 0.423 0.430 0.483 0.436 0.470 0.423 0.430 0.436
```

The two conditions of the order reading hold on no seed in any cell; the
smallest share of neighbouring pairs in which the target moves the other way
is 0.194, against the cut-off of 0.10. The warnings in the last column are the
first reading's, as before this pass.

### 2.4 How many extreme target cells the rows set aside absorb (A3b)

| run | target cells set to -999 | the copy excluded, of 30 | another column excluded | errors |
|---|---|---|---|---|
| leak_reg | one cell | 30 of 30 | 0 |  |
| leak_reg | 0.5% of the rows | 30 of 30 | 0 |  |
| leak_reg | 1% | 14 of 30 | 0 |  |
| leak_reg | 2% | 0 of 30 | 0 |  |
| leak_reg | 5% | 0 of 30 | 0 |  |
| random_walk | one cell | 30 of 30 | 0 |  |
| random_walk | 0.5% of the rows | 8 of 30 | 0 |  |
| random_walk | 1% | 0 of 30 | 0 |  |
| random_walk | 2% | 0 of 30 | 0 |  |
| random_walk | 5% | 0 of 30 | 0 |  |

Up to 1% of the training rows are set aside; beyond that nothing is, and the
screen scores every row, as it did before.

### 2.5 Where the edges of the two cut-offs are (spec 10.2)

The tables of the first pass's measurement (`docs\measurements\CP1\lab.py`,
`nearleak`), made the same way and run on the reference of this pass: a
rescaled copy of a regression target with noise as a share of the target's
deviation, and a copy of a two-class target with an exact share of its labels
flipped; 1,500 rows, the 30 seeds of `near_leak`. The whole run takes about
ten seconds.

```text
## regression, 1,500 rows, 30 seeds

| noise, share of the target's deviation | the reading rule 2 takes: range (mean, sd) | the reading rule 3 takes: range (mean, sd) | excluded | warned | kept in silence |
|---|---|---|---|---|---|
| 5% | 0.997 to 0.998 (0.998, 0.000) | 0.997 to 0.998 (0.997, 0.000) | 30 | 0 | 0 |
| 10% | 0.989 to 0.992 (0.990, 0.001) | 0.989 to 0.992 (0.990, 0.001) | 30 | 0 | 0 |
| 12% | 0.984 to 0.989 (0.986, 0.001) | 0.984 to 0.988 (0.986, 0.001) | 30 | 0 | 0 |
| 15% | 0.976 to 0.982 (0.978, 0.002) | 0.976 to 0.982 (0.978, 0.002) | 2 | 28 | 0 |
| 20% | 0.957 to 0.969 (0.961, 0.003) | 0.957 to 0.968 (0.961, 0.003) | 0 | 30 | 0 |
| 25% | 0.934 to 0.952 (0.941, 0.004) | 0.935 to 0.951 (0.941, 0.004) | 0 | 30 | 0 |
| 30% | 0.906 to 0.933 (0.917, 0.006) | 0.908 to 0.931 (0.917, 0.006) | 0 | 30 | 0 |
| 35% | 0.877 to 0.910 (0.890, 0.008) | 0.878 to 0.908 (0.890, 0.007) | 0 | 1 | 29 |
| 40% | 0.843 to 0.887 (0.861, 0.009) | 0.845 to 0.883 (0.861, 0.009) | 0 | 0 | 30 |
| 50% | 0.773 to 0.832 (0.799, 0.013) | 0.774 to 0.827 (0.797, 0.013) | 0 | 0 | 30 |

seed by seed (k = 0 to 29), the reading rule 3 takes:
  5%: 0.997 0.998 0.997 0.998 0.998 0.998 0.998 0.998 0.997 0.998 0.998 0.997 0.998 0.997 0.998 0.998 0.997 0.997 0.997 0.998 0.997 0.997 0.997 0.997 0.998 0.997 0.997 0.997 0.998 0.998
  10%: 0.990 0.990 0.989 0.992 0.991 0.991 0.991 0.991 0.990 0.990 0.991 0.990 0.990 0.990 0.990 0.991 0.989 0.989 0.989 0.990 0.989 0.990 0.990 0.989 0.991 0.990 0.989 0.989 0.990 0.991
  12%: 0.985 0.986 0.985 0.988 0.987 0.987 0.986 0.986 0.985 0.986 0.987 0.985 0.986 0.985 0.986 0.987 0.985 0.985 0.985 0.986 0.984 0.985 0.985 0.985 0.987 0.985 0.985 0.985 0.986 0.987
  15%: 0.977 0.979 0.976 0.982 0.980 0.980 0.979 0.979 0.977 0.978 0.979 0.977 0.978 0.977 0.978 0.980 0.976 0.976 0.976 0.979 0.976 0.977 0.977 0.976 0.980 0.977 0.976 0.976 0.978 0.979
  20%: 0.959 0.963 0.958 0.968 0.964 0.964 0.963 0.963 0.960 0.963 0.964 0.960 0.962 0.960 0.962 0.965 0.958 0.959 0.959 0.963 0.957 0.959 0.959 0.958 0.965 0.960 0.959 0.958 0.962 0.964
  25%: 0.937 0.943 0.936 0.951 0.944 0.945 0.943 0.943 0.939 0.943 0.944 0.938 0.942 0.939 0.941 0.945 0.936 0.936 0.938 0.943 0.935 0.937 0.938 0.935 0.946 0.938 0.937 0.936 0.942 0.945
  30%: 0.912 0.920 0.910 0.931 0.922 0.923 0.920 0.920 0.914 0.920 0.922 0.913 0.919 0.915 0.917 0.923 0.910 0.911 0.913 0.920 0.909 0.912 0.913 0.908 0.924 0.913 0.911 0.911 0.918 0.922
  35%: 0.884 0.895 0.882 0.908 0.896 0.898 0.894 0.894 0.887 0.895 0.896 0.885 0.893 0.888 0.891 0.899 0.881 0.882 0.886 0.894 0.881 0.884 0.884 0.878 0.900 0.886 0.882 0.883 0.892 0.897
  40%: 0.853 0.868 0.851 0.883 0.867 0.872 0.865 0.865 0.857 0.867 0.869 0.853 0.864 0.859 0.861 0.872 0.849 0.851 0.856 0.865 0.850 0.853 0.854 0.845 0.873 0.857 0.852 0.851 0.863 0.869
  50%: 0.787 0.807 0.784 0.827 0.806 0.814 0.804 0.806 0.792 0.808 0.809 0.785 0.803 0.797 0.798 0.814 0.782 0.783 0.794 0.805 0.783 0.785 0.788 0.774 0.813 0.793 0.785 0.785 0.801 0.810

## classification, 1,500 rows, 30 seeds

| labels flipped | the reading rule 2 takes: range (mean, sd) | the reading rule 3 takes: range (mean, sd) | excluded | warned | kept in silence |
|---|---|---|---|---|---|
| 0.5% | 0.979 to 0.990 (0.984, 0.003) | 0.979 to 0.990 (0.984, 0.003) | 28 | 2 | 0 |
| 1% | 0.965 to 0.980 (0.972, 0.004) | 0.965 to 0.980 (0.972, 0.004) | 0 | 30 | 0 |
| 1.5% | 0.952 to 0.972 (0.961, 0.004) | 0.952 to 0.972 (0.961, 0.004) | 0 | 30 | 0 |
| 2% | 0.938 to 0.960 (0.949, 0.005) | 0.938 to 0.960 (0.949, 0.005) | 0 | 30 | 0 |
| 2.5% | 0.925 to 0.947 (0.936, 0.005) | 0.925 to 0.947 (0.936, 0.005) | 0 | 30 | 0 |
| 3% | 0.915 to 0.937 (0.926, 0.006) | 0.915 to 0.937 (0.926, 0.006) | 0 | 30 | 0 |
| 4% | 0.892 to 0.916 (0.904, 0.007) | 0.892 to 0.916 (0.904, 0.007) | 0 | 22 | 8 |
| 5% | 0.872 to 0.897 (0.883, 0.007) | 0.872 to 0.897 (0.883, 0.007) | 0 | 0 | 30 |

seed by seed (k = 0 to 29), the reading rule 3 takes:
  0.5%: 0.986 0.986 0.983 0.979 0.987 0.980 0.988 0.981 0.983 0.980 0.981 0.981 0.987 0.980 0.982 0.989 0.982 0.986 0.986 0.981 0.990 0.985 0.980 0.987 0.982 0.986 0.984 0.986 0.981 0.984
  1%: 0.975 0.970 0.966 0.966 0.974 0.968 0.972 0.974 0.971 0.969 0.965 0.972 0.976 0.970 0.971 0.977 0.971 0.974 0.975 0.965 0.976 0.969 0.970 0.980 0.972 0.979 0.969 0.974 0.968 0.971
  1.5%: 0.965 0.958 0.955 0.961 0.957 0.958 0.962 0.957 0.963 0.960 0.952 0.959 0.967 0.959 0.960 0.963 0.961 0.958 0.962 0.960 0.967 0.957 0.962 0.972 0.963 0.971 0.962 0.966 0.960 0.960
  2%: 0.958 0.946 0.944 0.946 0.943 0.945 0.955 0.948 0.950 0.949 0.938 0.949 0.955 0.945 0.944 0.950 0.951 0.942 0.949 0.949 0.960 0.942 0.947 0.957 0.948 0.955 0.951 0.954 0.946 0.945
  2.5%: 0.938 0.935 0.932 0.937 0.928 0.931 0.947 0.934 0.932 0.931 0.925 0.938 0.938 0.933 0.937 0.937 0.938 0.932 0.940 0.936 0.943 0.933 0.937 0.946 0.935 0.944 0.939 0.938 0.932 0.930
  3%: 0.922 0.922 0.922 0.932 0.919 0.922 0.934 0.924 0.921 0.928 0.915 0.925 0.931 0.924 0.933 0.930 0.928 0.918 0.932 0.927 0.937 0.920 0.924 0.930 0.923 0.933 0.932 0.928 0.919 0.924
  4%: 0.899 0.906 0.905 0.914 0.893 0.896 0.913 0.894 0.898 0.914 0.892 0.904 0.916 0.900 0.904 0.911 0.904 0.902 0.908 0.904 0.907 0.903 0.906 0.906 0.896 0.907 0.906 0.905 0.892 0.907
  5%: 0.878 0.885 0.891 0.895 0.872 0.874 0.889 0.872 0.882 0.889 0.872 0.877 0.896 0.882 0.879 0.897 0.876 0.881 0.886 0.884 0.885 0.888 0.879 0.878 0.874 0.884 0.892 0.886 0.876 0.885
```

The regression edges moved: the line reads a noisy copy a little better than
the tree did. Before D74 the copy was excluded on 28 of 30 seeds at 12% and
warned about on 15 of 30 at 30%, never at 35% (spec 10.2 as it stood). The
classification edges are what they were.

## 3. One row that carries most of the error (A3; D74 point 3, D76 point 3)

### 3.1 One cell planted, three rows for each seed

One cell of a numeric feature, or of the target, set to -999 or to 999999, in
a row drawn three times for each of the 30 seeds, on nine regression runs and
three classification runs. Where the row fell, whether the skill left the
band of its run, and what the result then said:

| run | cell changed | seeds 0 to 29, three draws each | row in holdout: runs, out of band, of those warned | row in training: runs, out of band, of those warned |
|---|---|---|---|---|
| signal_reg | feature:f1:-999 | `TLHLTL TLTLHL HLTLTL TLHLHL TLTLHL HLTLHL TLTLTL TLTLHL TLTLTL HLTLHL TLTLHL TLHLTL HLTLTL TLTLTL HLTLTL TLTLTL TLTLTL TLHLTL TLHLHL TLTLTL TLTLTL TLTLTL HLHLTL TLHLTL TLTLHL TLTLTL TLTLTL TLTLTL TLTLTL TLHLTL` | 23, 23, 23 | 67, 67, 67 |
| signal_reg | feature:f1:999999 | `TLTLTL TLHLHL TLTLTL TLTLHL TLTLTL TLTLTL TLHLTL HLHLTL TLHLTL TLHLTL TLHLTL HLTLTL HLTLTL TLTLHL TLHLTL TLTLHL TLTLTL HLHLTL TLTLHL HLTLTL TLTLTL TLTLHL TLTLTL TLTLHL HLTLTL TLTLTL TLHLHL TLHLTL TLTLTL TLTLTL` | 24, 24, 24 | 66, 66, 66 |
| signal_reg | target_-999 | `TLHLTL HLTLTL TLHLTL HLTLTL TLTLTL TLHLTL TLTLTL HLTLTL TLHLTL TLHLTL HLHLHL TLTLTL HLTLTL HLHLTL TLTLTL HLTLTL TLTLTL TLTLHL TLHLTL HLHLTL TLTLTL T.TLHL HLHLHL TLTLHL HLHLTL TLTLTL TLTLTL TLTLTL TLHLTL TLTLTL` | 27, 27, 27 | 63, 62, 62 |
| signal_reg | target_999999 | `HLTLTL TLTLTL TLTLTL TLTLHL HLHLHL TLTLHL TLTLTL TLTLTL TLTLTL HLTLTL HLHLHL TLTLTL TLTLTL TLTLTL TLHLTL TLHLTL TLHLTL TLTLTL TLTLTL TLTLTL TLTLHL TLTLTL HLTLHL TLTLHL TLTLHL TLTLTL TLTLHL TLHLHL HLTLTL TLHLTL` | 23, 23, 23 | 67, 67, 67 |
| noise_reg | feature:n001:-999 | `TVT.TV TLTVT. TVT.HL T.T.T. T.T.T. TLHLTL TVT.TL TLT.T. TLTLTV T.T.T. TLTLHL HVT.TV TLTLHL T.HLTL TLTLTL TLT.TV HVT.T. HVTVT. T.T.T. HLTLTV TVTVTL HVT.HV TLHLHL TLT.T. HLTLTL HVT.HV T.HLTV T.TLTL HLHLTL HLTLTV` | 20, 20, 20 | 70, 40, 40 |
| noise_reg | feature:n001:999999 | `TLTLTL HLTLHL TLTLHL TLHLTL TLHLTL TLHLTL TLTLHL HLHLTL TLTLTL HLTLTL TLHLTL TLHLTL TLTLTL TLTLTL HLTLHL TLTLHL TLTLTL HLTLTL TLTLTL TLTLHL TLHLTL TLTLTL TLTLTL TLHLHL TLHLTL TLTLHL TLTLTL TLTLTL HLHLTL HLTLHL` | 26, 26, 26 | 64, 64, 64 |
| noise_reg | target_-999 | `HLHLTL HLHLTL TLT.TL HLTLTL TLTLTL TLTLTL TLHLHL T.HLTL HLT.TL HLTLTL T.T.T. TLHLTL TLTLTL T.HLT. TLHLHL HLTLT. TLTLTL TLTLT. HLTLT. TLHLT. TLT.TL T.TLHL TLTLHL TLHLT. TLTLTL TLTLTL TLTLTL TLTLTL HLTLT. TLTLTL` | 21, 21, 21 | 69, 53, 53 |
| noise_reg | target_999999 | `HLTLTL TLTLHL HLTLHL TLHLHL HLTLHL HLTLTL TLTLHL TLTLTL HLTLTL TLTLHL TLHLTL TLTLTL HLTLTL TLTLTL HLTLTL TLTLTL TLTLHL HLTLTL HLTLTL TLHLTL TLTLHL TLTLTL TLTLHL HLTLTL HLHLTL TLTLTL TLHLTL TLTLHL HLTLTL TLTLTL` | 27, 27, 27 | 63, 63, 63 |
| leak_reg | feature:f1:-999 | `TLTLHL HLTLTL TLTLHL TLTLTL TLTLTL HLTLTL TLTLTL TLTLTL TLTLHL TLTLTL TLTLTL TLHLTL TLTLTL TLHLTL TLTLTL TLHLTL TLHLTL TLHLTL HLTLHL TLTLTL TLTLHL TLTLTL TLTLTL TLTLTL HLTLHL TLTLHL TLTLTL TLHLTL HLTLTL HLTLHL` | 20, 20, 20 | 70, 70, 70 |
| leak_reg | feature:f1:999999 | `TLTLTL TLTLTL TLTLHL HLHLTL TLTLHL TLTLTL HLTLTL HLTLHL TLHLHL TLTLTL TLHLHL TLTLTL TLHLTL TLTLHL TLTLTL TLTLTL TLTLTL HLTLTL HLHLTL TLTLTL TLHLTL TLTLTL HLTLHL TLTLTL HLHLTL TLTLHL HLHLTL TLHLTL TLTLTL TLHLTL` | 26, 26, 26 | 64, 64, 64 |
| leak_reg | target_-999 | `TLHLTL TLTLHL TLHLHL TLTLTL TLHLTL HLTLHL HLHLHL TLTLHL TLHLHL HLTLHL HLHLHL TLHLTL TLHLTL TLHLHL HLHLHL TLTLHL TLTLTL HLHLTL HLTLHL TLTLTL TLTLTL TLTLHL HLHLTL TLHLTL HLTLTL TLTLTL TLTLHL TLTLTL TLHLTL TLTLTL` | 37, 37, 37 | 53, 53, 53 |
| leak_reg | target_999999 | `TLTLHL TLTLTL TLTLTL HLHLTL TLTLTL TLTLTL HLTLHL TLTLTL TLTLTL HLHLHL TLTLTL TLHLTL HLTLTL TLTLHL TLHLHL TLHLTL TLTLTL HLHLHL TLTLTL HLTLTL TLTLHL TLTLTL TLTLTL TLTLTL TLTLTL TLTLHL TLTLTL HLTLHL TLTLTL TLTLTL` | 22, 22, 22 | 68, 68, 68 |
| random_walk | feature:level_prev:-999 | `TLTLHL TLHLTL TVHLTL TLTVTL TVTLTL HLHLTL TLTLTL TVTLHL TVTLTL HLTLTL TLTVTL TLTLHL HLHLHL TLTLTL TLTLTL TLHLHL TVTLTL TVTLHL TLTLTL TLHLTL TLTVTL HLTLTL TLHLTL TVTLHL TLTLTL TLTLTV TVHLTL TLTLTL TLTLTL HLTLTV` | 20, 20, 20 | 70, 70, 70 |
| random_walk | feature:level_prev:999999 | `HLHLTL TLHLTV TLTLTV TLTLTV TVTLTL HLHLTL TLTLTL HLHLTV TLTLTL HLTLTV TLHLTL TLTLTL TLTLHL TLTLTL HLTVTL TVTLHL TLTLTL TLTLTV HLHLTL HLHLTL TLHLTL TLHLTL HLTLHL TLTLTL TLTLTL HLTVTL TLTLTV TVTVHL HLTLTL HLTLTL` | 24, 24, 24 | 66, 66, 66 |
| random_walk | target_-999 | `TLT.TL TLTLHL TLT.HL HLHLTV TLTVTL TLT.T. H.TLH. H.TLTL HLTLTL HLTLHL TLTVTL HLHLTL TLTLTL HLTLHL TLTLH. TVT.TL TLTLHL HLTVHL TLH.TL TLTLT. HLTLTL H.T.TL TVTLTL T.T.HL HLTLTL TVTLTL TLTLTL TVTLHL H.TVH. TLTVTL` | 26, 18, 18 | 64, 55, 55 |
| random_walk | target_999999 | `TLTLH. TLTLTL TLTLTL TLTLTL H.H.TL TVTLTL TLH.H. TLTLH. TLH.TV H.TVTL H.H.H. H.TLTL TVTLH. H.TLTL H.TLTL TLTLTV TLTLTL H.TLTL TLTLTL TLTLTL TVH.TL TVH.TL TLTVTV TLTVH. TLTLTL TLTLTL TLH.H. TLH.TV TLH.H. TLTLTL` | 24, 0, 0 | 66, 66, 66 |
| int_stamps | feature:level_prev:-999 | `TLTLHL TLHLTL HLTLTL TLTLTL HLTLHL TLTLHL TVTLTL TLTLTV TVTLTL HLTLTL TLTLTL TLTLTV HLHLTL TLTVTL HLTLTL TLTLTV TLHLHL TLTLTL TLTLHL TVHLTL TLTVTV TLTLHL TLTVTL HLTLHL TLTLTL TLTLTL TLTLTL HLTLHL TLTLTL HLTLTL` | 20, 20, 20 | 70, 70, 70 |
| int_stamps | feature:level_prev:999999 | `TLTLTL TLTLTL HLTLTL TVHLTL TVTLTL HLHLTL TVTLTL HLTLHL TLTLTL TLTLHL TLTLTL TLTLTL TLTVHL TVTLTL TLHLTL TLHLTV TLHLTL TLTLHL TLTLTL TLHLHL TLTLTV TLTLTL HLTLTL HLHLTL HLTLTV HLTLTV TLTLTV HLTLTL TLHLTL HLTLTL` | 22, 22, 22 | 68, 68, 68 |
| int_stamps | target_-999 | `T.TLT. TLHLTL TLTVTL T.TLTL TLTLT. TLH.TL TLHLHL TVTLTL TLTLHL TLTLH. TLTLTV TLTLTL TLTLTL TLHLTV H.HLTL H.H.TL HLTLTV TLTLTL TVTLTL HLTLHL TLTLTL T.HLTL TLTLHL HLT.TL TLHLTL TLTLTL T.TLTL TLTLH. TVTLHL TLTLT.` | 20, 14, 14 | 70, 62, 62 |
| int_stamps | target_999999 | `TLTLTV TLH.H. TLTLTL TVTLTL TLTLTV H.TLTL H.TLTL TLTVTL H.H.TL TLH.H. TLTLTL TLTLTL TLTVTL TVTLH. TLTLTL H.TLTL TLH.TL H.TLTL TLTLTL TLTLTL H.H.H. H.TLH. TLTLTL TLTLH. TLTLTL H.H.TL TLH.H. TLH.TL H.TVTL TLTVTL` | 24, 0, 0 | 66, 66, 66 |
| flat_tail | feature:footfall:-999 | `T.T.H. T.T.T. H.T.H. T.T.T. T.T.T. T.T.T. T.T.T. T.T.T. T.H.T. T.T.H. T.H.T. T.T.T. T.H.H. H.T.H. T.T.T. H.H.H. T.H.T. T.T.T. T.T.H. T.H.T. T.T.T. T.H.H. T.H.H. T.T.T. T.T.T. H.T.T. T.H.T. T.T.T. T.T.T. T.T.T.` | 22 (the run has no band) | 68 (the run has no band) |
| flat_tail | feature:footfall:999999 | `T.T.T. T.T.T. T.T.H. T.T.H. T.T.T. H.T.T. T.T.H. T.T.T. T.H.T. T.T.T. T.T.T. T.T.T. T.T.H. T.T.T. T.T.H. H.T.T. H.T.H. T.T.H. T.T.H. T.T.T. T.T.T. T.T.T. T.T.T. T.T.T. H.T.T. T.T.H. T.T.T. H.T.H. T.T.T. T.H.T.` | 17 (the run has no band) | 73 (the run has no band) |
| flat_tail | target_-999 | `T.T.T. T.T.T. T.T.T. T.T.H. T.T.T. T.T.T. T.H.T. T.T.H. H.H.T. T.T.T. T.T.T. T.H.H. H.T.H. T.T.T. T.T.H. T.H.T. T.T.T. T.T.T. H.H.H. H.H.H. T.T.H. H.H.T. T.H.T. T.T.T. T.T.T. T.T.H. T.T.T. T.T.T. T.T.T. T.T.T.` | 22 (the run has no band) | 68 (the run has no band) |
| flat_tail | target_999999 | `T.H.T. T.T.T. T.T.T. T.H.T. T.T.H. T.H.H. T.T.T. T.T.T. T.T.H. T.H.H. H.H.T. H.H.T. T.T.T. T.T.H. H.H.T. T.T.T. H.H.T. T.T.T. T.T.T. T.T.T. T.H.T. H.H.T. T.T.T. T.T.T. T.T.H. H.H.T. T.H.H. H.T.T. H.H.T. T.T.T.` | 28 (the run has no band) | 62 (the run has no band) |
| panel | feature:level_prev:-999 | `HLTLHL TLHLTL HLTLTL TLTLTL TLTLTL TLTLTL TLHLHL TVTLTL TLHLTL TVTLHL TLTLTL TVHLTL TLTLTL TVTLTL TLTLHL TLHLTV TLTLHL TLTLTV TLTLTL TLTLHL TLHLHL TLTLHL TLTLTL TLTVHL TLHLTL HLTLTL HLTLTL HLHLHL TLTLHL TLHLTL` | 25, 25, 25 | 65, 65, 65 |
| panel | feature:level_prev:999999 | `TVTLHL TLTLTL TLTLTL TVTLTL TVTLTV TLTVTV TLTLHL TLTLTL TLHLTL TVTVTL TLTLTV TLHLTL TVTLTL TLTLHL TLTLTL TLHLHL HLTLTL TVTLTL TLTLTL HLTVHL HLTLHL TLTLTL TLTLHL HLHLHL TVHLTL TLTLHL HLHLTL TLHLTL TLTLTV TVTLTL` | 21, 21, 21 | 69, 69, 69 |
| panel | target_-999 | `TLTLHL TLTVTL TLHLTL TLTLTL TLTLHL TLTVHL HLTLTL TLTLTL TLTLTL TLHLHL HLHLTL HLTLHL TVTLHL TLTLTL TLHLTL HLTLTL TLTLHL TLTLTL TLTLTV TLHLTL TLTLTL TLTLTL HLTLTL TLHLTL TLTLTL TLTLTL TLTLHL TLHLHL TLTLTL TLTLHL` | 22, 22, 22 | 68, 68, 68 |
| panel | target_999999 | `TLH.TL TLTVTL TLTLTL H.TLTL TLTLH. H.TLTL H.H.TL TLTLH. TLH.TL TLH.TL H.TLTL H.H.H. TLTLTL TLTLTL H.TLTL TLTLTL TLH.TL TLTLH. TLTLTL H.TLTL TLTLTL TVTLTL TLTLTL H.TLTV H.TLTV H.TLH. TLTLTL TLTLTL TLTLH. TVTLTV` | 22, 0, 0 | 68, 68, 68 |
| panel | feature:level_prev:empty | `T.T.T. HoT.T. HLT.Ho HoHLT. T.T.T. T.T.T. T.T.T. T.T.HL T.T.T. T.T.T. HoT.T. HLHoT. T.T.T. T.T.T. T.T.Ho T.T.T. T.T.T. T.T.T. T.T.T. H.T.T. T.T.HL T.T.T. T.T.T. HLT.T. HoHoT. HLT.T. T.T.T. T.HoHo T.H.T. T.T.Ho` | 20, 18, 7 | 70, 0, 0 |
| dominant_entity | feature:n1:-999 | `HLT.T. HVHVT. T.T.T. T.T.T. T.T.T. HLT.T. T.T.T. HLT.T. T.T.T. HLHLT. T.HVT. T.T.T. T.H.T. T.T.T. T.T.T. HLT.T. T.T.HL T.HVT. T.T.HL T.T.T. HLT.T. T.T.HL T.T.T. T.T.HV T.T.T. T.T.HL HLT.T. T.T.T. T.T.HL HVHVT.` | 21, 20, 20 | 69, 0, 0 |
| dominant_entity | feature:n1:999999 | `T.HLT. T.T.T. T.T.T. T.T.T. T.T.HL T.T.T. T.T.T. T.T.T. T.T.HL T.T.T. T.T.T. T.T.T. T.HLT. T.T.T. T.HLT. T.T.T. T.T.T. HLT.T. T.T.T. T.T.T. T.T.T. HLT.T. HLHLHL T.T.T. T.HLT. T.T.T. HLHLT. T.T.T. T.T.HL T.T.HL` | 15, 15, 15 | 75, 0, 0 |
| dominant_entity | target_-999 | `TLH.TL H.TLTL TLTLTL H.TLTL TLTLH. TLTLTL TLTLTL TLTLTL TLTLTL H.TLTL TLTLTL TLTLTL H.TLTL TLTLH. H.H.TL TLH.TL TLH.TL TLH.TL H.TLTL TLH.H. TLH.TL H.TLTL H.TLTL H.H.TL H.TLH. TLTLTL TLTLTL TLTLTL TLTLTL TLTLTL` | 22, 0, 0 | 68, 68, 68 |
| dominant_entity | target_999999 | `TLTLTL TLTLTL TLTLTL H.TLTL TLTLH. TLTLTL TLTLTL H.TLH. TLTVH. TLTLH. TLTLTL TLH.TL TLH.TL TLTVTL H.H.TL TLTLTL TLH.H. TLTLH. H.TLH. H.TLH. TLTLTL TLH.H. TLTLTL H.H.TL H.TLTL TLTLTL TLH.TL TLTVTL H.TLTL TLTLTL` | 24, 0, 0 | 66, 66, 66 |
| quarter_entity | feature:n1:-999 | `T.T.T. T.T.T. T.T.H. T.H.T. T.T.T. T.T.T. T.T.T. T.T.T. T.T.T. T.H.H. T.T.T. H.T.H. T.H.T. T.T.T. H.T.T. T.H.T. T.H.T. T.T.T. T.T.T. T.T.T. T.T.H. T.T.T. H.T.H. T.T.T. T.T.H. T.T.T. T.T.T. T.T.T. T.T.T. T.H.T.` | 15 (the run has no band) | 75 (the run has no band) |
| quarter_entity | feature:n1:999999 | `T.T.T. T.T.T. T.T.H. T.T.T. T.T.T. T.T.T. T.T.T. T.T.H. T.T.T. T.T.H. T.H.H. T.T.T. T.T.H. H.T.T. H.T.T. T.T.T. H.T.T. H.H.T. T.H.T. T.H.H. T.T.T. T.T.T. H.T.T. T.T.T. T.T.T. T.H.H. T.T.T. T.H.T. T.T.T. T.H.T.` | 19 (the run has no band) | 71 (the run has no band) |
| quarter_entity | target_-999 | `T.T.H. H.T.H. T.T.T. H.T.T. T.T.T. T.H.T. T.T.H. T.T.H. T.T.T. T.T.H. T.T.T. T.T.T. T.H.T. T.T.H. T.H.T. T.T.T. H.T.T. H.H.T. H.T.T. T.T.H. T.H.T. T.T.H. H.H.T. H.T.T. T.T.T. T.T.H. H.H.T. T.H.T. T.T.H. T.T.T.` | 26 (the run has no band) | 64 (the run has no band) |
| quarter_entity | target_999999 | `T.H.T. H.T.T. T.T.T. T.T.T. H.T.T. T.T.T. T.T.H. T.T.H. T.T.T. T.T.T. T.T.T. H.H.T. T.T.T. T.T.T. T.H.T. T.T.H. T.T.T. T.T.T. T.H.T. T.T.H. T.T.T. T.T.T. H.T.T. H.T.T. T.T.T. T.T.H. T.T.H. T.H.T. T.H.T. H.T.H.` | 19 (the run has no band) | 71 (the run has no band) |
| signal_clf | feature:f1:-999 | `H.T.T. T.H.H. TVH.TV T.H.TV TVH.TV H.T.T. H.T.H. TVT.T. H.H.TV T.T.TV TVTVTV H.TVH. TVH.H. T.H.H. T.T.TV T.H.H. T.T.H. T.T.T. T.TVT. T.TVT. H.TVTV TVT.H. H.T.T. TVTVTV T.H.H. TVT.TV TVT.H. T.T.T. TVH.TV H.H.TV` | 29, 0, 0 | 61, 28, 28 |
| signal_clf | feature:f1:999999 | `H.T.T. TVTVTV TVTVTV TVTVTV TVTVH. TVTVTV TVH.H. TVH.TV TVTVTV TVTVTV TVTVTV TVH.TV H.TVTV TVTVTV TVH.TV H.H.TV TVH.TV T.T.T. H.H.H. TVTVTV H.TVTV TVTVTV T.H.T. H.TVTV H.T.T. TVH.TV TVH.H. TVTVTV TVTVTV TVTVTV` | 21, 0, 0 | 69, 60, 60 |
| messy | feature:f1:-999 | `H.TVT. T.H.H. TVH.TV T.H.T. T.H.T. TVH.H. TVTVT. H.TVT. H.T.TV H.H.T. T.TVH. TVT.H. T.H.H. TVTVTV TVH.T. H.T.T. H.TVTV T.TVTV T.T.H. TVTVTV TVH.T. H.T.H. TVTVTV T.H.H. TVTVTV H.T.TV TVTVTV TVTVT. TVH.TV H.TVTV` | 28, 0, 0 | 62, 38, 38 |
| messy | feature:f1:999999 | `H.TVTV TVTVTV TVH.TV TVTVTV TVTVTV TVH.TV TVTVH. TVTVTV TVH.TV TVTVTV TVH.H. H.TVTV TVH.TV H.H.H. TVH.H. T.T.T. TVH.TV TVTVTV H.TVH. TVH.TV TVTVH. TVTVTV TVTVH. H.TVTV TVTVTV TVTVH. TVTVTV TVTVTV TVTVTV H.TVTV` | 23, 0, 0 | 67, 64, 64 |
| binary_numeric | feature:f1:-999 | `H.T.H. TVTVTV T.T.TV T.T.TV TVTVTV H.H.H. TVTVT. T.H.T. T.T.H. H.T.H. T.T.T. T.H.TV T.T.H. TVH.T. T.T.H. TVH.T. TVTVT. T.H.TV T.TVH. H.TVTV T.TVTV T.TVTV T.H.H. H.T.T. TVTVTV H.T.H. T.TVTV TVT.TV H.TVH. TVTVTV` | 24, 0, 0 | 66, 34, 34 |
| binary_numeric | feature:f1:999999 | `TVTVT. H.H.TV TVH.TV H.H.TV H.TVTV TVTVTV TVTVH. T.T.T. TVTVTV TVTVTV TVTVH. TVTVTV TVTVTV TVH.TV T.T.T. TVTVH. TVTVTV H.TVTV TVTVTV TVTVTV TVTVTV TVTVTV T.H.T. TVTVTV TVTVH. TVTVTV H.H.TV TVTVTV TVTVTV TVH.TV` | 16, 0, 0 | 74, 65, 65 |

| cell | runs | n | row in holdout | of those out of band | of those: a `largest_row_dominates` record, `extreme_value` only, nothing | row in training | of those out of band | of those: a holdout statement, a selection-fold statement, `extreme_value` only, nothing |
|---|---|---|---|---|---|---|---|---|
| feature | regression, random scheme | 540 | 139 | 139 | 132, 7, 0 | 401 | 371 | 0, 357, 14, 0 |
| feature | regression, time order | 540 | 132 | 132 | 132, 0, 0 | 408 | 408 | 0, 340, 68, 0 |
| feature | classification | 540 | 141 | 0 | 0, 0, 0 | 399 | 289 | 0, 0, 289, 0 |
| target | regression, random scheme | 540 | 157 | 157 | 157, 0, 0 | 383 | 366 | 39, 366, 0, 0 |
| target | regression, time order | 540 | 138 | 54 | 54, 0, 0 | 402 | 385 | 18, 335, 46, 0 |

In the last table every run whose skill left its band is named by a
`largest_row_dominates` record or by an `extreme_value` warning: the last
count of each group is 0.

### 3.2 The largest row's share on clean tables

| rows | holdout rows | errors | model: largest row's share | baseline: largest row's share | either above 0.5 | either above 0.25 |
|---|---|---|---|---|---|---|
| 100 | 25 | normal | 0.129 to 0.381 (mean 0.219) | 0.127 to 0.365 (mean 0.213) | 0 of 30 | 11 of 30 |
| 100 | 25 | t3 | 0.164 to 0.565 (mean 0.315) | 0.136 to 0.371 (mean 0.223) | 2 of 30 | 23 of 30 |
| 100 | 25 | lognormal | 0.162 to 0.977 (mean 0.456) | 0.102 to 0.980 (mean 0.499) | 14 of 30 | 25 of 30 |
| 200 | 50 | normal | 0.076 to 0.306 (mean 0.136) | 0.078 to 0.265 (mean 0.141) | 0 of 30 | 2 of 30 |
| 200 | 50 | t3 | 0.106 to 0.784 (mean 0.256) | 0.063 to 0.550 (mean 0.169) | 4 of 30 | 9 of 30 |
| 200 | 50 | lognormal | 0.076 to 0.987 (mean 0.453) | 0.067 to 0.984 (mean 0.484) | 14 of 30 | 26 of 30 |
| 400 | 100 | normal | 0.051 to 0.132 (mean 0.081) | 0.044 to 0.129 (mean 0.078) | 0 of 30 | 0 of 30 |
| 400 | 100 | t3 | 0.091 to 0.826 (mean 0.244) | 0.048 to 0.464 (mean 0.125) | 3 of 30 | 8 of 30 |
| 400 | 100 | lognormal | 0.108 to 0.943 (mean 0.426) | 0.100 to 0.937 (mean 0.422) | 9 of 30 | 24 of 30 |
| 1,000 | 250 | normal | 0.026 to 0.066 (mean 0.036) | 0.023 to 0.056 (mean 0.035) | 0 of 30 | 0 of 30 |
| 1,000 | 250 | t3 | 0.034 to 0.968 (mean 0.156) | 0.029 to 0.906 (mean 0.088) | 1 of 30 | 3 of 30 |
| 1,000 | 250 | lognormal | 0.092 to 0.886 (mean 0.329) | 0.093 to 0.872 (mean 0.317) | 8 of 30 | 16 of 30 |
| 4,000 | 1000 | normal | 0.014 to 0.032 (mean 0.021) | 0.014 to 0.030 (mean 0.021) | 0 of 30 | 0 of 30 |
| 4,000 | 1000 | t3 | 0.037 to 0.527 (mean 0.130) | 0.019 to 0.290 (mean 0.054) | 1 of 30 | 2 of 30 |
| 4,000 | 1000 | lognormal | 0.078 to 0.657 (mean 0.272) | 0.069 to 0.621 (mean 0.257) | 3 of 30 | 16 of 30 |

With errors that are normal the share never passes 0.5, at any size. With
heavy tails it does, and the statement is then true of the table.

### 3.3 The cut-off of `extreme_value`

```text
the 3 outermost distinct values at each end; the spread of the 10 values after; cut-offs K = 20, 25, 30, 40, 50

## 1. One cell planted

| table | numeric features and regression target | cell | named at K = 20 / 25 / 30 / 40 / 50, of seeds x columns | smallest ratio |
|---|---|---|---|---|
| signal_clf | f1, f2, n1, n2, n3 | -999 | 150 / 150 / 150 / 150 / 150 of 150 | 537 |
| signal_clf | f1, f2, n1, n2, n3 | 999999 | 150 / 150 / 150 / 150 / 150 of 150 | 5.79e+05 |
| signal_reg | f1, f2, n1, n2, n3, amount | -999 | 180 / 180 / 180 / 180 / 180 of 180 | 278 |
| signal_reg | f1, f2, n1, n2, n3, amount | 999999 | 180 / 180 / 180 / 180 / 180 of 180 | 2.21e+05 |
| noise_clf | n001 ... n300 (300) | -999 | 9000 / 9000 / 9000 / 9000 / 9000 of 9000 | 340 |
| noise_clf | n001 ... n300 (300) | 999999 | 9000 / 9000 / 9000 / 9000 / 9000 of 9000 | 3.52e+05 |
| noise_reg | n001 ... amount (301) | -999 | 9030 / 9030 / 9030 / 9030 / 9030 of 9030 | 346 |
| noise_reg | n001 ... amount (301) | 999999 | 9030 / 9030 / 9030 / 9030 / 9030 of 9030 | 3.23e+05 |
| imbalanced | n001, n002, n003, n004, n005, n006 | -999 | 180 / 180 / 180 / 180 / 180 of 180 | 523 |
| imbalanced | n001, n002, n003, n004, n005, n006 | 999999 | 180 / 180 / 180 / 180 / 180 of 180 | 4.18e+05 |
| leak | f1, f2, n1, n2, n3, strong_feature | -999 | 180 / 180 / 180 / 180 / 180 of 180 | 247 |
| leak | f1, f2, n1, n2, n3, strong_feature | 999999 | 180 / 180 / 180 / 180 / 180 of 180 | 3.2e+05 |
| leak_reg | f1, f2, n1, n2, n3, amount | -999 | 180 / 180 / 180 / 180 / 180 of 180 | 265 |
| leak_reg | f1, f2, n1, n2, n3, amount | 999999 | 180 / 180 / 180 / 180 / 180 of 180 | 2.05e+05 |
| near_leak | f1, f2, n1, n2, n3 | -999 | 150 / 150 / 150 / 150 / 150 of 150 | 445 |
| near_leak | f1, f2, n1, n2, n3 | 999999 | 150 / 150 / 150 / 150 / 150 of 150 | 5.55e+05 |
| binary_numeric | f1, f2, n1, n2, n3 | -999 | 150 / 150 / 150 / 150 / 150 of 150 | 583 |
| binary_numeric | f1, f2, n1, n2, n3 | 999999 | 150 / 150 / 150 / 150 / 150 of 150 | 5.23e+05 |
| ambiguous_target | (the run stops: task_ambiguous) | | | |
| messy | f1, f2, n1, spend, visits | -999 | 120 / 120 / 120 / 120 / 120 of 150 | 1.59 |
| messy | f1, f2, n1, spend, visits | 999999 | 150 / 150 / 150 / 150 / 150 of 150 | 778 |
| long_rows | f1, f2, n1, n2, n3 | -999 | 150 / 150 / 150 / 150 / 150 of 150 | 544 |
| long_rows | f1, f2, n1, n2, n3 | 999999 | 150 / 150 / 150 / 150 / 150 of 150 | 4.75e+05 |
| dup_noise | (the run stops: holdout_copies_in_training) | | | |
| time_blocks | b1, b2, b3, b4, b5, b6, b7, b8 | -999 | 240 / 240 / 240 / 240 / 240 of 240 | 499 |
| time_blocks | b1, b2, b3, b4, b5, b6, b7, b8 | 999999 | 240 / 240 / 240 / 240 / 240 of 240 | 5e+05 |
| time_dayfirst | (the run stops: time_column_invalid) | | | |
| random_walk | level_prev, level | -999 | 60 / 60 / 60 / 60 / 60 of 60 | 163 |
| random_walk | level_prev, level | 999999 | 60 / 60 / 60 / 60 / 60 of 60 | 2.47e+05 |
| int_stamps | level_prev, level | -999 | 60 / 60 / 60 / 60 / 60 of 60 | 119 |
| int_stamps | level_prev, level | 999999 | 60 / 60 / 60 / 60 / 60 of 60 | 3.07e+05 |
| flat_tail | footfall, promo, sales | -999 | 90 / 90 / 90 / 90 / 90 of 90 | 188 |
| flat_tail | footfall, promo, sales | 999999 | 90 / 90 / 90 / 90 / 90 of 90 | 8.55e+04 |
| entities | m1, m2, m3, m4, m5, m6 | -999 | 180 / 180 / 180 / 180 / 180 of 180 | 669 |
| entities | m1, m2, m3, m4, m5, m6 | 999999 | 180 / 180 / 180 / 180 / 180 of 180 | 6.36e+05 |
| panel | level_prev, level | -999 | 60 / 60 / 60 / 60 / 60 of 60 | 176 |
| panel | level_prev, level | 999999 | 60 / 60 / 60 / 60 / 60 of 60 | 2.43e+05 |
| noise_small | n001 ... n010 (10) | -999 | 300 / 300 / 300 / 300 / 300 of 300 | 416 |
| noise_small | n001 ... n010 (10) | 999999 | 300 / 300 / 300 / 300 / 300 of 300 | 3.72e+05 |
| signal_clf_small | f1, f2, n1, n2, n3 | -999 | 150 / 150 / 150 / 150 / 150 of 150 | 365 |
| signal_clf_small | f1, f2, n1, n2, n3 | 999999 | 150 / 150 / 150 / 150 / 150 of 150 | 3.99e+05 |
| signal_reg_small | f1, f2, n1, n2, n3, amount | -999 | 180 / 180 / 180 / 180 / 180 of 180 | 207 |
| signal_reg_small | f1, f2, n1, n2, n3, amount | 999999 | 180 / 180 / 180 / 180 / 180 of 180 | 2.25e+05 |
| few_valued | none | | | |
| dominant_entity | n1, n2, n3, n4, amount | -999 | 150 / 150 / 150 / 150 / 150 of 150 | 463 |
| dominant_entity | n1, n2, n3, n4, amount | 999999 | 150 / 150 / 150 / 150 / 150 of 150 | 4.96e+05 |
| day_numbers | n1, n2, n3, n4 | -999 | 120 / 120 / 120 / 120 / 120 of 120 | 424 |
| day_numbers | n1, n2, n3, n4 | 999999 | 120 / 120 / 120 / 120 / 120 of 120 | 4.04e+05 |
| day_sentinel | n1, n2, n3, n4 | -999 | 120 / 120 / 120 / 120 / 120 of 120 | 444 |
| day_sentinel | n1, n2, n3, n4 | 999999 | 120 / 120 / 120 / 120 / 120 of 120 | 4.11e+05 |
| few_valued_leak | (the run stops: holdout_copies_in_training) | | | |
| large_entity | (the run stops: fast_profile_group_too_large) | | | |
| quarter_entity | n1, n2, n3, n4, amount | -999 | 150 / 150 / 150 / 150 / 150 of 150 | 450 |
| quarter_entity | n1, n2, n3, n4, amount | 999999 | 150 / 150 / 150 / 150 / 150 of 150 | 5.02e+05 |

Every column and cell not named on all seeds at some cut-off (named of 30 seeds at K = 20 / 25 / 30 / 40 / 50; smallest ratio):
  messy.visits with -999: 0 / 0 / 0 / 0 / 0; 1.59

The weakest planted cell that is named on every seed at every cut-off: messy.spend with -999, smallest ratio 51.8

## 2. Nothing planted

| table | columns x seeds where a value is named at K = 20 / 25 / 30 / 40 / 50 | largest ratio |
|---|---|---|
| signal_clf | 0 / 0 / 0 / 0 / 0 of 150 | 2.46 |
| signal_reg | 0 / 0 / 0 / 0 / 0 of 180 | 3.07 |
| noise_clf | 0 / 0 / 0 / 0 / 0 of 9000 | 6.67 |
| noise_reg | 0 / 0 / 0 / 0 / 0 of 9030 | 8.24 |
| imbalanced | 0 / 0 / 0 / 0 / 0 of 180 | 3.02 |
| leak | 0 / 0 / 0 / 0 / 0 of 180 | 4.22 |
| leak_reg | 0 / 0 / 0 / 0 / 0 of 180 | 5.03 |
| near_leak | 0 / 0 / 0 / 0 / 0 of 150 | 3.83 |
| binary_numeric | 0 / 0 / 0 / 0 / 0 of 150 | 4.17 |
| messy | 0 / 0 / 0 / 0 / 0 of 150 | 2.52 |
| long_rows | 0 / 0 / 0 / 0 / 0 of 150 | 2.9 |
| time_blocks | 0 / 0 / 0 / 0 / 0 of 240 | 1 |
| random_walk | 0 / 0 / 0 / 0 / 0 of 60 | 1.61 |
| int_stamps | 0 / 0 / 0 / 0 / 0 of 60 | 1.46 |
| flat_tail | 0 / 0 / 0 / 0 / 0 of 90 | 2.87 |
| entities | 0 / 0 / 0 / 0 / 0 of 180 | 1.57 |
| panel | 0 / 0 / 0 / 0 / 0 of 60 | 1.09 |
| noise_small | 0 / 0 / 0 / 0 / 0 of 300 | 4.15 |
| signal_clf_small | 0 / 0 / 0 / 0 / 0 of 150 | 3.33 |
| signal_reg_small | 0 / 0 / 0 / 0 / 0 of 180 | 3.07 |
| dominant_entity | 0 / 0 / 0 / 0 / 0 of 150 | 3.73 |
| day_numbers | 0 / 0 / 0 / 0 / 0 of 120 | 5.51 |
| day_sentinel | 0 / 0 / 0 / 0 / 0 of 120 | 4.14 |
| quarter_entity | 0 / 0 / 0 / 0 / 0 of 150 | 4.88 |

## 3. examples/customers.csv

monthly_spend: named at K = 20 / 25 / 30 / 40 / 50: [False, False, False, False, False]; largest ratio 0.648
support_calls: named at K = 20 / 25 / 30 / 40 / 50: [False, False, False, False, False]; largest ratio 0.167
age: named at K = 20 / 25 / 30 / 40 / 50: [False, False, False, False, False]; largest ratio 0.1

## 4. Long tails

| column | rows | a value named at K = 20 / 25 / 30 / 40 / 50, of seeds | largest ratio |
|---|---|---|---|
| normal | 200 | 0 / 0 / 0 / 0 / 0 of 30 | 2.14 |
| log-normal, sigma 1 | 200 | 0 / 0 / 0 / 0 / 0 of 30 | 9.71 |
| log-normal, sigma 1.5 | 200 | 0 / 0 / 0 / 0 / 0 of 30 | 12.3 |
| log-normal, sigma 2 | 200 | 0 / 0 / 0 / 0 / 0 of 30 | 8.88 |
| traded volume (log-normal(13, 1.5), rounded) | 200 | 0 / 0 / 0 / 0 / 0 of 30 | 15 |
| Pareto, shape 2.5 | 200 | 0 / 0 / 0 / 0 / 0 of 30 | 7.55 |
| Pareto, shape 1.5 | 200 | 1 / 1 / 0 / 0 / 0 of 30 | 27.2 |
| Student t, 3 degrees of freedom | 200 | 0 / 0 / 0 / 0 / 0 of 30 | 4.9 |
| exponential | 200 | 0 / 0 / 0 / 0 / 0 of 30 | 1.89 |
| a count (Poisson, mean 3) | 200 | 0 / 0 / 0 / 0 / 0 of 30 | 0.429 |
| normal | 2,000 | 0 / 0 / 0 / 0 / 0 of 30 | 2.03 |
| log-normal, sigma 1 | 2,000 | 0 / 0 / 0 / 0 / 0 of 30 | 2.5 |
| log-normal, sigma 1.5 | 2,000 | 0 / 0 / 0 / 0 / 0 of 30 | 14.5 |
| log-normal, sigma 2 | 2,000 | 0 / 0 / 0 / 0 / 0 of 30 | 12.2 |
| traded volume (log-normal(13, 1.5), rounded) | 2,000 | 0 / 0 / 0 / 0 / 0 of 30 | 6.59 |
| Pareto, shape 2.5 | 2,000 | 0 / 0 / 0 / 0 / 0 of 30 | 6.63 |
| Pareto, shape 1.5 | 2,000 | 2 / 1 / 1 / 1 / 1 of 30 | 211 |
| Student t, 3 degrees of freedom | 2,000 | 0 / 0 / 0 / 0 / 0 of 30 | 9.36 |
| exponential | 2,000 | 0 / 0 / 0 / 0 / 0 of 30 | 6.18 |
| a count (Poisson, mean 3) | 2,000 | 0 / 0 / 0 / 0 / 0 of 30 | 0.3 |
| normal | 20,000 | 0 / 0 / 0 / 0 / 0 of 30 | 2.21 |
| log-normal, sigma 1 | 20,000 | 0 / 0 / 0 / 0 / 0 of 30 | 3.94 |
| log-normal, sigma 1.5 | 20,000 | 0 / 0 / 0 / 0 / 0 of 30 | 8.89 |
| log-normal, sigma 2 | 20,000 | 0 / 0 / 0 / 0 / 0 of 30 | 6.63 |
| traded volume (log-normal(13, 1.5), rounded) | 20,000 | 0 / 0 / 0 / 0 / 0 of 30 | 4.76 |
| Pareto, shape 2.5 | 20,000 | 0 / 0 / 0 / 0 / 0 of 30 | 3.95 |
| Pareto, shape 1.5 | 20,000 | 3 / 3 / 3 / 0 / 0 of 30 | 35.9 |
| Student t, 3 degrees of freedom | 20,000 | 0 / 0 / 0 / 0 / 0 of 30 | 5.17 |
| exponential | 20,000 | 0 / 0 / 0 / 0 / 0 of 30 | 2.25 |
| a count (Poisson, mean 3) | 20,000 | 0 / 0 / 0 / 0 / 0 of 30 | 0.3 |
| normal | 100,000 | 0 / 0 / 0 / 0 / 0 of 30 | 3.38 |
| log-normal, sigma 1 | 100,000 | 0 / 0 / 0 / 0 / 0 of 30 | 3.11 |
| log-normal, sigma 1.5 | 100,000 | 0 / 0 / 0 / 0 / 0 of 30 | 4.41 |
| log-normal, sigma 2 | 100,000 | 0 / 0 / 0 / 0 / 0 of 30 | 3.94 |
| traded volume (log-normal(13, 1.5), rounded) | 100,000 | 0 / 0 / 0 / 0 / 0 of 30 | 9.35 |
| Pareto, shape 2.5 | 100,000 | 0 / 0 / 0 / 0 / 0 of 30 | 6.61 |
| Pareto, shape 1.5 | 100,000 | 1 / 0 / 0 / 0 / 0 of 30 | 23.5 |
| Student t, 3 degrees of freedom | 100,000 | 0 / 0 / 0 / 0 / 0 of 30 | 10.7 |
| exponential | 100,000 | 0 / 0 / 0 / 0 / 0 of 30 | 3.46 |
| a count (Poisson, mean 3) | 100,000 | 0 / 0 / 0 / 0 / 0 of 30 | 0.3 |
```

30 is adopted: the largest ratio of a column that must stay quiet is 15 (a
log-normal column), the smallest of a planted cell that must be named is 52
(`spend` of `messy` with -999), so the cut-off is a factor of 2.0 above the
one and 1.7 below the other. A Pareto column of shape 1.5 is named on some
seeds at every cut-off; D76 accepts that as truthful.

### 3.4 The cell that is not named

| `messy` | runs | the value named | skill | inside 0.55 to 0.91 |
|---|---|---|---|---|
| as it is | 30 | 0 | +0.670 to +0.826 | 30 of 30 |
| feature:visits:-999 | 90 | 0 | +0.670 to +0.826 | 90 of 90 |
| feature:visits:999999 | 90 | 90 | +0.670 to +0.827 | 90 of 90 |
| feature:spend:-999 | 90 | 90 | +0.663 to +0.829 | 90 of 90 |
| feature:spend:999999 | 90 | 90 | +0.665 to +0.823 | 90 of 90 |

`visits` of `messy` holds whole numbers from 1,000 to 100,000; -999 is just
below its smallest value and moves no number. D76 point 3: the pair asserts
that the run is in band, or the value is named.

## 4. The identifier rule (A5; D74 point 4)

The rule as it stood (`old`), the second clause alone at three widths, and the
union adopted (`d50+ord`): on measures, which must not be called identifiers,
and on identifiers in every form found.

```text
## Measures: none is an identifier

| column | rows | old | dense50 | dense25 | dense10 | d50+ord |
|---|---|---|---|---|---|---|
| salary, rounded to the pound | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| salary, rounded to the pound, the table sorted by it | 200 | 18 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 18 of 30 |
| price in pence | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| price in pence, the table sorted by it | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| traded volume | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| traded volume, the table sorted by it | 200 | 30 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 |
| age in years | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| age in years, the table sorted by it | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a count (Poisson, mean 40) | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a count (Poisson, mean 40), the table sorted by it | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a score from 0 to n | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a score from 0 to n, the table sorted by it | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a distinct measure, 10% of its range | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 0 of 30 |
| a distinct measure, 10% of its range, the table sorted by it | 200 | 30 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 |
| a distinct measure, 33% of its range | 200 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 0 of 30 |
| a distinct measure, 33% of its range, the table sorted by it | 200 | 30 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| salary, rounded to the pound | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| salary, rounded to the pound, the table sorted by it | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| price in pence | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| price in pence, the table sorted by it | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| traded volume | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| traded volume, the table sorted by it | 2,000 | 10 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 10 of 30 |
| age in years | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| age in years, the table sorted by it | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a count (Poisson, mean 40) | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a count (Poisson, mean 40), the table sorted by it | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a score from 0 to n | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a score from 0 to n, the table sorted by it | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a distinct measure, 10% of its range | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 0 of 30 |
| a distinct measure, 10% of its range, the table sorted by it | 2,000 | 30 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 |
| a distinct measure, 33% of its range | 2,000 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 0 of 30 |
| a distinct measure, 33% of its range, the table sorted by it | 2,000 | 30 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| salary, rounded to the pound | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| salary, rounded to the pound, the table sorted by it | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| price in pence | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| price in pence, the table sorted by it | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| traded volume | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| traded volume, the table sorted by it | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| age in years | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| age in years, the table sorted by it | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a count (Poisson, mean 40) | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a count (Poisson, mean 40), the table sorted by it | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a score from 0 to n | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a score from 0 to n, the table sorted by it | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a distinct measure, 10% of its range | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 0 of 30 |
| a distinct measure, 10% of its range, the table sorted by it | 20,000 | 30 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 |
| a distinct measure, 33% of its range | 20,000 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 0 of 30 |
| a distinct measure, 33% of its range, the table sorted by it | 20,000 | 30 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| salary, rounded to the pound | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| salary, rounded to the pound, the table sorted by it | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| price in pence | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| price in pence, the table sorted by it | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| traded volume | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| traded volume, the table sorted by it | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| age in years | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| age in years, the table sorted by it | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a count (Poisson, mean 40) | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a count (Poisson, mean 40), the table sorted by it | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a score from 0 to n | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a score from 0 to n, the table sorted by it | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| a distinct measure, 10% of its range | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 0 of 30 |
| a distinct measure, 10% of its range, the table sorted by it | 100,000 | 30 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 |
| a distinct measure, 33% of its range | 100,000 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 0 of 30 |
| a distinct measure, 33% of its range, the table sorted by it | 100,000 | 30 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |

## Identifiers

| column | rows | old | dense50 | dense25 | dense10 | d50+ord |
|---|---|---|---|---|---|---|
| consecutive, in file order | 200 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, in order, 1% blank | 200 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, shuffled | 200 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, shuffled, 1% blank | 200 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 3% of the rows deleted, in order | 200 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 3% of the rows deleted, shuffled | 200 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, one row repeated, in order | 200 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, one row repeated, shuffled | 200 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 1% of the rows repeated, in order | 200 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 1% of the rows repeated, shuffled | 200 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 5% of the rows repeated, in order | 200 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 5% of the rows repeated, shuffled | 200 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 10% of the rows repeated, in order | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| consecutive, 10% of the rows repeated, shuffled | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| every row present twice, shuffled | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| 30% of the records gone, in order | 200 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 30% of the records gone, shuffled | 200 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 60% of the records gone, in order | 200 | 30 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 60% of the records gone, shuffled | 200 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 0 of 30 |
| increasing, gaps of 1 to 3, in order | 200 | 30 of 30 | 15 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| increasing, gaps of 1 to 3, shuffled | 200 | 0 of 30 | 15 of 30 | 30 of 30 | 30 of 30 | 15 of 30 |
| increasing, gaps of 1 to 10, in order | 200 | 30 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 |
| increasing, gaps of 1 to 10, shuffled | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 0 of 30 |
| increasing, steps of 7, in order | 200 | 30 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 |
| increasing, steps of 7, shuffled | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 0 of 30 |
| increasing, steps of 1,000, in order | 200 | 30 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 |
| increasing, steps of 1,000, shuffled | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| random, unordered (nine digits) | 200 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| consecutive, in file order | 2,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, in order, 1% blank | 2,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, shuffled | 2,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, shuffled, 1% blank | 2,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 3% of the rows deleted, in order | 2,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 3% of the rows deleted, shuffled | 2,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, one row repeated, in order | 2,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, one row repeated, shuffled | 2,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 1% of the rows repeated, in order | 2,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 1% of the rows repeated, shuffled | 2,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 5% of the rows repeated, in order | 2,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 5% of the rows repeated, shuffled | 2,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 10% of the rows repeated, in order | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| consecutive, 10% of the rows repeated, shuffled | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| every row present twice, shuffled | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| 30% of the records gone, in order | 2,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 30% of the records gone, shuffled | 2,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 60% of the records gone, in order | 2,000 | 30 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 60% of the records gone, shuffled | 2,000 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 0 of 30 |
| increasing, gaps of 1 to 3, in order | 2,000 | 30 of 30 | 15 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| increasing, gaps of 1 to 3, shuffled | 2,000 | 0 of 30 | 15 of 30 | 30 of 30 | 30 of 30 | 15 of 30 |
| increasing, gaps of 1 to 10, in order | 2,000 | 30 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 |
| increasing, gaps of 1 to 10, shuffled | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 0 of 30 |
| increasing, steps of 7, in order | 2,000 | 30 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 |
| increasing, steps of 7, shuffled | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 0 of 30 |
| increasing, steps of 1,000, in order | 2,000 | 30 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 |
| increasing, steps of 1,000, shuffled | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| random, unordered (nine digits) | 2,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| consecutive, in file order | 20,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, in order, 1% blank | 20,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, shuffled | 20,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, shuffled, 1% blank | 20,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 3% of the rows deleted, in order | 20,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 3% of the rows deleted, shuffled | 20,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, one row repeated, in order | 20,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, one row repeated, shuffled | 20,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 1% of the rows repeated, in order | 20,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 1% of the rows repeated, shuffled | 20,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 5% of the rows repeated, in order | 20,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 5% of the rows repeated, shuffled | 20,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 10% of the rows repeated, in order | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| consecutive, 10% of the rows repeated, shuffled | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| every row present twice, shuffled | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| 30% of the records gone, in order | 20,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 30% of the records gone, shuffled | 20,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 60% of the records gone, in order | 20,000 | 30 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 60% of the records gone, shuffled | 20,000 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 0 of 30 |
| increasing, gaps of 1 to 3, in order | 20,000 | 30 of 30 | 15 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| increasing, gaps of 1 to 3, shuffled | 20,000 | 0 of 30 | 15 of 30 | 30 of 30 | 30 of 30 | 15 of 30 |
| increasing, gaps of 1 to 10, in order | 20,000 | 30 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 |
| increasing, gaps of 1 to 10, shuffled | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 0 of 30 |
| increasing, steps of 7, in order | 20,000 | 30 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 |
| increasing, steps of 7, shuffled | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 0 of 30 |
| increasing, steps of 1,000, in order | 20,000 | 30 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 |
| increasing, steps of 1,000, shuffled | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| random, unordered (nine digits) | 20,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| consecutive, in file order | 100,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, in order, 1% blank | 100,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, shuffled | 100,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, shuffled, 1% blank | 100,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 3% of the rows deleted, in order | 100,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 3% of the rows deleted, shuffled | 100,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, one row repeated, in order | 100,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, one row repeated, shuffled | 100,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 1% of the rows repeated, in order | 100,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 1% of the rows repeated, shuffled | 100,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 5% of the rows repeated, in order | 100,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 5% of the rows repeated, shuffled | 100,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| consecutive, 10% of the rows repeated, in order | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| consecutive, 10% of the rows repeated, shuffled | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| every row present twice, shuffled | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| 30% of the records gone, in order | 100,000 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 30% of the records gone, shuffled | 100,000 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 60% of the records gone, in order | 100,000 | 30 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| 60% of the records gone, shuffled | 100,000 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 | 0 of 30 |
| increasing, gaps of 1 to 3, in order | 100,000 | 30 of 30 | 17 of 30 | 30 of 30 | 30 of 30 | 30 of 30 |
| increasing, gaps of 1 to 3, shuffled | 100,000 | 0 of 30 | 17 of 30 | 30 of 30 | 30 of 30 | 17 of 30 |
| increasing, gaps of 1 to 10, in order | 100,000 | 30 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 |
| increasing, gaps of 1 to 10, shuffled | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 0 of 30 |
| increasing, steps of 7, in order | 100,000 | 30 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 30 of 30 |
| increasing, steps of 7, shuffled | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 | 0 of 30 |
| increasing, steps of 1,000, in order | 100,000 | 30 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 30 of 30 |
| increasing, steps of 1,000, shuffled | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
| random, unordered (nine digits) | 100,000 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 | 0 of 30 |
```

`messy` at 30 seeds, as it is, shuffled, and with rows repeated:

| `messy` | rule | `record_id` excluded as id_like, of 30 | `visits` a feature | skill | inside 0.55 to 0.91 | stopped |
|---|---|---|---|---|---|---|
| as it is | D19 as it is | 30 of 30 | 30 of 30 | +0.670 to +0.826 | 30 of 30 |  |
| rows_shuffled | D19 as it is | 0 of 30 | 30 of 30 | +0.643 to +0.817 | 30 of 30 |  |
| row_repeated | D19 as it is | 0 of 30 | 30 of 30 | +0.647 to +0.824 | 30 of 30 |  |
| rows_repeated_1pct | D19 as it is | 0 of 29 | 29 of 29 | +0.673 to +0.803 | 29 of 29 | {'holdout_copies_in_training': 1} |
| rows_repeated_10pct | D19 as it is | 0 of 0 | 0 of 0 | none | 0 of 0 | {'holdout_copies_in_training': 30} |
| as it is | proposed | 30 of 30 | 30 of 30 | +0.670 to +0.826 | 30 of 30 |  |
| rows_shuffled | proposed | 30 of 30 | 30 of 30 | +0.640 to +0.818 | 30 of 30 |  |
| row_repeated | proposed | 30 of 30 | 30 of 30 | +0.650 to +0.822 | 30 of 30 |  |
| rows_repeated_1pct | proposed | 29 of 29 | 29 of 29 | +0.671 to +0.803 | 29 of 29 | {'holdout_copies_in_training': 1} |
| rows_repeated_10pct | proposed | 0 of 0 | 0 of 0 | none | 0 of 0 | {'holdout_copies_in_training': 30} |

The cost kept from D19's clause is the measure in a table sorted by it (D31):
the battery's pair `messy:sorted_by_measure`, a stated limit.

## 5. The choice of model beside a dominant entity (A6; D74 point 6)

```text
dominant_entity, (as it is)
  before  -0.8718 to +0.0030
    -0.006 -0.005 -0.005 -0.872 -0.011 +0.003 -0.005 -0.004 -0.806 -0.004 +0.003 +0.002 +0.002 -0.004 -0.006 +0.001 -0.004 -0.001 -0.006 +0.002 -0.008 -0.008 -0.009 -0.001 -0.004 -0.005 -0.003 -0.003 -0.004 +0.001
  new     -0.0117 to +0.0030
    -0.006 -0.005 -0.005 -0.007 -0.011 +0.003 -0.005 -0.004 -0.012 -0.004 +0.003 +0.002 +0.002 -0.004 -0.006 +0.001 -0.004 -0.001 -0.006 +0.002 -0.008 -0.008 -0.009 -0.001 -0.004 -0.005 -0.003 -0.003 -0.004 +0.001
  before: chosen  linear linear linear tree-0 linear linear linear linear tree-0 linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear
  before: `selection_not_made` on 0 of 30
  new: chosen  linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear
  new: `selection_not_made` on 30 of 30

quarter_entity, (as it is)
  before  -0.0110 to +0.0063
    +0.002 -0.002 -0.002 -0.011 +0.000 -0.005 +0.001 -0.003 -0.001 +0.000 -0.006 -0.001 -0.003 +0.006 -0.001 +0.003 -0.007 +0.005 -0.002 -0.002 -0.007 -0.001 +0.002 -0.004 -0.006 +0.003 +0.001 +0.000 +0.000 +0.003
  new     -0.0110 to +0.0063
    +0.002 -0.002 -0.002 -0.011 +0.000 -0.005 +0.001 -0.003 -0.001 +0.000 -0.006 -0.001 -0.003 +0.006 -0.001 +0.003 -0.007 +0.005 -0.002 -0.002 -0.007 -0.001 +0.002 -0.004 -0.006 +0.003 +0.001 +0.000 +0.000 +0.003
  before: chosen  linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear
  before: `selection_not_made` on 0 of 30
  new: chosen  linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear
  new: `selection_not_made` on 0 of 30

entities, (as it is)
  before  -0.2599 to +0.1681
    -0.081 +0.078 +0.041 -0.046 +0.004 -0.093 -0.035 -0.174 +0.029 -0.058 -0.059 -0.046 -0.204 +0.032 -0.083 +0.090 +0.153 -0.076 -0.046 -0.260 -0.201 +0.154 +0.142 +0.061 -0.063 +0.168 +0.150 -0.008 -0.240 -0.141
  new     -0.2599 to +0.1681
    -0.081 +0.078 +0.041 -0.046 +0.004 -0.093 -0.035 -0.174 +0.029 -0.058 -0.059 -0.046 -0.204 +0.032 -0.083 +0.090 +0.153 -0.076 -0.046 -0.260 -0.201 +0.154 +0.142 +0.061 -0.063 +0.168 +0.150 -0.008 -0.240 -0.141
  before: chosen  linear tree-0 tree-0 linear tree-0 linear linear tree-0 tree-0 linear tree-0 tree-0 linear tree-0 linear linear tree-0 tree-0 linear linear linear linear linear tree-0 linear linear tree-0 tree-0 linear tree-0
  before: `selection_not_made` on 0 of 30
  new: chosen  linear tree-0 tree-0 linear tree-0 linear linear tree-0 tree-0 linear tree-0 tree-0 linear tree-0 linear linear tree-0 tree-0 linear linear linear linear linear tree-0 linear linear tree-0 tree-0 linear tree-0
  new: `selection_not_made` on 0 of 30

few_valued, (as it is)
  before  -0.1214 to +0.0835
    +0.018 +0.000 -0.046 -0.020 +0.013 +0.084 -0.011 -0.088 +0.047 +0.038 -0.012 +0.044 +0.003 -0.005 +0.002 +0.008 -0.004 -0.086 +0.065 -0.011 -0.121 -0.025 -0.008 -0.002 +0.007 +0.066 +0.000 +0.000 +0.032 +0.015
  new     -0.1214 to +0.0835
    +0.018 +0.000 -0.046 -0.020 +0.013 +0.084 -0.011 -0.088 +0.047 +0.038 -0.012 +0.044 +0.003 -0.005 +0.002 +0.008 -0.004 -0.086 +0.065 -0.011 -0.121 -0.025 -0.008 -0.002 +0.007 +0.066 +0.000 +0.000 +0.032 +0.015
  before: chosen  tree-0 tree-0 tree-0 tree-0 linear linear linear linear tree-0 tree-0 tree-0 tree-0 linear linear linear linear linear tree-0 linear linear linear linear linear linear tree-0 linear tree-0 tree-0 tree-0 linear
  before: `selection_not_made` on 0 of 30
  new: chosen  tree-0 tree-0 tree-0 tree-0 linear linear linear linear tree-0 tree-0 tree-0 tree-0 linear linear linear linear linear tree-0 linear linear linear linear linear linear tree-0 linear tree-0 tree-0 tree-0 linear
  new: `selection_not_made` on 0 of 30

dominant_entity, copy_rescaled
  before  +0.9999 to +1.0000
    +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000 +1.000
  new     -0.0117 to +0.0030
    -0.006 -0.005 -0.005 -0.007 -0.011 +0.003 -0.005 -0.004 -0.012 -0.004 +0.003 +0.002 +0.002 -0.004 -0.006 +0.001 -0.004 -0.001 -0.006 +0.002 -0.008 -0.008 -0.009 -0.001 -0.004 -0.005 -0.003 -0.003 -0.004 +0.001
  before: chosen  linear linear linear linear linear linear linear linear linear tree-0 linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear
  before: `selection_not_made` on 0 of 30
  new: chosen  linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear linear
  new: `selection_not_made` on 30 of 30
```

On how few rows the choice fails, and whether leaving the small fold unscored
would have helped (it would not: the folds that validate on those rows decide):

| small-entity rows left among the training rows | folds | tree chosen, of 30 | skill when the linear model is chosen | when the tree is |
|---|---|---|---|---|
| 10 | every fold scored | 2 | -0.014 to +0.020 | -1.112 to -0.983 |
| 20 | every fold scored | 0 | -0.017 to +0.020 |  |
| 30 | every fold scored | 0 | -0.012 to +0.016 |  |
| 50 | every fold scored | 0 | -0.016 to +0.011 |  |
| 100 | every fold scored | 0 | -0.013 to +0.015 |  |
| 200 | every fold scored | 0 | -0.022 to +0.009 |  |
| 10 | the fold that trains on them not scored | 2 | -0.014 to +0.020 | -1.272 to -1.112 |
| 20 | the fold that trains on them not scored | 0 | -0.017 to +0.020 |  |
| 30 | the fold that trains on them not scored | 1 | -0.012 to +0.016 | -1.314 to -1.314 |
| 50 | the fold that trains on them not scored | 0 | -0.016 to +0.011 |  |
| 100 | the fold that trains on them not scored | 0 | -0.013 to +0.015 |  |
| 200 | the fold that trains on them not scored | 0 | -0.022 to +0.009 |  |

What the leak screen rests on in the same table:

| the fold that trains on ten rows | copy added | largest skill of a noise column, over the seeds | seeds with a noise column warned | excluded | the copy excluded | errors |
|---|---|---|---|---|---|---|
| scored | no | +0.001 to +0.204 | 0 of 30 | 0 of 30 |  |  |
| scored | yes | +0.001 to +0.204 | 0 of 30 | 0 of 30 | 30 |  |
| not scored | no | +0.208 to +0.563 | 0 of 30 | 0 of 30 |  |  |
| not scored | yes | +0.208 to +0.563 | 0 of 30 | 0 of 30 | 30 |  |

## 6. Every run of the gate at 30 seeds, before and now

```text
signal_clf
  new     +0.6719 to +0.8005
    +0.742 +0.708 +0.697 +0.734 +0.748 +0.703 +0.690 +0.791 +0.672 +0.726 +0.753 +0.744 +0.736 +0.771 +0.729 +0.779 +0.756 +0.801 +0.738 +0.756 +0.723 +0.779 +0.740 +0.760 +0.793 +0.696 +0.726 +0.754 +0.779 +0.719

signal_reg
  new     +0.8191 to +0.8914
    +0.831 +0.839 +0.843 +0.863 +0.846 +0.891 +0.850 +0.849 +0.819 +0.833 +0.835 +0.854 +0.854 +0.847 +0.861 +0.840 +0.857 +0.845 +0.871 +0.859 +0.868 +0.876 +0.830 +0.832 +0.835 +0.867 +0.856 +0.825 +0.848 +0.826

noise_clf
  new     -0.1145 to +0.1590
    -0.115 -0.050 +0.003 +0.012 -0.028 +0.023 -0.020 +0.032 +0.016 +0.055 +0.072 +0.002 +0.008 -0.018 +0.097 -0.032 -0.057 +0.033 -0.069 -0.008 +0.021 +0.072 +0.024 +0.018 -0.096 -0.072 -0.043 +0.030 +0.057 +0.159

noise_reg
  new     -0.5026 to -0.2423
    -0.407 -0.503 -0.318 -0.383 -0.429 -0.494 -0.355 -0.383 -0.380 -0.321 -0.278 -0.320 -0.472 -0.242 -0.352 -0.351 -0.425 -0.478 -0.352 -0.332 -0.453 -0.414 -0.387 -0.431 -0.290 -0.289 -0.497 -0.272 -0.449 -0.266

imbalanced
  new     -0.1499 to +0.3758
    +0.072 +0.052 -0.041 +0.040 +0.125 +0.083 +0.033 -0.047 -0.015 +0.113 -0.113 +0.010 +0.064 -0.012 +0.037 +0.376 -0.139 +0.016 -0.051 -0.031 -0.150 -0.022 -0.051 +0.077 +0.005 +0.008 +0.112 -0.028 +0.009 -0.034

leak
  new     +0.6479 to +0.8210
    +0.763 +0.712 +0.703 +0.814 +0.766 +0.749 +0.745 +0.758 +0.775 +0.753 +0.737 +0.782 +0.728 +0.787 +0.648 +0.821 +0.763 +0.735 +0.739 +0.753 +0.737 +0.713 +0.804 +0.747 +0.764 +0.687 +0.731 +0.762 +0.773 +0.786

leak_reg
  new     +0.8064 to +0.8593
    +0.820 +0.843 +0.832 +0.827 +0.859 +0.833 +0.852 +0.813 +0.857 +0.846 +0.849 +0.840 +0.842 +0.806 +0.821 +0.857 +0.844 +0.858 +0.842 +0.853 +0.827 +0.850 +0.839 +0.839 +0.830 +0.832 +0.833 +0.857 +0.859 +0.850

near_leak
  new     +0.9691 to +0.9996
    +0.969 +0.992 +0.989 +0.992 +1.000 +0.990 +0.989 +0.992 +0.979 +0.985 +0.996 +0.997 +0.991 +0.990 +0.993 +0.993 +0.988 +0.998 +0.987 +0.988 +0.989 +0.994 +0.992 +0.987 +0.990 +0.980 +0.978 +0.985 +0.994 +0.994

binary_numeric
  new     +0.6614 to +0.8130
    +0.734 +0.736 +0.711 +0.795 +0.684 +0.719 +0.769 +0.813 +0.734 +0.679 +0.755 +0.760 +0.774 +0.734 +0.730 +0.766 +0.757 +0.731 +0.664 +0.761 +0.702 +0.728 +0.774 +0.744 +0.751 +0.734 +0.753 +0.661 +0.770 +0.756

ambiguous_target
  new     no value; 30 stopped
    task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous task_ambiguous

messy
  new     +0.6702 to +0.8263
    +0.722 +0.826 +0.712 +0.739 +0.770 +0.757 +0.761 +0.762 +0.786 +0.696 +0.782 +0.709 +0.810 +0.773 +0.724 +0.814 +0.730 +0.736 +0.751 +0.722 +0.722 +0.761 +0.710 +0.745 +0.752 +0.748 +0.670 +0.724 +0.731 +0.734

long_rows
  new     no value; 30 stopped
    input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error input_error

dup_noise
  new     no value; 30 stopped
    holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training

time_blocks
  new     -0.2483 to +0.2170
    +0.122 +0.118 +0.217 -0.009 +0.031 +0.080 -0.104 +0.040 +0.028 +0.017 +0.014 -0.082 -0.155 +0.016 +0.050 +0.138 -0.068 -0.167 -0.144 +0.069 +0.217 -0.157 +0.085 -0.053 -0.173 -0.037 -0.007 -0.038 -0.023 -0.248

time_blocks_gap
  new     -0.4127 to +0.2264
    -0.413 +0.138 +0.226 -0.011 +0.031 -0.086 -0.094 +0.040 +0.035 +0.017 +0.041 -0.072 -0.111 -0.001 +0.033 +0.139 -0.071 -0.123 -0.030 +0.070 -0.095 -0.017 +0.091 -0.049 -0.181 -0.025 -0.003 -0.040 -0.024 -0.247

time_blocks_undeclared
  new     +0.9782 to +1.0000
    +0.999 +0.982 +0.978 +1.000 +0.994 +0.999 +1.000 +0.998 +1.000 +0.997 +0.999 +0.997 +1.000 +1.000 +0.999 +0.983 +0.998 +0.980 +0.998 +1.000 +0.999 +0.986 +0.996 +1.000 +1.000 +0.990 +1.000 +1.000 +0.999 +0.999

time_blocks_unstated
  new     no value; 30 stopped
    time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared

time_dayfirst
  new     no value; 30 stopped
    time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid time_column_invalid

random_walk
  new     -0.0998 to +0.0118
    +0.001 -0.001 -0.011 -0.004 +0.005 +0.001 -0.003 +0.012 -0.027 +0.002 +0.002 +0.002 +0.002 -0.036 +0.004 +0.001 -0.018 -0.100 +0.003 +0.002 +0.004 +0.003 +0.001 +0.003 -0.005 -0.021 -0.001 -0.003 +0.004 -0.011

int_stamps
  new     -0.1823 to +0.0052
    +0.002 -0.016 +0.005 +0.002 -0.182 +0.001 +0.002 +0.003 +0.003 -0.001 -0.001 -0.081 -0.038 -0.030 -0.011 -0.004 -0.005 -0.008 -0.016 -0.014 -0.176 +0.001 -0.038 -0.014 -0.000 -0.000 -0.001 +0.001 -0.003 -0.020

flat_tail
  new     no value
    null null null null null null null null null null null null null null null null null null null null null null null null null null null null null null

entities
  new     -0.2599 to +0.1681
    -0.081 +0.078 +0.041 -0.046 +0.004 -0.093 -0.035 -0.174 +0.029 -0.058 -0.059 -0.046 -0.204 +0.032 -0.083 +0.090 +0.153 -0.076 -0.046 -0.260 -0.201 +0.154 +0.142 +0.061 -0.063 +0.168 +0.150 -0.008 -0.240 -0.141

panel
  new     -0.0228 to +0.0062
    -0.010 -0.023 -0.000 +0.000 -0.002 +0.004 +0.002 -0.001 -0.012 -0.001 +0.002 -0.009 +0.005 -0.000 -0.000 -0.004 -0.009 +0.000 -0.004 +0.000 -0.003 -0.007 -0.007 -0.019 +0.000 +0.006 +0.005 -0.002 +0.003 -0.001

full_noise_small
  new     -0.3600 to +0.2825
    -0.038 -0.110 +0.023 +0.283 -0.134 -0.078 +0.111 -0.051 -0.135 +0.160 +0.124 +0.008 +0.024 -0.013 -0.037 +0.114 -0.050 -0.100 -0.032 +0.025 -0.031 +0.044 -0.132 +0.139 +0.090 -0.128 -0.098 -0.360 +0.013 +0.166

full_signal_clf_j1
  new     +0.5995 to +0.8204
    +0.645 +0.785 +0.723 +0.797 +0.780 +0.744 +0.779 +0.752 +0.659 +0.673 +0.599 +0.762 +0.635 +0.712 +0.752 +0.733 +0.799 +0.803 +0.705 +0.820 +0.743 +0.761 +0.674 +0.633 +0.670 +0.756 +0.696 +0.601 +0.776 +0.666

full_signal_clf_j4
  new     +0.5995 to +0.8204
    +0.645 +0.785 +0.723 +0.797 +0.780 +0.744 +0.779 +0.752 +0.659 +0.673 +0.599 +0.762 +0.635 +0.712 +0.752 +0.733 +0.799 +0.803 +0.705 +0.820 +0.743 +0.761 +0.674 +0.633 +0.670 +0.756 +0.696 +0.601 +0.776 +0.666

full_signal_reg_j1
  new     +0.7708 to +0.8896
    +0.818 +0.815 +0.872 +0.847 +0.853 +0.835 +0.847 +0.844 +0.890 +0.833 +0.856 +0.840 +0.771 +0.842 +0.883 +0.848 +0.863 +0.844 +0.866 +0.845 +0.865 +0.878 +0.857 +0.790 +0.881 +0.834 +0.798 +0.820 +0.826 +0.831

full_signal_reg_j4
  new     +0.7708 to +0.8896
    +0.818 +0.815 +0.872 +0.847 +0.853 +0.835 +0.847 +0.844 +0.890 +0.833 +0.856 +0.840 +0.771 +0.842 +0.883 +0.848 +0.863 +0.844 +0.866 +0.845 +0.865 +0.878 +0.857 +0.790 +0.881 +0.834 +0.798 +0.820 +0.826 +0.831

unsaid_both
  new     no value; 30 stopped
    time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared time_order_undeclared

unsaid_entities
  new     no value; 30 stopped
    entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared entities_undeclared

few_valued
  new     -0.1214 to +0.0835
    +0.018 +0.000 -0.046 -0.020 +0.013 +0.084 -0.011 -0.088 +0.047 +0.038 -0.012 +0.044 +0.003 -0.005 +0.002 +0.008 -0.004 -0.086 +0.065 -0.011 -0.121 -0.025 -0.008 -0.002 +0.007 +0.066 +0.000 +0.000 +0.032 +0.015

dominant_entity
  new     -0.0117 to +0.0030
    -0.006 -0.005 -0.005 -0.007 -0.011 +0.003 -0.005 -0.004 -0.012 -0.004 +0.003 +0.002 +0.002 -0.004 -0.006 +0.001 -0.004 -0.001 -0.006 +0.002 -0.008 -0.008 -0.009 -0.001 -0.004 -0.005 -0.003 -0.003 -0.004 +0.001
  before  -0.8718 to +0.0030
    -0.006 -0.005 -0.005 -0.872 -0.011 +0.003 -0.005 -0.004 -0.806 -0.004 +0.003 +0.002 +0.002 -0.004 -0.006 +0.001 -0.004 -0.001 -0.006 +0.002 -0.008 -0.008 -0.009 -0.001 -0.004 -0.005 -0.003 -0.003 -0.004 +0.001

day_numbers
  new     -0.0518 to +0.0806
    -0.033 +0.065 +0.062 +0.024 +0.029 +0.081 -0.026 +0.004 +0.036 +0.009 -0.046 -0.052 -0.035 -0.038 +0.049 -0.044 -0.006 -0.004 -0.038 +0.073 +0.014 +0.046 +0.020 +0.020 +0.059 -0.009 +0.030 -0.031 +0.021 +0.066

day_sentinel
  new     -0.1302 to +0.0755
    -0.048 -0.040 +0.011 -0.100 +0.019 +0.067 +0.075 +0.053 -0.018 +0.034 -0.072 +0.052 -0.035 +0.038 -0.072 +0.062 +0.015 -0.047 +0.038 +0.041 -0.042 -0.007 +0.064 -0.009 -0.130 +0.074 -0.060 -0.051 -0.061 +0.034

few_valued_leak
  new     no value; 30 stopped
    holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training holdout_copies_in_training

large_entity
  new     no value; 30 stopped
    fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large fast_profile_group_too_large

quarter_entity
  new     -0.0110 to +0.0063
    +0.002 -0.002 -0.002 -0.011 +0.000 -0.005 +0.001 -0.003 -0.001 +0.000 -0.006 -0.001 -0.003 +0.006 -0.001 +0.003 -0.007 +0.005 -0.002 -0.002 -0.007 -0.001 +0.002 -0.004 -0.006 +0.003 +0.001 +0.000 +0.000 +0.003

Fields that differ from the reference as it stood, by run (seeds of 30):
  dominant_entity: warnings on 30, skill on 2, score on 2, model on 2
  no field differs on 36 runs: signal_clf, signal_reg, noise_clf, noise_reg, imbalanced, leak, leak_reg, near_leak, binary_numeric, ambiguous_target, messy, long_rows, dup_noise, time_blocks, time_blocks_gap, time_blocks_undeclared, time_blocks_unstated, time_dayfirst, random_walk, int_stamps, flat_tail, entities, panel, full_noise_small, full_signal_clf_j1, full_signal_clf_j4, full_signal_reg_j1, full_signal_reg_j4, unsaid_both, unsaid_entities, few_valued, day_numbers, day_sentinel, few_valued_leak, large_entity, quarter_entity
```

One run moves: `dominant_entity`, by the rule of section 5. On the other 36 no
field of the eighteen differs on any seed.

## 7. The gate

Six checks, five datasets added last and one band came with the pass. The
scripts (after `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B
docs\measurements\CP9\`):

| Script | What it does | Command |
|---|---|---|
| `gate9_tables.py` | no seed, table, run, band or check of the gate as it was has moved | `gate9_tables.py <old verify>\smoke_test.py verify\smoke_test.py` |
| `gate9_seeds.py` | checks of the gate, run as the gate runs them, at 30 data seeds | `gate9_seeds.py verify\smoke_test.py <package folder> <work folder> <out>\seeds_<label>.json --only <the six checks>` |
| `gate9_summary.py` | the summary of those, check by check and package by package | `gate9_summary.py <out> --per-seed` |
| `gate9_compare.py` | what moved between two full gate runs, leaf by leaf | `gate9_compare.py <runs of the reference as it stood> <runs of the reference> --old-runs-only <old verify>\smoke_test.py` |
| `gate9_doctor.py` | each new check on a good output with one value changed | `gate9_doctor.py verify\smoke_test.py <verify\artifacts of a good run> <work folder>` |
| `gate9_times.py` | the time of a gate run, per edit and in full | `gate9_times.py E:\CODE\ml-professor verify/reference/mlprofessor 3` |

### 7.1 Each new check seen failing (proofs 1 and 2)

The gate on the reference: 53 checks pass.

```text
PASS data.rows_reconcile
PASS data.columns_accounted
PASS data.coercion_reported
PASS data.long_rows_stop
PASS data.holdout_copies_stop
PASS models.all_accounted
PASS task.binary_numeric
PASS task.ambiguous_stops
PASS leak.excluded
PASS leak.control_kept
PASS leak.regression
PASS leak.near_leak_warned
PASS model.signal_clf
PASS model.signal_reg
PASS model.noise_clf
PASS model.noise_p_value
PASS model.noise_reg
PASS model.headline_recomputed
PASS model.metric_clf
PASS model.selection_full
PASS time.order_respected
PASS time.no_skill_on_blocks
PASS time.gap_applied
PASS time.dayfirst_stops
PASS time.persistence_baseline
PASS time.integer_keys_distinct
PASS time.undeclared_warns
PASS time.undeclared_stops
PASS time.panel_previous_value
PASS groups.kept_together
PASS groups.no_skill_on_entities
PASS statements.time_order_asked
PASS statements.entities_asked
PASS duplicates.group_on_features
PASS duplicates.group_no_skill
PASS groups.holdout_bounded
PASS time.date_numbers_seen
PASS time.date_sentinel_seen
PASS duplicates.group_leak_stops
PASS fast.large_entity_stops
PASS fast.large_entity_kept
PASS time.previous_value_at_repeats
PASS leak.copy_excluded_under_time
PASS leak.excluded_despite_extreme_target
PASS result.largest_row_reported
PASS data.identifier_survives_shuffle
PASS groups.choice_not_made
PASS time.flat_holdout_reported
PASS uncertainty.reported
PASS result.matches_json
PASS output.complete
PASS output.deterministic
PASS output.same_at_any_n_jobs
GATE: PASS (53 checks)
```

On the reference as it stood: the 47 checks it had pass, and the six new ones
fail.

```text
PASS data.rows_reconcile
PASS data.columns_accounted
PASS data.coercion_reported
PASS data.long_rows_stop
PASS data.holdout_copies_stop
PASS models.all_accounted
PASS task.binary_numeric
PASS task.ambiguous_stops
PASS leak.excluded
PASS leak.control_kept
PASS leak.regression
PASS leak.near_leak_warned
PASS model.signal_clf
PASS model.signal_reg
PASS model.noise_clf
PASS model.noise_p_value
PASS model.noise_reg
PASS model.headline_recomputed
PASS model.metric_clf
PASS model.selection_full
PASS time.order_respected
PASS time.no_skill_on_blocks
PASS time.gap_applied
PASS time.dayfirst_stops
PASS time.persistence_baseline
PASS time.integer_keys_distinct
PASS time.undeclared_warns
PASS time.undeclared_stops
PASS time.panel_previous_value
PASS groups.kept_together
PASS groups.no_skill_on_entities
PASS statements.time_order_asked
PASS statements.entities_asked
PASS duplicates.group_on_features
PASS duplicates.group_no_skill
PASS groups.holdout_bounded
PASS time.date_numbers_seen
PASS time.date_sentinel_seen
PASS duplicates.group_leak_stops
PASS fast.large_entity_stops
PASS fast.large_entity_kept
FAIL time.previous_value_at_repeats: baselines on `walk_repeated` lists ['training_mean']. One row of this random walk is written twice, so the date 2021-01-21 holds two rows and the timestamps are not unique. With no entity column the previous value of a row is the mean target of the timestamp one before its own, so both training_mean and previous_value are scored and reported (D74; spec.md step 11). A baseline that is there only while every timestamp is unique is lost to one repeated row, and against the training mean a random walk looks almost perfectly predictable
FAIL leak.copy_excluded_under_time: `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.9875713998211745 against the baseline 'previous_value'. The column is the target on another scale, 1.07 x level + 3, and must be excluded as a suspected leak with the rule `predicts_target`. Under time order the screen's baseline is the previous value, and a tree fitted on the earlier rows cannot follow a target that wanders beyond the range it was fitted on: read with its tree alone, this column does not reach the cut-off. The screen reads a numeric column with a robust line as well as with its tree, and excludes it on the better of the two (D74; spec.md step 8)
FAIL leak.excluded_despite_extreme_target: `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is 0.35824851373882205 against the baseline 'previous_value'. The column is the target on another scale, 1.07 x level + 3, and must be excluded as a suspected leak with the rule `predicts_target`. One cell of the target is -999, in a training row (source row 365); in every other row the column is the target on another scale. With every row scored, the squared error of that one row is nearly all of the screen's sum for the column, and that row and the one after it, whose previous value is the cell, are nearly all of the baseline's: the column's skill comes out at one half or less, far below the cut-off. A training row whose target lies more than ten median deviations from the median target, or ten median steps from its previous value, is set aside from the whole screen, and the rows a column misses by most are not scored for exclusion (D74; spec.md step 8)
FAIL result.largest_row_reported: profile.json on `walk_late_cell` has no key headline.largest_row_share, headline.largest_row, headline.baseline_largest_row_share, headline.baseline_largest_row, cv.largest_row_share, cv.largest_row and 6 more. A regression run says what share of a squared error its single largest row carries, and which row that is: on the holdout, for the model and for the headline baseline (`headline`), and over the selection folds, for the chosen configuration and that baseline (`cv`) and for every configuration (`models`) (D74; spec.md 5.2)
FAIL data.identifier_survives_shuffle: `record_id` on `messy_shuffled` has role 'feature' and reason None. In the 1500 rows with a target it holds 1500 distinct whole numbers that fill 97% of the range from the smallest to the largest. A whole-number column in which more than 95% of the values are distinct and the distinct values fill at least half of their range is an identifier, whatever the order of the rows (D74; spec.md step 5). By D19's clause alone, all distinct and consecutive or in file order, it is one in `messy` and a numeric feature here, where the same rows are shuffled
FAIL groups.choice_not_made: 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill -0.0062. The training rows are the entity C0000, 1490 rows, and 10 rows of small entities. An entity is in one fold, so the fold that validates on the large one trains on at most 10 rows, fewer than 50 and fewer than it validates on, and each other fold validates on some of those same 10 rows. No choice between configurations is made on so few: every fold stays scored and reported, the configurations are taken in configuration order, so the one chosen is the first that has a cross-validation mean, and one `selection_not_made` warning says so (D74; spec.md step 9)
PASS time.flat_holdout_reported
PASS uncertainty.reported
PASS result.matches_json
PASS output.complete
PASS output.deterministic
PASS output.same_at_any_n_jobs
GATE: FAIL (6 of 53 checks failed)
```

On the application as built by the seventh pass: the same six fail, and
nothing else. Each is owned by a task of the pass (section 8).

```text
PASS data.rows_reconcile
PASS data.columns_accounted
PASS data.coercion_reported
PASS data.long_rows_stop
PASS data.holdout_copies_stop
PASS models.all_accounted
PASS task.binary_numeric
PASS task.ambiguous_stops
PASS leak.excluded
PASS leak.control_kept
PASS leak.regression
PASS leak.near_leak_warned
PASS model.signal_clf
PASS model.signal_reg
PASS model.noise_clf
PASS model.noise_p_value
PASS model.noise_reg
PASS model.headline_recomputed
PASS model.metric_clf
PASS model.selection_full
PASS time.order_respected
PASS time.no_skill_on_blocks
PASS time.gap_applied
PASS time.dayfirst_stops
PASS time.persistence_baseline
PASS time.integer_keys_distinct
PASS time.undeclared_warns
PASS time.undeclared_stops
PASS time.panel_previous_value
PASS groups.kept_together
PASS groups.no_skill_on_entities
PASS statements.time_order_asked
PASS statements.entities_asked
PASS duplicates.group_on_features
PASS duplicates.group_no_skill
PASS groups.holdout_bounded
PASS time.date_numbers_seen
PASS time.date_sentinel_seen
PASS duplicates.group_leak_stops
PASS fast.large_entity_stops
PASS fast.large_entity_kept
FAIL time.previous_value_at_repeats: baselines on `walk_repeated` lists ['training_mean']. One row of this random walk is written twice, so the date 2021-01-21 holds two rows and the timestamps are not unique. With no entity column the previous value of a row is the mean target of the timestamp one before its own, so both training_mean and previous_value are scored and reported (D74; spec.md step 11). A baseline that is there only while every timestamp is unique is lost to one repeated row, and against the training mean a random walk looks almost perfectly predictable
FAIL leak.copy_excluded_under_time: `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.9875713998211809 against the baseline 'previous_value'. The column is the target on another scale, 1.07 x level + 3, and must be excluded as a suspected leak with the rule `predicts_target`. Under time order the screen's baseline is the previous value, and a tree fitted on the earlier rows cannot follow a target that wanders beyond the range it was fitted on: read with its tree alone, this column does not reach the cut-off. The screen reads a numeric column with a robust line as well as with its tree, and excludes it on the better of the two (D74; spec.md step 8)
FAIL leak.excluded_despite_extreme_target: `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is 0.35824851373880295 against the baseline 'previous_value'. The column is the target on another scale, 1.07 x level + 3, and must be excluded as a suspected leak with the rule `predicts_target`. One cell of the target is -999, in a training row (source row 365); in every other row the column is the target on another scale. With every row scored, the squared error of that one row is nearly all of the screen's sum for the column, and that row and the one after it, whose previous value is the cell, are nearly all of the baseline's: the column's skill comes out at one half or less, far below the cut-off. A training row whose target lies more than ten median deviations from the median target, or ten median steps from its previous value, is set aside from the whole screen, and the rows a column misses by most are not scored for exclusion (D74; spec.md step 8)
FAIL result.largest_row_reported: profile.json on `walk_late_cell` has no key headline.largest_row_share, headline.largest_row, headline.baseline_largest_row_share, headline.baseline_largest_row, cv.largest_row_share, cv.largest_row and 6 more. A regression run says what share of a squared error its single largest row carries, and which row that is: on the holdout, for the model and for the headline baseline (`headline`), and over the selection folds, for the chosen configuration and that baseline (`cv`) and for every configuration (`models`) (D74; spec.md 5.2)
FAIL data.identifier_survives_shuffle: `record_id` on `messy_shuffled` has role 'feature' and reason None. In the 1500 rows with a target it holds 1500 distinct whole numbers that fill 97% of the range from the smallest to the largest. A whole-number column in which more than 95% of the values are distinct and the distinct values fill at least half of their range is an identifier, whatever the order of the rows (D74; spec.md step 5). By D19's clause alone, all distinct and consecutive or in file order, it is one in `messy` and a numeric feature here, where the same rows are shuffled
FAIL groups.choice_not_made: 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'tree-0' and skill -1.1278. The training rows are the entity C0000, 1490 rows, and 10 rows of small entities. An entity is in one fold, so the fold that validates on the large one trains on at most 10 rows, fewer than 50 and fewer than it validates on, and each other fold validates on some of those same 10 rows. No choice between configurations is made on so few: every fold stays scored and reported, the configurations are taken in configuration order, so the one chosen is the first that has a cross-validation mean, and one `selection_not_made` warning says so (D74; spec.md step 9)
PASS time.flat_holdout_reported
PASS uncertainty.reported
PASS result.matches_json
PASS output.complete
PASS output.deterministic
PASS output.same_at_any_n_jobs
GATE: FAIL (6 of 53 checks failed)
```

### 7.2 The new checks at 30 data seeds

For each check: the reference (`good`), its own broken copy, the reference as
it stood (`before`) and the application (`app`).

```text
## time.previous_value_at_repeats (run `walk_repeated`, names F23)
 good:
  passes on 30 of 30 seeds; the run finishes on 30
  skill: -0.0368 to +0.0084 (mean -0.0032, sd 0.0104, n 30)
  model: {'linear-0': 30}; baseline: {'previous_value': 30}; baselines listed: {'previous_value,training_mean': 30}
  level_prev: {'feature/None/None': 30}
  autocorrelated_panel_target, columns named -> seeds: ['level']: 30
 F23:
  passes on 0 of 30 seeds; the run finishes on 1
  runs that stopped: {'no_features': 29}
  skill: +0.9776 to +0.9776 (mean +0.9776, sd 0.0000, n 1)
  model: {'linear-0': 1}; baseline: {'training_mean': 1}; baselines listed: {'training_mean': 1}
  level_prev: {'feature/None/None': 1}
  strong_single_column, columns named -> seeds: ['level_prev']: 1
  autocorrelated_panel_target, columns named -> seeds: ['level']: 1
  fails (29): the tool exited 3 on dataset `walk_repeated` where it should have succeeded.
  command: python -m mlprofessor profile --data <scratch>...
  fails (1): baselines on `walk_repeated` lists ['training_mean']. One row of this random walk is written twice, so the date 2019-03-19 holds two rows and the time
 before:
  passes on 0 of 30 seeds; the run finishes on 29
  runs that stopped: {'no_features': 1}
  skill: +0.9606 to +0.9996 (mean +0.9904, sd 0.0103, n 29)
  model: {'linear-0': 29}; baseline: {'training_mean': 29}; baselines listed: {'training_mean': 29}
  level_prev: {'feature/None/None': 29}
  strong_single_column, columns named -> seeds: none: 12; ['level_prev']: 17
  autocorrelated_panel_target, columns named -> seeds: ['level']: 29
  fails (1): baselines on `walk_repeated` lists ['training_mean']. One row of this random walk is written twice, so the date 2021-01-21 holds two rows and the time
  fails (1): baselines on `walk_repeated` lists ['training_mean']. One row of this random walk is written twice, so the date 2020-10-08 holds two rows and the time
  fails (1): baselines on `walk_repeated` lists ['training_mean']. One row of this random walk is written twice, so the date 2022-09-30 holds two rows and the time
  fails (1): baselines on `walk_repeated` lists ['training_mean']. One row of this random walk is written twice, so the date 2021-09-21 holds two rows and the time
 app:
  passes on 0 of 30 seeds; the run finishes on 29
  runs that stopped: {'no_features': 1}
  skill: +0.9606 to +0.9996 (mean +0.9904, sd 0.0103, n 29)
  model: {'linear-0': 29}; baseline: {'training_mean': 29}; baselines listed: {'training_mean': 29}
  level_prev: {'feature/None/None': 29}
  strong_single_column, columns named -> seeds: none: 12; ['level_prev']: 17
  autocorrelated_panel_target, columns named -> seeds: ['level']: 29
  fails (1): baselines on `walk_repeated` lists ['training_mean']. One row of this random walk is written twice, so the date 2021-01-21 holds two rows and the time
  fails (1): baselines on `walk_repeated` lists ['training_mean']. One row of this random walk is written twice, so the date 2020-10-08 holds two rows and the time
  fails (1): baselines on `walk_repeated` lists ['training_mean']. One row of this random walk is written twice, so the date 2022-09-30 holds two rows and the time
  fails (1): baselines on `walk_repeated` lists ['training_mean']. One row of this random walk is written twice, so the date 2021-09-21 holds two rows and the time

## leak.copy_excluded_under_time (run `walk_copy`, names F24)
 good:
  passes on 30 of 30 seeds; the run finishes on 30
  skill: -0.1939 to +0.0072 (mean -0.0138, sd 0.0388, n 30)
  model: {'linear-0': 30}; baseline: {'previous_value': 30}; baselines listed: {'previous_value,training_mean': 30}
  level_scaled: {'excluded/suspected_leak/predicts_target': 30}
  level_prev: {'feature/None/None': 30}
  suspected_leak, columns named -> seeds: ['level_scaled']: 30
 F24:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: +0.8896 to +0.9987 (mean +0.9848, sd 0.0225, n 30)
  model: {'linear-0': 30}; baseline: {'previous_value': 30}; baselines listed: {'previous_value,training_mean': 30}
  level_scaled: {'feature/None/None': 30}
  level_prev: {'feature/None/None': 30}
  strong_single_column, columns named -> seeds: ['level_scaled']: 30
  fails (1): `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.9875713998211745 against the baseline 'previous_value'. The 
  fails (1): `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.9957381940292633 against the baseline 'previous_value'. The 
  fails (1): `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.9907225550886155 against the baseline 'previous_value'. The 
  fails (1): `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.994836594850423 against the baseline 'previous_value'. The c
 before:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: +0.8896 to +0.9987 (mean +0.9848, sd 0.0225, n 30)
  model: {'linear-0': 30}; baseline: {'previous_value': 30}; baselines listed: {'previous_value,training_mean': 30}
  level_scaled: {'feature/None/None': 30}
  level_prev: {'feature/None/None': 30}
  fails (1): `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.9875713998211745 against the baseline 'previous_value'. The 
  fails (1): `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.9957381940292633 against the baseline 'previous_value'. The 
  fails (1): `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.9907225550886155 against the baseline 'previous_value'. The 
  fails (1): `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.994836594850423 against the baseline 'previous_value'. The c
 app:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: +0.8896 to +0.9987 (mean +0.9848, sd 0.0225, n 30)
  model: {'linear-0': 30}; baseline: {'previous_value': 30}; baselines listed: {'previous_value,training_mean': 30}
  level_scaled: {'feature/None/None': 30}
  level_prev: {'feature/None/None': 30}
  fails (1): `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.9875713998211809 against the baseline 'previous_value'. The 
  fails (1): `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.99573819402928 against the baseline 'previous_value'. The co
  fails (1): `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.9907225550886116 against the baseline 'previous_value'. The 
  fails (1): `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.9948365948503914 against the baseline 'previous_value'. The 

## leak.excluded_despite_extreme_target (run `walk_copy_cell`, names F25)
 good:
  passes on 30 of 30 seeds; the run finishes on 30
  skill: -39.5289 to -0.1516 (mean -2.7560, sd 7.1602, n 30)
  model: {'linear-0': 27, 'tree-0': 3}; baseline: {'previous_value': 30}; baselines listed: {'previous_value,training_mean': 30}
  level_scaled: {'excluded/suspected_leak/predicts_target': 30}
  level_prev: {'feature/None/None': 30}
  extreme_value, columns named -> seeds: ['level']: 30
  largest_row_dominates, columns named -> seeds: ['None']: 9; ['None', 'None']: 21
  suspected_leak, columns named -> seeds: ['level_scaled']: 30
 F25:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: -37.8850 to +0.9994 (mean -1.9966, sd 7.0345, n 30)
  model: {'linear-0': 26, 'tree-0': 4}; baseline: {'previous_value': 30}; baselines listed: {'previous_value,training_mean': 30}
  level_scaled: {'feature/None/None': 30}
  level_prev: {'feature/None/None': 30}
  extreme_value, columns named -> seeds: ['level']: 30
  largest_row_dominates, columns named -> seeds: ['None']: 9; ['None', 'None']: 21
  strong_single_column, columns named -> seeds: ['level_scaled']: 30
  fails (1): `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is 0.35824851373882205 against the baseline 'previous_value'
  fails (1): `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is -0.0437479005228012 against the baseline 'previous_value'
  fails (1): `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is -3.64876277056136 against the baseline 'previous_value'. 
  fails (1): `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is 0.3625597824249591 against the baseline 'previous_value'.
 before:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: -37.8850 to +0.9994 (mean -1.9966, sd 7.0345, n 30)
  model: {'linear-0': 26, 'tree-0': 4}; baseline: {'previous_value': 30}; baselines listed: {'previous_value,training_mean': 30}
  level_scaled: {'feature/None/None': 30}
  level_prev: {'feature/None/None': 30}
  fails (1): `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is 0.35824851373882205 against the baseline 'previous_value'
  fails (1): `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is -0.0437479005228012 against the baseline 'previous_value'
  fails (1): `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is -3.64876277056136 against the baseline 'previous_value'. 
  fails (1): `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is 0.3625597824249591 against the baseline 'previous_value'.
 app:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: -37.8850 to +0.9994 (mean -1.9966, sd 7.0345, n 30)
  model: {'linear-0': 26, 'tree-0': 4}; baseline: {'previous_value': 30}; baselines listed: {'previous_value,training_mean': 30}
  level_scaled: {'feature/None/None': 30}
  level_prev: {'feature/None/None': 30}
  fails (1): `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is 0.35824851373880295 against the baseline 'previous_value'
  fails (1): `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is -0.043747900522785654 against the baseline 'previous_valu
  fails (1): `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is -3.6487627705612633 against the baseline 'previous_value'
  fails (1): `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is 0.36255978242499476 against the baseline 'previous_value'

## result.largest_row_reported (run `walk_late_cell`, names F26)
 good:
  passes on 30 of 30 seeds; the run finishes on 30
  skill: -3369746542.4728 to -2401347582.1617 (mean -2751809220.4122, sd 230571778.2449, n 30)
  model: {'linear-0': 30}; baseline: {'previous_value': 30}; baselines listed: {'previous_value,training_mean': 30}
  level_prev: {'feature/None/None': 30}
  extreme_value, columns named -> seeds: ['level_prev']: 30
  largest_row_dominates, columns named -> seeds: ['None']: 30
  headline.largest_row_share: +1.0000 to +1.0000 (mean +1.0000, sd 0.0000, n 30)
  cv.largest_row_share: +0.0085 to +0.0162 (mean +0.0118, sd 0.0021, n 30)
  headline.baseline_largest_row_share: +0.0197 to +0.0377 (mean +0.0269, sd 0.0047, n 30)
  cv.baseline_largest_row_share: +0.0087 to +0.0178 (mean +0.0116, sd 0.0022, n 30)
 F26:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: -3369746542.4728 to -2401347582.1617 (mean -2751809220.4122, sd 230571778.2449, n 30)
  model: {'linear-0': 30}; baseline: {'previous_value': 30}; baselines listed: {'previous_value,training_mean': 30}
  level_prev: {'feature/None/None': 30}
  extreme_value, columns named -> seeds: ['level_prev']: 30
  largest_row_dominates, columns named -> seeds: ['None']: 30
  headline.largest_row_share: +0.9997 to +0.9997 (mean +0.9997, sd 0.0000, n 30)
  cv.largest_row_share: +0.0085 to +0.0162 (mean +0.0118, sd 0.0021, n 30)
  headline.baseline_largest_row_share: +0.0089 to +0.0123 (mean +0.0106, sd 0.0010, n 30)
  cv.baseline_largest_row_share: +0.0087 to +0.0178 (mean +0.0116, sd 0.0022, n 30)
  fails (1): headline.largest_row_share on `walk_late_cell` is 0.999709711934073. The largest squared error of the chosen model on a holdout row, over the sum of t
  fails (1): headline.largest_row_share on `walk_late_cell` is 0.9997008473378556. The largest squared error of the chosen model on a holdout row, over the sum of 
  fails (1): headline.largest_row_share on `walk_late_cell` is 0.9997330584095413. The largest squared error of the chosen model on a holdout row, over the sum of 
  fails (1): headline.largest_row_share on `walk_late_cell` is 0.9996967613610889. The largest squared error of the chosen model on a holdout row, over the sum of 
 before:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: -3369746542.4728 to -2401347582.1617 (mean -2751809220.4122, sd 230571778.2449, n 30)
  model: {'linear-0': 30}; baseline: {'previous_value': 30}; baselines listed: {'previous_value,training_mean': 30}
  level_prev: {'feature/None/None': 30}
  headline.largest_row_share: no value
  cv.largest_row_share: no value
  headline.baseline_largest_row_share: no value
  cv.baseline_largest_row_share: no value
  fails (30): profile.json on `walk_late_cell` has no key headline.largest_row_share, headline.largest_row, headline.baseline_largest_row_share, headline.baseline_l
 app:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: -3369746542.4728 to -2401347582.1617 (mean -2751809220.4122, sd 230571778.2449, n 30)
  model: {'linear-0': 30}; baseline: {'previous_value': 30}; baselines listed: {'previous_value,training_mean': 30}
  level_prev: {'feature/None/None': 30}
  headline.largest_row_share: no value
  cv.largest_row_share: no value
  headline.baseline_largest_row_share: no value
  cv.baseline_largest_row_share: no value
  fails (30): profile.json on `walk_late_cell` has no key headline.largest_row_share, headline.largest_row, headline.baseline_largest_row_share, headline.baseline_l

## data.identifier_survives_shuffle (run `messy_shuffled`, names F27)
 good:
  passes on 30 of 30 seeds; the run finishes on 30
  skill: +0.6595 to +0.8204 (mean +0.7388, sd 0.0404, n 30)
  model: {'linear-0': 30}; baseline: {'chance': 30}; baselines listed: {'chance': 30}
  record_id: {'excluded/id_like/None': 30}
  visits: {'feature/None/None': 30}
 F27:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: +0.6602 to +0.8199 (mean +0.7385, sd 0.0407, n 30)
  model: {'linear-0': 30}; baseline: {'chance': 30}; baselines listed: {'chance': 30}
  record_id: {'feature/None/None': 30}
  visits: {'feature/None/None': 30}
  fails (30): `record_id` on `messy_shuffled` has role 'feature' and reason None. In the 1500 rows with a target it holds 1500 distinct whole numbers that fill 97% 
 before:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: +0.6602 to +0.8199 (mean +0.7385, sd 0.0407, n 30)
  model: {'linear-0': 30}; baseline: {'chance': 30}; baselines listed: {'chance': 30}
  record_id: {'feature/None/None': 30}
  visits: {'feature/None/None': 30}
  fails (30): `record_id` on `messy_shuffled` has role 'feature' and reason None. In the 1500 rows with a target it holds 1500 distinct whole numbers that fill 97% 
 app:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: +0.6457 to +0.7988 (mean +0.7310, sd 0.0356, n 30)
  model: {'linear-0': 30}; baseline: {'chance': 30}; baselines listed: {'chance': 30}
  record_id: {'feature/None/None': 30}
  visits: {'feature/None/None': 30}
  fails (30): `record_id` on `messy_shuffled` has role 'feature' and reason None. In the 1500 rows with a target it holds 1500 distinct whole numbers that fill 97% 

## groups.choice_not_made (run `dominant_entity`, names F28)
 good:
  passes on 30 of 30 seeds; the run finishes on 30
  skill: -0.0117 to +0.0030 (mean -0.0037, sd 0.0040, n 30)
  model: {'linear-0': 30}; baseline: {'training_mean': 30}; baselines listed: {'training_mean': 30}
  selection_not_made, columns named -> seeds: ['None']: 30
  the smallest fold, and the folds scored: {'10 training rows, scored 5': 30}
 F28:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: -0.8718 to +0.0030 (mean -0.0590, sd 0.2122, n 30)
  model: {'linear-0': 28, 'tree-0': 2}; baseline: {'training_mean': 30}; baselines listed: {'training_mean': 30}
  the smallest fold, and the folds scored: {'10 training rows, scored 5': 30}
  fails (2): 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill 0.0030. The training rows are the entity C0000
  fails (2): 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill -0.0037. The training rows are the entity C000
  fails (1): 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill -0.0062. The training rows are the entity C000
  fails (1): 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill -0.0045. The training rows are the entity C000
 before:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: -0.8718 to +0.0030 (mean -0.0590, sd 0.2122, n 30)
  model: {'linear-0': 28, 'tree-0': 2}; baseline: {'training_mean': 30}; baselines listed: {'training_mean': 30}
  the smallest fold, and the folds scored: {'10 training rows, scored 5': 30}
  fails (2): 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill 0.0030. The training rows are the entity C0000
  fails (2): 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill -0.0037. The training rows are the entity C000
  fails (1): 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill -0.0062. The training rows are the entity C000
  fails (1): 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill -0.0045. The training rows are the entity C000
 app:
  passes on 0 of 30 seeds; the run finishes on 30
  skill: -1.1278 to +0.0028 (mean -0.0726, sd 0.2650, n 30)
  model: {'tree-0': 2, 'linear-0': 28}; baseline: {'training_mean': 30}; baselines listed: {'training_mean': 30}
  the smallest fold, and the folds scored: {'10 training rows, scored 5': 30}
  fails (2): 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill -0.0044. The training rows are the entity C000
  fails (2): 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill -0.0034. The training rows are the entity C000
  fails (2): 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill -0.0017. The training rows are the entity C000
  fails (2): 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill -0.0076. The training rows are the entity C000
```

Two of the broken copies land inside the band of their run (F27 in the band
of `messy`, F28 in that of `dominant_entity` on 28 seeds of 30): they are
caught by the column's role and by the missing warning, not by the band,
which is why those checks are structural first.

Each check on a good output with one value changed (47 changes, each of which
must fail its check with that check's own message):

```text
ok  time.previous_value_at_repeats: unchanged outputs passed
ok  leak.copy_excluded_under_time: unchanged outputs passed
ok  leak.excluded_despite_extreme_target: unchanged outputs passed
ok  result.largest_row_reported: unchanged outputs passed
ok  data.identifier_survives_shuffle: unchanged outputs passed
ok  groups.choice_not_made: unchanged outputs passed
ok  time.previous_value_at_repeats | the previous value is not listed -> failed: baselines on `walk_repeated` lists ['training_mean']. One row of this random walk is written twice, so the date 2021-01-21 holds two rows and the time
ok  time.previous_value_at_repeats | the previous value scores 1% more -> failed: the previous_value baseline on `walk_repeated` scores 0.9729981516079783; recomputed by the gate from the source it is 0.963365. The previous value of
ok  time.previous_value_at_repeats | the training mean scores 1% less -> failed: the training_mean baseline on `walk_repeated` scores 295.8460174947532; recomputed by the gate from the source it is 298.834. The previous value of a 
ok  time.previous_value_at_repeats | the headline baseline is the training mean -> failed: the headline baseline on `walk_repeated` is 'training_mean', giving skill -0.0177. The previous value has the lower error (0.9634 against 298.8 for th
ok  time.previous_value_at_repeats | skill is off by 0.001 -> failed: headline.skill on `walk_repeated` is -0.016661; 1 - error / previous-value error, recomputed by the gate, is -0.017661
ok  time.previous_value_at_repeats | skill is null -> failed: `headline.skill` in profile.json on `walk_repeated` is None, not a finite number
ok  leak.copy_excluded_under_time | the copy is a feature -> failed: `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.0007304472852793786 against the baseline 'previous_value'. T
ok  leak.copy_excluded_under_time | the copy is excluded by the other rule -> failed: `level_scaled` on `walk_copy` has role 'excluded', reason 'suspected_leak', leak_rule 'missing_by_class'; skill is 0.0007304472852793786 against the b
ok  leak.copy_excluded_under_time | no warning names the copy -> failed: no warning with code `suspected_leak` names `level_scaled` on `walk_copy`
ok  leak.copy_excluded_under_time | the lag is excluded -> failed: `level_prev` on `walk_copy` has role 'excluded' and reason 'suspected_leak'. It is yesterday's level, known before today's, and the baseline of this r
ok  leak.copy_excluded_under_time | skill is 0.9 -> failed: skill on walk_copy is 0.9, expected between -0.3 and 0.05: with the copy excluded this is the skill of `random_walk`; near 1 means the copy is still a
ok  leak.excluded_despite_extreme_target | no warning names the target -> failed: `extreme_value` warnings on `walk_copy_cell` name no column. The target `level` holds -999 in one row, far outside all its other values, which are the
ok  leak.excluded_despite_extreme_target | the warning names another column -> failed: `extreme_value` warnings on `walk_copy_cell` name ['level_prev']. The target `level` holds -999 in one row, far outside all its other values, which ar
ok  leak.excluded_despite_extreme_target | the copy is a feature -> failed: `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is -0.7124242969623755 against the baseline 'previous_value'
ok  leak.excluded_despite_extreme_target | the lag is excluded -> failed: `level_prev` on `walk_copy_cell` has role 'excluded' and reason 'suspected_leak'. It is yesterday's level, known before today's, and the baseline of t
ok  result.largest_row_reported | headline.largest_row_share is absent -> failed: profile.json on `walk_late_cell` has no key headline.largest_row_share. A regression run says what share of a squared error its single largest row car
ok  result.largest_row_reported | a models record has no largest_row -> failed: profile.json on `walk_late_cell` has no key models[tree-0].largest_row. A regression run says what share of a squared error its single largest row car
ok  result.largest_row_reported | cv has none of the four keys -> failed: profile.json on `walk_late_cell` has no key cv.largest_row_share, cv.largest_row, cv.baseline_largest_row_share, cv.baseline_largest_row. A regression
ok  result.largest_row_reported | the share is of the absolute errors -> failed: headline.largest_row_share on `walk_late_cell` is 0.9997. The largest squared error of the chosen model on a holdout row, over the sum of them, recomp
ok  result.largest_row_reported | the share is null -> failed: `headline.largest_row_share` in profile.json on `walk_late_cell` is None, not a finite number
ok  result.largest_row_reported | the row is another holdout row -> failed: headline.largest_row on `walk_late_cell` is 1. The holdout row with the largest squared error of the chosen model, by the gate's own arithmetic, is so
ok  result.largest_row_reported | the row is a training row -> failed: headline.largest_row on `walk_late_cell` is 0. The holdout row with the largest squared error of the chosen model, by the gate's own arithmetic, is so
ok  result.largest_row_reported | the row is null -> failed: `headline.largest_row` in profile.json on `walk_late_cell` is None, expected int
ok  result.largest_row_reported | the baseline's share is halved -> failed: headline.baseline_largest_row_share on `walk_late_cell` is 0.01503666370360889. The largest squared error of the baseline previous_value on a holdout 
ok  result.largest_row_reported | the baseline's row is the model's -> failed: headline.baseline_largest_row on `walk_late_cell` is 248. The holdout row with the largest squared error of the baseline previous_value, by the gate's
ok  result.largest_row_reported | cv's share is not the chosen configuration's -> failed: cv.largest_row_share and cv.largest_row on `walk_late_cell` are 0.01830427874450439 and 230; they are those of the chosen configuration 'linear-0' in 
ok  result.largest_row_reported | a configuration's row is a holdout row -> failed: over the selection folds, models[linear-0] on `walk_late_cell` states the share 0.01220285249633626 at row 1. For a configuration with a cross-validat
ok  result.largest_row_reported | a configuration's share is null -> failed: over the selection folds, models[tree-0] on `walk_late_cell` states the share None at row 520. For a configuration with a cross-validation mean, and f
ok  result.largest_row_reported | the baseline's share over the folds is null -> failed: over the selection folds, cv.baseline on `walk_late_cell` states the share None at row 807. For a configuration with a cross-validation mean, and for 
ok  result.largest_row_reported | the baseline's share over the folds is 1.5 -> failed: over the selection folds, cv.baseline on `walk_late_cell` states the share 1.5 at row 807. For a configuration with a cross-validation mean, and for t
ok  result.largest_row_reported | no largest_row_dominates warning -> failed: 0 warning(s) with the code `largest_row_dominates` on `walk_late_cell`; the gate counts 1 statement(s) above one half. One warning for each: the model
ok  result.largest_row_reported | one largest_row_dominates warning too many -> failed: 2 warning(s) with the code `largest_row_dominates` on `walk_late_cell`; the gate counts 1 statement(s) above one half. One warning for each: the model
ok  result.largest_row_reported | the baseline over the folds is above one half and no warning is added -> failed: 1 warning(s) with the code `largest_row_dominates` on `walk_late_cell`; the gate counts 2 statement(s) above one half. One warning for each: the model
ok  result.largest_row_reported | no warning names the feature -> failed: `extreme_value` warnings on `walk_late_cell` name no column. The feature `level_prev` holds 999999 in one row, far outside all its other values, which
ok  result.largest_row_reported | the headline baseline is chance -> failed: the headline baseline on `walk_late_cell` is 'chance'; under time order with unique timestamps it is training_mean or previous_value (D10)
ok  data.identifier_survives_shuffle | the identifier is a feature -> failed: `record_id` on `messy_shuffled` has role 'feature' and reason None. In the 1500 rows with a target it holds 1500 distinct whole numbers that fill 97% 
ok  data.identifier_survives_shuffle | the measure is excluded as id-like -> failed: `visits` on `messy_shuffled` has role 'excluded', reason 'id_like' and kind None. Its 1500 distinct whole numbers fill 1.5% of the range from the smal
ok  data.identifier_survives_shuffle | the measure is categorical -> failed: `visits` on `messy_shuffled` has role 'feature', reason None and kind 'categorical'. Its 1500 distinct whole numbers fill 1.5% of the range from the s
ok  data.identifier_survives_shuffle | skill is 0.3 -> failed: skill on messy_shuffled is 0.3, expected between 0.55 and 0.91: the rows of `messy` in another order: the signal is still there under the mess; a wron
ok  groups.choice_not_made | no selection_not_made warning -> failed: 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill -0.0062. The training rows are the entity C000
ok  groups.choice_not_made | two selection_not_made warnings -> failed: 2 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'linear-0' and skill -0.0062. The training rows are the entity C000
ok  groups.choice_not_made | the tree is chosen -> failed: cv.chosen is 'tree-0' and headline.model 'tree-0' on `dominant_entity`; the first configuration that has a cross-validation mean is 'linear-0'. The tr
ok  groups.choice_not_made | the headline model is not the chosen one -> failed: cv.chosen is 'linear-0' and headline.model 'tree-0' on `dominant_entity`; the first configuration that has a cross-validation mean is 'linear-0'. The 
ok  groups.choice_not_made | a fold is not scored -> failed: validation.folds on `dominant_entity` reports scored = [True, True, False, True, True] and cv.n_scored_folds 5. The training rows are the entity C0000
ok  groups.choice_not_made | four folds scored -> failed: validation.folds on `dominant_entity` reports scored = [True, True, True, True, True] and cv.n_scored_folds 4. The training rows are the entity C0000,
ok  groups.choice_not_made | skill is -0.5 -> failed: skill on dominant_entity is -0.5, expected between -0.03 and 0.02: the target is noise, so honest skill is about zero, a little below for the first co
ok  groups.choice_not_made | the first configuration has no mean -> failed: cv.chosen is 'linear-0' and headline.model 'linear-0' on `dominant_entity`; the first configuration that has a cross-validation mean is 'tree-0'. The 

47 changed outputs; problems (with the six unchanged ones): 0
```

### 7.3 What else moved at the gate seed (proof 3)

No seed and no table of the gate as it was has moved:

```text
datasets 30 -> 35 | runs 38 -> 43 | fast runs 33 -> 38 | checks 47 -> 53 | failure modes 31 -> 37
order of the datasets and of the runs kept: True
every dataset seed kept: True
every table byte for byte: True
every run kept: True
every band kept: True | new bands: {'dominant_entity': (-0.03, 0.02)}
the runs of every check kept: True
new datasets and their seeds: {'walk_repeated': 1145, 'walk_copy': 1146, 'walk_copy_cell': 1147, 'walk_late_cell': 1148, 'messy_shuffled': 1149}
new runs: {'walk_repeated': ('walk_repeated', ['--time-col', 'date', '--no-entities']), 'walk_copy': ('walk_copy', ['--time-col', 'date', '--no-entities']), 'walk_copy_cell': ('walk_copy_cell', ['--time-col', 'date', '--no-entities']), 'walk_late_cell': ('walk_late_cell', ['--time-col', 'date', '--no-entities']), 'messy_shuffled': ('messy_shuffled', ['--no-time-order', '--no-entities'])}
new checks: {'time.previous_value_at_repeats': (['walk_repeated'], ['F23']), 'leak.copy_excluded_under_time': (['walk_copy'], ['F24']), 'leak.excluded_despite_extreme_target': (['walk_copy_cell'], ['F25']), 'result.largest_row_reported': (['walk_late_cell'], ['F26']), 'data.identifier_survives_shuffle': (['messy_shuffled'], ['F27']), 'groups.choice_not_made': (['dominant_entity'], ['F28'])}
```

A full gate run on the reference as it stood and one on the reference, with
the new gate, compared leaf by leaf (every value of `profile.json` but
`meta.timing`, every warning as code, column and message, `predictions.csv`
byte for byte):

```text
api_signal_clf: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
binary_numeric: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
day_numbers: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
day_sentinel: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
dominant_entity: MOVED
    warning only in B: selection_not_made [None] one fold trains on 10 rows, fewer than 50 and fewer than it validates on: one unit, or the units of one fold, hold nearly all of the training rows, and the othe
dominant_entity: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
entities: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
few_valued: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
flat_tail: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
full_noise_small: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
full_signal_clf_j1: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
full_signal_clf_j4: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
full_signal_reg_j1: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
full_signal_reg_j4: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
imbalanced: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
int_stamps: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
leak: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
leak_reg: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
messy: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
near_leak: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
noise_clf: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
noise_reg: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
panel: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
quarter_entity: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
random_walk: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
signal_clf: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
signal_reg: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
time_blocks: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
time_blocks_gap: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']
time_blocks_undeclared: keys only in B: ['cv.baseline_largest_row', 'cv.baseline_largest_row_share', 'cv.largest_row', 'cv.largest_row_share', 'headline.baseline_largest_row', 'headline.baseline_largest_row_share', 'headline.largest_row', 'headline.largest_row_share', 'models[].largest_row', 'models[].largest_row_share']

identical in every value: 0 runs: []
identical in every value A has, with keys only B has: 28 runs: ['api_signal_clf', 'binary_numeric', 'day_numbers', 'day_sentinel', 'entities', 'few_valued', 'flat_tail', 'full_noise_small', 'full_signal_clf_j1', 'full_signal_clf_j4', 'full_signal_reg_j1', 'full_signal_reg_j4', 'imbalanced', 'int_stamps', 'leak', 'leak_reg', 'messy', 'near_leak', 'noise_clf', 'noise_reg', 'panel', 'quarter_entity', 'random_walk', 'signal_clf', 'signal_reg', 'time_blocks', 'time_blocks_gap', 'time_blocks_undeclared']
stopped on both sides (no profile.json): 9 runs: ['ambiguous_target', 'dup_noise', 'few_valued_leak', 'large_entity', 'long_rows', 'time_blocks_unstated', 'time_dayfirst', 'unsaid_both', 'unsaid_entities']
moved: 1 runs: ['dominant_entity']
```

Of the 38 runs the gate had: 9 stop on both sides; 28 are identical in every
value the old output has, with only the ten new keys added; one moved,
`dominant_entity`, which gains the `selection_not_made` warning and nothing
else at the gate seed.

### 7.4 The self-test (proof 1)

37 broken copies; every check is seen to fail on the broken copy of the mode
it names, and the three checks that name none are each seen to fail on
another mode's. Run alone on 2026-11-25 it took 1,200 seconds, twenty
minutes (nineteen with the 31 copies of the seventh pass).

```text
ok  [temp dir] F1 caught by model.noise_clf, model.noise_p_value, model.noise_reg
ok  [temp dir] F2 caught by leak.excluded, leak.regression, leak.near_leak_warned
ok  [temp dir] F3 caught by model.headline_recomputed, model.selection_full
ok  [temp dir] F4 caught by model.headline_recomputed, model.metric_clf
ok  [temp dir] F5 caught by task.binary_numeric, task.ambiguous_stops
ok  [temp dir] F6 caught by data.rows_reconcile, data.coercion_reported
ok  [temp dir] F7 caught by time.order_respected, time.no_skill_on_blocks, time.gap_applied
ok  [temp dir] F8 caught by time.persistence_baseline
ok  [temp dir] F6b caught by data.columns_accounted
ok  [temp dir] F6c caught by models.all_accounted
ok  [temp dir] F6d caught by data.long_rows_stop
ok  [temp dir] F7b caught by time.dayfirst_stops
ok  [temp dir] F7c caught by time.integer_keys_distinct
ok  [temp dir] F9 caught by data.holdout_copies_stop
ok  [temp dir] F10 caught by groups.kept_together, groups.no_skill_on_entities
ok  [temp dir] F8b caught by time.panel_previous_value
ok  [temp dir] F7d caught by time.undeclared_stops
ok  [temp dir] F11 caught by statements.time_order_asked, statements.entities_asked
ok  [temp dir] F12 caught by duplicates.group_on_features, duplicates.group_no_skill
ok  [temp dir] F13 caught by groups.holdout_bounded
ok  [temp dir] F14 caught by time.date_numbers_seen
ok  [temp dir] F12b caught by duplicates.group_leak_stops
ok  [temp dir] F14b caught by time.date_sentinel_seen
ok  [temp dir] F15 caught by fast.large_entity_stops, fast.large_entity_kept
ok  [temp dir] F16 caught by model.signal_clf
ok  [temp dir] F17 caught by model.signal_reg
ok  [temp dir] F18 caught by time.undeclared_warns
ok  [temp dir] F19 caught by result.matches_json
ok  [temp dir] F20 caught by output.complete
ok  [temp dir] F21 caught by output.deterministic
ok  [temp dir] F22 caught by output.same_at_any_n_jobs
ok  [temp dir] F23 caught by time.previous_value_at_repeats
ok  [temp dir] F24 caught by leak.copy_excluded_under_time
ok  [temp dir] F25 caught by leak.excluded_despite_extreme_target
ok  [temp dir] F26 caught by result.largest_row_reported
ok  [temp dir] F27 caught by data.identifier_survives_shuffle
ok  [temp dir] F28 caught by groups.choice_not_made
ok  [project dir] F1 caught by model.noise_clf, model.noise_p_value, model.noise_reg
ok  [project dir] F2 caught by leak.excluded, leak.regression, leak.near_leak_warned
ok  [project dir] F3 caught by model.headline_recomputed, model.selection_full
ok  [project dir] F4 caught by model.headline_recomputed, model.metric_clf
ok  [project dir] F5 caught by task.binary_numeric, task.ambiguous_stops
ok  [project dir] F6 caught by data.rows_reconcile, data.coercion_reported
ok  [project dir] F7 caught by time.order_respected, time.no_skill_on_blocks, time.gap_applied
ok  [project dir] F8 caught by time.persistence_baseline
ok  [project dir] F6b caught by data.columns_accounted
ok  [project dir] F6c caught by models.all_accounted
ok  [project dir] F6d caught by data.long_rows_stop
ok  [project dir] F7b caught by time.dayfirst_stops
ok  [project dir] F7c caught by time.integer_keys_distinct
ok  [project dir] F9 caught by data.holdout_copies_stop
ok  [project dir] F10 caught by groups.kept_together, groups.no_skill_on_entities
ok  [project dir] F8b caught by time.panel_previous_value
ok  [project dir] F7d caught by time.undeclared_stops
ok  [project dir] F11 caught by statements.time_order_asked, statements.entities_asked
ok  [project dir] F12 caught by duplicates.group_on_features, duplicates.group_no_skill
ok  [project dir] F13 caught by groups.holdout_bounded
ok  [project dir] F14 caught by time.date_numbers_seen
ok  [project dir] F12b caught by duplicates.group_leak_stops
ok  [project dir] F14b caught by time.date_sentinel_seen
ok  [project dir] F15 caught by fast.large_entity_stops, fast.large_entity_kept
ok  [project dir] F16 caught by model.signal_clf
ok  [project dir] F17 caught by model.signal_reg
ok  [project dir] F18 caught by time.undeclared_warns
ok  [project dir] F19 caught by result.matches_json
ok  [project dir] F20 caught by output.complete
ok  [project dir] F21 caught by output.deterministic
ok  [project dir] F22 caught by output.same_at_any_n_jobs
ok  [project dir] F23 caught by time.previous_value_at_repeats
ok  [project dir] F24 caught by leak.copy_excluded_under_time
ok  [project dir] F25 caught by leak.excluded_despite_extreme_target
ok  [project dir] F26 caught by result.largest_row_reported
ok  [project dir] F27 caught by data.identifier_survives_shuffle
ok  [project dir] F28 caught by groups.choice_not_made
TIMING: full gate 15.3s, per-edit (--fast) 12.8s on the good fixture (2 slow check(s) skipped per edit)

PROOF: every check that names a failure mode, run on that mode's own fixture
| Check | Names | Failed on that fixture | Also failed on the fixture of |
|---|---|---|---|
| data.rows_reconcile | F6 | F6: yes, 2 of 2 runs | - |
| data.columns_accounted | F6b | F6b: yes, 2 of 2 runs | F4, F16 |
| data.coercion_reported | F6 | F6: yes, 2 of 2 runs | - |
| data.long_rows_stop | F6d | F6d: yes, 2 of 2 runs | - |
| data.holdout_copies_stop | F9 | F9: yes, 2 of 2 runs | - |
| models.all_accounted | F6c | F6c: yes, 2 of 2 runs | - |
| task.binary_numeric | F5 | F5: yes, 2 of 2 runs | F4 |
| task.ambiguous_stops | F5 | F5: yes, 2 of 2 runs | - |
| leak.excluded | F2 | F2: yes, 2 of 2 runs | F6, F16 |
| leak.regression | F2 | F2: yes, 2 of 2 runs | F17 |
| leak.near_leak_warned | F2 | F2: yes, 2 of 2 runs | - |
| model.signal_clf | F16 | F16: yes, 2 of 2 runs | - |
| model.signal_reg | F17 | F17: yes, 2 of 2 runs | - |
| model.noise_clf | F1 | F1: yes, 2 of 2 runs | - |
| model.noise_p_value | F1 | F1: yes, 2 of 2 runs | - |
| model.noise_reg | F1 | F1: yes, 2 of 2 runs | F17 |
| model.headline_recomputed | F3, F4 | F3: yes, 2 of 2 runs; F4: yes, 2 of 2 runs | - |
| model.metric_clf | F4 | F4: yes, 2 of 2 runs | F3 |
| model.selection_full | F3 | F3: yes, 2 of 2 runs | F1, F4, F6c |
| time.order_respected | F7 | F7: yes, 2 of 2 runs | - |
| time.no_skill_on_blocks | F7 | F7: yes, 2 of 2 runs | F1 |
| time.gap_applied | F7 | F7: yes, 2 of 2 runs | - |
| time.dayfirst_stops | F7b | F7b: yes, 2 of 2 runs | - |
| time.persistence_baseline | F8 | F8: yes, 2 of 2 runs | F3, F7, F17 |
| time.integer_keys_distinct | F7c | F7c: yes, 2 of 2 runs | F3, F7, F8 |
| time.undeclared_warns | F18 | F18: yes, 2 of 2 runs | - |
| time.undeclared_stops | F7d | F7d: yes, 2 of 2 runs | F11 |
| time.panel_previous_value | F8b | F8b: yes, 2 of 2 runs | F3, F7, F8, F17 |
| groups.kept_together | F10 | F10: yes, 2 of 2 runs | - |
| groups.no_skill_on_entities | F10 | F10: yes, 2 of 2 runs | - |
| statements.time_order_asked | F11 | F11: yes, 2 of 2 runs | - |
| statements.entities_asked | F11 | F11: yes, 2 of 2 runs | - |
| duplicates.group_on_features | F12 | F12: yes, 2 of 2 runs | F10 |
| duplicates.group_no_skill | F12 | F12: yes, 2 of 2 runs | F10 |
| groups.holdout_bounded | F13 | F13: yes, 2 of 2 runs | F10 |
| time.date_numbers_seen | F14 | F14: yes, 2 of 2 runs | F18 |
| time.date_sentinel_seen | F14b | F14b: yes, 2 of 2 runs | F14, F18 |
| duplicates.group_leak_stops | F12b | F12b: yes, 2 of 2 runs | F2 |
| fast.large_entity_stops | F15 | F15: yes, 2 of 2 runs | - |
| fast.large_entity_kept | F15 | F15: yes, 2 of 2 runs | F10, F13 |
| time.previous_value_at_repeats | F23 | F23: yes, 2 of 2 runs | F3, F7, F8, F17 |
| leak.copy_excluded_under_time | F24 | F24: yes, 2 of 2 runs | F2, F7, F8, F17 |
| leak.excluded_despite_extreme_target | F25 | F25: yes, 2 of 2 runs | F2, F7, F24 |
| result.largest_row_reported | F26 | F26: yes, 2 of 2 runs | F4, F7 |
| data.identifier_survives_shuffle | F27 | F27: yes, 2 of 2 runs | F16 |
| groups.choice_not_made | F28 | F28: yes, 2 of 2 runs | F3, F10, F13 |
| result.matches_json | F19 | F19: yes, 2 of 2 runs | - |
| output.complete | F20 | F20: yes, 2 of 2 runs | - |
| output.deterministic | F21 | F21: yes, 2 of 2 runs | - |
| output.same_at_any_n_jobs | F22 | F22: yes, 2 of 2 runs | F21 |
51 of 51 pairs of a check and a failure mode it names were seen failing, over 37 broken fixtures.

FOR INFORMATION: checks that give a reason in place of a failure mode, and the broken fixtures they failed on (no part of the verdict)
| Check | Failed on the fixture of |
|---|---|
| leak.control_kept | F6 |
| time.flat_holdout_reported | F3, F7, F8 |
| uncertainty.reported | F1, F4, F16 |
SELFTEST: PASS (good fixture passes twice with identical results; 37 failure mode(s) each caught by a check that names it; temp and project dirs agree)
```

The matrix (`verify\smoke_test.py --matrix`), and the list of checks:

| Failure mode | Description | Caught by |
|---|---|---|
| F1 | the chosen model is refitted on all rows before the holdout is scored | model.noise_clf, model.noise_p_value, model.noise_reg |
| F2 | the leak screen is removed, so a copy of the target is a feature | leak.excluded, leak.regression, leak.near_leak_warned |
| F3 | the headline is the best cross-validation score, not the holdout score | model.headline_recomputed, model.selection_full |
| F4 | the headline metric is accuracy | model.headline_recomputed, model.metric_clf |
| F5 | the task is decided from the column's type alone | task.binary_numeric, task.ambiguous_stops |
| F6 | rows with any missing value are dropped before counting | data.rows_reconcile, data.coercion_reported |
| F7 | the split is shuffled even when a time column is declared | time.order_respected, time.no_skill_on_blocks, time.gap_applied |
| F8 | the regression baseline is always the training mean | time.persistence_baseline |
| F6b | a column is lost from the accounting (constant columns are dropped with no record) | data.columns_accounted |
| F6c | a model family is lost from the models table (only the chosen configuration's family is listed) | models.all_accounted |
| F6d | data rows longer than the header are read silently, with no error | data.long_rows_stop |
| F7b | a day-first time column is parsed leniently, so the run goes ahead on the wrong order | time.dayfirst_stops |
| F7c | whole-number time keys are read as floats, so distinct instants above 2**53 become one timestamp | time.integer_keys_distinct |
| F9 | holdout rows with copies among the training rows are scored with no error | data.holdout_copies_stop |
| F10 | an entity has rows on both sides of the split although the entity column was declared | groups.kept_together, groups.no_skill_on_entities |
| F8b | a panel is scored against the training mean although the entity column was declared | time.panel_previous_value |
| F7d | a table with an undeclared date column is profiled under a random split without the caller's say | time.undeclared_stops |
| F11 | a statement the caller did not make is assumed: nothing said about time order, or about entities, and the run goes ahead | statements.time_order_asked, statements.entities_asked |
| F12 | duplicates='group' forms its sets with the target, so rows that agree in every feature are on both sides of the split | duplicates.group_on_features, duplicates.group_no_skill |
| F13 | the grouped holdout takes far more than a quarter because a large entity was drawn early | groups.holdout_bounded |
| F14 | dates written as whole numbers (YYYYMMDD) are a numeric feature | time.date_numbers_seen |
| F12b | duplicates='group' and a column the leak screen removes: feature combinations are on both sides and the run goes ahead | duplicates.group_leak_stops |
| F14b | one value that is no date, and a column of dates written YYYYMMDD is a numeric feature | time.date_sentinel_seen |
| F15 | the fast profile leaves out an entity larger than its cap, or one of a quarter or more of the rows, and reports on the rest | fast.large_entity_stops, fast.large_entity_kept |
| F16 | the positive class is inverted: skill that is there is reported as far below zero | model.signal_clf |
| F17 | squared error is read as higher-is-better: the worst configuration is chosen | model.signal_reg |
| F18 | a date column under --no-time-order gets no warning | time.undeclared_warns |
| F19 | the result object is not the file: ProfileResult.headline holds rounded numbers | result.matches_json |
| F20 | a part of the output is missing: the limitations statement is empty | output.complete |
| F21 | a clock value outside meta.timing: two identical runs differ | output.deterministic |
| F22 | the resamples depend on the number of cores | output.same_at_any_n_jobs |
| F23 | under time order with no entity column the previous value is a baseline only while every timestamp is unique: one repeated row leaves the training mean alone | time.previous_value_at_repeats |
| F24 | the leak screen reads a column with its tree alone: under time order a rescaled copy of the target stays a feature | leak.copy_excluded_under_time |
| F25 | no row is set aside from the leak screen: one extreme target cell switches it off, and a copy of the target stays a feature | leak.excluded_despite_extreme_target |
| F26 | the largest row's share is taken of the absolute errors, not of the squared errors | result.largest_row_reported |
| F27 | the id-like rule for whole numbers is D19's clause alone: an identifier is a feature once the rows are shuffled | data.identifier_survives_shuffle |
| F28 | under the grouped scheme the choice between configurations is made however few rows the folds beside one large entity train on | groups.choice_not_made |

```text
data.rows_reconcile  [F6]  every row count equals the gate's own count from the source, and they add up
data.columns_accounted  [F6b]  every source column appears exactly once, with the expected role and reason
data.coercion_reported  [F6]  text cells in a numeric column are counted as coerced, n/a cells as missing
data.long_rows_stop  [F6d]  data rows one field longer than the header stop with the named input error
data.holdout_copies_stop  [F9]  noise with every row present twice stops with the named copies error, and writes no headline
models.all_accounted  [F6c]  the models table lists every family of the profile, each with a status
task.binary_numeric  [F5]  a target stored as the integers 0 and 1 is classification, decided by the rule
task.ambiguous_stops  [F5]  an integer target with 7 distinct values stops with the named task error
leak.excluded  [F2]  both leak columns are excluded with their reasons; skill is not near 1
leak.control_kept  [-]  a strongly predictive but legitimate column stays a feature, with no warning
leak.regression  [F2]  a rescaled copy of a numeric target is excluded and listed
leak.near_leak_warned  [F2]  a copy of the target with 2% of its labels flipped stays a feature, with a warning
model.signal_clf  [F16]  clear skill where a binary target depends on three features
model.signal_reg  [F17]  clear skill where a numeric target is a known function of two features
model.noise_clf  [F1]  no skill where the target is a coin flip
model.noise_p_value  [F1]  a coin flip is not reported as significant
model.noise_reg  [F1]  no positive skill where a numeric target is independent of every feature
model.headline_recomputed  [F3, F4]  the gate recomputes the metric and the skill from predictions.csv and the source
model.metric_clf  [F4]  on a 95/5 target the metric is ROC AUC against 0.5, not accuracy
model.selection_full  [F3]  (slow)  full profile on noise: the headline is the holdout score, not the best CV score
time.order_respected  [F7]  with a time column the holdout is exactly the latest rows, cut at a whole date
time.no_skill_on_blocks  [F7]  no skill where only a shuffled split could find any
time.gap_applied  [F7]  --gap 3 removes exactly the 3 timestamps before the holdout from training
time.dayfirst_stops  [F7b]  a day/month/year time column stops with the named time-column error
time.persistence_baseline  [F8]  on a random walk the baseline is the previous value and skill is about zero
time.integer_keys_distinct  [F7c]  64-bit integer timestamps above 2**53 stay distinct: the timestamp counts equal the gate's own, and the baseline is the previous value
time.undeclared_warns  [F18]  a date column and --no-time-order: random scheme, with a warning, and the statement recorded
time.undeclared_stops  [F7d]  a date column, no --time-col and no --no-time-order: the run stops with the named error and writes no headline
time.panel_previous_value  [F8b]  twelve random walks with --group-col: the baseline is each series' own previous value, recomputed by the gate, and skill is about zero
groups.kept_together  [F10]  with --group-col no entity has rows on both sides of the split, by the gate's own count from predictions.csv, and the output reports the grouped scheme
groups.no_skill_on_entities  [F10]  no skill where only an entity on both sides of the split could give any
statements.time_order_asked  [F11]  nothing said about time order or entities, on a table with no date column: the run stops with the named error, and its message asks for both
statements.entities_asked  [F11]  a time column, one series with unique timestamps, and nothing said about entities: the run stops with the named error
duplicates.group_on_features  [F12]  with --duplicates group no combination of feature values is on both sides of the split, by the gate's own count from predictions.csv
duplicates.group_no_skill  [F12]  noise in a table of two few-valued columns, with --duplicates group: no skill
groups.holdout_bounded  [F13]  one entity of three quarters of the rows: the holdout is a quarter, made of whole small entities, by the gate's own count, and the large one is named
time.date_numbers_seen  [F14]  a day written as a whole number, YYYYMMDD, and --no-time-order: the column is excluded as a date and the warning for a date column names it
time.date_sentinel_seen  [F14b]  a day written YYYYMMDD with 99991231 in three rows of ten: the column is still excluded as a date and the warning names it
duplicates.group_leak_stops  [F12b]  --duplicates group and a column that copies the label: once the leak screen has excluded it, feature combinations are on both sides, and the run stops
fast.large_entity_stops  [F15]  the fast profile and an entity of more rows than its cap: the run stops and sends the caller to the full profile
fast.large_entity_kept  [F15]  the fast profile and an entity of a quarter or more of the rows that fits under its cap: the entity is kept, by the gate's own count
time.previous_value_at_repeats  [F23]  a random walk with one row written twice: both baselines are scored, the previous value as the mean target of the timestamp before, and skill is about zero
leak.copy_excluded_under_time  [F24]  under time order a rescaled copy of a random walk's target is excluded, the lag stays a feature, and skill is about zero
leak.excluded_despite_extreme_target  [F25]  one target cell of -999 in a training row: the rescaled copy of the target is still excluded, and a warning names the value
result.largest_row_reported  [F26]  one holdout cell of the feature set to 999999: the share of the squared error that its largest row carries is as the gate recomputes it, for the model and the baseline, with one warning for each share above one half
data.identifier_survives_shuffle  [F27]  `messy` with its rows shuffled: the identifier is still excluded as id-like, the whole-number measure is still a feature, and the signal is still there
groups.choice_not_made  [F28]  one entity of three quarters of the rows: no choice between configurations is made on folds that train on a few rows, the one chosen is the first that has a mean, a warning says so, and skill is about zero
time.flat_holdout_reported  [-]  a target flat across the holdout: the run finishes, skill is null with a warning, the score and both baselines are reported
uncertainty.reported  [-]  the interval contains the score; row bootstrap with a p-value, or block bootstrap labelled approximate
result.matches_json  [F19]  through the Python function, every part of ProfileResult equals profile.json
output.complete  [F20]  every required key is present and populated; no NaN or infinity anywhere
output.deterministic  [F21]  two runs with the same seed give identical files apart from meta.timing
output.same_at_any_n_jobs  [F22]  (slow)  full profile at --n-jobs 1 and 4: identical files apart from timing and n_jobs
```

### 7.5 The time of a gate run (proof 8)

Three rounds on the reference with nothing else running on the machine,
2026-11-25:

```text
round 1  per edit (--fast)    14.9 s  GATE: PASS (51 checks)
round 1  full                 20.5 s  GATE: PASS (53 checks)
round 2  per edit (--fast)    15.0 s  GATE: PASS (51 checks)
round 2  full                 22.3 s  GATE: PASS (53 checks)
round 3  per edit (--fast)    16.1 s  GATE: PASS (51 checks)
round 3  full                 20.9 s  GATE: PASS (53 checks)
```

The gate as it stood took 13.7 and 17.8 seconds on a quiet machine, one round, the night before: the
five new runs are five more starts, 0.25 seconds apart.

## 8. The suite against the changed judge (proofs 4 and 5)

No test was written by the harness session. The prototypes of every new and
every changed test are kept outside the repo, with the scripts that made them
and that ran them; the plan (`plan\test-plan.md`, "Eighth pass (CP9)") has
each test's recipe, and the tests stage writes the real ones.

### 8.1 The mutation check against the suite as it is (proof 4)

`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B verify\mutation_check.py --jobs 8 --keep`,
with the reference and the gate of this pass and the suite as the seventh pass
left it. The good reference fails exactly the seven cases the plan names under
D59, each of which pins the previous value as it was before D74; the eighteen
new variants have no test yet; every older variant is caught.

```text
Wed Nov 25 02:14:42 GMTST 2026
116caf510038874f
mutation check: 89 case(s), 8 at a time, projects under <temp>\mlprof-mutation-g86uxjkv
  note: docs/DECISIONS.md: copied from the repo
  note: examples/customers.csv: copied from the repo
  note: tests/test_judge_tools.py and tests/test_judge_tools_pass8.py are not run here: they run the judge's other tools on the application
good   fails 7 of 1921: test_guard_review5.py::test_m23_the_panel_warning_starts_just_above_one_half[just_above], test_guard_review5.py::test_m23_the_panel_warning_starts_just_above_one_half[just_below], test_metrics.py::test_f8_previous_value_refuses_repeated_keys, test_panel_stop.py::test_the_statement_lets_the_run_go_ahead_as_before, test_time_pipeline.py::test_panel_uses_training_mean_and_flags_persistence[False] (and 2 more)   [795 s]
F1     fails 29: test_chosen_failure.py::test_every_configuration_failing_after_selection_raises_all_models_failed, test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration, test_chosen_failure.py::test_holdout_rows_reach_one_configuration_at_a_time, test_duplicates.py::test_every_row_twice_stops_the_run[classification] (and 24 more)   [881 s]
F2     fails 18: test_duplicates_group.py::test_copies_that_appear_when_the_leak_screen_excludes_a_column_stop_the_run, test_group_after_leak.py::test_a_copy_of_the_target_under_group_stops_the_run, test_group_after_leak.py::test_no_combination_of_the_features_is_on_both_sides_in_a_finished_run[noise_with_status], test_group_after_leak.py::test_no_combination_of_the_features_is_on_both_sides_in_a_finished_run[twice_with_status], test_group_after_leak.py::test_one_combination_on_both_sides_is_enough (and 13 more)   [840 s]
F3     fails 24: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[holdout], test_chosen_failure.py::test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration, test_edges.py::test_negative_skill_reported_as_measured, test_full_profile.py::test_f3_headline_is_chosen_configuration_on_the_holdout[classification] (and 19 more)   [790 s]
F4     fails 14: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[holdout], test_chosen_failure.py::test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration, test_edges.py::test_negative_skill_reported_as_measured, test_full_profile.py::test_f3_headline_is_chosen_configuration_on_the_holdout[classification] (and 9 more)   [755 s]
F5     fails 58: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[holdout], test_chosen_failure.py::test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration, test_duplicates.py::test_values_are_compared_as_a_step_receives_them, test_edges.py::test_edge_raises_its_named_error[holdout_one_class] (and 53 more)   [831 s]
F6     fails 148: test_date_numbers.py::test_a_date_number_column_is_a_time_column_like_any_number[nullable_int], test_date_sentinels.py::test_the_time_column_is_judged_over_every_value_whatever_its_row, test_duplicates.py::test_values_are_compared_as_a_step_receives_them, test_edges.py::test_dataframe_outcomes_under_time_order, test_feature_names.py::test_the_missing_level_counts (and 143 more)   [751 s]
F7     fails 212: test_console_summary.py::test_summary_says_why_there_is_no_interval, test_console_summary.py::test_summary_under_time_order_says_why_there_is_no_p_value, test_date_numbers.py::test_a_date_number_column_is_a_time_column_like_any_number[int64], test_date_numbers.py::test_a_date_number_column_is_a_time_column_like_any_number[nullable_int], test_date_numbers.py::test_eight_digit_text_is_accepted_as_the_time_column (and 207 more)   [827 s]
F8     fails 16: test_groups_time.py::test_each_entity_has_its_own_previous_value[0], test_groups_time.py::test_each_entity_has_its_own_previous_value[2], test_groups_time.py::test_the_command_line_takes_both_columns, test_guard_review3.py::test_m9_the_leak_screen_uses_the_previous_value, test_object_numbers.py::test_object_whole_numbers_are_exact[decimal] (and 11 more)   [1227 s]
F9     fails 17: test_date_numbers.py::test_the_copies_stop_names_a_date_number_column, test_date_sentinels.py::test_it_is_named_when_nothing_is_said_and_by_the_copies_stop, test_duplicates.py::test_command_line_takes_duplicates, test_duplicates.py::test_every_row_twice_stops_the_run[classification], test_duplicates.py::test_every_row_twice_stops_the_run[regression] (and 12 more)   [1133 s]
F6b    fails 50: test_columns.py::test_f6_every_column_once_in_file_order, test_columns.py::test_roles_judged_on_usable_rows, test_columns.py::test_rule_table_one_column_per_rule[const_num], test_columns.py::test_rule_table_one_column_per_rule[const_text], test_columns.py::test_separator_rule_takes_precedence (and 45 more)   [1215 s]
F6c    fails 19: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[holdout], test_chosen_failure.py::test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration, test_chosen_failure.py::test_holdout_rows_reach_one_configuration_at_a_time, test_full_profile.py::test_eighteen_configurations_exactly[classification] (and 14 more)   [1249 s]
F6d    fails 6: test_csv_edges.py::test_a_line_pandas_reads_as_the_header_is_not_skipped[form_feed], test_csv_edges.py::test_a_line_pandas_reads_as_the_header_is_not_skipped[two_quotes], test_guard_header_lines.py::test_a_line_pandas_reads_as_the_header_before_a_two_column_header_is_an_input_error[form_feed], test_guard_header_lines.py::test_a_line_pandas_reads_as_the_header_before_a_two_column_header_is_an_input_error[no_break_space], test_guard_header_lines.py::test_a_line_pandas_reads_as_the_header_before_a_two_column_header_is_an_input_error[two_quotes_round_a_space] (and 1 more)   [1224 s]
F7b    fails 34: test_edges.py::test_edge_raises_its_named_error[time_column_invalid], test_split_time.py::test_f7_read_time_refuses_other_forms[day_first], test_split_time.py::test_f7_read_time_refuses_other_forms[month_first], test_split_time.py::test_f7_read_time_refuses_other_forms[month_name], test_split_time.py::test_f7_read_time_refuses_other_forms[year_month] (and 29 more)   [1254 s]
F7c    fails 11: test_object_numbers.py::test_object_whole_numbers_are_exact[decimal], test_object_numbers.py::test_object_whole_numbers_are_exact[python_int], test_whole_numbers.py::test_nanosecond_stamps_are_distinct_timestamps[Int64], test_whole_numbers.py::test_nanosecond_stamps_are_distinct_timestamps[UInt64], test_whole_numbers.py::test_nanosecond_stamps_are_distinct_timestamps[int64] (and 6 more)   [1217 s]
F10    fails 40: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_together_whatever_their_target, test_duplicates_features.py::test_the_command_line_uses_the_new_names, test_duplicates_features.py::test_the_count_and_the_warning_have_their_new_names, test_duplicates_features.py::test_the_count_is_of_rows_that_agree_in_every_feature (and 35 more)   [1211 s]
F7d    fails 7: test_date_numbers.py::test_it_is_named_when_nothing_is_said_about_time_order, test_date_sentinels.py::test_it_is_named_when_nothing_is_said_and_by_the_copies_stop, test_time_order_stated.py::test_a_date_column_and_no_word_about_time_stops_the_run[datetime], test_time_order_stated.py::test_a_date_column_and_no_word_about_time_stops_the_run[iso_text], test_time_order_stated.py::test_the_command_line_states_it (and 2 more)   [1458 s]
F8b    fails 6: test_groups_time.py::test_an_entity_with_no_earlier_row_gets_the_training_mean, test_groups_time.py::test_each_entity_has_its_own_previous_value[0], test_groups_time.py::test_each_entity_has_its_own_previous_value[2], test_groups_time.py::test_the_command_line_takes_both_columns, test_guard_review6.py::test_m28_eight_digit_text_orders_a_panel_and_read_time_reads_it (and 1 more)   [1532 s]
F11    fails 26: test_date_numbers.py::test_it_is_named_when_nothing_is_said_about_time_order, test_date_sentinels.py::test_it_is_named_when_nothing_is_said_and_by_the_copies_stop, test_panel_stop.py::test_a_persistent_panel_with_no_entity_column_stops, test_panel_stop.py::test_the_command_line_says_there_are_no_entities, test_statements.py::test_none_passed_explicitly_is_not_said (and 21 more)   [1054 s]
F12    fails 6: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_together_whatever_their_target, test_duplicates_features.py::test_the_count_is_of_rows_that_agree_in_every_feature, test_fast_cap_groups.py::test_it_is_for_an_entity_column_with_no_time_column_only, test_group_after_leak.py::test_no_combination_of_the_features_is_on_both_sides_in_a_finished_run[few_valued] (and 1 more)   [1494 s]
F13    fails 11: test_grouped_holdout.py::test_a_large_entity_is_never_held_out_when_smaller_ones_can_fill_a_quarter[100], test_grouped_holdout.py::test_a_large_entity_is_never_held_out_when_smaller_ones_can_fill_a_quarter[120], test_grouped_holdout.py::test_a_large_entity_is_never_held_out_when_smaller_ones_can_fill_a_quarter[240], test_grouped_holdout.py::test_a_large_unit_kept_in_training_is_named, test_grouped_holdout.py::test_a_unit_too_large_for_a_sound_split_is_the_holdout_and_is_named (and 6 more)   [1491 s]
F14    fails 47: test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[decimal], test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[float_with_gaps], test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[int64], test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[nullable_int], test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[python_ints] (and 42 more)   [1483 s]
F12b   fails 4: test_group_after_leak.py::test_a_copy_of_the_target_under_group_stops_the_run, test_group_after_leak.py::test_one_combination_on_both_sides_is_enough, test_group_after_leak.py::test_the_command_line_stops_too, test_group_after_leak.py::test_the_stop_names_a_date_column_that_was_excluded_too   [1501 s]
F14b   fails 37: test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[29_feb_1900], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[29_feb_2023], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[30_percent_of_the_rows_99991231], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[31_april], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[5_percent_of_the_rows_99991231] (and 32 more)   [1470 s]
F15    fails 7: test_fast_cap_groups.py::test_an_entity_larger_than_the_cap_stops_the_fast_profile, test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept, test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run, test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[0], test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[1] (and 2 more)   [1429 s]
F16    fails 13: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[holdout], test_chosen_failure.py::test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration, test_chosen_failure.py::test_holdout_rows_reach_one_configuration_at_a_time, test_edges.py::test_negative_skill_reported_as_measured (and 8 more)   [1417 s]
F17    fails 4: test_full_profile.py::test_f3_headline_is_chosen_configuration_on_the_holdout[regression], test_guard_review6.py::test_m6_an_infinite_holdout_prediction_in_a_regression_falls_back, test_non_finite.py::test_a_holdout_score_that_is_not_finite_stops_the_run_without_a_fallback, test_pipeline.py::test_selection_highest_auc_lowest_mse   [1452 s]
F18    fails 17: test_console_summary.py::test_summary_shows_both_intervals_the_p_value_and_each_warning, test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[decimal], test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[float_with_gaps], test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[int64], test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[nullable_int] (and 12 more)   [1420 s]
F19    fails 27: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[holdout], test_chosen_failure.py::test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration, test_edges.py::test_negative_skill_reported_as_measured, test_full_profile.py::test_f3_headline_is_chosen_configuration_on_the_holdout[classification] (and 22 more)   [1427 s]
F20    fails 4: test_edges.py::test_limitations_and_caveat_are_the_fixed_texts[random-classification], test_edges.py::test_limitations_and_caveat_are_the_fixed_texts[random-regression], test_edges.py::test_limitations_and_caveat_are_the_fixed_texts[time-classification], test_edges.py::test_limitations_and_caveat_are_the_fixed_texts[time-regression]   [1430 s]
F21    fails 12: test_duplicates.py::test_the_rule_needs_a_share_and_a_count[0_of_1000], test_duplicates.py::test_the_rule_needs_a_share_and_a_count[10_of_1000], test_duplicates.py::test_the_rule_needs_a_share_and_a_count[9_of_100], test_duplicates_group.py::test_under_time_order_it_changes_nothing, test_full_profile.py::test_command_line_defaults_to_full (and 7 more)   [1438 s]
F22    fails 1: test_full_profile.py::test_same_result_at_one_and_two_jobs   [1427 s]
F23    fails 0   NOT CAUGHT: no test fails on this variant   [1012 s]
F24    fails 0   NOT CAUGHT: no test fails on this variant   [816 s]
F25    fails 0   NOT CAUGHT: no test fails on this variant   [734 s]
F26    fails 0   NOT CAUGHT: no test fails on this variant   [729 s]
F27    fails 0   NOT CAUGHT: no test fails on this variant   [721 s]
F28    fails 0   NOT CAUGHT: no test fails on this variant   [723 s]
M1     fails 2: test_guard_final_fit.py::test_m1_final_fit_statistics_ignore_the_holdout, test_guard_review6.py::test_m1_the_final_fit_is_centred_on_its_own_rows   [700 s]
M2     fails 2: test_feature_names.py::test_a_level_only_in_rows_with_no_target_does_not_count, test_guard_final_fit.py::test_m2_levels_come_from_the_rows_of_each_fit   [691 s]
M3     fails 2: test_guard_leak_screen.py::test_m3_missing_by_class_counts_training_rows_only, test_guard_leak_screen.py::test_m3_single_column_skill_is_fitted_on_fold_training_rows_only   [543 s]
M4     fails 3: test_guard_uncertainty.py::test_m4_block_bootstrap_draws_whole_blocks, test_skill_undefined.py::test_no_skill_interval_when_fewer_than_half_the_resamples_have_a_baseline_error, test_skill_undefined.py::test_skill_undefined_and_no_skill_interval_are_two_codes   [535 s]
M5     fails 2: test_guard_review6.py::test_m5_a_shuffle_that_ties_counts_where_only_some_of_them_tie, test_guard_uncertainty.py::test_m5_ties_count_in_the_p_value   [519 s]
M6     fails 2: test_guard_review6.py::test_m6_an_infinite_holdout_prediction_in_a_regression_falls_back, test_non_finite.py::test_a_prediction_that_is_not_finite_counts_as_raising   [512 s]
M7     fails 4: test_fast_profile_error.py::test_fast_error_stands_when_the_full_profile_would_run[eight_dates_of_450], test_fast_profile_error.py::test_fast_error_stands_when_the_full_profile_would_run[twelve_dates_of_2500_latest_over_the_cap], test_fast_profile_error.py::test_fast_raises_the_full_profiles_error_when_full_would_stop_too[nine_dates_of_450_gap_2], test_time_pipeline.py::test_wide_panel_under_fast_raises   [511 s]
M8     fails 2: test_guard_review3.py::test_m8_the_interval_is_the_middle_95_percent, test_guard_review6.py::test_m8_the_interval_from_whole_entities_is_the_middle_95_percent   [515 s]
M9     fails 14: test_groups_time.py::test_an_entity_with_no_earlier_row_gets_the_training_mean, test_groups_time.py::test_each_entity_has_its_own_previous_value[0], test_groups_time.py::test_each_entity_has_its_own_previous_value[2], test_groups_time.py::test_the_command_line_takes_both_columns, test_groups_time.py::test_the_leak_screen_measures_a_lag_against_the_entity_baseline (and 9 more)   [515 s]
M10    fails 2: test_guard_review3.py::test_m10_the_fast_cap_is_a_sample_from_the_seed, test_guard_review6.py::test_m10_the_fast_cap_is_a_sample_whatever_the_order_of_the_file   [509 s]
M11    fails 2: test_guard_review3.py::test_m11_cv_std_is_the_sample_deviation, test_guard_review6.py::test_m11_cv_std_of_two_scored_folds_is_their_sample_deviation   [522 s]
M12    fails 3: test_guard_review3.py::test_m12_tied_predictions_in_a_resample, test_guard_review6.py::test_m12_tied_predictions_in_a_resample_of_blocks_and_of_entities[blocks], test_guard_review6.py::test_m12_tied_predictions_in_a_resample_of_blocks_and_of_entities[entities]   [505 s]
M13    fails 1: test_guard_review3.py::test_m13_n_imputed_counts_rows_that_reach_a_model   [517 s]
M14    fails 4: test_read_once.py::test_a_text_cell_of_an_object_target_in_a_row_with_no_time_value, test_read_once.py::test_a_text_cell_of_an_object_time_column_in_a_row_with_no_target, test_read_once.py::test_a_time_value_beyond_64_bits_in_a_row_with_no_target, test_read_once.py::test_a_whole_number_beyond_64_bits_in_a_target_row_with_no_time_value   [512 s]
M15    fails 4: test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_together_whatever_their_target, test_duplicates_group.py::test_each_set_of_copies_stays_on_one_side_and_in_one_fold, test_groups_random.py::test_no_entity_on_both_sides_or_in_two_folds[classification], test_groups_random.py::test_no_entity_on_both_sides_or_in_two_folds[regression]   [511 s]
M16    fails 7: test_groups_time.py::test_an_entity_with_no_earlier_row_gets_the_training_mean, test_groups_time.py::test_each_entity_has_its_own_previous_value[0], test_groups_time.py::test_each_entity_has_its_own_previous_value[2], test_groups_time.py::test_the_command_line_takes_both_columns, test_groups_time.py::test_the_leak_screen_measures_a_lag_against_the_entity_baseline (and 2 more)   [514 s]
M17    fails 7: test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept, test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run, test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[0], test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[1], test_fast_cap_groups.py::test_the_cap_keeps_whole_entities_every_large_one_and_no_more_than_it_can (and 2 more)   [515 s]
M18    fails 6: test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite[columns], test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite[input], test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite[meta], test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite[rows], test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite[task] (and 1 more)   [511 s]
M19    fails 7: test_panel_stop.py::test_a_persistent_panel_with_no_entity_column_stops, test_panel_stop.py::test_the_command_line_says_there_are_no_entities, test_statements.py::test_nothing_said_about_entities_stops_whatever_the_table_holds[a_panel_that_does_not_persist], test_statements.py::test_nothing_said_about_entities_stops_whatever_the_table_holds[classification], test_statements.py::test_nothing_said_about_entities_stops_whatever_the_table_holds[one_series] (and 2 more)   [482 s]
M20    fails 3: test_guard_review4.py::test_an_infinite_float_in_an_object_target_is_a_missing_target, test_guard_review4.py::test_an_object_target_of_numbers_with_an_infinite_one, test_guard_review4.py::test_m20_infinite_cells_of_the_target_and_the_time_column_are_counted   [510 s]
M21    fails 2: test_groups_random.py::test_the_interval_resamples_whole_entities_and_there_is_no_p_value, test_guard_review6.py::test_m21_a_resample_of_whole_entities_has_one_score   [522 s]
M22    fails 17: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_together_whatever_their_target, test_duplicates_features.py::test_the_command_line_uses_the_new_names, test_duplicates_features.py::test_the_count_and_the_warning_have_their_new_names, test_duplicates_features.py::test_the_count_is_of_rows_that_agree_in_every_feature (and 12 more)   [514 s]
M23    fails 1: test_guard_review6.py::test_m23_a_panel_between_the_two_cut_offs_is_warned_about_on_the_command_line   [514 s]
M24    fails 2: test_statements.py::test_one_message_asks_for_everything_that_is_missing, test_statements.py::test_the_command_line_asks_for_each_missing_pair   [501 s]
M25    fails 3: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_the_count_is_of_rows_that_agree_in_every_feature, test_grouped_holdout.py::test_the_rule_holds_for_rows_that_agree_in_every_feature   [519 s]
M26    fails 5: test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run, test_grouped_holdout.py::test_a_unit_too_large_for_a_sound_split_is_the_holdout_and_is_named, test_grouped_holdout.py::test_one_warning_for_each_large_unit_and_a_share_of_its_own_class, test_grouped_holdout.py::test_the_holdout_holds_a_quarter_and_needs_every_unit_it_has[a_handful_of_large], test_grouped_holdout.py::test_what_the_holdout_can_spare_goes_back   [513 s]
M27    fails 3: test_grouped_holdout.py::test_a_large_unit_kept_in_training_is_named, test_grouped_holdout.py::test_one_warning_for_each_large_unit_and_a_share_of_its_own_class, test_grouped_holdout.py::test_the_rule_holds_for_rows_that_agree_in_every_feature   [515 s]
M28    fails 4: test_date_numbers.py::test_eight_digit_text_is_accepted_as_the_time_column, test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[text], test_date_sentinels.py::test_read_time_refuses_it_too[text], test_guard_review6.py::test_m28_eight_digit_text_orders_a_panel_and_read_time_reads_it   [515 s]
M29    fails 4: test_date_numbers.py::test_the_date_rule_comes_before_the_id_like_rule[int64], test_date_numbers.py::test_the_date_rule_comes_before_the_id_like_rule[text], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[an_infinite_cell], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[floats_with_gaps_and_an_infinite_cell]   [521 s]
M30    fails 2: test_duplicates.py::test_the_rule_in_regression, test_guard_review5.py::test_m30_a_copy_at_step_8a_still_needs_the_same_target   [524 s]
M31    fails 19: test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[30_percent_of_the_rows_99991231], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[60_percent_of_the_rows_99991231], test_date_sentinels.py::test_a_sentinel_counts_once_however_many_cells_hold_it, test_date_sentinels.py::test_a_time_column_that_is_no_date_column_is_a_number_as_before, test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[float] (and 14 more)   [523 s]
M32    fails 8: test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[float], test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[int64], test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[text], test_date_sentinels.py::test_read_time_refuses_it_too[float], test_date_sentinels.py::test_read_time_refuses_it_too[int64] (and 3 more)   [514 s]
M33    fails 1: test_group_after_leak.py::test_one_combination_on_both_sides_is_enough   [513 s]
M34    fails 1: test_fast_cap_groups.py::test_when_the_full_profile_would_stop_too_its_error_is_raised   [516 s]
M35    fails 5: test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept, test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run, test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[0], test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[1], test_fast_cap_groups.py::test_the_cap_keeps_whole_entities_every_large_one_and_no_more_than_it_can   [515 s]
M36    fails 2: test_review6_small.py::test_the_date_warning_holds_the_caveat_of_its_scheme[grouped_by_an_entity_column], test_review6_small.py::test_the_date_warning_holds_the_caveat_of_its_scheme[grouped_by_rows_that_agree]   [514 s]
M37    fails 1: test_group_after_leak.py::test_the_grouped_warning_says_what_group_measures   [515 s]
M38    fails 2: test_guard_review6.py::test_m38_one_fold_not_scored_record_for_each_fold_not_scored[three_folds], test_guard_review6.py::test_m38_one_fold_not_scored_record_for_each_fold_not_scored[two_folds]   [520 s]
M39    fails 1: test_guard_review6.py::test_m39_rows_with_a_cell_that_cannot_be_hashed_are_counted_as_repeats   [517 s]
M40    fails 0   NOT CAUGHT: no test fails on this variant   [513 s]
M41    fails 0   NOT CAUGHT: no test fails on this variant   [513 s]
M42    fails 0   NOT CAUGHT: no test fails on this variant   [517 s]
M43    fails 0   NOT CAUGHT: no test fails on this variant   [516 s]
M44    fails 0   NOT CAUGHT: no test fails on this variant   [513 s]
M45    fails 0   NOT CAUGHT: no test fails on this variant   [512 s]
M46    fails 0   NOT CAUGHT: no test fails on this variant   [515 s]
M47    fails 0   NOT CAUGHT: no test fails on this variant   [506 s]
M48    fails 0   NOT CAUGHT: no test fails on this variant   [517 s]
M49    fails 0   NOT CAUGHT: no test fails on this variant   [514 s]
M50    fails 0   NOT CAUGHT: no test fails on this variant   [513 s]
M51    fails 0   NOT CAUGHT: no test fails on this variant   [491 s]
README tests (not counted: they judge README.md, the same file in every case): 0 of 14 failing
MUTATION CHECK: FAIL
exit 1
Wed Nov 25 04:46:05 GMTST 2026
```

### 8.2 The loosening (D59)

Seven cases in five files (`test_guard_review5.py`, `test_metrics.py`,
`test_panel_stop.py`, `test_time_pipeline.py`, `test_whole_numbers.py`) and
`test_judge_tools.py`; the plan has the exact form of each. The 57 cases of
the five files pass in their loosened form on the application, on the
reference and on the reference as it stood. No broken copy loses a failing
case by it, and four gain (F6, F7, F7c, M23).
`tests\test_skill_undefined.py` is not changed: the order reading of D76 is
not taken of a column whose skill has no value (D36), and the reference was
corrected to that when this test failed on it.

### 8.3 The prototypes, red and green today

Every prototype case on the application as built by the seventh pass, on the
reference and on the reference as it stood:

```text
test_guard_review5.py: 3 cases; application: 0 red, 3 green; staged reference: 0 red; reference as it was: 0 red of 3
   GREEN test_m23_the_panel_warning_starts_just_above_one_half[just_above]
   GREEN test_m23_the_panel_warning_starts_just_above_one_half[just_below]
   GREEN test_m30_a_copy_at_step_8a_still_needs_the_same_target
test_metrics.py: 15 cases; application: 0 red, 15 green; staged reference: 0 red; reference as it was: 0 red of 15
   GREEN test_f4_constant_prediction_on_rare_class_has_no_skill
   GREEN test_f8_previous_value_beats_the_mean_on_a_random_walk
   GREEN test_f8_previous_value_refuses_repeated_keys
   GREEN test_mse_and_skill_by_hand
   GREEN test_one_class_raises
   GREEN test_previous_value_on_shuffled_keys[datetime-0]
   GREEN test_previous_value_on_shuffled_keys[datetime-2]
   GREEN test_previous_value_on_shuffled_keys[float-0]
   GREEN test_previous_value_on_shuffled_keys[float-2]
   GREEN test_roc_auc_by_hand[y0-score0-0.75]
   GREEN test_roc_auc_by_hand[y1-score1-0.6666666666666666]
   GREEN test_roc_auc_by_hand[y2-score2-0.5]
   GREEN test_roc_auc_by_hand[y3-score3-1.0]
   GREEN test_roc_auc_by_hand[y4-score4-0.0]
   GREEN test_roc_auc_by_hand[y5-score5-0.7]
test_panel_stop.py: 5 cases; application: 0 red, 5 green; staged reference: 0 red; reference as it was: 0 red of 5
   GREEN test_a_classification_panel_is_not_asked
   GREEN test_a_panel_that_does_not_persist_runs_without_a_word
   GREEN test_a_persistent_panel_with_no_entity_column_stops
   GREEN test_the_command_line_says_there_are_no_entities
   GREEN test_the_statement_lets_the_run_go_ahead_as_before
test_time_pipeline.py: 15 cases; application: 0 red, 15 green; staged reference: 0 red; reference as it was: 0 red of 15
   GREEN test_date_column_undeclared
   GREEN test_f7_day_first_time_column_stops
   GREEN test_f7_no_fit_at_or_after_holdout_or_inside_gap
   GREEN test_f7_run_uses_split_time_of_usable_keys
   GREEN test_f7_shuffled_table_gives_same_holdout_rows
   GREEN test_f8_previous_value_baseline_on_a_random_walk
   GREEN test_f8_previous_value_respects_the_gap
   GREEN test_fast_cap_keeps_most_recent_timestamps
   GREEN test_fast_cap_keeps_whole_timestamps_in_a_panel
   GREEN test_fixture_csv_with_iso_time_column
   GREEN test_numeric_and_iso_time_forms_reported
   GREEN test_panel_uses_training_mean_and_flags_persistence[False]
   GREEN test_panel_uses_training_mean_and_flags_persistence[True]
   GREEN test_scheme_follows_time_col
   GREEN test_wide_panel_under_fast_raises
test_whole_numbers.py: 19 cases; application: 0 red, 19 green; staged reference: 0 red; reference as it was: 0 red of 19
   GREEN test_command_line_on_a_26_digit_column
   GREEN test_distinct_values_are_counted_exactly
   GREEN test_missing_and_repeated_integer_stamps
   GREEN test_nanosecond_stamps_are_distinct_timestamps[Int64]
   GREEN test_nanosecond_stamps_are_distinct_timestamps[UInt64]
   GREEN test_nanosecond_stamps_are_distinct_timestamps[int64]
   GREEN test_nanosecond_stamps_are_distinct_timestamps[uint64]
   GREEN test_read_time_keeps_integers_above_2_53[Int64]
   GREEN test_read_time_keeps_integers_above_2_53[UInt64]
   GREEN test_read_time_keeps_integers_above_2_53[int64]
   GREEN test_read_time_keeps_integers_above_2_53[uint64]
   GREEN test_the_boundary_is_a_magnitude_above_2_53
   GREEN test_whole_numbers_beyond_64_bits_in_every_role[both_signs_beyond_int64-frame]
   GREEN test_whole_numbers_beyond_64_bits_in_every_role[consecutive_26_digits-csv]
   GREEN test_whole_numbers_beyond_64_bits_in_every_role[consecutive_26_digits-frame]
   GREEN test_whole_numbers_beyond_64_bits_in_every_role[one_at_2_64-csv]
   GREEN test_whole_numbers_beyond_64_bits_in_every_role[one_at_2_64-frame]
   GREEN test_whole_numbers_beyond_64_bits_in_every_role[one_below_int64-csv]
   GREEN test_whole_numbers_beyond_64_bits_in_every_role[one_below_int64-frame]
test_id_dense.py: 23 cases; application: 11 red, 12 green; staged reference: 0 red
   RED test_a_shuffled_identifier_reaches_no_model
   RED test_the_id_like_rule_for_whole_numbers[286_distinct_of_300]
   RED test_the_id_like_rule_for_whole_numbers[file_order_one_row_repeated]
   RED test_the_id_like_rule_for_whole_numbers[fills_half_its_range]
   RED test_the_id_like_rule_for_whole_numbers[fills_just_over_half_its_range]
   RED test_the_id_like_rule_for_whole_numbers[no_order_beyond_2_to_the_53]
   RED test_the_id_like_rule_for_whole_numbers[no_order_empty_cells]
   RED test_the_id_like_rule_for_whole_numbers[no_order_one_row_repeated]
   RED test_the_id_like_rule_for_whole_numbers[no_order_some_rows_gone]
   RED test_the_id_like_rule_for_whole_numbers[no_order_some_rows_gone_as_floats]
   RED test_the_rule_reads_the_rows_that_have_a_target[no_order]
   GREEN test_the_cases_are_what_their_names_say
   GREEN test_the_id_like_rule_for_whole_numbers[285_distinct_of_300]
   GREEN test_the_id_like_rule_for_whole_numbers[a_measure_in_no_order]
   GREEN test_the_id_like_rule_for_whole_numbers[consecutive_in_no_order]
   GREEN test_the_id_like_rule_for_whole_numbers[falling_with_gaps]
   GREEN test_the_id_like_rule_for_whole_numbers[few_values]
   GREEN test_the_id_like_rule_for_whole_numbers[fills_just_under_half_its_range]
   GREEN test_the_id_like_rule_for_whole_numbers[fractions]
   GREEN test_the_id_like_rule_for_whole_numbers[in_file_order]
   GREEN test_the_id_like_rule_for_whole_numbers[rising_with_gaps]
   GREEN test_the_rule_reads_the_rows_that_have_a_target[file_order]
   GREEN test_the_time_column_and_the_entity_column_keep_their_roles
test_repeated_stamps.py: 15 cases; application: 13 red, 2 green; staged reference: 0 red
   RED test_both_baselines_are_scored_at_repeated_timestamps[gap_2]
   RED test_both_baselines_are_scored_at_repeated_timestamps[rows_shuffled]
   RED test_both_baselines_are_scored_at_repeated_timestamps[three_a_timestamp]
   RED test_both_baselines_are_scored_at_repeated_timestamps[uneven]
   RED test_one_repeated_row_leaves_the_previous_value_in_use
   RED test_the_command_line_reports_both_baselines
   RED test_the_leak_screen_measures_a_lag_against_the_timestamp_before
   RED test_the_panel_warning_names_the_baseline_the_run_used[late_noise-training_mean]
   RED test_the_panel_warning_names_the_baseline_the_run_used[persistent-previous_value]
   RED test_the_step_function_gives_the_mean_of_the_timestamp_before[0]
   RED test_the_step_function_gives_the_mean_of_the_timestamp_before[1]
   RED test_the_step_function_gives_the_mean_of_the_timestamp_before[2]
   RED test_the_training_mean_is_the_headline_baseline_where_its_error_is_lower
   GREEN test_the_step_function_is_unchanged_at_unique_keys
   GREEN test_unique_timestamps_are_scored_as_before
test_screen_line.py: 46 cases; application: 28 red, 18 green; staged reference: 0 red
   RED test_a_copy_of_the_class_is_excluded_when_a_fold_holds_few_of_one_class[number]
   RED test_a_copy_of_the_class_is_excluded_when_a_fold_holds_few_of_one_class[number_reversed]
   RED test_a_copy_of_the_class_is_excluded_when_a_fold_holds_few_of_one_class[text]
   RED test_a_copy_of_the_target_is_excluded_whatever_its_scale[grouped-scale_1e-9]
   RED test_a_copy_of_the_target_is_excluded_whatever_its_scale[random-scale_1e-9]
   RED test_a_copy_of_the_target_is_excluded_whatever_its_scale[time-exact]
   RED test_a_copy_of_the_target_is_excluded_whatever_its_scale[time-negative]
   RED test_a_copy_of_the_target_is_excluded_whatever_its_scale[time-rescaled]
   RED test_a_copy_of_the_target_is_excluded_whatever_its_scale[time-scale_1e-9]
   RED test_a_copy_of_the_target_is_excluded_whatever_its_scale[time_entities-exact]
   RED test_a_copy_of_the_target_is_excluded_whatever_its_scale[time_entities-negative]
   RED test_a_copy_of_the_target_is_excluded_whatever_its_scale[time_entities-rescaled]
   RED test_a_copy_of_the_target_is_excluded_whatever_its_scale[time_entities-scale_1e-9]
   RED test_a_copy_of_the_target_is_excluded_whatever_its_scale[time_repeated-exact]
   RED test_a_copy_of_the_target_is_excluded_whatever_its_scale[time_repeated-negative]
   RED test_a_copy_of_the_target_is_excluded_whatever_its_scale[time_repeated-rescaled]
   RED test_a_copy_of_the_target_is_excluded_whatever_its_scale[time_repeated-scale_1e-9]
   RED test_a_copy_with_empty_cells_is_excluded[time]
   RED test_a_copy_with_empty_cells_is_excluded[time_entities]
   RED test_a_copy_with_empty_cells_is_excluded[time_repeated]
   RED test_one_cell_far_off_in_the_copy_does_not_keep_it[grouped--999.0]
   RED test_one_cell_far_off_in_the_copy_does_not_keep_it[random--999.0]
   RED test_one_cell_far_off_in_the_copy_does_not_keep_it[time--999.0]
   RED test_one_cell_far_off_in_the_copy_does_not_keep_it[time-999999.0]
   RED test_one_cell_far_off_in_the_copy_does_not_keep_it[time_entities--999.0]
   RED test_one_cell_far_off_in_the_copy_does_not_keep_it[time_entities-999999.0]
   RED test_one_cell_far_off_in_the_copy_does_not_keep_it[time_repeated--999.0]
   RED test_one_cell_far_off_in_the_copy_does_not_keep_it[time_repeated-999999.0]
   GREEN test_a_column_that_drives_the_class_is_not_excluded[random]
   GREEN test_a_column_that_drives_the_class_is_not_excluded[time]
   GREEN test_a_copy_of_the_target_is_excluded_whatever_its_scale[grouped-exact]
   GREEN test_a_copy_of_the_target_is_excluded_whatever_its_scale[grouped-negative]
   GREEN test_a_copy_of_the_target_is_excluded_whatever_its_scale[grouped-rescaled]
   GREEN test_a_copy_of_the_target_is_excluded_whatever_its_scale[random-exact]
   GREEN test_a_copy_of_the_target_is_excluded_whatever_its_scale[random-negative]
   GREEN test_a_copy_of_the_target_is_excluded_whatever_its_scale[random-rescaled]
   GREEN test_a_copy_with_empty_cells_is_excluded[grouped]
   GREEN test_a_copy_with_empty_cells_is_excluded[random]
   GREEN test_a_driver_known_before_the_outcome_is_kept_under_time_order
   GREEN test_columns_that_do_not_follow_the_target_stay_features[grouped]
   GREEN test_columns_that_do_not_follow_the_target_stay_features[random]
   GREEN test_columns_that_do_not_follow_the_target_stay_features[time]
   GREEN test_columns_that_do_not_follow_the_target_stay_features[time_entities]
   GREEN test_columns_that_do_not_follow_the_target_stay_features[time_repeated]
   GREEN test_one_cell_far_off_in_the_copy_does_not_keep_it[grouped-999999.0]
   GREEN test_one_cell_far_off_in_the_copy_does_not_keep_it[random-999999.0]
test_screen_set_aside.py: 18 cases; application: 12 red, 6 green; staged reference: 0 red
   RED test_a_long_tail_of_the_target_is_scored_as_it_is
   RED test_at_most_one_percent_of_the_training_rows_is_set_aside[120-1-True]
   RED test_at_most_one_percent_of_the_training_rows_is_set_aside[120-2-False]
   RED test_at_most_one_percent_of_the_training_rows_is_set_aside[600-4-True]
   RED test_at_most_one_percent_of_the_training_rows_is_set_aside[600-5-False]
   RED test_one_wild_target_cell_does_not_hide_a_copy[grouped]
   RED test_one_wild_target_cell_does_not_hide_a_copy[random]
   RED test_one_wild_target_cell_does_not_hide_a_copy[time]
   RED test_one_wild_target_cell_does_not_hide_a_copy[time_entities]
   RED test_one_wild_target_cell_does_not_hide_a_copy[time_repeated]
   RED test_under_time_order_the_row_after_a_wild_cell_counts_too[2-True]
   RED test_under_time_order_the_row_after_a_wild_cell_counts_too[3-False]
   GREEN test_a_wild_target_cell_in_the_holdout_changes_nothing_in_the_screen
   GREEN test_one_wild_target_cell_makes_no_column_a_leak[grouped]
   GREEN test_one_wild_target_cell_makes_no_column_a_leak[random]
   GREEN test_one_wild_target_cell_makes_no_column_a_leak[time]
   GREEN test_one_wild_target_cell_makes_no_column_a_leak[time_entities]
   GREEN test_one_wild_target_cell_makes_no_column_a_leak[time_repeated]
test_order_reading.py: 23 cases; application: 7 red, 16 green; staged reference: 0 red
   RED test_a_copy_with_some_cells_moved_is_named_under_time_order[8501]
   RED test_a_copy_with_some_cells_moved_is_named_under_time_order[8502]
   RED test_a_copy_with_some_cells_moved_is_named_under_time_order[8503]
   RED test_a_curve_of_the_target_is_named_under_time_order[cube]
   RED test_a_curve_of_the_target_is_named_under_time_order[exponential]
   RED test_the_first_condition_is_a_share_of_at_most_one_tenth
   RED test_with_no_previous_value_the_second_condition_has_nothing_to_ask
   GREEN test_a_categorical_column_has_no_order_reading
   GREEN test_a_classification_has_no_order_reading[1]
   GREEN test_a_classification_has_no_order_reading[2]
   GREEN test_a_column_warned_for_its_skill_gets_one_record
   GREEN test_a_column_whose_skill_cannot_be_computed_is_named_by_neither_reading
   GREEN test_a_strong_driver_known_before_the_outcome_is_not_named[100-0.95]
   GREEN test_a_strong_driver_known_before_the_outcome_is_not_named[100-0.99]
   GREEN test_a_strong_driver_known_before_the_outcome_is_not_named[100-0.9]
   GREEN test_a_strong_driver_known_before_the_outcome_is_not_named[200-0.95]
   GREEN test_a_strong_driver_known_before_the_outcome_is_not_named[200-0.99]
   GREEN test_a_strong_driver_known_before_the_outcome_is_not_named[200-0.9]
   GREEN test_a_strong_driver_known_before_the_outcome_is_not_named[400-0.95]
   GREEN test_a_strong_driver_known_before_the_outcome_is_not_named[400-0.99]
   GREEN test_a_strong_driver_known_before_the_outcome_is_not_named[400-0.9]
   GREEN test_fewer_than_twenty_pairs_give_no_reading
   GREEN test_the_second_condition_keeps_the_lag_of_a_running_total_quiet
test_largest_row.py: 13 cases; application: 12 red, 1 green; staged reference: 0 red
   RED test_a_configuration_that_failed_has_no_share
   RED test_a_sum_of_zero_has_no_largest_row
   RED test_a_wild_holdout_target_is_a_record_for_the_model_and_one_for_the_baseline
   RED test_a_wild_training_target_is_one_record_for_the_configurations_and_one_for_the_baseline
   RED test_every_configuration_has_its_share_over_the_selection_folds
   RED test_exactly_one_half_is_not_above_one_half
   RED test_one_holdout_row_the_model_misses_is_one_record
   RED test_the_holdout_shares_are_those_of_the_predictions_file
   RED test_the_keys_are_in_every_part[classification]
   RED test_the_keys_are_in_every_part[regression]
   RED test_the_statements_leave_nothing_out_of_the_score
   RED test_under_time_order_the_baseline_is_the_one_the_headline_used
   GREEN test_no_warning_without_a_row_that_dominates
test_extreme_value.py: 33 cases; application: 15 red, 18 green; staged reference: 0 red
   RED test_a_value_counts_once_however_many_rows_hold_it
   RED test_columns_that_reach_no_model_are_not_looked_at
   RED test_one_cell_far_outside_a_feature_is_named[a--999.0]
   RED test_one_cell_far_outside_a_feature_is_named[a-999999.0]
   RED test_one_cell_far_outside_a_feature_is_named[b--999.0]
   RED test_one_cell_far_outside_a_feature_is_named[b-999999.0]
   RED test_only_rows_that_are_split_are_looked_at
   RED test_the_command_line_writes_the_warning
   RED test_the_gap_must_be_more_than_thirty_times_the_spread[311.0-True]
   RED test_the_target_of_a_regression_is_looked_at
   RED test_the_three_outermost_values_at_each_end_are_looked_at[both_ends]
   RED test_the_three_outermost_values_at_each_end_are_looked_at[three_beyond]
   RED test_the_three_outermost_values_at_each_end_are_looked_at[three_values]
   RED test_the_three_outermost_values_at_each_end_are_looked_at[two_beyond]
   RED test_the_warning_changes_no_number
   GREEN test_a_clean_table_has_no_such_warning_under_any_scheme
   GREEN test_a_long_tail_is_not_named[count-2000]
   GREEN test_a_long_tail_is_not_named[count-200]
   GREEN test_a_long_tail_is_not_named[exponential-2000]
   GREEN test_a_long_tail_is_not_named[exponential-200]
   GREEN test_a_long_tail_is_not_named[log_normal_1-2000]
   GREEN test_a_long_tail_is_not_named[log_normal_1-200]
   GREEN test_a_long_tail_is_not_named[log_normal_1.5-2000]
   GREEN test_a_long_tail_is_not_named[log_normal_1.5-200]
   GREEN test_a_long_tail_is_not_named[normal-2000]
   GREEN test_a_long_tail_is_not_named[normal-200]
   GREEN test_a_long_tail_is_not_named[volume-2000]
   GREEN test_a_long_tail_is_not_named[volume-200]
   GREEN test_an_infinite_cell_is_no_extreme_value
   GREEN test_the_gap_must_be_more_than_thirty_times_the_spread[310.0-False]
   GREEN test_the_three_outermost_values_at_each_end_are_looked_at[four_beyond]
   GREEN test_the_three_outermost_values_at_each_end_are_looked_at[three_values_near]
   GREEN test_the_three_outermost_values_at_each_end_are_looked_at[two_values]
test_fit_order.py: 6 cases; application: 5 red, 1 green; staged reference: 0 red
   RED test_every_fit_is_handed_its_rows_in_time_order[entities]
   RED test_every_fit_is_handed_its_rows_in_time_order[repeated_timestamps]
   RED test_every_fit_is_handed_its_rows_in_time_order[unique_timestamps]
   RED test_rows_of_one_timestamp_keep_their_file_order
   RED test_the_order_of_the_file_changes_no_number_under_time_order
   GREEN test_with_no_time_column_the_run_is_as_it_was
test_no_choice.py: 12 cases; application: 9 red, 3 green; staged reference: 0 red
   RED test_a_configuration_with_no_fold_scores_is_not_the_first
   RED test_one_entity_of_nearly_all_training_rows_and_no_choice_is_made[0]
   RED test_one_entity_of_nearly_all_training_rows_and_no_choice_is_made[1]
   RED test_one_entity_of_nearly_all_training_rows_and_no_choice_is_made[2]
   RED test_rows_that_agree_in_every_feature_are_units_of_the_rule_too
   RED test_the_dominant_shape_is_reported_at_about_zero[0]
   RED test_the_dominant_shape_is_reported_at_about_zero[1]
   RED test_the_dominant_shape_is_reported_at_about_zero[2]
   RED test_the_first_configuration_that_fits_is_used
   GREEN test_a_small_fold_that_trains_on_as_many_rows_as_it_validates_on_is_chosen_on
   GREEN test_an_ordinary_table_of_entities_is_chosen_on_as_before
   GREEN test_the_rule_is_for_the_grouped_scheme_only
test_guard_review7.py: 9 cases; application: 0 red, 9 green; staged reference: 0 red
   GREEN test_f22_the_interval_is_the_same_at_any_number_of_cores[bootstrap_blocks]
   GREEN test_f22_the_interval_is_the_same_at_any_number_of_cores[bootstrap_groups]
   GREEN test_f22_the_interval_is_the_same_at_any_number_of_cores[bootstrap_rows]
   GREEN test_m33_three_rows_alike_after_a_removed_leak_stop_a_regression
   GREEN test_m34_the_command_line_gives_the_full_profiles_own_error[fast]
   GREEN test_m34_the_command_line_gives_the_full_profiles_own_error[full]
   GREEN test_m37_the_file_says_what_group_measures
   GREEN test_m39_rows_with_a_dictionary_cell_are_counted_as_repeats[dictionary]
   GREEN test_m39_rows_with_a_dictionary_cell_are_counted_as_repeats[tuple_with_a_list]
test_readme_pass8.py: 3 cases; application: 3 red, 0 green; staged reference: 3 red
   RED test_the_readme_names_what_the_eighth_pass_added
   RED test_the_readme_says_what_the_baseline_is_at_repeated_timestamps
   RED test_the_readme_states_the_limits_of_the_eighth_pass
   RED ON THE STAGED REFERENCE test_the_readme_names_what_the_eighth_pass_added
   RED ON THE STAGED REFERENCE test_the_readme_says_what_the_baseline_is_at_repeated_timestamps
   RED ON THE STAGED REFERENCE test_the_readme_states_the_limits_of_the_eighth_pass
```

### 8.4 The mutation check with the prototypes laid over the suite (proof 5)

The same command with `--overlay <the prototypes>`. The good reference fails
nothing; each of the 88 variants fails at least one test.

```text
Wed Nov 25 02:24:33 GMTST 2026
116caf510038874f
982ae03cea46b2bf298549e411e8ab6ec77d70bdc4d3036871fd703a322a982a *cp9_helpers.py
613d2aa7c9ef234faeabd97aeec75ff08615e150894013823f4fa8c7d6559206 *test_extreme_value.py
37998610f03065a5f61969c1136aa7b0659ee35a80bf33607584f8df7ad7ccda *test_fit_order.py
5127be9b6573dc8ba395d452a9d65937486119165f853865e1925a8f3e15079e *test_guard_review5.py
a453c1f67de7dde806d4b9838ed3237dfa6ba553f92fd037cd3b6803c581be8c *test_guard_review7.py
22ac607800c138afd97f205de05303a556dde104e0014bf20bccb403e5f98151 *test_id_dense.py
4ffb57365f42ef6f23bc2906dac510764ab1a270bda8a075db721630d48196ec *test_judge_tools.py
c71c4b24bfcdca65d57c46a286df05acc49c2331a591ec3dea67eefb0b4c648a *test_judge_tools_pass8.py
76d9298284a0f638de21aa374cc0907ce8a7d18abe45259ed4afa8696cd89389 *test_largest_row.py
62c19333a2e5dd869049928ea6572b8fc5961bdf2141a30fab21bd4cfbea6acd *test_metrics.py
f4735ad193879c067e0a921b916bf329d7cad996a1ad86b0c507e3efc1cb43c0 *test_no_choice.py
baccb4c5344b2f3655834d923455325cc8d9eeb77cc23ac5df0cb3211f84e19a *test_order_reading.py
e95eb3747bd6b8f13f7b5210e6e813bc969a99cd77cfb77b32f4656b9c9eae0b *test_panel_stop.py
45efdd367b2b54c6611ce2f98d60a052572d0fb05d7e81734e4c05ffefcb96ec *test_readme_pass8.py
4969513873a230ccc862f66677388b1488a1591d2a5881ae12e8ff4e1725deb5 *test_repeated_stamps.py
415c5c887052867c40f557719737e835f2622fb906d5e411d8ad89637273e831 *test_screen_line.py
550a901be1dbc3d9ec6367ff046790bc602179b1f7c5535de12d94f009bdf7fd *test_screen_set_aside.py
9d98ddaacd19e0689d8777975dd71c07f855438b9a1133e8623746c5ebc02bcc *test_time_pipeline.py
b854f34f46c8245f4bdd8e048acf38a5504543953da4e8dde26452461dda789d *test_whole_numbers.py
mutation check: 89 case(s), 8 at a time, projects under <temp>\mlprof-mutation-c73zwkb0
  note: docs/DECISIONS.md: copied from the repo
  note: examples/customers.csv: copied from the repo
  note: overlay: tests/cp9_helpers.py added from <scratch>/tests_proto
  note: overlay: tests/test_extreme_value.py added from <scratch>/tests_proto
  note: overlay: tests/test_fit_order.py added from <scratch>/tests_proto
  note: overlay: tests/test_guard_review5.py replaced by <scratch>/tests_proto
  note: overlay: tests/test_guard_review7.py added from <scratch>/tests_proto
  note: overlay: tests/test_id_dense.py added from <scratch>/tests_proto
  note: overlay: tests/test_judge_tools.py replaced by <scratch>/tests_proto
  note: overlay: tests/test_judge_tools_pass8.py added from <scratch>/tests_proto
  note: overlay: tests/test_largest_row.py added from <scratch>/tests_proto
  note: overlay: tests/test_metrics.py replaced by <scratch>/tests_proto
  note: overlay: tests/test_no_choice.py added from <scratch>/tests_proto
  note: overlay: tests/test_order_reading.py added from <scratch>/tests_proto
  note: overlay: tests/test_panel_stop.py replaced by <scratch>/tests_proto
  note: overlay: tests/test_readme_pass8.py added from <scratch>/tests_proto
  note: overlay: tests/test_repeated_stamps.py added from <scratch>/tests_proto
  note: overlay: tests/test_screen_line.py added from <scratch>/tests_proto
  note: overlay: tests/test_screen_set_aside.py added from <scratch>/tests_proto
  note: overlay: tests/test_time_pipeline.py replaced by <scratch>/tests_proto
  note: overlay: tests/test_whole_numbers.py replaced by <scratch>/tests_proto
  note: tests/test_judge_tools.py and tests/test_judge_tools_pass8.py are not run here: they run the judge's other tools on the application
good   fails 0 of 2119   [1192 s]
F1     fails 34: test_chosen_failure.py::test_every_configuration_failing_after_selection_raises_all_models_failed, test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration, test_chosen_failure.py::test_holdout_rows_reach_one_configuration_at_a_time, test_duplicates.py::test_every_row_twice_stops_the_run[classification] (and 29 more)   [1135 s]
F2     fails 79: test_duplicates_group.py::test_copies_that_appear_when_the_leak_screen_excludes_a_column_stop_the_run, test_extreme_value.py::test_columns_that_reach_no_model_are_not_looked_at, test_group_after_leak.py::test_a_copy_of_the_target_under_group_stops_the_run, test_group_after_leak.py::test_no_combination_of_the_features_is_on_both_sides_in_a_finished_run[noise_with_status], test_group_after_leak.py::test_no_combination_of_the_features_is_on_both_sides_in_a_finished_run[twice_with_status] (and 74 more)   [1163 s]
F3     fails 28: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[holdout], test_chosen_failure.py::test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration, test_edges.py::test_negative_skill_reported_as_measured, test_full_profile.py::test_f3_headline_is_chosen_configuration_on_the_holdout[classification] (and 23 more)   [1162 s]
F4     fails 26: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[holdout], test_chosen_failure.py::test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration, test_edges.py::test_negative_skill_reported_as_measured, test_full_profile.py::test_f3_headline_is_chosen_configuration_on_the_holdout[classification] (and 21 more)   [1190 s]
F5     fails 67: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[holdout], test_chosen_failure.py::test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration, test_duplicates.py::test_values_are_compared_as_a_step_receives_them, test_edges.py::test_edge_raises_its_named_error[holdout_one_class] (and 62 more)   [1115 s]
F6     fails 160: test_date_numbers.py::test_a_date_number_column_is_a_time_column_like_any_number[nullable_int], test_date_sentinels.py::test_the_time_column_is_judged_over_every_value_whatever_its_row, test_duplicates.py::test_values_are_compared_as_a_step_receives_them, test_edges.py::test_dataframe_outcomes_under_time_order, test_extreme_value.py::test_only_rows_that_are_split_are_looked_at (and 155 more)   [1194 s]
F7     fails 283: test_console_summary.py::test_summary_says_why_there_is_no_interval, test_console_summary.py::test_summary_under_time_order_says_why_there_is_no_p_value, test_date_numbers.py::test_a_date_number_column_is_a_time_column_like_any_number[int64], test_date_numbers.py::test_a_date_number_column_is_a_time_column_like_any_number[nullable_int], test_date_numbers.py::test_eight_digit_text_is_accepted_as_the_time_column (and 278 more)   [1173 s]
F8     fails 27: test_groups_time.py::test_each_entity_has_its_own_previous_value[0], test_groups_time.py::test_each_entity_has_its_own_previous_value[2], test_groups_time.py::test_the_command_line_takes_both_columns, test_guard_review3.py::test_m9_the_leak_screen_uses_the_previous_value, test_largest_row.py::test_a_sum_of_zero_has_no_largest_row (and 22 more)   [1576 s]
F9     fails 17: test_date_numbers.py::test_the_copies_stop_names_a_date_number_column, test_date_sentinels.py::test_it_is_named_when_nothing_is_said_and_by_the_copies_stop, test_duplicates.py::test_command_line_takes_duplicates, test_duplicates.py::test_every_row_twice_stops_the_run[classification], test_duplicates.py::test_every_row_twice_stops_the_run[regression] (and 12 more)   [1507 s]
F6b    fails 50: test_columns.py::test_f6_every_column_once_in_file_order, test_columns.py::test_roles_judged_on_usable_rows, test_columns.py::test_rule_table_one_column_per_rule[const_num], test_columns.py::test_rule_table_one_column_per_rule[const_text], test_columns.py::test_separator_rule_takes_precedence (and 45 more)   [1622 s]
F6c    fails 28: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[holdout], test_chosen_failure.py::test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration, test_chosen_failure.py::test_holdout_rows_reach_one_configuration_at_a_time, test_full_profile.py::test_eighteen_configurations_exactly[classification] (and 23 more)   [1677 s]
F6d    fails 6: test_csv_edges.py::test_a_line_pandas_reads_as_the_header_is_not_skipped[form_feed], test_csv_edges.py::test_a_line_pandas_reads_as_the_header_is_not_skipped[two_quotes], test_guard_header_lines.py::test_a_line_pandas_reads_as_the_header_before_a_two_column_header_is_an_input_error[form_feed], test_guard_header_lines.py::test_a_line_pandas_reads_as_the_header_before_a_two_column_header_is_an_input_error[no_break_space], test_guard_header_lines.py::test_a_line_pandas_reads_as_the_header_before_a_two_column_header_is_an_input_error[two_quotes_round_a_space] (and 1 more)   [1639 s]
F7b    fails 34: test_edges.py::test_edge_raises_its_named_error[time_column_invalid], test_split_time.py::test_f7_read_time_refuses_other_forms[day_first], test_split_time.py::test_f7_read_time_refuses_other_forms[month_first], test_split_time.py::test_f7_read_time_refuses_other_forms[month_name], test_split_time.py::test_f7_read_time_refuses_other_forms[year_month] (and 29 more)   [1612 s]
F7c    fails 12: test_object_numbers.py::test_object_whole_numbers_are_exact[decimal], test_object_numbers.py::test_object_whole_numbers_are_exact[python_int], test_whole_numbers.py::test_missing_and_repeated_integer_stamps, test_whole_numbers.py::test_nanosecond_stamps_are_distinct_timestamps[Int64], test_whole_numbers.py::test_nanosecond_stamps_are_distinct_timestamps[UInt64] (and 7 more)   [1597 s]
F10    fails 52: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_together_whatever_their_target, test_duplicates_features.py::test_the_command_line_uses_the_new_names, test_duplicates_features.py::test_the_count_and_the_warning_have_their_new_names, test_duplicates_features.py::test_the_count_is_of_rows_that_agree_in_every_feature (and 47 more)   [1701 s]
F7d    fails 7: test_date_numbers.py::test_it_is_named_when_nothing_is_said_about_time_order, test_date_sentinels.py::test_it_is_named_when_nothing_is_said_and_by_the_copies_stop, test_time_order_stated.py::test_a_date_column_and_no_word_about_time_stops_the_run[datetime], test_time_order_stated.py::test_a_date_column_and_no_word_about_time_stops_the_run[iso_text], test_time_order_stated.py::test_the_command_line_states_it (and 2 more)   [1516 s]
F8b    fails 6: test_groups_time.py::test_an_entity_with_no_earlier_row_gets_the_training_mean, test_groups_time.py::test_each_entity_has_its_own_previous_value[0], test_groups_time.py::test_each_entity_has_its_own_previous_value[2], test_groups_time.py::test_the_command_line_takes_both_columns, test_guard_review6.py::test_m28_eight_digit_text_orders_a_panel_and_read_time_reads_it (and 1 more)   [1518 s]
F11    fails 26: test_date_numbers.py::test_it_is_named_when_nothing_is_said_about_time_order, test_date_sentinels.py::test_it_is_named_when_nothing_is_said_and_by_the_copies_stop, test_panel_stop.py::test_a_persistent_panel_with_no_entity_column_stops, test_panel_stop.py::test_the_command_line_says_there_are_no_entities, test_statements.py::test_none_passed_explicitly_is_not_said (and 21 more)   [1069 s]
F12    fails 7: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_together_whatever_their_target, test_duplicates_features.py::test_the_count_is_of_rows_that_agree_in_every_feature, test_fast_cap_groups.py::test_it_is_for_an_entity_column_with_no_time_column_only, test_group_after_leak.py::test_no_combination_of_the_features_is_on_both_sides_in_a_finished_run[few_valued] (and 2 more)   [1539 s]
F13    fails 20: test_grouped_holdout.py::test_a_large_entity_is_never_held_out_when_smaller_ones_can_fill_a_quarter[100], test_grouped_holdout.py::test_a_large_entity_is_never_held_out_when_smaller_ones_can_fill_a_quarter[120], test_grouped_holdout.py::test_a_large_entity_is_never_held_out_when_smaller_ones_can_fill_a_quarter[240], test_grouped_holdout.py::test_a_large_unit_kept_in_training_is_named, test_grouped_holdout.py::test_a_unit_too_large_for_a_sound_split_is_the_holdout_and_is_named (and 15 more)   [1540 s]
F14    fails 47: test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[decimal], test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[float_with_gaps], test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[int64], test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[nullable_int], test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[python_ints] (and 42 more)   [1533 s]
F12b   fails 5: test_group_after_leak.py::test_a_copy_of_the_target_under_group_stops_the_run, test_group_after_leak.py::test_one_combination_on_both_sides_is_enough, test_group_after_leak.py::test_the_command_line_stops_too, test_group_after_leak.py::test_the_stop_names_a_date_column_that_was_excluded_too, test_guard_review7.py::test_m33_three_rows_alike_after_a_removed_leak_stop_a_regression   [1552 s]
F14b   fails 37: test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[29_feb_1900], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[29_feb_2023], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[30_percent_of_the_rows_99991231], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[31_april], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[5_percent_of_the_rows_99991231] (and 32 more)   [1554 s]
F15    fails 7: test_fast_cap_groups.py::test_an_entity_larger_than_the_cap_stops_the_fast_profile, test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept, test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run, test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[0], test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[1] (and 2 more)   [1076 s]
F16    fails 13: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[holdout], test_chosen_failure.py::test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration, test_chosen_failure.py::test_holdout_rows_reach_one_configuration_at_a_time, test_edges.py::test_negative_skill_reported_as_measured (and 8 more)   [882 s]
F17    fails 15: test_full_profile.py::test_f3_headline_is_chosen_configuration_on_the_holdout[regression], test_guard_review6.py::test_m6_an_infinite_holdout_prediction_in_a_regression_falls_back, test_largest_row.py::test_a_sum_of_zero_has_no_largest_row, test_largest_row.py::test_a_wild_holdout_target_is_a_record_for_the_model_and_one_for_the_baseline, test_largest_row.py::test_a_wild_training_target_is_one_record_for_the_configurations_and_one_for_the_baseline (and 10 more)   [838 s]
F18    fails 17: test_console_summary.py::test_summary_shows_both_intervals_the_p_value_and_each_warning, test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[decimal], test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[float_with_gaps], test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[int64], test_date_numbers.py::test_a_column_of_dates_written_as_whole_numbers_is_excluded_as_a_date[nullable_int] (and 12 more)   [787 s]
F19    fails 30: test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[final_fit], test_chosen_failure.py::test_failure_after_the_choice_keeps_fold_scores_and_falls_back[holdout], test_chosen_failure.py::test_fallback_follows_cv_mean_and_the_headline_comes_from_one_configuration, test_edges.py::test_negative_skill_reported_as_measured, test_full_profile.py::test_f3_headline_is_chosen_configuration_on_the_holdout[classification] (and 25 more)   [791 s]
F20    fails 4: test_edges.py::test_limitations_and_caveat_are_the_fixed_texts[random-classification], test_edges.py::test_limitations_and_caveat_are_the_fixed_texts[random-regression], test_edges.py::test_limitations_and_caveat_are_the_fixed_texts[time-classification], test_edges.py::test_limitations_and_caveat_are_the_fixed_texts[time-regression]   [791 s]
F21    fails 15: test_duplicates.py::test_the_rule_needs_a_share_and_a_count[0_of_1000], test_duplicates.py::test_the_rule_needs_a_share_and_a_count[10_of_1000], test_duplicates.py::test_the_rule_needs_a_share_and_a_count[9_of_100], test_duplicates_group.py::test_under_time_order_it_changes_nothing, test_full_profile.py::test_command_line_defaults_to_full (and 10 more)   [750 s]
F22    fails 4: test_full_profile.py::test_same_result_at_one_and_two_jobs, test_guard_review7.py::test_f22_the_interval_is_the_same_at_any_number_of_cores[bootstrap_blocks], test_guard_review7.py::test_f22_the_interval_is_the_same_at_any_number_of_cores[bootstrap_groups], test_guard_review7.py::test_f22_the_interval_is_the_same_at_any_number_of_cores[bootstrap_rows]   [715 s]
F23    fails 10: test_repeated_stamps.py::test_both_baselines_are_scored_at_repeated_timestamps[gap_2], test_repeated_stamps.py::test_both_baselines_are_scored_at_repeated_timestamps[rows_shuffled], test_repeated_stamps.py::test_both_baselines_are_scored_at_repeated_timestamps[three_a_timestamp], test_repeated_stamps.py::test_both_baselines_are_scored_at_repeated_timestamps[uneven], test_repeated_stamps.py::test_one_repeated_row_leaves_the_previous_value_in_use (and 5 more)   [583 s]
F24    fails 29: test_screen_line.py::test_a_copy_of_the_target_is_excluded_whatever_its_scale[grouped-scale_1e-9], test_screen_line.py::test_a_copy_of_the_target_is_excluded_whatever_its_scale[random-scale_1e-9], test_screen_line.py::test_a_copy_of_the_target_is_excluded_whatever_its_scale[time-exact], test_screen_line.py::test_a_copy_of_the_target_is_excluded_whatever_its_scale[time-negative], test_screen_line.py::test_a_copy_of_the_target_is_excluded_whatever_its_scale[time-rescaled] (and 24 more)   [570 s]
F25    fails 14: test_extreme_value.py::test_columns_that_reach_no_model_are_not_looked_at, test_screen_line.py::test_a_copy_with_empty_cells_is_excluded[time_entities], test_screen_line.py::test_one_cell_far_off_in_the_copy_does_not_keep_it[grouped-999999.0], test_screen_line.py::test_one_cell_far_off_in_the_copy_does_not_keep_it[random--999.0], test_screen_line.py::test_one_cell_far_off_in_the_copy_does_not_keep_it[time_entities--999.0] (and 9 more)   [572 s]
F26    fails 3: test_largest_row.py::test_one_holdout_row_the_model_misses_is_one_record, test_largest_row.py::test_the_holdout_shares_are_those_of_the_predictions_file, test_largest_row.py::test_under_time_order_the_baseline_is_the_one_the_headline_used   [555 s]
F27    fails 11: test_id_dense.py::test_a_shuffled_identifier_reaches_no_model, test_id_dense.py::test_the_id_like_rule_for_whole_numbers[286_distinct_of_300], test_id_dense.py::test_the_id_like_rule_for_whole_numbers[file_order_one_row_repeated], test_id_dense.py::test_the_id_like_rule_for_whole_numbers[fills_half_its_range], test_id_dense.py::test_the_id_like_rule_for_whole_numbers[fills_just_over_half_its_range] (and 6 more)   [559 s]
F28    fails 9: test_no_choice.py::test_a_configuration_with_no_fold_scores_is_not_the_first, test_no_choice.py::test_one_entity_of_nearly_all_training_rows_and_no_choice_is_made[0], test_no_choice.py::test_one_entity_of_nearly_all_training_rows_and_no_choice_is_made[1], test_no_choice.py::test_one_entity_of_nearly_all_training_rows_and_no_choice_is_made[2], test_no_choice.py::test_rows_that_agree_in_every_feature_are_units_of_the_rule_too (and 4 more)   [553 s]
M1     fails 2: test_guard_final_fit.py::test_m1_final_fit_statistics_ignore_the_holdout, test_guard_review6.py::test_m1_the_final_fit_is_centred_on_its_own_rows   [555 s]
M2     fails 2: test_feature_names.py::test_a_level_only_in_rows_with_no_target_does_not_count, test_guard_final_fit.py::test_m2_levels_come_from_the_rows_of_each_fit   [555 s]
M3     fails 2: test_guard_leak_screen.py::test_m3_missing_by_class_counts_training_rows_only, test_guard_leak_screen.py::test_m3_single_column_skill_is_fitted_on_fold_training_rows_only   [554 s]
M4     fails 3: test_guard_uncertainty.py::test_m4_block_bootstrap_draws_whole_blocks, test_skill_undefined.py::test_no_skill_interval_when_fewer_than_half_the_resamples_have_a_baseline_error, test_skill_undefined.py::test_skill_undefined_and_no_skill_interval_are_two_codes   [551 s]
M5     fails 2: test_guard_review6.py::test_m5_a_shuffle_that_ties_counts_where_only_some_of_them_tie, test_guard_uncertainty.py::test_m5_ties_count_in_the_p_value   [554 s]
M6     fails 2: test_guard_review6.py::test_m6_an_infinite_holdout_prediction_in_a_regression_falls_back, test_non_finite.py::test_a_prediction_that_is_not_finite_counts_as_raising   [555 s]
M7     fails 4: test_fast_profile_error.py::test_fast_error_stands_when_the_full_profile_would_run[eight_dates_of_450], test_fast_profile_error.py::test_fast_error_stands_when_the_full_profile_would_run[twelve_dates_of_2500_latest_over_the_cap], test_fast_profile_error.py::test_fast_raises_the_full_profiles_error_when_full_would_stop_too[nine_dates_of_450_gap_2], test_time_pipeline.py::test_wide_panel_under_fast_raises   [553 s]
M8     fails 2: test_guard_review3.py::test_m8_the_interval_is_the_middle_95_percent, test_guard_review6.py::test_m8_the_interval_from_whole_entities_is_the_middle_95_percent   [555 s]
M9     fails 31: test_fit_order.py::test_every_fit_is_handed_its_rows_in_time_order[entities], test_groups_time.py::test_an_entity_with_no_earlier_row_gets_the_training_mean, test_groups_time.py::test_each_entity_has_its_own_previous_value[0], test_groups_time.py::test_each_entity_has_its_own_previous_value[2], test_groups_time.py::test_the_command_line_takes_both_columns (and 26 more)   [555 s]
M10    fails 2: test_guard_review3.py::test_m10_the_fast_cap_is_a_sample_from_the_seed, test_guard_review6.py::test_m10_the_fast_cap_is_a_sample_whatever_the_order_of_the_file   [548 s]
M11    fails 2: test_guard_review3.py::test_m11_cv_std_is_the_sample_deviation, test_guard_review6.py::test_m11_cv_std_of_two_scored_folds_is_their_sample_deviation   [551 s]
M12    fails 3: test_guard_review3.py::test_m12_tied_predictions_in_a_resample, test_guard_review6.py::test_m12_tied_predictions_in_a_resample_of_blocks_and_of_entities[blocks], test_guard_review6.py::test_m12_tied_predictions_in_a_resample_of_blocks_and_of_entities[entities]   [548 s]
M13    fails 1: test_guard_review3.py::test_m13_n_imputed_counts_rows_that_reach_a_model   [552 s]
M14    fails 4: test_read_once.py::test_a_text_cell_of_an_object_target_in_a_row_with_no_time_value, test_read_once.py::test_a_text_cell_of_an_object_time_column_in_a_row_with_no_target, test_read_once.py::test_a_time_value_beyond_64_bits_in_a_row_with_no_target, test_read_once.py::test_a_whole_number_beyond_64_bits_in_a_target_row_with_no_time_value   [552 s]
M15    fails 14: test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_together_whatever_their_target, test_duplicates_group.py::test_each_set_of_copies_stays_on_one_side_and_in_one_fold, test_groups_random.py::test_no_entity_on_both_sides_or_in_two_folds[classification], test_groups_random.py::test_no_entity_on_both_sides_or_in_two_folds[regression], test_no_choice.py::test_a_configuration_with_no_fold_scores_is_not_the_first (and 9 more)   [554 s]
M16    fails 8: test_fit_order.py::test_every_fit_is_handed_its_rows_in_time_order[entities], test_groups_time.py::test_an_entity_with_no_earlier_row_gets_the_training_mean, test_groups_time.py::test_each_entity_has_its_own_previous_value[0], test_groups_time.py::test_each_entity_has_its_own_previous_value[2], test_groups_time.py::test_the_command_line_takes_both_columns (and 3 more)   [560 s]
M17    fails 7: test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept, test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run, test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[0], test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[1], test_fast_cap_groups.py::test_the_cap_keeps_whole_entities_every_large_one_and_no_more_than_it_can (and 2 more)   [552 s]
M18    fails 6: test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite[columns], test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite[input], test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite[meta], test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite[rows], test_review4_small.py::test_no_part_of_the_output_can_hold_a_number_that_is_not_finite[task] (and 1 more)   [553 s]
M19    fails 7: test_panel_stop.py::test_a_persistent_panel_with_no_entity_column_stops, test_panel_stop.py::test_the_command_line_says_there_are_no_entities, test_statements.py::test_nothing_said_about_entities_stops_whatever_the_table_holds[a_panel_that_does_not_persist], test_statements.py::test_nothing_said_about_entities_stops_whatever_the_table_holds[classification], test_statements.py::test_nothing_said_about_entities_stops_whatever_the_table_holds[one_series] (and 2 more)   [532 s]
M20    fails 3: test_guard_review4.py::test_an_infinite_float_in_an_object_target_is_a_missing_target, test_guard_review4.py::test_an_object_target_of_numbers_with_an_infinite_one, test_guard_review4.py::test_m20_infinite_cells_of_the_target_and_the_time_column_are_counted   [555 s]
M21    fails 2: test_groups_random.py::test_the_interval_resamples_whole_entities_and_there_is_no_p_value, test_guard_review6.py::test_m21_a_resample_of_whole_entities_has_one_score   [551 s]
M22    fails 19: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_rows_that_agree_in_every_feature_stay_together_whatever_their_target, test_duplicates_features.py::test_the_command_line_uses_the_new_names, test_duplicates_features.py::test_the_count_and_the_warning_have_their_new_names, test_duplicates_features.py::test_the_count_is_of_rows_that_agree_in_every_feature (and 14 more)   [563 s]
M23    fails 2: test_guard_review5.py::test_m23_the_panel_warning_starts_just_above_one_half[just_above], test_guard_review6.py::test_m23_a_panel_between_the_two_cut_offs_is_warned_about_on_the_command_line   [556 s]
M24    fails 2: test_statements.py::test_one_message_asks_for_everything_that_is_missing, test_statements.py::test_the_command_line_asks_for_each_missing_pair   [544 s]
M25    fails 3: test_duplicates_features.py::test_a_set_that_spans_two_entities_joins_them, test_duplicates_features.py::test_the_count_is_of_rows_that_agree_in_every_feature, test_grouped_holdout.py::test_the_rule_holds_for_rows_that_agree_in_every_feature   [562 s]
M26    fails 5: test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run, test_grouped_holdout.py::test_a_unit_too_large_for_a_sound_split_is_the_holdout_and_is_named, test_grouped_holdout.py::test_one_warning_for_each_large_unit_and_a_share_of_its_own_class, test_grouped_holdout.py::test_the_holdout_holds_a_quarter_and_needs_every_unit_it_has[a_handful_of_large], test_grouped_holdout.py::test_what_the_holdout_can_spare_goes_back   [556 s]
M27    fails 6: test_grouped_holdout.py::test_a_large_unit_kept_in_training_is_named, test_grouped_holdout.py::test_one_warning_for_each_large_unit_and_a_share_of_its_own_class, test_grouped_holdout.py::test_the_rule_holds_for_rows_that_agree_in_every_feature, test_no_choice.py::test_the_dominant_shape_is_reported_at_about_zero[0], test_no_choice.py::test_the_dominant_shape_is_reported_at_about_zero[1] (and 1 more)   [557 s]
M28    fails 4: test_date_numbers.py::test_eight_digit_text_is_accepted_as_the_time_column, test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[text], test_date_sentinels.py::test_read_time_refuses_it_too[text], test_guard_review6.py::test_m28_eight_digit_text_orders_a_panel_and_read_time_reads_it   [555 s]
M29    fails 4: test_date_numbers.py::test_the_date_rule_comes_before_the_id_like_rule[int64], test_date_numbers.py::test_the_date_rule_comes_before_the_id_like_rule[text], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[an_infinite_cell], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[floats_with_gaps_and_an_infinite_cell]   [555 s]
M30    fails 2: test_duplicates.py::test_the_rule_in_regression, test_guard_review5.py::test_m30_a_copy_at_step_8a_still_needs_the_same_target   [554 s]
M31    fails 19: test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[30_percent_of_the_rows_99991231], test_date_sentinels.py::test_a_column_of_dates_with_a_value_that_is_no_date_is_still_a_date_column[60_percent_of_the_rows_99991231], test_date_sentinels.py::test_a_sentinel_counts_once_however_many_cells_hold_it, test_date_sentinels.py::test_a_time_column_that_is_no_date_column_is_a_number_as_before, test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[float] (and 14 more)   [567 s]
M32    fails 8: test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[float], test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[int64], test_date_sentinels.py::test_as_the_time_column_a_value_that_is_no_date_stops_the_run[text], test_date_sentinels.py::test_read_time_refuses_it_too[float], test_date_sentinels.py::test_read_time_refuses_it_too[int64] (and 3 more)   [554 s]
M33    fails 2: test_group_after_leak.py::test_one_combination_on_both_sides_is_enough, test_guard_review7.py::test_m33_three_rows_alike_after_a_removed_leak_stop_a_regression   [555 s]
M34    fails 2: test_fast_cap_groups.py::test_when_the_full_profile_would_stop_too_its_error_is_raised, test_guard_review7.py::test_m34_the_command_line_gives_the_full_profiles_own_error[fast]   [553 s]
M35    fails 5: test_fast_cap_groups.py::test_an_entity_of_a_quarter_or_more_that_fits_is_always_kept, test_fast_cap_groups.py::test_entities_of_a_quarter_or_more_that_do_not_fit_together_stop_the_run, test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[0], test_fast_cap_groups.py::test_the_cap_itself_is_the_boundary[1], test_fast_cap_groups.py::test_the_cap_keeps_whole_entities_every_large_one_and_no_more_than_it_can   [556 s]
M36    fails 2: test_review6_small.py::test_the_date_warning_holds_the_caveat_of_its_scheme[grouped_by_an_entity_column], test_review6_small.py::test_the_date_warning_holds_the_caveat_of_its_scheme[grouped_by_rows_that_agree]   [544 s]
M37    fails 2: test_group_after_leak.py::test_the_grouped_warning_says_what_group_measures, test_guard_review7.py::test_m37_the_file_says_what_group_measures   [540 s]
M38    fails 2: test_guard_review6.py::test_m38_one_fold_not_scored_record_for_each_fold_not_scored[three_folds], test_guard_review6.py::test_m38_one_fold_not_scored_record_for_each_fold_not_scored[two_folds]   [535 s]
M39    fails 3: test_guard_review6.py::test_m39_rows_with_a_cell_that_cannot_be_hashed_are_counted_as_repeats, test_guard_review7.py::test_m39_rows_with_a_dictionary_cell_are_counted_as_repeats[dictionary], test_guard_review7.py::test_m39_rows_with_a_dictionary_cell_are_counted_as_repeats[tuple_with_a_list]   [540 s]
M40    fails 7: test_repeated_stamps.py::test_both_baselines_are_scored_at_repeated_timestamps[gap_2], test_repeated_stamps.py::test_both_baselines_are_scored_at_repeated_timestamps[rows_shuffled], test_repeated_stamps.py::test_both_baselines_are_scored_at_repeated_timestamps[three_a_timestamp], test_repeated_stamps.py::test_both_baselines_are_scored_at_repeated_timestamps[uneven], test_repeated_stamps.py::test_the_panel_warning_names_the_baseline_the_run_used[late_noise-training_mean] (and 2 more)   [534 s]
M41    fails 6: test_screen_line.py::test_one_cell_far_off_in_the_copy_does_not_keep_it[time--999.0], test_screen_line.py::test_one_cell_far_off_in_the_copy_does_not_keep_it[time-999999.0], test_screen_line.py::test_one_cell_far_off_in_the_copy_does_not_keep_it[time_entities--999.0], test_screen_line.py::test_one_cell_far_off_in_the_copy_does_not_keep_it[time_entities-999999.0], test_screen_line.py::test_one_cell_far_off_in_the_copy_does_not_keep_it[time_repeated--999.0] (and 1 more)   [540 s]
M42    fails 8: test_extreme_value.py::test_the_target_of_a_regression_is_looked_at, test_largest_row.py::test_a_wild_training_target_is_one_record_for_the_configurations_and_one_for_the_baseline, test_screen_set_aside.py::test_at_most_one_percent_of_the_training_rows_is_set_aside[120-1-True], test_screen_set_aside.py::test_at_most_one_percent_of_the_training_rows_is_set_aside[600-4-True], test_screen_set_aside.py::test_one_wild_target_cell_does_not_hide_a_copy[grouped] (and 3 more)   [532 s]
M43    fails 1: test_order_reading.py::test_the_second_condition_keeps_the_lag_of_a_running_total_quiet   [513 s]
M44    fails 11: test_extreme_value.py::test_the_three_outermost_values_at_each_end_are_looked_at[both_ends], test_extreme_value.py::test_the_three_outermost_values_at_each_end_are_looked_at[three_beyond], test_extreme_value.py::test_the_three_outermost_values_at_each_end_are_looked_at[two_beyond], test_order_reading.py::test_a_copy_with_some_cells_moved_is_named_under_time_order[8501], test_order_reading.py::test_a_copy_with_some_cells_moved_is_named_under_time_order[8502] (and 6 more)   [513 s]
M45    fails 4: test_largest_row.py::test_a_sum_of_zero_has_no_largest_row, test_largest_row.py::test_a_wild_training_target_is_one_record_for_the_configurations_and_one_for_the_baseline, test_largest_row.py::test_every_configuration_has_its_share_over_the_selection_folds, test_largest_row.py::test_the_keys_are_in_every_part[regression]   [521 s]
M46    fails 4: test_extreme_value.py::test_a_value_counts_once_however_many_rows_hold_it, test_extreme_value.py::test_one_cell_far_outside_a_feature_is_named[a--999.0], test_extreme_value.py::test_the_command_line_writes_the_warning, test_extreme_value.py::test_the_three_outermost_values_at_each_end_are_looked_at[both_ends]   [511 s]
M47    fails 3: test_screen_line.py::test_a_copy_of_the_class_is_excluded_when_a_fold_holds_few_of_one_class[number], test_screen_line.py::test_a_copy_of_the_class_is_excluded_when_a_fold_holds_few_of_one_class[number_reversed], test_screen_line.py::test_a_copy_of_the_class_is_excluded_when_a_fold_holds_few_of_one_class[text]   [498 s]
M48    fails 1: test_no_choice.py::test_a_small_fold_that_trains_on_as_many_rows_as_it_validates_on_is_chosen_on   [509 s]
M49    fails 5: test_fit_order.py::test_every_fit_is_handed_its_rows_in_time_order[entities], test_fit_order.py::test_every_fit_is_handed_its_rows_in_time_order[repeated_timestamps], test_fit_order.py::test_every_fit_is_handed_its_rows_in_time_order[unique_timestamps], test_fit_order.py::test_rows_of_one_timestamp_keep_their_file_order, test_fit_order.py::test_the_order_of_the_file_changes_no_number_under_time_order   [507 s]
M50    fails 3: test_id_dense.py::test_the_id_like_rule_for_whole_numbers[286_distinct_of_300], test_id_dense.py::test_the_id_like_rule_for_whole_numbers[file_order_one_row_repeated], test_id_dense.py::test_the_id_like_rule_for_whole_numbers[no_order_one_row_repeated]   [506 s]
M51    fails 2: test_repeated_stamps.py::test_the_command_line_reports_both_baselines, test_repeated_stamps.py::test_the_panel_warning_names_the_baseline_the_run_used[persistent-previous_value]   [405 s]
README tests (not counted: they judge README.md, the same file in every case): 3 of 17 failing: test_readme_pass8.py::test_the_readme_names_what_the_eighth_pass_added, test_readme_pass8.py::test_the_readme_says_what_the_baseline_is_at_repeated_timestamps, test_readme_pass8.py::test_the_readme_states_the_limits_of_the_eighth_pass
MUTATION CHECK: PASS (the good reference fails nothing; each of 88 variant(s) fails at least one test)
exit 0
Wed Nov 25 04:57:36 GMTST 2026
```

The test that fails on each new variant, and how many cases of the new files
fail on it:

| Variant | Test that fails on it | Cases |
|---|---|---|
| F23 | `test_repeated_stamps.py::test_both_baselines_are_scored_at_repeated_timestamps` | 10 |
| F24 | `test_screen_line.py::test_a_copy_of_the_target_is_excluded_whatever_its_scale` | 29 |
| F25 | `test_screen_set_aside.py::test_one_wild_target_cell_does_not_hide_a_copy` | 14 |
| F26 | `test_largest_row.py::test_the_holdout_shares_are_those_of_the_predictions_file` | 3 |
| F27 | `test_id_dense.py::test_the_id_like_rule_for_whole_numbers` | 11 |
| F28 | `test_no_choice.py::test_one_entity_of_nearly_all_training_rows_and_no_choice_is_made` | 9 |
| M40 | `test_repeated_stamps.py::test_both_baselines_are_scored_at_repeated_timestamps` | 7 |
| M41 | `test_screen_line.py::test_one_cell_far_off_in_the_copy_does_not_keep_it` | 6 |
| M42 | `test_screen_set_aside.py::test_one_wild_target_cell_makes_no_column_a_leak` | 8 |
| M43 | `test_order_reading.py::test_the_second_condition_keeps_the_lag_of_a_running_total_quiet` | 1 |
| M44 | `test_order_reading.py::test_a_copy_with_some_cells_moved_is_named_under_time_order` | 11 |
| M45 | `test_largest_row.py::test_every_configuration_has_its_share_over_the_selection_folds` | 4 |
| M46 | `test_extreme_value.py::test_one_cell_far_outside_a_feature_is_named` | 4 |
| M47 | `test_screen_line.py::test_a_copy_of_the_class_is_excluded_when_a_fold_holds_few_of_one_class` | 3 |
| M48 | `test_no_choice.py::test_a_small_fold_that_trains_on_as_many_rows_as_it_validates_on_is_chosen_on` | 1 |
| M49 | `test_fit_order.py::test_every_fit_is_handed_its_rows_in_time_order` | 5 |
| M50 | `test_id_dense.py::test_the_id_like_rule_for_whole_numbers` | 3 |
| M51 | `test_repeated_stamps.py::test_the_panel_warning_names_the_baseline_the_run_used` | 2 |

The five faults that had one test and that the two tools do not catch each
have a second in `test_guard_review7.py`: F22
(`test_f22_the_interval_is_the_same_at_any_number_of_cores`), M33
(`test_m33_three_rows_alike_after_a_removed_leak_stop_a_regression`), M34
(`test_m34_the_command_line_gives_the_full_profiles_own_error`), M37
(`test_m37_the_file_says_what_group_measures`) and M39
(`test_m39_rows_with_a_dictionary_cell_are_counted_as_repeats`). M13's second
is the tools' (sections 9 and 10).

Failing cases for each broken copy the suite caught before: before the
loosening / after it, in the files that existed / with the new files.

```text
F1 29/29/34    F2 18/18/79    F3 24/24/28    F4 14/14/26    F5 58/58/67
F6 148/149/160 F7 212/218/283 F8 16/16/27    F9 17/17/17    F6b 50/50/50
F6c 19/19/28   F6d 6/6/6      F7b 34/34/34   F7c 11/12/12   F10 40/40/52
F7d 7/7/7      F8b 6/6/6      F11 26/26/26   F12 6/6/7      F13 11/11/20
F14 47/47/47   F12b 4/4/5     F14b 37/37/37  F15 7/7/7      F16 13/13/13
F17 4/4/15     F18 17/17/17   F19 27/27/30   F20 4/4/4      F21 12/12/15
F22 1/1/4
M1 2/2/2    M2 2/2/2    M3 2/2/2    M4 3/3/3    M5 2/2/2    M6 2/2/2
M7 4/4/4    M8 2/2/2    M9 14/14/31 M10 2/2/2   M11 2/2/2   M12 3/3/3
M13 1/1/1   M14 4/4/4   M15 4/4/14  M16 7/7/8   M17 7/7/7   M18 6/6/6
M19 7/7/7   M20 3/3/3   M21 2/2/2   M22 17/17/19 M23 1/2/2  M24 2/2/2
M25 3/3/3   M26 5/5/5   M27 3/3/6   M28 4/4/4   M29 4/4/4   M30 2/2/2
M31 19/19/19 M32 8/8/8  M33 1/1/2   M34 1/1/2   M35 5/5/5   M36 2/2/2
M37 1/1/2   M38 2/2/2   M39 1/1/3
```

### 8.5 The two judge-tool tests on the application

The loosened `test_judge_tools.py` (each tool run once, exit 1 accepted only
when every failing pair or table is on the list of `owned_by_pass8.json`) and
the strict `test_judge_tools_pass8.py`, in one session against the
application as built by the seventh pass: the loosened one passes, the strict
one fails, as each must until the pass is built.

```text
pytest exit 1; 475 s for the session
PASSES test_judge_tools.py::test_the_perturbation_battery_holds (359 s)
PASSES test_judge_tools.py::test_the_generated_tables_agree_with_the_reference (113 s)
FAILS  test_judge_tools_pass8.py::test_the_perturbation_battery_exits_0 (0 s)
      E       AssertionError: verify/battery.py exited 1; its last lines:
      E              the change: a column `target_copy` is added: the target rescaled, with noise of 5% of the target's spread
      E              must hold: the added column `target_copy` is excluded as a suspected leak, or at least named by a `strong_single_column` warning: a rescaled copy of the target with noise of 5% of the target's spread added. A stated limit (D76 point 2): above 1% such a column cannot be told from a str
      E              u_largest_row: `headline` has no ['largest_row_share', 'largest_row', 'baseline_largest_row_share', 'baseline_largest_row']
      E              copy_told:target_copy: the added column `target_copy` is a feature (role 'feature', reason None) and no `strong_single_column` warning names it; skill 0.4896780229461545 against 'previous_value', model linear-0
      E              seen: rc=0, error=None, skill=0.4897, scheme=time_ordered, baseline=previous_value, model=linear-0, holdout=375, copies=None, cap=0, share=None, base_share=None, folds=None, roles={'target_copy': 'feature/None'}
      E              table: <scratch>\tp\proj\tools\verify\artifacts\battery\data\random_walk__copy_noise_5pct.csv
      E              output: <scratch>\tp\proj\tools\verify\artifacts\battery\runs\build\random_walk__copy_noise_5pct
      E              again: python verify/battery.py mlprofessor --only random_walk:copy_noise_5pct
      E         STATED LIMIT (D55): int_stamps:missing_cell_date [structure]
      E              the change: one cell of the time column, or of a date column, is emptied
      E              must hold: the run's own conclusion (its band, its roles, its counts): a row with no time value is dropped and counted in `dropped_missing_time` (step 2)
      E              u_largest_row: `headline` has no ['largest_row_share', 'largest_row', 'baseline_largest_row_share', 'baseline_largest_row']
      E              time_split: the holdout is 377 rows; the judge expects the 375 latest (2 unexpected, 0 missing)
FAILS  test_judge_tools_pass8.py::test_the_generated_tables_exit_0 (0 s)
          [property] application: headline holds the largest row's share, for the model and for the baseline (5.2; D74 point 3): "absent: ['largest_row_share', 'largest_row', 'baseline_largest_row_share', 'baseline_largest_row']" | ''
      E       AssertionError: verify/generated_tables.py exited 1; its last lines:
      E           [property] application: headline holds the largest row's share, for the model and for the baseline (5.2; D74 point 3): "absent: ['largest_row_share', 'largest_row', 'baseline_largest_row_share', 'baseline_largest_row']" | ''
      E           again: python verify/generated_tables.py <scratch>\tp\proj\tools\mlprofessor --seed 20261122 --only 996 --keep
      E         
      E         table 997: csv, 225 rows x 7 columns ['f2', 'count', 'const', 'f1', 'when', 'y', 'odd']; task='auto', time_col=False, group_col=False, gap=0, profile='fast', seed=7, duplicates='group'; small; statement none; target clf_bool; some timestamps hold several rows; odd: a column of list; when: 
      E           application: result: random, skill -0.19969040247678016, model tree-0
      E           reference:   result: random, skill -0.29721362229102166, model linear-0
      E           [property] application: headline holds the largest row's share, for the model and for the baseline (5.2; D74 point 3): "absent: ['largest_row_share', 'largest_row', 'baseline_largest_row_share', 'baseline_largest_row']" | ''
      E           again: python verify/generated_tables.py <scratch>\tp\proj\tools\mlprofessor --seed 20261122 --only 997 --keep
      E         
      E         table 998: dataframe, 705 rows x 6 columns ['signup', 'when', 'y', 'f2', 'who', 'f1']; task='regression', time_col='when', group_col='who', gap=0, profile='fast', seed=1, duplicates='error'; large; statement time_entity; target reg; panel of 25 entities on 38 timestamps, irregular; when: i
      E           application: result: time_ordered, skill 0.8424286203875389, model linear-0
      E           reference:   result: time_ordered, skill 0.8424286203875389, model linear-0
```

## 9. The perturbation battery (D71 to D76)

The prototype of the tool on the prototype of the reference, which is what the
owner's second go rested on (D76): every pair holds or is one of the four
stated limits.

```text
STATED LIMIT (D31): messy:sorted_by_measure [band]
     the change: the table is sorted by the whole-number measure `visits`
     must hold: the run's own conclusion (its band, its roles, its counts): the table is sorted by the whole-number measure `visits`: it stays a feature (D19's own control; today the ordered clause calls it `id_like`, the limit D31 accepted)
     messy_roles: `visits` has role 'excluded', reason 'id_like'; expected 'feature', None
     seen: rc=0, error=None, skill=0.7122, scheme=random, baseline=chance, model=linear-0, holdout=375, copies=0, cap=0, share=None, base_share=None, folds=None, roles={'record_id': 'excluded/id_like'}
     table: <scratch>\bat\work9\data\messy__sorted_by_measure.csv
     output: <scratch>\bat\work9\runs\reference\messy__sorted_by_measure
     again: python verify/battery.py verify_final/reference/mlprofessor --only messy:sorted_by_measure
STATED LIMIT (D60): dup_noise:noise_added [stop-or-band]
     the change: a column `noise_added` of normal noise is added
     must hold: the stop `holdout_copies_in_training`, or no skill (the band of `noise_clf`, borrowed): every row is still present twice and the target is noise. By spec step 8a the rows are no longer copies and the run goes ahead: a copy that differs in one feature is not detected (D60)
     band:noise_clf: skill is 0.6742, expected between -0.27 and 0.27 (the band of `noise_clf`)
     seen: rc=0, error=None, skill=0.6742, scheme=random, baseline=chance, model=tree-0, holdout=375, copies=0, cap=0, share=None, base_share=None, folds=None, roles={'noise_added': 'feature/None'}
     table: <scratch>\bat\work9\data\dup_noise__noise_added.csv
     output: <scratch>\bat\work9\runs\reference\dup_noise__noise_added
     again: python verify/battery.py verify_final/reference/mlprofessor --only dup_noise:noise_added
STATED LIMIT (D76): random_walk:copy_noise_5pct [structure]
     the change: a column `target_copy` is added: the target rescaled, with noise of 5% of the target's spread
     must hold: the added column `target_copy` is excluded as a suspected leak, or at least named by a `strong_single_column` warning: a rescaled copy of the target with noise of 5% of the target's spread added: as large as the previous value's own miss. A stated limit (D76): such a column cannot be told from a strong feature measured at the same moment
     copy_told:target_copy: the added column `target_copy` is a feature (role 'feature', reason None) and no `strong_single_column` warning names it; skill 0.4896780229461525 against 'previous_value', model linear-0
     seen: rc=0, error=None, skill=0.4897, scheme=time_ordered, baseline=previous_value, model=linear-0, holdout=375, copies=None, cap=0, share=0.0286, base_share=0.0417, folds=None, roles={'target_copy': 'feature/None'}
     table: <scratch>\bat\work9\data\random_walk__copy_noise_5pct.csv
     output: <scratch>\bat\work9\runs\reference\random_walk__copy_noise_5pct
     again: python verify/battery.py verify_final/reference/mlprofessor --only random_walk:copy_noise_5pct
STATED LIMIT (D55): int_stamps:missing_cell_date [structure]
     the change: one cell of the time column, or of a date column, is emptied
     must hold: the run's own conclusion (its band, its roles, its counts): a row with no time value is dropped and counted in `dropped_missing_time` (step 2)
     u_largest_row: `headline.baseline_largest_row_share` is 0.02504022710989764; the judge computes 0.03031797829538831 for the headline baseline (row 760)
     time_split: the holdout is 377 rows; the judge expects the 375 latest (2 unexpected, 0 missing)
     walk_baselines: the training_mean baseline scores 209.50897449007985; the judge computes 208.75221214027528
     int_keys: training and holdout timestamps, unique, form: (439, 148, False, 'numeric'); the judge counts (1124, 375, True, 'numeric')
     seen: rc=0, error=None, skill=0.5384, scheme=time_ordered, baseline=previous_value, model=linear-0, holdout=377, copies=None, cap=0, share=0.0318, base_share=0.025, folds=None, roles={}
     table: <scratch>\bat\work9\data\int_stamps__missing_cell_date.csv
     output: <scratch>\bat\work9\runs\reference\int_stamps__missing_cell_date
     again: python verify/battery.py verify_final/reference/mlprofessor --only int_stamps:missing_cell_date
STATED LIMIT (D76): int_stamps:copy_noise_5pct [structure]
     the change: a column `target_copy` is added: the target rescaled, with noise of 5% of the target's spread
     must hold: the added column `target_copy` is excluded as a suspected leak, or at least named by a `strong_single_column` warning: a rescaled copy of the target with noise of 5% of the target's spread added: as large as the previous value's own miss. A stated limit (D76): such a column cannot be told from a strong feature measured at the same moment
     copy_told:target_copy: the added column `target_copy` is a feature (role 'feature', reason None) and no `strong_single_column` warning names it; skill 0.8018094664408543 against 'previous_value', model linear-0
     seen: rc=0, error=None, skill=0.8018, scheme=time_ordered, baseline=previous_value, model=linear-0, holdout=375, copies=None, cap=0, share=0.0331, base_share=0.0304, folds=None, roles={'target_copy': 'feature/None'}
     table: <scratch>\bat\work9\data\int_stamps__copy_noise_5pct.csv
     output: <scratch>\bat\work9\runs\reference\int_stamps__copy_noise_5pct
     again: python verify/battery.py verify_final/reference/mlprofessor --only int_stamps:copy_noise_5pct
STATED LIMIT (D76): panel:copy_noise_5pct [structure]
     the change: a column `target_copy` is added: the target rescaled, with noise of 5% of the target's spread
     must hold: the added column `target_copy` is excluded as a suspected leak, or at least named by a `strong_single_column` warning: a rescaled copy of the target with noise of 5% of the target's spread added: as large as the previous value's own miss. A stated limit (D76): such a column cannot be told from a strong feature measured at the same moment
     copy_told:target_copy: the added column `target_copy` is a feature (role 'feature', reason None) and no `strong_single_column` warning names it; skill 0.4276190101071946 against 'previous_value', model linear-0
     seen: rc=0, error=None, skill=0.4276, scheme=time_ordered, baseline=previous_value, model=linear-0, holdout=384, copies=None, cap=0, share=0.0309, base_share=0.0261, folds=None, roles={'target_copy': 'feature/None'}
     table: <scratch>\bat\work9\data\panel__copy_noise_5pct.csv
     output: <scratch>\bat\work9\runs\reference\panel__copy_noise_5pct
     again: python verify/battery.py verify_final/reference/mlprofessor --only panel:copy_noise_5pct
1016 tool runs in 412 s at 16 side by side (6386 s of tool time): 32 gate runs as they are and 984 pairs; 1256 pairs left out (--list says why)
holds 1010; stated limit 6 of 6 listed; open 0 of 0 listed (D73, pass 8); on neither list and failing 0
BATTERY: PASS (holds 1010, stated limit 6, open 0)
```

The tool as it is now, on the reference. The stated-limit list holds exactly
four kinds, six pairs; nothing is open.

```text
STATED LIMIT (D31): messy:sorted_by_measure [band]
     the change: the table is sorted by the whole-number measure `visits`
     must hold: the run's own conclusion (its band, its roles, its counts): the table is sorted by the whole-number measure `visits`: a measure is a feature whatever the order of the rows. A stated limit (D31): every value is distinct and in the order of the file, so the first clause of the id-like rule (D19) excludes it
     messy_roles: `visits` has role 'excluded', reason 'id_like'; expected 'feature', None
     seen: rc=0, error=None, skill=0.7122, scheme=random, baseline=chance, model=linear-0, holdout=375, copies=0, cap=0, share=None, base_share=None, folds=None, roles={'record_id': 'excluded/id_like'}
     table: <scratch>\out\tools_stage\final\work_reference\data\messy__sorted_by_measure.csv
     output: <scratch>\out\tools_stage\final\work_reference\runs\reference\messy__sorted_by_measure
     again: python verify/battery.py verify/reference/mlprofessor --only messy:sorted_by_measure
STATED LIMIT (D60): dup_noise:noise_added [stop-or-band]
     the change: a column `noise_added` of normal noise is added
     must hold: the stop `holdout_copies_in_training`, or no skill (the band of `noise_clf`, borrowed): every row is still present twice and the target is noise. By spec step 8a the rows are no longer copies and the run goes ahead: a copy that differs in one feature is not detected (D60)
     band:noise_clf: skill is 0.6742, expected between -0.27 and 0.27 (the band of `noise_clf`)
     seen: rc=0, error=None, skill=0.6742, scheme=random, baseline=chance, model=tree-0, holdout=375, copies=0, cap=0, share=None, base_share=None, folds=None, roles={'noise_added': 'feature/None'}
     table: <scratch>\out\tools_stage\final\work_reference\data\dup_noise__noise_added.csv
     output: <scratch>\out\tools_stage\final\work_reference\runs\reference\dup_noise__noise_added
     again: python verify/battery.py verify/reference/mlprofessor --only dup_noise:noise_added
STATED LIMIT (D76): random_walk:copy_noise_5pct [structure]
     the change: a column `target_copy` is added: the target rescaled, with noise of 5% of the target's spread
     must hold: the added column `target_copy` is excluded as a suspected leak, or at least named by a `strong_single_column` warning: a rescaled copy of the target with noise of 5% of the target's spread added. A stated limit (D76 point 2): above 1% such a column cannot be told from a strong feature measured at the same moment, and whether it is known before the outcome is the caller's to say
     copy_told:target_copy: the added column `target_copy` is a feature (role 'feature', reason None) and no `strong_single_column` warning names it; skill 0.4896780229461525 against 'previous_value', model linear-0
     seen: rc=0, error=None, skill=0.4897, scheme=time_ordered, baseline=previous_value, model=linear-0, holdout=375, copies=None, cap=0, share=0.0286, base_share=0.0417, folds=None, roles={'target_copy': 'feature/None'}
     table: <scratch>\out\tools_stage\final\work_reference\data\random_walk__copy_noise_5pct.csv
     output: <scratch>\out\tools_stage\final\work_reference\runs\reference\random_walk__copy_noise_5pct
     again: python verify/battery.py verify/reference/mlprofessor --only random_walk:copy_noise_5pct
STATED LIMIT (D55): int_stamps:missing_cell_date [structure]
     the change: one cell of the time column, or of a date column, is emptied
     must hold: the run's own conclusion (its band, its roles, its counts): a row with no time value is dropped and counted in `dropped_missing_time` (step 2)
     u_largest_row: `headline.baseline_largest_row_share` is 0.025040227109897645; the judge computes 0.03031797829538831 for the headline baseline (row 760)
     time_split: the holdout is 377 rows; the judge expects the 375 latest (2 unexpected, 0 missing)
     walk_baselines: the training_mean baseline scores 209.50897449007985; the judge computes 208.75221214027528
     int_keys: training and holdout timestamps, unique, form: (439, 148, False, 'numeric'); the judge counts (1124, 375, True, 'numeric')
     seen: rc=0, error=None, skill=0.5384, scheme=time_ordered, baseline=previous_value, model=linear-0, holdout=377, copies=None, cap=0, share=0.0318, base_share=0.025, folds=None, roles={}
     table: <scratch>\out\tools_stage\final\work_reference\data\int_stamps__missing_cell_date.csv
     output: <scratch>\out\tools_stage\final\work_reference\runs\reference\int_stamps__missing_cell_date
     again: python verify/battery.py verify/reference/mlprofessor --only int_stamps:missing_cell_date
STATED LIMIT (D76): int_stamps:copy_noise_5pct [structure]
     the change: a column `target_copy` is added: the target rescaled, with noise of 5% of the target's spread
     must hold: the added column `target_copy` is excluded as a suspected leak, or at least named by a `strong_single_column` warning: a rescaled copy of the target with noise of 5% of the target's spread added. A stated limit (D76 point 2): above 1% such a column cannot be told from a strong feature measured at the same moment, and whether it is known before the outcome is the caller's to say
     copy_told:target_copy: the added column `target_copy` is a feature (role 'feature', reason None) and no `strong_single_column` warning names it; skill 0.8018094664408543 against 'previous_value', model linear-0
     seen: rc=0, error=None, skill=0.8018, scheme=time_ordered, baseline=previous_value, model=linear-0, holdout=375, copies=None, cap=0, share=0.0331, base_share=0.0304, folds=None, roles={'target_copy': 'feature/None'}
     table: <scratch>\out\tools_stage\final\work_reference\data\int_stamps__copy_noise_5pct.csv
     output: <scratch>\out\tools_stage\final\work_reference\runs\reference\int_stamps__copy_noise_5pct
     again: python verify/battery.py verify/reference/mlprofessor --only int_stamps:copy_noise_5pct
STATED LIMIT (D76): panel:copy_noise_5pct [structure]
     the change: a column `target_copy` is added: the target rescaled, with noise of 5% of the target's spread
     must hold: the added column `target_copy` is excluded as a suspected leak, or at least named by a `strong_single_column` warning: a rescaled copy of the target with noise of 5% of the target's spread added. A stated limit (D76 point 2): above 1% such a column cannot be told from a strong feature measured at the same moment, and whether it is known before the outcome is the caller's to say
     copy_told:target_copy: the added column `target_copy` is a feature (role 'feature', reason None) and no `strong_single_column` warning names it; skill 0.4276190101071946 against 'previous_value', model linear-0
     seen: rc=0, error=None, skill=0.4276, scheme=time_ordered, baseline=previous_value, model=linear-0, holdout=384, copies=None, cap=0, share=0.0309, base_share=0.0261, folds=None, roles={'target_copy': 'feature/None'}
     table: <scratch>\out\tools_stage\final\work_reference\data\panel__copy_noise_5pct.csv
     output: <scratch>\out\tools_stage\final\work_reference\runs\reference\panel__copy_noise_5pct
     again: python verify/battery.py verify/reference/mlprofessor --only panel:copy_noise_5pct
1229 tool runs, and 143 second calls by the judge, in 1296 s at 16 side by side (20188 s of tool time): 37 gate runs as they are and 1192 pairs; 1398 pairs left out (--list says why)
holds 1223; stated limit 6 of 6 listed; open 0 of 0 listed; on neither list and failing 0
BATTERY: PASS (holds 1223, stated limit 6, open 0)
exit 0
```

On the application as built by the seventh pass, by the rule of this pass
that the build must change (`by_rule_battery.py <the battery's --json file>`;
the whole output is `battery\battery_application.txt`). A pair is counted
under every rule one of its failing claims has; none fits no rule. Nearly
every finished run fails for the keys of the largest row's share alone, which
the application does not write yet.

```text
package E:\CODE\ml-professor\mlprofessor
1229 runs; counts {'holds': 299, 'stated limit': 6, 'open': 0, 'fails': 924}; wall 1056.7 s at 16 side by side, 16148.7 s of tool time

924 pairs fail and are on neither list (930 of the 1229 runs finished)

pairs by rule (a pair is counted under every rule one of its failing claims has):
  A1 baseline: 73
  A2/A3b screen and order reading: 144
  A3 statements: 922
  A3 extreme value: 162
  A5 identifier: 66
  A6 no choice: 31
  none: 0

pairs by the set of rules they fail for:
    570  A3 statements
     68  A3 extreme value + A3 statements
     62  A2/A3b screen and order reading + A3 extreme value + A3 statements
     57  A2/A3b screen and order reading + A3 statements
     54  A3 statements + A5 identifier
     39  A1 baseline + A3 statements
     19  A3 statements + A6 no choice
     16  A1 baseline + A3 extreme value + A3 statements
     10  A1 baseline + A2/A3b screen and order reading + A3 statements
      6  A2/A3b screen and order reading + A3 statements + A6 no choice
      6  A1 baseline + A2/A3b screen and order reading + A3 extreme value + A3 statements
      5  A3 extreme value + A3 statements + A6 no choice
      4  A3 extreme value + A3 statements + A5 identifier
      2  A5 identifier
      2  A1 baseline + A3 statements + A5 identifier
      2  A2/A3b screen and order reading + A3 statements + A5 identifier
      1  A3 statements + A5 identifier + A6 no choice
      1  A2/A3b screen and order reading + A3 extreme value + A3 statements + A5 identifier

failing claims by rule and claim:
  A1 baseline | band: 31
  A1 baseline | flat: 6
  A1 baseline | walk_baselines: 67
  A2/A3b screen and order reading | added: 52
  A2/A3b screen and order reading | band: 36
  A2/A3b screen and order reading | copy_told: 21
  A2/A3b screen and order reading | leak_reg_roles: 3
  A2/A3b screen and order reading | roles_as_base: 3
  A2/A3b screen and order reading | same_as_base: 41
  A2/A3b screen and order reading | walk_copy_roles: 85
  A3 extreme value | band_or_named: 155
  A3 extreme value | named: 85
  A3 statements | u_largest_row: 922
  A5 identifier | added: 28
  A5 identifier | band: 4
  A5 identifier | messy_roles: 35
  A5 identifier | same_as_base: 28
  A5 identifier | stop: 2
  A6 no choice | band: 20
  A6 no choice | fold_rule: 31

pairs by rule, in full (the statements of A3 aside: every finished run lacks them):

  A1 baseline: 73 pairs
    base (1): walk_repeated
    blank_line_added (1): walk_repeated
    constant_added (1): walk_repeated
    copy_5_cells_differ (1): walk_repeated
    copy_and_target_sentinel (1): walk_repeated
    copy_distinct (1): walk_repeated
    copy_exact (1): walk_repeated
    copy_negative_scale (1): walk_repeated
    copy_one_cell_999999 (1): walk_repeated
    date_sentinel_1900 (1): walk_repeated
    feature_999999_early_row (1): walk_repeated
    feature_999999_latest_row (1): walk_repeated
    feature_renamed (1): walk_repeated
    feature_renamed_awkward (1): walk_repeated
    gap_2_row_repeated (3): random_walk, int_stamps, flat_tail
    id_added_gaps (1): walk_repeated
    id_added_int (1): walk_repeated
    id_added_text (1): walk_repeated
    missing_cell_date (1): walk_repeated
    missing_cell_feature (1): walk_repeated
    missing_cell_target (1): walk_repeated
    noise_added (1): walk_repeated
    number_sentinel_-999 (1): walk_repeated
    number_sentinel_1e308 (1): walk_repeated
    number_sentinel_999999 (1): walk_repeated
    number_two_sentinels (1): walk_repeated
    role_column_renamed (1): walk_repeated
    row_repeated (7): random_walk, int_stamps, flat_tail, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    row_repeated_other_target (7): random_walk, int_stamps, flat_tail, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    rows_repeated_10pct (7): random_walk, int_stamps, flat_tail, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    rows_repeated_1pct (7): random_walk, int_stamps, flat_tail, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    rows_shuffled (1): walk_repeated
    second_date_added (1): walk_repeated
    stamp_shared (7): random_walk, int_stamps, flat_tail, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    target_-999_early_row (1): walk_repeated
    target_-999_latest_row (1): walk_repeated
    target_first (1): walk_repeated
    target_renamed (1): walk_repeated
    target_sentinel_-999 (1): walk_repeated
    target_sentinel_999999 (1): walk_repeated
    target_two_sentinels (1): walk_repeated

  A2/A3b screen and order reading: 144 pairs
    base (2): walk_copy, walk_copy_cell
    blank_line_added (2): walk_copy, walk_copy_cell
    constant_added (2): walk_copy, walk_copy_cell
    copy_2pct_cells_altered (8): random_walk, int_stamps, flat_tail, panel, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    copy_5_cells_differ (9): random_walk, int_stamps, flat_tail, panel, dominant_entity, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    copy_and_target_sentinel (11): signal_reg, noise_reg, random_walk, int_stamps, flat_tail, panel, dominant_entity, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    copy_curve (7): random_walk, int_stamps, panel, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    copy_distinct (8): random_walk, int_stamps, panel, dominant_entity, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    copy_exact (8): random_walk, int_stamps, panel, dominant_entity, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    copy_negative_scale (8): random_walk, int_stamps, panel, dominant_entity, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    copy_noise_1pct (6): random_walk, int_stamps, panel, walk_repeated, walk_copy, walk_late_cell
    copy_one_cell_999999 (8): random_walk, int_stamps, panel, dominant_entity, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    date_sentinel_1900 (2): walk_copy, walk_copy_cell
    feature_999999_early_row (2): walk_copy, walk_copy_cell
    feature_999999_latest_row (2): walk_copy, walk_copy_cell
    feature_renamed (2): walk_copy, walk_copy_cell
    feature_renamed_awkward (2): walk_copy, walk_copy_cell
    id_added_gaps (2): walk_copy, walk_copy_cell
    id_added_int (2): walk_copy, walk_copy_cell
    id_added_text (2): walk_copy, walk_copy_cell
    missing_cell_date (2): walk_copy, walk_copy_cell
    missing_cell_feature (2): walk_copy, walk_copy_cell
    missing_cell_target (2): walk_copy, walk_copy_cell
    noise_added (2): walk_copy, walk_copy_cell
    number_sentinel_-999 (2): walk_copy, walk_copy_cell
    number_sentinel_1e308 (2): walk_copy, walk_copy_cell
    number_sentinel_999999 (2): walk_copy, walk_copy_cell
    number_two_sentinels (2): walk_copy, walk_copy_cell
    role_column_renamed (2): walk_copy, walk_copy_cell
    row_repeated (2): walk_copy, walk_copy_cell
    row_repeated_other_target (2): walk_copy, walk_copy_cell
    rows_repeated_10pct (2): walk_copy, walk_copy_cell
    rows_repeated_1pct (2): walk_copy, walk_copy_cell
    rows_shuffled (2): walk_copy, walk_copy_cell
    second_date_added (2): walk_copy, walk_copy_cell
    stamp_shared (2): walk_copy, walk_copy_cell
    target_-999_early_row (2): walk_copy, walk_copy_cell
    target_-999_latest_row (2): walk_copy, walk_copy_cell
    target_first (2): walk_copy, walk_copy_cell
    target_renamed (2): walk_copy, walk_copy_cell
    target_sentinel_-999 (3): leak_reg, walk_copy, walk_copy_cell
    target_sentinel_999999 (3): leak_reg, walk_copy, walk_copy_cell
    target_two_sentinels (3): leak_reg, walk_copy, walk_copy_cell

  A3 extreme value: 162 pairs
    base (2): walk_copy_cell, walk_late_cell
    blank_line_added (2): walk_copy_cell, walk_late_cell
    constant_added (2): walk_copy_cell, walk_late_cell
    copy_2pct_cells_altered (2): walk_copy_cell, walk_late_cell
    copy_5_cells_differ (2): walk_copy_cell, walk_late_cell
    copy_and_target_sentinel (9): signal_reg, noise_reg, leak_reg, random_walk, panel, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    copy_curve (2): walk_copy_cell, walk_late_cell
    copy_distinct (2): walk_copy_cell, walk_late_cell
    copy_exact (2): walk_copy_cell, walk_late_cell
    copy_negative_scale (2): walk_copy_cell, walk_late_cell
    copy_noise_1pct (1): walk_late_cell
    copy_one_cell_999999 (2): walk_copy_cell, walk_late_cell
    date_sentinel_1900 (2): walk_copy_cell, walk_late_cell
    feature_999999_early_row (6): random_walk, panel, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    feature_999999_latest_row (6): random_walk, panel, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    feature_renamed (2): walk_copy_cell, walk_late_cell
    feature_renamed_awkward (2): walk_copy_cell, walk_late_cell
    id_added_gaps (2): walk_copy_cell, walk_late_cell
    id_added_int (2): walk_copy_cell, walk_late_cell
    id_added_text (2): walk_copy_cell, walk_late_cell
    missing_cell_date (2): walk_copy_cell, walk_late_cell
    missing_cell_feature (6): panel, dominant_entity, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    missing_cell_target (2): walk_copy_cell, walk_late_cell
    noise_added (2): walk_copy_cell, walk_late_cell
    number_sentinel_-999 (12): signal_clf, signal_reg, leak_reg, messy, random_walk, panel, dominant_entity, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell, messy_shuffled
    number_sentinel_1e308 (2): walk_copy_cell, walk_late_cell
    number_sentinel_999999 (14): signal_clf, signal_reg, noise_reg, leak_reg, binary_numeric, messy, random_walk, panel, dominant_entity, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell, messy_shuffled
    number_two_sentinels (14): signal_clf, signal_reg, noise_reg, leak_reg, binary_numeric, messy, random_walk, panel, dominant_entity, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell, messy_shuffled
    role_column_renamed (2): walk_copy_cell, walk_late_cell
    row_repeated (2): walk_copy_cell, walk_late_cell
    row_repeated_other_target (2): walk_copy_cell, walk_late_cell
    rows_repeated_10pct (2): walk_copy_cell, walk_late_cell
    rows_repeated_1pct (2): walk_copy_cell, walk_late_cell
    rows_shuffled (2): walk_copy_cell, walk_late_cell
    second_date_added (2): walk_copy_cell, walk_late_cell
    stamp_shared (2): walk_copy_cell, walk_late_cell
    target_-999_early_row (6): random_walk, panel, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    target_-999_latest_row (3): walk_repeated, walk_copy_cell, walk_late_cell
    target_first (2): walk_copy_cell, walk_late_cell
    target_renamed (2): walk_copy_cell, walk_late_cell
    target_sentinel_-999 (8): signal_reg, leak_reg, random_walk, panel, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell
    target_sentinel_999999 (8): signal_reg, noise_reg, leak_reg, panel, dominant_entity, walk_repeated, walk_copy_cell, walk_late_cell
    target_two_sentinels (9): signal_reg, noise_reg, leak_reg, random_walk, panel, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell

  A5 identifier: 66 pairs
    base (1): messy_shuffled
    blank_line_added (1): messy_shuffled
    constant_added (1): messy_shuffled
    copy_distinct (1): messy_shuffled
    copy_two_valued (1): messy_shuffled
    date_sentinel_1900 (1): messy_shuffled
    date_sentinel_9999 (1): messy_shuffled
    date_sentinel_zero (1): messy_shuffled
    feature_renamed (1): messy_shuffled
    feature_renamed_awkward (1): messy_shuffled
    id_added_gaps (30): signal_clf, signal_reg, noise_clf, noise_reg, imbalanced, leak, leak_reg, near_leak, binary_numeric, messy, dup_noise, time_blocks, time_blocks_gap, time_blocks_undeclared, random_walk, int_stamps, flat_tail, entities, panel, few_valued, dominant_entity, day_numbers, day_sentinel, few_valued_leak, quarter_entity, walk_repeated, walk_copy, walk_copy_cell, walk_late_cell, messy_shuffled
    id_added_int (1): messy_shuffled
    id_added_text (1): messy_shuffled
    id_floats_shuffled (1): messy
    id_shuffled_and_repeated (1): messy
    missing_cell_date (1): messy_shuffled
    missing_cell_feature (1): messy_shuffled
    missing_cell_target (1): messy_shuffled
    noise_added (1): messy_shuffled
    number_sentinel_-999 (1): messy_shuffled
    number_sentinel_1e308 (1): messy_shuffled
    number_sentinel_999999 (1): messy_shuffled
    number_two_sentinels (1): messy_shuffled
    role_column_renamed (1): messy_shuffled
    row_repeated (2): messy, messy_shuffled
    rows_deleted_shuffled (1): messy
    rows_shuffled (4): messy, walk_repeated, walk_copy, messy_shuffled
    second_date_added (1): messy_shuffled
    target_first (1): messy_shuffled
    target_renamed (1): messy_shuffled
    target_respelt (1): messy_shuffled
    target_respelt_flipped (1): messy_shuffled
    white_space_cells (1): messy_shuffled

  A6 no choice: 31 pairs
    base (1): dominant_entity
    blank_line_added (1): dominant_entity
    constant_added (1): dominant_entity
    copy_5_cells_differ (1): dominant_entity
    copy_and_target_sentinel (1): dominant_entity
    copy_distinct (1): dominant_entity
    copy_exact (1): dominant_entity
    copy_negative_scale (1): dominant_entity
    copy_one_cell_999999 (1): dominant_entity
    feature_renamed (1): dominant_entity
    feature_renamed_awkward (1): dominant_entity
    id_added_gaps (1): dominant_entity
    id_added_int (1): dominant_entity
    id_added_text (1): dominant_entity
    missing_cell_feature (1): dominant_entity
    missing_cell_target (1): dominant_entity
    noise_added (1): dominant_entity
    number_sentinel_-999 (1): dominant_entity
    number_sentinel_1e308 (1): dominant_entity
    number_sentinel_999999 (1): dominant_entity
    number_two_sentinels (1): dominant_entity
    role_column_renamed (1): dominant_entity
    row_repeated (1): dominant_entity
    rows_shuffled (1): dominant_entity
    second_date_added (1): dominant_entity
    target_first (1): dominant_entity
    target_renamed (1): dominant_entity
    target_sentinel_-999 (1): dominant_entity
    target_sentinel_999999 (1): dominant_entity
    target_two_sentinels (1): dominant_entity
    white_space_cells (1): dominant_entity

  none: 0 pairs

failing claims that fit no rule, in full: 0

stated limit: 6: messy:sorted_by_measure (stated limit: D31); dup_noise:noise_added (stated limit: D60); random_walk:copy_noise_5pct (stated limit: D76); int_stamps:missing_cell_date (stated limit: D55); int_stamps:copy_noise_5pct (stated limit: D76); panel:copy_noise_5pct (stated limit: D76)

open: 0: 

listed and holds here: 0: 
```

The battery runs 32 side by side by default since this pass. Alone on a quiet
machine it takes 274 to 296 seconds so (two runs), against 716 seconds at 16
side by side; it took 226 seconds before this pass, with 766 runs. The two
runs at 32 give the same verdict on every one of the 1,229 pairs as the run
at 16, and no pair needed a second attempt in any of the three
(`same_verdicts.py`; the outputs are `battery\battery_reference.txt`, made at
16 beside other runs, `battery\battery_reference_32_side_by_side.txt` and
`battery\battery_reference_32_side_by_side_second_run.txt`):

```text
16 side by side: 16 side by side, 1296 s; 1229 pairs; {'holds': 1223, 'stated limit': 6, 'open': 0, 'fails': 0}; pairs that needed a second attempt: 0; second calls by the judge: 143
32 side by side, first run: 32 side by side, 296 s; 1229 pairs; {'holds': 1223, 'stated limit': 6, 'open': 0, 'fails': 0}; pairs that needed a second attempt: 0; second calls by the judge: 143
32 side by side, second run: 32 side by side, 274 s; 1229 pairs; {'holds': 1223, 'stated limit': 6, 'open': 0, 'fails': 0}; pairs that needed a second attempt: 0; second calls by the judge: 143
32 side by side, first run against 16 side by side: the same pairs: True; pairs whose class differs: 0; pairs whose verdict differs in any claim: 0
32 side by side, second run against 16 side by side: the same pairs: True; pairs whose class differs: 0; pairs whose verdict differs in any claim: 0
the same stated limits in every run: True
the same second calls in every run: True
```

On six broken copies of the reference that had one test each
(`variant_summary.py`): the battery catches M13 (66 pairs) and M33 (one pair,
`battery\battery_variant_M33.txt`), and not F22, M34, M37 or M39, which get
their second test from the suite (section 8).

The log of the final runs of both tools, with the checksums of the reference,
the two tools and the gate at the start and at the end:

```text
02:31:38 start; core.py battery.py generated_tables.py smoke_test.py: 116caf510038874f a76fbb5c570b7bb6 9dc7b4f3a022d573 7af964a499938a84 
02:36:34 generated reference_seed20261122: 295 s in all; GENERATED TABLES: PASS (1000 tables from seed 20261122: 1000 agree, 0 stated limit, 0 open, 0 fail)
02:41:56 generated reference_seed1: 322 s in all; GENERATED TABLES: PASS (1000 tables from seed 1: 1000 agree, 0 stated limit, 0 open, 0 fail)
02:47:08 generated reference_seed2: 312 s in all; GENERATED TABLES: PASS (1000 tables from seed 2: 1000 agree, 0 stated limit, 0 open, 0 fail)
02:49:33 battery application: 1074 s in all; BATTERY: FAIL (924 of 1229 do not hold and are on neither list; holds 299, stated limit 6, open 0)
02:52:30 generated application: 322 s in all; GENERATED TABLES: FAIL (1000 tables from seed 20261122: 415 agree, 0 stated limit, 0 open, 585 fail)
02:53:32 battery reference: 1313 s in all; BATTERY: PASS (holds 1223, stated limit 6, open 0)
02:53:33 reference and application done; 116caf510038874f a76fbb5c570b7bb6 9dc7b4f3a022d573 7af964a499938a84 
02:59:43 generated variant_F22: 370 s in all; GENERATED TABLES: PASS (1000 tables from seed 20261122: 1000 agree, 0 stated limit, 0 open, 0 fail)
03:05:23 generated variant_M13: 339 s in all; GENERATED TABLES: FAIL (1000 tables from seed 20261122: 849 agree, 0 stated limit, 0 open, 151 fail)
03:11:04 generated variant_M33: 341 s in all; GENERATED TABLES: PASS (1000 tables from seed 20261122: 1000 agree, 0 stated limit, 0 open, 0 fail)
03:17:53 battery variant_M33: 1460 s in all; BATTERY: FAIL (1 of 1229 do not hold and are on neither list; holds 1222, stated limit 6, open 0)
03:17:54 battery variant_F22: 1461 s in all; BATTERY: PASS (holds 1223, stated limit 6, open 0)
03:17:58 battery variant_M13: 1465 s in all; BATTERY: FAIL (66 of 1229 do not hold and are on neither list; holds 1157, stated limit 6, open 0)
03:23:25 generated variant_M34: 327 s in all; GENERATED TABLES: PASS (1000 tables from seed 20261122: 1000 agree, 0 stated limit, 0 open, 0 fail)
03:29:04 generated variant_M37: 339 s in all; GENERATED TABLES: PASS (1000 tables from seed 20261122: 1000 agree, 0 stated limit, 0 open, 0 fail)
03:34:45 generated variant_M39: 341 s in all; GENERATED TABLES: PASS (1000 tables from seed 20261122: 1000 agree, 0 stated limit, 0 open, 0 fail)
03:41:32 battery variant_M37: 1414 s in all; BATTERY: PASS (holds 1223, stated limit 6, open 0)
03:41:32 battery variant_M34: 1414 s in all; BATTERY: PASS (holds 1223, stated limit 6, open 0)
03:41:33 battery variant_M39: 1414 s in all; BATTERY: PASS (holds 1223, stated limit 6, open 0)
03:41:33 end; core.py battery.py generated_tables.py smoke_test.py: 116caf510038874f a76fbb5c570b7bb6 9dc7b4f3a022d573 7af964a499938a84 
```

`owned_by_pass8.json` lists every pair and every generated table that
fails on the application today, each with the rule that owns it
(`owned.py`). The loosened `tests\test_judge_tools.py` of the plan reads its
fixed list from it.

## 10. The generated tables (D56, D71 to D76)

The prototype of the tool on the prototype of the reference, three seeds:

```text
generated tables: 1000 table(s) from seed 20261122, 334 worker run(s), 24 at a time, under <scratch>\gen\snap9\verify_new\artifacts\generated\run-jmil2ctt
  package:   <scratch>\gen\snap9\proto_snapshot\mlprofessor
  reference: <scratch>\gen\snap9\verify_new\reference\mlprofessor

STATED LIMITS
  none is known

OPEN: owned, not yet built
  none is known

NOT COMPARED (no part of the verdict): differences by kind
      0  the table earns two named errors of steps 1 to 4, and which is raised is open (spec 2.4)
      0  it depends on which rows a random or a grouped split took, or the fast cap kept (spec 11)
      0  both sides raise the warning; how many records it has is not pinned (spec 2.5)
      0  which estimator of the lag-1 autocorrelation the panel warning uses is open (spec 11)

application: 585 result(s) {'grouped': 156, 'random': 190, 'time_ordered': 239}, 415 named error(s) {'column_not_found': 8, 'duplicate_columns': 14, 'entities_undeclared': 34, 'fast_profile_group_too_large': 3, 'fast_profile_too_few_rows': 2, 'fast_profile_too_few_timestamps': 5, 'feature_name_collision': 13, 'gap_too_large': 12, 'group_column_invalid': 31, 'holdout_copies_in_training': 47, 'input_error': 8, 'non_finite_score': 10, 'target_constant': 20, 'task_ambiguous': 35, 'task_unsupported': 38, 'time_column_invalid': 23, 'time_order_undeclared': 56, 'too_few_groups': 19, 'too_few_rows': 37}, 0 other
reference: 585 result(s) {'grouped': 156, 'random': 190, 'time_ordered': 239}, 415 named error(s) {'column_not_found': 8, 'duplicate_columns': 14, 'entities_undeclared': 34, 'fast_profile_group_too_large': 3, 'fast_profile_too_few_rows': 2, 'fast_profile_too_few_timestamps': 5, 'feature_name_collision': 13, 'gap_too_large': 12, 'group_column_invalid': 31, 'holdout_copies_in_training': 47, 'input_error': 8, 'non_finite_score': 10, 'target_constant': 20, 'task_ambiguous': 35, 'task_unsupported': 38, 'time_column_invalid': 23, 'time_order_undeclared': 56, 'too_few_groups': 19, 'too_few_rows': 37}, 0 other
time: 140 s wall-clock at 24 worker(s) (10 s making the tables); in the tool: application 337 s, reference 330 s
GENERATED TABLES: PASS (1000 tables from seed 20261122: 1000 agree, 0 stated limit, 0 open, 0 fail)
```

```text
generated tables: 1000 table(s) from seed 1, 334 worker run(s), 24 at a time, under <scratch>\gen\snap9\verify_new\artifacts\generated\run-7mfhbojt
  package:   <scratch>\gen\snap9\proto_snapshot\mlprofessor
  reference: <scratch>\gen\snap9\verify_new\reference\mlprofessor

STATED LIMITS
  none is known

OPEN: owned, not yet built
  none is known

NOT COMPARED (no part of the verdict): differences by kind
      0  the table earns two named errors of steps 1 to 4, and which is raised is open (spec 2.4)
      0  it depends on which rows a random or a grouped split took, or the fast cap kept (spec 11)
      0  both sides raise the warning; how many records it has is not pinned (spec 2.5)
      0  which estimator of the lag-1 autocorrelation the panel warning uses is open (spec 11)

application: 580 result(s) {'grouped': 128, 'random': 207, 'time_ordered': 245}, 420 named error(s) {'column_not_found': 12, 'duplicate_columns': 20, 'entities_undeclared': 30, 'fast_profile_group_too_large': 14, 'fast_profile_too_few_timestamps': 4, 'feature_name_collision': 14, 'gap_too_large': 14, 'group_column_invalid': 19, 'holdout_copies_in_training': 39, 'input_error': 8, 'non_finite_score': 10, 'one_class': 3, 'target_constant': 17, 'task_ambiguous': 29, 'task_unsupported': 27, 'time_column_invalid': 26, 'time_order_undeclared': 55, 'too_few_groups': 25, 'too_few_rows': 54}, 0 other
reference: 580 result(s) {'grouped': 128, 'random': 207, 'time_ordered': 245}, 420 named error(s) {'column_not_found': 12, 'duplicate_columns': 20, 'entities_undeclared': 30, 'fast_profile_group_too_large': 14, 'fast_profile_too_few_timestamps': 4, 'feature_name_collision': 14, 'gap_too_large': 14, 'group_column_invalid': 19, 'holdout_copies_in_training': 39, 'input_error': 8, 'non_finite_score': 10, 'one_class': 3, 'target_constant': 17, 'task_ambiguous': 29, 'task_unsupported': 27, 'time_column_invalid': 26, 'time_order_undeclared': 55, 'too_few_groups': 25, 'too_few_rows': 54}, 0 other
time: 103 s wall-clock at 24 worker(s) (8 s making the tables); in the tool: application 336 s, reference 328 s
GENERATED TABLES: PASS (1000 tables from seed 1: 1000 agree, 0 stated limit, 0 open, 0 fail)
```

```text
generated tables: 1000 table(s) from seed 2, 334 worker run(s), 24 at a time, under <scratch>\gen\snap9\verify_new\artifacts\generated\run-2mufvylm
  package:   <scratch>\gen\snap9\proto_snapshot\mlprofessor
  reference: <scratch>\gen\snap9\verify_new\reference\mlprofessor

STATED LIMITS
  none is known

OPEN: owned, not yet built
  none is known

NOT COMPARED (no part of the verdict): differences by kind
      0  the table earns two named errors of steps 1 to 4, and which is raised is open (spec 2.4)
      0  it depends on which rows a random or a grouped split took, or the fast cap kept (spec 11)
      0  both sides raise the warning; how many records it has is not pinned (spec 2.5)
      0  which estimator of the lag-1 autocorrelation the panel warning uses is open (spec 11)

application: 590 result(s) {'grouped': 129, 'random': 200, 'time_ordered': 261}, 410 named error(s) {'column_not_found': 8, 'duplicate_columns': 15, 'entities_undeclared': 30, 'fast_profile_group_too_large': 6, 'fast_profile_too_few_rows': 2, 'fast_profile_too_few_timestamps': 9, 'feature_name_collision': 11, 'gap_too_large': 11, 'group_column_invalid': 35, 'holdout_copies_in_training': 41, 'input_error': 5, 'non_finite_score': 15, 'one_class': 2, 'target_constant': 22, 'task_ambiguous': 22, 'task_unsupported': 28, 'time_column_invalid': 25, 'time_order_undeclared': 69, 'too_few_groups': 14, 'too_few_rows': 40}, 0 other
reference: 590 result(s) {'grouped': 129, 'random': 200, 'time_ordered': 261}, 410 named error(s) {'column_not_found': 8, 'duplicate_columns': 15, 'entities_undeclared': 30, 'fast_profile_group_too_large': 6, 'fast_profile_too_few_rows': 2, 'fast_profile_too_few_timestamps': 9, 'feature_name_collision': 11, 'gap_too_large': 11, 'group_column_invalid': 35, 'holdout_copies_in_training': 41, 'input_error': 5, 'non_finite_score': 15, 'one_class': 2, 'target_constant': 22, 'task_ambiguous': 22, 'task_unsupported': 28, 'time_column_invalid': 25, 'time_order_undeclared': 69, 'too_few_groups': 14, 'too_few_rows': 40}, 0 other
time: 100 s wall-clock at 24 worker(s) (8 s making the tables); in the tool: application 333 s, reference 323 s
GENERATED TABLES: PASS (1000 tables from seed 2: 1000 agree, 0 stated limit, 0 open, 0 fail)
```

The tool as it is now, the reference against itself as the package, three
seeds: 1,000 tables agree at each, nothing open.

```text
generated tables: 1000 table(s) from seed 20261122, 334 worker run(s), 24 at a time, under E:\CODE\ml-professor\verify\artifacts\generated\run-w7wj949r
  package:   E:\CODE\ml-professor\verify\reference\mlprofessor
  reference: E:\CODE\ml-professor\verify\reference\mlprofessor

STATED LIMITS
  none is known

OPEN: owned, not yet built
  none is known

NOT COMPARED (no part of the verdict): differences by kind
      0  the table earns two named errors of steps 1 to 4, and which is raised is open (spec 2.4)
      0  it depends on which rows a random or a grouped split took, or the fast cap kept (spec 11)
      0  both sides raise the warning; how many records it has is not pinned (spec 2.5)
      0  which estimator of the lag-1 autocorrelation the panel warning uses is open (spec 11)

application: 585 result(s) {'grouped': 156, 'random': 190, 'time_ordered': 239}, 415 named error(s) {'column_not_found': 8, 'duplicate_columns': 14, 'entities_undeclared': 34, 'fast_profile_group_too_large': 3, 'fast_profile_too_few_rows': 2, 'fast_profile_too_few_timestamps': 5, 'feature_name_collision': 13, 'gap_too_large': 12, 'group_column_invalid': 31, 'holdout_copies_in_training': 47, 'input_error': 8, 'non_finite_score': 10, 'target_constant': 20, 'task_ambiguous': 35, 'task_unsupported': 38, 'time_column_invalid': 23, 'time_order_undeclared': 56, 'too_few_groups': 19, 'too_few_rows': 37}, 0 other
reference: 585 result(s) {'grouped': 156, 'random': 190, 'time_ordered': 239}, 415 named error(s) {'column_not_found': 8, 'duplicate_columns': 14, 'entities_undeclared': 34, 'fast_profile_group_too_large': 3, 'fast_profile_too_few_rows': 2, 'fast_profile_too_few_timestamps': 5, 'feature_name_collision': 13, 'gap_too_large': 12, 'group_column_invalid': 31, 'holdout_copies_in_training': 47, 'input_error': 8, 'non_finite_score': 10, 'target_constant': 20, 'task_ambiguous': 35, 'task_unsupported': 38, 'time_column_invalid': 23, 'time_order_undeclared': 56, 'too_few_groups': 19, 'too_few_rows': 37}, 0 other
time: 246 s wall-clock at 24 worker(s) (10 s making the tables); in the tool: application 385 s, reference 360 s
GENERATED TABLES: PASS (1000 tables from seed 20261122: 1000 agree, 0 stated limit, 0 open, 0 fail)
exit 0
```

```text
generated tables: 1000 table(s) from seed 1, 334 worker run(s), 24 at a time, under E:\CODE\ml-professor\verify\artifacts\generated\run-8bx09a2i
  package:   E:\CODE\ml-professor\verify\reference\mlprofessor
  reference: E:\CODE\ml-professor\verify\reference\mlprofessor

STATED LIMITS
  none is known

OPEN: owned, not yet built
  none is known

NOT COMPARED (no part of the verdict): differences by kind
      0  the table earns two named errors of steps 1 to 4, and which is raised is open (spec 2.4)
      0  it depends on which rows a random or a grouped split took, or the fast cap kept (spec 11)
      0  both sides raise the warning; how many records it has is not pinned (spec 2.5)
      0  which estimator of the lag-1 autocorrelation the panel warning uses is open (spec 11)

application: 580 result(s) {'grouped': 128, 'random': 207, 'time_ordered': 245}, 420 named error(s) {'column_not_found': 12, 'duplicate_columns': 20, 'entities_undeclared': 30, 'fast_profile_group_too_large': 14, 'fast_profile_too_few_timestamps': 4, 'feature_name_collision': 14, 'gap_too_large': 14, 'group_column_invalid': 19, 'holdout_copies_in_training': 39, 'input_error': 8, 'non_finite_score': 10, 'one_class': 3, 'target_constant': 17, 'task_ambiguous': 29, 'task_unsupported': 27, 'time_column_invalid': 26, 'time_order_undeclared': 55, 'too_few_groups': 25, 'too_few_rows': 54}, 0 other
reference: 580 result(s) {'grouped': 128, 'random': 207, 'time_ordered': 245}, 420 named error(s) {'column_not_found': 12, 'duplicate_columns': 20, 'entities_undeclared': 30, 'fast_profile_group_too_large': 14, 'fast_profile_too_few_timestamps': 4, 'feature_name_collision': 14, 'gap_too_large': 14, 'group_column_invalid': 19, 'holdout_copies_in_training': 39, 'input_error': 8, 'non_finite_score': 10, 'one_class': 3, 'target_constant': 17, 'task_ambiguous': 29, 'task_unsupported': 27, 'time_column_invalid': 26, 'time_order_undeclared': 55, 'too_few_groups': 25, 'too_few_rows': 54}, 0 other
time: 281 s wall-clock at 24 worker(s) (12 s making the tables); in the tool: application 388 s, reference 362 s
GENERATED TABLES: PASS (1000 tables from seed 1: 1000 agree, 0 stated limit, 0 open, 0 fail)
exit 0
```

```text
generated tables: 1000 table(s) from seed 2, 334 worker run(s), 24 at a time, under E:\CODE\ml-professor\verify\artifacts\generated\run-_7dp3z2s
  package:   E:\CODE\ml-professor\verify\reference\mlprofessor
  reference: E:\CODE\ml-professor\verify\reference\mlprofessor

STATED LIMITS
  none is known

OPEN: owned, not yet built
  none is known

NOT COMPARED (no part of the verdict): differences by kind
      0  the table earns two named errors of steps 1 to 4, and which is raised is open (spec 2.4)
      0  it depends on which rows a random or a grouped split took, or the fast cap kept (spec 11)
      0  both sides raise the warning; how many records it has is not pinned (spec 2.5)
      0  which estimator of the lag-1 autocorrelation the panel warning uses is open (spec 11)

application: 590 result(s) {'grouped': 129, 'random': 200, 'time_ordered': 261}, 410 named error(s) {'column_not_found': 8, 'duplicate_columns': 15, 'entities_undeclared': 30, 'fast_profile_group_too_large': 6, 'fast_profile_too_few_rows': 2, 'fast_profile_too_few_timestamps': 9, 'feature_name_collision': 11, 'gap_too_large': 11, 'group_column_invalid': 35, 'holdout_copies_in_training': 41, 'input_error': 5, 'non_finite_score': 15, 'one_class': 2, 'target_constant': 22, 'task_ambiguous': 22, 'task_unsupported': 28, 'time_column_invalid': 25, 'time_order_undeclared': 69, 'too_few_groups': 14, 'too_few_rows': 40}, 0 other
reference: 590 result(s) {'grouped': 129, 'random': 200, 'time_ordered': 261}, 410 named error(s) {'column_not_found': 8, 'duplicate_columns': 15, 'entities_undeclared': 30, 'fast_profile_group_too_large': 6, 'fast_profile_too_few_rows': 2, 'fast_profile_too_few_timestamps': 9, 'feature_name_collision': 11, 'gap_too_large': 11, 'group_column_invalid': 35, 'holdout_copies_in_training': 41, 'input_error': 5, 'non_finite_score': 15, 'one_class': 2, 'target_constant': 22, 'task_ambiguous': 22, 'task_unsupported': 28, 'time_column_invalid': 25, 'time_order_undeclared': 69, 'too_few_groups': 14, 'too_few_rows': 40}, 0 other
time: 271 s wall-clock at 24 worker(s) (12 s making the tables); in the tool: application 365 s, reference 350 s
GENERATED TABLES: PASS (1000 tables from seed 2: 1000 agree, 0 stated limit, 0 open, 0 fail)
exit 0
```

Alone on a quiet machine one seed takes about 110 seconds
(`generated_tables\generated_reference_quiet_machine.txt`).

The application against the reference, by rule (`by_rule_generated.py`; the
whole output is `generated_tables\generated_application.txt`):

```text
package E:\CODE\ml-professor\mlprofessor
reference E:\CODE\ml-professor\verify\reference\mlprofessor
1000 tables of seed 20261122: {'agrees': 415, 'stated limit': 0, 'open': 0, 'fails': 585, 'harness': 0}; 275.4 s at 24 workers; in the tool {'application': 1926.3459999999993, 'reference': 384.4939999999997}
outcomes {'application': {'result': 585, 'error': 415, 'other': 0}, 'reference': {'result': 585, 'error': 415, 'other': 0}}
failing findings by class {'property': 662, 'exact': 1144, 'fit': 1163}

585 tables fail

tables by rule (a table is counted under every rule one of its findings has):
  A1 baseline: 47: 2, 11, 58, 79, 91, 136, 143, 180, 198, 235, 259, 261, 271, 296, 319, 322, 354, 388, 410, 422, 439, 450, 454, 462, 469, 524, 547, 551, 553, 567, 574, 590, 601, 644, 674, 700, 704, 706, 751, 775, 809, 829, 846, 925, 948, 967, 979
  A2/A3b screen and order reading: 27: 2, 21, 81, 159, 212, 221, 247, 301, 319, 339, 397, 445, 450, 461, 463, 478, 524, 635, 704, 742, 755, 775, 812, 912, 916, 918, 948
  A3 statements: 585: 0, 1, 2, 5, 6, 8, 11, 13, 14, 18, 21, 25, 29, 31, 32, 33, 34, 36, 38, 39, ...
  A3 warnings: 1: 918
  A4 order of rows: 8: 296, 332, 605, 626, 644, 816, 836, 974
  A5 identifier: 14: 78, 81, 234, 417, 452, 528, 745, 789, 809, 853, 854, 867, 942, 992
  A6 no choice: 3: 634, 778, 923

tables by the set of rules they fail for:
    497  A3 statements
     37  A1 baseline + A3 statements
     18  A2/A3b screen and order reading + A3 statements
     12  A3 statements + A5 identifier
      7  A1 baseline + A2/A3b screen and order reading + A3 statements
      6  A3 statements + A4 order of rows
      3  A3 statements + A6 no choice
      2  A1 baseline + A3 statements + A4 order of rows
      1  A2/A3b screen and order reading + A3 statements + A5 identifier
      1  A1 baseline + A3 statements + A5 identifier
      1  A2/A3b screen and order reading + A3 statements + A3 warnings

failing findings by rule, class and field:
  A1 baseline | exact | baselines: 47
  A1 baseline | exact | baselines (names): 47
  A1 baseline | exact | headline.baseline: 23
  A1 baseline | exact | headline.baseline_score: 23
  A1 baseline | property | application: under time order a regression scores both baselines, whet: 47
  A2/A3b screen and order reading | exact | columns: 6
  A2/A3b screen and order reading | exact | warnings: 26
  A2/A3b screen and order reading | property | application: an exact copy of the target is excluded by the leak scree: 14
  A3 statements | exact | cv.baseline_largest_row: 239
  A3 statements | exact | cv.baseline_largest_row_share: 239
  A3 statements | exact | headline.baseline_largest_row: 239
  A3 statements | exact | headline.baseline_largest_row_share: 239
  A3 statements | fit | cv.largest_row: 239
  A3 statements | fit | cv.largest_row_share: 239
  A3 statements | fit | headline.largest_row: 239
  A3 statements | fit | headline.largest_row_share: 239
  A3 statements | fit | models: 112
  A3 statements | property | application: headline holds the largest row's share, for the model and: 585
  A3 warnings | exact | warnings: 1
  A5 identifier | exact | columns: 15
  A5 identifier | property | application: a whole-number column is id-like exactly when every value: 13
  A6 no choice | property | application: one selection_not_made record exactly when, under the gro: 3
  follows | fit | cv.chosen: 2
  follows | fit | cv.chosen_mean: 11
  follows | fit | cv.chosen_std: 11
  follows | fit | headline.model: 2
  follows | fit | headline.score: 11
  follows | fit | headline.skill: 34
  follows | fit | models: 13
  follows | fit | predictions.csv (y_pred): 11

tables with a finding that no rule explains, in full: 0
```

How many of the reference's 585 finished tables carry one of the new
warnings:

```text
585 finished runs; runs with each new warning: {'selection_not_made': 3, 'largest_row_dominates': 1}
tables with an extreme_value warning: 0; columns named: []
```

On the six broken copies: M13 is caught on 151 tables
(`generated_tables\generated_variant_M13.txt`); F22, M33, M34, M37 and M39 on
none.

The SHA-256 of every judge file every proof above was made on, in the last
state of each:

```text
7af964a499938a84ef359ff3be085306ac2de173928d4b096e38f66e5046391c *verify/smoke_test.py
88eee281dd87fe04de33b022db6eae46a274f6924f5c91af646719ced0176047 *verify/battery.py
9dc7b4f3a022d573a76b623776fd967a7763672fdd4423dac224932f8099fffb *verify/generated_tables.py
a5a759438f28fe6f3a0e8d529364f48ddac023d9dfa06c00cb10ec5ec581455b *verify/mutation_check.py
116caf510038874fc9774413702ce7f5105e612f16bdc90d70404f6a8c96df8e *verify/reference/mlprofessor/core.py
d53cd3d1960e9e0d8babaa77ea23e1e7b3e021e0a8acfe9330f6e4a9e710f7ae *verify/reference/mlprofessor/_variant.py
4ec84288aeb1a21dbea044f13bf1e816f3ae0b339f1c883afa2ff8631c8e9284 *verify/reference/mlprofessor/stages.py
49bee0cabe9c7d681204d411d38976954802527b6bb84b0ceb69e05d75a56628 *spec.md
ccc91844b95fac616931f0a1bf7078c742a37839b71ab86a643ea7707ca571bb *CLAUDE.md
b36d37f75097f601603a0f9e98886d9689aba0e06c81a822f15e951fffbe6e64 *plan/tasks.json
59eeae29001d28a1ae23f02bcc6ba34a381aeb8bfd01b2a8c3285e6c863575e4 *plan/test-plan.md
8f1e4d5cf38cdab8d3634012b31becde076f0fcee00d36ad9a0227b3a05e35ff *.claude/skills/mlprofessor-build/SKILL.md
```

## 11. The harness after the change (proofs 8 and 9)

The judge was built and proved in a copy of the project outside the repo: the
session that built it carried no harness-stage label, and the guard refused
its writes to `verify\`, `plan\`, `spec.md` and `CLAUDE.md`. Every command of
sections 0 to 10 was run in that copy, where `verify\`, `plan\` and the
application stood as they do here. On 2026-11-25 a session that carried the
label placed it: twelve judge files and this folder, 112 files in all, none
under `tests\`, `mlprofessor\`, `loop\`, `examples\` or `.claude\hooks\`.
After placing, each of these was run from the repo with nothing else on the
machine, one after another.

`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B loop/run_loop.py doctor`:

```text
Loop doctor
  ok   interpreter E:\CODE\ml-professor\mlprof_env\Scripts\python.exe (3.12.13)
  ok   hooks registered with an absolute interpreter path
  ok   Claude Code 2.1.332 (Claude Code)
       harness   -> fable   effort=xhigh  turns<=-, $-/run
       tests     -> opus    effort=high   turns<=150, $15/run
       review    -> fable   effort=high   turns<=80, $10/run
       diagnose  -> fable   effort=high   turns<=40, $5/run
       support   -> haiku   effort=medium turns<=60, $2/run
       routine   -> sonnet  effort=medium turns<=100, $6/run
       complex   -> opus    effort=high   turns<=200, $20/run
       total budget $250, permission mode 'auto'
  ok   pytest importable
  ok   gate lists 53 check(s)
  ok   gate self-test: SELFTEST: PASS (good fixture passes twice with identical results; 37 failure mode(s) each caught by a check that names it; temp and project dirs agree)
  ok   gate timing: 12.8s per edit, 17.4s in full
  ok   plan valid: 68 task(s) (49 complex, 10 routine, 9 support)
  WARN 12 uncommitted change(s) in the harness; commit before an unattended run
Doctor: OK, 1 warning(s)
```

The one warning is the twelve judge files of this pass, which the owner
commits. The self-test inside it is the one of section 7.4; the gate's times
here, 12.8 seconds per edit and 17.4 in full, are on the reference.

`E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B loop/run_loop.py build --dry-run`:

```text
[09:50:03] Dry run. Nothing will be executed.
[09:50:03]   T1     support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  Package skeleton, command-line stub, requirements.txt
[09:50:03]   T2     complex  opus x3                            gate=0 tests=1  Load one table; count usable rows
[09:50:03]   T3     complex  opus x3                            gate=0 tests=1  Column roles that need no target
[09:50:03]   T4     complex  opus x3                            gate=0 tests=1  Task decision
[09:50:03]   T5     complex  opus x3                            gate=0 tests=1  Random holdout and folds
[09:50:03]   T6     complex  opus x3                            gate=0 tests=1  Time column, time-ordered holdout, expanding folds, gap
[09:50:03]   T7     complex  opus x3                            gate=0 tests=1  Metrics, skill and the previous-value baseline
[09:50:03]   T8     complex  opus x3                            gate=0 tests=1  Preprocessing fitted on the rows it is given
[09:50:03]   T9     complex  opus x3                            gate=0 tests=1  Fit, select, score once: the run in memory, random scheme, fast profile
[09:50:03]   T10    complex  opus x3                            gate=12 tests=1  profile.json, predictions.csv and the command line: the first run the gate can judge
[09:50:03]   T11    complex  opus x3                            gate=5 tests=1  Time-ordered scheme wired into the run, with its baselines
[09:50:03]   T12    complex  opus x3                            gate=4 tests=1  Leak screen
[09:50:03]   T13    complex  opus x3                            gate=2 tests=1  Interval and p-value, row and block forms
[09:50:03]   T14    routine  sonnet x2 -> opus x3               gate=0 tests=1  Random forest and histogram gradient boosting behind the existing family interface
[09:50:03]   T15    complex  opus x3                            gate=2 tests=1  The full profile's grids, and the same result at any n_jobs
[09:50:03]   T16    routine  sonnet x2 -> opus x3               gate=1 tests=1  Edges, limitations statement, console summary
[09:50:03]   T17    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  README with run instructions
[09:50:03]   T18    complex  opus x3                            gate=0 tests=4  Guards for the first review's test holes, and two CSV edges: no code change expected
[09:50:03]   T19    complex  opus x3                            gate=1 tests=1  Skill when a baseline's error is zero
[09:50:03]   T20    routine  sonnet x2 -> opus x3               gate=0 tests=1  Colliding feature names stop the run
[09:50:03]   T21    complex  opus x3                            gate=0 tests=1  A configuration that fails after it is chosen
[09:50:03]   T22    routine  sonnet x2 -> opus x3               gate=0 tests=2  Small defects: the output folder, process settings, the no-features message, the console summary
[09:50:03]   T23    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README's commands use a table kept in the repo
[09:50:03]   T24    complex  opus x3                            gate=0 tests=1  Guard of the third pass: a line pandas reads as the header. No code change expected
[09:50:03]   T25    complex  opus x3                            gate=0 tests=1  No number that is not finite is reported, or turned into null
[09:50:03]   T26    complex  opus x3                            gate=0 tests=1  White space round a target value; a target cell of only white space
[09:50:03]   T27    routine  sonnet x2 -> opus x3               gate=0 tests=1  The seed's range
[09:50:03]   T28    routine  sonnet x2 -> opus x3               gate=0 tests=1  The fast profile's error only when the full profile would run
[09:50:03]   T29    routine  sonnet x2 -> opus x3               gate=0 tests=1  The console summary prints four significant digits
[09:50:03]   T30    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: which rows the fast profile keeps, and what this pass changed
[09:50:03]   T31    complex  opus x3                            gate=0 tests=1  Guards of the fourth pass: six pinned behaviours the third review could break unseen. No code change expected
[09:50:03]   T32    complex  opus x3                            gate=1 tests=1  Copies of a holdout row among the training rows stop the run; the `duplicates` argument
[09:50:03]   T33    complex  opus x3                            gate=1 tests=1  Whole numbers are compared exactly: integer time keys, distinct values, and whole numbers too large for 64 bits
[09:50:03]   T34    complex  opus x3                            gate=0 tests=1  Number columns as a database driver returns them; a warning for every unsupported column
[09:50:03]   T35    complex  opus x3                            gate=0 tests=1  The type matrix: every kind of column, in every role, ends in a result or a named error
[09:50:03]   T36    routine  sonnet x2 -> opus x3               gate=0 tests=1  One to nine digits in a fraction of a second; a refusal that says what is wrong
[09:50:03]   T37    routine  sonnet x2 -> opus x3               gate=0 tests=1  The fast profile's cap leaving too few rows: its own error when the full profile would run
[09:50:03]   T38    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: copies of holdout rows, and what this pass changed
[09:50:03]   T39    complex  opus x3                            gate=0 tests=1  Guards of the fifth pass: what the fourth review pinned and the application already does. No code change expected
[09:50:03]   T40    complex  opus x3                            gate=0 tests=1  Each object column is read once, over all loaded rows (the fourth review, finding 1)
[09:50:03]   T41    complex  opus x3                            gate=0 tests=1  No part of the output can hold a number that is not finite; the boundary of a failed configuration; the three-class message (the fourth review, findings 4 and 5)
[09:50:03]   T42    complex  opus x3                            gate=5 tests=1  A date column with nothing said about time order stops the run; `time_col=False` and `--no-time-order`; the README's commands
[09:50:03]   T43    complex  opus x3                            gate=2 tests=1  A declared entity column under the random scheme: `group_col`, the grouped split, the cap, the interval
[09:50:03]   T44    complex  opus x3                            gate=1 tests=2  Entities under time order: each entity's own previous value, the leak screen's baseline, and the stop for a panel with no entity declared
[09:50:03]   T45    complex  opus x3                            gate=0 tests=1  `duplicates="group"`: each set of copies on one side of the split and of every fold
[09:50:03]   T46    complex  opus x3                            gate=0 tests=1  The type matrix gains a role: every kind of column as the entity column
[09:50:03]   T47    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: entities, the two statements, `duplicates="group"`, and what is not detected
[09:50:03]   T48    complex  opus x3                            gate=0 tests=1  Guards of the sixth pass: the cut-off of the panel warning, and what a copy is. No code change expected
[09:50:03]   T49    complex  opus x3                            gate=2 tests=1  Both statements on every call: `time_col` and `group_col` left unsaid stop the run, whatever the table holds
[09:50:03]   T50    complex  opus x3                            gate=2 tests=1  `duplicates="group"` keeps together the rows that agree in every feature column; the count and the warning renamed; the copies stop says when each answer fits
[09:50:03]   T51    complex  opus x3                            gate=1 tests=1  The grouped holdout holds a quarter and no more than it must; a unit of a quarter or more is named
[09:50:03]   T52    complex  opus x3                            gate=1 tests=1  Dates written as whole numbers (YYYYMMDD) are a date form: excluded as a date, named, and accepted as the time column
[09:50:03]   T53    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: both statements in every command and example, what `group` means, the two warnings for a large entity, dates as whole numbers, and the limits
[09:50:03]   T54    complex  opus x3                            gate=0 tests=1  Guards of the seventh pass: a second test for eleven thin spots, a date beyond the nanosecond range, and two readings D72 pins. No code change expected
[09:50:03]   T55    complex  opus x3                            gate=1 tests=1  A column of dates written YYYYMMDD stays a date column when it holds a sentinel; as the time column, a value that is no date stops the run
[09:50:03]   T56    complex  opus x3                            gate=1 tests=1  `duplicates="group"` and a column the leak screen removes: one combination of the features that are left on both sides stops the run; the grouped warning says what `group` measures
[09:50:03]   T57    complex  opus x3                            gate=2 tests=1  The fast profile's cap with an entity column: an entity of a quarter or more of the rows is always kept, and the run stops with `fast_profile_group_too_large` when the cap cannot do that
[09:50:03]   T58    routine  sonnet x2 -> opus x3               gate=0 tests=1  Two small things of the sixth review: the `date_column_undeclared` warning carries the caveat of its scheme, and a number that is not finite is never mapped to null
[09:50:03]   T59    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: a date column with a sentinel, the stop under `group` after a removed leak and what `group` measures, the fast profile with a large entity, and the limits
[09:50:03]   T60    complex  opus x3                            gate=0 tests=1  The judge's two other tools are clean on the build: the perturbation battery and the generated-table test
[09:50:03]   T61    complex  opus x3                            gate=1 tests=1  A whole-number identifier stays id-like when the rows are shuffled: the second clause of the id-like rule
[09:50:03]   T62    complex  opus x3                            gate=1 tests=1  The previous value at repeated timestamps: both baselines scored, `stages.previous_value`, and the passage of the panel warning
[09:50:03]   T63    complex  opus x3                            gate=2 tests=3  The leak screen: rows set aside, a line beside the tree, the direct reading of a class, and the order reading of the `strong_single_column` warning
[09:50:03]   T64    complex  opus x3                            gate=1 tests=2  The largest row's share of a squared error, on the holdout and over the selection folds, and the `extreme_value` warning
[09:50:03]   T65    complex  opus x3                            gate=0 tests=1  Under time order the rows of a fit are handed over in time order
[09:50:03]   T66    complex  opus x3                            gate=1 tests=1  No choice between configurations where the folds cannot choose: the `selection_not_made` rule of the grouped scheme
[09:50:03]   T67    support  haiku x2 -> sonnet x2 -> opus x3   gate=0 tests=1  The README: the baseline at repeated timestamps, the largest row's share and the extreme-value warning, no choice on too few rows, and the limits
[09:50:03]   T68    complex  opus x3                            gate=0 tests=2  The judge's two other tools are clean on the build, with nothing listed as open; and five guards
[09:50:03] Then: full gate (53 checks) + full suite, on failure an integration task at tier 'complex'.
```

`verify\smoke_test.py --matrix` from the repo is `output\smoke_test_matrix.txt`,
the same text as in 7.4.

The gate on the application, alone (23 seconds): the 47 checks it had pass,
`output.same_at_any_n_jobs` among them, and exactly the six new ones fail.

```text
PASS data.rows_reconcile
PASS data.columns_accounted
PASS data.coercion_reported
PASS data.long_rows_stop
PASS data.holdout_copies_stop
PASS models.all_accounted
PASS task.binary_numeric
PASS task.ambiguous_stops
PASS leak.excluded
PASS leak.control_kept
PASS leak.regression
PASS leak.near_leak_warned
PASS model.signal_clf
PASS model.signal_reg
PASS model.noise_clf
PASS model.noise_p_value
PASS model.noise_reg
PASS model.headline_recomputed
PASS model.metric_clf
PASS model.selection_full
PASS time.order_respected
PASS time.no_skill_on_blocks
PASS time.gap_applied
PASS time.dayfirst_stops
PASS time.persistence_baseline
PASS time.integer_keys_distinct
PASS time.undeclared_warns
PASS time.undeclared_stops
PASS time.panel_previous_value
PASS groups.kept_together
PASS groups.no_skill_on_entities
PASS statements.time_order_asked
PASS statements.entities_asked
PASS duplicates.group_on_features
PASS duplicates.group_no_skill
PASS groups.holdout_bounded
PASS time.date_numbers_seen
PASS time.date_sentinel_seen
PASS duplicates.group_leak_stops
PASS fast.large_entity_stops
PASS fast.large_entity_kept
FAIL time.previous_value_at_repeats: baselines on `walk_repeated` lists ['training_mean']. One row of this random walk is written twice, so the date 2021-01-21 holds two rows and the timestamps are not unique. With no entity column the previous value of a row is the mean target of the timestamp one before its own, so both training_mean and previous_value are scored and reported (D74; spec.md step 11). A baseline that is there only while every timestamp is unique is lost to one repeated row, and against the training mean a random walk looks almost perfectly predictable
FAIL leak.copy_excluded_under_time: `level_scaled` on `walk_copy` has role 'feature', reason None, leak_rule None; skill is 0.9875713998211809 against the baseline 'previous_value'. The column is the target on another scale, 1.07 x level + 3, and must be excluded as a suspected leak with the rule `predicts_target`. Under time order the screen's baseline is the previous value, and a tree fitted on the earlier rows cannot follow a target that wanders beyond the range it was fitted on: read with its tree alone, this column does not reach the cut-off. The screen reads a numeric column with a robust line as well as with its tree, and excludes it on the better of the two (D74; spec.md step 8)
FAIL leak.excluded_despite_extreme_target: `level_scaled` on `walk_copy_cell` has role 'feature', reason None, leak_rule None; skill is 0.35824851373880295 against the baseline 'previous_value'. The column is the target on another scale, 1.07 x level + 3, and must be excluded as a suspected leak with the rule `predicts_target`. One cell of the target is -999, in a training row (source row 365); in every other row the column is the target on another scale. With every row scored, the squared error of that one row is nearly all of the screen's sum for the column, and that row and the one after it, whose previous value is the cell, are nearly all of the baseline's: the column's skill comes out at one half or less, far below the cut-off. A training row whose target lies more than ten median deviations from the median target, or ten median steps from its previous value, is set aside from the whole screen, and the rows a column misses by most are not scored for exclusion (D74; spec.md step 8)
FAIL result.largest_row_reported: profile.json on `walk_late_cell` has no key headline.largest_row_share, headline.largest_row, headline.baseline_largest_row_share, headline.baseline_largest_row, cv.largest_row_share, cv.largest_row and 6 more. A regression run says what share of a squared error its single largest row carries, and which row that is: on the holdout, for the model and for the headline baseline (`headline`), and over the selection folds, for the chosen configuration and that baseline (`cv`) and for every configuration (`models`) (D74; spec.md 5.2)
FAIL data.identifier_survives_shuffle: `record_id` on `messy_shuffled` has role 'feature' and reason None. In the 1500 rows with a target it holds 1500 distinct whole numbers that fill 97% of the range from the smallest to the largest. A whole-number column in which more than 95% of the values are distinct and the distinct values fill at least half of their range is an identifier, whatever the order of the rows (D74; spec.md step 5). By D19's clause alone, all distinct and consecutive or in file order, it is one in `messy` and a numeric feature here, where the same rows are shuffled
FAIL groups.choice_not_made: 0 warning(s) with the code `selection_not_made` on `dominant_entity`; cv.chosen is 'tree-0' and skill -1.1278. The training rows are the entity C0000, 1490 rows, and 10 rows of small entities. An entity is in one fold, so the fold that validates on the large one trains on at most 10 rows, fewer than 50 and fewer than it validates on, and each other fold validates on some of those same 10 rows. No choice between configurations is made on so few: every fold stays scored and reported, the configurations are taken in configuration order, so the one chosen is the first that has a cross-validation mean, and one `selection_not_made` warning says so (D74; spec.md step 9)
PASS time.flat_holdout_reported
PASS uncertainty.reported
PASS result.matches_json
PASS output.complete
PASS output.deterministic
PASS output.same_at_any_n_jobs
GATE: FAIL (6 of 53 checks failed)
```

The battery from the repo, at its default of 32 side by side: on the
reference, then on the application.

```text
On the reference (verify\battery.py verify\reference\mlprofessor), 281 seconds in all:
1229 tool runs, and 143 second calls by the judge, in 273 s at 32 side by side (7893 s of tool time): 37 gate runs as they are and 1192 pairs; 1398 pairs left out (--list says why)
holds 1223; stated limit 6 of 6 listed; open 0 of 0 listed; on neither list and failing 0
BATTERY: PASS (holds 1223, stated limit 6, open 0)

STATED LIMIT (D31): messy:sorted_by_measure [band]
STATED LIMIT (D60): dup_noise:noise_added [stop-or-band]
STATED LIMIT (D76): random_walk:copy_noise_5pct [structure]
STATED LIMIT (D55): int_stamps:missing_cell_date [structure]
STATED LIMIT (D76): int_stamps:copy_noise_5pct [structure]
STATED LIMIT (D76): panel:copy_noise_5pct [structure]

On the application (verify\battery.py mlprofessor), 232 seconds in all:
holds 299; stated limit 6 of 6 listed; open 0 of 0 listed; on neither list and failing 924
BATTERY: FAIL (924 of 1229 do not hold and are on neither list; holds 299, stated limit 6, open 0)

924 pairs fail and are on neither list (930 of the 1229 runs finished)

pairs by rule (a pair is counted under every rule one of its failing claims has):
  A1 baseline: 73
  A2/A3b screen and order reading: 144
  A3 statements: 922
  A3 extreme value: 162
  A5 identifier: 66
  A6 no choice: 31
  none: 0

By the task that owns the rule (a pair is counted under every task one of its failing claims belongs to; in brackets, the pairs that fail for that task's rule alone):
  T61 identifier rule: 66 (2)
  T62 previous value at repeated timestamps: 73 (0)
  T63 leak screen and order reading: 144 (0)
  T64 largest row's share and extreme_value: 922 (638)
  T65 order of rows in a fit: 0 (no pair tests it; the generated tables do, on 8 tables)
  T66 no choice on too few rows: 31 (0)
```
