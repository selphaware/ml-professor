# CP1 measurements

The scripts, the seeds and the per-seed values behind every band and cut-off
measured in the CP1 session (2026-11-16), kept here by D35. The conclusions
drawn from them are in `spec.md` section 10; this folder is how to get the
numbers again.

| File | What it is |
|---|---|
| `lab.py` | Runs the reference implementation on the gate's datasets and on crafted near-copies, over the 30 seeds, and compares two states of the judge. |
| `cases.py` | Crafted columns and tables for the separator rule and the time-ordered recipes; the two bands at the gate's seed. |
| `starts.py` | Starts eight tool processes at the same instant, round after round (D32). |
| `per_seed.csv` | One row per seed, one column per measured value: 30 rows, 118 values. |

Raw outputs are not kept. Every command below prints its ranges and writes its
detail outside the repo.

## Seeds

- **30 base seeds: 1115 + 100*k*, *k* = 0 to 29.** *k* = 0 is the gate's own
  seed (`SEED` in the gate script).
- A dataset's own seed is the base seed plus the dataset's position in the
  gate's `DATASETS`: `signal_clf` +0, `signal_reg` +1, `noise_clf` +2,
  `noise_reg` +3, `imbalanced` +4, `leak` +5, `leak_reg` +6, `binary_numeric`
  +7, `ambiguous_target` +8, `messy` +9, `time_blocks` +10, `time_dayfirst`
  +11, `random_walk` +12, `noise_small` +13, `signal_clf_small` +14,
  `signal_reg_small` +15, `near_leak` +16.
- The tool always runs with `seed=0`, `n_jobs=1`, and BLAS and OpenMP held to
  one thread; fast profile unless a run of the gate says full.
- The crafted near-copies use the dataset seed of `near_leak` (base + 16).
  The rows to flip come from `random.Random("near_leak.outcome_noisy.<seed>")`
  and the regression noise from `random.Random("near_leak.amount_noisy.<seed>")`.
- The crafted columns of `cases.py roles` and `coerce` use the fixed seeds 7
  and 11.
- **The seeds of the first harness session (2026-11-15) were not recorded.**
  `lab.py scheme` holds the 22 schemes that were tried against its recorded
  range on `messy` (0.630 to 0.824); none reproduces it.

## Running a measurement again

From the repo root, with the project interpreter, one command at a time:

```
$env:MLPROF_LAB_WORK = "C:\somewhere\outside\the\repo"
$py = "E:\CODE\ml-professor\mlprof_env\Scripts\python.exe"
$W  = "$env:MLPROF_LAB_WORK\out"
```

`MLPROF_LAB_WORK` is where the generated datasets and results go; without it
they go to `mlprof-lab` in the system's temporary folder. In every command `-`
stands for the repo's own reference implementation or gate script; give a path
instead to measure another state of the judge (see "Comparing two states").

| What it backs | Command | Takes |
|---|---|---|
| Every band: headline skill on the gate's datasets, 30 seeds | `& $py -B docs/measurements/CP1/lab.py headlines now - - "$W\head_k30.json" k30 "signal_clf,signal_reg,noise_clf,noise_reg,imbalanced,leak,leak_reg,near_leak,binary_numeric,messy,time_blocks,time_blocks_gap,time_blocks_undeclared,random_walk"` | 4 min |
| The same at the gate's seed, the full-profile runs included | `& $py -B docs/measurements/CP1/lab.py headlines now - - "$W\head_gate.json" gate` | 30 s |
| Warnings and column roles in either file | `& $py -B docs/measurements/CP1/lab.py tally "$W\head_k30.json"` | |
| Both leak cut-offs: every single-column skill in every gate dataset | `& $py -B docs/measurements/CP1/lab.py screen now - - "$W\screen_k30.json" k30` | 3 min |
| Where the warning tier's edges are; the test recipes (random scheme, both tasks) | `& $py -B docs/measurements/CP1/lab.py nearleak - - "$W\nearleak.json"` | 5 min |
| The test recipes under time order | `& $py -B docs/measurements/CP1/cases.py - - neartime` | 1 min |
| The separator rule: how `revenue`, `fee` and `spend` of `messy` are written | `& $py -B docs/measurements/CP1/cases.py - - shares` | |
| The separator rule on crafted columns; the warnings of a run | `& $py -B docs/measurements/CP1/cases.py - - roles` | |
| The coercion measurement of 2,000 rows | `& $py -B docs/measurements/CP1/cases.py - - coerce` | |
| The pattern, value by value | `& $py -B docs/measurements/CP1/cases.py - - examples` | |
| `model.noise_reg` and `model.metric_clf` at the gate's seed | `& $py -B docs/measurements/CP1/cases.py - - gateseed` | |
| Simultaneous starts (quiet machine) | `& $py -B docs/measurements/CP1/starts.py 8 30` | 2 min |
| `per_seed.csv`, from the files the commands above wrote | `& $py -B docs/measurements/CP1/lab.py table docs/measurements/CP1/per_seed.csv` | |

`table` needs `head_k30.json`, `screen_k30.json`, `nearleak.json`,
`neartime.json` and `shares.json` in `$W`, under exactly those names.

### Comparing two states of the judge

"No band's measurement changed" was shown by running two states of the
reference implementation on the same seeds and comparing run by run:

```
& $py -B docs/measurements/CP1/lab.py headlines old <old>\verify\reference <old>\verify\smoke_test.py "$W\old_k30.json" k30 "<runs>"
& $py -B docs/measurements/CP1/lab.py compare "$W\old_k30.json" "$W\head_k30.json"
& $py -B docs/measurements/CP1/lab.py messycols <old>\verify\smoke_test.py -
```

`<old>` is a copy of the repo at an earlier commit, outside this folder; the
human makes it, since agents do not run git here. Three states were compared
in the session:

1. the first harness, commit `070e5b8`;
2. the harness after D27 to D30 as first built, when D28 still had two forms
   of its warning (reproducible only if that state was committed);
3. the harness after D34.

`compare` reports differences in the headline, the row counts, the task, the
validation record, the baselines, the models table and the predictions, and
lists the warnings gained and lost. The column accounting is left out on
purpose: `messy` gained columns.

## What `per_seed.csv` holds

One row per base seed (`k`, `base_seed`), then:

| Columns | Meaning |
|---|---|
| `skill.<run>` | `headline.skill` of the reference on that run of the gate. Identical in all three states for every run that existed in each. |
| `single_top.<run>` | The highest single-column skill of any legitimate column in that run's dataset. |
| `single.<run>.<column>` | The single-column skill of one named column: the planted leaks `outcome_copy` and `amount_scaled`, the near-leak `outcome_noisy`, and the legitimate `strong_feature`. |
| `flip.random.n<rows>.<exact or chance>.<share>` | Single-column skill of a copy of a two-class target with that share of its labels flipped, random scheme. `exact`: exactly that share of the rows. `chance`: each row at that chance. |
| `flip.time.n<rows>.exact.<share>` | The same under the time-ordered scheme, unique timestamps. |
| `regnoise.random.n<rows>.<share>` | Single-column skill of a rescaled copy of a continuous target plus noise whose deviation is that share of the target's. |
| `messy.revenue.share_*`, `messy.fee.*`, `messy.spend.*` | How those columns of `messy` are written, among the rows with a target unless the name says all rows: the share that parses, the share or count that matches the separator pattern, the share distinct, the count of text cells. |

`settlement_amount` has no single-column skill: the missing-by-class rule
excludes it before the skill is measured. `time_dayfirst` and
`ambiguous_target` stop with their named errors and have no values.

## What was found

Ranges are over the 30 seeds; mean and sd in brackets.

- **Bands.** The three states agree on every seed: 390 runs over 30 seeds and
  19 at the gate's seed between the first state and the last, 420 and 20
  between the second and the last, with no difference. No band was changed.
  `messy` is 0.670 to 0.826 (0.746, 0.035) in all three.
- **Warning cut-off, 0.90 (D30, D33).** `outcome_noisy` of `near_leak`: 0.938
  to 0.960 (0.949, 0.005). The strongest legitimate column in any gate
  dataset, `strong_feature` of `leak`: 0.609 to 0.748 (0.656, 0.032).
- **Separator rule (D28, D34).** `revenue` and `fee` are excluded with the
  reason `numeric_with_separators` on 30 of 30; `spend` stays numeric with
  `values_coerced` alone on 30 of 30. The reference of the first state calls
  `revenue` id-like and keeps `fee` as numeric with six cells coerced.
- **Two bands are narrower than 4.5 sd on these seeds** and are left as they
  are (D35). `noise_reg`: −0.503 to −0.242 (−0.377, 0.076) against a band of
  −0.65 to −0.05. `imbalanced`: −0.150 to +0.376 (+0.017, 0.097) against −0.42
  to 0.42. At the gate's seed the reference is at −0.407 and +0.072.
- **Simultaneous starts (D32).** Eight at the same instant, 30 rounds, twice:
  none blocked. Each process took 2.9 to 3.2 seconds, and 10.9 in the first
  round of the first set.

## Two things to know about these files

- The loop driver classifies anything under `docs/` as application code. Its
  build report lists application files that mention the gate, and these three
  scripts do, so they will appear in that list. They are measurement tools for
  the harness stage and no part of the package.
- For the same reason a build-stage agent is not prevented from editing them.
  If a number here ever disagrees with `spec.md`, run the command again.
