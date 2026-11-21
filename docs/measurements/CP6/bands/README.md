# CP6: the two new bands, and the interval and p-value under the grouped scheme

The scripts and the per-seed values behind spec section 10.7 and the two rows
that the fifth pass added to section 10.1 (D35, D61). Measured on 2026-11-21 on
the judge as installed for the fifth pass: `verify\smoke_test.py` and
`verify\reference`. Nothing here is part of the gate, and neither script
writes inside the repo.

## What each script measures

**`bands.py`** writes the gate's datasets `entities` and `panel` with the
gate's own generators, once for each data seed, and profiles each as the gate
profiles it (fast profile, tool seed 0, one core), with the reference and
with the broken copies of it that the two score checks must catch:

- `entities` with `group_col="customer"`: the reference (`good`); F10, where
  the split is made row by row although the entity column is declared; and
  M21, where the interval is drawn from single rows and still reported as
  `bootstrap_groups`;
- `panel` with `time_col="date"` and `group_col="series"`: the reference; and
  F8b, where the headline is scored against the training mean alone.

It writes one line of JSON for each run: the dataset, the variant, the seed
number `k`, `skill`, the tool's interval on skill (`low`, `high`), the
baseline, the model, the scheme, the holdout rows and the entities on each
side. `bands.py table` prints the range, the mean and the standard deviation
of `skill` for each dataset and variant. The bands of `verify\smoke_test.py`
are the reference's mean plus or minus about 4.5 standard deviations, rounded
outwards.

**`grouped_uncertainty.py`** asks what a row bootstrap and a row shuffle would
say under the grouped scheme. On `entities` nothing predicts the label of an
entity the model has not seen, so the true skill is zero. For each data seed
it profiles `entities` with the reference and, from the holdout predictions,
works out for itself: a 95% interval on skill from 1,000 resamples of rows,
and one from 1,000 resamples of whole entities; a p-value from 1,000 shuffles
of the holdout labels row by row, and one from 1,000 shuffles entity by
entity. It prints one line for each seed, then the totals. A sound interval
holds zero in about 19 runs of 20; a sound p-value is below 0.05 in about one
of 20.

## The commands

From `E:\CODE\ml-professor`, in PowerShell. The file `bands.py` writes goes
outside the repo. The last argument but one of the first command, and the
last of the third, is the number of data seeds: `1` runs the gate's own seed
alone.

```
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP6\bands\bands.py verify\smoke_test.py verify\reference 30 $env:TEMP\cp6_bands.jsonl
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP6\bands\bands.py table $env:TEMP\cp6_bands.jsonl
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe -B docs\measurements\CP6\bands\grouped_uncertainty.py verify\smoke_test.py verify\reference 30
```

The first takes 17 seconds for 30 seeds and the third 14, on this machine.

## The seeds

- **Data seeds:** 30 base seeds, 1115 + 100 × *k* for *k* = 0 to 29. 1115 is
  the gate's base seed, so *k* = 0 is the data the gate itself uses. A
  dataset's own seed is the base seed plus its position in the gate's list:
  1136 + 100 × *k* for `entities` and 1137 + 100 × *k* for `panel`.
- **Tool seed:** 0 in every run, as in the gate.
- **The resamples and shuffles of `grouped_uncertainty.py`:** one generator
  for each data seed, `numpy.random.default_rng([base seed, 1])`.

## The values, seed by seed

### The two bands (`bands.py`)

`skill` of each run, and for `entities` the tool's 95% interval on skill: the
reference's, from whole entities, and M21's, from single rows (its skill is
the reference's, since only the interval differs).

| k | base seed | `entities`: reference | its interval on skill | F10 | M21's interval on skill | `panel`: reference | F8b |
|---|---|---|---|---|---|---|---|
| 0 | 1115 | -0.0812 | -0.305 to +0.183 | +0.7974 | -0.193 to +0.022 | -0.0098 | +0.9985 |
| 1 | 1215 | +0.0782 | -0.143 to +0.292 | +0.8514 | -0.035 to +0.171 | -0.0228 | +0.9978 |
| 2 | 1315 | +0.0410 | -0.173 to +0.246 | +0.8401 | -0.068 to +0.158 | -0.0002 | +0.9967 |
| 3 | 1415 | -0.0460 | -0.306 to +0.252 | +0.8499 | -0.168 to +0.062 | +0.0002 | +0.9982 |
| 4 | 1515 | +0.0042 | -0.179 to +0.183 | +0.8422 | -0.095 to +0.096 | -0.0021 | +0.9965 |
| 5 | 1615 | -0.0931 | -0.367 to +0.093 | +0.8398 | -0.205 to +0.015 | +0.0038 | +0.9983 |
| 6 | 1715 | -0.0350 | -0.293 to +0.209 | +0.8027 | -0.150 to +0.076 | +0.0017 | +0.9982 |
| 7 | 1815 | -0.1737 | -0.356 to +0.017 | +0.7913 | -0.281 to -0.076 | -0.0012 | +0.9953 |
| 8 | 1915 | +0.0289 | -0.158 to +0.225 | +0.8746 | -0.073 to +0.139 | -0.0123 | +0.9982 |
| 9 | 2015 | -0.0575 | -0.344 to +0.189 | +0.8426 | -0.171 to +0.040 | -0.0012 | +0.9985 |
| 10 | 2115 | -0.0591 | -0.284 to +0.162 | +0.8446 | -0.163 to +0.030 | +0.0018 | +0.9983 |
| 11 | 2215 | -0.0459 | -0.244 to +0.148 | +0.8498 | -0.141 to +0.062 | -0.0089 | +0.9984 |
| 12 | 2315 | -0.2036 | -0.473 to +0.059 | +0.8179 | -0.324 to -0.075 | +0.0045 | +0.9978 |
| 13 | 2415 | +0.0323 | -0.144 to +0.216 | +0.8234 | -0.049 to +0.116 | -0.0001 | +0.9985 |
| 14 | 2515 | -0.0832 | -0.309 to +0.167 | +0.8514 | -0.191 to +0.041 | -0.0002 | +0.9977 |
| 15 | 2615 | +0.0904 | -0.152 to +0.325 | +0.7972 | -0.036 to +0.209 | -0.0036 | +0.9989 |
| 16 | 2715 | +0.1526 | -0.063 to +0.332 | +0.8024 | +0.041 to +0.248 | -0.0091 | +0.9980 |
| 17 | 2815 | -0.0758 | -0.283 to +0.144 | +0.8234 | -0.188 to +0.022 | +0.0002 | +0.9989 |
| 18 | 2915 | -0.0464 | -0.315 to +0.222 | +0.8288 | -0.158 to +0.056 | -0.0042 | +0.9970 |
| 19 | 3015 | -0.2599 | -0.495 to +0.050 | +0.8818 | -0.380 to -0.144 | +0.0003 | +0.9978 |
| 20 | 3115 | -0.2015 | -0.445 to +0.070 | +0.7600 | -0.315 to -0.089 | -0.0026 | +0.9966 |
| 21 | 3215 | +0.1542 | -0.121 to +0.461 | +0.8069 | +0.012 to +0.255 | -0.0073 | +0.9989 |
| 22 | 3315 | +0.1424 | -0.102 to +0.408 | +0.8487 | +0.050 to +0.259 | -0.0070 | +0.9976 |
| 23 | 3415 | +0.0606 | -0.174 to +0.239 | +0.8473 | -0.048 to +0.160 | -0.0185 | +0.9979 |
| 24 | 3515 | -0.0632 | -0.345 to +0.238 | +0.7782 | -0.166 to +0.036 | +0.0002 | +0.9976 |
| 25 | 3615 | +0.1681 | -0.097 to +0.414 | +0.7952 | +0.069 to +0.275 | +0.0062 | +0.9970 |
| 26 | 3715 | +0.1501 | -0.006 to +0.335 | +0.8116 | +0.050 to +0.256 | +0.0049 | +0.9956 |
| 27 | 3815 | -0.0083 | -0.207 to +0.166 | +0.8117 | -0.116 to +0.082 | -0.0024 | +0.9965 |
| 28 | 3915 | -0.2398 | -0.450 to +0.046 | +0.8363 | -0.352 to -0.132 | +0.0030 | +0.9979 |
| 29 | 4015 | -0.1405 | -0.321 to +0.052 | +0.7972 | -0.244 to -0.041 | -0.0010 | +0.9985 |

Over the 30 seeds:

| Dataset | Variant | Skill: range | Mean | Standard deviation | Band in the gate |
|---|---|---|---|---|---|
| `entities` | reference | -0.2599 to +0.1681 | -0.0270 | 0.1185 | -0.57 to 0.52 |
| `entities` | F10 | +0.7600 to +0.8818 | +0.8249 | 0.0280 | |
| `panel` | reference | -0.0228 to +0.0062 | -0.0029 | 0.0065 | -0.04 to 0.03 |
| `panel` | F8b | +0.9953 to +0.9989 | +0.9977 | 0.0009 | |

Every run finished. On `entities` the holdout is 375 or 380 rows of 75 or 76
entities; the scheme is `grouped` and the interval `bootstrap_groups`. On
`panel` the holdout is 384 rows, the baseline is `previous_value` in all 30
runs of the reference and `training_mean` in all 30 of F8b. The reference's
interval on skill holds zero in 30 of 30 runs on `entities`; M21's in 19.

### The interval and the p-value on `entities` (`grouped_uncertainty.py`)

The lines the script printed. Three intervals on skill: from resampled rows,
from resampled whole entities (both the script's own, 1,000 resamples), and
the tool's (`bootstrap_groups`, 200 resamples under the fast profile). Two
p-values, both the script's own.

| k | base seed | holdout rows | holdout entities | skill | interval, rows | interval, entities | the tool's interval | p, rows | p, entities |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1115 | 380 | 76 | -0.0812 | -0.194 to +0.036 | -0.339 to +0.176 | -0.305 to +0.183 | 0.9251 | 0.7073 |
| 1 | 1215 | 375 | 75 | +0.0782 | -0.023 to +0.180 | -0.138 to +0.275 | -0.143 to +0.292 | 0.0649 | 0.2498 |
| 2 | 1315 | 375 | 75 | +0.0410 | -0.058 to +0.146 | -0.168 to +0.247 | -0.173 to +0.246 | 0.2488 | 0.3806 |
| 3 | 1415 | 380 | 76 | -0.0460 | -0.168 to +0.072 | -0.313 to +0.215 | -0.306 to +0.252 | 0.7822 | 0.6324 |
| 4 | 1515 | 380 | 76 | +0.0042 | -0.095 to +0.105 | -0.196 to +0.218 | -0.179 to +0.183 | 0.4975 | 0.5065 |
| 5 | 1615 | 380 | 76 | -0.0931 | -0.203 to +0.023 | -0.361 to +0.164 | -0.367 to +0.093 | 0.9500 | 0.7433 |
| 6 | 1715 | 380 | 76 | -0.0350 | -0.151 to +0.084 | -0.293 to +0.233 | -0.293 to +0.209 | 0.7093 | 0.6004 |
| 7 | 1815 | 380 | 76 | -0.1737 | -0.273 to -0.079 | -0.365 to +0.019 | -0.356 to +0.017 | 0.9990 | 0.9650 |
| 8 | 1915 | 380 | 76 | +0.0289 | -0.069 to +0.130 | -0.176 to +0.227 | -0.158 to +0.225 | 0.3037 | 0.4196 |
| 9 | 2015 | 380 | 76 | -0.0575 | -0.166 to +0.061 | -0.330 to +0.209 | -0.344 to +0.189 | 0.8462 | 0.6713 |
| 10 | 2115 | 380 | 76 | -0.0591 | -0.157 to +0.039 | -0.256 to +0.147 | -0.284 to +0.162 | 0.8731 | 0.7173 |
| 11 | 2215 | 375 | 75 | -0.0459 | -0.146 to +0.050 | -0.249 to +0.166 | -0.244 to +0.148 | 0.8312 | 0.6763 |
| 12 | 2315 | 375 | 75 | -0.2036 | -0.319 to -0.086 | -0.469 to +0.060 | -0.473 to +0.059 | 1.0000 | 0.9341 |
| 13 | 2415 | 380 | 76 | +0.0323 | -0.072 to +0.138 | -0.182 to +0.243 | -0.144 to +0.216 | 0.3107 | 0.4066 |
| 14 | 2515 | 380 | 76 | -0.0832 | -0.206 to +0.027 | -0.333 to +0.176 | -0.309 to +0.167 | 0.9091 | 0.7163 |
| 15 | 2615 | 375 | 75 | +0.0904 | -0.030 to +0.202 | -0.185 to +0.319 | -0.152 to +0.325 | 0.0579 | 0.2547 |
| 16 | 2715 | 380 | 76 | +0.1526 | +0.058 to +0.252 | -0.054 to +0.338 | -0.063 to +0.332 | 0.0040 | 0.0799 |
| 17 | 2815 | 380 | 76 | -0.0758 | -0.184 to +0.023 | -0.277 to +0.136 | -0.283 to +0.144 | 0.9520 | 0.7902 |
| 18 | 2915 | 380 | 76 | -0.0464 | -0.160 to +0.071 | -0.308 to +0.229 | -0.315 to +0.222 | 0.8012 | 0.6154 |
| 19 | 3015 | 380 | 76 | -0.2599 | -0.378 to -0.148 | -0.503 to +0.016 | -0.495 to +0.050 | 1.0000 | 0.9670 |
| 20 | 3115 | 375 | 75 | -0.2015 | -0.315 to -0.088 | -0.460 to +0.064 | -0.445 to +0.070 | 1.0000 | 0.9520 |
| 21 | 3215 | 380 | 76 | +0.1542 | +0.043 to +0.272 | -0.126 to +0.400 | -0.121 to +0.461 | 0.0030 | 0.1309 |
| 22 | 3315 | 380 | 76 | +0.1424 | +0.016 to +0.257 | -0.127 to +0.408 | -0.102 to +0.408 | 0.0100 | 0.1239 |
| 23 | 3415 | 380 | 76 | +0.0606 | -0.040 to +0.150 | -0.152 to +0.271 | -0.174 to +0.239 | 0.1469 | 0.3037 |
| 24 | 3515 | 380 | 76 | -0.0632 | -0.178 to +0.065 | -0.331 to +0.217 | -0.345 to +0.238 | 0.8581 | 0.6883 |
| 25 | 3615 | 380 | 76 | +0.1681 | +0.054 to +0.285 | -0.085 to +0.438 | -0.097 to +0.414 | 0.0010 | 0.1079 |
| 26 | 3715 | 380 | 76 | +0.1501 | +0.059 to +0.250 | -0.068 to +0.350 | -0.006 to +0.335 | 0.0040 | 0.0729 |
| 27 | 3815 | 380 | 76 | -0.0083 | -0.104 to +0.085 | -0.188 to +0.186 | -0.207 to +0.166 | 0.6084 | 0.5704 |
| 28 | 3915 | 380 | 76 | -0.2398 | -0.360 to -0.126 | -0.471 to +0.014 | -0.450 to +0.046 | 1.0000 | 0.9600 |
| 29 | 4015 | 380 | 76 | -0.1405 | -0.234 to -0.041 | -0.353 to +0.064 | -0.321 to +0.052 | 0.9970 | 0.9291 |

The totals it printed, which are the table of spec section 10.7:

```
30 runs on `entities` under the grouped scheme; skill -0.260 to +0.168; the true skill is 0
  interval from resampled rows holds 0:          19 of 30
  interval from resampled entities holds 0:      30 of 30
  the tool's own interval (bootstrap_groups) holds 0: 30 of 30
  p-value from shuffled rows below 0.05:         5 of 30   (smallest 0.0010)
  p-value from shuffled entities below 0.05:     0 of 30   (smallest 0.0729)
```

The eleven runs in which the interval from rows leaves zero outside are *k* =
7, 12, 16, 19, 20, 21, 22, 25, 26, 28 and 29: six below zero and five above.
The five in which the row shuffle gives *p* below 0.05 are *k* = 16, 21, 22,
25 and 26, the five whose interval from rows lies above zero.
