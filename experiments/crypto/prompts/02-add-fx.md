# BTC daily dataset: add major FX pairs (tables C and D)

Start this session from `E:\CODE\ml-professor`:

```powershell
cd E:\CODE\ml-professor
claude --model opus --effort xhigh "Read E:\CODE\ml-professor\experiments\crypto\prompts\02-add-fx.md and carry it out."
```

Read this whole file before doing anything. If anything in it is wrong,
contradicts what you find, or cannot be done as written, say so plainly with
the evidence and stop. Do not work round it.

---

## What exists, and what this session adds

- `E:\CODE\ml-professor\experiments\crypto` holds a BTC daily dataset built
  by an earlier session from `prompts\01-build-dataset.md` and
  `prompts\01a-phase0-rulings.md`. Read both, then `README.md`,
  `reports\sources.md` and `reports\dataset.md`, before anything else.
  Tables A and B exist, they have been profiled, and their results stand.
- This session adds **one new feature family, major FX pairs from the ECB's
  euro reference rates**, and builds two new tables: **C = every column of
  A plus the FX columns**, and **D = every column of B plus the FX
  columns**. Labels, k and every existing column stay exactly as they are.
- The FX feature set below is fixed in advance. It will be profiled once and
  compared with A and B on the same holdout days. Do not add, drop or tune
  FX features to improve anything. If one cannot be built as written,
  report it.

## Rules for this session

All the rules of `01-build-dataset.md` hold, with these additions:

- Write only under `E:\CODE\ml-professor\experiments\crypto`. Never edit
  anything under `prompts\`.
- **Write every file through PowerShell, as the earlier session did.** The
  kit treats a session started this way as a build session, so the
  file-editing tools would run the library's gate after every edit. Keep
  comments that name locked folders out of commands; the guard refused one
  such command before.
- **Network access is allowed only in the new `scripts\fetch_ecb.py`.** Do
  not run `fetch_bitstamp.py`, `fetch_coinmetrics.py` or
  `fetch_binance.py`, and do not change any existing file under
  `data\raw\`. New days would change tables A and B and their holdouts.
- **Fetch the ECB rates through 2026-11-27 and no later**, the last day of
  the existing tables.
- **Tables A and B must not change by a single byte.** Before you change
  any script, record the SHA-256 of every file in `data\profiler\` and
  `data\built\`. `a_short_up_7d.csv` and `a_short_down_7d.csv` were written
  by the project's lead (they are A's rows from 2019-12-09). Leave them
  alone, and do not regenerate them.
- Build with `--k 0.36` from now on (ruling 5 fixed k at 0.36).
- Do not run ML Professor.

## Source

| Source | Used for | Starting point, to verify |
|---|---|---|
| ECB euro foreign exchange reference rates | daily fixings of USD, JPY, GBP, CHF, AUD, CAD and SEK against EUR | the ECB Data Portal API, `https://data-api.ecb.europa.eu/service/data/EXR/D.USD+JPY+GBP+CHF+AUD+CAD+SEK.EUR.SP00.A` with `format=csvdata`; or the historical file `https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip` |

These are starting points, not facts. Verify them, choose one, say which and
why, and use no other source and no API key. The ECB publishes these rates at
around 16:00 CET on each working day (TARGET closing days have none).

---

## Phase 0: fetch, inspect, report, then stop

1. Record the SHA-256 of every file in `data\profiler\` and `data\built\`.
2. Write `scripts\fetch_ecb.py` and run it. Save the responses as received
   under `data\raw\ecb\`. Add each file to `data\raw\manifest.json` (URL with
   parameters, UTC time fetched, row count, SHA-256) without changing any
   existing entry.
3. Write `reports\fx_sources.md`, establishing by measurement:
   - the SHA-256 values from step 1;
   - for each currency: first and last fixing date, the number of
     fixings, and every weekday with no fixing, by year. Say which of these
     are TARGET closing days and list any that are not;
   - the publication time: what the source shows about when a day's
     fixing is released. Confirm that a fixing for date F is known before
     00:00 UTC on F+1, or report that it is not;
   - the derived USD pairs, each as defined under "Phase 1": the ten
     largest absolute daily log moves per pair, with their dates;
   - whether the derived dollar index's daily log return agrees with the
     direction of USDJPY, USDCHF and USDCAD, and the opposite of EURUSD and
     GBPUSD, on most days (a sign check on the construction);
   - the first row tables C and D would have under rule 8 of
     `01-build-dataset.md`. The expected answer is table A's and table B's
     own first rows, 2012-12-17 and 2019-12-09. If either is later, say why
     and stop;
   - anything in this prompt you think is wrong.
4. **Stop. Write nothing else, and wait for the user.**

---

## Phase 1: after the user approves Phase 0

### FX series

From the EUR rates, with every series a fixing on the same date:

- `eurusd` = USD per EUR, as published.
- `usdjpy` = JPY per EUR / USD per EUR.
- `gbpusd` = USD per EUR / GBP per EUR.
- `usdchf` = CHF per EUR / USD per EUR.
- `audusd` = USD per EUR / AUD per EUR.
- `usdcad` = CAD per EUR / USD per EUR.
- `dxy`, the ICE dollar index rebuilt from the crosses, in logs:
  ln(dxy) = −0.576 ln(eurusd) + 0.136 ln(usdjpy) − 0.119 ln(gbpusd) +
  0.091 ln(usdcad) + 0.042 ln(usdsek) + 0.036 ln(usdchf), where `usdsek` =
  SEK per EUR / USD per EUR. Leave out the index's constant: only changes of
  ln(dxy) are used. `usdsek` is used only inside `dxy` and has no columns
  of its own.

A fixing date with any of the seven currencies missing has no value for
any series. Report how many such dates there are.

### Point-in-time rule

- The fixing for date F is known at the close of BTC day F (00:00 UTC on
  F+1).
- Row t uses the latest fixing date F that is on or before t. On a weekend
  or a TARGET closing day, that is the last working day before it.
- Every FX feature is computed on the sequence of fixing dates, and then
  carried to each BTC row by that rule. A window of n fixings means the n
  fixing dates ending with F, never calendar days.
- Only trailing statistics, as in `01-build-dataset.md`.

### FX features

For each series s in `eurusd`, `usdjpy`, `gbpusd`, `usdchf`, `audusd`,
`usdcad` and `dxy`, name the columns `fx_` followed by s, then:

- `_ret_1`, `_ret_5`, `_ret_20`, `_ret_60`: the log change of s over the
  last 1, 5, 20 and 60 fixings, ending at F;
- `_vol_20`: the standard deviation (ddof 1) of s's daily log changes over
  the last 20 fixings;
- `_vol_z_250`: the trailing 250-fixing z-score of ln(`_vol_20`).

That gives 42 columns. One more column:

- `btc_dxy_corr_60`: the correlation over the last 60 fixing dates ending
  at F between the change of ln(dxy) from the previous fixing date F′ to F,
  and BTC's log return from Bitstamp's close of F′ to its close of F. A
  fixing date whose BTC close, on either end, is missing is left out of the
  window. The 90% rule of "Missing data" applies.

That gives 43 FX columns in all. Table C has A's 99 columns plus these 43,
making 142; table D has B's 119 plus these 43, making 162. A value exists
only when its window is complete, under the missing-data rules of
`01a-phase0-rulings.md`, applied to the fixing sequence.

### Files

- `data\profiler\c_up_7d.csv`, `c_down_7d.csv`, `d_up_7d.csv` and
  `d_down_7d.csv`. They follow rule 9: `date`, then the features, then
  exactly one target. The dates and targets of C equal A's, and those of D
  equal B's.
- `data\built\btc_daily_full.csv` may gain the FX columns, but **only if**
  that leaves the files recorded in Phase 0 byte-identical. If it would
  not, write the FX columns to a new `data\built\fx_daily.csv` instead, and
  say which you did.
- `README.md`: add the FX family to the feature catalogue, and record this
  prompt's rules as rulings 22 onwards, with reasons.

### Checks: extend `scripts\check_dataset.py`

Keep checks 1 to 7 and extend them to tables C and D:

1. **labels**: C's targets equal A's, and D's equal B's, row for row.
2. **point_in_time**: the cut copies now include the ECB files, cut to the
   fixings known at each date's close. The 40 dates must include at least
   one Saturday, one Sunday, one TARGET closing day that falls on a
   weekday, and the first rows of C and D. Compare every row of every cut
   build with the full build, for tables C and D as well as A and B.
3. **one_target**: C's columns are A's followed by the 43 FX columns, in
   the same order in the up and down files, and likewise D's against B's.
   A's columns in C equal A's values; B's columns in D equal B's values;
   the FX columns in C and D equal each other on their common dates.
4. **rows**, 5. **finite**, 6. **deterministic**, 7. **level_screen**: as
   before, over all four tables.
8. **ab_unchanged** (new): every file recorded in Phase 0 step 1 still has
   the same SHA-256, and every raw file other than those under
   `data\raw\ecb\` still matches the manifest entry it had before this
   session.

Exit 0 only when checks 1 to 6 and 8 all pass. Explain or fix every FX
feature that the level screen puts above 10%, without changing the feature
set. If a feature truly cannot be kept, report it and stop.

### Phase 1 report: `reports\fx.md`

- Tables C and D: rows, first and last dates, column counts.
- The FX catalogue: name, formula in words, window in fixings, lag rule.
- The full output of `check_dataset.py`.
- Anything in this prompt you think is wrong, and anything you could not
  do.

## Done

Run from `E:\CODE\ml-professor`, both of these exit 0, and the second prints
PASS for checks 1 to 6 and 8:

```powershell
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe experiments\crypto\scripts\build_dataset.py --k 0.36
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe experiments\crypto\scripts\check_dataset.py
```

Then stop. Do not run ML Professor. That is a separate step, taken by the
user.
