# BTC daily dataset for ML Professor: build session

Start this session from `E:\CODE\ml-professor`:

```powershell
cd E:\CODE\ml-professor
claude --model opus "Read E:\CODE\ml-professor\experiments\crypto\prompts\01-build-dataset.md and carry it out."
```

Read this whole file before doing anything. If anything in it is wrong,
contradicts what you find, or cannot be done as written, say so plainly with
the evidence and stop. Do not work round it.

---

## What this is, and the rules for this session

- `E:\CODE\ml-professor` is ML Professor, a Python library that measures how
  predictable a target column is. **This session does not work on the
  library.** It builds a dataset the library will be run on later, in a new
  folder: `E:\CODE\ml-professor\experiments\crypto`.
- The root `CLAUDE.md` is written for the library's autonomous build. It
  limits writes to `mlprofessor\`, `examples\`, `requirements.txt` and
  `README.md`, and forbids network access at run time. Those two rules are
  for the library build. For this session the repo owner sets these instead:
  - write only under `E:\CODE\ml-professor\experiments\crypto`;
  - never write anything outside it: not `mlprofessor\`, `verify\`,
    `tests\`, `plan\`, `loop\`, `.claude\`, `docs\`, `examples\`, and not the
    root `spec.md`, `CLAUDE.md`, `README.md`, `requirements.txt` or
    `.gitignore`;
  - network access is allowed only inside this experiment's three fetch
    scripts.

  Everything else in the root `CLAUDE.md` still holds. In particular: no
  git, of any kind.
- Never edit anything under `experiments\crypto\prompts\`.
- If a hook or the guard blocks something this prompt asks for, stop and
  report it.

## The question the dataset serves

One row per UTC calendar day t. Two binary targets, which will be profiled
separately: `up_7d`, the next 7 days were a rally, and `down_7d`, they were a
sell-off, each defined under "Settled choices". Every feature on row t must
use only what was known at the close of day t.

## Environment

- Interpreter, for every script: `E:\CODE\ml-professor\mlprof_env\Scripts\python.exe`
  (Python 3.12, pandas 3.1.1, numpy 2.5.4, scipy 1.18.1). Do not install
  anything into it and do not create another environment. Downloads use the
  standard library (`urllib.request`, `json`); tables use pandas and numpy.
- Windows and PowerShell. Scripts are run from `E:\CODE\ml-professor`, and
  resolve every path from their own location, never from the working folder.
- The root `.gitignore` ignores folders named `lib`, `target`, `downloads`,
  `build`, `dist`, `var`, `parts` and `cover`, and every `*.log` file,
  anywhere in the repo. Use none of those names.
- No `__init__.py` anywhere under `experiments\`: the repo will be packaged
  for PyPI and this folder must never look like a package. No file named
  `test_*.py` or `*_test.py`, and no `conftest.py` or `pytest.ini`: the
  library's test run must never collect anything from here.

## Layout

```
experiments\crypto\
  README.md                      what this is, how to rebuild, the settled choices, the feature catalogue
  .gitignore                     one line: data/
  scripts\fetch_bitstamp.py
  scripts\fetch_coinmetrics.py
  scripts\fetch_binance.py
  scripts\labels.py              used by build_dataset.py
  scripts\features.py            used by build_dataset.py
  scripts\build_dataset.py       raw data to built tables; options --raw FOLDER, --out FOLDER, --k VALUE
  scripts\check_dataset.py       independent checks; exit 0 pass, 1 fail
  reports\sources.md             Phase 0 report
  reports\dataset.md             Phase 1 report
  data\raw\                      downloads as received, and manifest.json
  data\built\btc_daily_full.csv  every column, labels included
  data\profiler\a_up_7d.csv
  data\profiler\a_down_7d.csv
  data\profiler\b_up_7d.csv
  data\profiler\b_down_7d.csv
```

`build_dataset.py` with no options reads `data\raw\`, writes `data\built\`
and `data\profiler\`, and computes k by rule 5. `--raw`, `--out` and `--k`
let the checks run it on other inputs, with k fixed.

## Sources

| Source | Used for | Starting point, to verify |
|---|---|---|
| Bitstamp BTC/USD | the price of record: labels and table A | `https://www.bitstamp.net/api/v2/ohlc/btcusd/` with `step=86400`, `limit=1000`, `start` and `end` in Unix seconds; public, no key |
| Coin Metrics Community | PriceUSD for the cross-check; free network metrics for table A | root `https://community-api.coinmetrics.io/v4`, no key; at most 10 requests per 6 seconds |
| Binance spot BTCUSDT | daily klines with quote volume, trade count and taker-buy volume: table B | `https://api.binance.com/api/v3/klines` with `symbol=BTCUSDT`, `interval=1d`, `limit=1000`, `startTime` in milliseconds |
| Binance USD-M perpetual BTCUSDT | daily klines: table B | `https://fapi.binance.com/fapi/v1/klines`, same parameters |
| Binance funding, BTCUSDT perpetual | every funding print: table B | `https://fapi.binance.com/fapi/v1/fundingRate` with `symbol=BTCUSDT`, `limit=1000`, `startTime` |

These endpoints are where to start, not facts. Verify each one answers as
described. If one does not, report what it does instead and stop. Use no
other source and no API key.

"Until today" means through the last UTC day that had ended when the fetch
ran. Never keep a candle or a value for a day that has not ended.

---

## Phase 0: fetch, inspect, report, then stop

1. Run the interpreter and report its version and the pandas and numpy
   versions.
2. Write the three fetch scripts and run them. Save every response as
   received under `data\raw\`, and say how pages are stored. Write
   `data\raw\manifest.json`: for each file, the URL with its parameters, the
   UTC time it was fetched, its row count and its SHA-256. Write
   `experiments\crypto\.gitignore` with the one line `data/`.
3. Write `reports\sources.md`. Establish each of these by measurement:
   - for each source: first and last date, row count, the fields returned
     and their types, and every missing calendar day (the dates when there
     are 30 or fewer, otherwise counts by year);
   - Bitstamp days with zero volume, and days with open, high, low and close
     all equal, by year;
   - the day convention of Coin Metrics daily rows: does the row dated D
     carry the price at the end of D? Compare PriceUSD with Bitstamp's close
     of D and of the day before, and report the median absolute difference
     for each;
   - Bitstamp's close against Coin Metrics PriceUSD on the matched day: the
     days more than 1% apart, by year, and the ten largest with both values;
   - every free Coin Metrics metric for btc at daily frequency, with its
     first date. Propose a set for table A covering activity (addresses,
     transactions), security (hash rate, difficulty), fees, supply, and
     valuation (market cap against realised cap, and against adjusted
     transfer value, if those are free), choosing for each concept the
     metric with the longest history. Say which of these concepts are not
     free;
   - the halving dates 2012-11-28, 2016-07-09, 2020-05-11 and 2024-04-20:
     confirm each from the data if a free metric shows daily issuance or
     block height (issuance halves on those dates); otherwise say they are
     unverified;
   - Binance: the first date of the spot klines, of the perpetual klines and
     of the funding prints; the funding print times and their offset from
     the whole hour; prints per day, and every day with a number other than
     three;
   - the first row you propose for table A and for table B under rule 8
     below, the reasons, and the rows each table would then have;
   - how you propose to treat each kind of missing day you found, using the
     rules under "Missing data", or why none fits.
4. **Stop. Write nothing else, and wait for the user.**

---

## Phase 1: after the user approves Phase 0

### Settled choices

Copy these into `README.md` as written.

1. One row per UTC calendar day. C_t is Bitstamp's close of day t; O, H, L
   and V are its open, high, low and volume (in BTC).
2. σ30_t is the standard deviation (ddof 1) of the 30 daily log returns
   ending with day t.
3. z_t = ln(C_{t+7} / C_t) / (σ30_t × √7), where C_{t+7} is the close seven
   calendar days later.
4. Classes: `UP` when z_t is above +k; `DOWN` when z_t is below −k; `RANGE`
   otherwise.
5. k is the one-third quantile of |z_t| over the earliest 75% of table A's
   rows that have a z_t, in date order, rounded to two decimals, then fixed.
   Report k, and the class shares in that earliest 75%, in the latest 25%,
   and by year.
6. `up_7d` is 1 when the class is UP and 0 otherwise; `down_7d` is 1 when it
   is DOWN and 0 otherwise; both are written as the integers 0 and 1. Both
   are blank on the last seven days (no C_{t+7}) and wherever C_t, C_{t+7}
   or σ30_t is missing, and nowhere else.
7. Two tables. Table A holds the price, volume, calendar and on-chain
   features. Table B holds every column of table A plus the Binance
   features. A column that is in both is computed identically.
8. A table's first row is the first day on which every one of its feature
   columns has a value. Its last row is the last completed UTC day.
9. A profiler file holds `date` (ISO `YYYY-MM-DD` text) first, then the
   table's feature columns, then exactly one target. z_t, the forward
   return and the class are written only to `data\built\btc_daily_full.csv`,
   never to a profiler file.

### Point-in-time rules, for every feature

- A feature on row t uses only values known at the close of day t, which is
  00:00 UTC on day t+1.
- Bitstamp and Binance daily values for day t are known at its close.
- Coin Metrics network metrics are used one day late: row t takes the value
  for day t−1. PriceUSD is used for the cross-check only, never as a
  feature.
- Funding: floor each print's time to the hour first. Row t uses the prints
  at 00:00, 08:00 and 16:00 UTC of day t. The 00:00 print of day t+1 belongs
  to row t+1.
- A rolling window ends at row t, inclusive. A rolling value exists only
  when its window is complete, apart from what "Missing data" allows.
- Trailing statistics only: no centred window, no two-sided filter, and no
  mean, standard deviation, rank or quantile taken over the whole sample in
  any feature. The only constant fixed from the data is k (rule 5).
- No raw level of anything that trends (price, volume, hash rate, supply,
  market cap, addresses, trade count), and no time index or date ordinal.
  Each is turned into a return, a ratio, a growth rate or a trailing
  z-score.
- A trailing z-score of x over n days is (x_t − mean) / standard deviation,
  both taken over the n days ending with day t.

### Missing data

- A missing Bitstamp day stays missing. It is never filled from another
  source.
- Isolated gaps, at most two days in a row: a rolling value is computed when
  at least 90% of its window's days are present; a return over n days needs
  both of its ends.
- A longer gap: the days inside it, and every row whose window cannot meet
  the 90% rule because of it, have no value. If this happens after a
  table's first row, report the rows affected and stop for the user.
- The same rules apply to Coin Metrics and Binance, each on its own days.

### Features

Name every column exactly as below. n is a window in days.

**Returns (A).**
- `ret_n` = ln(C_t / C_{t−n}), for n = 1, 2, 3, 5, 7, 14, 21, 30, 60, 90, 180, 365.
- `zret_n` = ret_n / (σ30_t × √n), for n = 7, 14, 30.
- `past_z_7` = ln(C_t / C_{t−7}) / (σ30_{t−7} × √7), and `past_class_7` =
  1, 0 or −1 by the rule 4 thresholds with the same k: the label's own
  definition applied to the seven days just ended.

**Volatility (A).**
- `vol_cc_n`: the standard deviation of daily log returns, for n = 7, 14, 30, 90, 365.
- `vol_park_n`, `vol_gk_n`, `vol_rs_n`: the Parkinson, Garman–Klass and
  Rogers–Satchell daily variances averaged over the window, then
  square-rooted, for n = 7, 30.
- `vol_yz_30`: Yang–Zhang over 30 days.
- `vol_ratio_7_30`, `vol_ratio_30_90`, `vol_ratio_30_365`: ln(vol_cc of the
  shorter window / vol_cc of the longer).
- `vol_of_vol_30`: the standard deviation over 30 days of ln(vol_cc_7).
- `vol_z_365`: the trailing 365-day z-score of ln(vol_cc_30).

**Range and candle (A).**
- `hl_range` = ln(H_t / L_t), and `hl_range_mean_7`.
- `body` = (C − O) / (H − L); `clv` = (2C − H − L) / (H − L);
  `upper_wick` = (H − max(O, C)) / (H − L); `lower_wick` = (min(O, C) − L) / (H − L).
  Each is 0 when H equals L; count those days in the report.
- `body_mean_7`, `clv_mean_7`.

**Trend (A).**
- `dist_sma_n` = ln(C_t / SMA_n), for n = 7, 20, 50, 100, 200.
- `sma_50_200` = ln(SMA_50 / SMA_200).
- `macd` = (EMA_12 − EMA_26) / C_t; `macd_signal` = EMA_9 of `macd`;
  `macd_hist` = `macd` − `macd_signal`. Each EMA starts from the simple mean
  of its first window, so no value depends on a later row, and counts as
  complete after three times its span.
- `trend_t_n`: the t-statistic of the least-squares slope of ln C on day
  number over the last n days, for n = 30, 90.
- `dd_ath` = ln(C_t / the highest C up to and including t);
  `days_since_ath` = ln(1 + days since that highest C).
- `dist_high_365` = ln(C_t / the highest H over 365 days);
  `dist_low_365` = ln(C_t / the lowest L over 365 days).
- `adx_14` (Wilder) and `aroon_osc_25`.

**Oscillators (A).**
- `rsi_14` and `rsi_30` (Wilder smoothing), `stoch_k_14`, `stoch_d_14_3`,
  `bb_pctb_20`, `bb_width_20` (4 × the 20-day standard deviation of C, over
  SMA_20), `cci_20`.

**Volume and liquidity (A).** Dollar volume DV_t = V_t × C_t.
- `lvol_z_30`, `lvol_z_90`: trailing z-scores of ln(DV).
- `lvol_ratio_7_30` = ln(mean DV over 7 days / mean DV over 30 days).
- `amihud_z_365`: the trailing 365-day z-score of ln(the 30-day mean of
  |daily log return| / DV).
- `obv_slope_30`: the 30-day sum of sign(daily return) × V, over the 30-day
  sum of V.
- `cmf_20`: Chaikin money flow over 20 days.
- `up_vol_share_30`: the volume of up days over all volume, across 30 days.

**Return distribution (A).**
- `skew_n`, `kurt_n` and `acf1_n` (lag-1 autocorrelation) of daily log
  returns, for n = 30, 90.
- `up_share_n`: the share of days with a positive return, for n = 14, 30.
- `maxdd_30`: the largest fall from a peak to a later trough of C within the
  30 days, as a log ratio.

**Calendar and cycle (A).**
- `dow`, 0 for Monday to 6 for Sunday; `month`, 1 to 12.
- `days_since_halving` = ln(1 + days since the latest halving on or before
  day t; before the first halving, days since 2009-01-03).
- `halving_phase` = days since that halving / 1461. Note in the catalogue
  that this column can act as an era marker.

**On-chain (A).** For each metric the user approves in Phase 0, using the
value for day t−1, name columns `oc_` followed by the metric's Coin Metrics
id in lower case:
- that name followed by `_g7` and `_g30`: log growth over 7 and 30 days;
- that name followed by `_z365`: the trailing 365-day z-score of the log of
  the metric.
- If market cap and realised cap are both free: `mvrv` = ln(market cap /
  realised cap), and `mvrv_z365`.
- If adjusted transfer value is free: `nvt_30` = ln(market cap / its 30-day
  mean adjusted transfer value in USD), and `nvt_z365`.

**Binance (B only).**
- `fund_1d`: the sum of the day's three prints; `fund_7` and `fund_30`:
  sums over the window; `fund_z_90`: the trailing 90-day z-score of
  `fund_1d`.
- `basis` = ln(perpetual close / spot close); `basis_mean_7`; `basis_z_90`.
- `usdt_prem` = ln(Binance spot close / Bitstamp close); `usdt_prem_z_90`.
- `perp_spot_lvol` = ln(perpetual quote volume / spot quote volume), and
  `perp_spot_lvol_z_90`.
- `taker_share_spot` = taker-buy quote volume / quote volume, with
  `taker_share_spot_mean_7` and `taker_share_spot_z_90`; the same three for
  the perpetual, named `taker_share_perp`, `taker_share_perp_mean_7` and
  `taker_share_perp_z_90`.
- `trades_z_90`: the trailing 90-day z-score of ln(spot trade count);
  `avg_trade_z_90`: the same of ln(spot quote volume / spot trade count).
- `bn_lvol_z_30`: the trailing 30-day z-score of ln(spot quote volume).

You may add a feature only if it meets every rule above. List it in the
catalogue as added by you, with the reason. Never drop a listed feature
without reporting why.

### Checks: `scripts\check_dataset.py`

The checks must not import `labels.py`, `features.py` or `build_dataset.py`.
They read the raw and built files, and run `build_dataset.py` only as a
separate process. Print one line per check, PASS or FAIL with the reason.
Exit 0 only when checks 1 to 6 all pass.

1. **labels**: recompute z_t, the class, `up_7d` and `down_7d` from the raw
   Bitstamp file and k. They equal the built values on every row of every
   file, and they are blank exactly where rule 6 says.
2. **point_in_time**: choose 40 dates with a fixed seed, always including
   table A's first row, table B's first row, the last row with a target and
   the last row. For each date: copy every raw source, cut to what was known
   at that day's close (with the Coin Metrics one-day lag and the funding
   rule applied), into a temporary folder; run `build_dataset.py --raw` on
   it with `--k` set to the full build's k; compare that date's row with the
   full build. Every feature column must agree within 1e-9, absolute or
   relative. This check is the reason the dataset can be trusted. ML
   Professor cannot detect look-ahead inside a caller's own features.
3. **one_target**: each profiler file has `date` first, then the features,
   then exactly one target with the expected name. No column holds the
   forward return, z_t or the class (`past_class_7` and `past_z_7` are
   features, not labels). `a_up_7d.csv` and `a_down_7d.csv` are identical
   apart from the target, and the same holds for `b_up_7d.csv` and
   `b_down_7d.csv`. Table A's feature columns are all in table B, with equal
   values on their common dates.
4. **rows**: in each table the dates rise strictly, one per calendar day,
   with no day missing from the first row to the last, written as ISO text.
   The last row is the last day completed at fetch time, read from the
   manifest. Targets are blank on the last seven rows and wherever rule 6
   says.
5. **finite**: no infinite value anywhere. Every missing feature value is
   listed by column and date, and is allowed only under "Missing data".
6. **deterministic**: build twice into two temporary folders; the output
   files are byte-identical.
7. **level_screen**, reported and not a pass or fail: for each feature, the
   share of its values in the latest 25% of table A's rows that lie outside
   its range over the earliest 75%. Explain or fix every feature above 10%.

### Phase 1 report: `reports\dataset.md`

- The rows of each table: first and last date; rows with a target, and rows
  without one and why.
- k, and the class shares from rule 5. The shares of `up_7d` and `down_7d`
  by year.
- The feature catalogue: name, formula in words, window, source,
  availability lag, table. The same catalogue goes in `README.md`.
- The full output of `check_dataset.py`, and the level screen.
- Anything in this prompt you think is wrong, and anything you could not do.

## Done

Run from `E:\CODE\ml-professor`, both of these exit 0, and the second prints
PASS for checks 1 to 6:

```powershell
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe experiments\crypto\scripts\build_dataset.py
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe experiments\crypto\scripts\check_dataset.py
```

Then stop. Do not run ML Professor. That is a separate step, taken by the
user.
