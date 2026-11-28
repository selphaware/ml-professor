# Phase 1 of the FX columns: tables C and D

Built on 2026-11-28 with `--k 0.36`, from `data/raw/` as fetched:
- the ECB rates on 2026-11-28 between 12:24 and 12:31 UTC;
- the other three sources between 09:18 and 09:21 UTC;
- everything through 2026-11-27.

The rulings are in `README.md`: 21 to 27 answer Phase 0, and 28 to 34 are
the rules of `02-add-fx.md` (section 5, point 2). ML Professor was not run.

## 1. Tables C and D

| | Table C | Table D |
|---|---|---|
| Rows | table A's | table B's |
| First row | 2012-12-17 | 2019-12-09 |
| Last row | 2026-11-27 | 2026-11-27 |
| Rows | 5,094 | 2,546 |
| Feature columns | 142: A's 99, then the 43 FX columns | 162: B's 119, then the 43 FX columns |
| Columns of each profiler file | 144: `date`, the features, one target | 164 |
| Rows with a target | 5,054 | 2,539 |
| Rows without one | 40, A's own (see `reports/dataset.md`, section 1) | 7, the last seven |
| Blank feature cells | 1,015 on 49 rows: A's 1,014 and `btc_dxy_corr_60` on 2015-01-08 (ruling 23) | none |
| Rows that use their own day's fixing | 3,568 | 1,786 |
| Rows that carry an earlier fixing (1, 2, 3 or 4 days back) | 745, 732, 31, 18 | 370, 365, 16, 9 |

- **The targets are A's and B's.** C's dates and targets equal A's, row for
  row, and D's equal B's (check 1). Each line of a C file begins with the
  same line of the A file, character for character, minus its target; the
  same holds for D and B (check 3).
- **The first rows.** Rule 8 alone gives C 2012-12-01, as it does A, and
  ruling 12 moves both to 2012-12-17. D's rule-8 day is B's own,
  2019-12-09. The FX columns all have values from 2012-01-30, so they bind
  neither table.
- **Files written:**
  - `data/profiler/c_up_7d.csv` and `c_down_7d.csv` (14.3 MB each);
  - `data/profiler/d_up_7d.csv` and `d_down_7d.csv` (8.2 MB each);
  - `data/built/fx_daily.csv` (`date`, `fx_day` and the 43 FX columns on
    table A's rows, 4.7 MB; ruling 24);
  - `data/built/build_info_cd.json`: the rows and blank cells of C and D,
    the rule-8 days, the rows by days since their fixing, and the ECB file
    read. That file has 7,146 fixing dates, 1999-01-04 to 2026-11-27; 434
    holiday rows on 62 dates are dropped, and no date lacks a currency.
- **Tables A and B did not change** (check 8).
  - The build with `--k 0.36` writes `btc_daily_full.csv`,
    `build_info.json` and the four `a_`/`b_` files byte for byte as
    recorded in Phase 0.
  - `build_info.json` stays the same because the build now records rule 5
    whether or not k is given (ruling 22).

## 2. The FX catalogue

The window counts fixing dates, not days. F is the latest fixing date on or
before row t: on a weekend or a TARGET closing day, the last working day
before it. The lag rule is the same for every FX column (ruling 32): the
fixing of F is known at the close of BTC day F (published at about 16:00
CET, measured at 14:56 UTC on 2026-11-27), and row t takes F's values.

| Series s | Built from the EUR rates of one fixing date |
|---|---|
| `eurusd` | USD per EUR, as published |
| `usdjpy` | JPY per EUR / USD per EUR |
| `gbpusd` | USD per EUR / GBP per EUR |
| `usdchf` | CHF per EUR / USD per EUR |
| `audusd` | USD per EUR / AUD per EUR |
| `usdcad` | CAD per EUR / USD per EUR |
| `dxy` | ln dxy = −0.576 ln eurusd + 0.136 ln usdjpy − 0.119 ln gbpusd + 0.091 ln usdcad + 0.042 ln usdsek + 0.036 ln usdchf, with usdsek = SEK per EUR / USD per EUR and no constant: ICE's weights applied to the 14:15 CET fixings. usdsek has no columns of its own |

| Column (for each s) | Formula in words | Window, in fixings | Lag rule | Count |
|---|---|---|---|---|
| `fx_s_ret_1`, `_ret_5`, `_ret_20`, `_ret_60` | log change of s over the last 1, 5, 20 or 60 fixings, ending at F | its two ends: F and n fixings before | fixing of F | 28 |
| `fx_s_vol_20` | standard deviation (ddof 1) of s's daily log changes over the last 20 fixings | 20 changes, 21 fixings | fixing of F | 7 |
| `fx_s_vol_z_250` | trailing 250-fixing z-score of ln(`fx_s_vol_20`); needs F's own value and 225 of the 250 | 250 values, 270 fixings | fixing of F | 7 |
| `btc_dxy_corr_60` | correlation over the 60 fixing dates ending at F between the change of ln(dxy) from the previous fixing date F′ to F and BTC's log return from Bitstamp's close of F′ to its close of F. A date with either BTC close missing is left out, and 54 of 60 are needed. Blank on a day inside a longer Bitstamp gap | 60 pairs, 61 fixings | fixing of F, and Bitstamp's closes of F′ and F | 1 |

In table order: for `eurusd`, `usdjpy`, `gbpusd`, `usdchf`, `audusd`,
`usdcad` and `dxy` in turn, the six columns `fx_s_ret_1` to
`fx_s_vol_z_250`; then `btc_dxy_corr_60`. That is 43 columns.

**`btc_dxy_corr_60` correlates returns that are not taken at the same
time** (ruling 27). The dollar index changes from 14:15 CET to 14:15 CET,
and BTC from 00:00 UTC to 00:00 UTC. On consecutive weekdays the two
overlap for about 13 of their 24 hours, so the column reads weaker than a
correlation of returns taken at the same time would.

The same catalogue is in `README.md`, under "FX columns (tables C and D)".

## 3. The checks

Here is the full output of `check_dataset.py`, run from
`E:\CODE\ml-professor` right after `build_dataset.py --k 0.36`. Both exit 0,
and checks 1 to 6 and 8 print PASS. The run took 36 seconds.

```
PASS 1 labels: k = 0.36 (rule 5 recomputed: 0.36); z, class, up_7d, down_7d recomputed from raw Bitstamp agree on 5094 rows of btc_daily_full.csv and every row of the 8 profiler files; C's targets equal A's and D's equal B's, row for row; blank on 40 rows, exactly where C_t, C_t+7 or sigma30_t is missing
PASS 2 point_in_time: 40 dates (seed 20261128; the first rows of tables A to D, the last row with a target, the last row, and 2015-01-06, 2015-01-07, 2015-01-08, 2015-01-09, 2021-02-25, 2024-08-16, 2025-05-03, 2026-04-04, 2026-04-05, 2026-04-06 among them): each rebuilt from raw data cut at its close, the ECB file cut to fixing dates on or before it, with k = 0.36 (--k-unchecked: rule 5 on the cut history gives another k on 29 of them); its row and every earlier row agree in all 162 feature columns of tables A to D within 1e-09, and in cm_day and fx_day; every row of fx_daily.csv takes the latest fixing date on or before it
    dates: 2012-12-17, 2013-05-05, 2015-01-06, 2015-01-07, 2015-01-08, 2015-01-09, 2016-04-25, 2017-01-04, 2017-11-23, 2018-01-31, 2018-02-16, 2018-09-26, 2018-10-30, 2019-06-10, 2019-12-09, 2020-01-04, 2021-02-25, 2021-06-28, 2021-07-15, 2021-11-24, 2022-04-08, 2022-05-05, 2022-10-24, 2023-02-05, 2023-08-08, 2024-06-29, 2024-08-16, 2024-12-03, 2025-01-30, 2025-05-03, 2025-07-18, 2025-11-22, 2025-11-26, 2026-03-26, 2026-04-04, 2026-04-05, 2026-04-06, 2026-06-20, 2026-11-20, 2026-11-27
    among them: Saturday 2020-01-04, 2024-06-29, 2025-05-03, 2025-11-22, 2026-04-04, 2026-06-20; Sunday 2013-05-05, 2023-02-05, 2026-04-05; weekday with no fixing (a TARGET closing day) 2026-04-06
    rows compared: built/btc_daily_full.csv 124067, built/fx_daily.csv 124067, profiler/c_up_7d.csv 124067, profiler/d_up_7d.csv 39778
PASS 3 one_target: eight profiler files: date, then 99 (A), 119 (B), 142 (C) or 162 (D) features, then one target; C is A's columns followed by the 43 FX columns, D is B's followed by the same, in the same order in the up and down files; no column holds or equals the forward return, z or the class; up and down files identical apart from the target; A's columns all in B with equal values on 2546 common dates; C's lines begin with A's features as written, D's with B's; the FX columns equal in C, D and fx_daily.csv on 2546 common dates
PASS 4 rows: dates strictly rising by one day, ISO text, no day missing; tables A and C 2012-12-17..2026-11-27 (5094 and 5094 rows), tables B and D 2019-12-09..2026-11-27 (2546 and 2546 rows), fx_daily.csv as table A; last row is the manifest's last completed day, and no ECB fixing is later; targets blank on the last seven rows (and elsewhere exactly as rule 6, check 1); all 202 raw files match their SHA-256
PASS 5 finite: no infinite value in any file; every blank feature cell lies within its column's reach of a missing source day, or before its window lies inside its sources' history; an FX column's reach is counted in fixings, and 0 fixing dates lack a currency (missing days: bitstamp: 2011-08-23, 2011-08-26..2011-08-29 (4), 2011-08-31, 2011-09-03, 2011-09-05, 2011-09-09..2011-09-12 (4), 2011-09-30..2011-10-02 (3), 2011-10-15..2011-10-16 (2), 2011-10-18..2011-10-19 (2), 2011-10-22..2011-10-23 (2), 2011-10-27, 2011-11-02..2011-11-03 (2), 2011-11-07, 2011-11-23, 2011-11-27, 2011-12-04, 2011-12-09, 2011-12-17, 2015-01-06..2015-01-08 (3))
    table A: 1014 blank cells:
      ret_1: 2015-01-06..2015-01-09 (4)
      ret_2: 2015-01-06..2015-01-10 (5)
      ret_3: 2015-01-06..2015-01-11 (6)
      ret_5: 2015-01-06..2015-01-08 (3), 2015-01-11..2015-01-13 (3)
      ret_7: 2015-01-06..2015-01-08 (3), 2015-01-13..2015-01-15 (3)
      ret_14: 2015-01-06..2015-01-08 (3), 2015-01-20..2015-01-22 (3)
      ret_21: 2015-01-06..2015-01-08 (3), 2015-01-27..2015-01-29 (3)
      ret_30: 2015-01-06..2015-01-08 (3), 2015-02-05..2015-02-07 (3)
      ret_60: 2015-01-06..2015-01-08 (3), 2015-03-07..2015-03-09 (3)
      ret_90: 2015-01-06..2015-01-08 (3), 2015-04-06..2015-04-08 (3)
      ret_180: 2015-01-06..2015-01-08 (3), 2015-07-05..2015-07-07 (3)
      ret_365: 2015-01-06..2015-01-08 (3), 2016-01-06..2016-01-08 (3)
      zret_7: 2015-01-06..2015-02-04 (30)
      zret_14: 2015-01-06..2015-02-04 (30)
      zret_30: 2015-01-06..2015-02-07 (33)
      past_z_7: 2015-01-06..2015-01-08 (3), 2015-01-13..2015-02-11 (30)
      past_class_7: 2015-01-06..2015-01-08 (3), 2015-01-13..2015-02-11 (30)
      vol_cc_7: 2015-01-06..2015-01-15 (10)
      vol_cc_14: 2015-01-07..2015-01-21 (15)
      vol_cc_30: 2015-01-08..2015-02-04 (28)
      vol_cc_90: 2015-01-08
      vol_park_7: 2015-01-06..2015-01-14 (9)
      vol_gk_7: 2015-01-06..2015-01-14 (9)
      vol_rs_7: 2015-01-06..2015-01-14 (9)
      vol_park_30: 2015-01-08
      vol_gk_30: 2015-01-08
      vol_rs_30: 2015-01-08
      vol_yz_30: 2015-01-08..2015-02-04 (28)
      vol_ratio_7_30: 2015-01-06..2015-02-04 (30)
      vol_ratio_30_90: 2015-01-08..2015-02-04 (28)
      vol_ratio_30_365: 2015-01-08..2015-02-04 (28)
      vol_of_vol_30: 2015-01-08..2015-02-10 (34)
      vol_z_365: 2015-01-08..2015-02-04 (28)
      hl_range: 2015-01-06..2015-01-08 (3)
      hl_range_mean_7: 2015-01-06..2015-01-14 (9)
      body: 2015-01-06..2015-01-08 (3)
      clv: 2015-01-06..2015-01-08 (3)
      upper_wick: 2015-01-06..2015-01-08 (3)
      lower_wick: 2015-01-06..2015-01-08 (3)
      body_mean_7: 2015-01-06..2015-01-14 (9)
      clv_mean_7: 2015-01-06..2015-01-14 (9)
      dist_sma_7: 2015-01-06..2015-01-14 (9)
      dist_sma_20: 2015-01-06..2015-01-25 (20)
      dist_sma_50: 2015-01-06..2015-01-08 (3)
      dist_sma_100: 2015-01-06..2015-01-08 (3)
      dist_sma_200: 2015-01-06..2015-01-08 (3)
      sma_50_200: 2015-01-08
      macd: 2015-01-06..2015-01-08 (3)
      macd_signal: 2015-01-06..2015-02-01 (27)
      macd_hist: 2015-01-06..2015-02-01 (27)
      trend_t_30: 2015-01-08
      trend_t_90: 2015-01-08
      dd_ath: 2015-01-06..2015-01-08 (3)
      days_since_ath: 2015-01-06..2015-01-08 (3)
      dist_high_365: 2015-01-06..2015-01-08 (3)
      dist_low_365: 2015-01-06..2015-01-08 (3)
      adx_14: 2015-01-06..2015-01-09 (4)
      aroon_osc_25: 2015-01-08..2015-01-31 (24)
      rsi_14: 2015-01-06..2015-01-09 (4)
      rsi_30: 2015-01-06..2015-01-09 (4)
      stoch_k_14: 2015-01-06..2015-01-20 (15)
      stoch_d_14_3: 2015-01-06..2015-01-22 (17)
      bb_pctb_20: 2015-01-06..2015-01-25 (20)
      bb_width_20: 2015-01-08..2015-01-25 (18)
      cci_20: 2015-01-06..2015-01-25 (20)
      lvol_z_30: 2015-01-06..2015-01-08 (3)
      lvol_z_90: 2015-01-06..2015-01-08 (3)
      lvol_ratio_7_30: 2015-01-06..2015-01-14 (9)
      amihud_z_365: 2015-01-08..2015-02-04 (28)
      obv_slope_30: 2015-01-08..2015-02-04 (28)
      cmf_20: 2015-01-08..2015-01-25 (18)
      up_vol_share_30: 2015-01-08..2015-02-04 (28)
      skew_30: 2015-01-08..2015-02-04 (28)
      kurt_30: 2015-01-08..2015-02-04 (28)
      acf1_30: 2015-01-08..2015-02-04 (28)
      skew_90: 2015-01-08
      kurt_90: 2015-01-08
      acf1_90: 2015-01-08
      up_share_14: 2015-01-07..2015-01-21 (15)
      up_share_30: 2015-01-08..2015-02-04 (28)
      maxdd_30: 2015-01-08
    table B: 0 blank cells
    table C, its FX columns (its A columns are table A's, check 3): 1 blank cells:
      btc_dxy_corr_60: 2015-01-08
    table D, its FX columns (its B columns are table B's, check 3): 0 blank cells
    full file, table B columns before table B: 43910 blank cells:
      every one of the 20 columns is blank only before its window lies inside its sources' history
PASS 6 deterministic: two builds with --k 0.36 into two temporary folders wrote byte-identical files (12 files), identical to those in data/; a build with --k 0.37 stops, since rule 5 gives 0.36; --k-unchecked on data/raw is refused and writes nothing (ruling 22)
REPORT 7 level_screen: 0 of 162 features above 10% (0 of the 43 FX columns): none
    over table A's rows; earliest 75%: 2012-12-17..2023-06-02; latest 25%: 2023-06-03..2026-11-27
    fx_usdcad_vol_20: 6.4% of 1274 late values outside the early range
    btc_dxy_corr_60: 3.3% of 1274 late values outside the early range
    days_since_halving: 3.0% of 1274 late values outside the early range
    halving_phase: 3.0% of 1274 late values outside the early range
    fx_usdjpy_vol_20: 2.3% of 1274 late values outside the early range
    amihud_z_365: 1.4% of 1274 late values outside the early range
    fx_usdjpy_vol_z_250: 1.2% of 1274 late values outside the early range
    oc_txcnt_g7: 0.9% of 1274 late values outside the early range
    oc_txcnt_g30: 0.5% of 1274 late values outside the early range
    fx_usdcad_vol_z_250: 0.5% of 1274 late values outside the early range
    obv_slope_30: 0.5% of 1274 late values outside the early range
    up_vol_share_30: 0.5% of 1274 late values outside the early range
    fx_usdchf_ret_60: 0.5% of 1274 late values outside the early range
    zret_30: 0.4% of 1274 late values outside the early range
    oc_feetotntv_g30: 0.4% of 1274 late values outside the early range
    basis_z_90: 0.4% of 1274 late values outside the early range
    fx_usdjpy_ret_20: 0.3% of 1274 late values outside the early range
    oc_feetotntv_g7: 0.2% of 1274 late values outside the early range
    fx_audusd_ret_1: 0.2% of 1274 late values outside the early range
    hl_range: 0.2% of 1274 late values outside the early range
    trend_t_30: 0.2% of 1274 late values outside the early range
    perp_spot_lvol: 0.2% of 1274 late values outside the early range
    avg_trade_z_90: 0.2% of 1274 late values outside the early range
    fx_usdjpy_ret_5: 0.2% of 1274 late values outside the early range
    zret_7: 0.1% of 1274 late values outside the early range
    zret_14: 0.1% of 1274 late values outside the early range
    cmf_20: 0.1% of 1274 late values outside the early range
    oc_adractcnt_z365: 0.1% of 1274 late values outside the early range
    fx_usdjpy_ret_1: 0.1% of 1274 late values outside the early range
    fx_audusd_vol_z_250: 0.1% of 1274 late values outside the early range
    fx_dxy_vol_z_250: 0.1% of 1274 late values outside the early range
PASS 8 ab_unchanged: the 8 files of data/built/ and data/profiler/ recorded in Phase 0 keep their SHA-256; the manifest's Bitstamp, Coin Metrics and Binance entries are those it had before the FX session, and the 25 raw files outside data/raw/ecb/ are exactly those entries' files, unchanged
```

What each check does for tables C and D:
- **1 labels.** It recomputes z, the class and both targets from raw
  Bitstamp for every row of all eight profiler files and
  `btc_daily_full.csv`, and k by rule 5. C's and D's dates and targets
  equal A's and B's, row for row.
- **2 point_in_time.**
  - Each cut copy holds the ECB API file cut to fixing dates on or before
    the date (ruling 25).
  - Each cut build runs with `--k 0.36 --k-unchecked`. On 29 of the 40
    dates, rule 5 on the cut history gives another k.
  - Every row of each cut build is compared with the full build in
    `btc_daily_full.csv` (A's and B's columns and `cm_day`),
    `fx_daily.csv` (the FX columns and `fx_day`), `c_up_7d.csv` and
    `d_up_7d.csv`: 124,067, 124,067, 124,067 and 39,778 rows.
  - Every row's `fx_day` is recomputed from the raw ECB file as the latest
    fixing date on or before the row.
  - The 40 dates hold six Saturdays, three Sundays and a weekday TARGET
    closing day, Easter Monday 2026-04-06, besides the first rows of all
    four tables.
  - Adding the Easter dates changed which 26 dates the seed draws, compared
    with the run of `reports/dataset.md`.
- **3 one_target.** C is A's columns followed by the 43 FX columns, in this
  file's own list and order, and D is B's followed by the same. The up and
  down files are identical apart from the target. The FX columns are equal
  in C, D and `fx_daily.csv`.
- **4 rows.** It covers all eight profiler files and `fx_daily.csv`.
  - Every raw file, now 202 with the 177 ECB files, matches its SHA-256.
  - No ECB fixing is later than the manifest's last day.
- **5 finite.** A blank FX cell is allowed only with a gap of its sources
  inside its window, counted in fixings. The only one is
  `btc_dxy_corr_60` on 2015-01-08, inside Bitstamp's gap.
- **6 deterministic.**
  - It compares 12 files from two builds with `--k 0.36` with each other
    and with `data/`.
  - It also proves ruling 22: a build with `--k 0.37` stops, since rule 5
    gives 0.36, and `--k-unchecked` on `data/raw` is refused without
    writing anything.
- **7 level_screen.** It screens all 162 distinct features over table A's
  rows, as before (section 4).
- **8 ab_unchanged.**
  - The eight files recorded in Phase 0 keep their SHA-256.
  - The manifest's Bitstamp, Coin Metrics and Binance entries re-serialise
    to the pre-session manifest's SHA-256.
  - The 25 raw files outside `data/raw/ecb/` are exactly those entries'
    files, unchanged.

Beyond the checks:
- **The FX values, recomputed independently.** Every value of the 43 FX
  columns was recomputed with plain pandas rolling windows, in the model
  that is now `scripts/inspect_ecb.py`, and compared with `fx_daily.csv`
  on all 5,094 rows.
  - The blank cells are the same, `fx_day` is the model's fixing date on
    every row, and the column order is the same.
  - The largest difference is 1.0e-12, on `fx_usdchf_vol_z_250`, from
    pandas' running sums.
- **The guards of ruling 22, tested by hand.** The build refuses
  `--k-unchecked` on `data/raw`, written with a trailing backslash or in
  another letter case as well, and refuses it without `--k`. In each case
  it writes nothing.

## 4. The level screen

**No FX column is above 10%**, nor any other feature. The FX columns
outside the earliest 75%'s range on more than 1% of the latest 25% of rows:

| Column | Share | Why |
|---|---|---|
| `fx_usdcad_vol_20` | 6.4% | USDCAD's 20-fixing volatility fell below its 2012-2023 low (0.00186) in ten quiet spells, 82 rows in all (low 0.00142): July to August 2024, the turn of 2024 and 2025, October 2025, and May, June, July to August and October 2026. It never rose above the earlier high |
| `btc_dxy_corr_60` | 3.3% | above its 2012-2023 high (0.401) on 42 rows from August to November 2024, up to 0.472. Its late range starts at −0.433, inside the earlier one |
| `fx_usdjpy_vol_20` | 2.3% | below its earlier low (0.00175) from 2026-06-03 to 2026-07-01 (29 rows, low 0.00110) |
| `fx_usdjpy_vol_z_250` | 1.2% | the same quiet month, against the year before |

These are spells, not a level that has moved away, so no FX column marks
the era the way `vol_cc_365` and SplyCur did. The feature set is fixed
(ruling 28), and nothing was changed.

## 5. What I think is wrong, and what I could not do

1. **Ruling 22, as first written, would have stopped check 2.**
   - Check 2's cut builds run with `--k 0.36` on a shorter history, and on
     most dates rule 5 gives another k there: on 31 of the 40 dates of the
     earlier run (for example 1.45 at 2013-05-05, 0.40 at 2019-12-09, 0.37
     through 2025), and on 29 of the 40 now.
   - I asked. The owner chose an explicit `--k-unchecked` that only check 2
     passes, which the build refuses when `--raw` is the real `data/raw`.
     That condition is recorded in ruling 22.
2. **The numbering of `02-add-fx.md`'s own rules.**
   - `02-add-fx.md` asks for its rules from ruling 22, and `02a` takes 21
     to 27 for its own rulings.
   - I asked whether to number the prompt's rules from 28, but that
     question went unanswered when the k question was answered. I followed
     the default I had proposed: they are 28 to 34. Say if you want them
     placed otherwise.
3. **My Phase 0 preview of the level screen left out the seven `_vol_20`
   columns.** `reports/fx_sources.md`, section 8, said every column other
   than the four named was at 0.3% or less. With the `_vol_20` columns,
   `fx_usdcad_vol_20` is at 6.4% and `fx_usdjpy_vol_20` at 2.3% (section
   4). None is above 10%, so the conclusion stands. I added a one-line
   correction to that section.
4. **`btc_dxy_corr_60` reads weaker than a same-time correlation** (ruling
   27). It is built as specified.
5. **Publication times are measured, not proved, for each day.** The probes
   cover 60 fixing dates, and check 2 cannot see publication times
   (`reports/fx_sources.md`, section 5).
6. **The README's layout now lists `scripts/compare_b.py`,
   `scripts/read_runs.py` and `runs/`**, which this session did not write,
   as not part of the build.
7. **Writing.**
   - Every file was written through PowerShell, and only under
     `experiments/crypto`. Nothing was blocked by the guard or a hook.
   - Changed: `scripts/features.py`, `scripts/build_dataset.py`,
     `scripts/check_dataset.py`, `README.md` and the one line of
     `reports/fx_sources.md`.
   - New: `scripts/inspect_ecb.py`, this report, and the data files of
     section 1.
   - Copies of the three scripts and the README as they were before this
     phase are in `data/scratch/before_fx/`, for review, since this folder
     is not under version control. The copies of A and B made there as a
     safeguard were deleted once check 8 passed.
8. **Nothing else was left undone.** The other fetch scripts were not run,
   and ML Professor was not run.
