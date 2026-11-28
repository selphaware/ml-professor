# Phase 1: the dataset

Built on 2026-11-28 from the raw data of `reports/sources.md`, fetched on
2026-11-28 between 09:18 and 09:21 UTC, through 2026-11-27. The rulings are
in `README.md` (10 to 20, and "Decided during Phase 1"). ML Professor was not
run.

## 1. The rows of each table

| | Table A | Table B |
|---|---|---|
| First row | 2012-12-17 (ruling 12) | 2019-12-09 (ruling 13) |
| Last row | 2026-11-27 | 2026-11-27 |
| Rows | 5,094 | 2,546 |
| Feature columns | 99 | 119 |
| Rows with a target | 5,054 | 2,539 |
| Rows without a target | 40 | 7 |
| Blank feature cells | 1,014, on 49 rows | none |

**Why the 40 rows of table A have no target:**
- The last seven rows, 2026-11-21..27, have no C_{t+7}.
- 2014-12-30..2015-01-01 (3 rows): C_{t+7} falls on a missing Bitstamp day.
- 2015-01-06..08 (3 rows): C_t is missing. These are the zero-volume days.
- 2015-01-09..2015-02-04 (27 rows): σ30_t is missing, because its 30
  returns hold only 26 (the returns of 2015-01-06..09 need a missing close).

**Table B** lacks a target only on its last seven rows.

**Rule 8 and the first rows.**
- Rule 8 alone would start table A on 2012-12-01. Ruling 12 moves it to
  2012-12-17, so that no blank cell comes from the 2011 zero-volume days.
- Table B's first row, 2019-12-09, is rule 8's own. The binding column is
  `fund_z_90`.
- The build refuses to write a table if any feature column lacks a value
  on its first row.

**The blank cells of table A** all come from the 2015-01-06..08 gap: 49
rows.
- On 2015-01-06..2015-02-11 (37 rows), 2015-01-08 is blank in every
  Bitstamp-based column. On 2015-01-06 and 01-07, the columns that need
  that day's own candle are blank, and 30 and 28 rolling columns keep a
  value (the point-in-time reading).
- Three-row holes follow where a return's far end falls in the gap:
  `ret_60` on 2015-03-07..09, `ret_90` on 2015-04-06..08, `ret_180` on
  2015-07-05..07 and `ret_365` on 2016-01-06..08.
- Check 5 below lists every cell by column and date. Calendar and on-chain
  columns are never blank in either table.

**The eight Coin Metrics rows** (2021-02-25, 2024-08-15..18, 2025-05-02..04)
take an earlier Coin Metrics day under ruling 15, and have no blank cell.

## 2. k and the class shares

**k = 0.36.**
- It is the 1/3 quantile of |z_t| over the earliest 75% of table A's rows
  that have a z: the first 3,790 of 5,054, from 2012-12-17 to 2023-06-05.
- The quantile is 0.358625, which rounds to 0.36.

| Rows | n | UP | RANGE | DOWN |
|---|---|---|---|---|
| earliest 75% (2012-12-17..2023-06-05) | 3,790 | 38.7% | 33.5% | 27.8% |
| latest 25% (2023-06-06..2026-11-20) | 1,264 | 35.1% | 33.9% | 31.0% |
| all of table A | 5,054 | 37.8% | 33.6% | 28.6% |
| table B (2019-12-09..2026-11-20) | 2,539 | 35.7% | 35.3% | 29.0% |

The share of `up_7d` is the UP share, and that of `down_7d` the DOWN share.
By year, over table A's rows with a target (table B's rows are the same
rows from 2019-12-09):

| Year | Rows | up_7d = 1 | RANGE | down_7d = 1 |
|---|---|---|---|---|
| 2012 | 15 | 0.0% | 93.3% | 6.7% |
| 2013 | 365 | 60.6% | 20.5% | 18.9% |
| 2014 | 363 | 19.8% | 38.0% | 42.1% |
| 2015 | 334 | 42.5% | 29.9% | 27.5% |
| 2016 | 366 | 41.5% | 40.2% | 18.3% |
| 2017 | 365 | 54.2% | 24.4% | 21.4% |
| 2018 | 365 | 28.2% | 30.4% | 41.4% |
| 2019 | 365 | 33.1% | 38.9% | 28.0% |
| 2020 | 366 | 47.8% | 36.1% | 16.1% |
| 2021 | 365 | 36.4% | 31.8% | 31.8% |
| 2022 | 365 | 24.4% | 42.5% | 33.1% |
| 2023 | 365 | 38.4% | 38.1% | 23.6% |
| 2024 | 366 | 41.0% | 31.7% | 27.3% |
| 2025 | 365 | 32.3% | 34.8% | 32.9% |
| 2026 | 324 | 29.6% | 29.9% | 40.4% |

The year-to-year swings are the market's own: 2013, 2017 and 2020 rallied,
while 2014, 2018 and 2026 fell. That is what the targets measure.
## 3. The feature catalogue

C, O, H, L and V are Bitstamp's close, open, high, low and volume (BTC) of
day t, and r_t = ln(C_t / C_{t−1}) is the daily log return. σ30_t is the
standard deviation (ddof 1) of the 30 r ending with t. Every window ends at
row t, inclusive, and gives a value only under the missing-data rules
(below). "Lag" is when the inputs of row t are known:
- **close**: at the close of day t, 00:00 UTC on t+1;
- **as-of**: Coin Metrics' value for the latest day D ≤ t−1 that was known
  at that close (ruling 15). D is t−1 on every row but eight.

The table column says which tables carry the feature. k = 0.36 (rule 5).

| Column | Formula | Window | Source | Lag | Table |
|---|---|---|---|---|---|
| `ret_1`, `ret_2`, `ret_3`, `ret_5`, `ret_7`, `ret_14`, `ret_21`, `ret_30`, `ret_60`, `ret_90`, `ret_180`, `ret_365` | ret_n = ln(C_t / C_{t−n}) | its two ends, t and t−n | Bitstamp | close | A, B |
| `zret_7`, `zret_14`, `zret_30` | ret_n / (σ30_t × √n) | n, and 30 returns for σ30 | Bitstamp | close | A, B |
| `past_z_7` | ln(C_t / C_{t−7}) / (σ30_{t−7} × √7): the label's z for the seven days just ended | 7, and 30 returns ending t−7 | Bitstamp | close | A, B |
| `past_class_7` | 1 when past_z_7 > k, −1 when past_z_7 < −k, else 0 | as past_z_7 | Bitstamp | close | A, B |
| `vol_cc_7`, `vol_cc_14`, `vol_cc_30`, `vol_cc_90` | standard deviation (ddof 1) of r; vol_cc_30 is σ30 | n returns | Bitstamp | close | A, B |
| `vol_park_7`, `vol_park_30` | √(mean of the Parkinson variance (ln(H/L))² / (4 ln 2)) | n | Bitstamp | close | A, B |
| `vol_gk_7`, `vol_gk_30` | √(mean of the Garman–Klass variance ½(ln(H/L))² − (2 ln 2 − 1)(ln(C/O))²) | n | Bitstamp | close | A, B |
| `vol_rs_7`, `vol_rs_30` | √(mean of the Rogers–Satchell variance ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)) | n | Bitstamp | close | A, B |
| `vol_yz_30` | Yang–Zhang: √(σ²_o + k_yz·σ²_c + (1 − k_yz)·RS). σ²_o is the variance of ln(O_t / C_{t−1}), σ²_c that of ln(C_t / O_t), RS the mean Rogers–Satchell variance, and k_yz = 0.34 / (1.34 + (m+1)/(m−1)) over m days | 30 | Bitstamp | close | A, B |
| `vol_ratio_7_30`, `vol_ratio_30_90`, `vol_ratio_30_365` | ln(vol_cc_a / vol_cc_b). The 365-day σ is computed for this ratio only and is not a column | b returns | Bitstamp | close | A, B |
| `vol_of_vol_30` | standard deviation over 30 days of ln(vol_cc_7) | 30 + 7 | Bitstamp | close | A, B |
| `vol_z_365` | trailing 365-day z-score of ln(vol_cc_30) | 365 + 30 | Bitstamp | close | A, B |
| `hl_range` | ln(H_t / L_t) | 1 | Bitstamp | close | A, B |
| `hl_range_mean_7` | mean of hl_range | 7 | Bitstamp | close | A, B |
| `body` | (C − O) / (H − L); 0 when H = L | 1 | Bitstamp | close | A, B |
| `clv` | (2C − H − L) / (H − L); 0 when H = L | 1 | Bitstamp | close | A, B |
| `upper_wick` | (H − max(O, C)) / (H − L); 0 when H = L | 1 | Bitstamp | close | A, B |
| `lower_wick` | (min(O, C) − L) / (H − L); 0 when H = L | 1 | Bitstamp | close | A, B |
| `body_mean_7`, `clv_mean_7` | mean of body, of clv | 7 | Bitstamp | close | A, B |
| `dist_sma_7`, `dist_sma_20`, `dist_sma_50`, `dist_sma_100`, `dist_sma_200` | ln(C_t / SMA_n), where SMA_n is the mean close over n days | n | Bitstamp | close | A, B |
| `sma_50_200` | ln(SMA_50 / SMA_200) | 200 | Bitstamp | close | A, B |
| `macd` | (EMA_12 − EMA_26) / C_t. An EMA of span s has α = 2/(s+1), starts from the mean of its first s inputs, and counts as complete after 3s days | 78 | Bitstamp | close | A, B |
| `macd_signal` | EMA_9 of macd, from macd's first complete value | 78 + 27 | Bitstamp | close | A, B |
| `macd_hist` | macd − macd_signal | 78 + 27 | Bitstamp | close | A, B |
| `trend_t_30`, `trend_t_90` | t-statistic of the least-squares slope of ln C on day number | n | Bitstamp | close | A, B |
| `dd_ath` | ln(C_t / the highest C up to and including t) | all of Bitstamp's history | Bitstamp | close | A, B |
| `days_since_ath` | ln(1 + days since the latest day on which that highest C was reached) | all of Bitstamp's history | Bitstamp | close | A, B |
| `dist_high_365` | ln(C_t / the highest H over 365 days) | 365 | Bitstamp | close | A, B |
| `dist_low_365` | ln(C_t / the lowest L over 365 days) | 365 | Bitstamp | close | A, B |
| `adx_14` | Wilder's ADX. TR, +DM and −DM are each Wilder-averaged (α = 1/14, from the mean of the first 14). DX = 100·\|+DI − −DI\| / (+DI + −DI), with ±DI = 100·DM/TR. ADX is the Wilder average of DX | about 85 | Bitstamp | close | A, B |
| `aroon_osc_25` | Aroon up − Aroon down. Up is 100·(25 − days since the highest H)/25 over today and the 25 days before; down likewise for the lowest L. On ties the latest day counts | 26 | Bitstamp | close | A, B |
| `rsi_14`, `rsi_30` | 100 · avg gain / (avg gain + avg loss) of the daily change C_t − C_{t−1}. Both are Wilder averages (α = 1/n, from the mean of the first n) | 3n | Bitstamp | close | A, B |
| `stoch_k_14` | 100 · (C − lowest L) / (highest H − lowest L) over 14 days | 14 | Bitstamp | close | A, B |
| `stoch_d_14_3` | mean of the last 3 stoch_k_14 | 16 | Bitstamp | close | A, B |
| `bb_pctb_20` | (C − (SMA_20 − 2·sd)) / (4·sd), where sd is the standard deviation (ddof 1) of C over 20 days | 20 | Bitstamp | close | A, B |
| `bb_width_20` | 4·sd / SMA_20 | 20 | Bitstamp | close | A, B |
| `cci_20` | (TP − mean TP) / (0.015 · mean absolute deviation of TP from that mean), where TP = (H + L + C)/3 | 20 | Bitstamp | close | A, B |
| `lvol_z_30`, `lvol_z_90` | trailing z-score of ln(DV), where DV = V × C | n | Bitstamp | close | A, B |
| `lvol_ratio_7_30` | ln(mean DV over 7 days / mean DV over 30 days) | 30 | Bitstamp | close | A, B |
| `amihud_z_365` | trailing 365-day z-score of ln(30-day mean of \|r\| / DV) | 365 + 30 | Bitstamp | close | A, B |
| `obv_slope_30` | Σ sign(r)·V / Σ V over 30 days | 30 | Bitstamp | close | A, B |
| `cmf_20` | Chaikin money flow: Σ clv·V / Σ V over 20 days | 20 | Bitstamp | close | A, B |
| `up_vol_share_30` | Σ V on days with r > 0, over Σ V, across 30 days | 30 | Bitstamp | close | A, B |
| `skew_30`, `skew_90` | sample skewness (G1) of r | n | Bitstamp | close | A, B |
| `kurt_30`, `kurt_90` | sample excess kurtosis (G2) of r | n | Bitstamp | close | A, B |
| `acf1_30`, `acf1_90` | lag-1 autocorrelation: the correlation of r_s with r_{s−1}, both in the window | n | Bitstamp | close | A, B |
| `up_share_14`, `up_share_30` | share of days with r > 0 | n | Bitstamp | close | A, B |
| `maxdd_30` | the largest fall from a peak to a later trough of C within the window, as ln(trough / peak) (≤ 0) | 30 | Bitstamp | close | A, B |
| `dow` | day of week, 0 Monday to 6 Sunday | none | calendar | none | A, B |
| `month` | 1 to 12 | none | calendar | none | A, B |
| `days_since_halving` | ln(1 + days since the latest halving on or before t: 2012-11-28, 2016-07-09, 2020-05-11, 2024-04-20; before the first, since 2009-01-03) | none | calendar | none | A, B |
| `halving_phase` | days since that halving / 1461. **This column can act as an era marker** | none | calendar | none | A, B |
| `oc_adractcnt_g7`, `oc_adractcnt_g30` | ln(v_t / v_{t−n}), where v is the as-of AdrActCnt (active addresses) | n | Coin Metrics | as-of | A, B |
| `oc_adractcnt_z365` | trailing 365-day z-score of ln v | 365 | Coin Metrics | as-of | A, B |
| `oc_txcnt_g7`, `oc_txcnt_g30`, `oc_txcnt_z365` | the same for TxCnt (transactions) | 7, 30, 365 | Coin Metrics | as-of | A, B |
| `oc_hashrate_g7`, `oc_hashrate_g30`, `oc_hashrate_z365` | the same for HashRate | 7, 30, 365 | Coin Metrics | as-of | A, B |
| `oc_feetotntv_g7`, `oc_feetotntv_g30`, `oc_feetotntv_z365` | the same for FeeTotNtv (total fees, BTC) | 7, 30, 365 | Coin Metrics | as-of | A, B |
| `mvrv` | ln(CapMVRVCur): market cap over realised cap | 1 | Coin Metrics | as-of | A, B |
| `mvrv_z365` | trailing 365-day z-score of mvrv | 365 | Coin Metrics | as-of | A, B |
| `fund_1d` | sum of the funding rates printed at 00:00, 08:00 and 16:00 UTC of day t, each time floored to the hour | 1 | Binance funding | close | B |
| `fund_7`, `fund_30` | sum of fund_1d over the window, scaled to n days if a day were missing (none is) | n | Binance funding | close | B |
| `fund_z_90` | trailing 90-day z-score of fund_1d | 90 | Binance funding | close | B |
| `basis` | ln(perpetual close / spot close) | 1 | Binance spot, perpetual | close | B |
| `basis_mean_7` | mean of basis | 7 | Binance spot, perpetual | close | B |
| `basis_z_90` | trailing 90-day z-score of basis | 90 | Binance spot, perpetual | close | B |
| `usdt_prem` | ln(Binance spot close / Bitstamp close) | 1 | Binance spot, Bitstamp | close | B |
| `usdt_prem_z_90` | trailing 90-day z-score of usdt_prem | 90 | Binance spot, Bitstamp | close | B |
| `perp_spot_lvol` | ln(perpetual quote volume / spot quote volume) | 1 | Binance spot, perpetual | close | B |
| `perp_spot_lvol_z_90` | trailing 90-day z-score of perp_spot_lvol | 90 | Binance spot, perpetual | close | B |
| `taker_share_spot` | spot taker-buy quote volume / spot quote volume | 1 | Binance spot | close | B |
| `taker_share_spot_mean_7`, `taker_share_spot_z_90` | its 7-day mean and trailing 90-day z-score | 7, 90 | Binance spot | close | B |
| `taker_share_perp` | perpetual taker-buy quote volume / perpetual quote volume | 1 | Binance perpetual | close | B |
| `taker_share_perp_mean_7`, `taker_share_perp_z_90` | its 7-day mean and trailing 90-day z-score | 7, 90 | Binance perpetual | close | B |
| `trades_z_90` | trailing 90-day z-score of ln(spot trade count) | 90 | Binance spot | close | B |
| `avg_trade_z_90` | trailing 90-day z-score of ln(spot quote volume / spot trade count) | 90 | Binance spot | close | B |
| `bn_lvol_z_30` | trailing 30-day z-score of ln(spot quote volume) | 30 | Binance spot | close | B |

Table A has 99 feature columns and table B 119. A trailing z-score of x over
n days is (x_t − mean) / standard deviation (ddof 1), both taken over the n
days ending with t.

**Listed in the prompt but not built:**
- `oc_splycur_g7`, `oc_splycur_g30`, `oc_splycur_z365`: dropped by ruling 14.
- `nvt_30`, `nvt_z365`: adjusted transfer value is not free in Coin Metrics
  Community (ruling 14).
- `vol_cc_365`: **dropped in Phase 1 by the level screen** (check 7). From
  2023-09-09 the one-year volatility lies below every value of 2012-2023:
  62.6% of the latest 25% of rows fall outside the earliest 75%'s range. It
  marks the era, as SplyCur did (ruling 14). It is still computed as the
  denominator of `vol_ratio_30_365`, which stays inside the earlier range.

**Added beyond the prompt:** no feature. `btc_daily_full.csv` also carries
`cm_day`, the Coin Metrics day that row used. It is never written to a
profiler file.
## 4. The checks

Here is the full output of `check_dataset.py`, run from
`E:\CODE\ml-professor` after the build. It exits 0, and checks 1 to 6 print
PASS:

```
PASS 1 labels: k = 0.36 (rule 5 recomputed: 0.36); z, class, up_7d, down_7d recomputed from raw Bitstamp agree on 5094 rows of every file; blank on 40 rows, exactly where C_t, C_t+7 or sigma30_t is missing
PASS 2 point_in_time: 40 dates (seed 20261128; table A's and B's first rows, the last row with a target, the last row, and 2015-01-06, 2015-01-07, 2015-01-08, 2015-01-09, 2021-02-25, 2024-08-16, 2025-05-03 among them): each rebuilt from raw data cut at its close with k = 0.36; its row, and every earlier row (118678 rows in all), agree in all 119 feature columns within 1e-09
    dates: 2012-12-17, 2013-05-05, 2014-03-04, 2015-01-06, 2015-01-07, 2015-01-08, 2015-01-09, 2016-04-25, 2017-01-04, 2017-11-23, 2018-01-31, 2018-02-16, 2018-09-26, 2018-10-30, 2019-06-10, 2019-12-09, 2020-01-04, 2021-02-25, 2021-06-28, 2021-07-15, 2021-11-24, 2022-04-08, 2022-05-05, 2022-10-24, 2022-12-26, 2023-02-05, 2023-08-08, 2024-06-29, 2024-08-16, 2024-12-03, 2025-01-30, 2025-05-03, 2025-07-18, 2025-11-22, 2025-11-26, 2026-03-26, 2026-06-17, 2026-11-16, 2026-11-20, 2026-11-27
PASS 3 one_target: four profiler files: date, then 99 (A) or 119 (B) features, then one target; no column holds or equals the forward return, z or the class; up and down files identical apart from the target; table A's columns all in B with equal values on 2546 common dates
PASS 4 rows: dates strictly rising by one day, ISO text, no day missing; table A 2012-12-17..2026-11-27 (5094 rows), table B 2019-12-09..2026-11-27 (2546 rows); last row is the manifest's last completed day; targets blank on the last seven rows (and elsewhere exactly as rule 6, check 1); every raw file matches its SHA-256
PASS 5 finite: no infinite value in any file; every blank feature cell lies within its column's reach of a missing source day, or before its window lies inside its sources' history (missing days: bitstamp: 2011-08-23, 2011-08-26..2011-08-29 (4), 2011-08-31, 2011-09-03, 2011-09-05, 2011-09-09..2011-09-12 (4), 2011-09-30..2011-10-02 (3), 2011-10-15..2011-10-16 (2), 2011-10-18..2011-10-19 (2), 2011-10-22..2011-10-23 (2), 2011-10-27, 2011-11-02..2011-11-03 (2), 2011-11-07, 2011-11-23, 2011-11-27, 2011-12-04, 2011-12-09, 2011-12-17, 2015-01-06..2015-01-08 (3))
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
    full file, table B columns before table B: 43910 blank cells:
      every one of the 20 columns is blank only before its window lies inside its sources' history
PASS 6 deterministic: two builds into two temporary folders wrote byte-identical files (6 files), identical to those in data/
REPORT 7 level_screen: 0 of 119 features above 10%: none
    earliest 75%: 2012-12-17..2023-06-02; latest 25%: 2023-06-03..2026-11-27
    days_since_halving: 3.0% of 1274 late values outside the early range
    halving_phase: 3.0% of 1274 late values outside the early range
    amihud_z_365: 1.4% of 1274 late values outside the early range
    oc_txcnt_g7: 0.9% of 1274 late values outside the early range
    oc_txcnt_g30: 0.5% of 1274 late values outside the early range
    obv_slope_30: 0.5% of 1274 late values outside the early range
    up_vol_share_30: 0.5% of 1274 late values outside the early range
    zret_30: 0.4% of 1274 late values outside the early range
    oc_feetotntv_g30: 0.4% of 1274 late values outside the early range
    basis_z_90: 0.4% of 1274 late values outside the early range
    oc_feetotntv_g7: 0.2% of 1274 late values outside the early range
    hl_range: 0.2% of 1274 late values outside the early range
    trend_t_30: 0.2% of 1274 late values outside the early range
    perp_spot_lvol: 0.2% of 1274 late values outside the early range
    avg_trade_z_90: 0.2% of 1274 late values outside the early range
    zret_7: 0.1% of 1274 late values outside the early range
    zret_14: 0.1% of 1274 late values outside the early range
    cmf_20: 0.1% of 1274 late values outside the early range
    oc_adractcnt_z365: 0.1% of 1274 late values outside the early range
```

What each check does beyond the letter of the prompt:
- **Check 2.**
  - Besides the four required dates, it always includes 2015-01-06..09
    (the gap) and 2021-02-25, 2024-08-16 and 2025-05-03 (Coin Metrics rows
    that use an earlier day). The other 29 dates are drawn with seed
    20261128.
  - It compares every row of each cut build with the full build, 118,678
    rows in all, not only the cut date's row. It also compares `cm_day`.
  - Coin Metrics is cut by ruling 15: by completion time from 2021-02-24,
    and by date before.
  - Funding is cut by each print's floored hour.
- **Check 1** recomputes k by rule 5 as well as z, the class and the
  targets, in its own code from raw Bitstamp.
- **Check 4** also confirms every raw file against its SHA-256 in the
  manifest.
- **Check 5** has its own table of the sources each column is built from,
  and how far back it reads. It accepts a blank cell only within that
  reach of a missing source day, or before the column's window lies inside
  its sources' history.

Beyond the checks, I recomputed about 40 features for 2024-06-29
independently, with scipy and plain pandas on the raw files. Every one
agrees with the build: 1.8e-9 at most, on `amihud_z_365`, from pandas'
running sums.

### The level screen (check 7)

The screen takes each feature over table A's rows. It reports the share of
the values in the latest 25% (2023-06-03..2026-11-27, 1,274 rows) that lie
outside the feature's range over the earliest 75% (2012-12-17..2023-06-02).
Binance columns have early values from 2017 or 2019 only.

- **`vol_cc_365` was 62.6% on the first build, and is fixed by not being a
  column.**
  - Bitstamp's one-year volatility has fallen as the market matured:
    yearly medians run from 0.064 in 2013 and 0.061 in 2014 to 0.025 in
    2024 and 0.023 in 2026.
  - From 2023-09-09 it lies below every value of 2012-2023. Against a
    holdout of the latest rows it would act as an era marker, the reason
    ruling 14 dropped SplyCur.
  - It is still computed as the denominator of `vol_ratio_30_365`, which
    stays inside the earlier range, and so do `vol_cc_90` (0%) and
    `vol_z_365` (0%).
- **After that, no feature is above 10%.**
  - The highest are `days_since_halving` and `halving_phase`, at 3.0%. The
    2020-2024 cycle ran 1,440 days, longer than the 1,319 and 1,402 days
    the early part saw, so its days 1,402 to 1,439 (38 rows) lie past the early range.
  - Next is `amihud_z_365` at 1.4%, and every other feature is at 0.9% or
    below.

## 5. What I think is wrong, and what I could not do

1. **The longer-gap rule as written used hindsight.** "The days inside a
   longer gap have no value" blanks the first two days of a gap. Whether a
   gap is long is known only on its third day, so a build cut at 2015-01-06
   or 01-07 gave those rows values the full build did not, and check 2
   failed on them. I asked, and the owner chose the point-in-time reading
   (`README.md`, "Decided during Phase 1").
   - So sections 10 and 11 of `reports/sources.md`, and the model in
     `scripts/inspect_sources.py`, differ from the build in 58 cells on
     those two rows. Both rows have no target.
   - Every other blank cell is exactly as listed there.
2. **`vol_cc_365`, one of the listed features, is not built**, for the
   reason in section 4.
3. **k is recomputed by every build without `--k`.** A later fetch adds
   rows to the earliest 75% and can move k, and with it every class. Rule
   5 says k is "then fixed". To keep the targets of this build, rebuild a
   new fetch with `--k 0.36`.
4. **Coin Metrics' history cannot be proved point-in-time.** Before
   2021-02-24 no publication times exist, and any revision since cannot be
   seen. Check 2 proves only that the build uses nothing after each row's
   close in the files as fetched today. This is ruling 15's stated limit.
5. **Coin Metrics has zero values** for TxCnt, FeeTotNtv and AdrActCnt in
   2009-2010, which have no logarithm. They count as missing days of that
   metric. They lie outside every window of the tables.
6. **One command was refused by the guard.** A comment in it held the name
   of a locked folder on the same line as a file write, so the guard
   blocked it. Nothing ran; I resent the command without the comment.
7. **Writing.** Every file was written through PowerShell, and nothing
   outside `experiments/crypto`. The checks' temporary folders live under
   `data/scratch/` and are removed. Two staging files used to assemble this
   report were deleted.
8. **Nothing else was left undone.**