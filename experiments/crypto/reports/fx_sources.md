# Phase 0: the ECB reference rates (tables C and D)

Fetched on 2026-11-28 between 12:24 and 12:31 UTC, through **2026-11-27**,
the last day of tables A and B. That day is a Friday, and 2026-11-28 is a
Saturday with no fixing, so nothing later could have been published.
Every number below was measured on `data/raw/` as fetched (section 10 says
how), unless it says otherwise.

## Decisions I need from you

Each is explained in the section named.

1. **`--k 0.36` rewrites `data/built/build_info.json`**, so "build with
   `--k 0.36`" and "A and B must not change by a single byte" cannot both
   hold with the build as it is (section 9.1). I recommend option (a): the
   build records rule 5 whether or not k is given, and C and D get a
   summary file of their own.
2. **The source: the Data Portal API's SDMX-CSV**, with the historical zip
   kept as received as a cross-check that the build does not read
   (section 2). The two agree on every value.
3. **The API's holiday rows are not fixing dates** (section 3). It lists 62
   TARGET closing days up to 2012 with no value in any currency. No date
   has only some of the seven currencies.
4. **`btc_dxy_corr_60` is blank on one row of table C, 2015-01-08**, the
   third day of Bitstamp's 2015 gap. That is a longer gap after the table's
   first row, so the rules say to stop for you (section 8). I recommend
   keeping the row with that blank, as ruling 11 did for table A. The row
   is already blank in every Bitstamp column and has no target.
5. **Ruling numbers.** The prompt says rulings 22 onwards, but `README.md`
   ends at ruling 20 and no ruling 21 exists in this folder. Should the
   new ones start at 21? (section 9.3)
6. **`btc_daily_full.csv` cannot gain the FX columns**, because it is one of
   the files that must stay byte-identical. So the FX columns go to a new
   `data/built/fx_daily.csv`, which the prompt's own condition requires
   (section 9.2). Please confirm the layout proposed there.
7. **How check 2 cuts the ECB files** (section 9.4).
8. **Keep the Phase 0 measurements as `scripts/inspect_ecb.py`?** (section 10)

The first rows are the expected ones: **table C starts on 2012-12-17 and
table D on 2019-12-09** (section 8). Tables A and B are unchanged
(section 1).

## 1. Tables A and B as found (SHA-256, recorded 2026-11-28 12:15 UTC)

Recorded before any script was written or changed. The table went into
this report then, before anything else, and is as recorded.

| File | Bytes | Last written (UTC) | SHA-256 |
|---|---|---|---|
| `data/built/btc_daily_full.csv` | 11,147,236 | 2026-11-28 10:56:11 | `c2f44ab747f6e403c0a802c44ac677bb23a5080bbc62be604b0741e3b1abc72e` |
| `data/built/build_info.json` | 7,855 | 2026-11-28 10:56:13 | `fecb623d730ffabf641212ad57984e70cce4835b3e1be3edce632f6bdfd4d72a` |
| `data/profiler/a_down_7d.csv` | 9,679,477 | 2026-11-28 10:56:12 | `f962cdfc2063d1e1a12ee45b5c44a8cfe9f32a9390722e1e61b885e6a0483383` |
| `data/profiler/a_short_down_7d.csv` | 4,862,585 | 2026-11-28 11:50:56 | `7d8d34e00a2f8db4b8a1122e71756c04d27e8017a8e2af39a5fd5ab4b00bfa60` |
| `data/profiler/a_short_up_7d.csv` | 4,862,583 | 2026-11-28 11:50:55 | `d067c2285c422f291a6b34a7b59b4474b914c89b12662c79c1a972c26dee7aae` |
| `data/profiler/a_up_7d.csv` | 9,679,475 | 2026-11-28 10:56:12 | `64b652325c3c50c75223d53b89f2863b03db3e892c74e1f8f2d5dc9c6875903a` |
| `data/profiler/b_down_7d.csv` | 5,853,303 | 2026-11-28 10:56:13 | `b3016a45489c765f96d0202c91a97c70c84364b5ffa3dd225d797edfe424a555` |
| `data/profiler/b_up_7d.csv` | 5,853,301 | 2026-11-28 10:56:12 | `8c76019717bcd3ba9e585d203ed4b5056610548673bfbabb5d6e289e1c5f9d21` |

Measured afterwards:
- **After the ECB fetch**, these are still the only files in `data/built/`
  and `data/profiler/`, and every hash is unchanged.
- **The other raw files are untouched.** All 25 raw files of Bitstamp, Coin
  Metrics and Binance still match their manifest entries.
- **The current build reproduces these files.** Run in memory, it writes
  `btc_daily_full.csv`, `build_info.json` and the four `a_`/`b_` files
  byte for byte, when k is computed (section 9.1 has the case of `--k`).
- **The two `a_short_` files** hold the header of A and A's rows from
  2019-12-09 on (2,546 rows), byte for byte. No script writes them.

## 2. Fetching, storage, and the choice of file

`scripts/fetch_ecb.py` is the only new code that touches the network. It
takes the last day from the manifest's Bitstamp entry (2026-11-27), so it
cannot fetch past tables A and B. It refuses a day that has not ended, and
writes nothing until every answer has passed its checks.

| Request | URL with every parameter | File | Rows | Bytes | Last-Modified |
|---|---|---|---|---|---|
| API, SDMX-CSV | `https://data-api.ecb.europa.eu/service/data/EXR/D.USD+JPY+GBP+CHF+AUD+CAD+SEK.EUR.SP00.A?format=csvdata&endPeriod=2026-11-27` | `ecb/exr_daily.csv` | 50,456 | 10,860,195 | Fri, 27 Nov 2026 14:56:58 GMT |
| historical file | `https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip` | `ecb/eurofxref-hist.zip` | 7,146 | 644,068 | Fri, 27 Nov 2026 14:56:38 GMT |
| 175 probes (section 5) | the API URL with `startPeriod`, `endPeriod` (the same date) and `updatedAfter` | `ecb/probes/<date>_after_<instant>.csv` | 0 or 7 each | | |

- **Storage.** Each response body is saved byte for byte, one file per
  request. A probe with no result is saved as received: an empty body with
  HTTP 200 (83 probes), or a JSON error body with HTTP 404 (7, all for
  2026-11-27).
- **The manifest.** The new `ecb` entry follows the others: `script`,
  `run_started_utc`, `last_complete_day` (2026-11-27) and `storage`. Each
  file record holds `file`, `url`, `fetched_utc`, `rows`, `sha256`,
  `http_status` and `bytes`; the two main files also hold their first and
  last date and the response headers, and the probes the instant asked.
  The other three entries are unchanged (section 10).
- **Both starting points answer as described.**
  - The API, with `format=csvdata`, returns one row per currency and day,
    with 32 columns: the series key, its five dimensions, `TIME_PERIOD`,
    `OBS_VALUE` and 24 attributes, `OBS_STATUS` among them. I added
    `endPeriod=2026-11-27`; no key was used.
  - The zip holds one file, `eurofxref-hist.csv`: `Date` and 42 currency
    columns, one row per fixing date.
- **The two agree exactly.** On their 7,146 common dates, all 50,022 values
  of the seven currencies are the same text. The zip has no `N/A` among
  them. The API has 62 more dates, the holiday rows of section 3. The zip
  has no date that the API lacks.
- **Proposed: the build reads `exr_daily.csv` only.** The zip stays in
  `data/raw/ecb/`, as received, as the cross-check. Reasons:
  - the server cuts the API at `endPeriod`, so no fixing after 2026-11-27
    was received. The zip cannot be cut: the fetch refuses to keep it if
    it ever holds a later day;
  - the API marks each observation's status (`A` normal, `H` holiday), and
    carries the series attributes that bear on timing (section 5);
  - the publication probes measure the API's own records (section 5).

## 3. Each currency

| Currency | First fixing | Last fixing | Fixings | Decimals |
|---|---|---|---|---|
| USD | 1999-01-04 | 2026-11-27 | 7,146 | 4 |
| JPY | 1999-01-04 | 2026-11-27 | 7,146 | 2 |
| GBP | 1999-01-04 | 2026-11-27 | 7,146 | 5 |
| CHF | 1999-01-04 | 2026-11-27 | 7,146 | 4 |
| AUD | 1999-01-04 | 2026-11-27 | 7,146 | 4 |
| CAD | 1999-01-04 | 2026-11-27 | 7,146 | 4 |
| SEK | 1999-01-04 | 2026-11-27 | 7,146 | 4 |

- **All 50,022 values have status `A`**, and none is empty, zero or
  negative.
- **No date has some of the seven currencies but not all: 0 such dates.**
  So the rule "a fixing date with any of the seven currencies missing has
  no value for any series" never applies.
- **The holiday rows.** The API lists 62 more dates, 1999-12-31 to
  2012-05-01. On each, all seven currencies have an empty `OBS_VALUE` with
  `OBS_STATUS` `H`: 434 rows. `H` is, as I know it, the code for a value
  missing because of a holiday (I did not fetch the code list).
  - Every one of these dates is a TARGET closing day (section 4).
  - After 2012-05-01 the API lists no row at all for a holiday, and the zip
    never does.
  - Proposed: **these are not fixing dates.** The fixing sequence is the
    7,146 dates with a value. Counting them as fixing dates with every
    value missing would put a gap into every window across a holiday
    before May 2012, and none after it, for the same kind of day.

## 4. Weekdays with no fixing

1999-01-04 to 2026-11-27 holds 7,280 weekdays. 7,146 have a fixing and
**134 have none**. No fixing falls on a Saturday or a Sunday. The dates of
every year are the same for all seven currencies (section 3).

| Year | Weekdays with no fixing | Dates |
|---|---|---|
| 1999 | 1 | 12-31 |
| 2000 | 5 | GF 04-21, EM 04-24, 05-01, 12-25, 12-26 |
| 2001 | 7 | 01-01, GF 04-13, EM 04-16, 05-01, 12-25, 12-26, 12-31 |
| 2002 | 6 | 01-01, GF 03-29, EM 04-01, 05-01, 12-25, 12-26 |
| 2003 | 6 | 01-01, GF 04-18, EM 04-21, 05-01, 12-25, 12-26 |
| 2004 | 3 | 01-01, GF 04-09, EM 04-12 |
| 2005 | 3 | GF 03-25, EM 03-28, 12-26 |
| 2006 | 5 | GF 04-14, EM 04-17, 05-01, 12-25, 12-26 |
| 2007 | 6 | 01-01, GF 04-06, EM 04-09, 05-01, 12-25, 12-26 |
| 2008 | 6 | 01-01, GF 03-21, EM 03-24, 05-01, 12-25, 12-26 |
| 2009 | 5 | 01-01, GF 04-10, EM 04-13, 05-01, 12-25 |
| 2010 | 3 | 01-01, GF 04-02, EM 04-05 |
| 2011 | 3 | GF 04-22, EM 04-25, 12-26 |
| 2012 | 5 | GF 04-06, EM 04-09, 05-01, 12-25, 12-26 |
| 2013 | 6 | 01-01, GF 03-29, EM 04-01, 05-01, 12-25, 12-26 |
| 2014 | 6 | 01-01, GF 04-18, EM 04-21, 05-01, 12-25, 12-26 |
| 2015 | 5 | 01-01, GF 04-03, EM 04-06, 05-01, 12-25 |
| 2016 | 4 | 01-01, GF 03-25, EM 03-28, 12-26 |
| 2017 | 5 | GF 04-14, EM 04-17, 05-01, 12-25, 12-26 |
| 2018 | 6 | 01-01, GF 03-30, EM 04-02, 05-01, 12-25, 12-26 |
| 2019 | 6 | 01-01, GF 04-19, EM 04-22, 05-01, 12-25, 12-26 |
| 2020 | 5 | 01-01, GF 04-10, EM 04-13, 05-01, 12-25 |
| 2021 | 3 | 01-01, GF 04-02, EM 04-05 |
| 2022 | 3 | GF 04-15, EM 04-18, 12-26 |
| 2023 | 5 | GF 04-07, EM 04-10, 05-01, 12-25, 12-26 |
| 2024 | 6 | 01-01, GF 03-29, EM 04-01, 05-01, 12-25, 12-26 |
| 2025 | 6 | 01-01, GF 04-18, EM 04-21, 05-01, 12-25, 12-26 |
| 2026 | 4 | 01-01, GF 04-03, EM 04-06, 05-01 (to 11-27) |

GF is Good Friday and EM Easter Monday, from the Gregorian Easter. The API
marks every date from 1999-12-31 to 2012-05-01 with `H` (section 3).

**Every one of the 134 is a TARGET closing day. None is not.** The
calendar used, from my own knowledge (no other source may be fetched):
- **Since 2000:** New Year's Day, Good Friday, Easter Monday, 1 May, 25 and
  26 December. Every year from 2000 on matches it exactly, apart from 31
  December 2001 (below): each of those days that fell on a weekday has no
  fixing, and every other weekday has one.
- **1999:** TARGET closed only on Christmas Day and New Year's Day, and,
  as a one-off, on 31 December 1999. Christmas 1999 fell on a Saturday, so
  31 December is the year's only weekday with no fixing. The data agree:
  Good Friday 1999-04-02 and Easter Monday 1999-04-05 have fixings.
- **31 December 2001** has no fixing and is marked `H`. I believe TARGET
  closed that day for the euro cash changeover, but have not verified it
  against an ECB document. In every other year whose 31 December fell on a
  weekday, that day has a fixing.

## 5. When a day's fixing is released

What the source shows:
- **The fixing time.** Every series carries the attribute `TITLE_COMPL`
  "ECB reference exchange rate, ..., 2.15 pm (C.E.T.)". The rates are
  those of 14:15 Central European time, so 13:15 UTC in winter and 12:15
  UTC in summer.
- **The publication time, for the last fixing.** Both files were last
  modified on Friday 2026-11-27, the zip at 14:56:38 GMT and the API at
  14:56:58 GMT. That is 15:56 CET, the ECB's "around 16:00 CET", and nine
  hours before 00:00 UTC on 2026-11-28.
- **The publication time, for 60 fixing dates: the `updatedAfter`
  probes.** The API returns an observation only if it was updated after
  the instant given. Each probe asked for one date D:

  | Fixing dates probed | after 12:00 UTC on D | after 13:30 to 15:30 UTC on D | after 16:00 to 22:00 UTC on D | after 00:00 UTC on D+1 |
  |---|---|---|---|---|
  | 40 drawn (seed 20261128) from the fixing dates since 2011-08-18, Bitstamp's first day: 2011-11-08 to 2026-07-28, 25 of them in summer time | 7 rows, on 40 of 40 | not asked | not asked | nothing, on 40 of 40 |
  | the other 15 of the last 20, 2026-11-02 to 2026-11-20 | 7 rows, on 15 of 15 | not asked | not asked | nothing, on 15 of 15 |
  | the last 5, 2026-11-23 to 2026-11-27 | 7 rows, on 5 of 5 | 7 rows at 13:30, 14:00, 14:30, 15:00 and 15:30 | nothing at 16:00, 16:30, 17:00, 18:00, 20:00 or 22:00 | nothing, on 5 of 5 |

  - So, as the API records it, each of the 60 fixings was last updated
    after 12:00 UTC on its own date and before 00:00 UTC on the next day.
    The 40 sampled since 2011 have not been changed since that day.
  - **On the last five days the probes put the update between 15:30 and
    16:00 UTC, an hour after the `Last-Modified` headers (14:56 UTC).** The
    likeliest reading is that the API compares `updatedAfter` with a time
    it keeps in Central European time (15:56 CET read as 15:56 UTC). The
    source cannot settle this. Under either reading each of the last five
    was last updated before 16:00 UTC on its own date.

**Confirmed: the fixing for date F is known before 00:00 UTC on F+1.**
- Margin: about nine hours on the last five days, where the probes
  bracket the update. On the other 55 dates they show only that it came
  before 00:00 UTC on D+1.
- Limits:
  - the API records only each observation's last update, not its first
    release. A value released early and never revised looks the same as
    one released late, if both come before the probe's instant. Neither
    matters here;
  - the 60 dates are a sample. The other fixing dates since 2011 were not
    probed one by one;
  - check 2 cannot see publication times, as with Coin Metrics.

## 6. The derived USD pairs: the ten largest daily log moves

Each series is built as in Phase 1 from the same fixing date, and a daily
move is ln(s_F / s_F′), where F′ is the fixing date before F. Each cell
gives F and the move; "from" gives F′ when it is not the weekday before F.
`usdsek` enters only `dxy`.

| Rank | eurusd | usdjpy | gbpusd | usdchf | audusd | usdcad | dxy |
|---|---|---|---|---|---|---|---|
| 1 | 2008-12-19 -0.0474 | 2024-08-05 -0.0460 (from 08-02) | 2016-06-24 -0.0816 | 2015-01-15 -0.1498 | 2008-10-24 -0.0737 | 2008-10-30 -0.0443 | 2009-03-19 -0.0388 |
| 2 | 2000-09-22 +0.0420 | 2022-11-11 -0.0459 | 2009-01-20 -0.0481 | 2011-09-06 +0.0819 | 2008-10-14 +0.0719 | 2008-10-22 +0.0362 | 2008-12-19 +0.0375 |
| 3 | 2009-03-19 +0.0404 | 2008-03-17 -0.0429 (from 03-14) | 2008-10-22 -0.0405 | 2008-12-18 -0.0564 | 2008-10-10 -0.0593 | 2008-10-10 +0.0358 | 2022-11-11 -0.0351 |
| 4 | 2008-12-18 +0.0389 | 2008-10-24 -0.0411 | 2009-03-19 +0.0389 | 2008-12-19 +0.0525 | 2008-10-08 -0.0565 | 2011-09-22 +0.0357 | 2000-09-22 -0.0287 |
| 5 | 2015-01-23 -0.0368 | 2013-06-07 -0.0353 | 2016-06-27 -0.0385 (from 06-24) | 2008-12-17 -0.0377 | 2007-08-16 -0.0466 | 2008-12-08 -0.0325 (from 12-05) | 2008-12-18 -0.0277 |
| 6 | 2022-11-11 +0.0349 | 2009-03-19 -0.0347 | 2008-10-24 -0.0349 | 2011-08-12 +0.0376 | 2008-10-30 +0.0446 | 2009-03-19 -0.0302 | 2016-06-24 +0.0264 |
| 7 | 2016-06-24 -0.0288 | 2016-06-24 -0.0325 | 2009-01-07 +0.0341 | 2009-03-19 -0.0369 | 2009-03-19 +0.0439 | 2009-01-26 -0.0291 (from 01-23) | 2015-01-23 +0.0259 |
| 8 | 2011-11-01 -0.0271 | 2011-08-04 +0.0324 | 2000-09-22 +0.0323 | 2022-11-11 -0.0339 | 2008-12-08 +0.0412 (from 12-05) | 2008-03-20 +0.0289 | 2025-04-03 -0.0243 |
| 9 | 2025-04-03 +0.0269 | 2013-06-10 +0.0323 (from 06-07) | 2022-11-11 +0.0322 | 2009-01-05 +0.0331 (from 01-02) | 2025-04-04 -0.0402 | 2016-03-17 -0.0287 | 2016-03-17 -0.0225 |
| 10 | 2008-12-17 +0.0266 | 2020-03-13 +0.0314 | 2009-03-09 -0.0311 (from 03-06) | 2009-03-12 +0.0305 | 2007-11-12 -0.0393 (from 11-09) | 2009-01-13 +0.0284 | 2008-10-22 +0.0223 |

- **They match known events**, a check that each pair is built the right
  way up:
  - 2015-01-15, when the Swiss National Bank ended its floor under EUR/CHF
    (a log move of −0.150 in USDCHF), and 2011-09-06, when it set that
    floor;
  - 2016-06-24, the day after the Brexit vote (−0.082 in GBPUSD);
  - the autumn of 2008;
  - 2022-11-11: US inflation figures came out on 2022-11-10, just after
    that day's fixing;
  - 2024-08-05, the yen's carry unwind.
- **The largest move of any pair, USDCHF on 2015-01-15, is inside the
  tables' history**, so it enters `fx_usdchf_ret_*` and
  `fx_usdchf_vol_*` on table C's rows.
- **On 2026-11-27 the pairs read:** eurusd 1.0957, usdjpy 159.54,
  gbpusd 1.3122, usdchf 0.8357, audusd 0.6669, usdcad 1.4245 and
  usdsek 10.229.

## 7. Sign check of the dollar index

ln(dxy) = −0.576 ln(eurusd) + 0.136 ln(usdjpy) − 0.119 ln(gbpusd) +
0.091 ln(usdcad) + 0.042 ln(usdsek) + 0.036 ln(usdchf), with no constant.
The table gives, for each pair, the share of fixing dates on which dxy
moved the expected way, over the days on which both moved. dxy never moved
by exactly zero.

| Pair | dxy should move | 1999-01-05 to 2026-11-27 (7,145 moves) | 2011-08-18 to 2026-11-27 (3,911 moves) |
|---|---|---|---|
| usdjpy | with it | 67.0% | 68.3% |
| usdchf | with it | 84.1% | 82.5% |
| usdcad | with it | 69.0% | 71.2% |
| eurusd | against it | 92.8% | 92.3% |
| gbpusd | against it | 78.5% | 78.0% |
| audusd (not asked) | against it | 72.2% | 71.9% |

Every share is a clear majority, so the construction has the right signs.
The daily change of ln(dxy) has a correlation of −0.972 with that of
ln(eurusd), as a 57.6% euro weight implies.

## 8. The first rows of tables C and D (rule 8)

I modelled the 43 FX columns as Phase 1 defines them: on the 7,146 fixing
dates, then carried to each BTC row t from the latest fixing date F ≤ t.
This is a model of when each column has a value, with the values
computed, not the build.

- **The 42 series columns.** Every fixing date has all seven currencies,
  so a window of fixings is complete once it lies inside the fixing
  history. The first fixing dates with a value are:
  - 1999-01-05 for `_ret_1`, 1999-03-29 for `_ret_60`, and 1999-02-01 for
    `_ret_20` and `_vol_20`;
  - 2000-01-17 for `_vol_z_250`: 250 values of `_vol_20`, each from 20
    log changes, so 270 fixings.

  So all 42 have a value on every BTC day from Bitstamp's first day,
  2011-08-18, on.
- **`btc_dxy_corr_60`**
  - Its first value is on **2012-01-30**. Its 60 pairs must lie inside
    Bitstamp's history, which starts on 2011-08-18. The zero-volume days
    of late 2011 (ruling 10) then hold it back further under the 90% rule,
    which needs 54 of the 60 pairs.
  - From 2012-01-30 to 2026-11-27, the only fixing dates with a BTC close
    missing at one end are 2015-01-06, 07, 08 and 09.
  - Pairs in the window: 60 on 2015-01-05, then 59, 58, 57, and 56 on
    2015-01-09 and 2015-01-12, rising back to 60 as the gap leaves the
    window. That is never below 54, so the 90% rule keeps a value.
  - **But 2015-01-08 is, by its own close, the third missing Bitstamp day
    in a row: a longer gap.** Under the point-in-time reading of the
    longer-gap rule ("Decided during Phase 1" in `README.md`), a day
    inside a longer gap has no value in any column built from that source.
    `btc_dxy_corr_60` is built from Bitstamp, so it is **blank on BTC row
    2015-01-08**. On 2015-01-06, 07 and 09 it has a value.
- **So all 43 FX columns have a value on every BTC day from 2012-01-30**,
  except 2015-01-08.
- **Table C: 2012-12-17 to 2026-11-27, 5,094 rows.**
  - Rule 8 alone gives C the same day as A, 2012-12-01, because the FX
    columns have values long before it. Ruling 12 moved A to 2012-12-17 so
    that no blank cell comes from the 2011 days, and the same reason holds
    for C. This is table A's own first row, as expected.
  - **One blank FX cell after the first row:** `btc_dxy_corr_60` on
    2015-01-08. The rules say: "If this happens after a table's first row,
    report the rows affected and stop for the user." The row affected is
    2015-01-08 alone.
  - Proposed: keep that row with the blank cell, as ruling 11 did for
    table A. The row is already blank in every Bitstamp column of A and
    has no target. Table C would then have 1,015 blank feature cells on
    49 rows: A's 1,014 and this one.
- **Table D: 2019-12-09 to 2026-11-27, 2,546 rows**, table B's own first
  row. No FX cell is blank.
- **Which fixing each row uses.**

  | Table | Rows using their own day's fixing (F = t) | One day back | Two | Three | Four |
  |---|---|---|---|---|---|
  | C | 3,568 | 745 | 732 | 31 | 18 |
  | D | 1,786 | 370 | 365 | 16 | 9 |

  One day back is a Saturday, or a holiday on a weekday. Two is mostly a
  Sunday. Three and four occur where a holiday joins a weekend: over
  Easter, the Sunday and the Monday use Thursday's fixing.
- **A preview of the level screen (check 7)**, on the model's values over
  table A's rows (earliest 75%: 2012-12-17 to 2023-06-02; latest 25%:
  2023-06-03 to 2026-11-27).
  - **No FX column is above 10%.**
  - The highest are `btc_dxy_corr_60` at 3.3% (early range −0.587 to
    0.401, late range −0.433 to 0.472), `fx_usdjpy_vol_z_250` at 1.2%, and
    `fx_usdcad_vol_z_250` and `fx_usdchf_ret_60` at 0.5%. Every other
    column is at 0.3% or less.
  - Every model value on C's rows is finite, apart from the blank of
    2015-01-08.
  - **Correction (Phase 1):** this preview left out the seven `_vol_20`
    columns. With them, `fx_usdcad_vol_20` is at 6.4% and
    `fx_usdjpy_vol_20` at 2.3%, still under 10% (`reports/fx.md`,
    section 4).

## 9. What I think is wrong or open in the prompt

1. **`--k 0.36` changes a file that must not change.**
   - Measured in memory, with nothing written: the current
     `build_dataset.py` on today's raw data writes every recorded file
     byte for byte when k is computed. With `--k 0.36`, the five CSV files
     are still identical, but `build_info.json` comes out at 5,626 bytes
     instead of 7,855.
   - The cause: the build records rule 5 (`k_rule`, and `class_shares`
     with the shares of the earliest 75%, the latest 25% and each year)
     only when it computes k. With k given, `k_rule` is null and
     `class_shares` is dropped.
   - So "build with `--k 0.36` from now on" and check 8 over every file of
     `data/built/` cannot both pass as the build stands. The present check
     6 has the same trouble: it builds without `--k` and compares with
     `data/`. Options:
     - **(a), recommended:** the build records rule 5 in `build_info.json`
       whenever table A has rows with a z, whether or not `--k` is given.
       k itself stays the given value. On today's data the file is then
       byte-identical, since rule 5 gives 0.36. Facts about C and D go to a
       new `data/built/fx_build_info.json`, never into `build_info.json`.
       Check 6 builds with `--k 0.36`.
     - **(b):** count `build_info.json` as a summary, not a table, leave it
       out of check 8, and let `--k 0.36` rewrite it.
2. **`btc_daily_full.csv` cannot gain the FX columns.** It is one of the
   recorded files, so any new column changes its bytes. The prompt's own
   condition therefore sends the FX columns to `data/built/fx_daily.csv`.
   - Proposed layout: `date`, then `fx_day` (the fixing date F the row
     used, like `cm_day`), then the 43 FX columns, on table A's rows
     (2012-12-17 to 2026-11-27).
   - `fx_day` would never be written to a profiler file.
3. **Rulings 22 onwards.** `README.md` numbers its rulings 10 to 20, then
   has an unnumbered "Decided during Phase 1". I found no ruling 21 in
   this folder. If none is reserved, the new rulings would start at 21.
4. **How check 2 cuts the ECB files.** Proposed:
   - the cut copy holds `exr_daily.csv`, keeping the rows with
     `TIME_PERIOD` ≤ t, which section 5 shows are known by t's close;
   - it also holds the zip, rewritten with the same rows, so that no cut
     copy holds a later fixing;
   - the probes are not copied, since the build does not read them;
   - besides the dates it has now, the 40 dates include a Saturday, a
     Sunday and a weekday TARGET closing day (for example Easter Monday
     2026-04-06, which uses the fixing of 2026-04-02).
5. **`btc_dxy_corr_60` correlates two intervals that are not the same.**
   This is not an error: it is what the prompt defines, and it is
   point-in-time. It matters for reading the results.
   - The dollar index changes from 14:15 CET on F′ to 14:15 CET on F.
     BTC's return runs from 00:00 UTC on F′+1 to 00:00 UTC on F+1.
   - On consecutive weekdays the two overlap for 13¼ of their 24 hours in
     winter and 12¼ in summer, so the correlation is weaker than a
     synchronous one would be.
   - Likewise, `dxy` is the index rebuilt from the 14:15 CET fixings, not
     ICE's published index.
6. **"TARGET closing days have none" holds, with one detail.** Before May
   2012 the API also returns an explicit empty row, status `H`, for each
   such day (section 3). The build must drop those rows rather than count
   them as fixing dates with values missing.
7. **The longer-gap stop** of section 8, which Phase 0's stop covers.

## 10. Files written in Phase 0, and how I worked

- `reports/fx_sources.md`, this report. Section 1 was written first, at
  12:15 UTC, before any script.
- `scripts/fetch_ecb.py`, the only code here that uses the network.
  - It ran twice. The first run, a few minutes earlier, had 65 probes and
    showed the last updates falling after 15:30 UTC.
  - I then extended the grid to 22:00 UTC and added the 40 sampled dates,
    and ran it again from 12:24 to 12:31 UTC. The second run replaced
    every file of the first.
- `data/raw/ecb/`: 177 files, 11,671,223 bytes.
- `data/raw/manifest.json`: the `ecb` entry is added. Without it, the
  manifest re-serialises byte for byte to the file as it was before this
  session: 11,567 bytes, SHA-256
  `78c02d3c6641856f245bf7af3526187d1984743556bbfc27551d561ae9a9f187`. I
  checked that the file re-serialised to its own bytes before the first
  fetch wrote it. That entry set, and the 25 file hashes in it, are the
  reference for check 8.
- **Nothing else.**
  - The measurements were run as code piped to the interpreter, so no
    inspection script was written ("write nothing else"). To make them
    repeatable, I would save that code in Phase 1 as
    `scripts/inspect_ecb.py`: reading only, like `inspect_sources.py`.
  - Neither the other fetch scripts nor ML Professor were run. No file of
    `data/raw/` outside `ecb/` changed.
- Every file was written through PowerShell. Nothing was blocked by the
  guard or a hook.

To repeat the fetch, run from `E:\CODE\ml-professor`:

```powershell
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe experiments\crypto\scripts\fetch_ecb.py
```
