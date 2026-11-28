# BTC daily dataset for ML Professor

One row per UTC calendar day of Bitcoin, priced on Bitstamp BTC/USD, with
features known at the close of each day and two binary targets:
- `up_7d`: the next seven days were a rally;
- `down_7d`: they were a sell-off.

Each target is profiled separately by ML Professor, a separate step that
this folder does not take.

There are four tables:
- **Table A** holds price, volume, candle, calendar and on-chain features
  (Bitstamp and Coin Metrics Community). It runs 2012-12-17 to 2026-11-27,
  5,094 rows.
- **Table B** holds every column of table A plus the Binance features (spot,
  USD-M perpetual and funding). It runs 2019-12-09 to 2026-11-27, 2,546 rows.
- **Table C** holds every column of table A plus 43 FX columns from the
  ECB's euro reference rates: six USD pairs and a rebuilt dollar index. It
  has table A's rows.
- **Table D** holds every column of table B plus the same 43 FX columns, on
  table B's rows.

Tables C and D were added after A and B had been profiled. A and B did not
change by a single byte (ruling 29, check 8), so C and D can be compared with
them on the same holdout days.

The reports:
- `reports/sources.md` (Phase 0: the sources, measured) and
  `reports/dataset.md` (Phase 1: tables A and B, k, the checks);
- `reports/fx_sources.md` (Phase 0 of the FX columns: the ECB rates,
  measured) and `reports/fx.md` (Phase 1: tables C and D, the checks).

## Layout

```
experiments/crypto/
  README.md                      this file
  .gitignore                     data/
  scripts/fetch_bitstamp.py      the only code that touches the network,
  scripts/fetch_coinmetrics.py   one script per source; each saves every
  scripts/fetch_binance.py       response as received and its manifest entry
  scripts/fetch_ecb.py
  scripts/inspect_sources.py     the Phase 0 measurements of A and B (reads only)
  scripts/inspect_ecb.py         the Phase 0 measurements of the ECB rates (reads only)
  scripts/labels.py              targets and k, used by build_dataset.py
  scripts/features.py            feature columns, used by build_dataset.py
  scripts/build_dataset.py       raw data to built tables
  scripts/check_dataset.py       independent checks; exit 0 pass, 1 fail
  scripts/compare_b.py, scripts/read_runs.py   readings of the ML Professor runs; not part of the build
  reports/sources.md, reports/dataset.md, reports/fx_sources.md, reports/fx.md
  runs/                          ML Professor's runs on the profiler files; not part of the build
  data/raw/                      downloads as received, and manifest.json
  data/raw/ecb/                  the ECB API file, the historical zip, and probes/ (ruling 21)
  data/built/btc_daily_full.csv  every A and B column, labels included (table A's rows)
  data/built/build_info.json     k, the class shares, rows and blank cells of A and B
  data/built/fx_daily.csv        the 43 FX columns on table A's rows (ruling 24)
  data/built/build_info_cd.json  rows and blank cells of C and D, and the ECB rates read
  data/profiler/a_up_7d.csv, a_down_7d.csv, b_up_7d.csv, b_down_7d.csv
  data/profiler/c_up_7d.csv, c_down_7d.csv, d_up_7d.csv, d_down_7d.csv
  data/profiler/a_short_up_7d.csv, a_short_down_7d.csv   table A from 2019-12-09; the lead's, not the build's
```

Each profiler file holds `date`, the table's features, then one target.
`btc_daily_full.csv` holds table A's rows with every feature of both
tables, followed by:
- `cm_day`, the Coin Metrics day the row used;
- `fwd_ret_7` = ln(C_{t+7}/C_t);
- `z_7`, `class_7` (UP, RANGE or DOWN), `up_7d` and `down_7d`.

The Binance columns there are blank before their windows lie inside
Binance's history.

`fx_daily.csv` holds table A's rows, with:
- `date`;
- `fx_day`, the ECB fixing date the row used;
- the 43 FX columns, in catalogue order.

`btc_daily_full.csv` does not change (ruling 24). `data/` is not committed.
The checks make their temporary folders under `data/scratch/` and remove
them.

## Rebuilding

Run from `E:\CODE\ml-professor`, with the project's interpreter:

```powershell
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe experiments\crypto\scripts\fetch_bitstamp.py
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe experiments\crypto\scripts\fetch_coinmetrics.py
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe experiments\crypto\scripts\fetch_binance.py
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe experiments\crypto\scripts\fetch_ecb.py
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe experiments\crypto\scripts\build_dataset.py --k 0.36
E:\CODE\ml-professor\mlprof_env\Scripts\python.exe experiments\crypto\scripts\check_dataset.py
```

- **Tables A and B are fixed** (rulings 29 and 30). Do not fetch again for
  them: a new fetch moves the last row to the last UTC day completed at
  that time, and every count in the reports moves with it. The four fetch
  scripts are listed for the record. Run only the last two lines to
  rebuild and check.
- `fetch_ecb.py` fetches through the Bitstamp entry's last completed day in
  the manifest, so it runs after the other three. Fetch all four sources on
  the same UTC day.
- `build_dataset.py` takes four options:
  - `--raw FOLDER` reads another raw folder with the same layout and
    manifest;
  - `--out FOLDER` writes `built/` and `profiler/` under another folder;
  - `--k VALUE` fixes k. Build with `--k 0.36` (ruling 31). Rule 5 is still
    computed and recorded, and the build stops if it gives another value
    (ruling 22);
  - `--k-unchecked` turns that stop off. Only check 2 passes it, for its
    copies of the raw data cut at earlier dates, and the build refuses it
    on `data/raw` itself (ruling 22).
- The build takes about 5 seconds. The checks take about 35, which includes
  44 builds run as separate processes.
- The scripts resolve every path from their own location. Only the four
  fetch scripts use the network. They use no key.

## Settled choices

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

## Rulings on Phase 0 (2026-11-28)

10. **Zero-volume Bitstamp days are missing Bitstamp days, never filled.**
    Reason: no trade took place on them. The candle repeats the previous
    close (33 of 33 days), while other markets moved (2015-01-06..08: 276.80
    on Bitstamp against 287.55, 297.54 and 284.34 at Coin Metrics). A zero
    volume would also make ln(DV) infinite.
11. **The 2015-01-06 to 2015-01-08 gap: option (a).** Those rows are kept
    with their blank cells. Their targets stay blank in the profiler files,
    and ML Professor drops and counts those rows itself. Reason: it is what
    the missing-data rules prescribe, and every other row is untouched.
12. **Table A's first row is 2012-12-17**, so that no blank cell comes from
    the 2011 zero-volume days. That gives 5,094 rows to 2026-11-27. k (rule
    5) is computed on this table. Reason: rule 8 alone would start on
    2012-12-01, but `ret_365` would then be blank on 2012-12-03, 12-08 and
    12-16, a year after the isolated zero-volume days of December 2011.
13. **Table B's first row is 2019-12-09**, giving 2,546 rows. Reason: it is
    rule 8's first day. The binding column is `fund_z_90`, 90 days of
    `fund_1d` from 2019-09-11.
14. **The on-chain set is AdrActCnt, TxCnt, HashRate and FeeTotNtv.**
    Reasons:
    - SplyCur and its three columns are dropped. Its growth is the issuance
      schedule, which steps down at each halving. Since April 2024 it sits
      below every value of the earlier years, so it marks the era rather
      than carrying information, and the cycle is already in
      `days_since_halving` and `halving_phase`.
    - `mvrv` = ln(CapMVRVCur) and `mvrv_z365` are built, because CapMVRVCur
      is the ratio the prompt asks for, although realised cap itself is not
      free.
    - `nvt_30` and `nvt_z365` are not built: adjusted transfer value is not
      free.
15. **Coin Metrics timing is an as-of rule, instead of a missing-day
    treatment.**
    - On row t, every Coin Metrics value is the value for the latest day D,
      with D no later than t−1, that was known at the close of t (00:00 UTC
      on t+1).
    - From 2021-02-24 on, day D counts as known when its
      AssetEODCompletionTime is at or before that close.
    - Before 2021-02-24 the recorded completion times are a 2021 backfill.
      There, day D counts as known at the close of D+1, which is the
      one-day lag as written.
    - On most rows D is t−1. On eight rows it is an earlier day:
      2021-02-25, 2024-08-15..18 and 2025-05-02..04. Those rows are not
      gaps, and no row loses a value.
    - The `_g7`, `_g30`, `_z365` and `mvrv` columns are computed on this
      as-of series: `_g7` on row t is ln(the as-of value on t / the as-of
      value on t−7).
    - **Publication times before 2021-02-24 are unknown, and Coin Metrics
      may have revised values since. Check 2 cannot detect either.**
    - Check 2 cuts Coin Metrics by this rule: by completion time from
      2021-02-24, and by date before.
    - Reason: the one-day lag as written would use, on those eight rows, a
      value that was published after the row's close.
16. **The 2024 halving stays at 2024-04-20.** Reason: Coin Metrics' daily
    data dates the halving block one day earlier (four blocks at the new
    reward on 2024-04-19). The settled date is the first whole UTC day at
    the new reward, and the difference touches one row.
17. **The four readings of the missing-data rules** proposed in
    `reports/sources.md`, section 12, are accepted:
    - recursive averages step over a day with no input. They are reported
      when the day's input exists and 90% of the last three-times-span days
      are present, and never inside a longer gap;
    - days before a source's first day are not gaps, and a window lies
      wholly inside its source's history;
    - a z-score needs its own day's value as well as 90% of its window;
    - funding history starts on 2019-09-11, with no `fund_1d` on 2019-09-10
      (that day has two prints because the series begins at 08:00);
    - a source's gaps blank only the columns built from that source.

    Reason: the prompt's rules leave each of these open.
18. **`endTime` is kept on the Binance requests.** Reason: without it, the
    current day's kline and the 00:00 print of today would be received.
19. **`scripts/inspect_sources.py` is kept.** It reads only and writes
    nothing. Reason: the Phase 0 measurements can be repeated.
20. **Files are written through PowerShell, as in Phase 0, and nothing is
    written outside `experiments/crypto`.** Reason: the repository's kit
    treats this session as a build session, so the file-editing tools would
    run the library's gate after every edit.

## Decided during Phase 1

- **Longer gaps are read point-in-time** (the owner's answer, 2026-11-28).
  A missing day is inside a longer gap only once the run of missing days
  ending on it has reached three, as known at that day's close. So
  2015-01-08 has no Bitstamp-based value. On 2015-01-06 and 2015-01-07, 30
  and 28 rolling columns carry the values an isolated gap allows, where
  section 11 of `reports/sources.md` had listed them blank. Reason: read
  with hindsight, the blank cells of those two days depend on whether
  Bitstamp stayed shut afterwards, and check 2 fails on them. Neither row
  has a target.
- **`vol_cc_365` is not a column** (the level screen, check 7; see the
  catalogue).
- **A Coin Metrics value of zero or less is a missing day of that metric.**
  It has no logarithm. This happens only for TxCnt, FeeTotNtv and AdrActCnt
  in 2009-2010, outside every window of the tables.

## Rulings on the FX Phase 0 (2026-11-28)

These answer `reports/fx_sources.md`.

21. **Source.** The build reads the ECB Data Portal file
    `data/raw/ecb/exr_daily.csv`. The historical zip is kept only as a
    cross-check, and the probes only as evidence of publication times.
    Neither is read by the build. Rows the API marks as holidays with no
    values are dropped: they are not fixing dates.
    - Reason: the API and the zip agree on all 50,022 values. The server
      cuts the API at `endPeriod`, the API marks each observation's status,
      and the probes measure its own records.
    - The 62 holiday rows (1999-12-31 to 2012-05-01) are TARGET closing
      days with no value in any currency. Counted as fixing dates, they
      would put a gap into windows before May 2012 and none after.
22. **`build_info.json` stays byte-identical.**
    - The build records rule 5 whether or not `--k` is given. When `--k` is
      given, it still computes rule 5's value from the data, records it
      exactly as before, and stops with an error if that value differs from
      the k given.
    - Built this way with `--k 0.36`, `build_info.json` is byte-identical
      to the file recorded in Phase 0.
    - Anything about tables C and D goes to `data/built/build_info_cd.json`.
    - Check 8 exempts nothing.
    - **`--k-unchecked`** turns the stop off, and only check 2 passes it.
      The build refuses it when `--raw` is the real `data/raw` folder, so
      no build of the real data can skip the k check (the owner's
      condition).
    - Reason: as the build stood, `--k` dropped rule 5's record from
      `build_info.json` (`k_rule` null and no class shares), so "build with
      `--k 0.36`" changed a file that must not change. The stop keeps k
      "then fixed" honest.
    - Check 2's copies of the raw data cut at earlier dates have a shorter
      history, on which rule 5 gives another k (on 29 of its 40 dates), so
      they would stop without `--k-unchecked`.
23. **The blank `btc_dxy_corr_60` cell on 2015-01-08 is kept**, as ruling
    11 kept A's rows in the 2015 gap. Reason: that row already has every
    Bitstamp column blank and no target.
24. **The FX columns go to `data/built/fx_daily.csv`**, laid out as
    `date`, `fx_day` (the fixing date the row used), then the 43 FX columns
    in catalogue order. `btc_daily_full.csv` does not change. Reason: it is
    one of the files that must stay byte-identical, so it cannot gain
    columns.
25. **Check 2 cuts the ECB data by fixing date.** For each test date, the
    API file is cut to fixing dates on or before that date.
    - The 40 dates include a Saturday, a Sunday and Easter Monday,
      2026-04-06, in addition to those the prompt requires. Check 2 always
      includes 2026-04-04 to 2026-04-06, which all carry the fixing of
      Thursday 2026-04-02.
    - Reason: a fixing of F is published by about 16:00 CET on F
      (`reports/fx_sources.md`, section 5), before F's close. The
      weekend and the holiday test the carry rule.
26. **The Phase 0 measurement code is saved as `scripts/inspect_ecb.py`.**
    It reads only and writes nothing, like `inspect_sources.py`. Reason:
    the measurements can be repeated.
27. **`btc_dxy_corr_60` stays as specified.** Its two returns overlap by
    about 13 of 24 hours, so it reads weaker than a correlation of returns
    taken at the same time would. The catalogue says so. Reason: the
    feature set is fixed (ruling 28).

## Rules of the FX session (02-add-fx.md, 2026-11-28)

`02-add-fx.md` asks for its rules to be recorded from ruling 22, but rulings
21 to 27 went to the FX Phase 0 rulings. Its rules are therefore 28 to 34.

28. **The FX feature set is fixed in advance:** the 43 columns of the
    catalogue. They are profiled once and compared with tables A and B on
    the same holdout days. No FX feature is added, dropped or tuned to
    improve anything. Reason: a set chosen after seeing results would make
    that comparison meaningless.
29. **Tables A and B do not change by a single byte.**
    - The SHA-256 of every file in `data/built/` and `data/profiler/` was
      recorded before any script changed (`reports/fx_sources.md`,
      section 1), and check 8 holds them.
    - `a_short_up_7d.csv` and `a_short_down_7d.csv` were written by the
      project's lead: table A's rows from 2019-12-09. The build never
      writes them.
    - Reason: A and B have been profiled, and their results stand.
30. **No new days.**
    - `fetch_bitstamp.py`, `fetch_coinmetrics.py` and `fetch_binance.py`
      are not run again, and no existing file of `data/raw/` changes.
    - The ECB rates are fetched through 2026-11-27, the last day of A and B.
    - Reason: new days would change tables A and B and their holdouts.
31. **k is fixed at 0.36: build with `--k 0.36`.** Reason: rule 5 fixes k
    ("then fixed"). A later fetch adds rows to the earliest 75% and could
    move rule 5's value, and with it every class.
32. **The FX point-in-time rule.**
    - The fixing for date F is known at the close of BTC day F (00:00 UTC
      on F+1). Row t uses the latest fixing date F on or before t. On a
      weekend or a TARGET closing day, that is the last working day before
      it.
    - Every FX feature is computed on the sequence of fixing dates, then
      carried to each BTC row. A window of n fixings means the n fixing
      dates ending with F, never calendar days.
    - Only trailing statistics.
    - Reason: the ECB publishes at about 16:00 CET. On 2026-11-27 both
      files were last modified at 14:56 UTC, and every probed fixing was
      last updated before 00:00 UTC on the next day. The rates come once
      per working day.
33. **Missing data on the fixing sequence.**
    - The missing-data rules of ruling 17 apply to the fixing sequence. A
      fixing date with any of the seven currencies missing has no value for
      any series. There is no such date.
    - `btc_dxy_corr_60` leaves out a fixing date whose BTC close, on either
      end, is missing, and needs 90% of its 60 pairs.
    - Reason: the rules of tables A and B, applied to the unit the rates
      come in.
34. **How the session worked.**
    - Network access only in `scripts/fetch_ecb.py`.
    - Every file written through PowerShell, and only under
      `experiments/crypto`; nothing under `prompts/` edited.
    - ML Professor was not run.
    - Reason: the repository's kit treats the session as a build session
      of the library (ruling 20), whose own rules forbid network access.
      Profiling is the owner's separate step.

## Point-in-time and missing data, as built

- **When a value is known.** A feature on row t uses only values known at
  the close of day t, 00:00 UTC on t+1.
  - Bitstamp and Binance values of day t are known at that close.
  - Coin Metrics follows ruling 15. PriceUSD is not a feature.
  - Funding times are floored to the hour, and row t takes the prints at
    00:00, 08:00 and 16:00 of day t.
  - There are no centred windows, two-sided filters or whole-sample
    statistics. The only constant fixed from the data is k.
- **Windows.**
  - A window of n days needs ceil(0.9·n) of its days present, so all 7 of
    7, 13 of 14, 27 of 30 and 329 of 365.
  - A window lies inside its source's history.
  - A return needs both of its ends.
  - A z-score also needs its own day's value.
  - A recursive average steps over days with no input, and needs its input
    on the day and 90% of its last 3 × span days.
- **Longer gaps and sources.**
  - A day that is, by its own close, the third or later missing day in a
    row has no value in any column of that source (see above).
  - A source's gaps blank only its own columns. Calendar and on-chain
    columns keep their values on 2015-01-06..08.
- **Implementation choices.**
  - Every standard deviation, including Bollinger's, is ddof 1.
  - Skewness and kurtosis are the sample (bias-corrected) G1 and G2.
  - The RSI averages daily changes of the close.
  - Aroon looks at today and the 25 days before.
  - A z-score whose window has no spread would be 0; no such window occurs.
  - The earliest 75% of rule 5 is the first floor(0.75·n) rows with a z.
    The quantile is numpy's default (linear interpolation), and k is
    Python's round(·, 2) of it.
- **Table first rows.** The first rows are those of rulings 12 and 13, and
  tables C and D have the first rows of A and B. The build refuses to
  write if any feature column of a table has no value on its first row.
- **FX columns (tables C and D).**
  - The ECB fixing of date F is known at the close of BTC day F (ruling 32).
    Row t takes the values of the latest fixing date F on or before t. On a
    weekend or a TARGET closing day, that is the last working day before
    it.
  - Every FX column is computed on the sequence of the 7,146 fixing dates,
    whose windows count fixings, not days. The rules above apply to that
    sequence: 90% of a window, both ends of a return, the z-score's own
    value, and a window inside the fixing history (ruling 33).
  - No fixing date lacks one of the seven currencies, so no FX series has a
    gap. The API's 62 holiday rows are not fixing dates (ruling 21).
  - `btc_dxy_corr_60` is also a Bitstamp column. A fixing date whose BTC
    close on either end is missing is left out of its window, and it has no
    value on a day inside a longer Bitstamp gap. That happens once, on
    2015-01-08 (ruling 23).

## Feature catalogue

C, O, H, L and V are Bitstamp's close, open, high, low and volume (BTC) of
day t, and r_t = ln(C_t / C_{t−1}) is the daily log return. σ30_t is the
standard deviation (ddof 1) of the 30 r ending with t. Every window ends at
row t, inclusive, and gives a value only under the missing-data rules
(below). "Lag" is when the inputs of row t are known:
- **close**: at the close of day t, 00:00 UTC on t+1;
- **as-of**: Coin Metrics' value for the latest day D ≤ t−1 that was known
  at that close (ruling 15). D is t−1 on every row but eight;
- **fixing**: the ECB fixing of the latest fixing date F ≤ t, known at the
  close of F (ruling 32). The FX columns are catalogued at the end.

The table column says which tables carry the feature. Every column of table
A is also in table C, and every column of table B in table D. k = 0.36
(rule 5).

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

Table A has 99 feature columns and table B 119. Table C has 142 (A's 99
and the 43 FX columns) and table D 162 (B's 119 and the 43). A trailing
z-score of x over n days is (x_t − mean) / standard deviation (ddof 1), both taken over the n
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

### FX columns (tables C and D)

Each series s is built from the EUR rates of one fixing date. A fixing
date that lacks any of the seven currencies has no value in any series
(there is none).

| s | Series |
|---|---|
| `eurusd` | USD per EUR, as published |
| `usdjpy` | JPY per EUR / USD per EUR |
| `gbpusd` | USD per EUR / GBP per EUR |
| `usdchf` | CHF per EUR / USD per EUR |
| `audusd` | USD per EUR / AUD per EUR |
| `usdcad` | CAD per EUR / USD per EUR |
| `dxy` | the ICE dollar index rebuilt from the crosses, in logs: ln dxy = −0.576 ln eurusd + 0.136 ln usdjpy − 0.119 ln gbpusd + 0.091 ln usdcad + 0.042 ln usdsek + 0.036 ln usdchf, where usdsek = SEK per EUR / USD per EUR. The index's constant is left out, since only changes are used. It is rebuilt from the 14:15 CET fixings, not ICE's published index. usdsek has no columns of its own |

Below, F is the latest fixing date on or before row t, F_{−n} the fixing
date n fixings before F, and a daily change runs from one fixing date to
the next.

| Column | Formula | Window, in fixings | Source | Lag | Table |
|---|---|---|---|---|---|
| `fx_s_ret_1`, `fx_s_ret_5`, `fx_s_ret_20`, `fx_s_ret_60` | ln(s_F / s_{F−n}): the log change over the last n fixings, ending at F | its two ends, F and F_{−n} | ECB | fixing | C, D |
| `fx_s_vol_20` | standard deviation (ddof 1) of the 20 daily log changes of s ending at F | 20 changes (21 fixings) | ECB | fixing | C, D |
| `fx_s_vol_z_250` | trailing 250-fixing z-score of ln(`fx_s_vol_20`) | 250 + 20 | ECB | fixing | C, D |
| `btc_dxy_corr_60` | correlation, over the 60 fixing dates ending at F, between the change of ln(dxy) from the fixing date before, F′, to F, and BTC's log return from Bitstamp's close of F′ to its close of F. A date whose BTC close is missing at either end is left out, and 54 of the 60 are needed. **The two returns are not taken at the same time** (ruling 27): dxy changes from 14:15 CET to 14:15 CET, and BTC from 00:00 UTC to 00:00 UTC. On consecutive weekdays they overlap for about 13 of their 24 hours, so this reads weaker than a correlation of returns taken at the same time | 60 pairs (61 fixings) | ECB, Bitstamp | fixing | C, D |

Each of the first three rows stands for seven columns, one for each series
s. There are 43 FX columns in all. In order: for `eurusd`, `usdjpy`,
`gbpusd`, `usdchf`, `audusd`, `usdcad` and `dxy` in turn, `_ret_1`,
`_ret_5`, `_ret_20`, `_ret_60`, `_vol_20` and `_vol_z_250`, named `fx_`
followed by s; then `btc_dxy_corr_60`.

- Every FX column has a value on every row of tables C and D but one:
  `btc_dxy_corr_60` on 2015-01-08, inside Bitstamp's 2015 gap (ruling 23).
- On a Saturday a row carries Friday's fixing, on a Sunday it carries the
  same, and over Easter the Sunday and the Monday carry Thursday's.
  `fx_day` in `fx_daily.csv` says which fixing each row used.
- **Level screen (check 7):** no FX column is above 10%.
  - The highest are `fx_usdcad_vol_20` at 6.4%, where USDCAD's 20-fixing
    volatility fell below its 2012-2023 low in ten quiet spells from 2024
    to 2026, and `btc_dxy_corr_60` at 3.3%, above its earlier high in
    autumn 2024.
  - Next is `fx_usdjpy_vol_20` at 2.3%, below its earlier low in June
    2026. No FX column marks the era.
