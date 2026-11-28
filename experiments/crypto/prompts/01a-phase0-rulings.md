# Phase 0 rulings, 2026-11-28

These answer `experiments\crypto\reports\sources.md`. Record them in
`README.md` after the settled choices, as rulings 10 to 20, with the reason
given for each. Then carry out Phase 1 of `01-build-dataset.md`.

10. **Zero-volume Bitstamp days** are missing Bitstamp days, never filled
    (report section 11, kind 1).
11. **The 2015-01-06 to 2015-01-08 gap: option (a).** Keep those rows with
    the blank cells listed in section 11. Their targets stay blank in the
    profiler files; ML Professor drops and counts those rows itself.
12. **Table A's first row is 2012-12-17**, so that no blank cell comes from
    the 2011 days. That gives 5,094 rows to 2026-11-27. k (rule 5) is
    computed on this table.
13. **Table B's first row is 2019-12-09**, giving 2,546 rows.
14. **The on-chain set is AdrActCnt, TxCnt, HashRate and FeeTotNtv.**
    - Drop SplyCur and its three columns. Its growth is the issuance
      schedule, which steps down at each halving. Since April 2024 it sits
      below every value of the earlier years, so it marks the era rather
      than carrying information, and the cycle is already in
      `days_since_halving` and `halving_phase`.
    - Build `mvrv` = ln(CapMVRVCur) and `mvrv_z365`: CapMVRVCur is the ratio
      the prompt asks for.
    - `nvt_30` and `nvt_z365` are not built. Say so in the catalogue.
15. **Coin Metrics timing: an as-of rule, instead of the missing-day
    treatment proposed in section 7.3.**
    - On row t, every Coin Metrics value is the value for the latest day D,
      with D no later than t−1, that was known at the close of t (00:00 UTC
      on t+1).
    - From 2021-02-24 on, day D counts as known when its
      AssetEODCompletionTime is at or before that close.
    - Before 2021-02-24 the recorded completion times are a 2021 backfill.
      There, day D counts as known at the close of D+1, which is the
      one-day lag as written.
    - On most rows D is t−1. On the eight rows in section 7.3 it is an
      earlier day. Those rows are therefore not gaps, and no row loses a
      value.
    - The `_g7`, `_g30`, `_z365` and `mvrv` columns are computed on this
      as-of series: `_g7` on row t is ln(the as-of value on t / the as-of
      value on t−7).
    - State in `README.md` that publication times before 2021-02-24 are
      unknown, and that Coin Metrics may have revised values since. Check 2
      cannot detect either.
    - Check 2 cuts Coin Metrics by this rule: by completion time from
      2021-02-24, and by date before.
16. **The 2024 halving stays at 2024-04-20.**
17. **The four proposals in section 12, points 1 to 4, are accepted as
    written:**
    - recursive averages step over a day with no input, and are reported
      when the day's input exists and 90% of the last three-times-span days
      are present, never inside a longer gap;
    - days before a source's first day are not gaps, and a window lies
      wholly inside its source's history;
    - a z-score needs its own day's value as well as 90% of its window;
    - funding history starts on 2019-09-11, with no `fund_1d` on
      2019-09-10;
    - a source's gaps blank only the columns built from that source.
18. **Keep `endTime`** on the Binance requests.
19. **Keep `scripts\inspect_sources.py`.** It reads only and writes nothing.
20. **Keep writing files through PowerShell, as in Phase 0.** The kit treats
    this session as a build session, so the file-editing tools would run the
    library's gate after every edit. Write nothing outside
    `experiments\crypto`.

Everything else in `01-build-dataset.md` stands. When Phase 1 is done, stop.
Do not run ML Professor.
