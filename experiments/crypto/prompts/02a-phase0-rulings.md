# FX Phase 0 rulings, 2026-11-28

These answer `experiments\crypto\reports\fx_sources.md`. Record them in
`README.md` as rulings 21 to 27, with the reason given for each. The
point-in-time gap decision stays where it is, under "Decided during
Phase 1". Then carry out Phase 1 of `02-add-fx.md`.

21. **Source.** The build reads the ECB Data Portal file
    `data\raw\ecb\exr_daily.csv`. The historical zip is kept only as a
    cross-check, and the probes only as evidence of publication times.
    Neither is read by the build. Rows the API marks as holidays with no
    values are dropped: they are not fixing dates.
22. **`build_info.json` stays byte-identical.**
    - The build records rule 5 whether or not `--k` is given. When `--k` is
      given, it still computes rule 5's value from the data, records it
      exactly as before, and stops with an error if that value differs from
      the k given.
    - Built this way with `--k 0.36`, `build_info.json` must be
      byte-identical to the file recorded in Phase 0.
    - Anything about tables C and D goes to a new
      `data\built\build_info_cd.json`.
    - Check 8 exempts nothing.
23. **The blank `btc_dxy_corr_60` cell on 2015-01-08 is kept**, as ruling 11
    kept A's rows in the 2015 gap. That row already has every Bitstamp
    column blank and no target.
24. **The FX columns go to `data\built\fx_daily.csv`**, laid out as `date`,
    `fx_day` (the fixing date the row used), then the 43 FX columns in
    catalogue order. `btc_daily_full.csv` does not change.
25. **Check 2 cuts the ECB data by fixing date.** For each test date, the
    API file is cut to fixing dates on or before that date. The 40 dates
    include a Saturday, a Sunday and Easter Monday, 2026-04-06, in addition
    to those the prompt requires.
26. **Save the Phase 0 measurement code as `scripts\inspect_ecb.py`.** It
    reads only and writes nothing, like `inspect_sources.py`.
27. **`btc_dxy_corr_60` stays as specified.** Its two returns overlap by
    about 13 of 24 hours, so it reads weaker than a correlation of returns
    taken at the same time would. State this in the catalogue. The feature
    set is fixed, so do not change it.

Everything else in `02-add-fx.md` stands. When Phase 1 is done, stop. Do
not run ML Professor.
