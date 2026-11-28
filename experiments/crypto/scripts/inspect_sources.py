"""Measure the raw downloads for reports/sources.md (Phase 0).

Reads data/raw/ only and writes nothing; prints every measurement the
report uses. Paths resolve from this file, so it runs from any folder.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "raw"
DAY = pd.Timedelta(days=1)
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
HALVINGS = {"2012-11-28": 210_000, "2016-07-09": 420_000,
            "2020-05-11": 630_000, "2024-04-20": 840_000}
KLINE_FIELDS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def section(title: str) -> None:
    print()
    print("## " + title)


def by_year(dates) -> str:
    counts = Counter(pd.Timestamp(d).year for d in dates)
    return ", ".join(f"{year}: {n}" for year, n in sorted(counts.items())) or "none"


def describe_missing(missing) -> str:
    missing = list(missing)
    if not missing:
        return "none"
    if len(missing) <= 30:
        return ", ".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in missing)
    return f"{len(missing)} days; by year: {by_year(missing)}"


def missing_days(dates: pd.DatetimeIndex, first=None, last=None) -> pd.DatetimeIndex:
    first = dates.min() if first is None else first
    last = dates.max() if last is None else last
    return pd.date_range(first, last, freq="D").difference(dates)


def runs(dates) -> list[tuple[pd.Timestamp, int]]:
    """Consecutive runs of days: (first day, length)."""
    out = []
    for day in sorted(pd.Timestamp(d) for d in dates):
        if out and day == out[-1][0] + out[-1][1] * DAY:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((day, 1))
    return out


def field_types(records) -> str:
    types: dict[str, Counter] = {}
    for record in records:
        for key, value in record.items():
            types.setdefault(key, Counter())[type(value).__name__] += 1
    return "; ".join(f"{k}: " + "/".join(f"{t}" + (f" x{n}" if len(c) > 1 else "")
                                        for t, n in c.items())
                     for k, c in types.items())


# ------------------------------------------------------------------ load --

def load_bitstamp():
    candles = {}
    for path in sorted((RAW / "bitstamp").glob("ohlc_*.json")):
        for candle in json.loads(path.read_bytes())["data"]["ohlc"]:
            candles[int(candle["timestamp"])] = candle
    records = [candles[k] for k in sorted(candles)]
    frame = pd.DataFrame(records)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"].astype("int64"), unit="s"),
                                   name="date")
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = frame[col].astype(float)
    return frame, records


def load_cm():
    rows: dict[str, dict] = {}
    for path in sorted((RAW / "coinmetrics").glob("series_g*_p*.json")):
        for row in json.loads(path.read_bytes())["data"]:
            rows.setdefault(row["time"], {}).update(row)
    records = [rows[k] for k in sorted(rows)]
    frame = pd.DataFrame(records)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame["time"].str[:10]), name="date")
    catalog = {}
    for name in ("catalog_free", "catalog_all"):
        found = {}
        for path in sorted((RAW / "coinmetrics").glob(f"{name}_p*.json")):
            for item in json.loads(path.read_bytes())["data"]:
                for metric in item["metrics"]:
                    for freq in metric["frequencies"]:
                        if freq["frequency"] == "1d":
                            found[metric["metric"]] = freq
        catalog[name] = found
    return frame, records, catalog


def load_klines(stem: str):
    rows = {}
    for path in sorted((RAW / "binance").glob(f"{stem}_*.json")):
        for row in json.loads(path.read_bytes()):
            rows[row[0]] = row
    raw = [rows[k] for k in sorted(rows)]
    records = [dict(zip(KLINE_FIELDS, row)) for row in raw]
    frame = pd.DataFrame(records)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame["open_time"].astype("int64"), unit="ms"),
                                   name="date")
    for col in ("open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_base", "taker_buy_quote"):
        frame[col] = frame[col].astype(float)
    return frame, records


def load_funding():
    prints = {}
    for path in sorted((RAW / "binance").glob("funding_*.json")):
        for row in json.loads(path.read_bytes()):
            key = (row["fundingTime"], row["fundingRate"])
            prints[key] = row
    records = [prints[k] for k in sorted(prints)]
    frame = pd.DataFrame(records)
    frame["time"] = pd.to_datetime(frame["fundingTime"].astype("int64"), unit="ms")
    frame["offset_ms"] = frame["fundingTime"].astype("int64") % HOUR_MS
    frame["hour_start"] = pd.to_datetime(
        (frame["fundingTime"].astype("int64") // HOUR_MS) * HOUR_MS, unit="ms")
    frame["day"] = frame["hour_start"].dt.normalize()
    frame["hour"] = frame["hour_start"].dt.hour
    frame["rate"] = frame["fundingRate"].astype(float)
    return frame, records


# --------------------------------------------------------------- reports --

def report_bitstamp(bs, records):
    section("Bitstamp BTC/USD daily")
    print(f"first {bs.index.min().date()}, last {bs.index.max().date()}, "
          f"{len(bs)} distinct days ({sum(1 for _ in (RAW / 'bitstamp').glob('*.json'))} files, "
          f"overlapping candles identical)")
    print("fields and JSON types:", field_types(records))
    print("missing calendar days:", describe_missing(missing_days(bs.index)))
    zero = bs.index[bs["volume"] == 0]
    flat = bs.index[(bs["open"] == bs["high"]) & (bs["high"] == bs["low"]) & (bs["low"] == bs["close"])]
    both = zero.intersection(flat)
    print(f"zero volume: {len(zero)} days; by year: {by_year(zero)}")
    print(f"open = high = low = close: {len(flat)} days; by year: {by_year(flat)}")
    print(f"both: {len(both)}; zero volume but not flat: {len(zero.difference(flat))}; "
          f"flat with volume > 0: {len(flat.difference(zero))}")
    if len(zero):
        carried = (bs["close"].shift(1).reindex(zero) == bs.loc[zero, "close"]).mean()
        print(f"zero-volume days whose close equals the previous close: {carried:.1%}")
        zr = runs(zero)
        print("runs of zero-volume days, length: count ->",
              dict(sorted(Counter(n for _, n in zr).items())))
        print("longest runs:", [(d.date().isoformat(), n) for d, n in sorted(zr, key=lambda r: -r[1])[:8]])
        print("last zero-volume day:", zero.max().date())
    if len(flat.difference(zero)):
        print("flat with volume, by year:", by_year(flat.difference(zero)))
        print("  last:", flat.difference(zero).max().date())
    bad = bs[(bs["high"] < bs[["open", "close"]].max(axis=1)) |
             (bs["low"] > bs[["open", "close"]].min(axis=1)) |
             (bs[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    print(f"candles with high below open/close, low above open/close, or a price <= 0: {len(bad)}")
    if len(bad):
        print(bad.head(10).to_string())
    smallest = bs.loc[bs["volume"] > 0, "volume"]
    print(f"volume > 0: smallest {smallest.min()}, median {smallest.median():.2f} BTC")
    pos = bs.loc["2012":]
    print("days with volume below 1 BTC, by year:", by_year(bs.index[(bs['volume'] > 0) & (bs['volume'] < 1)]))


def report_cm(cm, records, catalog, bs):
    section("Coin Metrics Community, btc, 1d")
    free = catalog["catalog_free"]
    every = catalog["catalog_all"]
    print(f"rows: {len(cm)}, first {cm.index.min().date()}, last {cm.index.max().date()}; "
          f"missing calendar days in the rows: {describe_missing(missing_days(cm.index))}")
    print(f"free metrics at 1d: {len(free)} (catalog-v2); all metrics at 1d: {len(every)} "
          f"(catalog-all-v2, of which flagged community: "
          f"{sum(1 for f in every.values() if f.get('community'))})")
    print(f"free catalog == metrics flagged community in the full catalog: "
          f"{set(free) == {m for m, f in every.items() if f.get('community')}}")
    extra = sorted(c for c in cm.columns if c not in free and c not in ("asset", "time"))
    print("fields besides asset, time and the metrics:", ", ".join(extra))
    print("JSON types: asset str, time str; every metric value is str (a decimal) or null")
    types = Counter()
    for record in records:
        for key, value in record.items():
            if key in free:
                types[type(value).__name__] += 1
    print("  metric cell types over all rows:", dict(types))
    print()
    print("| metric | catalog min_time | first non-null | first > 0 | last | nulls after first non-null | zeros after first > 0 |")
    print("|---|---|---|---|---|---|---|")
    for metric in sorted(free, key=str.lower):
        col = pd.to_numeric(cm[metric], errors="coerce")
        present = col.dropna()
        if present.empty:
            print(f"| {metric} | {free[metric]['min_time'][:10]} | none | | | | |")
            continue
        first, last = present.index.min(), present.index.max()
        positive = col[col > 0]
        firstpos = positive.index.min() if len(positive) else None
        inside = col.loc[first:last]
        nulls = inside.index[inside.isna()]
        zeros = col.loc[firstpos:].index[col.loc[firstpos:] == 0] if firstpos is not None else []
        print(f"| {metric} | {free[metric]['min_time'][:10]} | {first.date()} | "
              f"{firstpos.date() if firstpos is not None else 'never'} | {last.date()} | "
              f"{describe_missing(nulls)} | {describe_missing(zeros)} |")
    print()
    status_cols = [c for c in cm.columns if c.endswith("-status")]
    for col in status_cols:
        counts = cm[col].value_counts(dropna=False).to_dict()
        flash = cm.index[cm[col] == "flash"]
        span = f"{flash.min().date()}..{flash.max().date()}" if len(flash) else "-"
        print(f"{col}: {counts}; flash days {span}")
    done = pd.to_numeric(cm["AssetEODCompletionTime"], errors="coerce")
    done_at = pd.to_datetime(done, unit="s")
    lag_h = (done_at - (cm.index + DAY)) / pd.Timedelta(hours=1)
    recent = lag_h.loc[cm.index.max() - pd.Timedelta(days=364):]
    print(f"AssetEODCompletionTime minus 00:00 UTC of D+1, last 365 rows, hours: "
          f"min {recent.min():.2f}, median {recent.median():.2f}, max {recent.max():.2f}; "
          f"rows before 00:00 of D+1: {(recent < 0).sum()}")
    backfill = done_at.value_counts().iloc[0]
    print(f"most common completion time {done_at.value_counts().index[0]} shared by {backfill} rows "
          f"(first {done_at[done_at == done_at.value_counts().index[0]].index.min().date()}, "
          f"last {done_at[done_at == done_at.value_counts().index[0]].index.max().date()})")
    first_live = lag_h[lag_h.between(0, 48)].index.min()
    print(f"first row completed within 48 h of its day's end: {first_live.date() if pd.notna(first_live) else 'none'}")

    section("Coin Metrics day convention: PriceUSD of D against Bitstamp's close")
    price = pd.to_numeric(cm["PriceUSD"], errors="coerce").dropna()
    close = bs["close"]
    prev = close.shift(1, freq="D")
    nxt = close.shift(-1, freq="D")
    for label, ref in (("close of D", close), ("close of D-1", prev), ("close of D+1", nxt)):
        both = pd.concat([price, ref], axis=1, join="inner").dropna()
        both.columns = ["p", "c"]
        absdiff = (both.p - both.c).abs()
        rel = np.log(both.p / both.c).abs()
        print(f"against {label}: {len(both)} days, median |PriceUSD - close| = {absdiff.median():.4f} USD, "
              f"median |ln(PriceUSD / close)| = {rel.median():.6f}")
    rows = []
    for year in sorted(set(price.index.year) & set(close.index.year)):
        line = [str(year)]
        for ref in (close, prev):
            both = pd.concat([price, ref], axis=1, join="inner").dropna()
            both.columns = ["p", "c"]
            both = both[both.index.year == year]
            line.append(f"{np.log(both.p / both.c).abs().median():.5f}")
            line.append(f"{(both.p - both.c).abs().median():.2f}")
        rows.append(line)
    print("| year | median abs ln diff, D | median abs USD diff, D | median abs ln diff, D-1 | median abs USD diff, D-1 |")
    print("|---|---|---|---|---|")
    for line in rows:
        print("| " + " | ".join(line) + " |")

    section("Bitstamp close against Coin Metrics PriceUSD, same day D")
    both = pd.concat([close, price], axis=1, join="inner").dropna()
    both.columns = ["bitstamp_close", "cm_price"]
    both["rel"] = both.cm_price / both.bitstamp_close - 1
    far = both[both.rel.abs() > 0.01]
    print(f"matched days: {len(both)} ({both.index.min().date()}..{both.index.max().date()}); "
          f"more than 1% apart: {len(far)}")
    total = Counter(both.index.year)
    farc = Counter(far.index.year)
    print("| year | matched days | > 1% apart | share |")
    print("|---|---|---|---|")
    for year in sorted(total):
        print(f"| {year} | {total[year]} | {farc.get(year, 0)} | {farc.get(year, 0) / total[year]:.1%} |")
    top = both.reindex(both.rel.abs().sort_values(ascending=False).index[:10])
    print("ten largest:")
    print("| date | Bitstamp close | CM PriceUSD | PriceUSD / close - 1 |")
    print("|---|---|---|---|")
    for day, row in top.iterrows():
        print(f"| {day.date()} | {row.bitstamp_close:.2f} | {row.cm_price:.4f} | {row.rel:+.2%} |")
    later = far.loc["2014":]
    print(f"more than 1% apart from 2014 on: {len(later)}; from 2017 on: {len(far.loc['2017':])}")

    section("Halvings")
    iss = pd.to_numeric(cm["IssTotNtv"], errors="coerce")
    blk = pd.to_numeric(cm["BlkCnt"], errors="coerce")
    supply = pd.to_numeric(cm["SplyCur"], errors="coerce")
    height = blk.fillna(0).cumsum()
    per_block = iss / blk
    diff_ok = (supply.diff() - iss).abs().loc["2009-01-10":].max()
    print(f"max |SplyCur(D) - SplyCur(D-1) - IssTotNtv(D)| from 2009-01-10: {diff_ok:.8f}")
    for day, target in HALVINGS.items():
        d = pd.Timestamp(day)
        crossed = height.index[height >= target].min()
        window = per_block.loc[d - 2 * DAY: d + 2 * DAY]
        print(f"{day}: IssTotNtv / BlkCnt on D-2..D+2 = " +
              ", ".join(f"{v:.4f}" for v in window.values) +
              f"; cumulative BlkCnt first reaches {target} on {crossed.date()}; "
              f"blocks that day {int(blk.loc[crossed])}, cumulative before {int(height.loc[crossed - DAY])}")
    print("block reward by era (median IssTotNtv / BlkCnt):")
    edges = ["2009-01-10"] + list(HALVINGS) + [str(cm.index.max().date())]
    for start, end in zip(edges[:-1], edges[1:]):
        seg = per_block.loc[pd.Timestamp(start) + DAY: pd.Timestamp(end) - DAY]
        print(f"  {start}..{end}: median {seg.median():.4f}, min {seg.min():.4f}, max {seg.max():.4f}")


def report_klines(name: str, frame, records):
    section(f"Binance {name} BTCUSDT 1d klines")
    print(f"first {frame.index.min().date()}, last {frame.index.max().date()}, {len(frame)} rows")
    print("fields (array positions, named here) and JSON types:", field_types(records))
    print("missing calendar days:", describe_missing(missing_days(frame.index)))
    short = frame[frame["close_time"] != frame["open_time"] + DAY_MS - 1]
    print(f"klines closing before the end of their day: {len(short)}")
    for day, row in short.iterrows():
        closes = pd.to_datetime(int(row.close_time), unit="ms")
        print(f"  {day.date()}: close_time {closes}, volume {row.volume}, trades {row.trades}")
    zero = frame.index[(frame["volume"] == 0) | (frame["trades"] == 0)]
    print("days with zero volume or zero trades:", describe_missing(zero))
    print("ignore field values:", dict(Counter(r["ignore"] for r in records)))


def report_funding(fund, records):
    section("Binance funding, BTCUSDT perpetual")
    print(f"prints: {len(fund)}; first {fund['time'].min()}, last {fund['time'].max()}")
    print("fields and JSON types:", field_types(records))
    print("key sets:", dict(Counter(tuple(sorted(r)) for r in records)))
    print("rateType:", dict(Counter(r.get("rateType") for r in records)))
    print("markPrice empty:", sum(1 for r in records if r.get("markPrice") in ("", None)),
          "; last empty:",
          max((pd.to_datetime(r["fundingTime"], unit="ms") for r in records
               if r.get("markPrice") in ("", None)), default="none"))
    off = fund["offset_ms"]
    print(f"offset from the whole hour, ms: zero {int((off == 0).sum())}, 1-999 "
          f"{int(off.between(1, 999).sum())}, 1000-59999 {int(off.between(1000, 59999).sum())}, "
          f">= 60000 {int((off >= 60000).sum())}; max {int(off.max())}; "
          f"quantiles 50/90/99%: {off.quantile([0.5, 0.9, 0.99]).round(1).tolist()}")
    if (off >= 1000).any():
        print("  prints >= 1 s after the hour:")
        for _, row in fund[off >= 1000].iterrows():
            print(f"    {row.time}")
    print("hours after flooring:", dict(sorted(Counter(fund['hour']).items())))
    dup = fund[fund.duplicated("hour_start", keep=False)]
    print(f"floored hours with more than one print: {dup['hour_start'].nunique()}")
    per_day = fund.groupby("day").size()
    first, last = per_day.index.min(), per_day.index.max()
    full = per_day.reindex(pd.date_range(first, last, freq="D"), fill_value=0)
    print("prints per day:", dict(sorted(Counter(full.values).items())))
    odd = full[full != 3]
    print(f"days with a number other than three ({len(odd)}):")
    for day, n in odd.items():
        hours = fund.loc[fund["day"] == day, "time"].dt.strftime("%H:%M:%S.%f").str[:-3].tolist()
        print(f"  {day.date()}: {n} -> {hours}")
    regular = fund[fund["hour"].isin([0, 8, 16])]
    print(f"prints at 00, 08 or 16 after flooring: {len(regular)} of {len(fund)}")


def report_first_rows(bs, cm, spot, perp, fund):
    section("First rows under rule 8 (complete windows, data as fetched)")
    start = bs.index.min()
    print(f"Bitstamp first day {start.date()}")
    for lookback, what in ((365, "ret_365, vol_cc_365, vol_ratio_30_365"),
                           (364, "dist_high_365, dist_low_365"),
                           (394, "vol_z_365, amihud_z_365: 365 values of a 30-day statistic of returns")):
        print(f"  needs {lookback} days before t ({what}): first t = {(start + lookback * DAY).date()}")
    first_a = start + 394 * DAY
    last = bs.index.max()
    print(f"table A first row {first_a.date()}, last {last.date()}, rows {(last - first_a).days + 1}")
    zero = bs.index[bs["volume"] == 0]
    print(f"zero-volume Bitstamp days on or after {first_a.date()}: {describe_missing(zero[zero >= first_a])}")
    print(f"zero-volume days within 394 days before it: {len(zero[(zero < first_a) & (zero >= first_a - 394 * DAY)])}")
    for metric in ("AdrActCnt", "TxCnt", "HashRate", "FeeTotNtv", "SplyCur", "IssTotNtv", "CapMVRVCur", "CapMrktCurUSD"):
        col = pd.to_numeric(cm[metric], errors="coerce")
        need_from = first_a - 365 * DAY
        window = col.loc[need_from: last - DAY]
        print(f"  {metric}: from {need_from.date()} (365 days before t-1 of the first row): "
              f"null {int(window.isna().sum())}, <= 0 {int((window <= 0).sum())}")
    full_days = fund.groupby("day").size()
    full_days = full_days[full_days == 3]
    first_full = full_days.index.min()
    print(f"perpetual first kline {perp.index.min().date()}; first funding day with three prints {first_full.date()}")
    b_from = max(perp.index.min() + 89 * DAY, first_full + 89 * DAY, spot.index.min() + 89 * DAY)
    print(f"  basis_z_90, perp_spot_lvol_z_90, taker_share_perp_z_90 need 89 days before t: "
          f"{(perp.index.min() + 89 * DAY).date()}")
    print(f"  fund_z_90 needs fund_1d on 90 days: {(first_full + 89 * DAY).date()}")
    print(f"table B first row {b_from.date()}, last {last.date()}, rows {(last - b_from).days + 1}")
    gaps_spot = missing_days(spot.index)
    print(f"spot missing days on or after {(b_from - 89 * DAY).date()}: "
          f"{describe_missing(gaps_spot[gaps_spot >= b_from - 89 * DAY])}")


# ------------------------------------------- presence under the proposal --
# Which feature columns would have a value on which rows, under the treatment
# proposed in reports/sources.md: a Bitstamp day with zero volume is a missing
# day; days before a source's first day are not gaps (a window must lie inside
# the source's history); a rolling value needs 90% of its window; a return
# needs both ends; a z-score needs its own day's value; an EMA or Wilder
# average needs its input on the day and 90% of its last 3 x span inputs; the
# days inside a gap of three or more days have no Bitstamp-based value.
# Formulas are not computed here, only whether each column has a value.

RET_N = (1, 2, 3, 5, 7, 14, 21, 30, 60, 90, 180, 365)


def need(n: int, strict: bool) -> int:
    return n if strict else int(np.ceil(0.9 * n - 1e-9))


def win(mask: pd.Series, n: int, strict: bool) -> pd.Series:
    counts = mask.astype("int64").rolling(n, min_periods=n).sum()
    return counts >= need(n, strict)


def smooth(mask: pd.Series, span: int, strict: bool) -> pd.Series:
    return mask & win(mask, 3 * span, strict)


def lag(mask: pd.Series, k: int) -> pd.Series:
    return mask.shift(k, fill_value=False)


def presence_a(P: pd.Series, strict: bool) -> dict:
    r = P & lag(P, 1)
    f = {}
    for n in RET_N:
        f[f"ret_{n}"] = P & lag(P, n)
    s30 = win(r, 30, strict)
    for n in (7, 14, 30):
        f[f"zret_{n}"] = f[f"ret_{n}"] & s30
    f["past_z_7"] = f["past_class_7"] = P & lag(P, 7) & lag(s30, 7)
    vc = {n: win(r, n, strict) for n in (7, 14, 30, 90, 365)}
    for n, m in vc.items():
        f[f"vol_cc_{n}"] = m
    for n in (7, 30):
        for k in ("park", "gk", "rs"):
            f[f"vol_{k}_{n}"] = win(P, n, strict)
    f["vol_yz_30"] = win(r, 30, strict)
    f["vol_ratio_7_30"] = vc[7] & vc[30]
    f["vol_ratio_30_90"] = vc[30] & vc[90]
    f["vol_ratio_30_365"] = vc[30] & vc[365]
    f["vol_of_vol_30"] = win(vc[7], 30, strict)
    f["vol_z_365"] = vc[30] & win(vc[30], 365, strict)
    f["hl_range"] = P
    f["hl_range_mean_7"] = win(P, 7, strict)
    for k in ("body", "clv", "upper_wick", "lower_wick"):
        f[k] = P
    f["body_mean_7"] = f["clv_mean_7"] = win(P, 7, strict)
    sma = {n: win(P, n, strict) for n in (7, 20, 50, 100, 200)}
    for n, m in sma.items():
        f[f"dist_sma_{n}"] = P & m
    f["sma_50_200"] = sma[50] & sma[200]
    macd = P & smooth(P, 12, strict) & smooth(P, 26, strict)
    f["macd"] = macd
    f["macd_signal"] = smooth(macd, 9, strict)
    f["macd_hist"] = macd & f["macd_signal"]
    for n in (30, 90):
        f[f"trend_t_{n}"] = win(P, n, strict)
    f["dd_ath"] = f["days_since_ath"] = P
    f["dist_high_365"] = f["dist_low_365"] = P & win(P, 365, strict)
    f["adx_14"] = smooth(smooth(r, 14, strict), 14, strict)
    f["aroon_osc_25"] = win(P, 26, strict)
    f["rsi_14"] = smooth(r, 14, strict)
    f["rsi_30"] = smooth(r, 30, strict)
    stoch = P & win(P, 14, strict)
    f["stoch_k_14"] = stoch
    f["stoch_d_14_3"] = win(stoch, 3, strict)
    f["bb_pctb_20"] = f["cci_20"] = P & win(P, 20, strict)
    f["bb_width_20"] = win(P, 20, strict)
    f["lvol_z_30"] = P & win(P, 30, strict)
    f["lvol_z_90"] = P & win(P, 90, strict)
    f["lvol_ratio_7_30"] = win(P, 7, strict) & win(P, 30, strict)
    ami = win(r, 30, strict)
    f["amihud_z_365"] = ami & win(ami, 365, strict)
    f["obv_slope_30"] = f["up_vol_share_30"] = win(r, 30, strict)
    f["cmf_20"] = win(P, 20, strict)
    for n in (30, 90):
        for k in ("skew", "kurt", "acf1"):
            f[f"{k}_{n}"] = win(r, n, strict)
    for n in (14, 30):
        f[f"up_share_{n}"] = win(r, n, strict)
    f["maxdd_30"] = win(P, 30, strict)
    return f


def presence_b(Ps, Pp, Pf, Pb, strict: bool) -> dict:
    f = {"fund_1d": Pf, "fund_7": win(Pf, 7, strict), "fund_30": win(Pf, 30, strict),
         "fund_z_90": Pf & win(Pf, 90, strict)}
    basis = Pp & Ps
    f.update(basis=basis, basis_mean_7=win(basis, 7, strict),
             basis_z_90=basis & win(basis, 90, strict))
    prem = Ps & Pb
    f.update(usdt_prem=prem, usdt_prem_z_90=prem & win(prem, 90, strict))
    f.update(perp_spot_lvol=basis, perp_spot_lvol_z_90=basis & win(basis, 90, strict))
    for name, m in (("spot", Ps), ("perp", Pp)):
        f[f"taker_share_{name}"] = m
        f[f"taker_share_{name}_mean_7"] = win(m, 7, strict)
        f[f"taker_share_{name}_z_90"] = m & win(m, 90, strict)
    f["trades_z_90"] = f["avg_trade_z_90"] = Ps & win(Ps, 90, strict)
    f["bn_lvol_z_30"] = Ps & win(Ps, 30, strict)
    return f


def gap_runs(present: pd.Series):
    missing = present.index[~present]
    return [(d, n) for d, n in runs(missing)]


def inside_long(present: pd.Series) -> pd.Series:
    inside = pd.Series(False, index=present.index)
    for day, n in gap_runs(present):
        if n >= 3:
            inside.loc[day: day + (n - 1) * DAY] = True
    return inside


def table_a(P: pd.Series) -> pd.DataFrame:
    strict = presence_a(pd.Series(True, index=P.index), True)
    lenient = presence_a(P, False)
    inside = inside_long(P)
    return pd.DataFrame({k: strict[k] & lenient[k] & ~inside for k in strict})


def summarise_holes(frame: pd.DataFrame, first) -> None:
    after = frame.loc[first:]
    bad = after.index[~after.all(axis=1)]
    print(f"  rows from the first row on with a feature missing: {len(bad)}")
    for day, n in runs(bad):
        block = after.loc[day: day + (n - 1) * DAY]
        cols = [c for c in block.columns if not block[c].all()]
        shown = ", ".join(cols[:8]) + (f" and {len(cols) - 8} more" if len(cols) > 8 else "")
        print(f"    {day.date()}..{(day + (n - 1) * DAY).date()} ({n} rows): {len(cols)} columns: {shown}")
    if len(bad):
        counts = (~after.loc[bad]).sum().sort_values(ascending=False)
        counts = counts[counts > 0]
        print("  missing cells by column: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


def report_presence(bs, spot, perp, fund):
    section("Presence under the proposed treatment of missing days")
    idx = bs.index
    P = bs["volume"] > 0
    zero = idx[~P]
    print("Bitstamp zero-volume days:", ", ".join(d.strftime("%Y-%m-%d") for d in zero))
    print("their runs:", [(d.date().isoformat(), n) for d, n in gap_runs(P)])
    frame = table_a(P)
    print(f"table A: {frame.shape[1]} Bitstamp-based columns modelled "
          f"(calendar and on-chain columns have a value on every row)")
    first = frame.index[frame.all(axis=1)].min()
    print(f"table A first row under rule 8: {first.date()}; rows to {idx.max().date()}: "
          f"{(idx.max() - first).days + 1}")
    only_2011 = P | (idx.year >= 2015)
    only_2015 = P | (idx.year < 2015)
    for label, mask in (("all zero-volume days missing", P),
                        ("only the 2011 days missing", only_2011),
                        ("only the 2015 days missing", only_2015)):
        print(f" {label}:")
        summarise_holes(table_a(mask), first)
    frame_2011 = table_a(only_2011)
    after = frame_2011.loc[first:]
    bad = after.index[~after.all(axis=1)]
    clean = (bad.max() + DAY) if len(bad) else first
    print(f"first row after which no row lacks a value because of the 2011 days: {clean.date()}; "
          f"rows {(idx.max() - clean).days + 1}")
    r = P & lag(P, 1)
    s30 = win(pd.Series(True, index=idx), 30, True) & win(r, 30, False)
    label = P & P.shift(-7, fill_value=False) & s30 & ~inside_long(P)
    has_future = idx <= idx.max() - 7 * DAY
    for start in (first, clean):
        rows = (idx >= start) & has_future
        print(f"labels from {start.date()} to {(idx.max() - 7 * DAY).date()}: {int(rows.sum())} rows "
              f"could carry one; blank there because of missing days: {int((rows & ~label).sum())} "
              f"({', '.join(f'{d.date()}+{n}' for d, n in runs(idx[rows & ~label.values]))})")

    def on_calendar(frame_src, first_day, mask=None):
        present = pd.Series(False, index=idx)
        days = frame_src.index if mask is None else frame_src.index[mask]
        present.loc[present.index.isin(days) & (idx >= first_day)] = True
        return present

    per_day = fund.groupby("day").size()
    full_days = per_day.index[per_day == 3]
    first_fund = full_days.min()
    Ps = on_calendar(spot, spot.index.min())
    Pp = on_calendar(perp, perp.index.min())
    Pf = pd.Series(idx.isin(full_days), index=idx)
    strict_b = presence_b(*(pd.Series(idx >= d, index=idx) for d in
                            (spot.index.min(), perp.index.min(), first_fund, idx.min())), True)
    lenient_b = presence_b(Ps, Pp, Pf, P, False)
    frame_b = pd.concat([frame, pd.DataFrame({k: strict_b[k] & lenient_b[k] for k in strict_b})], axis=1)
    first_b = frame_b.index[frame_b.all(axis=1)].min()
    print(f"table B ({frame_b.shape[1]} modelled columns) first row under rule 8: {first_b.date()}; "
          f"rows {(idx.max() - first_b).days + 1} (funding history taken from {first_fund.date()}, "
          f"the first day with three prints)")
    binding = [k for k in strict_b if not (strict_b[k] & lenient_b[k]).loc[first_b - DAY]]
    print(f"  columns without a value the day before: {', '.join(binding)}")
    summarise_holes(frame_b, first_b)


def main() -> int:
    bs, bs_records = load_bitstamp()
    cm, cm_records, catalog = load_cm()
    spot, spot_records = load_klines("spot_klines")
    perp, perp_records = load_klines("perp_klines")
    fund, fund_records = load_funding()
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    section("Manifest")
    for source, entry in sorted(manifest.items()):
        print(f"{source}: run {entry['run_started_utc']}, last complete day {entry['last_complete_day']}, "
              f"{len(entry['files'])} files, {sum(f['rows'] for f in entry['files'])} rows")
    report_bitstamp(bs, bs_records)
    report_cm(cm, cm_records, catalog, bs)
    report_klines("spot", spot, spot_records)
    report_klines("USD-M perpetual", perp, perp_records)
    report_funding(fund, fund_records)
    report_first_rows(bs, cm, spot, perp, fund)
    report_presence(bs, spot, perp, fund)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())