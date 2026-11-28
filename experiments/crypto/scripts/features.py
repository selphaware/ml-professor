"""Feature columns of tables A to D, each under the missing-data rules.

Every column is computed on the calendar (one row per UTC day) from values
known at the close of its day, and is then kept only on the days where the
rules give it a value. Used by build_dataset.py; never run on its own.

The rules, as settled in README.md (rulings 10 to 20):
- a window of n days needs 90% of its days present (ceil(0.9 n)), and lies
  wholly inside its source's history: days before a source's first day are
  not gaps;
- a return over n days needs both of its ends;
- a z-score needs its own day's value as well as 90% of its window;
- a recursive average (EMA, Wilder) steps over a day with no input, and is
  reported when the day's input exists and 90% of its last 3 x span days
  are present;
- a day inside a longer gap (the third or later missing day in a row, as
  known at that day's close) has no value in any column of that source;
- a source's gaps blank only the columns built from that source.

The FX columns of tables C and D (rulings 21 to 34) are computed on the
sequence of ECB fixing dates, under the same rules applied to fixings, and
row t then takes the values of the latest fixing date on or before t.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

HALVINGS = tuple(pd.Timestamp(d) for d in
                 ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"))
GENESIS = pd.Timestamp("2009-01-03")
CYCLE_DAYS = 1461
RET_N = (1, 2, 3, 5, 7, 14, 21, 30, 60, 90, 180, 365)
ONCHAIN = ("AdrActCnt", "TxCnt", "HashRate", "FeeTotNtv")
LONG_GAP = 3  # a gap of this many missing days in a row is a longer gap
LN2 = math.log(2.0)

A_COLUMNS = (
    [f"ret_{n}" for n in RET_N]
    + ["zret_7", "zret_14", "zret_30", "past_z_7", "past_class_7"]
    + ["vol_cc_7", "vol_cc_14", "vol_cc_30", "vol_cc_90",
       "vol_park_7", "vol_gk_7", "vol_rs_7", "vol_park_30", "vol_gk_30", "vol_rs_30",
       "vol_yz_30", "vol_ratio_7_30", "vol_ratio_30_90", "vol_ratio_30_365",
       "vol_of_vol_30", "vol_z_365"]
    + ["hl_range", "hl_range_mean_7", "body", "clv", "upper_wick", "lower_wick",
       "body_mean_7", "clv_mean_7"]
    + ["dist_sma_7", "dist_sma_20", "dist_sma_50", "dist_sma_100", "dist_sma_200",
       "sma_50_200", "macd", "macd_signal", "macd_hist", "trend_t_30", "trend_t_90",
       "dd_ath", "days_since_ath", "dist_high_365", "dist_low_365", "adx_14",
       "aroon_osc_25"]
    + ["rsi_14", "rsi_30", "stoch_k_14", "stoch_d_14_3", "bb_pctb_20", "bb_width_20",
       "cci_20"]
    + ["lvol_z_30", "lvol_z_90", "lvol_ratio_7_30", "amihud_z_365", "obv_slope_30",
       "cmf_20", "up_vol_share_30"]
    + ["skew_30", "kurt_30", "acf1_30", "skew_90", "kurt_90", "acf1_90",
       "up_share_14", "up_share_30", "maxdd_30"]
    + ["dow", "month", "days_since_halving", "halving_phase"]
    + [f"oc_{m.lower()}_{s}" for m in ONCHAIN for s in ("g7", "g30", "z365")]
    + ["mvrv", "mvrv_z365"]
)
B_ONLY_COLUMNS = [
    "fund_1d", "fund_7", "fund_30", "fund_z_90",
    "basis", "basis_mean_7", "basis_z_90",
    "usdt_prem", "usdt_prem_z_90",
    "perp_spot_lvol", "perp_spot_lvol_z_90",
    "taker_share_spot", "taker_share_spot_mean_7", "taker_share_spot_z_90",
    "taker_share_perp", "taker_share_perp_mean_7", "taker_share_perp_z_90",
    "trades_z_90", "avg_trade_z_90", "bn_lvol_z_30",
]
B_COLUMNS = A_COLUMNS + B_ONLY_COLUMNS
INTEGER_COLUMNS = frozenset({"past_class_7", "dow", "month"})

# Tables C and D: table A or B followed by the FX columns, from the ECB's
# euro reference rates.
ECB_CURRENCIES = ("USD", "JPY", "GBP", "CHF", "AUD", "CAD", "SEK")
FX_SERIES = ("eurusd", "usdjpy", "gbpusd", "usdchf", "audusd", "usdcad", "dxy")
FX_RET_N = (1, 5, 20, 60)
# ICE's dollar index weights, applied in logs; the index's constant is left
# out, since only changes of ln(dxy) are used. usdsek enters dxy only.
DXY_WEIGHTS = (("eurusd", -0.576), ("usdjpy", 0.136), ("gbpusd", -0.119),
               ("usdcad", 0.091), ("usdsek", 0.042), ("usdchf", 0.036))
FX_COLUMNS = ([f"fx_{s}_{part}" for s in FX_SERIES
               for part in [f"ret_{n}" for n in FX_RET_N] + ["vol_20", "vol_z_250"]]
              + ["btc_dxy_corr_60"])
C_COLUMNS = A_COLUMNS + FX_COLUMNS
D_COLUMNS = B_COLUMNS + FX_COLUMNS


# ------------------------------------------------------------ availability --

def need(n: int) -> int:
    """Days a window of n days needs: 90% of n, rounded up."""
    return math.ceil(0.9 * n - 1e-9)


def rolling_count(mask: np.ndarray, n: int) -> np.ndarray:
    """True days in each window of n days ending on a day; -1 before n days."""
    totals = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    out = np.full(len(mask), -1, dtype=np.int64)
    if len(mask) >= n:
        out[n - 1:] = totals[n:] - totals[:-n]
    return out


def lag_mask(mask: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(len(mask), dtype=bool)
    if k < len(mask):
        out[k:] = mask[:len(mask) - k]
    return out


@dataclass(frozen=True)
class Avail:
    """Where a value can exist.

    ``have``: the inputs exist under the rules (90% of each window);
    ``full``: the same with no gap at all, i.e. every window lies inside the
    source's history. A value exists where both hold and the day is not
    inside a longer gap of its sources.
    """

    have: np.ndarray
    full: np.ndarray

    def __and__(self, other: "Avail") -> "Avail":
        return Avail(self.have & other.have, self.full & other.full)

    def lag(self, k: int) -> "Avail":
        return Avail(lag_mask(self.have, k), lag_mask(self.full, k))

    def window(self, n: int) -> "Avail":
        return Avail(rolling_count(self.have, n) >= need(n),
                     rolling_count(self.full, n) >= n)

    def smooth(self, span: int) -> "Avail":
        """A recursive average: the day's input and 90% of 3 x span days."""
        return self & self.window(3 * span)


def inside_long_gaps(present: np.ndarray, history: np.ndarray) -> np.ndarray:
    """Days that are, by their own close, the third or later missing day in a row."""
    out = np.zeros(len(present), dtype=bool)
    run = 0
    for i in range(len(present)):
        if history[i] and not present[i]:
            run += 1
            out[i] = run >= LONG_GAP
        else:
            run = 0
    return out


class Table:
    """Columns on one calendar, each kept only where it has a value."""

    def __init__(self, index: pd.DatetimeIndex):
        self.index = index
        self.data: dict[str, np.ndarray] = {}
        self.problems: list[str] = []

    def settle(self, name: str, values, avail: Avail, inside: np.ndarray):
        mask = avail.have & avail.full & ~inside
        values = np.asarray(values, dtype=float)
        bad = mask & ~np.isfinite(values)
        if bad.any():
            days = ", ".join(d.strftime("%Y-%m-%d") for d in self.index[bad][:5])
            self.problems.append(f"{name}: no finite value on {int(bad.sum())} days "
                                 f"whose inputs exist ({days})")
        return np.where(mask, values, np.nan), Avail(mask, avail.full)

    def put(self, name: str, values, avail: Avail, inside: np.ndarray) -> Avail:
        self.data[name], settled = self.settle(name, values, avail, inside)
        return settled


# ------------------------------------------------------------- arithmetic --

def shift(x: np.ndarray, k: int) -> np.ndarray:
    """x moved k days later (the value of day t-k on day t)."""
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


def windows(x: np.ndarray, n: int) -> np.ndarray:
    """Row i holds the n days ending on day i (NaN before the calendar)."""
    padded = np.concatenate([np.full(n - 1, np.nan), np.asarray(x, dtype=float)])
    return np.lib.stride_tricks.sliding_window_view(padded, n)


def quiet(fn):
    """Run a numpy computation without warnings about empty windows."""
    def inner(*args, **kwargs):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            return fn(*args, **kwargs)
    inner.__name__ = fn.__name__
    return inner


@quiet
def w_mean(x, n):
    return np.nanmean(windows(x, n), axis=1)


@quiet
def w_sum(x, n):
    w = windows(x, n)
    return np.where(np.isnan(w).all(axis=1), np.nan, np.nansum(w, axis=1))


@quiet
def w_std(x, n):
    return np.nanstd(windows(x, n), axis=1, ddof=1)


@quiet
def w_var(x, n):
    return np.nanvar(windows(x, n), axis=1, ddof=1)


@quiet
def w_max(x, n):
    return np.nanmax(windows(x, n), axis=1)


@quiet
def w_min(x, n):
    return np.nanmin(windows(x, n), axis=1)


def w_count(x, n):
    return (~np.isnan(windows(x, n))).sum(axis=1)


@quiet
def zscore(x, n):
    """(x_t - mean) / sd over the n days ending with t; 0 when sd is 0."""
    w = windows(x, n)
    mean = np.nanmean(w, axis=1)
    sd = np.nanstd(w, axis=1, ddof=1)
    z = (np.asarray(x, dtype=float) - mean) / sd
    return np.where(sd == 0, 0.0, z)


@quiet
def w_moments(x, n):
    """Sample skewness (G1) and excess kurtosis (G2) of each window."""
    w = windows(x, n)
    m = (~np.isnan(w)).sum(axis=1).astype(float)
    d = w - np.nanmean(w, axis=1)[:, None]
    m2 = np.nanmean(d ** 2, axis=1)
    m3 = np.nanmean(d ** 3, axis=1)
    m4 = np.nanmean(d ** 4, axis=1)
    skew = np.sqrt(m * (m - 1)) / (m - 2) * m3 / m2 ** 1.5
    kurt = (m - 1) / ((m - 2) * (m - 3)) * ((m + 1) * m4 / m2 ** 2 - 3 * (m - 1))
    return skew, kurt


@quiet
def w_acf1(x, n):
    """Correlation of each return with the one before, both in the window."""
    w = windows(x, n)
    a, b = w[:, 1:], w[:, :-1]
    ok = ~np.isnan(a) & ~np.isnan(b)
    a = np.where(ok, a, np.nan)
    b = np.where(ok, b, np.nan)
    da = a - np.nanmean(a, axis=1)[:, None]
    db = b - np.nanmean(b, axis=1)[:, None]
    return np.nansum(da * db, axis=1) / np.sqrt(np.nansum(da ** 2, axis=1)
                                                * np.nansum(db ** 2, axis=1))


@quiet
def w_corr(x, y, n):
    """Correlation of x with y over the pairs of each window where both exist."""
    a, b = windows(x, n), windows(y, n)
    ok = ~np.isnan(a) & ~np.isnan(b)
    a = np.where(ok, a, np.nan)
    b = np.where(ok, b, np.nan)
    da = a - np.nanmean(a, axis=1)[:, None]
    db = b - np.nanmean(b, axis=1)[:, None]
    return np.nansum(da * db, axis=1) / np.sqrt(np.nansum(da ** 2, axis=1)
                                                * np.nansum(db ** 2, axis=1))


@quiet
def w_up_share(x, n):
    w = windows(x, n)
    return (w > 0).sum(axis=1) / (~np.isnan(w)).sum(axis=1)


@quiet
def w_maxdd(c, n):
    """Largest fall from a peak to a later trough within the window, as ln."""
    w = windows(c, n)
    peak = np.fmax.accumulate(w, axis=1)
    return np.nanmin(np.log(w / peak), axis=1)


@quiet
def w_total(x, n):
    """Sum over the window, scaled to n days when a day is missing."""
    count = w_count(x, n)
    total = w_sum(x, n)
    return np.where(count == n, total, total * n / count)


def ema_skip(x: np.ndarray, alpha: float, seed: int) -> np.ndarray:
    """Recursive average over the days that have an input, skipping the rest.

    It starts from the simple mean of its first ``seed`` inputs. Each later
    input moves it by ``alpha``.
    """
    out = np.full(len(x), np.nan)
    state = None
    first: list[float] = []
    for i, value in enumerate(x):
        if value != value:
            continue
        if state is None:
            first.append(float(value))
            if len(first) == seed:
                state = math.fsum(first) / seed
                out[i] = state
            continue
        state = alpha * value + (1.0 - alpha) * state
        out[i] = state
    return out


@quiet
def trend_t(logc: np.ndarray, n: int) -> np.ndarray:
    """t-statistic of the least-squares slope of ln C on day number."""
    w = windows(logc, n)
    days = np.arange(n, dtype=float)
    out = np.full(len(logc), np.nan)
    for i in range(n - 1, len(logc)):
        ok = ~np.isnan(w[i])
        m = int(ok.sum())
        if m < 3:
            continue
        x, y = days[ok], w[i][ok]
        xd, yd = x - x.mean(), y - y.mean()
        sxx = float((xd ** 2).sum())
        slope = float((xd * yd).sum()) / sxx
        resid = yd - slope * xd
        se = math.sqrt(float((resid ** 2).sum()) / (m - 2) / sxx)
        out[i] = slope / se if se > 0 else np.nan
    return out


@quiet
def aroon_up(x: np.ndarray, n: int, highest: bool) -> np.ndarray:
    """100 x (n - days since the window's extreme) / n, over today and n days back."""
    w = windows(x, n + 1)
    fill = -np.inf if highest else np.inf
    filled = np.where(np.isnan(w), fill, w)[:, ::-1]
    since = np.argmax(filled, axis=1) if highest else np.argmin(filled, axis=1)
    return 100.0 * (n - since) / n


@quiet
def mean_deviation(x: np.ndarray, centre: np.ndarray, n: int) -> np.ndarray:
    return np.nanmean(np.abs(windows(x, n) - centre[:, None]), axis=1)


def where_flat(rng: np.ndarray, numerator: np.ndarray) -> np.ndarray:
    """numerator / (H - L), and 0 when H equals L."""
    flat = rng == 0
    return np.where(flat, 0.0, numerator / np.where(flat, 1.0, rng))


# -------------------------------------------------------------- table A --

@dataclass(frozen=True)
class Source:
    """A source's presence on the calendar."""

    present: np.ndarray
    history: np.ndarray

    @property
    def base(self) -> Avail:
        return Avail(self.present, self.history)

    @property
    def inside(self) -> np.ndarray:
        return inside_long_gaps(self.present, self.history)


def bitstamp_columns(table: Table, ohlcv: pd.DataFrame, first_day: pd.Timestamp) -> dict:
    """The price, volume and candle columns of table A, from Bitstamp.

    `past_class_7` needs k, so it is added later by past_class_column.
    """
    O, H, L, C, V = (ohlcv[c].to_numpy(dtype=float) for c in
                     ("open", "high", "low", "close", "volume"))
    src = Source(~np.isnan(C), np.asarray(table.index >= first_day))
    inside = src.inside
    base = src.base
    ra = base & base.lag(1)
    on_r = ra.have

    def put(name, values, avail):
        return table.put(name, values, avail, inside)

    def hidden(values, avail):
        return table.settle("(intermediate)", values, avail, inside)

    data = table.data
    with np.errstate(all="ignore"):
        cp = shift(C, 1)
        r = np.log(C / cp)

        # Returns
        ret_a = {}
        for n in RET_N:
            ret_a[n] = put(f"ret_{n}", np.log(C / shift(C, n)), base & base.lag(n))
        vc, vcv = {}, {}
        for n in (7, 14, 30, 90):
            vc[n] = put(f"vol_cc_{n}", w_std(r, n), ra.window(n))
            vcv[n] = data[f"vol_cc_{n}"]
        # vol_cc_365 is not a column: from 2023-09 it lies below every earlier
        # value (the level screen). It enters vol_ratio_30_365 only.
        vcv[365], vc[365] = hidden(w_std(r, 365), ra.window(365))
        s30, s30a = data["vol_cc_30"], vc[30]
        for n in (7, 14, 30):
            put(f"zret_{n}", data[f"ret_{n}"] / (s30 * math.sqrt(n)), ret_a[n] & s30a)
        past = np.log(C / shift(C, 7)) / (shift(s30, 7) * math.sqrt(7))
        past_a = base & base.lag(7) & s30a.lag(7)
        put("past_z_7", past, past_a)

        # Volatility
        lhl, lco = np.log(H / L), np.log(C / O)
        rs = np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O)
        daily = {"park": lhl ** 2 / (4 * LN2),
                 "gk": 0.5 * lhl ** 2 - (2 * LN2 - 1) * lco ** 2,
                 "rs": rs}
        for n in (7, 30):
            for name in ("park", "gk", "rs"):
                put(f"vol_{name}_{n}", np.sqrt(w_mean(daily[name], n)), base.window(n))
        overnight = np.where(on_r, np.log(O / cp), np.nan)
        intraday = np.where(on_r, lco, np.nan)
        rs_r = np.where(on_r, rs, np.nan)
        m = w_count(overnight, 30).astype(float)
        kk = 0.34 / (1.34 + (m + 1) / (m - 1))
        put("vol_yz_30", np.sqrt(w_var(overnight, 30) + kk * w_var(intraday, 30)
                                 + (1 - kk) * w_mean(rs_r, 30)), ra.window(30))
        for a, b in ((7, 30), (30, 90), (30, 365)):
            put(f"vol_ratio_{a}_{b}", np.log(vcv[a] / vcv[b]), vc[a] & vc[b])
        put("vol_of_vol_30", w_std(np.log(data["vol_cc_7"]), 30), vc[7].window(30))
        put("vol_z_365", zscore(np.log(data["vol_cc_30"]), 365), vc[30] & vc[30].window(365))

        # Range and candle
        put("hl_range", lhl, base)
        put("hl_range_mean_7", w_mean(lhl, 7), base.window(7))
        rng = H - L
        clv = where_flat(rng, 2 * C - H - L)
        body = where_flat(rng, C - O)
        put("body", body, base)
        put("clv", clv, base)
        put("upper_wick", where_flat(rng, H - np.maximum(O, C)), base)
        put("lower_wick", where_flat(rng, np.minimum(O, C) - L), base)
        put("body_mean_7", w_mean(body, 7), base.window(7))
        put("clv_mean_7", w_mean(clv, 7), base.window(7))

        # Trend
        sma, sma_a = {}, {}
        for n in (7, 20, 50, 100, 200):
            sma[n], sma_a[n] = w_mean(C, n), base.window(n)
            put(f"dist_sma_{n}", np.log(C / sma[n]), base & sma_a[n])
        put("sma_50_200", np.log(sma[50] / sma[200]), sma_a[50] & sma_a[200])
        ema12, ema26 = ema_skip(C, 2 / 13, 12), ema_skip(C, 2 / 27, 26)
        macd_a = put("macd", (ema12 - ema26) / C, base & base.smooth(12) & base.smooth(26))
        sig_a = put("macd_signal", ema_skip(data["macd"], 2 / 10, 9), macd_a.smooth(9))
        put("macd_hist", data["macd"] - data["macd_signal"], macd_a & sig_a)
        logc = np.log(C)
        for n in (30, 90):
            put(f"trend_t_{n}", trend_t(logc, n), base.window(n))
        peak = np.fmax.accumulate(C)
        put("dd_ath", np.log(C / peak), base)
        since = np.full(len(C), np.nan)
        best, best_i = -np.inf, None
        for i, c in enumerate(C):
            if c == c and c >= best:
                best, best_i = c, i
            if best_i is not None:
                since[i] = i - best_i
        put("days_since_ath", np.log1p(since), base)
        year_a = base & base.window(365)
        put("dist_high_365", np.log(C / w_max(H, 365)), year_a)
        put("dist_low_365", np.log(C / w_min(L, 365)), year_a)
        hp, lp = shift(H, 1), shift(L, 1)
        tr = np.where(on_r, np.maximum(np.maximum(H - L, np.abs(H - cp)), np.abs(L - cp)), np.nan)
        up, dn = H - hp, lp - L
        pdm = np.where(on_r, np.where((up > dn) & (up > 0), up, 0.0), np.nan)
        mdm = np.where(on_r, np.where((dn > up) & (dn > 0), dn, 0.0), np.nan)
        s_tr, s_p, s_m = (ema_skip(v, 1 / 14, 14) for v in (tr, pdm, mdm))
        pdi, mdi = 100 * s_p / s_tr, 100 * s_m / s_tr
        total = pdi + mdi
        dx = np.where(total > 0, 100 * np.abs(pdi - mdi) / np.where(total > 0, total, 1.0), 0.0)
        dx_v, dx_a = hidden(dx, ra.smooth(14))
        put("adx_14", ema_skip(dx_v, 1 / 14, 14), dx_a.smooth(14))
        put("aroon_osc_25", aroon_up(H, 25, True) - aroon_up(L, 25, False), base.window(26))

        # Oscillators
        change = np.where(on_r, C - cp, np.nan)
        gain = np.where(np.isnan(change), np.nan, np.maximum(change, 0.0))
        loss = np.where(np.isnan(change), np.nan, np.maximum(-change, 0.0))
        for n in (14, 30):
            avg_gain, avg_loss = ema_skip(gain, 1 / n, n), ema_skip(loss, 1 / n, n)
            both = avg_gain + avg_loss
            put(f"rsi_{n}", np.where(both > 0, 100 * avg_gain / np.where(both > 0, both, 1.0), 50.0),
                ra.smooth(n))
        hh, ll = w_max(H, 14), w_min(L, 14)
        span = hh - ll
        stoch = np.where(span > 0, 100 * (C - ll) / np.where(span > 0, span, 1.0), 50.0)
        stoch_a = put("stoch_k_14", stoch, base & base.window(14))
        put("stoch_d_14_3", w_mean(data["stoch_k_14"], 3), stoch_a.window(3))
        sd20 = w_std(C, 20)
        put("bb_pctb_20", (C - (sma[20] - 2 * sd20)) / (4 * sd20), base & sma_a[20])
        put("bb_width_20", 4 * sd20 / sma[20], sma_a[20])
        tp = (H + L + C) / 3
        tp_mean = w_mean(tp, 20)
        put("cci_20", (tp - tp_mean) / (0.015 * mean_deviation(tp, tp_mean, 20)),
            base & base.window(20))

        # Volume and liquidity
        dv = V * C
        lv = np.log(dv)
        for n in (30, 90):
            put(f"lvol_z_{n}", zscore(lv, n), base & base.window(n))
        put("lvol_ratio_7_30", np.log(w_mean(dv, 7) / w_mean(dv, 30)),
            base.window(7) & base.window(30))
        illiq = np.where(on_r, np.abs(r) / dv, np.nan)
        ami_v, ami_a = hidden(np.log(w_mean(illiq, 30)), ra.window(30))
        put("amihud_z_365", zscore(ami_v, 365), ami_a & ami_a.window(365))
        v_r = np.where(on_r, V, np.nan)
        put("obv_slope_30", w_sum(np.sign(r) * v_r, 30) / w_sum(v_r, 30), ra.window(30))
        put("cmf_20", w_sum(clv * V, 20) / w_sum(V, 20), base.window(20))
        up_v = np.where(on_r, np.where(r > 0, V, 0.0), np.nan)
        put("up_vol_share_30", w_sum(up_v, 30) / w_sum(v_r, 30), ra.window(30))

        # Return distribution
        for n in (30, 90):
            skew, kurt = w_moments(r, n)
            put(f"skew_{n}", skew, ra.window(n))
            put(f"kurt_{n}", kurt, ra.window(n))
            put(f"acf1_{n}", w_acf1(r, n), ra.window(n))
        for n in (14, 30):
            put(f"up_share_{n}", w_up_share(r, n), ra.window(n))
        put("maxdd_30", w_maxdd(C, 30), base.window(30))
    return {"source": src, "sigma30": data["vol_cc_30"]}


def past_class_column(table: Table, k: float) -> None:
    """1, 0 or -1 by the rule 4 thresholds applied to past_z_7."""
    past = table.data["past_z_7"]
    have = ~np.isnan(past)
    values = np.where(past > k, 1.0, np.where(past < -k, -1.0, 0.0))
    table.put("past_class_7", values, Avail(have, have), np.zeros(len(past), dtype=bool))


def calendar_columns(table: Table) -> None:
    idx = table.index
    always = Avail(np.ones(len(idx), dtype=bool), np.ones(len(idx), dtype=bool))
    none = np.zeros(len(idx), dtype=bool)
    refs = pd.DatetimeIndex([GENESIS, *HALVINGS])
    which = refs.searchsorted(idx, side="right") - 1
    days = np.asarray((idx - refs[np.clip(which, 0, None)]).days, dtype=float)
    days = np.where(which >= 0, days, np.nan)
    table.put("dow", idx.dayofweek.to_numpy(dtype=float), always, none)
    table.put("month", idx.month.to_numpy(dtype=float), always, none)
    table.put("days_since_halving", np.log1p(days), always, none)
    table.put("halving_phase", days / CYCLE_DAYS, always, none)


def onchain_columns(table: Table, asof: pd.DataFrame) -> None:
    """Coin Metrics columns from the as-of series (ruling 15)."""
    with np.errstate(all="ignore"):
        for metric in ONCHAIN + ("CapMVRVCur",):
            v = asof[metric].to_numpy(dtype=float)
            # A value of zero or less has no logarithm: a missing day of that
            # metric (only TxCnt, FeeTotNtv and AdrActCnt, in 2009-2010).
            v = np.where(v > 0, v, np.nan)
            have = ~np.isnan(v)
            first = int(np.argmax(have)) if have.any() else len(v)
            source = Source(have, np.arange(len(v)) >= first)
            src, inside = source.base, source.inside
            if metric == "CapMVRVCur":
                mv = np.log(v)
                table.put("mvrv", mv, src, inside)
                table.put("mvrv_z365", zscore(mv, 365), src & src.window(365), inside)
                continue
            name = f"oc_{metric.lower()}"
            table.put(f"{name}_g7", np.log(v / shift(v, 7)), src & src.lag(7), inside)
            table.put(f"{name}_g30", np.log(v / shift(v, 30)), src & src.lag(30), inside)
            table.put(f"{name}_z365", zscore(np.log(v), 365), src & src.window(365), inside)


# ------------------------------------------------------------- Binance --

def binance_columns(table: Table, spot: pd.DataFrame, perp: pd.DataFrame,
                    funding: pd.Series, bitstamp: Source, bitstamp_close: np.ndarray,
                    firsts: dict) -> None:
    """The Binance columns of table B. ``firsts`` holds each source's first day."""
    idx = table.index
    sc = spot["close"].to_numpy(dtype=float)
    pc = perp["close"].to_numpy(dtype=float)
    fund = funding.to_numpy(dtype=float)
    s_src = Source(~np.isnan(sc), np.asarray(idx >= firsts["spot"]))
    p_src = Source(~np.isnan(pc), np.asarray(idx >= firsts["perp"]))
    f_src = Source(~np.isnan(fund), np.asarray(idx >= firsts["funding"]))
    s_in, p_in, f_in, b_in = s_src.inside, p_src.inside, f_src.inside, bitstamp.inside
    As, Ap, Af = s_src.base, p_src.base, f_src.base
    data = table.data
    with np.errstate(all="ignore"):
        table.put("fund_1d", fund, Af, f_in)
        for n in (7, 30):
            table.put(f"fund_{n}", w_total(fund, n), Af.window(n), f_in)
        table.put("fund_z_90", zscore(fund, 90), Af & Af.window(90), f_in)

        both_in = s_in | p_in
        basis_a = table.put("basis", np.log(pc / sc), Ap & As, both_in)
        table.put("basis_mean_7", w_mean(data["basis"], 7), basis_a.window(7), both_in)
        table.put("basis_z_90", zscore(data["basis"], 90), basis_a & basis_a.window(90), both_in)

        prem_in = s_in | b_in
        prem_a = table.put("usdt_prem", np.log(sc / bitstamp_close), As & bitstamp.base, prem_in)
        table.put("usdt_prem_z_90", zscore(data["usdt_prem"], 90), prem_a & prem_a.window(90),
                  prem_in)

        sqv = spot["quote_volume"].to_numpy(dtype=float)
        pqv = perp["quote_volume"].to_numpy(dtype=float)
        lvol_a = table.put("perp_spot_lvol", np.log(pqv / sqv), Ap & As, both_in)
        table.put("perp_spot_lvol_z_90", zscore(data["perp_spot_lvol"], 90),
                  lvol_a & lvol_a.window(90), both_in)

        for name, frame, avail, inside in (("spot", spot, As, s_in), ("perp", perp, Ap, p_in)):
            share = frame["taker_buy_quote"].to_numpy(dtype=float) / frame["quote_volume"].to_numpy(dtype=float)
            col = f"taker_share_{name}"
            share_a = table.put(col, share, avail, inside)
            table.put(f"{col}_mean_7", w_mean(data[col], 7), share_a.window(7), inside)
            table.put(f"{col}_z_90", zscore(data[col], 90), share_a & share_a.window(90), inside)

        trades = spot["trades"].to_numpy(dtype=float)
        quarter = As & As.window(90)
        table.put("trades_z_90", zscore(np.log(trades), 90), quarter, s_in)
        table.put("avg_trade_z_90", zscore(np.log(sqv / trades), 90), quarter, s_in)
        table.put("bn_lvol_z_30", zscore(np.log(sqv), 30), As & As.window(30), s_in)


# ------------------------------------------------------------------ FX --

def fx_logs(rates: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """ln of each FX series on the fixing dates, and where all seven exist.

    ``rates`` holds the seven currencies per EUR, one row per fixing date.
    A fixing date that lacks any of them has no value in any series.
    """
    per_eur = {c: rates[c].to_numpy(dtype=float) for c in ECB_CURRENCIES}
    usd = per_eur["USD"]
    with np.errstate(all="ignore"):
        logs = {"eurusd": np.log(usd),
                "usdjpy": np.log(per_eur["JPY"] / usd),
                "gbpusd": np.log(usd / per_eur["GBP"]),
                "usdchf": np.log(per_eur["CHF"] / usd),
                "audusd": np.log(usd / per_eur["AUD"]),
                "usdcad": np.log(per_eur["CAD"] / usd)}
        parts = dict(logs, usdsek=np.log(per_eur["SEK"] / usd))
        dxy = np.zeros(len(usd))
        for name, weight in DXY_WEIGHTS:
            dxy = dxy + weight * parts[name]
        logs["dxy"] = dxy
    present = np.ones(len(usd), dtype=bool)
    for c in ECB_CURRENCIES:
        present &= ~np.isnan(per_eur[c])
    return {name: np.where(present, v, np.nan) for name, v in logs.items()}, present


def fx_columns(fx_table: Table, logs: dict, present: np.ndarray,
               btc_close: np.ndarray, btc_history: np.ndarray) -> None:
    """The 43 FX columns on the fixing sequence, one row per fixing date.

    A window of n fixings is the n fixing dates ending with the row's own.
    A fixing date that lacks a currency is a missing day of the ECB source.
    ``btc_close`` is Bitstamp's close of each fixing date (missing on its
    missing days); ``btc_history`` says whether that date lies inside
    Bitstamp's history.
    """
    ecb = Source(present, np.ones(len(present), dtype=bool))
    inside = ecb.inside
    base = ecb.base
    change = base & base.lag(1)
    with np.errstate(all="ignore"):
        for s in FX_SERIES:
            x = logs[s]
            d = x - shift(x, 1)
            for n in FX_RET_N:
                fx_table.put(f"fx_{s}_ret_{n}", x - shift(x, n), base & base.lag(n), inside)
            vol_a = fx_table.put(f"fx_{s}_vol_20", w_std(d, 20), change.window(20), inside)
            fx_table.put(f"fx_{s}_vol_z_250",
                         zscore(np.log(fx_table.data[f"fx_{s}_vol_20"]), 250),
                         vol_a & vol_a.window(250), inside)
        # Pairs from the fixing date before, F', to F: the change of ln(dxy),
        # and BTC's log return from Bitstamp's close of F' to its close of F.
        # A pair with either BTC close missing is left out of the window.
        btc = Source(~np.isnan(btc_close), np.asarray(btc_history, dtype=bool)).base
        pair = change & btc & btc.lag(1)
        dxy_change = np.where(pair.have, logs["dxy"] - shift(logs["dxy"], 1), np.nan)
        btc_return = np.where(pair.have, np.log(btc_close / shift(btc_close, 1)), np.nan)
        fx_table.put("btc_dxy_corr_60", w_corr(dxy_change, btc_return, 60),
                     pair.window(60), inside)


def carry_fx(table: Table, fx_table: Table, fix_dates: pd.DatetimeIndex,
             bitstamp_inside: np.ndarray) -> pd.Series:
    """Row t takes the FX columns of the latest fixing date F on or before t.

    The fixing of F is known at the close of BTC day F. btc_dxy_corr_60 is
    also a Bitstamp column, so it has no value on a day inside a longer
    Bitstamp gap (ruling 23). Returns F for each row, NaT before the first
    fixing date.
    """
    pos = fix_dates.searchsorted(table.index, side="right") - 1
    known = pos >= 0
    take = np.clip(pos, 0, None)
    for name in FX_COLUMNS:
        if len(fix_dates):
            values = np.where(known, fx_table.data[name][take], np.nan)
        else:
            values = np.full(len(table.index), np.nan)
        if name == "btc_dxy_corr_60":
            values = np.where(bitstamp_inside, np.nan, values)
        table.data[name] = values
    day = pd.Series(pd.NaT, index=table.index, dtype="datetime64[ns]")
    if len(fix_dates):
        day[known] = fix_dates[take[known]]
    return day
