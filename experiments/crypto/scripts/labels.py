"""The targets: sigma30, z, the class, up_7d and down_7d, and k.

Rules 2 to 6 of README.md. Used by build_dataset.py.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

HORIZON = 7
EARLY_SHARE = 0.75
K_QUANTILE = 1 / 3


def forward(close: pd.Series, sigma30: pd.Series) -> pd.DataFrame:
    """ln(C_{t+7} / C_t) and z_t = that / (sigma30_t x sqrt 7).

    ``close`` is Bitstamp's close on the calendar (missing on missing days),
    ``sigma30`` the 30-day standard deviation of daily log returns where the
    missing-data rules give it a value. Both are blank wherever C_t, C_{t+7}
    or sigma30_t is missing, and on the last seven days.
    """
    with np.errstate(all="ignore"):
        fwd = np.log(close.shift(-HORIZON) / close)
        z = fwd / (sigma30 * math.sqrt(HORIZON))
    return pd.DataFrame({"fwd_ret_7": fwd, "z_7": z}, index=close.index)


def rule5_k(z: pd.Series) -> dict:
    """k: the 1/3 quantile of |z| over the earliest 75% of rows that have z.

    ``z`` holds table A's rows in date order. The earliest 75% is the first
    floor(0.75 n) of the n rows with a z. The quantile is numpy's default
    (linear), and k is that rounded to two decimals.
    """
    present = z.dropna()
    n_early = int(math.floor(EARLY_SHARE * len(present)))
    early = present.iloc[:n_early]
    quantile = float(np.quantile(np.abs(early.to_numpy()), K_QUANTILE))
    return {"k": round(quantile, 2), "quantile": quantile, "rows": n_early,
            "first": early.index[0].strftime("%Y-%m-%d"),
            "last": early.index[-1].strftime("%Y-%m-%d")}


def classify(z: pd.Series, k: float) -> pd.DataFrame:
    """UP above +k, DOWN below -k, RANGE otherwise; the two 0/1 targets."""
    have = z.notna()
    cls = pd.Series(pd.NA, index=z.index, dtype="object")
    cls[have & (z > k)] = "UP"
    cls[have & (z < -k)] = "DOWN"
    cls[have & (z >= -k) & (z <= k)] = "RANGE"
    up = pd.Series(pd.NA, index=z.index, dtype="Int64")
    down = pd.Series(pd.NA, index=z.index, dtype="Int64")
    up[have] = (cls[have] == "UP").astype("int64")
    down[have] = (cls[have] == "DOWN").astype("int64")
    return pd.DataFrame({"class_7": cls, "up_7d": up, "down_7d": down}, index=z.index)


def shares(cls: pd.Series) -> dict:
    """Shares of UP, RANGE and DOWN among the rows that have a class."""
    have = cls.dropna()
    n = len(have)
    return {"rows": n, **{c: (round(float((have == c).sum()) / n, 4) if n else None)
                          for c in ("UP", "RANGE", "DOWN")}}