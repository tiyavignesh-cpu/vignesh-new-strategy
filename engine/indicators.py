"""
Indicator library for the pullback strategy.

Everything here is causal: a value at bar t uses only bars <= t. Donchian
channels in particular EXCLUDE the current bar, because a channel that includes
today's high can never be "breached" today -- it just tracks price, and the
breakout signal becomes tautological.
"""

from __future__ import annotations

import pandas as pd


def moving_average(series: pd.Series, period: int, kind: str = "EMA") -> pd.Series:
    kind = kind.upper()
    if kind == "SMA":
        return series.rolling(period, min_periods=period).mean()
    if kind == "EMA":
        return series.ewm(span=period, adjust=False, min_periods=period).mean()
    raise ValueError("kind must be SMA or EMA")


def bollinger(
    close: pd.Series, period: int = 20, mult: float = 2.0, kind: str = "SMA"
) -> pd.DataFrame:
    """Standard Bollinger Bands. Population std (ddof=0), the platform default."""
    mid = moving_average(close, period, kind)
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": mid + mult * sd, "bb_lower": mid - mult * sd,
         "bb_width": 2 * mult * sd},
        index=close.index,
    )


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
    """Donchian channel over the PRIOR `period` bars, excluding the current bar.

    Shifting by one is what makes "close breaks the upper band" a real event
    rather than a restatement of the current bar's own high.
    """
    upper = high.rolling(period, min_periods=period).max().shift(1)
    lower = low.rolling(period, min_periods=period).min().shift(1)
    return pd.DataFrame(
        {"dc_upper": upper, "dc_lower": lower, "dc_width": upper - lower},
        index=high.index,
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ATR."""
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def slope(series: pd.Series, lookback: int = 3) -> pd.Series:
    """Simple rise-over-run slope of a line, normalised by its own level.

    Normalising means the slope threshold means the same thing on a Rs 80 stock
    and a Rs 8,000 one.
    """
    return (series - series.shift(lookback)) / (lookback * series.abs())


def rejection_candle(
    o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
    direction: str, close_pct: float = 0.6, min_range_frac: float = 0.0,
    atr_series: pd.Series | None = None,
) -> pd.Series:
    """Bullish/bearish rejection bar.

    Bullish: closes up, and closes in the top `close_pct` of its own range --
    i.e. it probed lower and was rejected. Optionally require the bar to have
    real range relative to ATR, which filters doji-on-doji noise.
    """
    rng = (h - l).replace(0, pd.NA)
    if direction == "LONG":
        body_ok = c > o
        pos_ok = (c - l) / rng >= close_pct
    elif direction == "SHORT":
        body_ok = c < o
        pos_ok = (h - c) / rng >= close_pct
    else:
        raise ValueError("direction must be LONG or SHORT")

    ok = body_ok & pos_ok.fillna(False).astype(bool)
    if atr_series is not None and min_range_frac > 0:
        ok = ok & ((h - l) >= min_range_frac * atr_series)
    return ok.fillna(False).astype(bool)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0, 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ADX."""
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr_series = atr(high, low, close, period=period)
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(alpha=1.0/period, adjust=False).mean() / tr_series
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(alpha=1.0/period, adjust=False).mean() / tr_series
    
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    return dx.ewm(alpha=1.0/period, adjust=False).mean()


import numpy as np

__all__ = [
    "moving_average", "bollinger", "donchian", "atr", "slope", "rejection_candle",
    "rsi", "adx",
]

