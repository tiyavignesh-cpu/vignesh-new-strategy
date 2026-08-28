"""
engine/regime.py — Trend Regime Filter (50-DMA and 200-DMA).

Strictly causal macro market regime classification:
- BULLISH : Close > SMA50 > SMA200  --> LONG setups only
- BEARISH : Close < SMA50 < SMA200  --> SHORT setups only
- NEUTRAL : Everything else         --> NO TRADE
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

import numpy as np
import pandas as pd


class TrendRegime(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Causal Simple Moving Average."""
    return series.rolling(window, min_periods=window).mean()


def compute_regime_series(benchmark_close: pd.Series) -> pd.Series:
    """Compute the historical series of trend regimes bar-by-bar.
    
    Returns:
        pd.Series with values 'BULLISH', 'BEARISH', 'NEUTRAL'
    """
    sma50 = compute_sma(benchmark_close, 50)
    sma200 = compute_sma(benchmark_close, 200)

    bullish_mask = (benchmark_close > sma50) & (sma50 > sma200)
    bearish_mask = (benchmark_close < sma50) & (sma50 < sma200)

    regimes = pd.Series(TrendRegime.NEUTRAL.value, index=benchmark_close.index)
    regimes[bullish_mask] = TrendRegime.BULLISH.value
    regimes[bearish_mask] = TrendRegime.BEARISH.value

    # Bars before 200 periods are NEUTRAL
    regimes[sma200.isna()] = TrendRegime.NEUTRAL.value
    return regimes


def get_market_regime(
    benchmark_close: pd.Series,
    asof: pd.Timestamp,
    regime_series: Optional[pd.Series] = None,
) -> TrendRegime:
    """Determine the trend regime in force at `asof` timestamp using only completed data."""
    if regime_series is not None and asof in regime_series.index:
        return TrendRegime(regime_series.loc[asof])

    if asof not in benchmark_close.index:
        prior_dates = benchmark_close.index[benchmark_close.index <= asof]
        if len(prior_dates) == 0:
            return TrendRegime.NEUTRAL
        asof = prior_dates[-1]

    sma50 = compute_sma(benchmark_close, 50).loc[asof]
    sma200 = compute_sma(benchmark_close, 200).loc[asof]
    c = benchmark_close.loc[asof]

    if pd.isna(sma50) or pd.isna(sma200):
        return TrendRegime.NEUTRAL

    if c > sma50 > sma200:
        return TrendRegime.BULLISH
    elif c < sma50 < sma200:
        return TrendRegime.BEARISH
    else:
        return TrendRegime.NEUTRAL


def is_side_permitted_by_regime(side: str, regime: Union[TrendRegime, str]) -> bool:
    """Check if trade side is permitted by current market regime."""
    r_val = regime.value if isinstance(regime, TrendRegime) else str(regime).upper()
    side_up = side.upper()

    if r_val == TrendRegime.BULLISH.value and side_up in ("BUY", "LONG"):
        return True
    if r_val == TrendRegime.BEARISH.value and side_up in ("SELL", "SHORT"):
        return True
    return False


__all__ = [
    "TrendRegime",
    "compute_sma",
    "compute_regime_series",
    "get_market_regime",
    "is_side_permitted_by_regime",
]
