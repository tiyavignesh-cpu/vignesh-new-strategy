"""
engine/strategies/t3_momentum.py — Tillson T3 Trend and Momentum Strategy.

Implements Tim Tillson's T3 Moving Average (triple-smoothed DEMA):
- Six cascaded EMAs for ultra-smooth, low-lag trend filtering
- Bullish regime (Close > SMA50 > SMA200):
    - Price crosses above T3 with upward slope
    - LONG Signal generated
- Bearish regime (Close < SMA50 < SMA200):
    - Price crosses below T3 with downward slope
    - SHORT Signal generated
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from engine.regime import compute_regime_series, TrendRegime
from engine.indicators import atr
from .signals import StrategySignal


def compute_t3(series: pd.Series, period: int = 8, vfactor: float = 0.7) -> pd.Series:
    """Calculate Tillson T3 Moving Average."""
    e1 = series.ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    e3 = e2.ewm(span=period, adjust=False).mean()
    e4 = e3.ewm(span=period, adjust=False).mean()
    e5 = e4.ewm(span=period, adjust=False).mean()
    e6 = e5.ewm(span=period, adjust=False).mean()

    c1 = -vfactor ** 3
    c2 = 3 * vfactor ** 2 + 3 * vfactor ** 3
    c3 = -6 * vfactor ** 2 - 3 * vfactor - 3 * vfactor ** 3
    c4 = 1 + 3 * vfactor + vfactor ** 3 + 3 * vfactor ** 2

    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


@dataclass(frozen=True)
class T3Config:
    t3_period: int = 8
    t3_vfactor: float = 0.7
    min_tp_r: float = 2.0
    atr_period: int = 14
    stop_atr_mult: float = 1.5
    style: str = "SWING"
    timeframe: str = "1D"


def scan_t3_momentum(
    symbol: str,
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    cfg: Optional[T3Config] = None,
) -> List[StrategySignal]:
    """Scan symbol for causal Tillson T3 momentum and trend continuation signals."""
    cfg = cfg or T3Config()
    signals: List[StrategySignal] = []

    if len(df) < 200:
        return signals

    df = df.sort_index()
    bench = benchmark_df.sort_index()

    regimes = compute_regime_series(bench["Close"])
    t3_series = compute_t3(df["Close"], period=cfg.t3_period, vfactor=cfg.t3_vfactor)
    t3_slope = t3_series.diff()
    atr_s = atr(df["High"], df["Low"], df["Close"], period=cfg.atr_period)

    for i in range(200, len(df)):
        t = df.index[i]
        c = float(df["Close"].iloc[i])
        c_prior = float(df["Close"].iloc[i-1])

        current_regime = regimes.get(t, TrendRegime.NEUTRAL.value)
        if current_regime == TrendRegime.NEUTRAL.value:
            continue

        t3_val = float(t3_series.iloc[i])
        t3_prior = float(t3_series.iloc[i-1])
        slope = float(t3_slope.iloc[i])
        cur_atr = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else (c * 0.015)

        # Bullish Regime -> T3 Bullish Cross & Upward Slope
        if current_regime == TrendRegime.BULLISH.value and slope > 0:
            if c_prior <= t3_prior and c > t3_val:
                stop_px = round(max(t3_val - cfg.stop_atr_mult * cur_atr, c - 2.0 * cur_atr), 2)
                stop_dist = c - stop_px
                if stop_dist > 0:
                    target_px = round(c + cfg.min_tp_r * stop_dist, 2)
                    signals.append(
                        StrategySignal(
                            symbol=symbol,
                            strategy="T3_MOMENTUM",
                            style=cfg.style, # type: ignore
                            side="BUY",
                            timestamp=t,
                            entry=c,
                            stop=stop_px,
                            target=target_px,
                            rules=f"T3 Momentum Bullish Cross above {t3_val:.2f} (Slope > 0)",
                            timeframe=cfg.timeframe,
                            tp1=round(c + 1.5 * stop_dist, 2),
                            tp2=target_px,
                        )
                    )

        # Bearish Regime -> T3 Bearish Cross & Downward Slope
        elif current_regime == TrendRegime.BEARISH.value and slope < 0:
            if c_prior >= t3_prior and c < t3_val:
                stop_px = round(min(t3_val + cfg.stop_atr_mult * cur_atr, c + 2.0 * cur_atr), 2)
                stop_dist = stop_px - c
                if stop_dist > 0:
                    target_px = round(c - cfg.min_tp_r * stop_dist, 2)
                    signals.append(
                        StrategySignal(
                            symbol=symbol,
                            strategy="T3_MOMENTUM",
                            style=cfg.style, # type: ignore
                            side="SELL",
                            timestamp=t,
                            entry=c,
                            stop=stop_px,
                            target=target_px,
                            rules=f"T3 Momentum Bearish Cross below {t3_val:.2f} (Slope < 0)",
                            timeframe=cfg.timeframe,
                            tp1=round(c - 1.5 * stop_dist, 2),
                            tp2=target_px,
                        )
                    )

    return signals


__all__ = ["T3Config", "compute_t3", "scan_t3_momentum"]
