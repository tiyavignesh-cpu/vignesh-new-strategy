"""
engine/strategies/pivot_pullback.py — Pivot Pullback Strategy (Swing & Intraday).

Implements support/resistance bounce and rejection off classic daily/weekly pivots:
- Bullish regime (Close > SMA50 > SMA200):
    1. Price approaches/trades through support (Pivot, S1, or SMA50)
    2. Price closes back above support (confirmation)
    3. Volume confirms reversal (Volume > 20-SMA Volume)
    4. LONG Signal generated with initial stop below reaction low and target >= 1.5R
- Bearish regime (Close < SMA50 < SMA200):
    1. Price approaches/trades through resistance (Pivot, R1, or SMA50)
    2. Price closes back below resistance (rejection)
    3. Volume confirms rejection (Volume > 20-SMA Volume)
    4. SHORT Signal generated with initial stop above reaction high and target >= 1.5R
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from engine.pivots import daily_pivots, weekly_pivots_daily
from engine.regime import compute_regime_series, TrendRegime
from engine.indicators import atr
from .signals import StrategySignal


@dataclass(frozen=True)
class PivotPullbackConfig:
    min_tp_r: float = 1.5
    atr_period: int = 14
    vol_sma_period: int = 20
    touch_tolerance: float = 0.008      # within 0.8% of level
    stop_buffer_atr: float = 0.5        # buffer below reaction extreme in ATRs
    timeframe: str = "1D"               # '1D' for swing, '15m'/'30m' for intraday
    style: str = "SWING"                # 'SWING' or 'INTRADAY'


def scan_pivot_pullback(
    symbol: str,
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    cfg: Optional[PivotPullbackConfig] = None,
) -> List[StrategySignal]:
    """Scan symbol bar-by-bar for causal, confirmed Pivot Pullback setups."""
    cfg = cfg or PivotPullbackConfig()
    signals: List[StrategySignal] = []

    if len(df) < 200:
        return signals

    # Ensure monotonic datetime index
    df = df.sort_index()
    bench = benchmark_df.sort_index()

    # 1. Macro Regime Series
    regimes = compute_regime_series(bench["Close"])
    
    # 2. Pivot Levels (Causal prior-day completed pivots)
    pivots = daily_pivots(df)
    
    # 3. Indicators
    sma50 = df["Close"].rolling(50, min_periods=50).mean()
    sma200 = df["Close"].rolling(200, min_periods=200).mean()
    vol_ma = df["Volume"].rolling(cfg.vol_sma_period, min_periods=cfg.vol_sma_period).mean() if "Volume" in df else None
    atr_s = atr(df["High"], df["Low"], df["Close"], period=cfg.atr_period)

    # Walk bar-by-bar starting after warmup
    for i in range(200, len(df)):
        t = df.index[i]
        c = float(df["Close"].iloc[i])
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        v = float(df["Volume"].iloc[i]) if vol_ma is not None else 1.0
        v_thresh = float(vol_ma.iloc[i]) if vol_ma is not None and not np.isnan(vol_ma.iloc[i]) else 0.0

        current_regime = regimes.get(t, TrendRegime.NEUTRAL.value)
        if current_regime == TrendRegime.NEUTRAL.value:
            continue

        pp = float(pivots["PP"].iloc[i]) if "PP" in pivots and not np.isnan(pivots["PP"].iloc[i]) else np.nan
        s1 = float(pivots["S1"].iloc[i]) if "S1" in pivots and not np.isnan(pivots["S1"].iloc[i]) else np.nan
        r1 = float(pivots["R1"].iloc[i]) if "R1" in pivots and not np.isnan(pivots["R1"].iloc[i]) else np.nan
        s50 = float(sma50.iloc[i]) if not np.isnan(sma50.iloc[i]) else np.nan
        cur_atr = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else (c * 0.015)

        # -------------------------------------------------------------
        # BULLISH REGIME --> LONG SETUPS ONLY
        # -------------------------------------------------------------
        if current_regime == TrendRegime.BULLISH.value:
            # Check candidate support levels: S1, PP, SMA50
            support_levels = [lvl for lvl in (s1, pp, s50) if not np.isnan(lvl) and lvl < c * 1.05]
            for supp in support_levels:
                # 1. Price traded through/touched support zone
                touched = l <= supp * (1.0 + cfg.touch_tolerance)
                # 2. Price closed back above support
                closed_above = c > supp
                # 3. Volume confirms reversal (or positive candle body)
                vol_ok = (v >= v_thresh * 0.9) and (c >= df["Open"].iloc[i])

                if touched and closed_above and vol_ok:
                    stop_px = round(l - (cfg.stop_buffer_atr * cur_atr), 2)
                    stop_dist = c - stop_px
                    if stop_dist > 0:
                        target_px = round(c + max(cfg.min_tp_r * stop_dist, (r1 - c) if not np.isnan(r1) and r1 > c else cfg.min_tp_r * stop_dist), 2)
                        signals.append(
                            StrategySignal(
                                symbol=symbol,
                                strategy="PIVOT_PULLBACK",
                                style=cfg.style, # type: ignore
                                side="BUY",
                                timestamp=t,
                                entry=c,
                                stop=stop_px,
                                target=target_px,
                                rules=f"Bullish Pivot Reversal off {supp:.2f} (S1/PP/SMA50)",
                                timeframe=cfg.timeframe,
                                tp1=round(c + cfg.min_tp_r * stop_dist, 2),
                                tp2=target_px,
                            )
                        )
                        break # One signal per bar

        # -------------------------------------------------------------
        # BEARISH REGIME --> SHORT SETUPS ONLY
        # -------------------------------------------------------------
        elif current_regime == TrendRegime.BEARISH.value:
            # Check candidate resistance levels: R1, PP, SMA50
            resistance_levels = [lvl for lvl in (r1, pp, s50) if not np.isnan(lvl) and lvl > c * 0.95]
            for res in resistance_levels:
                # 1. Price traded through/touched resistance zone
                touched = h >= res * (1.0 - cfg.touch_tolerance)
                # 2. Price closed back below resistance
                closed_below = c < res
                # 3. Volume confirms rejection (or bearish candle body)
                vol_ok = (v >= v_thresh * 0.9) and (c <= df["Open"].iloc[i])

                if touched and closed_below and vol_ok:
                    stop_px = round(h + (cfg.stop_buffer_atr * cur_atr), 2)
                    stop_dist = stop_px - c
                    if stop_dist > 0:
                        target_px = round(c - max(cfg.min_tp_r * stop_dist, (c - s1) if not np.isnan(s1) and s1 < c else cfg.min_tp_r * stop_dist), 2)
                        signals.append(
                            StrategySignal(
                                symbol=symbol,
                                strategy="PIVOT_PULLBACK",
                                style=cfg.style, # type: ignore
                                side="SELL",
                                timestamp=t,
                                entry=c,
                                stop=stop_px,
                                target=target_px,
                                rules=f"Bearish Pivot Rejection off {res:.2f} (R1/PP/SMA50)",
                                timeframe=cfg.timeframe,
                                tp1=round(c - cfg.min_tp_r * stop_dist, 2),
                                tp2=target_px,
                            )
                        )
                        break # One signal per bar

    return signals


__all__ = ["PivotPullbackConfig", "scan_pivot_pullback"]
