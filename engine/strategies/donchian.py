"""
engine/strategies/donchian.py — Donchian Multi-Timeframe Breakout Strategy.

Strictly causal channel breakout:
- Excludes the current bar (uses shift 1 on rolling channel extremes).
- Bullish regime (Close > SMA50 > SMA200):
    - Price crosses above prior 20-bar Upper Channel
    - Volume confirms breakout
    - LONG Signal generated
- Bearish regime (Close < SMA50 < SMA200):
    - Price crosses below prior 20-bar Lower Channel
    - Volume confirms breakdown
    - SHORT Signal generated
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from engine.regime import compute_regime_series, TrendRegime
from engine.indicators import donchian, atr
from .signals import StrategySignal


@dataclass(frozen=True)
class DonchianConfig:
    channel_period: int = 20
    min_tp_r: float = 2.0
    atr_period: int = 14
    vol_sma_period: int = 20
    stop_atr_mult: float = 1.5
    style: str = "SWING"
    timeframe: str = "1D"


def scan_donchian(
    symbol: str,
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    cfg: Optional[DonchianConfig] = None,
) -> List[StrategySignal]:
    """Scan symbol for causal Donchian channel breakouts."""
    cfg = cfg or DonchianConfig()
    signals: List[StrategySignal] = []

    if len(df) < 200:
        return signals

    df = df.sort_index()
    bench = benchmark_df.sort_index()

    regimes = compute_regime_series(bench["Close"])
    
    # Donchian channel (strictly shifted by 1 to exclude current bar)
    upper = df["High"].shift(1).rolling(cfg.channel_period).max()
    lower = df["Low"].shift(1).rolling(cfg.channel_period).min()
    middle = (upper + lower) / 2.0

    atr_s = atr(df["High"], df["Low"], df["Close"], period=cfg.atr_period)
    vol_ma = df["Volume"].rolling(cfg.vol_sma_period, min_periods=cfg.vol_sma_period).mean() if "Volume" in df else None

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

        u_lvl = float(upper.iloc[i]) if not np.isnan(upper.iloc[i]) else np.nan
        l_lvl = float(lower.iloc[i]) if not np.isnan(lower.iloc[i]) else np.nan
        m_lvl = float(middle.iloc[i]) if not np.isnan(middle.iloc[i]) else np.nan
        cur_atr = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else (c * 0.015)

        # Bullish Regime -> LONG Breakout
        if current_regime == TrendRegime.BULLISH.value and not np.isnan(u_lvl):
            prior_close = float(df["Close"].iloc[i-1])
            # Breakout: close crosses above upper channel with volume confirmation
            if prior_close <= u_lvl and c > u_lvl and v >= v_thresh:
                stop_px = round(max(m_lvl, c - cfg.stop_atr_mult * cur_atr), 2)
                stop_dist = c - stop_px
                if stop_dist > 0:
                    target_px = round(c + cfg.min_tp_r * stop_dist, 2)
                    signals.append(
                        StrategySignal(
                            symbol=symbol,
                            strategy="DONCHIAN",
                            style=cfg.style, # type: ignore
                            side="BUY",
                            timestamp=t,
                            entry=c,
                            stop=stop_px,
                            target=target_px,
                            rules=f"Donchian 20-day High Breakout above {u_lvl:.2f}",
                            timeframe=cfg.timeframe,
                            tp1=round(c + 1.5 * stop_dist, 2),
                            tp2=target_px,
                        )
                    )

        # Bearish Regime -> SHORT Breakdown
        elif current_regime == TrendRegime.BEARISH.value and not np.isnan(l_lvl):
            prior_close = float(df["Close"].iloc[i-1])
            # Breakdown: close crosses below lower channel with volume confirmation
            if prior_close >= l_lvl and c < l_lvl and v >= v_thresh:
                stop_px = round(min(m_lvl, c + cfg.stop_atr_mult * cur_atr), 2)
                stop_dist = stop_px - c
                if stop_dist > 0:
                    target_px = round(c - cfg.min_tp_r * stop_dist, 2)
                    signals.append(
                        StrategySignal(
                            symbol=symbol,
                            strategy="DONCHIAN",
                            style=cfg.style, # type: ignore
                            side="SELL",
                            timestamp=t,
                            entry=c,
                            stop=stop_px,
                            target=target_px,
                            rules=f"Donchian 20-day Low Breakdown below {l_lvl:.2f}",
                            timeframe=cfg.timeframe,
                            tp1=round(c - 1.5 * stop_dist, 2),
                            tp2=target_px,
                        )
                    )

    return signals


__all__ = ["DonchianConfig", "scan_donchian"]
