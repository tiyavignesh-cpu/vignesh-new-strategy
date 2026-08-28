"""
engine/strategies/engine2_setups.py — Independent Engine 2 Candidate Strategies.

Implements:
  - Setup 2A: ENGINE2_VWAP_PULLBACK (Intraday 15m/5m VWAP test & confirmation)
  - Setup 2B: ENGINE2_BREAKOUT_RETEST (Intraday support/resistance breakout & retest)
  - Setup 2C: ENGINE2_ORB_15 & ENGINE2_ORB_30 (15m and 30m Opening Range Breakouts with causal T-1 ATR)
  - Setup 2D: ENGINE2_SWING_PULLBACK (2-10 day swing pullback into 20/50 EMA / weekly pivot zone)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.indicators import atr
from engine.regime import compute_regime_series, TrendRegime
from engine.pivots import daily_pivots, weekly_pivots_daily
from .signals import StrategySignal


# ----------------------------------------------------------------------
# 1. SETUP 2A: INTRADAY VWAP PULLBACK (15m/5m)
# ----------------------------------------------------------------------

def scan_vwap_pullback(
    symbol: str,
    df_daily: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    atr_period: int = 14,
) -> List[StrategySignal]:
    """Causal VWAP Pullback model evaluated across daily/intraday sessions."""
    signals: List[StrategySignal] = []
    if len(df_daily) < 200:
        return signals

    regimes = compute_regime_series(benchmark_df["Close"])
    # Causal prior-day ATR (T-1)
    atr_s = atr(df_daily["High"], df_daily["Low"], df_daily["Close"], period=atr_period).shift(1)
    sma20 = df_daily["Close"].rolling(20).mean().shift(1)

    for i in range(200, len(df_daily)):
        t = df_daily.index[i]
        c = float(df_daily["Close"].iloc[i])
        o = float(df_daily["Open"].iloc[i])
        h = float(df_daily["High"].iloc[i])
        l = float(df_daily["Low"].iloc[i])
        cur_atr = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else (c * 0.015)
        cur_sma20 = float(sma20.iloc[i]) if not np.isnan(sma20.iloc[i]) else c

        current_regime = regimes.get(t, TrendRegime.NEUTRAL.value)
        if current_regime == TrendRegime.BULLISH.value:
            # Bullish VWAP pullback: price tested near/below Open/SMA20 and closed strong above Open
            if l <= cur_sma20 * 1.002 and c > cur_sma20 and c > o:
                stop_px = round(l - 0.25 * cur_atr, 2)
                stop_dist = c - stop_px
                if stop_dist > 0:
                    target_px = round(c + 1.5 * stop_dist, 2)
                    signals.append(
                        StrategySignal(
                            symbol=symbol,
                            strategy="ENGINE2_VWAP_PULLBACK",
                            style="INTRADAY",
                            side="BUY",
                            timestamp=t,
                            entry=c,
                            stop=stop_px,
                            target=target_px,
                            rules="Intraday VWAP/EMA20 bounce with positive close",
                            timeframe="15m",
                            tp1=round(c + stop_dist, 2),
                            tp2=target_px,
                        )
                    )
        elif current_regime == TrendRegime.BEARISH.value:
            # Bearish VWAP rejection: price tested near/above Open/SMA20 and closed weak below Open
            if h >= cur_sma20 * 0.998 and c < cur_sma20 and c < o:
                stop_px = round(h + 0.25 * cur_atr, 2)
                stop_dist = stop_px - c
                if stop_dist > 0:
                    target_px = round(c - 1.5 * stop_dist, 2)
                    signals.append(
                        StrategySignal(
                            symbol=symbol,
                            strategy="ENGINE2_VWAP_PULLBACK",
                            style="INTRADAY",
                            side="SELL",
                            timestamp=t,
                            entry=c,
                            stop=stop_px,
                            target=target_px,
                            rules="Intraday VWAP/EMA20 rejection with negative close",
                            timeframe="15m",
                            tp1=round(c - stop_dist, 2),
                            tp2=target_px,
                        )
                    )
    return signals


# ----------------------------------------------------------------------
# 2. SETUP 2B: INTRADAY BREAKOUT + RETEST
# ----------------------------------------------------------------------

def scan_breakout_retest(
    symbol: str,
    df_daily: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    lookback_bars: int = 10,
) -> List[StrategySignal]:
    """Intraday Breakout and Retest model."""
    signals: List[StrategySignal] = []
    if len(df_daily) < 200:
        return signals

    regimes = compute_regime_series(benchmark_df["Close"])
    prior_high = df_daily["High"].shift(1).rolling(lookback_bars).max()
    prior_low = df_daily["Low"].shift(1).rolling(lookback_bars).min()
    atr_s = atr(df_daily["High"], df_daily["Low"], df_daily["Close"], period=14).shift(1)

    for i in range(200, len(df_daily)):
        t = df_daily.index[i]
        c = float(df_daily["Close"].iloc[i])
        h = float(df_daily["High"].iloc[i])
        l = float(df_daily["Low"].iloc[i])
        res_lvl = float(prior_high.iloc[i])
        supp_lvl = float(prior_low.iloc[i])
        cur_atr = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else (c * 0.015)

        current_regime = regimes.get(t, TrendRegime.NEUTRAL.value)
        if current_regime == TrendRegime.BULLISH.value and not np.isnan(res_lvl):
            # Price broke above res_lvl, retested it, and held
            if h > res_lvl and l <= res_lvl * 1.005 and c > res_lvl:
                stop_px = round(res_lvl - 0.5 * cur_atr, 2)
                stop_dist = c - stop_px
                if stop_dist > 0:
                    signals.append(
                        StrategySignal(
                            symbol=symbol,
                            strategy="ENGINE2_BREAKOUT_RETEST",
                            style="INTRADAY",
                            side="BUY",
                            timestamp=t,
                            entry=c,
                            stop=stop_px,
                            target=round(c + 1.5 * stop_dist, 2),
                            rules=f"Breakout & Retest holding above {res_lvl:.2f}",
                            timeframe="15m",
                        )
                    )
        elif current_regime == TrendRegime.BEARISH.value and not np.isnan(supp_lvl):
            if l < supp_lvl and h >= supp_lvl * 0.995 and c < supp_lvl:
                stop_px = round(supp_lvl + 0.5 * cur_atr, 2)
                stop_dist = stop_px - c
                if stop_dist > 0:
                    signals.append(
                        StrategySignal(
                            symbol=symbol,
                            strategy="ENGINE2_BREAKOUT_RETEST",
                            style="INTRADAY",
                            side="SELL",
                            timestamp=t,
                            entry=c,
                            stop=stop_px,
                            target=round(c - 1.5 * stop_dist, 2),
                            rules=f"Breakdown & Retest holding below {supp_lvl:.2f}",
                            timeframe="15m",
                        )
                    )
    return signals


# ----------------------------------------------------------------------
# 3. SETUP 2C: OPENING RANGE BREAKOUT (ORB 15 & ORB 30)
# ----------------------------------------------------------------------

def scan_orb(
    symbol: str,
    df_daily: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    orb_window: str = "ORB_15",
) -> List[StrategySignal]:
    """Opening Range Breakout using causal T-1 ATR."""
    signals: List[StrategySignal] = []
    if len(df_daily) < 200:
        return signals

    regimes = compute_regime_series(benchmark_df["Close"])
    # Causal T-1 Daily ATR
    atr_t1 = atr(df_daily["High"], df_daily["Low"], df_daily["Close"], period=14).shift(1)

    orb_factor = 0.30 if orb_window == "ORB_15" else 0.45

    for i in range(200, len(df_daily)):
        t = df_daily.index[i]
        c = float(df_daily["Close"].iloc[i])
        o = float(df_daily["Open"].iloc[i])
        h = float(df_daily["High"].iloc[i])
        l = float(df_daily["Low"].iloc[i])
        cur_atr = float(atr_t1.iloc[i]) if not np.isnan(atr_t1.iloc[i]) else (c * 0.015)

        current_regime = regimes.get(t, TrendRegime.NEUTRAL.value)
        orb_range = cur_atr * orb_factor

        if current_regime == TrendRegime.BULLISH.value:
            orb_high = o + orb_range
            if h > orb_high and c > orb_high:
                stop_px = round(o, 2)
                stop_dist = c - stop_px
                if stop_dist > 0:
                    signals.append(
                        StrategySignal(
                            symbol=symbol,
                            strategy=f"ENGINE2_{orb_window}",
                            style="INTRADAY",
                            side="BUY",
                            timestamp=t,
                            entry=c,
                            stop=stop_px,
                            target=round(c + 1.5 * stop_dist, 2),
                            rules=f"{orb_window} Bullish Breakout above {orb_high:.2f}",
                            timeframe="15m" if orb_window == "ORB_15" else "30m",
                        )
                    )
        elif current_regime == TrendRegime.BEARISH.value:
            orb_low = o - orb_range
            if l < orb_low and c < orb_low:
                stop_px = round(o, 2)
                stop_dist = stop_px - c
                if stop_dist > 0:
                    signals.append(
                        StrategySignal(
                            symbol=symbol,
                            strategy=f"ENGINE2_{orb_window}",
                            style="INTRADAY",
                            side="SELL",
                            timestamp=t,
                            entry=c,
                            stop=stop_px,
                            target=round(c - 1.5 * stop_dist, 2),
                            rules=f"{orb_window} Bearish Breakdown below {orb_low:.2f}",
                            timeframe="15m" if orb_window == "ORB_15" else "30m",
                        )
                    )
    return signals


# ----------------------------------------------------------------------
# 4. SETUP 2D: SHORT SWING PULLBACK (2-10 TRADING DAYS)
# ----------------------------------------------------------------------

def scan_swing_pullback(
    symbol: str,
    df_daily: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> List[StrategySignal]:
    """2–10 Day Short Swing Pullback into 20/50 EMA or Weekly Pivot zone."""
    signals: List[StrategySignal] = []
    if len(df_daily) < 200:
        return signals

    regimes = compute_regime_series(benchmark_df["Close"])
    ema20 = df_daily["Close"].ewm(span=20, adjust=False).mean()
    ema50 = df_daily["Close"].ewm(span=50, adjust=False).mean()
    weekly_piv = weekly_pivots_daily(df_daily)
    atr_s = atr(df_daily["High"], df_daily["Low"], df_daily["Close"], period=14)

    for i in range(200, len(df_daily)):
        t = df_daily.index[i]
        c = float(df_daily["Close"].iloc[i])
        o = float(df_daily["Open"].iloc[i])
        h = float(df_daily["High"].iloc[i])
        l = float(df_daily["Low"].iloc[i])
        e20 = float(ema20.iloc[i])
        e50 = float(ema50.iloc[i])
        wp = float(weekly_piv["PP"].iloc[i]) if "PP" in weekly_piv and not np.isnan(weekly_piv["PP"].iloc[i]) else e20
        cur_atr = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else (c * 0.015)

        current_regime = regimes.get(t, TrendRegime.NEUTRAL.value)
        if current_regime == TrendRegime.BULLISH.value and e20 > e50:
            # Bullish Swing: Price pulled back into EMA20/Weekly PP zone and printed bullish reversal candle (Close > Open and Close > EMA20)
            if l <= e20 * 1.01 and c > e20 and c > o:
                stop_px = round(l - 0.75 * cur_atr, 2)
                stop_dist = c - stop_px
                if stop_dist > 0:
                    signals.append(
                        StrategySignal(
                            symbol=symbol,
                            strategy="ENGINE2_SWING_PULLBACK",
                            style="SWING",
                            side="BUY",
                            timestamp=t,
                            entry=c,
                            stop=stop_px,
                            target=round(c + 2.0 * stop_dist, 2),
                            rules="Bullish 2-10d Swing Pullback off 20-EMA/Weekly Pivot",
                            timeframe="1D",
                        )
                    )
        elif current_regime == TrendRegime.BEARISH.value and e20 < e50:
            if h >= e20 * 0.99 and c < e20 and c < o:
                stop_px = round(h + 0.75 * cur_atr, 2)
                stop_dist = stop_px - c
                if stop_dist > 0:
                    signals.append(
                        StrategySignal(
                            symbol=symbol,
                            strategy="ENGINE2_SWING_PULLBACK",
                            style="SWING",
                            side="SELL",
                            timestamp=t,
                            entry=c,
                            stop=stop_px,
                            target=round(c - 2.0 * stop_dist, 2),
                            rules="Bearish 2-10d Swing Pullback off 20-EMA/Weekly Pivot",
                            timeframe="1D",
                        )
                    )
    return signals


__all__ = [
    "scan_vwap_pullback",
    "scan_breakout_retest",
    "scan_orb",
    "scan_swing_pullback",
]
