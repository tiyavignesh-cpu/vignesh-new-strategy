"""
Engine 1: cross-sectional 12-1 momentum rotation, long only.

Two things the original spec got wrong that are fixed here:

1. RAW vs ADJUSTED PRICES. `auto_adjust=True` back-adjusts for splits AND
   dividends, so adjusted prices are not prices anyone ever traded. Running the
   Rs 50 floor or a turnover floor against them filters on numbers that never
   existed. Signals and returns use ADJUSTED; price and liquidity filters use
   RAW. Both frames are required arguments -- there is no single-frame path,
   deliberately.

2. MONTH-END ANCHORING. `resample('ME')` lands on the calendar month end, which
   is regularly a weekend or an NSE holiday. Anchors snap to the last available
   trading bar instead.

All frames are wide: index = DatetimeIndex of trading days, columns = symbols.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOP_N = 10
HOLD_RANK_BUFFER = 20
REBALANCE_BAND = 0.25
MIN_PRICE = 50.0
MIN_ADT_CR = 5.0          # Rs 5 crore average daily turnover
ADT_WINDOW = 20
VOL_WINDOW = 60
VOL_PERCENTILE_DROP = 0.90
SMA_TREND = 200


def sma(frame: pd.DataFrame | pd.Series, window: int) -> pd.DataFrame | pd.Series:
    return frame.rolling(window, min_periods=window).mean()


def month_end_anchors(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last ACTUAL trading bar of each calendar month."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).last().values)


def momentum_12_1(adj_close: pd.DataFrame) -> pd.DataFrame:
    """12-1 score at each month-end anchor.

    Return from the close 12 months ago to the close 1 month ago, skipping the
    most recent month to shed short-term reversal drag.
    """
    anchors = month_end_anchors(adj_close.index)
    me = adj_close.loc[anchors]
    score = me.shift(1) / me.shift(12) - 1.0
    return score


def volatility_60d(adj_close: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    rets = adj_close.pct_change()
    return rets.rolling(window, min_periods=window).std()


def average_daily_turnover(
    raw_close: pd.DataFrame, raw_volume: pd.DataFrame, window: int = ADT_WINDOW
) -> pd.DataFrame:
    """Rupee turnover, rolling mean. Raw price times raw volume -- never adjusted."""
    return (raw_close * raw_volume).rolling(window, min_periods=window).mean()


def institutional_momentum(adj_close: pd.DataFrame) -> pd.DataFrame:
    """Multi-Timeframe Normalized Momentum Factor (Skip-1-Month Institutional Standard):
    Score = 0.35 * (R12-1 / vol12) + 0.35 * (R6-1 / vol6) + 0.15 * (R3-1 / vol3) + 0.15 * (Price / 52W High)
    """
    p = adj_close
    p_lag = p.shift(21) # Skip most recent month to avoid short-term mean-reversion drag

    r12 = p_lag / p.shift(252) - 1.0
    r6 = p_lag / p.shift(126) - 1.0
    r3 = p_lag / p.shift(63) - 1.0

    ret = p.pct_change()
    vol12 = ret.rolling(252, min_periods=126).std() * np.sqrt(252)
    vol6 = ret.rolling(126, min_periods=63).std() * np.sqrt(252)
    vol3 = ret.rolling(63, min_periods=30).std() * np.sqrt(252)

    vol12 = vol12.clip(lower=0.10)
    vol6 = vol6.clip(lower=0.10)
    vol3 = vol3.clip(lower=0.10)

    score12 = r12 / vol12
    score6 = r6 / vol6
    score3 = r3 / vol3

    high52 = p.rolling(252, min_periods=126).max()
    prox52 = p / high52

    composite = 0.35 * score12 + 0.35 * score6 + 0.15 * score3 + 0.15 * prox52
    return composite


def inverse_volatility_weights(
    selected_syms: list[str],
    vol_series: pd.Series,
    max_weight: float = 0.18,
    min_vol: float = 0.10,
) -> dict[str, float]:
    """Allocate portfolio weights inversely proportional to annualized volatility."""
    if not selected_syms:
        return {}
    
    cur_vols = {s: float(vol_series.get(s, 0.25)) for s in selected_syms}
    inv_vols = {s: 1.0 / max(v if not np.isnan(v) else 0.25, min_vol) for s, v in cur_vols.items()}
    tot_inv = sum(inv_vols.values())
    if tot_inv <= 0:
        return {s: 1.0 / len(selected_syms) for s in selected_syms}
    
    target_weights = {s: min(inv / tot_inv, max_weight) for s, inv in inv_vols.items()}
    tot_w = sum(target_weights.values())
    return {s: w / tot_w for s, w in target_weights.items()}


def dynamic_regime_allocation(benchmark_close: pd.Series, asof: pd.Timestamp) -> float:
    """3-State Dynamic Macro Exposure Model.
    
    1.0 = Bull Trend (NIFTY > 50-DMA and > 200-DMA) -> 100% Equity Exposure
    0.5 = Pullback/Correction (NIFTY > 200-DMA but <= 50-DMA) -> 50% Equity Exposure
    0.0 = Bear Market (NIFTY <= 200-DMA) -> 100% Cash Exposure
    """
    ma50 = benchmark_close.rolling(50, min_periods=50).mean()
    ma200 = benchmark_close.rolling(SMA_TREND, min_periods=SMA_TREND).mean()
    if asof not in benchmark_close.index:
        prior = benchmark_close.index[benchmark_close.index <= asof]
        if len(prior) == 0:
            return 0.0
        asof = prior[-1]
    
    c = benchmark_close.loc[asof]
    m50 = ma50.loc[asof]
    m200 = ma200.loc[asof]
    
    if pd.isna(m200):
        return 0.0
    if c > m200:
        if not pd.isna(m50) and c > m50:
            return 1.0
        return 0.5
    return 0.0


def regime_is_risk_on(benchmark_close: pd.Series, asof: pd.Timestamp) -> bool:
    """Macro circuit breaker: index must be strictly above its own 200-DMA."""
    return dynamic_regime_allocation(benchmark_close, asof) > 0.0


def eligible_universe(
    asof: pd.Timestamp,
    raw_close: pd.DataFrame,
    adj_close: pd.DataFrame,
    raw_volume: pd.DataFrame,
    scores: pd.Series,
    *,
    min_price: float = MIN_PRICE,
    min_adt_cr: float = MIN_ADT_CR,
    vol_percentile_drop: float = VOL_PERCENTILE_DROP,
) -> pd.DataFrame:
    """Apply all four universe filters plus liquidity. Returns a diagnostic frame.

    The `eligible` column is the answer; the rest exist so you can see which
    filter is doing the work in any given month.
    """
    trend = adj_close > sma(adj_close, SMA_TREND)
    vol = volatility_60d(adj_close)
    adt = average_daily_turnover(raw_close, raw_volume)

    px = raw_close.loc[asof]
    out = pd.DataFrame(
        {
            "score": scores.reindex(px.index),
            "raw_price": px,
            "above_200dma": trend.loc[asof].reindex(px.index).fillna(False),
            "vol60": vol.loc[asof].reindex(px.index),
            "adt_cr": adt.loc[asof].reindex(px.index) / 1e7,
        }
    )

    out["f_price"] = out["raw_price"] >= min_price
    out["f_trend"] = out["above_200dma"].astype(bool)
    out["f_score"] = out["score"] > 0.0
    out["f_liquidity"] = out["adt_cr"] >= min_adt_cr

    # Volatility decile is measured only across names that passed the others,
    # otherwise illiquid junk sets the cutoff.
    pre = out["f_price"] & out["f_trend"] & out["f_score"] & out["f_liquidity"]
    if pre.any():
        cutoff = out.loc[pre, "vol60"].quantile(vol_percentile_drop)
        out["f_vol"] = out["vol60"] <= cutoff
    else:
        out["f_vol"] = False

    out["eligible"] = pre & out["f_vol"].fillna(False)
    return out


def select_holdings(
    scores: pd.Series,
    eligible: pd.Series,
    current_holdings: list[str],
    *,
    top_n: int = TOP_N,
    hold_buffer: int = HOLD_RANK_BUFFER,
) -> list[str]:
    """Rank-buffered selection.

    An existing holding is retained while it stays inside the top `hold_buffer`
    AND remains eligible. Losing eligibility (falling under its 200-DMA, going
    illiquid) evicts it regardless of rank -- rank buffering is a turnover
    control, not an override of the risk filters.
    """
    ranked = scores.dropna().sort_values(ascending=False)
    rank = pd.Series(range(1, len(ranked) + 1), index=ranked.index)

    keep = [
        s
        for s in current_holdings
        if s in rank.index
        and rank[s] <= hold_buffer
        and bool(eligible.get(s, False))
    ]

    slots = top_n - len(keep)
    if slots > 0:
        for sym in ranked.index:
            if slots == 0:
                break
            if sym in keep or not bool(eligible.get(sym, False)):
                continue
            keep.append(sym)
            slots -= 1

    # If buffered holds overflow the book, drop the worst-ranked.
    if len(keep) > top_n:
        keep = sorted(keep, key=lambda s: rank[s])[:top_n]

    return keep


def target_weights(
    selected: list[str],
    current_weights: dict[str, float],
    *,
    band: float = REBALANCE_BAND,
) -> dict[str, float]:
    """Equal weight the book, but leave drifted-but-close positions alone.

    Returns the weight to hold. A name already inside the band keeps its
    CURRENT weight (i.e. no order is generated), which is the point of the band.
    """
    if not selected:
        return {}
    target = 1.0 / len(selected)
    out: dict[str, float] = {}
    for sym in selected:
        cw = current_weights.get(sym, 0.0)
        if cw == 0.0:
            out[sym] = target
            continue
        drift = abs(cw - target) / target
        out[sym] = target if drift > band else cw
    return out


def orders_from_weights(
    target: dict[str, float], current: dict[str, float], equity: float
) -> dict[str, float]:
    """Rupee deltas per symbol. Negative = sell. Exits appear as full sells."""
    syms = set(target) | set(current)
    deltas = {}
    for s in syms:
        d = (target.get(s, 0.0) - current.get(s, 0.0)) * equity
        if abs(d) > 1e-9:
            deltas[s] = d
    return deltas


__all__ = [
    "sma",
    "month_end_anchors",
    "momentum_12_1",
    "ensemble_momentum",
    "dynamic_regime_allocation",
    "volatility_60d",
    "average_daily_turnover",
    "regime_is_risk_on",
    "eligible_universe",
    "select_holdings",
    "target_weights",
    "orders_from_weights",
    "TOP_N",
    "HOLD_RANK_BUFFER",
    "REBALANCE_BAND",
]
