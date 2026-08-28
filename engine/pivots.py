"""
Weekly classical pivots, correctly phased.

The bug being fixed: a naive `df.resample('W')` labels weeks on Sunday and,
depending on how you shift and forward-fill, either leaks the current week's
own high/low into its pivots (lookahead) or serves last-week-but-one's pivots
(a full week of lag).

The contract here:
    For any trading day d, the pivot levels are computed from the OHLC of the
    week that COMPLETED strictly before the week containing d.

So Monday morning of week W already has week W-1's pivots, and those same
levels hold constant Monday through Friday. That is the behaviour a discretionary
trader would have, and it contains no lookahead.

Anchoring is W-FRI, not the pandas default W-SUN, so a week runs Sat..Fri and
is labelled with its Friday.
"""

from __future__ import annotations

import pandas as pd

ANCHOR = "W-FRI"


def weekly_ohlc(df: pd.DataFrame, anchor: str = ANCHOR) -> pd.DataFrame:
    """Resample daily OHLC to weekly bars labelled on the week's Friday."""
    required = {"High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    agg = {"High": "max", "Low": "min", "Close": "last"}
    if "Open" in df.columns:
        agg["Open"] = "first"
    if "Volume" in df.columns:
        agg["Volume"] = "sum"

    weekly = df.resample(anchor, closed="right", label="right").agg(agg)
    return weekly.dropna(subset=["High", "Low", "Close"])


def classical_pivots(high, low, close) -> dict:
    """Five-point classical pivots from a completed period's H/L/C."""
    pp = (high + low + close) / 3.0
    r1 = 2.0 * pp - low
    s1 = 2.0 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    return {"PP": pp, "R1": r1, "S1": s1, "R2": r2, "S2": s2}


def weekly_pivots_daily(df: pd.DataFrame, anchor: str = ANCHOR) -> pd.DataFrame:
    """Return a daily-indexed frame of weekly pivot levels, correctly phased.

    Columns: PP, R1, S1, R2, S2, src_week_end (the Friday of the source week).
    Rows before the first complete source week are NaN/NaT.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must be indexed by a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    weekly = weekly_ohlc(df, anchor)

    # Levels valid DURING week W are built from week W-1's completed bar.
    prior = weekly.shift(1)
    levels = pd.DataFrame(
        classical_pivots(prior["High"], prior["Low"], prior["Close"]),
        index=weekly.index,
    )
    levels["src_week_end"] = weekly.index.to_series().shift(1).values

    # Align by calendar week period so every daily bar picks up its own week's
    # entry -- not the previous label via forward-fill, which is where the
    # one-week lag crept in.
    daily_period = df.index.to_period(anchor)
    levels.index = weekly.index.to_period(anchor)

    out = levels.reindex(daily_period)
    out.index = df.index
    return out


def assert_no_lookahead(df: pd.DataFrame, pivots: pd.DataFrame, anchor: str = ANCHOR) -> None:
    """Raise if any pivot row sources a week that had not completed by that bar.

    Cheap enough to run in CI on every symbol; keep it in the test suite.
    """
    src = pivots["src_week_end"].dropna()
    if src.empty:
        return
    week_start = df.index.to_period(anchor).start_time
    ws = pd.Series(week_start, index=df.index).loc[src.index]
    bad = pd.to_datetime(src.values) >= ws.values
    if bad.any():
        first = src.index[bad][0]
        raise AssertionError(
            f"lookahead at {first}: pivot sources week ending "
            f"{src.loc[first]} which is not strictly before that bar's week"
        )


def daily_pivots(df: pd.DataFrame) -> pd.DataFrame:
    """Return daily pivot levels from the prior completed trading day (shift 1).
    
    Columns: PP, R1, S1, R2, S2, src_date
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must be indexed by a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    prior_h = df["High"].shift(1)
    prior_l = df["Low"].shift(1)
    prior_c = df["Close"].shift(1)

    levels = pd.DataFrame(
        classical_pivots(prior_h, prior_l, prior_c),
        index=df.index,
    )
    levels["src_date"] = df.index.to_series().shift(1).values
    return levels


def intraday_pivots(intraday_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Align previous day's completed classical pivots across intraday bars.
    
    Each intraday bar on date D receives the pivot calculated from day D-1's completed daily bar.
    """
    d_pivots = daily_pivots(daily_df)
    # Map by date
    daily_date_map = d_pivots.copy()
    daily_date_map["date_key"] = daily_date_map.index.date

    intraday_dates = pd.Series(intraday_df.index.date, index=intraday_df.index)
    aligned = pd.merge(
        intraday_dates.rename("date_key").to_frame(),
        daily_date_map.reset_index(),
        on="date_key",
        how="left",
    )
    aligned.index = intraday_df.index
    return aligned[["PP", "R1", "S1", "R2", "S2", "src_date"]]


def assert_constant_within_week(pivots: pd.DataFrame, anchor: str = ANCHOR) -> None:
    """Raise if pivot levels change mid-week (they must not)."""
    period = pivots.index.to_period(anchor)
    for col in ("PP", "R1", "S1", "R2", "S2"):
        nun = pivots[col].groupby(period).nunique(dropna=True)
        if (nun > 1).any():
            bad = nun[nun > 1].index[0]
            raise AssertionError(f"{col} changes within week {bad}")


__all__ = [
    "ANCHOR",
    "weekly_ohlc",
    "classical_pivots",
    "weekly_pivots_daily",
    "daily_pivots",
    "intraday_pivots",
    "assert_no_lookahead",
    "assert_constant_within_week",
]
