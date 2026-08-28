"""
Swing pivot detection, repaint-safe.

This is the single place where the pullback strategy could cheat, so it gets its
own module and its own tests.

A swing high at bar `t` cannot be known at bar `t`. It is only confirmed once
`right` further bars have printed without exceeding it. Every charting package
draws the pivot back at `t` -- which is correct for a chart and catastrophic for
a backtest, because it hands you `right` bars of hindsight.

The contract here: `confirmed_at` is the bar on which the pivot became knowable.
No consumer may use a pivot before its `confirmed_at` bar. `pivot_series()`
enforces this by construction -- values are placed at the confirmation bar, not
at the pivot bar, and `pivot_index` carries the original location for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Pivot:
    kind: str          # "HIGH" or "LOW"
    index: int         # positional index of the pivot bar itself
    price: float
    confirmed_at: int  # positional index of the bar on which it became knowable

    @property
    def lag(self) -> int:
        return self.confirmed_at - self.index


def find_pivots(
    high: pd.Series, low: pd.Series, left: int = 3, right: int = 3
) -> list[Pivot]:
    """Fractal pivots requiring `left` lower highs before and `right` after.

    Ties are resolved strictly on the right side (a later equal high invalidates
    the earlier one), so a flat top yields one pivot, not several.
    """
    if left < 1 or right < 1:
        raise ValueError("left and right must be >= 1")

    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    n = len(h)
    out: list[Pivot] = []

    for i in range(left, n - right):
        if h[i] > h[i - left:i].max() and np.all(h[i + 1:i + 1 + right] < h[i]):
            out.append(Pivot("HIGH", i, float(h[i]), i + right))

        if l[i] < l[i - left:i].min() and np.all(l[i + 1:i + 1 + right] > l[i]):
            out.append(Pivot("LOW", i, float(l[i]), i + right))

    return sorted(out, key=lambda p: p.confirmed_at)


def pivot_series(
    df: pd.DataFrame, left: int = 3, right: int = 3
) -> pd.DataFrame:
    """Pivot info aligned to the CONFIRMATION bar, never the pivot bar.

    Columns:
        last_high, last_low          -- most recently confirmed pivot prices
        last_high_idx, last_low_idx  -- positional index of those pivot bars
    Rows before the first confirmation are NaN. Values forward-fill, so at any
    bar you see the latest pivot that was actually knowable by then.
    """
    pivots = find_pivots(df["High"], df["Low"], left, right)
    n = len(df)

    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    hi_idx = np.full(n, np.nan)
    lo_idx = np.full(n, np.nan)

    for p in pivots:
        c = p.confirmed_at
        if c >= n:
            continue
        if p.kind == "HIGH":
            hi[c], hi_idx[c] = p.price, p.index
        else:
            lo[c], lo_idx[c] = p.price, p.index

    out = pd.DataFrame(
        {
            "last_high": hi,
            "last_low": lo,
            "last_high_idx": hi_idx,
            "last_low_idx": lo_idx,
        },
        index=df.index,
    )
    return out.ffill()


def assert_pivots_are_causal(df: pd.DataFrame, left: int = 3, right: int = 3) -> None:
    """Raise if any pivot claims to be knowable before its confirmation window."""
    for p in find_pivots(df["High"], df["Low"], left, right):
        if p.lag != right:
            raise AssertionError(
                f"{p.kind} pivot at {p.index} has lag {p.lag}, expected {right}"
            )
        if p.confirmed_at <= p.index:
            raise AssertionError("pivot confirmed at or before its own bar")


__all__ = ["Pivot", "find_pivots", "pivot_series", "assert_pivots_are_causal"]
