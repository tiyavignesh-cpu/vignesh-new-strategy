"""
engine/strategies/signals.py — Unified Strategy Signal Schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Literal, Optional, Dict, Any


@dataclass(frozen=True)
class StrategySignal:
    """Standardized trading signal schema across all engines."""
    symbol: str
    strategy: str                     # 'PIVOT_PULLBACK', 'DONCHIAN', 'T3_MOMENTUM'
    style: Literal["SWING", "INTRADAY"]
    side: Literal["BUY", "SELL", "LONG", "SHORT"]
    timestamp: datetime | date
    entry: float
    stop: float
    target: float
    rules: str
    qty_hint: Optional[int] = None
    timeframe: str = "1D"             # '1D', '15m', '30m'
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None

    @property
    def r_multiple(self) -> float:
        """Target distance in terms of initial stop distance (R)."""
        stop_dist = abs(self.entry - self.stop)
        if stop_dist <= 0:
            return 0.0
        return abs(self.target - self.entry) / stop_dist

    def is_long(self) -> bool:
        return self.side.upper() in ("BUY", "LONG")

    def is_short(self) -> bool:
        return self.side.upper() in ("SELL", "SHORT")


__all__ = ["StrategySignal"]
