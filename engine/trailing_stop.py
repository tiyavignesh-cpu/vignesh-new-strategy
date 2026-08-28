"""
Dynamic exit engine.

Semantics, settled:
  - Initial stop at entry -/+ 1.5 * ATR14. R = |entry - stop|.
  - Initial target at 2.0R.
  - Touching the target ARMS the trail on that bar, but the trail only becomes
    LIVE from the next bar. This removes intrabar path ambiguity: on the arming
    bar we cannot know whether the high or the pullback came first.
  - Once live, the stop ratchets to 1% behind the high-water mark and never
    loosens, with a hard floor at the original target so a reached target is
    always banked.
  - Time exit after MAX_BARS daily bars.

Longs and shorts are mirror images throughout. (If you drop the short leg -- and
for cash-segment NSE you must, since shorts cannot be held overnight -- the
short path stays here only for futures work later.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Direction = Literal["LONG", "SHORT"]

TRAIL_PCT = 0.01
TARGET_R = 2.0
MAX_BARS = 40


class ExitReason:
    STOP = "STOP"
    TIME = "TIME"
    NONE = None


@dataclass
class Position:
    symbol: str
    direction: Direction
    entry_price: float
    initial_stop: float
    trail_pct: float = TRAIL_PCT
    target_r: float = TARGET_R
    max_bars: int = MAX_BARS

    stop: float = field(init=False)
    target: float = field(init=False)
    r_value: float = field(init=False)
    hwm: float = field(init=False)
    armed: bool = field(default=False, init=False)
    trail_live: bool = field(default=False, init=False)
    bars_held: int = field(default=0, init=False)

    def __post_init__(self):
        if self.direction not in ("LONG", "SHORT"):
            raise ValueError("direction must be LONG or SHORT")
        self.r_value = abs(self.entry_price - self.initial_stop)
        if self.r_value <= 0:
            raise ValueError("stop distance must be positive")
        self.stop = self.initial_stop
        if self.direction == "LONG":
            self.target = self.entry_price + self.target_r * self.r_value
        else:
            self.target = self.entry_price - self.target_r * self.r_value
        self.hwm = self.entry_price

    # -- helpers -------------------------------------------------------

    def _tighter(self, candidate: float) -> float:
        """Monotonic ratchet: a stop may only ever move toward the price."""
        if self.direction == "LONG":
            return max(self.stop, candidate)
        return min(self.stop, candidate)

    def _trail_level(self) -> float:
        if self.direction == "LONG":
            return self.hwm * (1.0 - self.trail_pct)
        return self.hwm * (1.0 + self.trail_pct)

    def _clamped_to_target(self, level: float) -> float:
        """Never let the trail fall back below the banked target."""
        if self.direction == "LONG":
            return max(level, self.target)
        return min(level, self.target)

    # -- main step -----------------------------------------------------

    def update(self, high: float, low: float, close: float) -> Optional[str]:
        """Advance one daily bar. Returns an exit reason, or None to stay open.

        Order of operations matters and is deliberate:
          1. Stop check uses the stop that was ALREADY in force at bar open.
          2. Then the trail (if live) tightens for the next bar.
          3. Then arming is evaluated.
        """
        self.bars_held += 1

        # 1. Stop is checked before any tightening this bar.
        if self.direction == "LONG":
            if low <= self.stop:
                return ExitReason.STOP
        else:
            if high >= self.stop:
                return ExitReason.STOP

        # 2. High-water mark and trail, only if the trail went live on a prior bar.
        if self.direction == "LONG":
            self.hwm = max(self.hwm, high)
        else:
            self.hwm = min(self.hwm, low)

        if self.trail_live:
            self.stop = self._tighter(self._clamped_to_target(self._trail_level()))

        # 3. Arming. The trail goes live from the NEXT bar.
        if not self.armed:
            hit = high >= self.target if self.direction == "LONG" else low <= self.target
            if hit:
                self.armed = True
        elif not self.trail_live:
            self.trail_live = True
            self.stop = self._tighter(self._clamped_to_target(self._trail_level()))

        if self.bars_held >= self.max_bars:
            return ExitReason.TIME

        return ExitReason.NONE

    # -- reporting -----------------------------------------------------

    def unrealised_r(self, price: float) -> float:
        if self.direction == "LONG":
            return (price - self.entry_price) / self.r_value
        return (self.entry_price - price) / self.r_value

    def status(self) -> str:
        if self.trail_live:
            return "TRAILING"
        if self.armed:
            return "ARMED (live next bar)"
        return "INITIAL STOP"


def atr(high, low, close, window: int = 14):
    """Wilder ATR. Accepts pandas Series."""
    import pandas as pd

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


__all__ = ["Position", "ExitReason", "atr", "TRAIL_PCT", "TARGET_R", "MAX_BARS"]
