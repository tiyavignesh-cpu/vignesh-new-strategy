"""
Two-target exit manager: 50% out at TP1, breakeven + trail on the remainder,
final exit at TP2.

Two conventions worth being explicit about, because they are where backtest
optimism usually enters:

* PESSIMISTIC INTRABAR. When a bar's range spans both the stop and a target, we
  cannot know which came first from daily data. We assume the stop. This makes
  the backtest slightly worse than reality on average, which is the correct
  direction to be wrong in.

* ARM-NEXT-BAR. The breakeven shift and the trail activate on the bar AFTER the
  one that touched TP1, matching the discipline already used in
  trailing_stop.py. Moving the stop on the touching bar assumes you saw the
  touch before the bar's low printed, which you did not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple

Direction = Literal["LONG", "SHORT"]


class Event(NamedTuple):
    kind: str          # SCALE_OUT | EXIT
    price: float
    qty: int
    reason: str        # TP1 | TP2 | STOP | TIME


@dataclass
class ScaledPosition:
    symbol: str
    direction: Direction
    entry_price: float
    initial_stop: float
    tp1: float
    tp2: float
    qty: int

    scale_out_pct: float = 0.5
    breakeven_after_tp1: bool = True
    trail_after_tp1: bool = True
    trail_pct: float = 0.01
    max_bars: int = 40
    pessimistic_intrabar: bool = True

    stop: float = field(init=False)
    r_value: float = field(init=False)
    remaining: int = field(init=False)
    hwm: float = field(init=False)
    tp1_hit: bool = field(default=False, init=False)
    trail_live: bool = field(default=False, init=False)
    bars_held: int = field(default=0, init=False)
    closed: bool = field(default=False, init=False)
    realised: list = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.direction not in ("LONG", "SHORT"):
            raise ValueError("direction must be LONG or SHORT")
        self.r_value = abs(self.entry_price - self.initial_stop)
        if self.r_value <= 0:
            raise ValueError("stop distance must be positive")
        if self.direction == "LONG" and not (self.initial_stop < self.entry_price <= self.tp1 <= self.tp2):
            raise ValueError("long requires stop < entry <= tp1 <= tp2")
        if self.direction == "SHORT" and not (self.initial_stop > self.entry_price >= self.tp1 >= self.tp2):
            raise ValueError("short requires stop > entry >= tp1 >= tp2")
        self.stop = self.initial_stop
        self.remaining = self.qty
        self.hwm = self.entry_price

    # -- direction helpers ---------------------------------------------

    def _tighter(self, level: float) -> float:
        return max(self.stop, level) if self.direction == "LONG" else min(self.stop, level)

    def _reached(self, level: float, high: float, low: float) -> bool:
        return high >= level if self.direction == "LONG" else low <= level

    def _stopped(self, high: float, low: float) -> bool:
        return low <= self.stop if self.direction == "LONG" else high >= self.stop

    def _trail_level(self) -> float:
        f = 1.0 - self.trail_pct if self.direction == "LONG" else 1.0 + self.trail_pct
        return self.hwm * f

    def _floor(self, level: float) -> float:
        """Never let the trail give back the breakeven shift."""
        if not self.breakeven_after_tp1:
            return level
        return max(level, self.entry_price) if self.direction == "LONG" else min(level, self.entry_price)

    # -- main step -----------------------------------------------------

    def update(self, high: float, low: float, close: float) -> list[Event]:
        if self.closed:
            return []

        self.bars_held += 1
        events: list[Event] = []

        # Pending arm from the previous bar's TP1 touch.
        if self.tp1_hit and not self.trail_live:
            self.trail_live = True
            if self.breakeven_after_tp1:
                self.stop = self._tighter(self.entry_price)

        # 1. Stop, checked against the level in force at bar open.
        if self._stopped(high, low):
            return self._close_out(self.stop, "STOP", events)

        # 2. High-water mark, then trail if it is live.
        self.hwm = max(self.hwm, high) if self.direction == "LONG" else min(self.hwm, low)
        if self.trail_live and self.trail_after_tp1:
            self.stop = self._tighter(self._floor(self._trail_level()))

        # 3. TP1 scale-out. Arms breakeven/trail for the NEXT bar.
        if not self.tp1_hit and self._reached(self.tp1, high, low):
            out = int(round(self.remaining * self.scale_out_pct))
            out = max(1, min(out, self.remaining - 1)) if self.remaining > 1 else self.remaining
            self.remaining -= out
            self.tp1_hit = True
            ev = Event("SCALE_OUT", self.tp1, out, "TP1")
            events.append(ev)
            self.realised.append(ev)
            if self.remaining == 0:
                self.closed = True
                return events

        # 4. TP2 on the remainder.
        if self.tp1_hit and self._reached(self.tp2, high, low):
            return self._close_out(self.tp2, "TP2", events)

        # 5. Time stop.
        if self.bars_held >= self.max_bars:
            return self._close_out(close, "TIME", events)

        return events

    def _close_out(self, price: float, reason: str, events: list[Event]) -> list[Event]:
        if self.remaining > 0:
            ev = Event("EXIT", price, self.remaining, reason)
            events.append(ev)
            self.realised.append(ev)
            self.remaining = 0
        self.closed = True
        return events

    # -- reporting -----------------------------------------------------

    def realised_r(self) -> float:
        """Net R-multiple across all fills, weighted by quantity."""
        if not self.realised:
            return 0.0
        sign = 1.0 if self.direction == "LONG" else -1.0
        pnl = sum(sign * (e.price - self.entry_price) * e.qty for e in self.realised)
        return pnl / (self.r_value * self.qty)

    def status(self) -> str:
        if self.closed:
            return "CLOSED"
        if self.trail_live:
            return f"TRAILING ({self.remaining}/{self.qty} open)"
        if self.tp1_hit:
            return f"TP1 HIT, breakeven live next bar ({self.remaining}/{self.qty})"
        return "INITIAL STOP"


__all__ = ["ScaledPosition", "Event"]
