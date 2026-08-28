"""
Centralised RiskGate.

The headline fix over the original spec: CAPITAL IS ALLOCATED PER ENGINE.

Previously Engine 1 wanted 10 positions at ~10% and Engine 2 allowed 6 at up to
20%, both drawing on the same ring fence -- up to 220% gross on undrawn capital.
Here the ring fence is hard-split at construction, the split is validated to sum
to no more than the fence, and each engine's gate only ever sees its own slice.
Percentages are of the ENGINE's allocation, never of the whole fence.

Second fix: position sizing records which constraint actually bound. With a
1.5*ATR stop on a low-volatility large cap, a 2%-risk budget implies a position
far above the 20% cap, so the cap binds and the risk budget silently stops
doing anything. If `binding` is POSITION_CAP on nearly every trade, your risk
model is the cap, not the 2%.

Third: live routing is gated on SEBI algo-trading preconditions as well as the
paper-trading proving gate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Optional

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class EngineConfig:
    name: str
    allocation: float             # rupees, hard slice of the ring fence
    max_positions: int
    max_position_pct: float       # of THIS engine's allocation
    max_drawdown_pct: float       # hard halt threshold
    risk_per_trade_pct: float = 0.02
    sector_cap: int = 3
    min_price: float = 50.0
    min_adt_cr: float = 5.0
    proving_min_days: int = 180
    proving_min_trades: int = 100


ROTATION = EngineConfig(
    name="ROTATION",
    allocation=300_000.0,
    max_positions=10,
    max_position_pct=0.20,
    max_drawdown_pct=0.18,
    proving_min_trades=40,        # ~monthly cadence cannot reach 100 in 180 days
)

SWING = EngineConfig(
    name="SWING",
    allocation=200_000.0,
    max_positions=6,
    max_position_pct=0.20,
    max_drawdown_pct=0.15,
    proving_min_trades=100,
)


@dataclass(frozen=True)
class ComplianceConfig:
    """SEBI retail algo framework preconditions for live order routing.

    Full compliance has been mandatory since 1 April 2026. Self-built strategies
    for personal use sit below the 10-orders-per-second registration threshold,
    but every order still needs its exchange-assigned identifier, and API access
    requires a whitelisted static IP with 2FA.

    None of this is legal advice -- confirm the current position with your broker
    before flipping live_enabled.
    """

    algo_strategy_id: Optional[str] = None
    static_ip_whitelisted: bool = False
    two_factor_enabled: bool = False
    observed_peak_orders_per_second: float = 0.0
    registration_threshold_ops: int = 10

    def blockers(self) -> list[str]:
        out = []
        if not self.algo_strategy_id:
            out.append("no exchange algo/strategy ID configured")
        if not self.static_ip_whitelisted:
            out.append("static IP not whitelisted with broker")
        if not self.two_factor_enabled:
            out.append("2FA not enabled on API session")
        if self.observed_peak_orders_per_second >= self.registration_threshold_ops:
            out.append(
                f"order rate {self.observed_peak_orders_per_second}/s meets the "
                f"{self.registration_threshold_ops}/s registration threshold"
            )
        return out


# ----------------------------------------------------------------------
# Decisions
# ----------------------------------------------------------------------


@dataclass
class Decision:
    allowed: bool
    reason: str = "ok"
    binding: str = ""

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class SizingResult:
    qty: int
    notional: float
    binding: str          # RISK | POSITION_CAP | CASH | NONE
    risk_amount: float


class HaltState:
    NONE = "NONE"
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    HARD = "HARD"


# ----------------------------------------------------------------------
# Per-engine book state
# ----------------------------------------------------------------------


@dataclass
class EngineState:
    equity: float
    peak_equity: float
    day_start_equity: float
    week_start_equity: float
    month_start_equity: float
    as_of: str
    halt: str = HaltState.NONE
    halt_until: Optional[str] = None
    halt_reason: str = ""
    paper_days: int = 0
    paper_trades: int = 0
    positions: dict = field(default_factory=dict)   # symbol -> {"notional":, "sector":}


class RiskGate:
    DAILY_LOSS_LIMIT = 0.04
    WEEKLY_LOSS_LIMIT = 0.07
    MONTHLY_LOSS_LIMIT = 0.12

    def __init__(
        self,
        ring_fence: float,
        engines: list[EngineConfig],
        compliance: ComplianceConfig | None = None,
        state_path: str = "risk_gate.json",
        live_enabled: bool = False,
    ):
        total = sum(e.allocation for e in engines)
        if total > ring_fence + 1e-6:
            raise ValueError(
                f"engine allocations {total:,.0f} exceed ring fence "
                f"{ring_fence:,.0f} -- define a split that fits before running"
            )
        names = [e.name for e in engines]
        if len(set(names)) != len(names):
            raise ValueError("duplicate engine names")

        self.ring_fence = ring_fence
        self.configs = {e.name: e for e in engines}
        self.compliance = compliance or ComplianceConfig()
        self.state_path = state_path
        self.live_enabled = live_enabled
        self.states: dict[str, EngineState] = {}
        for e in engines:
            self.states[e.name] = EngineState(
                equity=e.allocation,
                peak_equity=e.allocation,
                day_start_equity=e.allocation,
                week_start_equity=e.allocation,
                month_start_equity=e.allocation,
                as_of=date.today().isoformat(),
            )

    @property
    def unallocated(self) -> float:
        return self.ring_fence - sum(c.allocation for c in self.configs.values())

    # -- equity marking ------------------------------------------------

    def mark(self, engine: str, on: date, equity: float) -> Decision:
        """Mark an engine's equity for the day and evaluate circuit breakers."""
        cfg, st = self._get(engine)
        prev = date.fromisoformat(st.as_of)

        if on > prev:
            st.day_start_equity = st.equity
            if on.isocalendar()[1] != prev.isocalendar()[1] or on.year != prev.year:
                st.week_start_equity = st.equity
            if (on.year, on.month) != (prev.year, prev.month):
                st.month_start_equity = st.equity
            st.paper_days += (on - prev).days
            st.as_of = on.isoformat()

        st.equity = equity
        st.peak_equity = max(st.peak_equity, equity)

        # Release expired time-based halts before re-testing.
        if st.halt in (HaltState.DAY, HaltState.WEEK, HaltState.MONTH):
            if st.halt_until and on >= date.fromisoformat(st.halt_until):
                st.halt, st.halt_until, st.halt_reason = HaltState.NONE, None, ""

        dd = (st.peak_equity - equity) / st.peak_equity if st.peak_equity else 0.0
        if dd > cfg.max_drawdown_pct:
            return self._halt(st, HaltState.HARD, None,
                              f"max drawdown {dd:.1%} > {cfg.max_drawdown_pct:.0%}")

        if st.halt == HaltState.HARD:
            return Decision(False, st.halt_reason, HaltState.HARD)

        def drop(base):
            return (base - equity) / base if base else 0.0

        if drop(st.month_start_equity) > self.MONTHLY_LOSS_LIMIT:
            return self._halt(st, HaltState.MONTH, on + timedelta(days=30),
                              f"monthly loss {drop(st.month_start_equity):.1%}")
        if drop(st.week_start_equity) > self.WEEKLY_LOSS_LIMIT:
            return self._halt(st, HaltState.WEEK, on + timedelta(days=7),
                              f"weekly loss {drop(st.week_start_equity):.1%}")
        if drop(st.day_start_equity) > self.DAILY_LOSS_LIMIT:
            return self._halt(st, HaltState.DAY, on + timedelta(days=1),
                              f"daily loss {drop(st.day_start_equity):.1%}")

        return Decision(True, "ok")

    def _halt(self, st: EngineState, kind: str, until, reason: str) -> Decision:
        st.halt = kind
        st.halt_until = until.isoformat() if until else None
        st.halt_reason = reason
        return Decision(False, reason, kind)

    def reset_hard_halt(self, engine: str, confirm: str) -> None:
        """A hard halt requires a deliberate manual restart, by design."""
        if confirm != "CONFIRM":
            raise PermissionError("hard halt reset requires confirm='CONFIRM'")
        st = self.states[engine]
        st.halt, st.halt_until, st.halt_reason = HaltState.NONE, None, ""
        st.peak_equity = st.equity

    # -- sizing --------------------------------------------------------

    def size_position(
        self, engine: str, entry: float, stop: float, cash_available: float | None = None
    ) -> SizingResult:
        cfg, st = self._get(engine)
        stop_dist = abs(entry - stop)
        if stop_dist <= 0 or entry <= 0:
            return SizingResult(0, 0.0, "NONE", 0.0)

        risk_budget = st.equity * cfg.risk_per_trade_pct
        qty_risk = int(risk_budget // stop_dist)
        qty_cap = int((st.equity * cfg.max_position_pct) // entry)

        cash = st.equity if cash_available is None else cash_available
        qty_cash = int(cash // entry)

        qty = min(qty_risk, qty_cap, qty_cash)
        if qty == qty_cash and qty_cash < min(qty_risk, qty_cap):
            binding = "CASH"
        elif qty_cap <= qty_risk:
            binding = "POSITION_CAP"
        else:
            binding = "RISK"

        return SizingResult(
            qty=max(0, qty),
            notional=max(0, qty) * entry,
            binding=binding if qty > 0 else "NONE",
            risk_amount=max(0, qty) * stop_dist,
        )

    # -- the gate ------------------------------------------------------

    def check_order(
        self,
        engine: str,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        *,
        sector: str = "Unmapped",
        adt_cr: float | None = None,
        is_live: bool = False,
    ) -> Decision:
        cfg, st = self._get(engine)

        if is_live:
            gate = self.check_live_ready(engine)
            if not gate:
                return gate

        # Exits come BEFORE the halt check on purpose. A circuit breaker must
        # stop new risk, never trap you inside an open position.
        if side.upper() == "SELL" and symbol in st.positions:
            return Decision(True, "exit always permitted", "")

        if st.halt != HaltState.NONE:
            return Decision(False, f"{engine} halted: {st.halt_reason}", st.halt)

        if price < cfg.min_price:
            return Decision(False, f"price {price} below Rs {cfg.min_price:.0f} floor",
                            "MIN_PRICE")
        if adt_cr is not None and adt_cr < cfg.min_adt_cr:
            return Decision(False, f"ADT {adt_cr:.2f}cr below {cfg.min_adt_cr}cr floor",
                            "LIQUIDITY")

        if symbol not in st.positions and len(st.positions) >= cfg.max_positions:
            return Decision(False, f"{engine} at max {cfg.max_positions} positions",
                            "MAX_POSITIONS")

        notional = qty * price
        cap = st.equity * cfg.max_position_pct
        existing = st.positions.get(symbol, {}).get("notional", 0.0)
        if existing + notional > cap + 1e-6:
            return Decision(
                False,
                f"position {existing + notional:,.0f} exceeds "
                f"{cfg.max_position_pct:.0%} cap ({cap:,.0f})",
                "POSITION_CAP",
            )

        gross = sum(p["notional"] for p in st.positions.values()) + notional
        if gross > st.equity + 1e-6:
            return Decision(False,
                            f"gross exposure {gross:,.0f} exceeds engine allocation "
                            f"{st.equity:,.0f}", "GROSS_EXPOSURE")

        # "Unmapped" is not a sector -- capping it would create false positives.
        if sector and sector != "Unmapped":
            same = sum(
                1 for s, p in st.positions.items()
                if p.get("sector") == sector and s != symbol
            )
            if same >= cfg.sector_cap:
                return Decision(False, f"sector {sector} at cap {cfg.sector_cap}",
                                "SECTOR_CAP")

        return Decision(True, "ok")

    # -- live gating ---------------------------------------------------

    def check_live_ready(self, engine: str) -> Decision:
        cfg, st = self._get(engine)
        if not self.live_enabled:
            return Decision(False, "live routing disabled (paper mode)", "PAPER_MODE")

        blockers = self.compliance.blockers()
        if blockers:
            return Decision(False, "SEBI algo preconditions unmet: " +
                            "; ".join(blockers), "COMPLIANCE")

        if st.paper_days < cfg.proving_min_days:
            return Decision(False,
                            f"proving gate: {st.paper_days}/{cfg.proving_min_days} days",
                            "PROVING_GATE")
        if st.paper_trades < cfg.proving_min_trades:
            return Decision(False,
                            f"proving gate: {st.paper_trades}/"
                            f"{cfg.proving_min_trades} trades", "PROVING_GATE")
        return Decision(True, "ok")

    # -- book bookkeeping ----------------------------------------------

    def register_fill(self, engine: str, symbol: str, notional: float,
                      sector: str = "Unmapped") -> None:
        st = self.states[engine]
        pos = st.positions.setdefault(symbol, {"notional": 0.0, "sector": sector})
        pos["notional"] += notional
        pos["sector"] = sector
        if pos["notional"] <= 1e-6:
            st.positions.pop(symbol, None)
            st.paper_trades += 1

    def _get(self, engine: str):
        if engine not in self.configs:
            raise KeyError(f"unknown engine {engine!r}")
        return self.configs[engine], self.states[engine]

    # -- persistence ---------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "ring_fence": self.ring_fence,
            "unallocated": self.unallocated,
            "live_enabled": self.live_enabled,
            "compliance": asdict(self.compliance),
            "configs": {k: asdict(v) for k, v in self.configs.items()},
            "states": {k: asdict(v) for k, v in self.states.items()},
        }

    def save(self, path: str | None = None) -> str:
        path = path or self.state_path
        tmp, bak = path + ".tmp", path + ".bak"
        with open(tmp, "w") as fh:
            json.dump(self.snapshot(), fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        if os.path.exists(path):
            try:
                os.replace(path, bak)
            except OSError:
                pass
        os.replace(tmp, path)
        return path

    def load(self, path: str | None = None) -> None:
        path = path or self.state_path
        with open(path) as fh:
            blob = json.load(fh)
        for name, s in blob.get("states", {}).items():
            if name in self.states:
                self.states[name] = EngineState(**s)


__all__ = [
    "RiskGate", "EngineConfig", "ComplianceConfig", "Decision", "SizingResult",
    "EngineState", "HaltState", "ROTATION", "SWING",
]
