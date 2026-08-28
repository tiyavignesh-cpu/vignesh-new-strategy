"""
Itemised Indian (NSE) transaction cost model.

Every charge is a named component. Nothing is a single hardcoded percentage.
Slippage is modelled separately from statutory charges so you can stress it
independently in walk-forward runs.

Segments supported:
    DELIVERY  -> CNC equity delivery
    INTRADAY  -> MIS equity intraday

Note: capital gains tax is NOT here. It is a portfolio-level, financial-year
level charge and lives in taxes.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class CostConfig:
    """All rates are fractions of turnover unless the name says otherwise."""

    # Brokerage: charged as min(flat_cap, pct * turnover) when pct > 0,
    # otherwise flat_per_order (which may be 0 for delivery on many brokers).
    brokerage_pct: float = 0.0
    brokerage_flat_cap: float = 20.0
    brokerage_flat_per_order: float = 0.0

    stt_buy_pct: float = 0.0010          # 0.10% delivery buy
    stt_sell_pct: float = 0.0010         # 0.10% delivery sell
    exchange_txn_pct: float = 0.0000297  # 0.00297% each side (NSE equity)
    sebi_pct: float = 0.000001           # Rs 10 per crore
    stamp_duty_buy_pct: float = 0.00015  # 0.015% buy only (delivery)
    gst_pct: float = 0.18                # on brokerage + exchange txn + SEBI
    dp_charge_per_scrip_sell: float = 15.93  # CDSL, per scrip per sell day
    gst_on_dp: bool = False              # set True if your broker adds GST on DP

    slippage_pct: float = 0.0005         # 0.05% per leg, modelled separately

    label: str = "DELIVERY"


DELIVERY = CostConfig()

INTRADAY = CostConfig(
    stt_buy_pct=0.0,
    stt_sell_pct=0.00025,      # 0.025% sell side only
    stamp_duty_buy_pct=0.00003,  # 0.003% buy only
    dp_charge_per_scrip_sell=0.0,
    label="INTRADAY",
)


def _brokerage(turnover: float, cfg: CostConfig) -> float:
    if cfg.brokerage_pct > 0:
        return min(cfg.brokerage_flat_cap, cfg.brokerage_pct * turnover)
    return cfg.brokerage_flat_per_order


def leg_costs(
    side: Side,
    price: float,
    qty: int,
    cfg: CostConfig = DELIVERY,
    *,
    is_last_sell_of_day_for_scrip: bool = True,
) -> dict:
    """Itemised statutory + broker charges for one leg. Excludes slippage.

    `is_last_sell_of_day_for_scrip` exists because DP is charged once per
    scrip per sell day, not once per sell order. Set False for the 2nd..Nth
    partial sell of the same scrip on the same day.
    """
    if side not in ("BUY", "SELL"):
        raise ValueError(f"side must be BUY or SELL, got {side!r}")
    if qty < 0 or price < 0:
        raise ValueError("price and qty must be non-negative")

    turnover = price * qty

    brokerage = _brokerage(turnover, cfg)
    stt = turnover * (cfg.stt_buy_pct if side == "BUY" else cfg.stt_sell_pct)
    exch = turnover * cfg.exchange_txn_pct
    sebi = turnover * cfg.sebi_pct
    stamp = turnover * cfg.stamp_duty_buy_pct if side == "BUY" else 0.0

    gst_base = brokerage + exch + sebi
    dp = 0.0
    if side == "SELL" and is_last_sell_of_day_for_scrip:
        dp = cfg.dp_charge_per_scrip_sell
        if cfg.gst_on_dp:
            gst_base += dp

    gst = cfg.gst_pct * gst_base

    total = brokerage + stt + exch + sebi + stamp + gst + dp

    return {
        "side": side,
        "turnover": turnover,
        "brokerage": brokerage,
        "stt": stt,
        "exchange_txn": exch,
        "sebi": sebi,
        "stamp_duty": stamp,
        "gst": gst,
        "dp_charge": dp,
        "total": total,
        "total_pct_of_turnover": (total / turnover) if turnover else 0.0,
    }


def slippage_cost(price: float, qty: int, cfg: CostConfig = DELIVERY) -> float:
    """Modelled execution shortfall for one leg."""
    return price * qty * cfg.slippage_pct


def round_trip(
    buy_price: float,
    sell_price: float,
    qty: int,
    cfg: CostConfig = DELIVERY,
    *,
    include_slippage: bool = True,
) -> dict:
    """Full round-trip breakdown plus net P&L before capital gains tax."""
    buy = leg_costs("BUY", buy_price, qty, cfg)
    sell = leg_costs("SELL", sell_price, qty, cfg)

    slip = 0.0
    if include_slippage:
        slip = slippage_cost(buy_price, qty, cfg) + slippage_cost(sell_price, qty, cfg)

    charges = buy["total"] + sell["total"]
    gross_pnl = (sell_price - buy_price) * qty
    net_pnl = gross_pnl - charges - slip

    notional = buy_price * qty
    return {
        "buy": buy,
        "sell": sell,
        "statutory_and_broker": charges,
        "slippage": slip,
        "total_cost": charges + slip,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "cost_pct_of_notional": (charges + slip) / notional if notional else 0.0,
        "statutory_pct_of_notional": charges / notional if notional else 0.0,
    }


def baseline_drag_pct(cfg: CostConfig = DELIVERY, notional: float = 500_000.0) -> float:
    """Round-trip cost as a fraction of notional at a flat price.

    Useful as a single headline number for sanity checks, but always derive it
    from the components rather than hardcoding it -- DP charges are a fixed
    rupee amount, so the drag is size-dependent.
    """
    price = 1000.0
    qty = max(1, int(notional // price))
    return round_trip(price, price, qty, cfg)["cost_pct_of_notional"]


def apply_costs_to_fill(price: float, side: Side, cfg: CostConfig = DELIVERY) -> float:
    """Slippage-adjusted fill price: buys fill worse (higher), sells worse (lower)."""
    if side == "BUY":
        return price * (1.0 + cfg.slippage_pct)
    return price * (1.0 - cfg.slippage_pct)


__all__ = [
    "CostConfig",
    "DELIVERY",
    "INTRADAY",
    "leg_costs",
    "slippage_cost",
    "round_trip",
    "baseline_drag_pct",
    "apply_costs_to_fill",
    "replace",
    "field",
]
