"""
costs/nse_friction.py — Explicit, Itemised, Intraday-Correct NSE Equity Friction Engine.

Implements statutory charges for MIS/intraday equity plus adaptive liquidity-impact slippage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Dict, Any

Side = Literal["BUY", "SELL"]

# Statutory Intraday Equity Rates (MIS / Square-off same day)
STT_SELL_INTRADAY = 0.00025    # 0.025% on SELL side only (0.0% on BUY)
EXCHANGE_TXN_NSE = 0.0000297  # 0.00297% both sides
SEBI_CHARGES = 0.000001       # 0.0001% (Rs 10 / crore) both sides
STAMP_DUTY_BUY = 0.00003      # 0.003% BUY side only (0.0% on SELL)
GST_RATE = 0.18               # 18% on (Brokerage + Exchange Txn + SEBI)
BROKERAGE_FLAT_CAP = 20.0     # Max Rs 20 per executed order
BROKERAGE_PCT = 0.0003        # 0.03% whichever is lower
IPFT_RATE = 0.000001          # Investor Protection Fund Trust charge (0.0001%)


def calculate_brokerage(turnover: float) -> float:
    """Discount broker standard: min(Rs 20, 0.03% of turnover)."""
    return min(BROKERAGE_FLAT_CAP, BROKERAGE_PCT * turnover)


def calculate_adaptive_slippage_bps(
    order_value: float,
    slot_turnover_20d_median: float = 5_000_000.0,
    spread_bps: float = 2.0,
) -> float:
    """
    Adaptive Slippage Model (Amendment A10):
      impact_bps = 10.0 * sqrt(order_value / slot_turnover_20d_median)
      slippage_bps = min(25.0, max(1.5, 0.5 * spread_bps + impact_bps))
    """
    if slot_turnover_20d_median <= 0:
        slot_turnover_20d_median = 5_000_000.0 # fallback Rs 50L default 2-min slot volume
    
    impact_bps = 10.0 * math.sqrt(max(0.0, order_value) / slot_turnover_20d_median)
    slippage_bps = min(25.0, max(1.5, 0.5 * spread_bps + impact_bps))
    return slippage_bps


def compute_leg_friction(
    side: Side,
    price: float,
    qty: int,
    slot_turnover_20d_median: float = 5_000_000.0,
    spread_bps: float = 2.0,
) -> Dict[str, float]:
    """Itemised calculation for a single execution leg."""
    if qty <= 0 or price <= 0:
        return {
            "turnover": 0.0, "brokerage": 0.0, "stt": 0.0, "exchange_txn": 0.0,
            "sebi": 0.0, "stamp_duty": 0.0, "ipft": 0.0, "gst": 0.0,
            "statutory_and_broker": 0.0, "slippage_bps": 0.0, "slippage_inr": 0.0, "total": 0.0
        }

    turnover = price * qty
    brokerage = calculate_brokerage(turnover)
    stt = turnover * STT_SELL_INTRADAY if side == "SELL" else 0.0
    exchange_txn = turnover * EXCHANGE_TXN_NSE
    sebi = turnover * SEBI_CHARGES
    stamp_duty = turnover * STAMP_DUTY_BUY if side == "BUY" else 0.0
    ipft = turnover * IPFT_RATE

    gst_base = brokerage + exchange_txn + sebi
    gst = GST_RATE * gst_base

    statutory_and_broker = brokerage + stt + exchange_txn + sebi + stamp_duty + ipft + gst

    slip_bps = calculate_adaptive_slippage_bps(turnover, slot_turnover_20d_median, spread_bps)
    slippage_inr = turnover * (slip_bps / 10000.0)

    total = statutory_and_broker + slippage_inr

    return {
        "turnover": turnover,
        "brokerage": brokerage,
        "stt": stt,
        "exchange_txn": exchange_txn,
        "sebi": sebi,
        "stamp_duty": stamp_duty,
        "ipft": ipft,
        "gst": gst,
        "statutory_and_broker": statutory_and_broker,
        "slippage_bps": slip_bps,
        "slippage_inr": slippage_inr,
        "total": total,
    }


def compute_round_trip_friction(
    entry_price: float,
    exit_price: float,
    qty: int,
    side: Side = "BUY",
    slot_turnover_20d_median: float = 5_000_000.0,
    spread_bps: float = 2.0,
) -> Dict[str, Any]:
    """Itemised full round-trip friction for an intraday trade."""
    open_side: Side = "BUY" if side == "BUY" else "SELL"
    close_side: Side = "SELL" if side == "BUY" else "BUY"

    leg1 = compute_leg_friction(open_side, entry_price, qty, slot_turnover_20d_median, spread_bps)
    leg2 = compute_leg_friction(close_side, exit_price, qty, slot_turnover_20d_median, spread_bps)

    gross_pnl = (exit_price - entry_price) * qty if side == "BUY" else (entry_price - exit_price) * qty

    total_statutory = leg1["statutory_and_broker"] + leg2["statutory_and_broker"]
    total_slippage = leg1["slippage_inr"] + leg2["slippage_inr"]
    total_friction = total_statutory + total_slippage

    net_pnl = gross_pnl - total_friction
    round_trip_turnover = leg1["turnover"] + leg2["turnover"]
    cost_pct_of_turnover = (total_friction / round_trip_turnover) * 100.0 if round_trip_turnover > 0 else 0.0

    return {
        "side": side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "qty": qty,
        "gross_pnl": gross_pnl,
        "brokerage": leg1["brokerage"] + leg2["brokerage"],
        "stt": leg1["stt"] + leg2["stt"],
        "exchange_txn": leg1["exchange_txn"] + leg2["exchange_txn"],
        "sebi": leg1["sebi"] + leg2["sebi"],
        "stamp_duty": leg1["stamp_duty"] + leg2["stamp_duty"],
        "ipft": leg1["ipft"] + leg2["ipft"],
        "gst": leg1["gst"] + leg2["gst"],
        "total_statutory": total_statutory,
        "total_slippage": total_slippage,
        "total_friction": total_friction,
        "net_pnl": net_pnl,
        "cost_pct_of_turnover": cost_pct_of_turnover,
        "avg_slippage_bps": (leg1["slippage_bps"] + leg2["slippage_bps"]) / 2.0,
    }


def estimate_round_trip_cost_pct(
    entry_price: float,
    qty: int,
    slot_turnover_20d_median: float = 5_000_000.0,
    spread_bps: float = 2.0,
) -> float:
    """Returns estimated round-trip cost as a percentage of entry notional."""
    notional = entry_price * qty
    if notional <= 0:
        return 0.0030
    res = compute_round_trip_friction(entry_price, entry_price, qty, "BUY", slot_turnover_20d_median, spread_bps)
    return res["total_friction"] / notional


__all__ = [
    "STT_SELL_INTRADAY",
    "EXCHANGE_TXN_NSE",
    "SEBI_CHARGES",
    "STAMP_DUTY_BUY",
    "GST_RATE",
    "calculate_brokerage",
    "calculate_adaptive_slippage_bps",
    "compute_leg_friction",
    "compute_round_trip_friction",
    "estimate_round_trip_cost_pct",
]
