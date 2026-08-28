"""
Capital gains tax layer for listed Indian equity (STT paid).

This is the charge the original spec was missing. A monthly rotation realises
essentially everything as short-term gains, so tax is a larger drag than the
whole transaction cost stack combined. Backtest equity curves must be reported
net of this or the CAGR is not real.

Rates encoded (post 23 July 2024 regime):
    STCG (holding <= 12 months) : 20%
    LTCG (holding  > 12 months) : 12.5%, with an annual exemption on LTCG gains

Set-off rules implemented:
    Short-term capital LOSS  -> may offset STCG and LTCG (applied to STCG first,
                                since STCG is taxed higher)
    Long-term capital LOSS   -> may offset LTCG only
    Unabsorbed losses carry forward (8 years) retaining their character

This is a modelling tool for backtests, not tax advice. Confirm the rates and
your own set-off position with your CA before relying on the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

STCG_RATE = 0.20
LTCG_RATE = 0.125
LTCG_ANNUAL_EXEMPTION = 125_000.0
LONG_TERM_DAYS = 365
CARRY_FORWARD_YEARS = 8


def financial_year(d: date) -> int:
    """Indian FY starting year. 15 Jun 2026 -> 2026 (FY2026-27)."""
    return d.year if d.month >= 4 else d.year - 1


def is_long_term(entry: date, exit_: date) -> bool:
    return (exit_ - entry) > timedelta(days=LONG_TERM_DAYS)


@dataclass
class RealisedTrade:
    symbol: str
    entry_date: date
    exit_date: date
    pnl: float  # net of transaction costs

    @property
    def fy(self) -> int:
        return financial_year(self.exit_date)

    @property
    def long_term(self) -> bool:
        return is_long_term(self.entry_date, self.exit_date)


@dataclass
class FYTaxResult:
    fy: int
    stcg_gross: float
    ltcg_gross: float
    stcl: float           # positive magnitude of short-term losses
    ltcl: float           # positive magnitude of long-term losses
    stcg_taxable: float
    ltcg_taxable: float   # after exemption
    ltcg_exemption_used: float
    tax: float
    carried_forward_stcl: float
    carried_forward_ltcl: float

    @property
    def effective_rate(self) -> float:
        net = self.stcg_gross + self.ltcg_gross - self.stcl - self.ltcl
        return self.tax / net if net > 0 else 0.0


class TaxLedger:
    """Accumulates realised trades and computes tax per financial year."""

    def __init__(
        self,
        stcg_rate: float = STCG_RATE,
        ltcg_rate: float = LTCG_RATE,
        ltcg_exemption: float = LTCG_ANNUAL_EXEMPTION,
    ):
        self.stcg_rate = stcg_rate
        self.ltcg_rate = ltcg_rate
        self.ltcg_exemption = ltcg_exemption
        self.trades: list[RealisedTrade] = []

    def add(self, trade: RealisedTrade) -> None:
        self.trades.append(trade)

    def add_many(self, trades) -> None:
        self.trades.extend(trades)

    def compute(self) -> list[FYTaxResult]:
        """Walk financial years in order, carrying losses forward."""
        by_fy: dict[int, list[RealisedTrade]] = {}
        for t in self.trades:
            by_fy.setdefault(t.fy, []).append(t)

        cf_stcl = 0.0
        cf_ltcl = 0.0
        results: list[FYTaxResult] = []

        for fy in sorted(by_fy):
            ts = by_fy[fy]
            stcg = sum(t.pnl for t in ts if not t.long_term and t.pnl > 0)
            ltcg = sum(t.pnl for t in ts if t.long_term and t.pnl > 0)
            stcl = -sum(t.pnl for t in ts if not t.long_term and t.pnl < 0) + cf_stcl
            ltcl = -sum(t.pnl for t in ts if t.long_term and t.pnl < 0) + cf_ltcl

            # Short-term losses first against STCG (higher rate), then LTCG.
            use = min(stcl, stcg)
            stcg_net = stcg - use
            stcl_left = stcl - use

            use2 = min(stcl_left, ltcg)
            ltcg_net = ltcg - use2
            stcl_left -= use2

            # Long-term losses only against LTCG.
            use3 = min(ltcl, ltcg_net)
            ltcg_net -= use3
            ltcl_left = ltcl - use3

            exemption_used = min(self.ltcg_exemption, ltcg_net)
            ltcg_taxable = ltcg_net - exemption_used

            tax = stcg_net * self.stcg_rate + ltcg_taxable * self.ltcg_rate

            results.append(
                FYTaxResult(
                    fy=fy,
                    stcg_gross=stcg,
                    ltcg_gross=ltcg,
                    stcl=stcl,
                    ltcl=ltcl,
                    stcg_taxable=stcg_net,
                    ltcg_taxable=ltcg_taxable,
                    ltcg_exemption_used=exemption_used,
                    tax=tax,
                    carried_forward_stcl=stcl_left,
                    carried_forward_ltcl=ltcl_left,
                )
            )
            cf_stcl, cf_ltcl = stcl_left, ltcl_left

        return results

    def total_tax(self) -> float:
        return sum(r.tax for r in self.compute())

    def net_of_tax_pnl(self) -> float:
        gross = sum(t.pnl for t in self.trades)
        return gross - self.total_tax()


__all__ = [
    "RealisedTrade",
    "FYTaxResult",
    "TaxLedger",
    "financial_year",
    "is_long_term",
    "STCG_RATE",
    "LTCG_RATE",
    "LTCG_ANNUAL_EXEMPTION",
]
