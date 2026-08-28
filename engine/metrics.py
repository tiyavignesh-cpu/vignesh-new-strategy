"""
Performance metrics, with the invariants asserted rather than assumed.

The v3.1 report contained metrics that cannot coexist:

  * |Sortino| < |Sharpe| in all three columns. Downside deviation is bounded
    above by RMS deviation about the same target, so for a common numerator
    |Sortino| >= |Sharpe| ALWAYS. The usual cause is dividing the sum of
    squared downside deviations by the COUNT OF LOSING PERIODS instead of the
    total period count. That inflates downside deviation and flips the ratio.
    `downside_deviation()` below uses the full-sample denominator and
    `check_invariants()` fails loudly if the relation ever breaks.

  * A full-period profit factor equal to the in-sample figure while the
    out-of-sample figure was wildly different. Pooled PF = (W1+W2)/(L1+L2) is
    the mediant of the fold PFs and must lie BETWEEN them. If your "overall"
    number sits outside that range you are reporting one fold and labelling it
    the whole period. `check_fold_consistency()` catches it.

  * A Sharpe implying ~0.9% annualised volatility alongside a 5.5% drawdown.
    `sanity_notes()` flags returns/volatility/drawdown combinations that are
    internally implausible.

None of this makes a strategy better. It makes the number you act on true.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------- core

def to_returns(equity: pd.Series) -> pd.Series:
    """Simple period returns from an equity curve."""
    eq = pd.Series(equity).astype(float).dropna()
    if (eq <= 0).any():
        raise ValueError("equity curve touches zero or goes negative")
    return eq.pct_change().dropna()


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    eq = pd.Series(equity).astype(float).dropna()
    n = len(eq) - 1
    if n <= 0 or eq.iloc[0] <= 0:
        return 0.0
    years = n / periods_per_year
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)


def volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of returns. ddof=1."""
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def rms_deviation(returns: pd.Series, target: float,
                  periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised RMS deviation about a target. The upper bound on downside dev."""
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return 0.0
    return float(np.sqrt(np.mean((r - target) ** 2)) * np.sqrt(periods_per_year))


def downside_deviation(returns: pd.Series, target: float = 0.0,
                       periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised downside deviation.

    CRITICAL: the denominator is the TOTAL number of periods, not the number of
    losing periods. Using the losing count is the single most common Sortino
    bug and it produces |Sortino| < |Sharpe|, which is impossible.
    """
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return 0.0
    shortfall = np.minimum(r - target, 0.0)
    return float(np.sqrt(np.sum(shortfall ** 2) / len(r)) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, rf: float = 0.06,
           periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0
    rf_period = (1 + rf) ** (1 / periods_per_year) - 1
    excess = r - rf_period
    vol = r.std(ddof=1)
    if vol <= 1e-8:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, rf: float = 0.06,
            periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return 0.0
    rf_period = (1 + rf) ** (1 / periods_per_year) - 1
    dd = downside_deviation(r, rf_period, periods_per_year)
    if dd <= 1e-8:
        return 0.0
    excess_annual = (r - rf_period).mean() * periods_per_year
    return float(excess_annual / dd)


def max_drawdown(equity: pd.Series) -> float:
    eq = pd.Series(equity).astype(float).dropna()
    if eq.empty:
        return 0.0
    return float((eq / eq.cummax() - 1.0).min())


def calmar(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return cagr(equity, periods_per_year) / mdd


def profit_factor(pnls) -> float:
    p = np.asarray(list(pnls), dtype=float)
    wins = p[p > 0].sum()
    losses = -p[p < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def win_rate(pnls) -> float:
    p = np.asarray(list(pnls), dtype=float)
    return float((p > 0).mean()) if len(p) else 0.0


# ------------------------------------------------------------- report

@dataclass
class PerformanceReport:
    label: str
    start_equity: float
    end_equity: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    trades: int
    win_rate: float
    profit_factor: float
    cash_yield_credited: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    def check_invariants(self) -> None:
        """Fail loudly on mathematically impossible combinations."""
        if self.volatility < 0 or self.max_drawdown > 0:
            raise AssertionError("volatility must be >= 0 and max_drawdown <= 0")

        # |Sortino| >= |Sharpe| for a common numerator. Tolerance covers the
        # ddof=1 vs population difference in the two denominators.
        if abs(self.sortino) + 1e-6 < abs(self.sharpe) * 0.98:
            raise AssertionError(
                f"{self.label}: |Sortino| ({self.sortino:.2f}) < |Sharpe| "
                f"({self.sharpe:.2f}) -- downside deviation is almost certainly "
                f"being divided by the count of losing periods"
            )

    def sanity_notes(self) -> list[str]:
        """Non-fatal warnings about implausible metric combinations."""
        notes = []
        if self.volatility > 0 and abs(self.max_drawdown) > 4 * self.volatility:
            notes.append(
                f"max drawdown {self.max_drawdown:.1%} is more than 4x annual "
                f"volatility {self.volatility:.1%} -- check the equity curve"
            )
        if self.trades and self.trades < 30:
            notes.append(
                f"only {self.trades} trades -- Sharpe and profit factor have "
                f"standard errors wide enough to cover almost any conclusion"
            )
        if self.volatility < 0.02 and abs(self.max_drawdown) > 0.03:
            notes.append(
                f"volatility {self.volatility:.1%} is implausibly low next to a "
                f"{self.max_drawdown:.1%} drawdown -- the equity series is "
                f"probably not being marked daily"
            )
        return notes


def build_report(
    label: str,
    equity: pd.Series,
    trade_pnls=None,
    rf: float = 0.06,
    periods_per_year: int = TRADING_DAYS,
    cash_yield_credited: float = 0.0,
) -> PerformanceReport:
    eq = pd.Series(equity).astype(float).dropna()
    r = to_returns(eq)
    pnls = list(trade_pnls) if trade_pnls is not None else []

    rep = PerformanceReport(
        label=label,
        start_equity=float(eq.iloc[0]),
        end_equity=float(eq.iloc[-1]),
        cagr=cagr(eq, periods_per_year),
        volatility=volatility(r, periods_per_year),
        sharpe=sharpe(r, rf, periods_per_year),
        sortino=sortino(r, rf, periods_per_year),
        max_drawdown=max_drawdown(eq),
        calmar=calmar(eq, periods_per_year),
        trades=len(pnls),
        win_rate=win_rate(pnls),
        profit_factor=profit_factor(pnls),
        cash_yield_credited=cash_yield_credited,
    )
    rep.check_invariants()
    return rep


def check_fold_consistency(
    pooled_pf: float, fold_pfs: list[float],
    pooled_trades: int, fold_trades: list[int],
) -> None:
    """Necessary condition: pooled PF is the mediant of the fold PFs.

    (W1+W2)/(L1+L2) always sits between W1/L1 and W2/L2, so a pooled figure
    outside that range is definitely wrong. Note this is NECESSARY, NOT
    SUFFICIENT -- the v3.1 Engine 1 numbers (pooled 1.63 from folds 1.64 and
    0.22) pass this and are still wrong. Use `check_fold_exact` when you have
    the gross figures, and `flag_suspicious_pooling` when you don't.
    """
    if sum(fold_trades) != pooled_trades:
        raise AssertionError(
            f"fold trade counts {fold_trades} sum to {sum(fold_trades)}, "
            f"pooled reports {pooled_trades}"
        )
    finite = [p for p in fold_pfs if np.isfinite(p)]
    if not finite:
        return
    lo, hi = min(finite), max(finite)
    if not (lo - 1e-6 <= pooled_pf <= hi + 1e-6):
        raise AssertionError(
            f"pooled profit factor {pooled_pf:.3f} lies outside the fold range "
            f"[{lo:.3f}, {hi:.3f}] -- a fold is being mislabelled as the "
            f"full period"
        )


def check_fold_exact(pooled_pf: float, folds: list[tuple[float, float]]) -> None:
    """Exact check. `folds` is [(gross_wins, gross_losses), ...] per fold.

    This is the one to use. Report gross wins and losses per fold from the
    blotter and the pooled figure is fully determined -- no room for a fold to
    masquerade as the full period.
    """
    w = sum(f[0] for f in folds)
    l = sum(f[1] for f in folds)
    if l <= 0:
        return
    expected = w / l
    if abs(pooled_pf - expected) > 1e-3 * max(1.0, expected):
        raise AssertionError(
            f"pooled profit factor {pooled_pf:.3f} does not match the folds "
            f"({w:.0f} wins / {l:.0f} losses = {expected:.3f})"
        )


def flag_suspicious_pooling(
    pooled_pf: float, fold_pfs: list[float], fold_trades: list[int],
    tol: float = 0.05, divergence: float = 2.0,
) -> list[str]:
    """Warn when a pooled figure hugs one fold while another fold diverges.

    The v3.1 signature exactly: Engine 1 reported 1.63 overall against folds of
    1.64 and 0.22, on similar trade counts. With comparable trade counts the
    pooled value should be dragged well toward the weaker fold. Sitting within
    5% of the stronger one says the "overall" column is the in-sample fold
    wearing a different label.
    """
    notes = []
    finite = [(p, n) for p, n in zip(fold_pfs, fold_trades) if np.isfinite(p)]
    if len(finite) < 2:
        return notes
    best = max(finite, key=lambda x: x[0])
    worst = min(finite, key=lambda x: x[0])
    if worst[0] <= 0:
        return notes
    if best[0] / worst[0] < divergence:
        return notes

    share = min(fold_trades) / max(1, sum(fold_trades))
    for label, (pf, _) in (("best", best), ("worst", worst)):
        if abs(pooled_pf - pf) <= tol * max(pf, 1e-9) and share > 0.25:
            notes.append(
                f"pooled profit factor {pooled_pf:.2f} is within {tol:.0%} of the "
                f"{label} fold ({pf:.2f}) while the other fold is "
                f"{(worst if label == 'best' else best)[0]:.2f} on a comparable "
                f"trade count -- the 'overall' column is probably one fold "
                f"mislabelled as the full period"
            )
    return notes


def credit_cash_yield(
    equity: pd.Series, invested_fraction: pd.Series,
    annual_yield: float = 0.065, periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Credit idle cash at a liquid-fund rate.

    A rotation with a macro circuit breaker sits in cash for long stretches.
    Modelling that cash at 0% is not conservative, it is wrong -- the real
    alternative earns something. This RAISES backtested returns, so it is not
    a tuning knob: it corrects an omission that was biasing results downward.
    """
    eq = pd.Series(equity).astype(float)
    inv = pd.Series(invested_fraction).reindex(eq.index).fillna(0.0).clip(0, 1)
    daily = (1 + annual_yield) ** (1 / periods_per_year) - 1
    idle_return = (1 - inv) * daily

    base = to_returns(eq)
    combined = (base + idle_return.reindex(base.index).fillna(0.0))
    out = (1 + combined).cumprod() * float(eq.iloc[0])
    return pd.concat([pd.Series([float(eq.iloc[0])], index=[eq.index[0]]), out])


__all__ = [
    "to_returns", "cagr", "volatility", "downside_deviation", "rms_deviation",
    "sharpe", "sortino", "max_drawdown", "calmar", "profit_factor", "win_rate",
    "PerformanceReport", "build_report", "check_fold_consistency",
    "credit_cash_yield", "TRADING_DAYS",
]
