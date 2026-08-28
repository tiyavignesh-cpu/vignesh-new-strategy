"""
Probability of Backtest Overfitting (PBO), via combinatorially symmetric
cross-validation (Bailey, Borwein, Lopez de Prado, Zhu).

This was specified in Phase 4 and silently dropped from the v3.1 run. It is the
one number that most directly addressed what the walk-forward showed: Engine 1's
profit factor collapsing from 1.64 in-sample to 0.22 out-of-sample is the
textbook overfitting signature, and PBO quantifies how likely that collapse was
to happen by construction.

Method: take a matrix of period returns, one column per parameter configuration
tried. Split the rows into S equal blocks. For every way of choosing S/2 blocks
as in-sample (the complement is out-of-sample), find the configuration that
ranked best in-sample and record where it ranked out-of-sample. PBO is the
fraction of splits where the in-sample winner landed in the bottom half
out-of-sample.

Reading it:
    PBO < 0.2   the selection process carries real information
    0.2 - 0.5   weak; the winner is partly luck
    > 0.5       the search is worse than random -- picking the best in-sample
                config actively predicts below-median out-of-sample results

CRITICAL: the matrix must contain EVERY configuration you actually tried, not a
tidy subset chosen afterwards. Feeding it five configs when you searched two
hundred produces a reassuring number that means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass
class PBOResult:
    pbo: float
    n_splits: int
    n_configs: int
    n_blocks: int
    logits: np.ndarray
    oos_ranks: np.ndarray
    best_config_counts: dict

    @property
    def verdict(self) -> str:
        if self.pbo < 0.2:
            return "acceptable -- selection carries information"
        if self.pbo < 0.5:
            return "weak -- the in-sample winner is substantially luck"
        return "FAIL -- selecting on in-sample performance predicts BELOW-median OOS"

    def summary(self) -> str:
        return (
            f"PBO = {self.pbo:.1%} over {self.n_splits} splits, "
            f"{self.n_configs} configs, {self.n_blocks} blocks\n"
            f"verdict: {self.verdict}"
        )


def _sharpe_like(block: np.ndarray) -> np.ndarray:
    """Per-column performance statistic. Mean/std, undefined columns -> -inf."""
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    out = np.full_like(mu, -np.inf, dtype=float)
    ok = sd > 0
    out[ok] = mu[ok] / sd[ok]
    return out


def compute_pbo(returns_matrix: pd.DataFrame, n_blocks: int = 8) -> PBOResult:
    """returns_matrix: rows = periods, columns = parameter configurations."""
    m = returns_matrix.dropna(how="any")
    if m.shape[1] < 2:
        raise ValueError("need at least 2 configurations to measure PBO")
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even")
    if m.shape[0] < n_blocks * 2:
        raise ValueError(f"need at least {n_blocks*2} periods for {n_blocks} blocks")

    arr = m.to_numpy(dtype=float)
    n_rows, n_cfg = arr.shape
    blocks = np.array_split(np.arange(n_rows), n_blocks)

    logits, ranks = [], []
    winners: dict = {}

    for is_idx in combinations(range(n_blocks), n_blocks // 2):
        oos_idx = [b for b in range(n_blocks) if b not in is_idx]
        is_rows = np.concatenate([blocks[b] for b in is_idx])
        oos_rows = np.concatenate([blocks[b] for b in oos_idx])

        is_perf = _sharpe_like(arr[is_rows])
        oos_perf = _sharpe_like(arr[oos_rows])

        best = int(np.argmax(is_perf))
        name = m.columns[best]
        winners[name] = winners.get(name, 0) + 1

        # Relative rank of the in-sample winner within the OOS ordering.
        order = np.argsort(np.argsort(oos_perf))       # 0 = worst
        rel = (order[best] + 1) / (n_cfg + 1)
        rel = min(max(rel, 1e-9), 1 - 1e-9)
        ranks.append(rel)
        logits.append(np.log(rel / (1 - rel)))

    logits = np.asarray(logits)
    ranks = np.asarray(ranks)
    return PBOResult(
        pbo=float((logits <= 0).mean()),
        n_splits=len(logits),
        n_configs=n_cfg,
        n_blocks=n_blocks,
        logits=logits,
        oos_ranks=ranks,
        best_config_counts=winners,
    )


def block_bootstrap_drawdown(
    returns: pd.Series, n_sims: int = 1000, block: int = 21, seed: int = 0
) -> dict:
    """Monte Carlo that preserves autocorrelation.

    Plain trade-sequence reshuffling assumes trades are independent. They are
    not -- momentum clusters wins and losses by regime, so reshuffling destroys
    exactly the structure that produces real drawdowns and returns a
    flatteringly narrow distribution. A moving-block bootstrap keeps local
    dependence intact.
    """
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = len(r)
    if n < block * 2:
        raise ValueError("series too short for the chosen block length")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block

    dds, terminals = [], []
    for _ in range(n_sims):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        path = np.concatenate([r[s:s + block] for s in starts])[:n]
        eq = np.cumprod(1 + path)
        dds.append(float((eq / np.maximum.accumulate(eq) - 1).min()))
        terminals.append(float(eq[-1]))

    dds, terminals = np.array(dds), np.array(terminals)
    return {
        "median_drawdown": float(np.median(dds)),
        "p05_drawdown": float(np.percentile(dds, 5)),
        "p01_drawdown": float(np.percentile(dds, 1)),
        "median_terminal_multiple": float(np.median(terminals)),
        "p05_terminal_multiple": float(np.percentile(terminals, 5)),
        "prob_loss": float((terminals < 1.0).mean()),
        "n_sims": n_sims,
        "block": block,
    }


__all__ = ["compute_pbo", "PBOResult", "block_bootstrap_drawdown"]
