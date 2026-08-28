"""
Chart plots for the pullback strategy.

Draws bands, the equilibrium MA, the Donchian channel, the shaded Fibonacci
pullback zone for each detected leg, and entry/stop/target markers.

Pivot markers are drawn at the pivot bar (which is what you want to LOOK at) but
carry a confirmation tick at the bar where they became knowable, so the chart
does not quietly imply you could have acted on them earlier.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .pullback import PullbackConfig, prepare, scan
from .swings import find_pivots


def plot_setups(
    df,
    cfg: PullbackConfig | None = None,
    path: str = "pullback_setups.png",
    title: str = "Pullback setups",
    last_n: int | None = 260,
):
    cfg = cfg or PullbackConfig()
    d = prepare(df, cfg)
    signals = scan(df, cfg)
    pivots = find_pivots(df["High"], df["Low"], cfg.pivot_left, cfg.pivot_right)

    # Frame around the setups when there are any -- showing the tail of the
    # series and cropping the signal out is the classic useless chart.
    n = len(d)
    if signals and last_n is not None:
        first = min(s.index for s in signals)
        last = max(s.index for s in signals)
        start = max(0, first - 100)
        end = min(n, max(last + 80, start + last_n // 2))
    elif last_n is None:
        start, end = 0, n
    else:
        start, end = max(0, n - last_n), n

    view = d.iloc[start:end]
    x = range(len(view))
    idx_of = {i: k for k, i in enumerate(range(start, end))}

    fig, ax = plt.subplots(figsize=(16, 8))

    # Candles, drawn thin so structure stays readable.
    for k, (_, row) in enumerate(view.iterrows()):
        up = row["Close"] >= row["Open"]
        col = "#2e7d32" if up else "#c62828"
        ax.plot([k, k], [row["Low"], row["High"]], color=col, linewidth=0.6, zorder=2)
        ax.plot([k, k], [row["Open"], row["Close"]], color=col, linewidth=2.2, zorder=2)

    ax.plot(x, view["bb_upper"], color="#5c6bc0", linewidth=0.9, label="BB upper")
    ax.plot(x, view["bb_lower"], color="#5c6bc0", linewidth=0.9, label="BB lower")
    ax.fill_between(x, view["bb_lower"], view["bb_upper"], color="#5c6bc0", alpha=0.05)
    ax.plot(x, view["ma"], color="#f57c00", linewidth=1.6,
            label=f"{cfg.ma_period} {cfg.ma_kind} equilibrium")
    ax.plot(x, view["dc_upper"], color="#455a64", linewidth=0.8, linestyle="--",
            label="Donchian")
    ax.plot(x, view["dc_lower"], color="#455a64", linewidth=0.8, linestyle="--")

    for p in pivots:
        if not (start <= p.index < end):
            continue
        k = idx_of.get(p.index)
        kc = idx_of.get(p.confirmed_at)
        if k is None:
            continue
        marker = "v" if p.kind == "HIGH" else "^"
        col = "#c62828" if p.kind == "HIGH" else "#2e7d32"
        ax.scatter([k], [p.price], marker=marker, s=28, color=col, zorder=4, alpha=0.7)
        if kc is not None:
            ax.plot([k, kc], [p.price, p.price], color=col, linewidth=0.5,
                    alpha=0.35, linestyle=":", zorder=1)

    for s in signals:
        if not (start <= s.index < end):
            continue
        k = idx_of[s.index]
        lo_z, hi_z = sorted((s.zone_deep, s.zone_shallow))
        left = idx_of.get(max(start, s.index - cfg.max_leg_age), 0)
        ax.add_patch(Rectangle((left, lo_z), k - left + 3, hi_z - lo_z,
                               color="#ffb300", alpha=0.18, zorder=1))

        ax.scatter([k], [s.entry_ref], marker="o", s=70,
                   facecolor="#1565c0", edgecolor="white", zorder=6)
        ax.annotate(f"L{s.leg_id}  {s.tp1_r:.2f}R", (k, s.entry_ref),
                    textcoords="offset points", xytext=(6, -14), fontsize=8)
        for level, col, lab in ((s.stop, "#c62828", "SL"),
                                (s.tp1, "#2e7d32", "TP1"),
                                (s.tp2, "#00695c", "TP2")):
            ax.hlines(level, k, min(k + 22, len(view) - 1), color=col,
                      linewidth=1.0, linestyle="-", alpha=0.8, zorder=5)
            ax.annotate(lab, (min(k + 22, len(view) - 1), level),
                        textcoords="offset points", xytext=(3, -3),
                        fontsize=7, color=col)

    ax.set_title(f"{title}  --  {len(signals)} signal(s)", fontsize=12)
    ax.set_xlabel("bar")
    ax.set_ylabel("price")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


__all__ = ["plot_setups"]
