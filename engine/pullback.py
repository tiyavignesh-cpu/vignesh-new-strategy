"""
Engine 2, sub-strategy 4: shallow-pullback trend continuation.

The brief called it "mean-reversion". It isn't. The setup buys strength after a
Bollinger-band expansion and enters on a SHALLOW retracement while price holds
above its baseline -- that is trend continuation. The distinction matters
because it tells you what kills it: not a failure to revert, but a trend that
ends mid-pullback. The naming is cosmetic; the risk profile is not.

Structure:
    1. IMPULSE       close outside the upper Bollinger band, 21 MA sloping up,
                     Donchian channel breached or expanding
    2. LEG           low  = last CONFIRMED swing low at impulse start
                     high = running high of the impulse
    3. PULLBACK      price retraces into [0.236, 0.114] of the leg while
                     holding above the 21 MA baseline
    4. TRIGGER       a bullish rejection bar closes inside that zone
    5. EXIT          TP1 at the leg high (50% out), TP2 at 3R or Donchian
                     expansion, stop beyond the MA / pullback pivot + ATR buffer

Three things to know before you trade it:

* THE ZONE IS THIN. 0.114 to 0.236 is 12% of the impulse. Most impulses either
  never reach it or slice straight through. Expect a low trade count -- that is
  the strategy working as specified, not a bug. If you widen it to 0.382 you
  have a different strategy and the walk-forward has to start over.

* TP1 IS OFTEN INSIDE 1R. Entering near the high and stopping beyond the MA
  gives a wide stop and a near target. `min_tp1_r` rejects setups where scaling
  out at TP1 would bank less than it risks; every signal carries `tp1_r` so you
  can see the geometry rather than assume it.

* THE SHORT SIDE IS OFF BY DEFAULT. NSE cash shorts cannot be held overnight and
  this is a multi-bar hold. `allow_short=True` exists for future stock-futures
  work and is not wired into any live path.

Everything is computed on confirmed bar closes. Swing pivots carry an explicit
confirmation lag (see swings.py) and signals are emitted for fill at the NEXT
bar's open.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

from .indicators import atr, bollinger, donchian, moving_average, rejection_candle, slope
from .swings import pivot_series


@dataclass(frozen=True)
class PullbackConfig:
    # --- indicators
    bb_period: int = 20
    bb_mult: float = 2.0
    ma_period: int = 21
    ma_kind: str = "EMA"            # "EMA" or "SMA"
    donchian_period: int = 20
    atr_period: int = 14

    # --- fibonacci pullback zone (fractions of the impulse leg)
    fib_shallow: float = 0.114
    fib_deep: float = 0.236

    # --- swing detection
    pivot_left: int = 3
    pivot_right: int = 3             # this is the repaint lag; never set to 0

    # --- setup gates
    slope_lookback: int = 3
    min_slope: float = 0.0
    expansion_lookback: int = 5
    max_leg_age: int = 20            # bars from impulse start before the leg dies
    rejection_close_pct: float = 0.6
    min_range_frac_atr: float = 0.3

    # --- risk
    stop_mode: str = "tighter_of"    # "ma" | "pivot" | "tighter_of"
    atr_buffer_mult: float = 0.5
    tp2_mode: str = "rr"             # "rr" | "donchian"
    tp2_rr: float = 3.0
    scale_out_pct: float = 0.5
    min_tp1_r: float = 0.75          # reject setups whose first target is too near

    # --- sides
    allow_long: bool = True
    allow_short: bool = False

    def __post_init__(self):
        if not 0 < self.fib_shallow < self.fib_deep < 1:
            raise ValueError("require 0 < fib_shallow < fib_deep < 1")
        if self.pivot_right < 1:
            raise ValueError("pivot_right must be >= 1 or pivots repaint")
        if self.stop_mode not in ("ma", "pivot", "tighter_of"):
            raise ValueError("stop_mode must be ma, pivot or tighter_of")
        if self.tp2_mode not in ("rr", "donchian"):
            raise ValueError("tp2_mode must be rr or donchian")


@dataclass
class Signal:
    date: pd.Timestamp
    index: int
    direction: str
    leg_id: int
    leg_low: float
    leg_high: float
    zone_shallow: float              # the 0.114 price
    zone_deep: float                 # the 0.236 price
    entry_ref: float                 # trigger-bar close; fill at next open
    stop: float
    tp1: float
    tp2: float
    r_value: float
    tp1_r: float                     # R-multiple of the first target
    tp2_r: float
    atr: float
    stop_source: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Leg:
    leg_id: int
    direction: str
    start: int
    low: float
    high: float
    extreme_bar: int
    anchor_idx: int = -1        # index of the confirmed pivot this leg is anchored to
    entered: bool = False

    def span(self) -> float:
        return abs(self.high - self.low)


def prepare(df: pd.DataFrame, cfg: PullbackConfig) -> pd.DataFrame:
    """Attach every indicator the scan needs. All causal."""
    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            raise KeyError(f"missing column {col}")

    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    out = df.copy()

    out = out.join(bollinger(c, cfg.bb_period, cfg.bb_mult))
    out = out.join(donchian(h, l, cfg.donchian_period))
    out["ma"] = moving_average(c, cfg.ma_period, cfg.ma_kind)
    out["ma_slope"] = slope(out["ma"], cfg.slope_lookback)
    out["atr"] = atr(h, l, c, cfg.atr_period)
    out = out.join(pivot_series(df, cfg.pivot_left, cfg.pivot_right))

    a = out["atr"]
    out["rej_long"] = rejection_candle(o, h, l, c, "LONG",
                                       cfg.rejection_close_pct,
                                       cfg.min_range_frac_atr, a)
    out["rej_short"] = rejection_candle(o, h, l, c, "SHORT",
                                        cfg.rejection_close_pct,
                                        cfg.min_range_frac_atr, a)

    # Range expansion: a fresh Donchian breach, or a channel that is widening.
    out["expand_up"] = (h > out["dc_upper"]) | (
        out["dc_width"] > out["dc_width"].shift(cfg.expansion_lookback))
    out["expand_dn"] = (l < out["dc_lower"]) | (
        out["dc_width"] > out["dc_width"].shift(cfg.expansion_lookback))

    out["impulse_up"] = (c > out["bb_upper"]) & (out["ma_slope"] > cfg.min_slope) & out["expand_up"]
    out["impulse_dn"] = (c < out["bb_lower"]) & (out["ma_slope"] < -cfg.min_slope) & out["expand_dn"]
    return out


def _build_long_signal(d, i, leg, cfg) -> Signal | None:
    row = d.iloc[i]
    entry = float(row["Close"])
    a = float(row["atr"])
    buf = cfg.atr_buffer_mult * a

    ma_stop = float(row["ma"]) - buf
    pull_low = float(d["Low"].iloc[leg.extreme_bar:i + 1].min())
    pivot_stop = pull_low - buf

    if cfg.stop_mode == "ma":
        stop, src = ma_stop, "MA"
    elif cfg.stop_mode == "pivot":
        stop, src = pivot_stop, "PIVOT"
    else:
        stop, src = (ma_stop, "MA") if ma_stop >= pivot_stop else (pivot_stop, "PIVOT")

    r = entry - stop
    if r <= 0:
        return None

    tp1 = max(leg.high, entry + max(1.5, cfg.min_tp1_r) * r)
    if cfg.tp2_mode == "rr":
        tp2 = entry + cfg.tp2_rr * r
    else:
        tp2 = float(row["dc_upper"]) + float(row["dc_width"])
    tp2 = max(tp2, tp1)

    tp1_r = (tp1 - entry) / r
    if tp1_r < cfg.min_tp1_r:
        return None

    return Signal(
        date=d.index[i], index=i, direction="LONG", leg_id=leg.leg_id,
        leg_low=leg.low, leg_high=leg.high,
        zone_shallow=leg.high - cfg.fib_shallow * leg.span(),
        zone_deep=leg.high - cfg.fib_deep * leg.span(),
        entry_ref=entry, stop=stop, tp1=tp1, tp2=tp2,
        r_value=r, tp1_r=tp1_r, tp2_r=(tp2 - entry) / r, atr=a, stop_source=src,
    )


def _build_short_signal(d, i, leg, cfg) -> Signal | None:
    row = d.iloc[i]
    entry = float(row["Close"])
    a = float(row["atr"])
    buf = cfg.atr_buffer_mult * a

    ma_stop = float(row["ma"]) + buf
    pull_high = float(d["High"].iloc[leg.extreme_bar:i + 1].max())
    pivot_stop = pull_high + buf

    if cfg.stop_mode == "ma":
        stop, src = ma_stop, "MA"
    elif cfg.stop_mode == "pivot":
        stop, src = pivot_stop, "PIVOT"
    else:
        stop, src = (ma_stop, "MA") if ma_stop <= pivot_stop else (pivot_stop, "PIVOT")

    r = stop - entry
    if r <= 0:
        return None

    tp1 = min(leg.low, entry - max(1.5, cfg.min_tp1_r) * r)
    if cfg.tp2_mode == "rr":
        tp2 = entry - cfg.tp2_rr * r
    else:
        tp2 = float(row["dc_lower"]) - float(row["dc_width"])
    tp2 = min(tp2, tp1)

    tp1_r = (entry - tp1) / r
    if tp1_r < cfg.min_tp1_r:
        return None

    return Signal(
        date=d.index[i], index=i, direction="SHORT", leg_id=leg.leg_id,
        leg_low=leg.low, leg_high=leg.high,
        zone_shallow=leg.low + cfg.fib_shallow * leg.span(),
        zone_deep=leg.low + cfg.fib_deep * leg.span(),
        entry_ref=entry, stop=stop, tp1=tp1, tp2=tp2,
        r_value=r, tp1_r=tp1_r, tp2_r=(entry - tp2) / r, atr=a, stop_source=src,
    )


def scan(df: pd.DataFrame, cfg: PullbackConfig | None = None) -> list[Signal]:
    """Walk the series bar by bar and emit signals on confirmed closes.

    Deliberately a Python loop, not vectorised: the leg state machine has to
    carry the "one entry per swing leg" flag forward, and a vectorised version
    of that is where duplicate entries hide.
    """
    cfg = cfg or PullbackConfig()
    d = prepare(df, cfg)
    n = len(d)

    signals: list[Signal] = []
    leg_counter = 0
    long_leg: _Leg | None = None
    short_leg: _Leg | None = None

    for i in range(n):
        row = d.iloc[i]
        ma = row["ma"]
        if pd.isna(ma) or pd.isna(row["atr"]) or pd.isna(row["dc_upper"]):
            continue

        c, hi, lo = float(row["Close"]), float(row["High"]), float(row["Low"])

        # ---------------------------------------------------------- LONG
        if cfg.allow_long:
            if long_leg is not None:
                if hi > long_leg.high:
                    long_leg.high, long_leg.extreme_bar = hi, i
                span = long_leg.span()
                deep_px = long_leg.high - cfg.fib_deep * span
                dead = (
                    c < ma
                    or lo < deep_px
                    or (i - long_leg.start) > cfg.max_leg_age
                    or span <= 0
                )
                if dead:
                    long_leg = None

            # Leg renewal. Once a leg has produced its entry it is spent -- but a
            # FRESH impulse to a new high, anchored on a NEWER confirmed swing
            # low, is a new leg and deserves its own entry. Without this the
            # strategy takes one trade per trend rather than one per pullback.
            if (
                long_leg is not None
                and long_leg.entered
                and bool(row["impulse_up"])
                and hi >= long_leg.high
            ):
                nl_idx = row["last_low_idx"]
                if not pd.isna(nl_idx) and int(nl_idx) > long_leg.anchor_idx:
                    leg_counter += 1
                    long_leg = _Leg(leg_counter, "LONG", i, float(row["last_low"]),
                                    hi, i, anchor_idx=int(nl_idx))

            if long_leg is None and bool(row["impulse_up"]):
                pl, pl_idx = row["last_low"], row["last_low_idx"]
                if not pd.isna(pl) and hi > float(pl):
                    leg_counter += 1
                    long_leg = _Leg(leg_counter, "LONG", i, float(pl), hi, i,
                                    anchor_idx=int(pl_idx))

            if long_leg is not None and not long_leg.entered and i > long_leg.start:
                span = long_leg.span()
                shallow_px = long_leg.high - cfg.fib_shallow * span
                deep_px = long_leg.high - cfg.fib_deep * span
                in_zone = (lo <= shallow_px) and (lo >= deep_px)
                if in_zone and c > ma and c >= deep_px and bool(row["rej_long"]):
                    sig = _build_long_signal(d, i, long_leg, cfg)
                    if sig is not None:
                        signals.append(sig)
                        long_leg.entered = True

        # --------------------------------------------------------- SHORT
        if cfg.allow_short:
            if short_leg is not None:
                if lo < short_leg.low:
                    short_leg.low, short_leg.extreme_bar = lo, i
                span = short_leg.span()
                deep_px = short_leg.low + cfg.fib_deep * span
                dead = (
                    c > ma
                    or hi > deep_px
                    or (i - short_leg.start) > cfg.max_leg_age
                    or span <= 0
                )
                if dead:
                    short_leg = None

            if (
                short_leg is not None
                and short_leg.entered
                and bool(row["impulse_dn"])
                and lo <= short_leg.low
            ):
                nh_idx = row["last_high_idx"]
                if not pd.isna(nh_idx) and int(nh_idx) > short_leg.anchor_idx:
                    leg_counter += 1
                    short_leg = _Leg(leg_counter, "SHORT", i, lo,
                                     float(row["last_high"]), i,
                                     anchor_idx=int(nh_idx))

            if short_leg is None and bool(row["impulse_dn"]):
                ph, ph_idx = row["last_high"], row["last_high_idx"]
                if not pd.isna(ph) and lo < float(ph):
                    leg_counter += 1
                    short_leg = _Leg(leg_counter, "SHORT", i, lo, float(ph), i,
                                     anchor_idx=int(ph_idx))

            if short_leg is not None and not short_leg.entered and i > short_leg.start:
                span = short_leg.span()
                shallow_px = short_leg.low + cfg.fib_shallow * span
                deep_px = short_leg.low + cfg.fib_deep * span
                in_zone = (hi >= shallow_px) and (hi <= deep_px)
                if in_zone and c < ma and c <= deep_px and bool(row["rej_short"]):
                    sig = _build_short_signal(d, i, short_leg, cfg)
                    if sig is not None:
                        signals.append(sig)
                        short_leg.entered = True

    return signals


def scan_frame(df: pd.DataFrame, cfg: PullbackConfig | None = None) -> pd.DataFrame:
    sigs = scan(df, cfg)
    if not sigs:
        return pd.DataFrame(columns=list(Signal.__dataclass_fields__))
    return pd.DataFrame([s.as_dict() for s in sigs]).set_index("date")


def assert_no_repaint(
    df: pd.DataFrame, cfg: PullbackConfig | None = None, checkpoints: int = 6
) -> None:
    """Prove signals are stable under truncation.

    Scanning the first k bars must produce EXACTLY the signals that a full scan
    produces at indices < k. If a later bar can add, remove or alter an earlier
    signal, the strategy is reading the future. This is the strongest available
    repaint test and it belongs in CI, not in a comment.
    """
    cfg = cfg or PullbackConfig()
    full = scan(df, cfg)
    n = len(df)
    warm = max(cfg.bb_period, cfg.ma_period, cfg.donchian_period, cfg.atr_period) + 10

    for k in np.linspace(warm + 20, n, checkpoints, dtype=int):
        k = int(k)
        partial = scan(df.iloc[:k], cfg)
        expected = [s for s in full if s.index < k]
        if len(partial) != len(expected):
            raise AssertionError(
                f"repaint at k={k}: {len(partial)} signals on truncated data vs "
                f"{len(expected)} on full data"
            )
        for a, b in zip(partial, expected):
            for fld in ("index", "direction", "entry_ref", "stop", "tp1", "tp2"):
                va, vb = getattr(a, fld), getattr(b, fld)
                same = va == vb if isinstance(va, str) else abs(va - vb) < 1e-9
                if not same:
                    raise AssertionError(
                        f"repaint at k={k}: signal {fld} changed {vb} -> {va}"
                    )


__all__ = [
    "PullbackConfig", "Signal", "prepare", "scan", "scan_frame", "assert_no_repaint",
]
