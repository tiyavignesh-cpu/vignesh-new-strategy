"""
Data loader with raw vs adjusted price separation, IST calendar alignment,
and parquet caching.

Contract:
- Signals and returns are computed on ADJUSTED close.
- Price floors (>= Rs 50) and liquidity floors (>= Rs 5 Cr ADT) use RAW close and RAW volume.
- All wide frames: index = DatetimeIndex of NSE trading days, columns = symbols.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
DELISTED_CACHE_FILE = os.path.join(os.path.dirname(__file__), "delisted_cache.json")
IST = timezone(timedelta(hours=5, minutes=30))


def _load_delisted_cache() -> Dict[str, dict]:
    if os.path.exists(DELISTED_CACHE_FILE):
        try:
            with open(DELISTED_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_delisted_cache(cache: Dict[str, dict]) -> None:
    try:
        with open(DELISTED_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def fetch_symbol_ohlcv(
    symbol: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV for one symbol, keeping both raw and adjusted columns."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{symbol}.parquet")

    if use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                return df
        except Exception:
            pass

    delisted_cache = _load_delisted_cache()
    if symbol in delisted_cache and delisted_cache[symbol].get("strikes", 0) >= 3:
        return None

    # Attempt download via yfinance
    try:
        import yfinance as yf

        ticker_str = f"{symbol}.NS" if not symbol.startswith("^") else symbol
        t = yf.Ticker(ticker_str)
        hist = t.history(start=start_date, end=end_date, auto_adjust=False)

        if hist.empty or len(hist) < 30:
            delisted_cache[symbol] = {
                "last_attempt": datetime.now(IST).isoformat(),
                "strikes": delisted_cache.get(symbol, {}).get("strikes", 0) + 1,
            }
            _save_delisted_cache(delisted_cache)
            return None

        # Clean index and columns
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in hist.columns for c in required):
            return None

        if "Adj Close" not in hist.columns:
            hist["Adj Close"] = hist["Close"]

        out = hist[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
        out = out.dropna(subset=["Close", "Adj Close"])

        # Cache to parquet
        try:
            out.to_parquet(cache_path)
        except Exception:
            pass

        return out

    except Exception:
        delisted_cache[symbol] = {
            "last_attempt": datetime.now(IST).isoformat(),
            "strikes": delisted_cache.get(symbol, {}).get("strikes", 0) + 1,
        }
        _save_delisted_cache(delisted_cache)
        return None


def generate_synthetic_universe(
    symbols: List[str],
    start_date: str = "2015-01-01",
    end_date: str = "2026-08-01",
    seed: int = 42,
) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """Generate realistic synthetic OHLCV universe with benchmark when offline."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)

    # Synthetic Benchmark (NIFTY 50)
    bench_returns = rng.normal(0.00045, 0.010, n)
    bench_prices = 8000.0 * np.cumprod(1 + bench_returns)
    bench_series = pd.Series(bench_prices, index=dates, name="^NSEI")

    data: Dict[str, pd.DataFrame] = {}

    for idx, sym in enumerate(symbols):
        beta = rng.uniform(0.6, 1.4)
        idio = rng.normal(0.0002, 0.016, n)
        sym_returns = beta * bench_returns + idio

        # Occasional corporate action (split or dividend drift)
        adj_factor = np.ones(n)
        if idx % 5 == 0:
            split_idx = n // 2
            adj_factor[:split_idx] = 0.5

        raw_close = rng.uniform(80, 2500) * np.cumprod(1 + sym_returns)
        raw_close = np.maximum(raw_close, 10.0)
        adj_close = raw_close * adj_factor

        noise = np.abs(rng.normal(0, 0.006, n))
        raw_open = np.concatenate([[raw_close[0]], raw_close[:-1]]) * (1 + rng.normal(0, 0.004, n))
        raw_high = np.maximum(raw_close, raw_open) * (1 + noise)
        raw_low = np.minimum(raw_close, raw_open) * (1 - noise)
        raw_volume = rng.lognormal(13.5, 0.8, n)

        df = pd.DataFrame(
            {
                "Open": raw_open,
                "High": raw_high,
                "Low": raw_low,
                "Close": raw_close,
                "Adj Close": adj_close,
                "Volume": raw_volume,
            },
            index=dates,
        )
        data[sym] = df

    return data, bench_series


def load_dataset(
    symbols: List[str],
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    allow_synthetic_fallback: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    """Load complete universe data, returning {symbol: DataFrame} and benchmark series."""
    data: Dict[str, pd.DataFrame] = {}

    # Fetch benchmark first
    bench_df = fetch_symbol_ohlcv("^NSEI", start_date, end_date)
    bench_series = bench_df["Close"] if bench_df is not None else None

    for sym in symbols:
        df = fetch_symbol_ohlcv(sym, start_date, end_date)
        if df is not None and len(df) >= 200:
            data[sym] = df

    # Fallback to deterministic synthetic universe if offline or no network data
    if (len(data) < len(symbols) * 0.3 or bench_series is None) and allow_synthetic_fallback:
        end_str = end_date or datetime.now(IST).strftime("%Y-%m-%d")
        synth_data, synth_bench = generate_synthetic_universe(symbols, start_date, end_str)
        # Use synthetic data for missing symbols
        for sym, df in synth_data.items():
            if sym not in data:
                data[sym] = df
        if bench_series is None:
            bench_series = synth_bench

    return data, bench_series


def build_wide_frames(
    data: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assemble aligned wide matrices for vectorized momentum rotation."""
    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    idx = pd.DatetimeIndex(all_dates)

    raw_close = pd.DataFrame(index=idx)
    adj_close = pd.DataFrame(index=idx)
    raw_volume = pd.DataFrame(index=idx)
    raw_high = pd.DataFrame(index=idx)
    raw_low = pd.DataFrame(index=idx)

    for sym, df in data.items():
        raw_close[sym] = df["Close"].reindex(idx)
        adj_close[sym] = df["Adj Close"].reindex(idx)
        raw_volume[sym] = df["Volume"].reindex(idx).fillna(0.0)
        raw_high[sym] = df["High"].reindex(idx)
        raw_low[sym] = df["Low"].reindex(idx)

    return raw_close, adj_close, raw_volume, raw_high, raw_low


__all__ = [
    "fetch_symbol_ohlcv",
    "generate_synthetic_universe",
    "load_dataset",
    "build_wide_frames",
]
