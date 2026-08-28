"""
Residual survivorship bias and data coverage reporting.

Measures the proportion of historical point-in-time index members that have
valid price data versus those omitted or missing.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
from .universe import PointInTimeUniverse


def generate_coverage_report(
    universe: PointInTimeUniverse,
    loaded_data: Dict[str, pd.DataFrame],
    asof_dates: Optional[List[pd.Timestamp]] = None,
) -> dict:
    """Evaluate point-in-time constituent coverage and survivorship gap."""
    all_syms = universe.all_historical_symbols()
    loaded_syms = set(loaded_data.keys())

    available = [s for s in all_syms if s in loaded_syms]
    missing = [s for s in all_syms if s not in loaded_syms]

    coverage_pct = len(available) / len(all_syms) if all_syms else 1.0
    survivorship_gap_pct = 1.0 - coverage_pct

    report = {
        "total_pit_constituents": len(all_syms),
        "available_with_data": len(available),
        "missing_or_delisted": len(missing),
        "coverage_pct": coverage_pct,
        "residual_survivorship_gap_pct": survivorship_gap_pct,
        "missing_symbols": missing,
    }
    import json
    import os
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/coverage_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def print_coverage_report(report: dict) -> None:
    print("=" * 70)
    print("POINT-IN-TIME CONSTITUENT COVERAGE & SURVIVORSHIP AUDIT")
    print("=" * 70)
    print(f"Total Historical Constituents: {report['total_pit_constituents']}")
    print(f"Available with Price Series  : {report['available_with_data']} ({report['coverage_pct']:.1%})")
    print(f"Missing / Delisted Scrips    : {report['missing_or_delisted']} ({report['residual_survivorship_gap_pct']:.1%})")
    if report["missing_symbols"]:
        print(f"Missing Scrips List          : {', '.join(report['missing_symbols'][:15])}")
    print("=" * 70)


__all__ = ["generate_coverage_report", "print_coverage_report"]
