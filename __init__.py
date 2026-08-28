"""
Data ingestion, point-in-time universe, and coverage reporting package.
"""

from .universe import PointInTimeUniverse, KNOWN_RENAMES, DEFAULT_PIT_CONSTITUENTS
from .loader import fetch_symbol_ohlcv, generate_synthetic_universe, load_dataset, build_wide_frames
from .coverage import generate_coverage_report, print_coverage_report

__all__ = [
    "PointInTimeUniverse",
    "KNOWN_RENAMES",
    "DEFAULT_PIT_CONSTITUENTS",
    "fetch_symbol_ohlcv",
    "generate_synthetic_universe",
    "load_dataset",
    "build_wide_frames",
    "generate_coverage_report",
    "print_coverage_report",
]
