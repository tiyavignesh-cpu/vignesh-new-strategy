"""
engine/state.py — Atomic state writer for bot_state.json and trade_log.csv.

Follows the atomic persistence pattern: .tmp -> fsync -> os.replace with .bak backup.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

IST = timezone(timedelta(hours=5, minutes=30))
DEFAULT_STATE_FILE = "bot_state.json"
DEFAULT_TRADE_LOG = "trade_log.csv"


class StateWriter:
    """Handles atomic persistence of trading bot state and trade logs."""

    def __init__(self, state_file: str = DEFAULT_STATE_FILE, trade_log_file: str = DEFAULT_TRADE_LOG):
        self.state_file = state_file
        self.trade_log_file = trade_log_file

    def load_state(self) -> dict:
        """Load state JSON with automatic .bak fallback."""
        if not os.path.exists(self.state_file):
            if os.path.exists(self.state_file + ".bak"):
                try:
                    with open(self.state_file + ".bak", "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return {}
            return {}
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            if os.path.exists(self.state_file + ".bak"):
                try:
                    with open(self.state_file + ".bak", "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
            return {}

    def save_state(self, state: dict) -> str:
        """Atomically persist state dictionary."""
        tmp_file = self.state_file + ".tmp"
        bak_file = self.state_file + ".bak"

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())

        if os.path.exists(self.state_file):
            try:
                shutil.copy2(self.state_file, bak_file)
            except Exception:
                pass

        os.replace(tmp_file, self.state_file)
        return self.state_file

    def append_trade_log(self, trade_records: List[dict]) -> str:
        """Append closed trades to trade_log.csv."""
        if not trade_records:
            return self.trade_log_file

        file_exists = os.path.exists(self.trade_log_file)
        fieldnames = list(trade_records[0].keys())

        with open(self.trade_log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(trade_records)
            f.flush()
            os.fsync(f.fileno())

        return self.trade_log_file


__all__ = ["StateWriter", "DEFAULT_STATE_FILE", "DEFAULT_TRADE_LOG"]
