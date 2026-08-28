"""
engine/oos_lock.py — Strict OOS Partition Access Lock (Amendment A8).

Ensures the 2025–2026 Final Untouched OOS dataset partition can only be read exactly once.
"""

from __future__ import annotations

import os
import hashlib
from datetime import datetime

LOCK_FILE_PATH = os.path.abspath(r"c:\Users\Vignesh\Desktop\vignesh new strategy\reports\engine2_2\.oos_locked")


class OOSAccessViolationError(RuntimeError):
    """Raised when the 2025-2026 OOS dataset is accessed more than once."""
    pass


def acquire_oos_access_token(caller_id: str = "FINAL_OOS_EVALUATION") -> str:
    """
    Acquires an exclusive, single-use lock token for 2025–2026 OOS data.
    If the lock file already exists, raises OOSAccessViolationError and aborts execution.
    """
    os.makedirs(os.path.dirname(LOCK_FILE_PATH), exist_ok=True)
    if os.path.exists(LOCK_FILE_PATH):
        with open(LOCK_FILE_PATH, "r", encoding="utf-8") as f:
            prev_info = f.read()
        raise OOSAccessViolationError(
            f"CRITICAL RESEARCH INTEGRITY VIOLATION: Final Untouched OOS (2025-2026) has already been accessed!\n"
            f"Previous access record:\n{prev_info}\n"
            f"Execution halted to prevent post-hoc curve fitting."
        )

    timestamp = datetime.now().isoformat()
    token_str = f"{caller_id}_{timestamp}_{os.getpid()}"
    token_hash = hashlib.sha256(token_str.encode("utf-8")).hexdigest()

    with open(LOCK_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(f"caller_id: {caller_id}\ntimestamp: {timestamp}\ntoken_hash: {token_hash}\n")

    print(f"OOS Lock Acquired: {token_hash[:16]}... Recorded in {LOCK_FILE_PATH}")
    return token_hash


def is_oos_locked() -> bool:
    """Checks if OOS data has been locked."""
    return os.path.exists(LOCK_FILE_PATH)


def reset_oos_lock_for_testing() -> None:
    """Explicit test reset helper (used only in clean development resets)."""
    if os.path.exists(LOCK_FILE_PATH):
        os.remove(LOCK_FILE_PATH)
