"""Append-only batch audit log (who ran what, and the outcome)."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from src.config import LOG_DIR

AUDIT_FIELDS = (
    "timestamp",
    "actor",
    "action",
    "provider",
    "model",
    "total",
    "processed",
    "failed",
    "detail",
)


def audit_path() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / "audit.csv"


def record_audit(
    *,
    actor: str,
    action: str,
    provider: str = "",
    model: str = "",
    total: int = 0,
    processed: int = 0,
    failed: int = 0,
    detail: str = "",
) -> Path:
    path = audit_path()
    new_file = not path.is_file()
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "actor": (actor or "unknown").strip()[:80],
        "action": (action or "run_batch").strip()[:40],
        "provider": (provider or "")[:40],
        "model": (model or "")[:80],
        "total": str(total),
        "processed": str(processed),
        "failed": str(failed),
        "detail": (detail or "").replace("\n", " ")[:240],
    }
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return path


def read_audit(limit: int = 12) -> list[dict[str, str]]:
    path = audit_path()
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return []
    return rows[-limit:]
