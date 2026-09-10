"""Human-in-the-loop review flags for generated customer emails.

Emails are written as files. They stay Draft until an admin marks Approved.
This app does not send mail.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DRAFT = "draft"
APPROVED = "approved"
KNOWN = frozenset({DRAFT, APPROVED})


def review_path(output_dir: Path) -> Path:
    return Path(output_dir) / "review_status.json"


def load_reviews(output_dir: Path) -> dict[str, dict[str, str]]:
    path = review_path(output_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    cleaned: dict[str, dict[str, str]] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            cleaned[str(key)] = {
                "status": str(value.get("status") or DRAFT),
                "reviewer": str(value.get("reviewer") or ""),
                "updated_at": str(value.get("updated_at") or ""),
            }
    return cleaned


def get_status(output_dir: Path, source_file: str) -> str:
    entry = load_reviews(output_dir).get(source_file) or {}
    status = entry.get("status", DRAFT)
    return status if status in KNOWN else DRAFT


def set_status(
    output_dir: Path,
    source_file: str,
    status: str,
    reviewer: str,
) -> None:
    if status not in KNOWN:
        raise ValueError("Review status must be draft or approved.")
    records = load_reviews(output_dir)
    records[source_file] = {
        "status": status,
        "reviewer": (reviewer or "").strip()[:80],
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = review_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
