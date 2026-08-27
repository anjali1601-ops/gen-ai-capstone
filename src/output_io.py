"""Read pipeline artefacts from disk for the Streamlit evaluation window."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.models import CaseSummary, CustomerEmail

REPORT_NAME = "final_report.csv"
EMAILS_DIR = "customer_emails"
SUMMARIES_DIR = "case_summaries"
STRUCTURED_DIR = "structured_data"

_SUMMARY_LABELS: tuple[tuple[str, str], ...] = (
    ("Case overview", "case_overview"),
    ("Key issue", "key_issue"),
    ("Action taken", "action_taken"),
    ("Current status", "current_status"),
    ("Recommended next action", "recommended_next_action"),
)


def report_path(output_dir: Path) -> Path:
    return output_dir / REPORT_NAME


def artefact_path(output_dir: Path, folder: str, source_file: str, suffix: str) -> Path:
    return output_dir / folder / f"{Path(source_file).stem}{suffix}"


def read_text_file(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def parse_customer_email(text: str) -> CustomerEmail:
    lines = text.replace("\r\n", "\n").split("\n")
    subject = ""
    body_lines = lines
    if lines and lines[0].startswith("Subject:"):
        subject = lines[0].split(":", 1)[1].strip()
        body_lines = lines[1:]
        if body_lines and body_lines[0] == "":
            body_lines = body_lines[1:]
    return CustomerEmail(subject=subject, body="\n".join(body_lines).rstrip("\n"))


def parse_case_summary(text: str) -> CaseSummary:
    chunks: dict[str, list[str]] = {field: [] for _, field in _SUMMARY_LABELS}
    current: str | None = None
    for raw in text.replace("\r\n", "\n").split("\n"):
        matched: str | None = None
        for label, field in _SUMMARY_LABELS:
            prefix = f"{label}:"
            if raw.startswith(prefix):
                remainder = raw[len(prefix) :].strip()
                chunks[field] = [remainder] if remainder else []
                matched = field
                break
        if matched:
            current = matched
        elif current is not None and raw.strip():
            chunks[current].append(raw)
    return CaseSummary(
        **{field: "\n".join(parts).strip() for field, parts in chunks.items()}
    )


def load_customer_email(output_dir: Path, source_file: str) -> CustomerEmail | None:
    text = read_text_file(artefact_path(output_dir, EMAILS_DIR, source_file, ".txt"))
    if text is None:
        return None
    try:
        return parse_customer_email(text)
    except (ValidationError, ValueError):
        return CustomerEmail(subject="", body=text.strip())


def load_case_summary(output_dir: Path, source_file: str) -> CaseSummary | None:
    text = read_text_file(artefact_path(output_dir, SUMMARIES_DIR, source_file, ".txt"))
    if text is None:
        return None
    try:
        return parse_case_summary(text)
    except (ValidationError, ValueError):
        return None


def load_structured_json_text(output_dir: Path, source_file: str) -> str | None:
    return read_text_file(artefact_path(output_dir, STRUCTURED_DIR, source_file, ".json"))


def read_final_report(output_dir: Path) -> pd.DataFrame | None:
    path = report_path(output_dir)
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return None
    return frame


def _stems_in_folder(folder: Path, suffix: str) -> set[str]:
    names: set[str] = set()
    if not folder.is_dir():
        return names
    try:
        entries = folder.iterdir()
    except OSError:
        return names
    for path in entries:
        if path.is_file() and path.suffix.lower() == suffix:
            names.add(path.stem)
    return names


def list_source_files(output_dir: Path, report: pd.DataFrame | None = None) -> list[str]:
    """Prefer CSV order; fall back to artefact folder stems."""
    names: list[str] = []
    seen: set[str] = set()
    frame = report if report is not None else read_final_report(output_dir)
    if frame is not None and "source_file" in frame.columns:
        for value in frame["source_file"].tolist():
            name = str(value).strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    if names:
        return names

    stems = (
        _stems_in_folder(output_dir / EMAILS_DIR, ".txt")
        | _stems_in_folder(output_dir / SUMMARIES_DIR, ".txt")
        | _stems_in_folder(output_dir / STRUCTURED_DIR, ".json")
    )
    return sorted(stems)


def row_for_source(report: pd.DataFrame | None, source_file: str) -> dict[str, str] | None:
    if report is None or "source_file" not in report.columns:
        return None
    matched = report[report["source_file"].astype(str) == source_file]
    if matched.empty:
        return None
    return {str(key): str(value) for key, value in matched.iloc[0].to_dict().items()}
