"""Save per-document outputs and the consolidated CSV report."""

from __future__ import annotations

import csv
from pathlib import Path

from src.models import CaseRecord, CaseSummary, CustomerEmail


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    folders = {
        "structured": output_dir / "structured_data",
        "emails": output_dir / "customer_emails",
        "summaries": output_dir / "case_summaries",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def save_case_outputs(
    output_dir: Path,
    stem: str,
    case: CaseRecord,
    email: CustomerEmail,
    summary: CaseSummary,
) -> None:
    folders = ensure_output_dirs(output_dir)
    (folders["structured"] / f"{stem}.json").write_text(
        case.model_dump_json(indent=2),
        encoding="utf-8",
    )
    email_text = f"Subject: {email.subject}\n\n{email.body}\n"
    (folders["emails"] / f"{stem}.txt").write_text(email_text, encoding="utf-8")

    summary_text = "\n".join(
        [
            f"Case overview: {summary.case_overview}",
            f"Key issue: {summary.key_issue}",
            f"Action taken: {summary.action_taken}",
            f"Current status: {summary.current_status}",
            f"Recommended next action: {summary.recommended_next_action}",
            "",
        ]
    )
    (folders["summaries"] / f"{stem}.txt").write_text(summary_text, encoding="utf-8")


def write_final_report(output_dir: Path, results) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "final_report.csv"
    fieldnames = [
        "source_file",
        "status",
        "customer_name",
        "email",
        "phone_number",
        "complaint_category",
        "issue_description",
        "resolution_provided",
        "is_complaint",
        "escalation_required",
        "supporting_document_available",
        "overall_case_status",
        "error",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            case = result.case
            writer.writerow(
                {
                    "source_file": result.source_file,
                    "status": result.status,
                    "customer_name": case.customer_name if case else "",
                    "email": case.email if case else "",
                    "phone_number": case.phone_number if case else "",
                    "complaint_category": case.complaint_category if case else "",
                    "issue_description": case.issue_description if case else "",
                    "resolution_provided": case.resolution_provided if case else "",
                    "is_complaint": case.is_complaint.value if case else "",
                    "escalation_required": case.escalation_required.value if case else "",
                    "supporting_document_available": (
                        case.supporting_document_available.value if case else ""
                    ),
                    "overall_case_status": case.overall_case_status if case else "",
                    "error": result.error or "",
                }
            )
    return report_path
