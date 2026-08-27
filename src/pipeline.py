"""Three-step workflow: extract, then email and summary in parallel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src import prompts
from src.loaders import DocumentLoadError, extract_text, list_documents
from src.logging_setup import setup_logging
from src.llm import LLMClient, LLMError
from src.models import CaseRecord, CaseSummary, CustomerEmail
from src.report import save_case_outputs, write_final_report

logger = setup_logging()

EventCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]


@dataclass
class DocumentResult:
    source_file: str
    status: str
    case: CaseRecord | None = None
    email: CustomerEmail | None = None
    summary: CaseSummary | None = None
    error: str | None = None


def notify(callback: EventCallback | None, message: str) -> None:
    logger.info(message)
    if callback:
        callback(message)


def process_folder(
    data_dir: Path,
    output_dir: Path,
    client: LLMClient,
    on_event: EventCallback | None = None,
    on_progress: Optional[ProgressCallback] = None,
) -> list[DocumentResult]:
    notify(on_event, f"Scanning input folder: {data_dir}")
    files = list_documents(data_dir)
    results: list[DocumentResult] = []
    if not files:
        notify(on_event, f"No supported documents found in {data_dir}")
        write_final_report(output_dir, results)
        return results
    notify(on_event, f"Found {len(files)} document(s) in {data_dir}")

    for index, path in enumerate(files, start=1):
        notify(on_event, f"[{index}/{len(files)}] Starting {path.name}")
        results.append(process_one_document(path, output_dir, client, on_event))
        if on_progress:
            on_progress(index, len(files) or 1)

    write_final_report(output_dir, results)
    notify(on_event, f"Wrote consolidated report to {output_dir / 'final_report.csv'}")
    return results


def process_one_document(
    path: Path,
    output_dir: Path,
    client: LLMClient,
    on_event: EventCallback | None = None,
) -> DocumentResult:
    try:
        text = extract_text(path)
        notify(on_event, f"{path.name}: extracted {len(text)} characters")
    except (OSError, DocumentLoadError) as exc:
        notify(on_event, f"{path.name}: failed to read file ({exc})")
        return DocumentResult(source_file=path.name, status="load_error", error=str(exc))

    try:
        notify(on_event, f"{path.name}: step 1/3 structured extraction")
        case = client.generate_structured(
            CaseRecord,
            prompts.EXTRACTION_SYSTEM,
            prompts.EXTRACTION_USER.format(filename=path.name, document_text=text),
        )

        case_json = case.model_dump_json(indent=2)
        notify(on_event, f"{path.name}: step 2/3 and 3/3 email + internal summary (parallel)")

        with ThreadPoolExecutor(max_workers=2) as pool:
            email_future = pool.submit(
                client.generate_structured,
                CustomerEmail,
                prompts.EMAIL_SYSTEM,
                prompts.EMAIL_USER.format(case_json=case_json),
            )
            summary_future = pool.submit(
                client.generate_structured,
                CaseSummary,
                prompts.SUMMARY_SYSTEM,
                prompts.SUMMARY_USER.format(case_json=case_json),
            )
            email = email_future.result()
            summary = summary_future.result()

        save_case_outputs(output_dir, path.stem, case, email, summary)
        notify(on_event, f"{path.name}: saved outputs")
        return DocumentResult(
            source_file=path.name,
            status="processed",
            case=case,
            email=email,
            summary=summary,
        )
    except LLMError as exc:
        notify(on_event, f"{path.name}: LLM step failed ({exc})")
        return DocumentResult(source_file=path.name, status="llm_error", error=str(exc))
    except Exception as exc:
        notify(on_event, f"{path.name}: unexpected error ({exc})")
        return DocumentResult(source_file=path.name, status="error", error=str(exc))
