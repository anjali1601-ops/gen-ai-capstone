"""CLI entry point. Shares the batch workflow with the Streamlit UI in app.py."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from src import config
from src.llm import LLMClient
from src.pipeline import DocumentResult, process_folder

EventCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]

__all__ = ["DocumentResult", "default_folders", "process_folder", "run_batch", "main"]


def default_folders() -> tuple[Path, Path]:
    """Project data/ and output/ folders used by the evaluation UI and CLI."""
    root = config.PROJECT_ROOT.resolve()
    return (root / "data").resolve(), (root / "output").resolve()


def run_batch(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    on_event: EventCallback | None = None,
    on_progress: Optional[ProgressCallback] = None,
) -> list[DocumentResult]:
    """Run the Project 1 folder workflow (extract, then email + summary)."""
    config.reload_env()
    data_dir, out_dir = default_folders()
    if input_dir is not None:
        data_dir = Path(input_dir)
    if output_dir is not None:
        out_dir = Path(output_dir)
    client = LLMClient(provider=provider, model=model)
    if on_event:
        on_event(f"Provider: {client.provider} | Model: {client.model}")
        on_event(f"Input: {data_dir}")
        on_event(f"Output: {out_dir}")
    return process_folder(
        data_dir,
        out_dir,
        client,
        on_event=on_event,
        on_progress=on_progress,
    )


def main() -> None:
    _, output_dir = default_folders()
    results = run_batch(on_event=print)
    processed = sum(1 for item in results if item.status == "processed")
    print(f"Done. {processed}/{len(results)} document(s) processed.")
    print(f"Report: {output_dir / 'final_report.csv'}")


if __name__ == "__main__":
    main()
