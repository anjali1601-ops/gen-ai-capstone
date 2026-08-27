"""Read text from local .txt, .pdf, and .docx files."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


class DocumentLoadError(Exception):
    pass


def list_documents(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")
    files = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".txt":
            return _read_txt(path)
        if suffix == ".pdf":
            return _read_pdf(path)
        if suffix == ".docx":
            return _read_docx(path)
        raise DocumentLoadError(f"Unsupported file type: {path.name}")
    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(f"Could not read {path.name}: {exc}") from exc


def _read_txt(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise DocumentLoadError(f"{path.name} is empty")
    return text


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n".join(pages).strip()
    if not text:
        raise DocumentLoadError(f"No readable text found in {path.name}")
    return text


def _read_docx(path: Path) -> str:
    document = Document(str(path))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
    if not text:
        raise DocumentLoadError(f"No readable text found in {path.name}")
    return text
