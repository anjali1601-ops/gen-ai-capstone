"""Keyword retrieval over local complaint files and batch outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.loaders import DocumentLoadError, extract_text, list_documents
from src.output_io import EMAILS_DIR, STRUCTURED_DIR, SUMMARIES_DIR, read_text_file

_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "the",
    "and",
    "for",
    "are",
    "was",
    "were",
    "this",
    "that",
    "with",
    "from",
    "have",
    "has",
    "had",
    "not",
    "but",
    "you",
    "your",
    "our",
    "their",
    "what",
    "when",
    "where",
    "which",
    "who",
    "how",
    "why",
    "can",
    "could",
    "would",
    "should",
    "please",
    "about",
    "into",
    "over",
    "also",
    "any",
    "all",
}


@dataclass(frozen=True)
class RetrievedSource:
    name: str
    text: str
    score: float


def tokenize(text: str) -> set[str]:
    tokens = {match.group(0).lower() for match in _TOKEN.finditer(text)}
    return {token for token in tokens if len(token) > 2 and token not in _STOPWORDS}


def _add_file(corpus: list[tuple[str, str]], name: str, text: str | None) -> None:
    cleaned = (text or "").strip()
    if cleaned:
        corpus.append((name, cleaned))


def build_corpus(input_dir: Path, output_dir: Path) -> list[tuple[str, str]]:
    """Index input complaints and generated output artefacts."""
    corpus: list[tuple[str, str]] = []
    try:
        for path in list_documents(input_dir):
            try:
                _add_file(corpus, path.name, extract_text(path))
            except (OSError, DocumentLoadError):
                continue
    except (FileNotFoundError, OSError):
        pass

    for folder, suffix, prefix in (
        (EMAILS_DIR, ".txt", "customer email"),
        (SUMMARIES_DIR, ".txt", "case summary"),
        (STRUCTURED_DIR, ".json", "structured data"),
    ):
        directory = output_dir / folder
        if not directory.is_dir():
            continue
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        for path in entries:
            if not path.is_file() or path.suffix.lower() != suffix:
                continue
            label = f"{prefix}: {path.name}"
            _add_file(corpus, label, read_text_file(path))

    report = output_dir / "final_report.csv"
    _add_file(corpus, "final_report.csv", read_text_file(report))
    return corpus


def retrieve(
    query: str,
    corpus: list[tuple[str, str]],
    top_k: int = 4,
) -> list[RetrievedSource]:
    query_tokens = tokenize(query)
    if not query_tokens or not corpus:
        return []

    tokenized: list[tuple[str, str, set[str]]] = []
    document_frequency: dict[str, int] = {}
    for name, text in corpus:
        tokens = tokenize(f"{name} {text}")
        tokenized.append((name, text, tokens))
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    doc_count = len(corpus)
    phrase = " ".join(query.lower().split())
    ranked: list[RetrievedSource] = []
    for name, text, tokens in tokenized:
        haystack = f"{name}\n{text}".lower()
        score = 0.0
        rare_hits = 0
        for token in query_tokens:
            if token not in tokens:
                continue
            common = document_frequency.get(token, 0) / doc_count > 0.5
            if common:
                score += 0.15
            else:
                score += 1.0
                rare_hits += 1
        if phrase and len(phrase) > 5 and phrase in haystack:
            score += 2.0
            rare_hits += 1
        if ":" not in name:
            score += 0.35
        if rare_hits <= 0:
            continue
        ranked.append(RetrievedSource(name=name, text=text, score=score))

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:top_k]
