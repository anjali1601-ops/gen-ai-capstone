"""IIT Patna USDC GenAI — Project 1 evaluation window.

Wraps ``src.pipeline.process_folder`` (same engine as ``main.py``).
Keys load from ``.env`` and are never shown.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.loaders import list_documents
from src.llm import LLMClient, LLMError
from src.logging_setup import setup_logging
from src.models import CaseSummary, CustomerEmail
from src.output_io import (
    SUMMARIES_DIR,
    artefact_path,
    list_source_files,
    load_case_summary,
    load_customer_email,
    read_final_report,
    read_text_file,
    report_path,
    row_for_source,
)
from src.pipeline import process_folder
from src.retrieve import RetrievedSource, build_corpus, retrieve

logger = setup_logging()

PAGE_TITLE = "IIT Patna · Complaint Case Processor"
NOT_FOUND_REPLY = (
    "That information was not found in the local complaint files or generated outputs."
)
CHAT_SYSTEM = (
    "You answer questions about customer complaints using ONLY the retrieved sources. "
    "If the sources do not contain the answer, say the information was not found. "
    "Do not invent names, dates, amounts, or outcomes. Keep the reply concise."
)


def project_folders() -> tuple[Path, Path]:
    root = config.PROJECT_ROOT.resolve()
    return (root / "data").resolve(), (root / "output").resolve()


def init_session_state() -> None:
    defaults: dict[str, object] = {
        "logs": [],
        "just_finished": False,
        "last_error": None,
        "chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _unique_models(*names: str) -> list[str]:
    options: list[str] = []
    seen: set[str] = set()
    for name in names:
        cleaned = name.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            options.append(cleaned)
    return options


def _model_options(provider: str) -> list[str]:
    if provider == "openai":
        return _unique_models(config.OPENAI_MODEL, "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini")
    return _unique_models(
        config.GEMINI_MODEL,
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-2.5-flash",
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2.8rem; max-width: 1200px; }
        .hero-kicker {
            font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em;
            text-transform: uppercase; color: #8c1d18; margin: 0 0 0.25rem 0;
        }
        .hero-title {
            font-size: 1.7rem; font-weight: 750; color: #13294b;
            line-height: 1.25; margin: 0 0 0.85rem 0;
        }
        section[data-testid="stSidebar"] {
            background: #f7f8fa;
        }
        section[data-testid="stSidebar"] .settings-section {
            font-size: 0.74rem; font-weight: 700; letter-spacing: 0.07em;
            text-transform: uppercase; color: rgba(49, 51, 63, 0.55);
            margin: 0.15rem 0 0.45rem 0;
        }
        div[data-testid="stAppViewContainer"] .stButton > button[kind="primary"] {
            min-height: 3.4rem; font-size: 1.14rem; font-weight: 700;
            letter-spacing: 0.01em; background: #13294b; border: 0;
        }
        div[data-testid="stAppViewContainer"] .stButton > button[kind="primary"]:hover {
            background: #1c3d6e; border: 0;
        }
        .detail-card {
            background: #f4f7fb;
            border: 1px solid #d5deea;
            border-radius: 0.75rem;
            padding: 1rem 1.1rem 1.15rem 1.1rem;
            min-height: 280px;
        }
        .detail-card h4 {
            margin: 0 0 0.7rem 0;
            font-size: 0.92rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #13294b;
        }
        .detail-card pre {
            white-space: pre-wrap;
            word-break: break-word;
            font-family: "Source Sans Pro", sans-serif;
            font-size: 0.92rem;
            line-height: 1.5;
            color: rgba(19, 41, 75, 0.88);
            margin: 0;
        }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(input_dir: Path, output_dir: Path) -> tuple[str, str]:
    with st.sidebar:
        st.markdown(
            '<p class="settings-section">Configuration</p>',
            unsafe_allow_html=True,
        )
        st.text_input(
            "📁 Input Folder",
            value=str(input_dir),
            disabled=True,
            key="input_folder_display",
        )
        st.text_input(
            "📁 Output Folder",
            value=str(output_dir),
            disabled=True,
            key="output_folder_display",
        )
        st.caption("Resolved from the project root.")

        st.divider()
        st.markdown('<p class="settings-section">Model</p>', unsafe_allow_html=True)
        provider_label = st.segmented_control(
            "Provider",
            options=["OpenAI", "Gemini"],
            default="OpenAI" if config.LLM_PROVIDER != "gemini" else "Gemini",
            key="settings_provider",
            help="Used for batch processing and grounded chat.",
        )
        if provider_label is None:
            provider_label = "OpenAI" if config.LLM_PROVIDER != "gemini" else "Gemini"
        provider = "gemini" if provider_label == "Gemini" else "openai"
        default_model = config.OPENAI_MODEL if provider == "openai" else config.GEMINI_MODEL
        model = st.selectbox(
            "Model",
            options=_model_options(provider),
            index=0,
            accept_new_options=True,
            key=f"settings_model_{provider}",
            help="Choose a preset or type another model name.",
        )
        model = str(model or "").strip() or default_model

    return provider, model


def display_value(value: object) -> str:
    if value is None or value == "":
        return "—"
    return str(value)


def _input_files(data_dir: Path) -> tuple[list[Path] | None, str | None]:
    if not data_dir.exists():
        return None, f"The input folder was not found: `{data_dir}`"
    if not data_dir.is_dir():
        return None, f"The input path is not a folder: `{data_dir}`"
    try:
        return list_documents(data_dir), None
    except FileNotFoundError:
        return None, f"The input folder was not found: `{data_dir}`"
    except OSError as exc:
        return None, f"Could not read the input folder: {exc}"


def run_batch(provider: str, model: str, data_dir: Path, output_dir: Path) -> None:
    logs: list[str] = []
    st.session_state.just_finished = False
    st.session_state.last_error = None
    st.session_state.logs = []
    progress = st.progress(0)
    log_box = st.empty()

    def on_event(message: str) -> None:
        logs.append(message)
        log_box.code("\n".join(logs), language="text")

    def on_progress(done: int, total: int) -> None:
        fraction = min(done / total, 1.0) if total else 1.0
        progress.progress(fraction)

    try:
        with st.spinner("Processing business documents, please wait..."):
            client = LLMClient(provider=provider, model=model)
            on_event(f"Provider: {client.provider} | Model: {client.model}")
            results = process_folder(
                data_dir,
                output_dir,
                client,
                on_event=on_event,
                on_progress=on_progress,
            )
        processed = sum(1 for item in results if item.status == "processed")
        on_event(f"Done. {processed}/{len(results)} document(s) processed.")
        progress.progress(1.0)
        st.session_state.just_finished = True
        st.session_state.last_error = None
    except LLMError as exc:
        logger.exception("Batch processing failed (LLM)")
        on_event(f"Error: {exc}")
        st.session_state.last_error = str(exc)
        st.error(str(exc))
    except (FileNotFoundError, OSError) as exc:
        logger.exception("Batch processing failed (filesystem)")
        on_event(f"Error: {exc}")
        st.session_state.last_error = str(exc)
        st.error(str(exc))
    except Exception as exc:
        logger.exception("Batch processing failed")
        on_event(f"Error: {exc}")
        st.session_state.last_error = f"Unexpected error: {exc}"
        st.error(st.session_state.last_error)

    st.session_state.logs = logs


def _email_block_text(email: CustomerEmail | None, raw: str | None) -> str:
    if email is not None:
        subject = display_value(email.subject)
        body = (email.body or "").strip() or "—"
        return f"Subject: {subject}\n\n{body}"
    if raw:
        return raw.strip()
    return "No customer email was saved for this document."


def _brief_block_text(summary: CaseSummary | None, raw: str | None) -> str:
    if summary is not None:
        return "\n".join(
            [
                f"Case overview: {display_value(summary.case_overview)}",
                f"Key issue: {display_value(summary.key_issue)}",
                f"Action taken: {display_value(summary.action_taken)}",
                f"Current status: {display_value(summary.current_status)}",
                f"Recommended next action: {display_value(summary.recommended_next_action)}",
            ]
        )
    if raw:
        return raw.strip()
    return "No management brief was saved for this document."


def _render_detail_card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="detail-card"><h4>{html.escape(title)}</h4>'
        f"<pre>{html.escape(body)}</pre></div>",
        unsafe_allow_html=True,
    )


def render_document_details(output_dir: Path, report: pd.DataFrame | None) -> None:
    names = list_source_files(output_dir, report)
    if not names:
        st.info("No generated documents are available yet. Run batch processing first.")
        return

    selected = st.selectbox(
        "Select Document to View Details",
        options=names,
        key="detail_document",
    )
    row = row_for_source(report, selected)
    if row:
        status = row.get("status", "") or "unknown"
        st.caption(f"Status: **{status}**")
        error_text = (row.get("error") or "").strip()
        if error_text:
            st.error(error_text)

    email = load_customer_email(output_dir, selected)
    summary = load_case_summary(output_dir, selected)
    summary_raw = read_text_file(artefact_path(output_dir, SUMMARIES_DIR, selected, ".txt"))
    email_col, brief_col = st.columns(2)
    with email_col:
        _render_detail_card("Customer Response Email", _email_block_text(email, None))
    with brief_col:
        _render_detail_card("Management Brief", _brief_block_text(summary, summary_raw))


def render_dashboard(
    provider: str,
    model: str,
    input_dir: Path,
    output_dir: Path,
) -> None:
    files, input_error = _input_files(input_dir)
    left, center, right = st.columns([1, 2, 1])
    with center:
        run_clicked = st.button(
            "🚀 Run Batch Processing",
            type="primary",
            use_container_width=True,
            key="run_batch",
            help="Runs the extract → email → summary pipeline on every file in the input folder.",
        )

    if files:
        st.caption("Ready: " + ", ".join(path.name for path in files))
    elif input_error:
        st.error(input_error)
    else:
        st.warning("No .txt, .pdf, or .docx files found in the input folder.")

    if run_clicked:
        if input_error:
            st.session_state.last_error = input_error
            st.error(input_error)
        else:
            run_batch(provider, model, input_dir, output_dir)

    if st.session_state.last_error and not run_clicked:
        st.error(st.session_state.last_error)

    csv_file = report_path(output_dir)
    report = read_final_report(output_dir)

    if st.session_state.just_finished:
        if report is None:
            st.warning(
                "Processing finished, but `final_report.csv` could not be read "
                f"from `{output_dir}`."
            )
        else:
            processed = 0
            if "status" in report.columns:
                processed = int((report["status"] == "processed").sum())
            st.success(
                f"Batch complete. {processed} of {len(report)} document(s) processed. "
                f"Saved to `{csv_file}`."
            )
            if processed < len(report):
                st.warning("Some files did not finish successfully. See the report table.")

    if report is None:
        if not st.session_state.just_finished:
            st.caption("Click **Run Batch Processing** to generate `final_report.csv`.")
        if st.session_state.logs:
            with st.expander("Processing log", expanded=False):
                st.code("\n".join(st.session_state.logs), language="text")
        return

    if report.empty:
        st.warning("The report has no document rows. The input folder may have been empty.")

    st.dataframe(report, use_container_width=True, hide_index=True, height=360)
    render_document_details(output_dir, report)

    if st.session_state.logs:
        with st.expander("Processing log", expanded=False):
            st.code("\n".join(st.session_state.logs), language="text")


def _format_sources_for_prompt(sources: list[RetrievedSource]) -> str:
    blocks = []
    for item in sources:
        excerpt = item.text.strip()
        if len(excerpt) > 3500:
            excerpt = excerpt[:3500] + "\n…"
        blocks.append(f"SOURCE: {item.name}\n{excerpt}")
    return "\n\n".join(blocks)


def answer_chat_question(question: str, provider: str, model: str, input_dir: Path, output_dir: Path) -> None:
    history: list[dict] = list(st.session_state.chat_history)
    history.append({"role": "user", "content": question, "sources": []})

    corpus = build_corpus(input_dir, output_dir)
    sources = retrieve(question, corpus, top_k=6)
    if not sources:
        history.append(
            {
                "role": "assistant",
                "content": NOT_FOUND_REPLY,
                "sources": [],
            }
        )
        st.session_state.chat_history = history
        return

    prior = []
    for turn in history[:-1][-6:]:
        prior.append(f"{turn['role'].upper()}: {turn['content']}")
    prior_text = "\n".join(prior) if prior else "(none)"
    user_prompt = (
        f"Conversation so far:\n{prior_text}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved sources:\n{_format_sources_for_prompt(sources)}"
    )
    try:
        client = LLMClient(provider=provider, model=model)
        answer = client.generate_text(CHAT_SYSTEM, user_prompt).strip()
        if not answer:
            answer = NOT_FOUND_REPLY
    except LLMError as exc:
        answer = f"The chat model could not answer: {exc}"
        sources = []

    history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [item.name for item in sources],
        }
    )
    st.session_state.chat_history = history


def render_chat(provider: str, model: str, input_dir: Path, output_dir: Path) -> None:
    st.caption(
        "Ask about local complaint files and generated emails, summaries, or structured records. "
        "Answers are grounded in retrieved sources."
    )
    if st.button("Clear conversation", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

    history: list[dict] = st.session_state.chat_history
    if not history:
        st.info("No conversation yet. Type a question below.")
    for turn in history:
        role = "user" if turn.get("role") == "user" else "assistant"
        with st.chat_message(role):
            st.write(turn.get("content") or "")
            if role == "assistant":
                source_names = turn.get("sources") or []
                if source_names:
                    st.caption("Retrieved sources: " + ", ".join(source_names))
                else:
                    st.caption("Retrieved sources: none — information was not found in local files.")

    prompt = st.chat_input("Ask about a complaint, email, or case summary")
    if prompt and prompt.strip():
        answer_chat_question(prompt.strip(), provider, model, input_dir, output_dir)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    config.reload_env()
    _inject_styles()
    init_session_state()

    input_dir, output_dir = project_folders()
    st.markdown(
        '<p class="hero-kicker">IIT Patna · USDC GenAI Development Program · Final Evaluation</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-title">AI Customer Complaint &amp; Case Processing</p>',
        unsafe_allow_html=True,
    )
    provider, model = render_sidebar(input_dir, output_dir)

    dashboard_tab, chat_tab = st.tabs(["Dashboard", "Chat"])
    with dashboard_tab:
        render_dashboard(provider, model, input_dir, output_dir)
    with chat_tab:
        render_chat(provider, model, input_dir, output_dir)


main()
