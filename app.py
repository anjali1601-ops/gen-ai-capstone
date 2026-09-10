"""Complaint Case Processor — Streamlit operations console.

Wraps ``src.pipeline.process_folder`` (same engine as ``main.py``).
API keys load from ``.env`` and are never rendered.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.auth import (
    MAX_LOGIN_ATTEMPTS,
    AuthUser,
    authenticate,
    directory_rows,
    load_user_store,
)
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

PAGE_TITLE = "Complaint Case Processor"
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
        "chat_histories": {},
        "auth_user": None,
        "login_error": None,
        "login_attempts": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_user() -> AuthUser | None:
    user = st.session_state.get("auth_user")
    if isinstance(user, AuthUser):
        return user
    return None


def _chat_history(username: str) -> list[dict]:
    store = st.session_state.chat_histories
    if username not in store:
        store[username] = []
    return store[username]


def _set_chat_history(username: str, history: list[dict]) -> None:
    st.session_state.chat_histories[username] = history


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


def _default_provider_model() -> tuple[str, str]:
    provider = config.LLM_PROVIDER if config.LLM_PROVIDER == "gemini" else "openai"
    model = config.OPENAI_MODEL if provider == "openai" else config.GEMINI_MODEL
    return provider, (model.strip() or ("gpt-4o-mini" if provider == "openai" else "gemini-2.0-flash"))


def _inject_styles(*, login: bool = False) -> None:
    hide_sidebar = ""
    if login:
        hide_sidebar = """
        section[data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
        """
    st.markdown(
        f"""
        <style>
        :root {{
          --navy: #0b1c33;
          --navy-2: #132844;
          --slate: #5b6b7c;
          --line: #c5d0dc;
          --ice: #e8eef4;
          --paper: #f5f7fa;
          --ok: #1f7a4d;
          --bad: #b42318;
        }}
        html, body, [data-testid="stAppViewContainer"] {{
          font-family: "Segoe UI", "Source Sans 3", Calibri, Arial, sans-serif;
          color: #122033;
        }}
        .block-container {{
          padding-top: 1.15rem;
          padding-bottom: 2.4rem;
          max-width: 1180px;
        }}
        header[data-testid="stHeader"] {{ background: var(--paper); }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        [data-testid="stToolbar"] {{ visibility: hidden; }}
        {hide_sidebar}

        .app-mast {{
          display: flex;
          justify-content: space-between;
          align-items: flex-end;
          gap: 1.5rem;
          padding: 0 0 0.95rem 0;
          margin: 0 0 1.05rem 0;
          border-bottom: 1px solid var(--line);
        }}
        .app-kicker {{
          margin: 0 0 0.28rem 0;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          color: #3d6f8f;
        }}
        .app-mast h1 {{
          margin: 0;
          font-size: 1.55rem;
          font-weight: 650;
          letter-spacing: -0.02em;
          color: var(--navy);
          line-height: 1.15;
        }}
        .app-tag {{
          margin: 0.38rem 0 0 0;
          font-size: 0.95rem;
          line-height: 1.4;
          color: var(--slate);
          max-width: 42rem;
        }}
        .app-meta {{
          text-align: right;
          font-size: 0.78rem;
          letter-spacing: 0.04em;
          color: var(--slate);
          white-space: nowrap;
          padding-bottom: 0.15rem;
        }}

        section[data-testid="stSidebar"] {{
          background: #eef3f7;
          border-right: 1px solid var(--line);
        }}
        section[data-testid="stSidebar"] .block-container {{ padding-top: 1.1rem; }}
        .settings-section {{
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #3d6f8f;
          margin: 0.1rem 0 0.55rem 0;
        }}
        .settings-note {{
          font-size: 0.78rem;
          line-height: 1.4;
          color: var(--slate);
          margin: 0.35rem 0 0 0;
        }}
        .identity-name {{
          margin: 0;
          font-size: 1.02rem;
          font-weight: 650;
          color: var(--navy);
        }}
        .identity-role {{
          margin: 0.2rem 0 0.55rem 0;
          font-size: 0.78rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--slate);
        }}
        .login-hint {{
          font-size: 0.8rem;
          color: var(--slate);
          margin: 0.75rem 0 0 0;
          line-height: 1.45;
        }}

        .section-label {{
          margin: 1.15rem 0 0.45rem 0;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #3d6f8f;
        }}

        div[data-testid="stAppViewContainer"] .stButton > button[kind="primary"] {{
          min-height: 2.85rem;
          font-size: 0.98rem;
          font-weight: 650;
          letter-spacing: 0.02em;
          background: var(--navy);
          color: #f7fbff;
          border: 1px solid var(--navy);
          border-radius: 4px;
        }}
        div[data-testid="stAppViewContainer"] .stButton > button[kind="primary"]:hover {{
          background: var(--navy-2);
          border-color: var(--navy-2);
        }}
        div[data-testid="stAppViewContainer"] .stButton > button[kind="secondary"] {{
          border-radius: 4px;
          border-color: var(--line);
        }}

        .queue-line {{
          font-size: 0.86rem;
          color: var(--slate);
          margin: 0.35rem 0 0.15rem 0;
        }}
        .queue-line code {{
          font-size: 0.8rem;
          color: var(--navy);
        }}

        .empty-state {{
          border: 1px solid var(--line);
          background: #fbfcfd;
          border-radius: 4px;
          padding: 1.15rem 1.2rem;
        }}
        .empty-state p {{ margin: 0; }}
        .empty-state .empty-title {{
          font-weight: 650;
          color: var(--navy);
          margin-bottom: 0.3rem;
        }}
        .empty-state .empty-body {{ color: var(--slate); font-size: 0.92rem; line-height: 1.45; }}

        .status-pill {{
          display: inline-block;
          font-family: Consolas, "Cascadia Mono", monospace;
          font-size: 0.78rem;
          font-weight: 650;
          letter-spacing: 0.02em;
          padding: 0.12rem 0.45rem;
          border-radius: 3px;
          border: 1px solid transparent;
        }}
        .status-pill.ok {{ color: var(--ok); background: #e8f6ee; border-color: #b7e0c7; }}
        .status-pill.bad {{ color: var(--bad); background: #fdecea; border-color: #f0c2bd; }}

        .detail-card {{
          background: #fbfcfd;
          border: 1px solid var(--line);
          border-radius: 4px;
          padding: 0.85rem 0.95rem 1rem;
          min-height: 268px;
        }}
        .detail-card h4 {{
          margin: 0 0 0.65rem 0;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: #3d6f8f;
        }}
        .detail-card pre {{
          white-space: pre-wrap;
          word-break: break-word;
          font-family: Consolas, "Cascadia Mono", "SF Mono", monospace;
          font-size: 0.82rem;
          line-height: 1.5;
          color: #1a2a3a;
          margin: 0;
        }}

        [data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: 4px; }}
        [data-testid="stMetric"] {{
          background: #fbfcfd;
          border: 1px solid var(--line);
          border-radius: 4px;
          padding: 0.45rem 0.7rem;
        }}
        [data-testid="stMetricLabel"] {{ color: var(--slate); }}
        [data-testid="stMetricValue"] {{ color: var(--navy); }}

        .stTabs [data-baseweb="tab-list"] {{
          gap: 0.25rem;
          border-bottom: 1px solid var(--line);
        }}
        .stTabs [data-baseweb="tab"] {{
          font-weight: 650;
          letter-spacing: 0.02em;
          color: var(--slate);
        }}
        .stTabs [aria-selected="true"] {{ color: var(--navy); }}

        [data-testid="stChatMessage"] {{
          border: 1px solid var(--line);
          background: #fbfcfd;
          border-radius: 4px;
          padding: 0.35rem 0.2rem;
        }}
        .source-line {{
          font-family: Consolas, "Cascadia Mono", monospace;
          font-size: 0.75rem;
          color: var(--slate);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_masthead(user: AuthUser | None = None) -> None:
    if user is None:
        meta = "Local sign-in · keys never displayed"
    else:
        meta = f"{html.escape(user.name)} · {html.escape(user.role_label)}"
    st.markdown(
        f"""
        <div class="app-mast">
          <div>
            <p class="app-kicker">IIT Patna · USDC GenAI · Project 1</p>
            <h1>Complaint Case Processor</h1>
            <p class="app-tag">Local batch workflow for structured cases, customer email, and internal summaries.</p>
          </div>
          <p class="app-meta">{meta}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_login() -> None:
    _render_masthead()
    store = load_user_store()
    left, panel, right = st.columns([1, 1.25, 1])
    with panel:
        st.markdown('<p class="section-label">Sign in</p>', unsafe_allow_html=True)
        if store.error:
            st.error(store.error)
        locked = int(st.session_state.login_attempts) >= MAX_LOGIN_ATTEMPTS
        with st.form("sign_in", clear_on_submit=False):
            username = st.text_input("Username", disabled=locked or bool(store.error))
            password = st.text_input(
                "Password",
                type="password",
                disabled=locked or bool(store.error),
            )
            submitted = st.form_submit_button(
                "Sign in",
                type="primary",
                use_container_width=True,
                disabled=locked or bool(store.error),
            )
        if locked:
            st.error("Too many failed attempts in this session. Restart the app to try again.")
        elif submitted:
            if not username.strip() or password == "":
                st.session_state.login_error = "Enter your username and password."
            else:
                user = authenticate(username, password, store)
                if user is None:
                    st.session_state.login_attempts = int(st.session_state.login_attempts) + 1
                    remaining = MAX_LOGIN_ATTEMPTS - int(st.session_state.login_attempts)
                    if remaining <= 0:
                        st.session_state.login_error = None
                        st.rerun()
                    st.session_state.login_error = "Invalid username or password."
                else:
                    st.session_state.auth_user = user
                    st.session_state.login_error = None
                    st.session_state.login_attempts = 0
                    st.rerun()
        if st.session_state.login_error:
            st.error(st.session_state.login_error)
        st.markdown(
            '<p class="login-hint">Local evaluation accounts only. '
            "Demo usernames and passwords are listed in README.md. "
            "API keys are never shown.</p>",
            unsafe_allow_html=True,
        )


def _logout() -> None:
    st.session_state.auth_user = None
    st.session_state.login_error = None
    st.session_state.just_finished = False
    st.session_state.last_error = None
    st.session_state.logs = []
    st.rerun()


def render_sidebar(user: AuthUser, input_dir: Path, output_dir: Path) -> tuple[str, str]:
    with st.sidebar:
        st.markdown('<p class="settings-section">Signed in</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="identity-name">{html.escape(user.name)}</p>'
            f'<p class="identity-role">{html.escape(user.role_label)}</p>',
            unsafe_allow_html=True,
        )
        if st.button("Log out", use_container_width=True, key="logout"):
            _logout()

        st.divider()
        st.markdown('<p class="settings-section">Configuration</p>', unsafe_allow_html=True)
        st.text_input(
            "Input folder",
            value=str(input_dir),
            disabled=True,
            key="input_folder_display",
        )
        st.text_input(
            "Output folder",
            value=str(output_dir),
            disabled=True,
            key="output_folder_display",
        )
        st.markdown(
            '<p class="settings-note">Resolved from the project root. '
            "Live artefacts write to <code>output/</code>.</p>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown('<p class="settings-section">Model</p>', unsafe_allow_html=True)
        if user.is_admin:
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
        else:
            provider, model = _default_provider_model()
            st.text_input(
                "Provider",
                value="Gemini" if provider == "gemini" else "OpenAI",
                disabled=True,
                key="readonly_provider",
            )
            st.text_input(
                "Model",
                value=model,
                disabled=True,
                key="readonly_model",
            )
            st.markdown(
                '<p class="settings-note">Provider and model are set by an administrator.</p>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<p class="settings-note">Keys load from <code>.env</code> and are never shown.</p>',
            unsafe_allow_html=True,
        )

        if user.is_admin:
            st.divider()
            st.markdown('<p class="settings-section">Directory</p>', unsafe_allow_html=True)
            rows = directory_rows()
            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                    height=min(220, 52 + 36 * len(rows)),
                )
            else:
                st.caption("No directory entries could be loaded.")

    return provider, model


def display_value(value: object) -> str:
    if value is None or value == "":
        return "—"
    return str(value)


def _input_files(data_dir: Path) -> tuple[list[Path] | None, str | None]:
    if not data_dir.exists():
        return None, f"Input folder not found: `{data_dir}`"
    if not data_dir.is_dir():
        return None, f"Input path is not a folder: `{data_dir}`"
    try:
        return list_documents(data_dir), None
    except FileNotFoundError:
        return None, f"Input folder not found: `{data_dir}`"
    except OSError as exc:
        return None, f"Could not read the input folder: {exc}"


def _empty_state(title: str, body: str) -> None:
    st.markdown(
        f'<div class="empty-state"><p class="empty-title">{html.escape(title)}</p>'
        f'<p class="empty-body">{html.escape(body)}</p></div>',
        unsafe_allow_html=True,
    )


def _status_pill(status: str) -> str:
    cls = "ok" if status == "processed" else "bad"
    return f'<span class="status-pill {cls}">{html.escape(status)}</span>'


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
        with st.spinner("Running batch…"):
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
        on_event(f"Complete. {processed}/{len(results)} document(s) processed.")
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
        _empty_state(
            "No document artefacts yet",
            "An administrator must run the batch to generate customer emails and management briefs.",
        )
        return

    st.markdown('<p class="section-label">Document</p>', unsafe_allow_html=True)
    selected = st.selectbox(
        "Source file",
        options=names,
        key="detail_document",
        help="Opens the saved customer email and internal summary for this file.",
    )
    row = row_for_source(report, selected)
    if row:
        status = str(row.get("status", "") or "unknown")
        st.markdown(
            f'Status {_status_pill(status)}',
            unsafe_allow_html=True,
        )
        error_text = (row.get("error") or "").strip()
        if error_text:
            st.error(error_text)

    email = load_customer_email(output_dir, selected)
    summary = load_case_summary(output_dir, selected)
    summary_raw = read_text_file(artefact_path(output_dir, SUMMARIES_DIR, selected, ".txt"))
    email_col, brief_col = st.columns(2)
    with email_col:
        _render_detail_card("Customer response email", _email_block_text(email, None))
    with brief_col:
        _render_detail_card("Management brief", _brief_block_text(summary, summary_raw))


def _queue_caption(files: list[Path]) -> str:
    names = [path.name for path in files]
    shown = ", ".join(names[:8])
    extra = f" +{len(names) - 8} more" if len(names) > 8 else ""
    return f"{len(names)} file(s) queued · {shown}{extra}"


def render_dashboard(
    provider: str,
    model: str,
    input_dir: Path,
    output_dir: Path,
    *,
    is_admin: bool,
) -> None:
    files, input_error = _input_files(input_dir)

    if is_admin:
        st.markdown('<p class="section-label">Batch</p>', unsafe_allow_html=True)
        left, center, right = st.columns([1, 2, 1])
        with center:
            run_clicked = st.button(
                "Run batch",
                type="primary",
                use_container_width=True,
                key="run_batch",
                help="Runs extract → email + summary on every supported file in the input folder.",
            )

        if files:
            st.markdown(
                f'<p class="queue-line">{html.escape(_queue_caption(files))}</p>',
                unsafe_allow_html=True,
            )
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
    else:
        run_clicked = False
        if input_error:
            st.error(input_error)

    csv_file = report_path(output_dir)
    report = read_final_report(output_dir)

    if is_admin and st.session_state.just_finished:
        if report is None:
            st.warning(
                "Batch finished, but `final_report.csv` could not be read "
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
                st.warning("Some files did not finish. Review status and error in the report.")

    if report is None:
        if not st.session_state.just_finished:
            if is_admin:
                _empty_state(
                    "No report yet",
                    "Run the batch to write final_report.csv and per-file artefacts under output/.",
                )
            else:
                _empty_state(
                    "No report yet",
                    "An administrator must run batch processing before results appear here.",
                )
        if is_admin and st.session_state.logs:
            st.markdown('<p class="section-label">Run log</p>', unsafe_allow_html=True)
            with st.expander("Processing log", expanded=False):
                st.code("\n".join(st.session_state.logs), language="text")
        return

    if report.empty:
        st.warning("The report has no document rows. The input folder may have been empty.")

    total = len(report)
    processed = int((report["status"] == "processed").sum()) if "status" in report.columns else 0
    failed = total - processed
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", total)
    m2.metric("Processed", processed)
    m3.metric("Failed", failed)

    st.markdown('<p class="section-label">Report</p>', unsafe_allow_html=True)
    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True,
        height=360,
        column_config={
            "source_file": st.column_config.TextColumn("source_file", width="medium"),
            "status": st.column_config.TextColumn(
                "status",
                help="processed · load_error · llm_error · error",
                width="small",
            ),
            "error": st.column_config.TextColumn("error", width="medium"),
        },
    )
    render_document_details(output_dir, report)

    if is_admin and st.session_state.logs:
        st.markdown('<p class="section-label">Run log</p>', unsafe_allow_html=True)
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


def answer_chat_question(
    question: str,
    provider: str,
    model: str,
    input_dir: Path,
    output_dir: Path,
    username: str,
) -> None:
    history: list[dict] = list(_chat_history(username))
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
        _set_chat_history(username, history)
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
    _set_chat_history(username, history)


def render_chat(
    provider: str,
    model: str,
    input_dir: Path,
    output_dir: Path,
    username: str,
) -> None:
    st.markdown('<p class="section-label">Grounded Q&amp;A</p>', unsafe_allow_html=True)
    st.caption(
        "Questions are answered from retrieved text in data/ and output/ only. "
        "The model is not allowed to invent facts."
    )
    if st.button("Clear conversation", key="clear_chat"):
        _set_chat_history(username, [])
        st.rerun()

    history: list[dict] = _chat_history(username)
    if not history:
        _empty_state(
            "No conversation yet",
            "Ask about a complaint, customer email, or case summary. Retrieved source names appear under each reply.",
        )
    for turn in history:
        role = "user" if turn.get("role") == "user" else "assistant"
        with st.chat_message(role):
            st.write(turn.get("content") or "")
            if role == "assistant":
                source_names = turn.get("sources") or []
                if source_names:
                    st.markdown(
                        f'<p class="source-line">sources · {html.escape(", ".join(source_names))}</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<p class="source-line">sources · none — not found in local files</p>',
                        unsafe_allow_html=True,
                    )

    prompt = st.chat_input("Ask about a complaint, email, or case summary")
    if prompt and prompt.strip():
        answer_chat_question(prompt.strip(), provider, model, input_dir, output_dir, username)
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    config.reload_env()
    init_session_state()
    user = current_user()
    _inject_styles(login=user is None)

    if user is None:
        render_login()
        return

    input_dir, output_dir = project_folders()
    _render_masthead(user)
    provider, model = render_sidebar(user, input_dir, output_dir)

    dashboard_tab, chat_tab = st.tabs(["Dashboard", "Chat"])
    with dashboard_tab:
        render_dashboard(provider, model, input_dir, output_dir, is_admin=user.is_admin)
    with chat_tab:
        render_chat(provider, model, input_dir, output_dir, user.username)


main()
