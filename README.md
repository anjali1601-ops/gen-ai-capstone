# Complaint Case Processor

Local batch workflow that turns customer complaint files into structured case records, customer emails, and manager summaries. Built for IIT Patna USDC GenAI **Project 1**.

Each file is processed in three LLM steps: extract a Pydantic `CaseRecord`, then generate a customer email and an internal summary **in parallel**. The Streamlit console and the CLI share the same engine (`src.pipeline.process_folder`).

## What it does

Support teams receive complaints as `.txt`, `.pdf`, and `.docx`. This product:

1. Reads every supported file in `data/`
2. Extracts a validated `CaseRecord`
3. Writes `customer_emails/*.txt` and `case_summaries/*.txt` from those fields (not a second unrestricted read of the raw file)
4. Consolidates results in `output/final_report.csv`

Optional **Chat** answers questions from retrieved local files and generated artefacts only.

Architecture poster: [docs/pipeline-one-pager.html](docs/pipeline-one-pager.html)

```text
data/*.txt, *.pdf, *.docx
        │
        ▼
 loaders.list_documents / extract_text
        │
        ▼
 process_folder → process_one_document
        │
        ▼
 1/3  generate_structured(CaseRecord)
        │
        │  case.model_dump_json()
        ▼
 2/3–3/3  ThreadPoolExecutor(max_workers=2)
        ├── CustomerEmail
        └── CaseSummary
        │
        ▼
 output/structured_data  customer_emails  case_summaries  final_report.csv
        │
        ├── Streamlit  app.py
        └── CLI        main.py
```

## Stack

- Python 3.11+
- OpenAI or Gemini (Pydantic-validated structured output)
- pypdf, python-docx
- Streamlit operations console with local sign-in (`admin` vs `analyst` / `user`)
- pandas for `final_report.csv`
- File log: `logs/app.log`

## Repository

```text
gen-ai-capstone/
  app.py                      Streamlit console (login, Dashboard + Chat)
  main.py                     CLI entry
  run_batch.py                CLI alias of the same batch
  requirements.txt
  .env.example
  config/users.example.yaml   Demo accounts (copy or first-run bootstrap)
  .streamlit/config.toml      Theme
  data/                       Sample complaints (.txt, .pdf, .docx)
  docs/pipeline-one-pager.html
  sample_output/              Example artefacts from a local run
  scripts/create_sample_files.py
    src/
    pipeline.py               Extract, then email + summary in parallel
    report.py                 JSON, emails, summaries, CSV
    models.py                 CaseRecord, CustomerEmail, CaseSummary
    loaders.py                list_documents / extract_text
    llm.py                    OpenAI / Gemini client with retries
    prompts.py
    retrieve.py               Keyword retrieval for Chat
    output_io.py              Read artefacts for the console
    auth.py                   Local hashed users (no cloud IdP)
    audit.py                  Batch audit log
    review.py                 Draft / approve email flags
    config.py
    logging_setup.py
  tests/                      pytest (no live API calls)
  .github/workflows/ci.yml
```

Runtime folders (gitignored): `output/`, `logs/`. Keys live only in `.env`. Hashed demo accounts write to `config/users.yaml` (gitignored).

## Setup (Windows)

1. Install Python 3.11+ from https://www.python.org/downloads/ and tick **Add python.exe to PATH**.

2. In PowerShell, from this repository:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Copy the example env file and add **one or both** provider keys:

   ```powershell
   copy .env.example .env
   ```

   Edit `.env` locally. Do not commit it.

   ```text
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key
   LLM_PROVIDER=openai
   OPENAI_MODEL=gpt-4o-mini
   GEMINI_MODEL=gemini-2.0-flash
   ```

   If Gemini rejects the default model, try `gemini-1.5-flash` or `gemini-2.5-flash`.

4. Local sign-in uses `config/users.example.yaml`. On first Streamlit launch the app hashes those demo passwords and writes `config/users.yaml` (gitignored). To create that file yourself:

   ```powershell
   copy config\users.example.yaml config\users.yaml
   ```

   The app still hashes plaintext passwords on load. Do not commit `config/users.yaml`.

5. Sample files are already in `data/`. Rebuild Word/PDF samples only if needed:

   ```powershell
   python scripts\create_sample_files.py
   ```

## Environment

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | If using OpenAI | Provider key |
| `GEMINI_API_KEY` | If using Gemini | Provider key |
| `LLM_PROVIDER` | Optional | `openai` (default) or `gemini` |
| `OPENAI_MODEL` | Optional | Default `gpt-4o-mini` |
| `GEMINI_MODEL` | Optional | Default `gemini-2.0-flash` |
| `DATA_DIR` | Optional | Override input folder (default `data/`) |
| `OUTPUT_DIR` | Optional | Override output folder (default `output/`) |
| `APP_SECRET` | Optional | Extra hash pepper. Leave blank for the class demo. If you set it, do so before the first `users.yaml` bootstrap. |

This README does not contain real keys.

## Run

Keep the virtual environment activated. This is a **local** workflow.

### Streamlit

```powershell
streamlit run app.py
```

Or: `.\.venv\Scripts\streamlit run app.py`

Opens `http://localhost:8501`. Sign in before the console appears. After changing `.env`, restart Streamlit (Ctrl+C, then run again).

**Demo credentials (evaluation only — not for production):**

| Username | Password | Role | What they can do |
| --- | --- | --- | --- |
| `admin` | `DemoAdmin!2026` | Admin | Run batch, change provider/model, view the user directory, chat, see all outputs |
| `priya` | `DemoAnalyst1!2026` | Analyst | View dashboard results and chat. Cannot run batch or change Settings |
| `arjun` | `DemoAnalyst2!2026` | Analyst | Same as priya |
| `kavya` | `DemoUser!2026` | User | Same as priya |

1. Sign in as **admin** to operate the system, or as an analyst/user to review results.
2. Admin only: in the sidebar, choose **OpenAI** or **Gemini** and a model.
3. Confirm input `data/` and output `output/`.
4. Admin only: on **Dashboard**, click **Run batch**.
5. Review `final_report.csv`, then open a source file for the email and brief.
6. **Chat** is optional and stored per signed-in username. Grounded Q&A over `data/` and `output/` only.
7. Use **Log out** in the sidebar to switch accounts.

### CLI

```powershell
python main.py
```

`python run_batch.py` is the same folder workflow.

## Sample inputs

| File | Format | Contents |
| --- | --- | --- |
| complaint_001.txt | text | Duplicate billing, attachment, supervisor requested |
| complaint_002.txt | text | Late delivery, crushed carton |
| complaint_003.docx | Word | Defective smart watch, already escalated |
| complaint_004.pdf | PDF | Password-reset email never arrives |
| complaint_005.pdf | PDF | Product feedback — not a complaint |

## Sample outputs

`sample_output/` is a snapshot from a successful local run. Wording can change if you re-run the models.

```text
sample_output/
  structured_data/*.json
  customer_emails/*.txt
  case_summaries/*.txt
  final_report.csv
```

Live runs write the same layout to `output/`.

## Design

- **Three LLM calls, not one.** Extraction is separate from writing.
- **Pydantic validation.** Raw model text is not the system of record.
- **Grounding.** Email and summary prompts must not invent refunds, dates, or contacts.
- **Per-file errors.** One bad PDF does not stop the folder.
- **Parallel generation.** Email and summary share `case_json` after extract.
- **Shared engine.** `app.py` and `main.py` both call `process_folder`.
- **Keys stay in `.env`.** Never rendered in the UI.
- **Local roles.** Streamlit sign-in is file-based (`admin` vs `analyst` / `user`). Passwords are PBKDF2-hashed. Only admin can run the batch or change provider/model.
- **LLM retries.** Provider calls retry up to three times on transient failures (missing keys fail immediately).
- **Audit trail.** Each batch run appends who ran it, provider/model, and counts to `logs/audit.csv` (admin can see recent rows in the sidebar).
- **Email review.** Generated emails stay **draft** until an admin clicks **Approve**. The app does not send mail.

## Tests

Automated checks cover schemas, password hashing, JSON parsing, retrieval, review flags, and audit writes. They do not call OpenAI or Gemini.

```powershell
pytest
```

GitHub Actions (`.github/workflows/ci.yml`) runs the same tests on `main`.

## Limits

- Free API tiers can rate-limit if you run the batch repeatedly.
- Scanned PDFs with no selectable text cannot be read.
- Missing fields are filled with `Unknown` / `Not specified` when the prompt is followed.
- No database — a re-run overwrites same-named files in `output/`. Local users are a YAML file, not IAM.
- Chat is keyword retrieval, not a vector index. If local files do not contain the answer, the app says so.
- This is a **local evaluation console**, not a bank production stack: no SSO, no email gateway, no Kubernetes. Those would be the next hardening steps.

## Demo

- Open the Streamlit console and sign in as `admin` (see demo table above)
- Optionally log out and sign in as `priya` to show view-only dashboard + chat
- Run the sample `data/` folder (admin)
- Show JSON, email, summary, and `final_report.csv`
- Point to log lines for extract → parallel email/summary
- Show **Approve email** (human-in-the-loop) and the sidebar **Audit** rows
- Switch provider if both keys are present
- Chat is extra, not required for Project 1
- Print the architecture poster from `docs/pipeline-one-pager.html` (landscape, background graphics on)
