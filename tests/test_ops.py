from pydantic import ValidationError

from src.audit import record_audit, read_audit
from src.auth import hash_password, verify_password
from src.llm import parse_json_object
from src.models import CaseRecord, YesNo
from src.retrieve import retrieve, tokenize
from src.review import APPROVED, DRAFT, get_status, set_status


def test_case_record_rejects_invalid_flag() -> None:
    try:
        CaseRecord(
            customer_name="A",
            complaint_category="Billing",
            issue_description="Overcharged",
            resolution_provided="Not specified",
            is_complaint="Maybe",  # type: ignore[arg-type]
            escalation_required=YesNo.NO,
            supporting_document_available=YesNo.NO,
            overall_case_status="Open",
        )
    except ValidationError:
        return
    raise AssertionError("invalid YesNo should fail")


def test_case_record_accepts_required_fields() -> None:
    record = CaseRecord(
        customer_name="Ada",
        email="ada@example.com",
        phone_number=None,
        complaint_category="Billing",
        issue_description="Duplicate charge",
        resolution_provided="Not specified",
        is_complaint=YesNo.YES,
        escalation_required=YesNo.NO,
        supporting_document_available=YesNo.YES,
        overall_case_status="Open",
    )
    assert record.customer_name == "Ada"
    assert record.is_complaint == YesNo.YES


def test_password_hash_roundtrip() -> None:
    stored = hash_password("DemoAdmin!2026", pepper="")
    assert verify_password("DemoAdmin!2026", stored, pepper="")
    assert not verify_password("wrong", stored, pepper="")


def test_parse_json_object_strips_fence() -> None:
    payload = parse_json_object('```json\n{"a": 1}\n```')
    assert payload == {"a": 1}


def test_retrieve_ranks_relevant_document() -> None:
    corpus = [
        ("complaint_001.txt", "Duplicate billing on invoice 4412 for Ada West."),
        ("notes.txt", "The office kitchen needs coffee."),
    ]
    hits = retrieve("duplicate billing invoice", corpus, top_k=2)
    assert hits
    assert hits[0].name == "complaint_001.txt"


def test_tokenize_drops_stopwords() -> None:
    tokens = tokenize("What is the leave policy for the team")
    assert "the" not in tokens
    assert "leave" in tokens
    assert "policy" in tokens


def test_review_defaults_to_draft(tmp_path) -> None:
    assert get_status(tmp_path, "complaint_001.txt") == DRAFT
    set_status(tmp_path, "complaint_001.txt", APPROVED, "admin")
    assert get_status(tmp_path, "complaint_001.txt") == APPROVED


def test_audit_appends_row(tmp_path, monkeypatch) -> None:
    import src.audit as audit

    monkeypatch.setattr(audit, "LOG_DIR", tmp_path)
    audit.record_audit(actor="admin", action="run_batch", processed=4, failed=1, total=5)
    rows = audit.read_audit(limit=5)
    assert len(rows) == 1
    assert rows[0]["actor"] == "admin"
    assert rows[0]["processed"] == "4"
