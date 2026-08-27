"""Pydantic schemas for structured LLM output."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class YesNo(str, Enum):
    YES = "Yes"
    NO = "No"


class CaseRecord(BaseModel):
    """Fields extracted from one customer complaint document."""

    customer_name: str = Field(description="Full customer name if present, otherwise Unknown")
    email: Optional[str] = Field(default=None, description="Customer email if present")
    phone_number: Optional[str] = Field(default=None, description="Customer phone if present")
    complaint_category: str = Field(
        description="Short category such as Billing, Product Defect, Delivery, Service, Account"
    )
    issue_description: str = Field(description="Neutral summary of the issue using only source facts")
    resolution_provided: str = Field(
        description="Resolution stated in the document, or 'Not specified' if missing"
    )
    is_complaint: YesNo
    escalation_required: YesNo
    supporting_document_available: YesNo
    overall_case_status: str = Field(
        description="Status such as Open, In Progress, Resolved, Escalated, Closed"
    )


class CustomerEmail(BaseModel):
    subject: str
    body: str


class CaseSummary(BaseModel):
    case_overview: str
    key_issue: str
    action_taken: str
    current_status: str
    recommended_next_action: str
