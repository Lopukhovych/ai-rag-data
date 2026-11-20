"""Schemas for every file build.py writes. validate.py checks files against these."""

from typing import Literal

from pydantic import BaseModel, Field

ROLES = ["employee", "engineering", "sales", "hr", "exec"]
TENANTS = ["enterprise", "subsidiary"]
Role = Literal["employee", "engineering", "sales", "hr", "exec"]
Tenant = Literal["enterprise", "subsidiary"]


class Doc(BaseModel):
    doc_id: str
    source_type: str
    title: str
    content: str
    tenant_id: Tenant
    clone_of: str | None = None


class AclRecord(BaseModel):
    doc_id: str
    tenant_id: Tenant
    source_type: str
    visibility: Literal["tenant", "roles", "principals"]
    allow_roles: list[Role] = Field(default_factory=list)
    allow_emails: list[str] = Field(default_factory=list)
    rule: str


class User(BaseModel):
    user_id: str
    email: str
    role: Role
    tenant_id: Tenant


class Question(BaseModel):
    question_id: str
    question_type: str
    source_types: list[str] = Field(default_factory=list)
    question: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    gold_answer: str = ""
    answer_facts: list[str] = Field(default_factory=list)
    tenant_id: Tenant = "enterprise"
    split: Literal["dev", "heldout"]
    origin_question_id: str | None = None


class Probe(BaseModel):
    """Hand-written permission-probe or injection case (questions/probes.jsonl)."""

    probe_id: str
    kind: Literal["permission", "injection"]
    user_id: str
    question: str
    expected_behavior: Literal["answer", "deny", "ignore_injection"]
    expected_doc_ids: list[str] = Field(default_factory=list)
    forbidden_doc_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class RedteamDoc(BaseModel):
    doc_id: str
    source_type: str
    title: str
    content: str
    tenant_id: Tenant = "enterprise"
    attack: Literal["indirect_injection", "fake_instruction", "exfiltration_lure"]
    redteam: Literal[True] = True


class ExtractionGold(BaseModel):
    doc_id: str
    owner: str
    date: str
    decisions: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    systems: list[str] = Field(default_factory=list)
    labeled_by: str = ""
    labeled_on: str = ""


class Ticket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    answer: str = ""
    type: str = ""
    queue: str
    priority: Literal["low", "medium", "high"]
    tags: list[str] = Field(default_factory=list)
    split: Literal["train", "val", "dev", "heldout"]
