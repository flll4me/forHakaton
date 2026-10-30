"""Pydantic-схемы контракта POST /triage (официальное задание Grand Mobile)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Lang = Literal["ru", "kk", "uz", "en", "es", "pt", "und"]
Priority = Literal["p0", "p1", "p2", "p3"]


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = "player"
    text: Optional[str] = ""
    attachments: Optional[int] = Field(default=0, ge=0)
    ts: Optional[str] = None

    @field_validator("text", mode="before")
    @classmethod
    def _none_to_empty(cls, v):
        return "" if v is None else v


class TriageRequest(BaseModel):
    """Обязательны только `kind` и `messages`. Остальные поля из выгрузки (reported_nick,
    rule_refs_raw, images_count, created_at, reporter_hash, ...) принимаются опционально:
    в продакшене они приходят из лаунчера и улучшают извлечение фактов и дедупликацию."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["complaint", "report"]
    messages: list[Message] = Field(min_length=1)

    ticket_id: Optional[str] = None
    created_at: Optional[str] = None
    region: Optional[str] = None
    server_lang: Optional[str] = None
    lang: Optional[str] = None
    reporter_hash: Optional[str] = None
    reported_hash: Optional[str] = None
    reported_nick: Optional[str] = None
    rule_refs_raw: Optional[str] = None
    images_count: Optional[int] = None


class Extracted(BaseModel):
    nickname: Optional[str] = None
    rule_refs: list[str] = Field(default_factory=list)
    has_evidence: bool = False
    server: Optional[str] = None


class DraftReply(BaseModel):
    lang: Lang
    text: str
    cited: list[str] = Field(default_factory=list)


class TriageResponse(BaseModel):
    lang: Lang
    category: str
    priority: Priority
    is_spam_or_duplicate: bool
    extracted: Extracted
    draft_reply: DraftReply
    needs_human: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class TriageDebug(BaseModel):
    """Расширенная диагностика (GET /triage?debug=true, дашборд). Не часть контракта."""

    model_config = ConfigDict(extra="allow")

    normalized_text: str = ""
    lang_confidence: float = 0.0
    category_confidence: float = 0.0
    priority_reason: str = ""
    extraction: dict = Field(default_factory=dict)
    duplicate_status: str = "new_issue"
    retrieved: list[dict] = Field(default_factory=list)
    retrieval_confidence: float = 0.0
    generator: str = "template"
    llm_cost_usd: float = 0.0
    latency_ms: float = 0.0
    citation_check: dict = Field(default_factory=dict)
    safety_checks: dict = Field(default_factory=dict)
