"""Единый пайплайн триажа: используется и в POST /triage, и в make predict.

INPUT → PREPROCESSING → LANGUAGE → CLASSIFICATION → FACTS → DEDUP → RAG → GENERATION → SAFETY → OUTPUT
Каждый этап — отдельный модуль. Ошибка любого этапа не роняет обработку: ответ
собирается с безопасными значениями и needs_human=true.
"""
from __future__ import annotations

import time
import uuid

from app.core import safety
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import DraftReply, Extracted, TriageDebug, TriageRequest, TriageResponse
from app.services import embeddings
from app.services.classifier import CategoryClassifier, is_spam, priority_for
from app.services.deduplicator import Deduplicator
from app.services.fact_extractor import Facts, extract_facts
from app.services.language_detector import detect_language
from app.services.llm_provider import get_provider
from app.services.normalizer import normalize_ticket
from app.services.rag import RulesRetriever, retrieval_confidence, validate_citations
from app.services.response_generator import RULE_NEEDED, Draft, ResponseGenerator, reply_language

log = get_logger(__name__)


class TriagePipeline:
    def __init__(self, use_embeddings: bool | None = None):
        t0 = time.perf_counter()
        self.settings = get_settings()
        if use_embeddings is None:
            use_embeddings = embeddings.available()
        self.classifier = CategoryClassifier()
        self.retriever = RulesRetriever(use_embeddings=use_embeddings)
        self.deduplicator = Deduplicator(use_embeddings=use_embeddings)
        self.llm = get_provider()
        self.generator = ResponseGenerator(self.retriever, self.llm)
        self.load_s = round(time.perf_counter() - t0, 2)
        log.info("pipeline ready in %.1fs (classifier=%s, dense=%s, llm=%s)",
                 self.load_s, self.classifier.ready, self.retriever.dense, self.llm.name)

    def status(self) -> dict:
        return {
            "classifier": "ml+rules" if self.classifier.ready else "rules-only",
            "classifier_approach": self.classifier.meta.get("approach"),
            "rag_index_points": len(self.retriever.ids),
            "rag_dense": self.retriever.dense,
            "llm_provider": self.llm.name,
            "llm_model": self.settings.llm_model or None,
            "llm_disabled_reason": self.llm.disabled_reason,
        }

    def triage(self, req: TriageRequest) -> tuple[TriageResponse, TriageDebug]:
        t0 = time.perf_counter()
        try:
            return self._triage(req, t0)
        except Exception as e:  # последняя линия обороны: один тикет не роняет сервис/батч
            log.exception("triage failed: %s", e)
            lang = reply_language("und", req.region)
            resp = TriageResponse(
                lang="und", category="report.offtopic" if req.kind == "report" else "complaint.other", priority="p2",
                is_spam_or_duplicate=False, extracted=Extracted(),
                draft_reply=DraftReply(lang=lang, text="", cited=[]), needs_human=True, confidence=0.0,
                reasons=[safety.R_ERROR],
            )
            return resp, TriageDebug(latency_ms=round((time.perf_counter() - t0) * 1000, 1), generator="error")

    def _triage(self, req: TriageRequest, t0: float) -> tuple[TriageResponse, TriageDebug]:
        messages = [m.model_dump() for m in req.messages]
        norm = normalize_ticket(messages, req.region, req.server_lang)
        text = norm.clean_text

        lang_res = detect_language(text, req.region)

        cat = self.classifier.predict(req.kind, text, req.rule_refs_raw)
        if norm.is_empty and not req.rule_refs_raw:
            cat.category = "report.offtopic" if req.kind == "report" else "complaint.invalid"
            cat.confidence = max(cat.confidence, 0.7)

        facts: Facts = extract_facts(req.kind, text, messages, req.reported_nick, req.rule_refs_raw,
                                     req.images_count, norm.accused_text)
        prio = priority_for(cat.category, facts.has_evidence, text)

        dedup = self.deduplicator.check_and_add(req.ticket_id or uuid.uuid4().hex, req.kind, text,
                                                req.reporter_hash, req.created_at, req.reported_hash)
        spam = is_spam(req.kind, text, facts.has_evidence)

        query = text if text.strip() else (req.rule_refs_raw or "")
        retrieved = self.retriever.search(query, cat.category, facts.rule_refs, k=3, category_confidence=cat.confidence)
        r_conf = retrieval_confidence(retrieved)

        lang = reply_language(lang_res.lang, req.region)
        draft: Draft = self.generator.template(kind=req.kind, category=cat.category, lang=lang, facts=facts,
                                               retrieved=retrieved, text=text, retrieval_conf=r_conf)
        llm_reason = None
        if self.generator.is_complex(req.kind, cat.category, text):
            ld = self.generator.llm_draft(category=cat.category, lang=lang, facts=facts, retrieved=retrieved,
                                          procedural=draft.procedural, text=text)
            if ld is not None and ld.generator == "llm":
                check = validate_citations(ld.cited, retrieved + ld.procedural, cat.category)
                if check.valid and ld.text and not safety.find_forbidden_promise(ld.text) and safety.language_matches(ld.text, lang):
                    ld.auto_answer = draft.auto_answer
                    draft = ld
                else:
                    llm_reason = safety.R_LLM_FAIL.format(err="ответ не прошёл проверки")
                    draft.llm_cost_usd = ld.llm_cost_usd
            elif ld is not None:
                llm_reason = safety.R_LLM_FAIL.format(err=ld.llm_error)
                draft.llm_cost_usd = ld.llm_cost_usd

        citation = validate_citations(draft.cited, retrieved + draft.procedural, cat.category)
        rule_needed = cat.category in RULE_NEEDED
        decision = safety.safety_gate(kind=req.kind, category=cat.category, category_conf=cat.confidence, text=text,
                                      facts=facts, draft=draft, citation=citation, rule_needed=rule_needed,
                                      retrieval_conf=r_conf, is_empty=norm.is_empty)
        if llm_reason:
            decision.add(llm_reason)
        if decision.replace_draft or not citation.valid:
            # безопасный шаблон без неподтверждённых цитат
            draft = self.generator.template(kind=req.kind, category=cat.category, lang=lang, facts=facts,
                                            retrieved=retrieved, text=text, retrieval_conf=r_conf)
            draft.cited = [c for c in draft.cited if c in citation.accepted or c in [p.rule_id for p in draft.procedural]]
        if dedup.status == "duplicate":
            decision.add(safety.R_DUPLICATE.format(n=dedup.series_count), human=False)

        confidence = safety.combine_confidence(cat.confidence, r_conf if rule_needed else None, lang_res.confidence if lang_res.lang != "und" else 0.5)
        resp = TriageResponse(
            lang=lang_res.lang, category=cat.category, priority=prio.priority,
            is_spam_or_duplicate=spam or dedup.status == "duplicate",
            extracted=Extracted(**facts.contract()),
            draft_reply=DraftReply(lang=draft.lang, text=draft.text, cited=draft.cited),
            needs_human=decision.needs_human, confidence=confidence, reasons=decision.reasons,
        )
        debug = TriageDebug(
            normalized_text=norm.normalized_text, lang_confidence=lang_res.confidence, category_confidence=round(cat.confidence, 3),
            priority_reason=prio.reason, extraction={k: v for k, v in facts.__dict__.items()},
            duplicate_status=dedup.status, retrieved=[r.as_dict() for r in retrieved], retrieval_confidence=r_conf,
            generator=draft.generator, llm_cost_usd=round(draft.llm_cost_usd, 6),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1), citation_check=citation.as_dict(),
            safety_checks=decision.checks, category_source=cat.source, dedup={**dedup.__dict__}, spam=spam,
            kazakh_fixed=norm.kazakh_fixed,
        )
        return resp, debug
