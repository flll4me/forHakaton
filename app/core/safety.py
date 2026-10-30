"""Safety gate и централизованные условия эскалации.

Безусловно человеку (официальное задание + taxonomy.json `needs_human_always`):
  платежи, возврат средств, апелляции банов и запросы на разблокировку, подозрение на взлом,
  категории report.account, report.payment, complaint.admin.
Дополнительно: низкая уверенность классификации, жалоба, по которой нужно решение о наказании,
не найдены подходящие правила, ошибка генерации или проверки цитат, запрещённое обещание
в черновике, несоответствие языка, некорректная структура ответа.

Confidence итогового ответа НЕ является вероятностью правильности: это минимум из
независимых оценок (классификация, поиск, язык) — см. combine_confidence().
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.knowledge_base.rules_index import load_taxonomy
from app.services import category_rules as cr
from app.services.classifier import THREAT_OUTSIDE_RX
from app.services.language_detector import detect_language
from app.services.rag import CitationCheck

# Причины эскалации (единый словарь — используется в API, отчётах и дашборде)
R_PAYMENT = "платёж/донат — всегда решает человек"
R_REFUND = "запрос возврата средств — всегда решает человек"
R_BAN_APPEAL = "апелляция наказания или запрос разблокировки — всегда решает человек"
R_HACK = "подозрение на взлом аккаунта — всегда решает человек"
R_ALWAYS_CATEGORY = "категория из needs_human_always (taxonomy.json): {cat}"
R_PUNISH_DECISION = "решение о наказании подтверждает администратор"
R_LOW_CONF = "низкая уверенность классификации ({conf:.2f})"
R_NO_RULES = "не найдено подходящих пунктов правил"
R_NO_EVIDENCE = "нет доказательств (вложений или ссылок)"
R_INVALID_REFS = "игрок указал несуществующие пункты правил: {refs}"
R_CITATION = "цитата не прошла проверку: {detail}"
R_PROMISE = "в черновике найдено запрещённое обещание — заменён безопасным шаблоном"
R_LANG = "язык черновика не совпадает с языком игрока"
R_LLM_FAIL = "ошибка LLM ({err}) — использован шаблон"
R_IN_GAME_ACTION = "нужно действие администратора в игре"
R_NO_ANSWER = "ответ по существу должен дать администратор"
R_DUPLICATE = "повторное обращение ({n} за час)"
R_EMPTY = "пустое сообщение"
R_ERROR = "внутренняя ошибка обработки — безопаснее передать человеку"

R_THREAT = "угрозы вне игры — всегда решает человек"
R_POSSIBLE_APPEAL = "в «не жалобе» упоминается наказание — возможна апелляция, нужен человек"

# Страховка для автоответа complaint.invalid: короткие апелляции («за что?», «почему перевыдали на 7?»)
# классификатор иногда относит к «не жалобе». Найдено аудитом на valid вне test (scripts/safety_audit.py).
PUNISHMENT_MENTION_RX = re.compile(
    r"\b(бан\w*|забан\w*|мут\w*|варн\w*|деморган\w*|дмг|джайл\w*|наказ\w*|посадил\w*|перевыдал\w*|выдал\w*|блокировк\w*|"
    r"ban\w*|mut\w*|jazo\w*)\b|за\s+что|где\s+я\s+\w+(ил|ал)\b|в\s+чём\s+(я\s+)?(виноват|нарушил)", re.I)

REFUND_RX = re.compile(r"(?<![а-яё])верн(и|ите|уть)\s+(мне\s+)?(деньги|донат|средства)|возврат\w*\s+(денег|средств|доната)|refund|reembols|pul(ni|imni)?\s+qaytar|ақшамды\s+қайтар|"
                       # обратный порядок слов: «деньги вернут», «донат мне верните» (аудит 2026-10-30)
                       r"(деньги|донат|средства)\s+(\w+\s+){0,2}(?<![а-яё])верн", re.I)
# Запрос снятия наказания при любом порядке слов: «бан снимут», «уберите мут». Дополняет BAN_APPEAL_RX
# только для эскалации — классификатор и слабая разметка остаются прежними.
UNBAN_REQUEST_RX = re.compile(r"(бан|блокировк\w*|наказани\w*|мут|варн|деморган|джайл)\s+(\w+\s+){0,2}(сним|убер)|"
                              r"(сним|убер)\w*\s+(\w+\s+){0,2}(бан\b|бань|блокировк|наказани|мут\b|варн|деморган|джайл)|разблокир", re.I)

# Запрещённые обещания в черновике (на всех языках шаблонов)
FORBIDDEN_PROMISES = [
    re.compile(r"(вы\s+будете|будете|будет|будут)\s+(разбанен|разблокирован|амнистирован)|(снимем|снимут|будет\s+снят[оа]?)\s+(ваш\s+)?(бан|наказани|блокировк|мут)|"
               r"(бан|наказани\w*|блокировк\w*|мут)\s+(\w+\s+)?(снимем|снимут|будет\s+снят[оа]?|будут\s+сняты)", re.I),
    re.compile(r"(вернём|вернем|вернут|будут\s+возвращен|будет\s+возвращ)\w*\s*(вам\s+)?(деньги|средства|донат|имущество|машин)?", re.I),
    re.compile(r"(игрок|нарушитель|он|она|\w+_\w+)\s+(получит|будет\s+наказан|будет\s+забанен|понес[её]т\s+наказание)", re.I),
    re.compile(r"гарантир\w+|обязательно\s+(накажем|разбаним|вернём|вернем)|жалоба\s+(будет\s+)?(одобрена|удовлетворена)", re.I),
    re.compile(r"нарушение\s+(подтверждено|доказано)|по\s+скриншотам\s+видно\s+нарушение", re.I),
    re.compile(r"\b(will\s+be\s+(unbanned|banned|punished|refunded)|we\s+will\s+(refund|unban|ban|punish)|guarantee)", re.I),
    re.compile(r"(ban(dan)?\s+(chiqariladi|olib\s+tashlanadi)|jazolanadi|ban\s+oladi|mut\s+oladi|pul(ingiz)?\s+qaytariladi)", re.I),
    re.compile(r"(жазаланады|бан\s+алады|ақша(ңыз)?\s+қайтарылады|бан\s+алынады)", re.I),
    re.compile(r"(será\s+(desbaneado|baneado|castigado|reembolsado)|serás\s+desbaneado|vamos\s+a\s+(reembolsar|banear))", re.I),
    re.compile(r"(será\s+(desbanido|banido|punido|reembolsado)|vamos\s+(reembolsar|banir))", re.I),
]


@dataclass
class SafetyDecision:
    needs_human: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    replace_draft: bool = False

    def add(self, reason: str, human: bool = True) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)
        if human:
            self.needs_human = True


def mandatory_escalations(kind: str, category: str, text: str) -> list[str]:
    """Условия безусловной передачи человеку по тексту и категории."""
    reasons = []
    always = set(load_taxonomy().get("needs_human_always", []))
    if category in always:
        reasons.append(R_ALWAYS_CATEGORY.format(cat=category))
    if cr.has_any(cr.PAYMENT_RX, text) or category == "report.payment":
        reasons.append(R_PAYMENT)
    if REFUND_RX.search(text):
        reasons.append(R_REFUND)
    if cr.has_any(cr.BAN_APPEAL_RX, text) or UNBAN_REQUEST_RX.search(text):
        reasons.append(R_BAN_APPEAL)
    if cr.has_any(cr.ACCOUNT_HACK_RX, text):
        reasons.append(R_HACK)
    if THREAT_OUTSIDE_RX.search(text):
        reasons.append(R_THREAT)
    return reasons


def find_forbidden_promise(draft_text: str) -> str | None:
    for rx in FORBIDDEN_PROMISES:
        m = rx.search(draft_text)
        if m:
            return m.group(0)
    return None


def language_matches(draft_text: str, expected: str) -> bool:
    """Проверка языка черновика для языков с надёжными маркерами (ru/kk/uz)."""
    if expected not in {"ru", "kk", "uz"}:
        return True
    got = detect_language(draft_text).lang
    return got == expected


def combine_confidence(category_conf: float, retrieval_conf: float | None, lang_conf: float) -> float:
    parts = [category_conf, max(lang_conf, 0.5)]
    if retrieval_conf is not None:
        parts.append(max(retrieval_conf, 0.3))
    return round(max(0.0, min(1.0, min(parts))), 3)


def safety_gate(*, kind: str, category: str, category_conf: float, text: str, facts, draft, citation: CitationCheck,
                rule_needed: bool, retrieval_conf: float, is_empty: bool) -> SafetyDecision:
    s = get_settings()
    d = SafetyDecision(needs_human=False)

    for r in mandatory_escalations(kind, category, text):
        d.add(r)
    if kind == "complaint" and category not in {"complaint.invalid"}:
        d.add(R_PUNISH_DECISION)
        if not facts.has_evidence:
            d.add(R_NO_EVIDENCE)
    if category == "complaint.invalid" and PUNISHMENT_MENTION_RX.search(text):
        d.add(R_POSSIBLE_APPEAL)
    if category_conf < s.min_category_confidence:
        d.add(R_LOW_CONF.format(conf=category_conf))
    if rule_needed and retrieval_conf < s.min_retrieval_score:
        d.add(R_NO_RULES)
    if facts.invalid_rule_refs:
        d.add(R_INVALID_REFS.format(refs=", ".join(facts.invalid_rule_refs)), human=False)
    if is_empty:
        d.add(R_EMPTY, human=False)
    if category in {"report.complaint"}:
        d.add(R_IN_GAME_ACTION)
    if category == "report.question" and not draft.cited:
        d.add(R_NO_ANSWER)
    if category == "report.help" and not draft.auto_answer:
        d.add(R_IN_GAME_ACTION)

    if not citation.valid:
        d.add(R_CITATION.format(detail="; ".join(f"{k}: {v}" for k, v in citation.rejected.items())))
    promise = find_forbidden_promise(draft.text)
    d.checks["forbidden_promise"] = promise
    if promise:
        d.add(R_PROMISE)
        d.replace_draft = True
    lang_ok = language_matches(draft.text, draft.lang)
    d.checks["language_ok"] = lang_ok
    if not lang_ok:
        d.add(R_LANG)
    d.checks["citations"] = citation.as_dict()
    return d
