"""Классификация: категория (taxonomy.json), приоритет, спам.

Категория — гибрид:
  1. ML-модель (выбирается по результатам scripts/train_classifier.py, по умолчанию
     TF-IDF слова + символьные n-граммы → LogisticRegression), отдельная для complaint и report:
     категории taxonomy привязаны к kind, поэтому модель не может выдать чужую категорию.
  2. Высокоточные правила (category_rules) для категорий, где ошибка дорогая:
     report.payment, report.account, complaint.admin (апелляции). Они перекрывают ML.
  3. Без файла модели работает только слой правил (с низкой confidence → эскалация).

Приоритет — детерминированная функция по критериям taxonomy.json (не ML: критерии
заданы явно, а меток приоритета в данных нет).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services import category_rules as cr

log = get_logger(__name__)

MODEL_FILE = "classifier.joblib"
DEFAULT_CATEGORY = {"complaint": "complaint.other", "report": "report.offtopic"}
SAFETY_RULE_CATEGORIES = {"report.payment", "report.account", "complaint.admin"}
ML_OVERRIDE_PROBA = 0.6  # ниже этой вероятности ML уступает сработавшему правилу

THREAT_OUTSIDE_RX = re.compile(
    r"найду\s+тебя|вычислю|по\s+айпи|по\s+ip\b|твой\s+адрес|знаю\s+где\s+ты\s+живешь|знаю\s+где\s+ты\s+живёшь|в\s+реал(е|ьной\s+жизни)\s+(убью|найду|приеду)|"
    r"приеду\s+к\s+тебе|доксинг|сват|слили\s+(мои\s+)?(данные|фото|номер)|угрожа\w*\s+(в\s+реал|убийством|расправой|семье)|seni\s+topaman|"
    # от третьего лица — так угрозы пересказывают в жалобах (аудит безопасности 2026-10-30)
    r"найд[её]т\s+(меня|мою\s+семью)|знает\s+где\s+я\s+живу|в\s+реал(е|ьной\s+жизни)\s+(убь[её]т|найд[её]т|приед)|приед[её]т\s+ко\s+мне|"
    r"(найд[её]т|найду)\s+\w*\s*в\s+реал",
    re.IGNORECASE,
)
MASS_BUG_RX = re.compile(
    r"у\s+всех\s+(игроков\s+)?(не\s+работает|баг|пропал|вылета)|сервер\s+(упал|лежит|не\s+работает)|массов\w*\s+баг|никто\s+не\s+может\s+(зайти|войти)|"
    r"все\s+игроки\s+(не\s+могут|вылетели)",
    re.IGNORECASE,
)
RULE_QUESTION_RX = re.compile(r"правил|пункт|\b\d\.\d|нарушени|варн|можно\s+ли|разрешено|запрещено|\bрп\b|наказан|qoida", re.IGNORECASE)


def model_input(kind: str, text: str, rule_refs_raw: str | None) -> str:
    """Единое представление для обучения и инференса."""
    refs = (rule_refs_raw or "").strip().lower()
    return f"{text.lower()} || refs: {refs}" if kind == "complaint" else text.lower()


@dataclass
class CategoryResult:
    category: str
    confidence: float
    source: str                      # ml | rule | rule+ml | fallback
    probabilities: dict = field(default_factory=dict)
    rule_reason: str = ""


@dataclass
class PriorityResult:
    priority: str
    reason: str


class CategoryClassifier:
    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or get_settings().models_dir / MODEL_FILE
        self.models: dict = {}
        self.meta: dict = {}
        self.load()

    def load(self) -> None:
        if self.model_path.exists():
            bundle = joblib.load(self.model_path)
            self.models = bundle["models"]
            self.meta = bundle.get("meta", {})
            log.info("classifier loaded: %s", self.meta.get("approach"))
        else:
            log.warning("classifier model not found at %s — rules-only mode", self.model_path)

    @property
    def ready(self) -> bool:
        return bool(self.models)

    def _ml(self, kind: str, x: str) -> dict:
        model = self.models.get(kind)
        if model is None:
            return {}
        proba = model.predict_proba([x])[0]
        return {c: float(p) for c, p in zip(model.classes_, proba)}

    def predict(self, kind: str, text: str, rule_refs_raw: str | None = None) -> CategoryResult:
        if kind == "complaint":
            rule_cat, reason = cr.complaint_rule_category(text, rule_refs_raw)
        else:
            rule_cat, reason = cr.report_rule_category(text)
        probs = self._ml(kind, model_input(kind, text, rule_refs_raw))
        if probs:
            ml_cat = max(probs, key=probs.get)
            p = probs[ml_cat]
            if rule_cat in SAFETY_RULE_CATEGORIES and rule_cat != ml_cat:
                return CategoryResult(rule_cat, 0.8, "rule", probs, reason)
            if rule_cat == ml_cat:
                return CategoryResult(ml_cat, max(p, 0.85), "rule+ml", probs, reason)
            if rule_cat and p < ML_OVERRIDE_PROBA:
                # модель не уверена, а правило сработало — берём правило, но с умеренной уверенностью
                return CategoryResult(rule_cat, 0.55, "rule", probs, reason)
            return CategoryResult(ml_cat, p, "ml", probs, reason)
        if rule_cat:
            return CategoryResult(rule_cat, 0.6, "rule", {}, reason)
        return CategoryResult(DEFAULT_CATEGORY[kind], 0.2, "fallback", {}, reason)


def priority_for(category: str, has_evidence: bool, text: str) -> PriorityResult:
    """Критерии taxonomy.json:
    p0 — взлом/потеря доступа, платежи, массовый баг, угрозы вне игры;
    p1 — читы, мошенничество, реклама RMT, жалобы с доказательствами;
    p2 — оскорбления, DM, нарушения РП, вопросы по правилам;
    p3 — вопросы по механике, предложения, оффтоп, жалобы без доказательств.
    """
    t = text or ""
    if category in {"report.account", "report.payment"}:
        return PriorityResult("p0", "аккаунт/платёж (taxonomy p0)")
    if cr.has_any(cr.PAYMENT_RX, t) or cr.has_any(cr.ACCOUNT_HACK_RX, t):
        return PriorityResult("p0", "упоминание платежа или взлома (taxonomy p0)")
    if THREAT_OUTSIDE_RX.search(t):
        return PriorityResult("p0", "угрозы вне игры (taxonomy p0)")
    if MASS_BUG_RX.search(t):
        return PriorityResult("p0", "массовый баг (taxonomy p0)")
    if category in {"complaint.cheat", "complaint.scam", "complaint.rmt_ads"}:
        return PriorityResult("p1", "читы/мошенничество/RMT (taxonomy p1)")
    if category in {"complaint.insult", "complaint.dm"}:
        if has_evidence:
            return PriorityResult("p2", "оскорбления/DM/РП (taxonomy p2)")
        return PriorityResult("p3", "жалоба без доказательств (taxonomy p3)")
    if category in {"complaint.admin", "complaint.other"}:
        if has_evidence:
            return PriorityResult("p1", "жалоба с доказательствами (taxonomy p1)")
        return PriorityResult("p3", "жалоба без доказательств (taxonomy p3)")
    if category == "report.complaint":
        return PriorityResult("p2", "оскорбления/DM через репорт (taxonomy p2)")
    if category == "report.question":
        if RULE_QUESTION_RX.search(t):
            return PriorityResult("p2", "вопрос по правилам (taxonomy p2)")
        return PriorityResult("p3", "вопрос по механике (taxonomy p3)")
    return PriorityResult("p3", "оффтоп/помощь/невалидная жалоба (taxonomy p3)")


def is_spam(kind: str, text: str, has_evidence: bool) -> bool:
    """Спам/флуд: сообщение без содержательных слов («+», «.», цифры) без вложений."""
    return cr.is_content_free(text) and not has_evidence
