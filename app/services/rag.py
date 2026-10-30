"""RAG по правилам игры (rules.json): гибридный поиск BM25 + FAISS и контроль цитирования.

Индекс: каждый пункт `points[]` (без заголовков подразделов) — отдельный документ
с каноническим rule_id. Текст документа = название раздела + текст пункта.

Поиск:
  - BM25 (rank-bm25) по нормализованным токенам с усечением до 5 символов (грубый стемминг
    для русского: «оскорбление/оскорбил/оскорбляет» → «оскор»);
  - FAISS (IndexFlatIP) по мультиязычным эмбеддингам — запросы на uz/kk/en находят русские правила;
  - объединение Reciprocal Rank Fusion + приоры: правила категории из taxonomy.json и
    пункты, заявленные игроком (если существуют).
Возвращается top-k (по умолчанию 3) — весь документ в запрос не передаётся.

validate_citations(): каждая цитата должна существовать, быть в результатах поиска и
соответствовать категории обращения. Иначе — needs_human с причиной.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.logging import get_logger
from app.knowledge_base.rules_index import RulePoint, category_rule_ids, load_rules, rules_by_id
from app.services import embeddings

log = get_logger(__name__)

RRF_K = 60
CATEGORY_PRIOR = 0.035   # ≈ вклад ранга-1 от обоих методов RRF (2/61)
CLAIMED_PRIOR = 0.012

# Сленг игроков → слова из текста правил, чтобы BM25 находил пункты по коротким жалобам
SLANG_EXPANSION = {
    "оск": "оскорбление", "осk": "оскорбление", "osk": "оскорбление", "родни": "родственников", "род": "родственников",
    "упом": "упоминание", "мат": "нецензурная брань", "дм": "убийство урон без причины deathmatch", "dm": "убийство урон без причины deathmatch",
    "дб": "убийство транспортом drive by", "db": "убийство транспортом drive by", "тк": "убийство членов своей организации teamkill",
    "tk": "убийство членов своей организации teamkill", "ск": "убийство на месте появления spawn kill", "sk": "убийство на месте появления spawn kill",
    "софт": "стороннее программное обеспечение читы", "soft": "стороннее программное обеспечение читы", "чит": "стороннее программное обеспечение",
    "сб": "сборка стороннее программное обеспечение", "сборка": "стороннее программное обеспечение", "скам": "мошенничество обман",
    "scam": "мошенничество обман", "зз": "зелёная зона", "gz": "зелёная зона", "гз": "зелёная зона", "нрп": "nonrp ролевая игра",
    "nrp": "nonrp ролевая игра", "реклама": "реклама", "репорт": "репорт", "оффтоп": "оффтоп репорт", "haqorat": "оскорбление",
    "sokdi": "оскорбление", "onadan": "оскорбление родственников", "aldab": "обман мошенничество", "oldirdi": "убийство",
}

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_RU_ENDING_RE = re.compile(r"(ями|ами|ого|ему|ому|ыми|ими|ов|ев|ей|ой|ий|ый|ая|яя|ое|ее|ые|ие|ую|юю|ах|ях|ам|ям|ом|ем|а|я|о|е|ы|и|у|ю)$")


def _stem(w: str) -> str:
    """Грубый стемминг: отсечение типового русского окончания + усечение до 5 символов."""
    if w.isdigit() or len(w) <= 3:
        return w
    s = _RU_ENDING_RE.sub("", w) if len(w) > 4 else w
    return (s if len(s) >= 3 else w)[:5]


def _tokens(text: str, expand: bool = False) -> list[str]:
    """Токены со стеммингом. expand=True (только для запросов) раскрывает игровой сленг."""
    out = []
    for w in _TOKEN_RE.findall(text.lower().replace("ё", "е")):
        if expand and w in SLANG_EXPANSION:
            out.extend(_stem(x) for x in _TOKEN_RE.findall(SLANG_EXPANSION[w].lower().replace("ё", "е")) if len(x) >= 2)
            continue
        if len(w) < 2:
            continue
        out.append(_stem(w))
    return out


@dataclass
class Retrieved:
    rule_id: str
    section: str
    text: str
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    dense_sim: float | None = None
    prior: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"rule_id": self.rule_id, "section": self.section, "text": self.text[:400], "score": round(self.score, 4),
                "bm25_rank": self.bm25_rank, "dense_rank": self.dense_rank,
                "dense_sim": None if self.dense_sim is None else round(self.dense_sim, 3), "prior": self.prior}


class RulesRetriever:
    def __init__(self, use_embeddings: bool | None = None):
        self.rules: tuple[RulePoint, ...] = load_rules()
        self.ids = [r.rule_id for r in self.rules]
        docs = [f"{r.section} {r.text}" for r in self.rules]
        # b=0.5: пункты правил сильно различаются по длине, стандартный b=0.75 слишком штрафует длинные (напр. 2.3)
        self.bm25 = BM25Okapi([_tokens(d) for d in docs], b=0.5)
        self.index = None
        self._lock = threading.Lock()
        if use_embeddings is None:
            use_embeddings = embeddings.available()
        if use_embeddings:
            vecs = embeddings.encode([f"{r.section}. {r.short(400)}" for r in self.rules])
            if vecs is not None:
                import faiss

                self.index = faiss.IndexFlatIP(vecs.shape[1])
                self.index.add(vecs)
        log.info("rules index: %d points, dense=%s", len(self.rules), self.index is not None)

    @property
    def dense(self) -> bool:
        return self.index is not None

    def search(self, query: str, category: str | None = None, claimed: list[str] | None = None,
               k: int = 3, pool: int = 20, category_confidence: float = 1.0) -> list[Retrieved]:
        q_tokens = _tokens(query, expand=True)
        bm25_scores = self.bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(self.ids))
        bm25_order = [i for i in np.argsort(-bm25_scores)[:pool] if bm25_scores[i] > 0]
        dense_order, dense_sims = [], {}
        if self.index is not None and query.strip():
            qv = embeddings.encode([query])
            if qv is not None:
                with self._lock:
                    sims, idx = self.index.search(qv, pool)
                dense_order = [int(i) for i in idx[0] if i >= 0]
                dense_sims = {int(i): float(s) for i, s in zip(idx[0], sims[0]) if i >= 0}

        scores: dict[int, float] = {}
        for rank, i in enumerate(bm25_order):
            scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(dense_order):
            scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        priors: dict[int, list[str]] = {}
        pos = {rid: i for i, rid in enumerate(self.ids)}
        cat_ids = category_rule_ids(category) if category else []
        for rid in cat_ids:
            if rid in pos:
                i = pos[rid]
                # правило категории получает приор, только если поиск его тоже видит, либо категория узкая (≤3 пунктов)
                if i in scores or len(cat_ids) <= 3:
                    # приор масштабируется уверенностью классификатора: неуверенная категория почти не влияет
                    scores[i] = scores.get(i, 0.0) + CATEGORY_PRIOR * category_confidence
                    priors.setdefault(i, []).append("taxonomy")
        for rid in claimed or []:
            if rid in pos:
                i = pos[rid]
                scores[i] = scores.get(i, 0.0) + CLAIMED_PRIOR
                priors.setdefault(i, []).append("claimed")

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        bm25_rank = {i: r + 1 for r, i in enumerate(bm25_order)}
        dense_rank = {i: r + 1 for r, i in enumerate(dense_order)}
        out = []
        for i, s in ranked:
            r = self.rules[i]
            out.append(Retrieved(r.rule_id, r.section, r.text, s, bm25_rank.get(i), dense_rank.get(i), dense_sims.get(i), priors.get(i, [])))
        return out


def retrieval_confidence(results: list[Retrieved]) -> float:
    """Уверенность поиска 0..1: найден ли топ-пункт обоими методами/приором и насколько отрыв."""
    if not results:
        return 0.0
    top = results[0]
    votes = sum([top.bm25_rank is not None and top.bm25_rank <= 5, top.dense_rank is not None and top.dense_rank <= 5,
                 "taxonomy" in top.prior, "claimed" in top.prior])
    conf = 0.2 + 0.2 * votes
    if top.dense_sim is not None:
        conf += 0.2 * max(0.0, min(1.0, (top.dense_sim - 0.2) / 0.5))
    return round(min(1.0, conf), 3)


@dataclass
class CitationCheck:
    valid: bool
    accepted: list[str]
    rejected: dict[str, str]

    def as_dict(self) -> dict:
        return {"valid": self.valid, "accepted": self.accepted, "rejected": self.rejected}


def validate_citations(cited: list[str], retrieved: list[Retrieved], category: str | None = None) -> CitationCheck:
    """Проверка каждой цитаты черновика:
    1) пункт существует в rules.json (каноничный rule_id);
    2) пункт присутствует в результатах поиска для этого обращения;
    3) пункт соответствует обращению: входит в правила категории taxonomy.json или найден
       поиском с поддержкой (BM25/dense в топе, а не только приор).
    """
    index = rules_by_id()
    retrieved_ids = {r.rule_id: r for r in retrieved}
    cat_ids = set(category_rule_ids(category)) if category else set()
    accepted, rejected = [], {}
    for rid in cited:
        if rid not in index:
            rejected[rid] = "пункт не существует в rules.json"
        elif rid not in retrieved_ids:
            rejected[rid] = "пункт не найден поиском для этого обращения"
        else:
            r = retrieved_ids[rid]
            # вне правил категории цитата допустима, только если её нашли оба метода поиска или игрок сам указал пункт
            both = (r.bm25_rank or 99) <= 5 and (r.dense_rank or 99) <= 10
            claimed_found = "claimed" in r.prior and ((r.bm25_rank or 99) <= 10 or (r.dense_rank or 99) <= 10)
            if rid in cat_ids or both or claimed_found:
                accepted.append(rid)
            else:
                rejected[rid] = "пункт не соответствует категории обращения"
    return CitationCheck(valid=not rejected, accepted=accepted, rejected=rejected)
