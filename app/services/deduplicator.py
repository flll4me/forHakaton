"""Поиск повторных обращений (бонус задания: «1 игрок, 1 проблема, 5 тикетов за час»).

Кандидаты — предыдущие обращения того же reporter_hash в окне DUPLICATE_WINDOW_MINUTES.
Решение:
  duplicate — тот же kind и та же проблема: для жалоб тот же обвиняемый (reported_hash)
              и похожий текст/пустой текст; для репортов — высокая семантическая схожесть;
  related   — тот же игрок в окне, схожесть средняя или другой обвиняемый с похожим текстом;
  new_issue — иначе, а также когда нет reporter_hash/created_at (данных недостаточно).
Схожесть: косинус мультиязычных эмбеддингов; без модели — символьные 3-граммы (Жаккар).
Общие слова сами по себе дубликат не дают: нужен тот же игрок, окно времени и порог.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from app.core.config import get_settings
from app.services import embeddings
from app.services.category_rules import is_content_free


@dataclass
class DedupResult:
    status: str                       # duplicate | related | new_issue
    similarity: float = 0.0
    matched_ticket_id: str | None = None
    series_count: int = 1             # сколько обращений игрока в окне, включая текущее
    note: str = ""


@dataclass
class _Entry:
    ticket_id: str
    created_at: datetime
    kind: str
    reported_hash: str | None
    text: str
    vec: np.ndarray | None = field(default=None, repr=False)


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _trigrams(t: str) -> set[str]:
    t = f"  {t.lower()}  "
    return {t[i:i + 3] for i in range(len(t) - 2)}


def _jaccard(a: str, b: str) -> float:
    x, y = _trigrams(a), _trigrams(b)
    return len(x & y) / len(x | y) if x and y else 0.0


class Deduplicator:
    def __init__(self, use_embeddings: bool | None = None, max_per_reporter: int = 50):
        s = get_settings()
        self.window = timedelta(minutes=s.duplicate_window_minutes)
        self.dup_thr = s.duplicate_similarity
        self.rel_thr = s.related_similarity
        self.use_embeddings = embeddings.available() if use_embeddings is None else use_embeddings
        self._store: dict[str, deque[_Entry]] = defaultdict(lambda: deque(maxlen=max_per_reporter))
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._store.clear()

    def _vec(self, text: str) -> np.ndarray | None:
        if not self.use_embeddings or not text.strip():
            return None
        v = embeddings.encode([text])
        return None if v is None else v[0]

    def _sim(self, a: _Entry, b: _Entry) -> float:
        if a.vec is not None and b.vec is not None:
            return float(np.dot(a.vec, b.vec))
        return _jaccard(a.text, b.text)

    def check_and_add(self, ticket_id: str, kind: str, text: str, reporter_hash: str | None,
                      created_at: str | None, reported_hash: str | None = None) -> DedupResult:
        ts = _parse_ts(created_at)
        if not reporter_hash or ts is None:
            return DedupResult("new_issue", note="нет reporter_hash/created_at — повтор не проверялся")
        cur = _Entry(ticket_id, ts, kind, reported_hash, text, self._vec(text))
        with self._lock:
            history = [e for e in self._store[reporter_hash] if timedelta(0) <= ts - e.created_at <= self.window]
            self._store[reporter_hash].append(cur)
        series = 1 + len(history)
        best, best_sim = None, 0.0
        for e in history:
            if e.kind != kind:
                continue
            s = self._sim(cur, e)
            if kind == "complaint":
                same_target = bool(reported_hash) and e.reported_hash == reported_hash
                if not same_target:
                    s *= 0.5  # другой обвиняемый — другая проблема, если только текст не совпадает почти дословно
                elif is_content_free(text) or is_content_free(e.text):
                    s = max(s, self.dup_thr)  # жалоба без текста на того же игрока в течение часа
            if s > best_sim:
                best, best_sim = e, s
        if best is None:
            return DedupResult("new_issue", series_count=series,
                               note="нет обращений игрока за окно" if series == 1 else "в окне есть обращения другого типа")
        status = "duplicate" if best_sim >= self.dup_thr else "related" if best_sim >= self.rel_thr else "new_issue"
        return DedupResult(status, round(best_sim, 3), best.ticket_id, series,
                           note=f"{series} обращ. игрока за {int(self.window.total_seconds() // 60)} мин")
