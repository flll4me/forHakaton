"""Хранилище обратной связи администраторов (SQLite).

Сохраняется исходный черновик, исправленный текст, категория (в т.ч. исправленная админом),
действие (send | edit | escalate) и оценка качества 1–5. Автоматического переобучения нет:
накопленные данные проверяются вручную перед использованием как обучающий сигнал.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from app.core.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('send', 'edit', 'escalate')),
    predicted_category TEXT,
    final_category TEXT,
    priority TEXT,
    lang TEXT,
    draft_text TEXT,
    final_text TEXT,
    cited TEXT,
    rating INTEGER CHECK (rating IS NULL OR rating BETWEEN 1 AND 5),
    comment TEXT
);
CREATE INDEX IF NOT EXISTS ix_feedback_ticket ON feedback(ticket_id);
"""


class FeedbackStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or get_settings().feedback_db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, *, ticket_id: str, action: str, predicted_category: str, final_category: str, priority: str, lang: str,
            draft_text: str, final_text: str, cited: list[str], rating: int | None = None, comment: str = "") -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO feedback (ticket_id, created_at, action, predicted_category, final_category, priority, lang,"
                " draft_text, final_text, cited, rating, comment) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (ticket_id, time.strftime("%Y-%m-%dT%H:%M:%S"), action, predicted_category, final_category, priority, lang,
                 draft_text, final_text, ",".join(cited), rating, comment))
            return int(cur.lastrowid)

    def status_by_ticket(self) -> dict[str, str]:
        with self._conn() as c:
            rows = c.execute("SELECT ticket_id, action FROM feedback ORDER BY id").fetchall()
        return {r["ticket_id"]: r["action"] for r in rows}

    def all(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM feedback ORDER BY id DESC").fetchall()]

    def stats(self) -> dict:
        with self._conn() as c:
            r = c.execute("SELECT COUNT(*) n, AVG(rating) avg_rating, SUM(action='edit') edits, SUM(action='escalate') esc,"
                          " SUM(predicted_category != final_category) recategorized FROM feedback").fetchone()
        return dict(r)
