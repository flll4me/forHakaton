"""Оценка дедупликации на train (серии обращений одного игрока там сохранены целиком).

Прокси-разметка: verdict=duplicate в ответе админа (жалоба признана дубликатом).
Это неполная разметка: админ помечает не все повторы, а дубликат может относиться
к жалобе, поданной вне часового окна. Поэтому считаются:
  - recall по verdict=duplicate (какую долю админских дублей сервис поймал в окне часа);
  - доля verdict=duplicate среди помеченных нами как duplicate (нижняя граница точности);
  - статистика серий «≥5 обращений одного игрока за час» (условие из задания).
Используется символьная схожесть (без эмбеддингов) — так же, как fallback сервиса.
Выход: дописывает раздел в reports/dedup_evaluation.md.
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.deduplicator import Deduplicator  # noqa: E402
from app.services.normalizer import normalize_ticket  # noqa: E402


def main() -> int:
    rows = []
    with gzip.open(get_settings().data_dir / "tickets_train.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            rows.append((t.get("created_at") or "", t["ticket_id"], t["kind"], t.get("reporter_hash"), t.get("reported_hash"),
                         t.get("verdict"), normalize_ticket(t["messages"], t.get("region"), t.get("server_lang")).clean_text))
    rows.sort()
    d = Deduplicator(use_embeddings=False)
    stat = collections.Counter()
    series5 = 0
    for ts, tid, kind, rep, rped, verdict, text in rows:
        r = d.check_and_add(tid, kind, text, rep, ts, rped)
        stat[(kind, r.status)] += 1
        if r.series_count >= 5:
            series5 += 1
        if kind == "complaint":
            stat[("admin_dup", verdict == "duplicate", r.status)] += 1

    dup_total = sum(v for k, v in stat.items() if k[0] == "admin_dup" and k[1])
    dup_caught = stat[("admin_dup", True, "duplicate")] + stat[("admin_dup", True, "related")]
    ours_dup = stat[("complaint", "duplicate")]
    ours_dup_admin = stat[("admin_dup", True, "duplicate")]
    lines = ["# Оценка дедупликации на train\n", "Сгенерировано `scripts/evaluate_dedup.py`. Только агрегаты.\n",
             "| показатель | значение |", "|---|---|",
             f"| Обращений | {len(rows)} |",
             f"| report: duplicate / related / new_issue | {stat[('report', 'duplicate')]} / {stat[('report', 'related')]} / {stat[('report', 'new_issue')]} |",
             f"| complaint: duplicate / related / new_issue | {stat[('complaint', 'duplicate')]} / {stat[('complaint', 'related')]} / {stat[('complaint', 'new_issue')]} |",
             f"| Обращений, которые ≥5-е в серии игрока за час | {series5} |",
             f"| Жалоб с verdict=duplicate у админа | {dup_total} |",
             f"| из них найдено как duplicate/related (recall-прокси) | {dup_caught} ({100 * dup_caught / max(1, dup_total):.1f}%) |",
             f"| Наших complaint-duplicate, которые админ тоже пометил duplicate | {ours_dup_admin} из {ours_dup} ({100 * ours_dup_admin / max(1, ours_dup):.1f}%) |",
             "",
             "Интерпретация: админ помечает дубликатом только часть повторов (чаще повторную жалобу после уже рассмотренной), "
             "поэтому низкая доля совпадений с verdict не означает ложные срабатывания; это нижняя граница точности."]
    (ROOT / "reports" / "dedup_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
