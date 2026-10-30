"""Пакетный прогноз: make predict IN=tickets.jsonl OUT=pred.jsonl

- та же логика, что POST /triage (TriagePipeline);
- одна строка JSON (контракт ответа) на каждую входную строку, исходный порядок сохраняется;
- дедупликация причинная: для каждого тикета учитываются только более ранние по created_at;
- ошибка в отдельной строке (битый JSON, неверная схема) не останавливает обработку:
  для неё пишется безопасный ответ с needs_human=true;
- логи только в stderr; в OUT нет ничего, кроме ответов.
Опции: --debug добавляет поле _debug (не для сдачи), --ids добавляет ticket_id.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.models.schemas import DraftReply, Extracted, TriageRequest, TriageResponse  # noqa: E402
from app.services.pipeline import TriagePipeline  # noqa: E402

log = get_logger("predict")


def error_response(kind: str | None, reason: str) -> dict:
    return TriageResponse(
        lang="und", category="report.offtopic" if kind == "report" else "complaint.other", priority="p2",
        is_spam_or_duplicate=False, extracted=Extracted(), draft_reply=DraftReply(lang="ru", text="", cited=[]),
        needs_human=True, confidence=0.0, reasons=[reason],
    ).model_dump()


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--ids", action="store_true")
    args = ap.parse_args()

    lines = Path(args.inp).read_text(encoding="utf-8-sig").splitlines()
    parsed: list[tuple[int, dict | None, str | None]] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            parsed.append((i, json.loads(line), None))
        except json.JSONDecodeError as e:
            parsed.append((i, None, f"некорректный JSON в строке {i + 1}: {e.msg}"))

    pipeline = TriagePipeline()
    # причинный порядок для дедупликации; записи без created_at — в конце, в исходном порядке
    order = sorted(range(len(parsed)), key=lambda j: ((parsed[j][1] or {}).get("created_at") or "9999", j))
    results: dict[int, dict] = {}
    latencies = []
    t_all = time.perf_counter()
    for n, j in enumerate(order, 1):
        _, rec, err = parsed[j]
        if err:
            results[j] = error_response(None, err)
            continue
        try:
            req = TriageRequest(**rec)
        except Exception as e:
            results[j] = error_response(rec.get("kind") if isinstance(rec, dict) else None, f"запись не соответствует схеме: {type(e).__name__}")
            continue
        t0 = time.perf_counter()
        resp, dbg = pipeline.triage(req)
        latencies.append((time.perf_counter() - t0) * 1000)
        out = resp.model_dump()
        if args.ids:
            out = {"ticket_id": rec.get("ticket_id"), **out}
        if args.debug:
            out["_debug"] = dbg.model_dump()
        results[j] = out
        if n % 200 == 0:
            log.info("processed %d/%d", n, len(order))

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for j in range(len(parsed)):
            f.write(json.dumps(results[j], ensure_ascii=False) + "\n")
    if latencies:
        ls = sorted(latencies)
        log.info("done: %d records in %.1fs, p50=%.1fms p95=%.1fms", len(parsed), time.perf_counter() - t_all,
                 ls[len(ls) // 2], ls[int(0.95 * (len(ls) - 1))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
