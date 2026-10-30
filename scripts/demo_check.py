"""Проверка демо-сценария перед показом: python scripts/demo_check.py [--file demo/demo_tickets.jsonl]

Каждое демо-обращение (написано вручную, не из данных кейса) прогоняется через тот же TriagePipeline,
что и API; поле demo_expect сверяется с ответом. Дополнительно для каждого ответа проверяются
инварианты безопасности: нет запрещённых обещаний, пункты только из rules.json, у эскалации есть причина.
Код выхода 0 — демо готово к показу.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import safety  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.knowledge_base.rules_index import rules_by_id  # noqa: E402
from app.models.schemas import TriageRequest  # noqa: E402
from app.services.pipeline import TriagePipeline  # noqa: E402

DEMO_FILE = ROOT / "demo" / "demo_tickets.jsonl"


def load_demo(path: Path = DEMO_FILE) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return sorted(rows, key=lambda t: t.get("created_at") or "")


def check(resp: dict, expect: dict) -> list[str]:
    errors = []
    for key in ("category", "priority", "lang", "needs_human", "is_spam_or_duplicate"):
        if key in expect and resp[key] != expect[key]:
            errors.append(f"{key}: ожидалось {expect[key]}, получено {resp[key]}")
    cited = resp["draft_reply"]["cited"]
    if "cited_has" in expect and expect["cited_has"] not in cited:
        errors.append(f"нет цитаты {expect['cited_has']} (cited={cited})")
    if "cited_not" in expect and expect["cited_not"] in cited:
        errors.append(f"процитирован {expect['cited_not']}")
    if "text_not" in expect and expect["text_not"] in resp["draft_reply"]["text"]:
        errors.append(f"в черновике «{expect['text_not']}»")
    if safety.find_forbidden_promise(resp["draft_reply"]["text"]):
        errors.append("запрещённое обещание в черновике")
    if any(c not in rules_by_id() for c in cited):
        errors.append(f"несуществующий пункт в cited: {cited}")
    if resp["needs_human"] and not resp["reasons"]:
        errors.append("эскалация без причины")
    if not resp["draft_reply"]["text"]:
        errors.append("пустой черновик")
    return errors


def run(pipeline: TriagePipeline, rows: list[dict]) -> list[tuple[dict, dict, list[str]]]:
    pipeline.deduplicator.reset()
    out = []
    for t in rows:
        resp, _ = pipeline.triage(TriageRequest(**t))
        r = resp.model_dump()
        out.append((t, r, check(r, t.get("demo_expect", {}))))
    return out


def main() -> int:
    setup_logging("WARNING")
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEMO_FILE))
    args = ap.parse_args()
    results = run(TriagePipeline(), load_demo(Path(args.file)))
    failed = 0
    for t, r, errors in results:
        mark = "OK  " if not errors else "FAIL"
        failed += bool(errors)
        print(f"{mark} {t['ticket_id']} {t.get('demo_title', '')}: {r['category']} {r['priority']} lang={r['lang']} "
              f"needs_human={r['needs_human']}")
        for e in errors:
            print(f"       - {e}")
    print(f"\n{len(results) - failed}/{len(results)} сценариев прошли проверку")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
