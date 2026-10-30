"""Weak supervision: слабая разметка категорий для обучения классификатора.

Категорий в данных нет. Метка ставится набором labeling functions (LF) по приоритету:

complaint:
  LF1 verdict ∈ {appeal_approved, appeal_rejected}                → complaint.admin
  LF1b ответ админа: «наказание выдано верно», «с администратором проведена беседа»,
       «напишите жалобу на администратора», «скриншот выдачи наказания»   → complaint.admin
  LF2 текст игрока: обжалование / претензия к админу                → complaint.admin
  LF3 ответ админа ссылается на ППЖ 1.1.2 (не жалоба)               → complaint.invalid
  LF4 поле violation/punishment из ответа админа                    → категория нарушения
  LF5 правила по тексту игрока + заявленный пункт                   → категория
  LF6 пустой текст без заявленного пункта                           → complaint.invalid
report:
  LF1 правила по тексту игрока (платёж, аккаунт, помощь, жалоба, вопрос, пусто)
  LF2 типовой ответ админа («слежу за игроком» → report.complaint, «уточните вопрос» → offtopic, ...)
Если ни одна LF не сработала — запись не размечается (abstain) и в обучение не идёт.

Качество слабых меток оценивается на ручной контрольной выборке (scripts/evaluate.py).
Выход: data/processed/{train,valid}_weak.jsonl (в .gitignore — производные данные кейса).
"""
from __future__ import annotations

import collections
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services import category_rules as cr  # noqa: E402
from app.services.normalizer import normalize_ticket  # noqa: E402

RAW = ROOT / "data" / "raw" / "handout"
OUT = ROOT / "data" / "processed"

# Ответ админа на жалобу явно говорит об обжаловании наказания / жалобе на администратора, хотя
# verdict извлечён как other/rejected (≈2300 жалоб train; до этой LF ~750 из них размечались как dm/insult/...).
ADMIN_APPEAL_REPLY_RX = re.compile(
    r"наказани\w*\s+(было\s+)?выдан\w*\s+(верно|правильно|правомерно)|будет\s+проведена\s+(соответствующая\s+)?беседа|"
    r"наказани\w*\s+(будет\s+)?снят|жалоб\w*\s+на\s+(администратора|имя\s+администратора)|скриншот\w*\s+(выдачи|выданного)\s+наказани|"
    r"администратор\w*\s+(ошибся|выдал\s+неверное)|jazo\s+yechilmaydi|jazo\w*\s+to.g.ri\s+berilgan",
    re.IGNORECASE,
)

ADMIN_REPORT_CUES = [
    (re.compile(r"слежу\s+за\s+игроком|работаю\s+по\s+(вашей\s+)?жалоб|при\s+наличии\s+доказательств|пишите\s+жалобу|подайте\s+жалобу|kuzatyapman\s+o'?yinchi", re.I), "report.complaint"),
    (re.compile(r"уточните\s+(ваш\s+)?(вопрос|жалоб)|оффтоп|мут\s+репорта|не\s+предоставляем|приятной\s+игры\.?$|aniqlashtiring", re.I), "report.offtopic"),
    (re.compile(r"эвакуатор|кнопк\w*\s+на\s+спидометр|телепорт\w*\s+(закрыт|на\s+мероприят)|не\s+телепортируем|перезайдите|баг,?\s+ожидайте", re.I), "report.help"),
    (re.compile(r"изучите\s+правила|ознакомьтесь\s+с\s+правилами|узнайте\s+(самостоятельно|сами)|навигатор|в\s+правительстве|на\s+стоянке|попытайтесь\s+узнать", re.I), "report.question"),
]


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def admin_text(t: dict) -> str:
    return " ".join(m.get("text") or "" for m in t.get("messages") or [] if m.get("role") == "admin")


def weak_label(t: dict, text: str) -> tuple[str | None, str]:
    kind = t.get("kind")
    if kind == "complaint":
        verdict = t.get("verdict")
        if verdict in {"appeal_approved", "appeal_rejected"}:
            return "complaint.admin", "LF1_verdict_appeal"
        if ADMIN_APPEAL_REPLY_RX.search(admin_text(t)):
            return "complaint.admin", "LF1b_admin_reply_appeal"
        cat, _ = cr.complaint_rule_category(text, t.get("rule_refs_raw"))
        if cat == "complaint.admin":
            return cat, "LF2_text_appeal"
        if re.search(r"1\.1\.2", admin_text(t)):
            return "complaint.invalid", "LF3_admin_112"
        vcat = cr.violation_category(t.get("violation"), t.get("punishment"))
        if vcat:
            return vcat, "LF4_violation"
        if cat:
            return cat, "LF5_text" if cat != "complaint.invalid" else "LF6_empty"
        return None, "abstain"
    cat, _ = cr.report_rule_category(text)
    if cat:
        return cat, "LF1_text"
    a = admin_text(t)
    for rx, c in ADMIN_REPORT_CUES:
        if rx.search(a):
            return c, "LF2_admin_reply"
    return None, "abstain"


def process(src: Path, dst: Path) -> collections.Counter:
    stats = collections.Counter()
    with dst.open("w", encoding="utf-8") as f:
        for t in read_jsonl(src):
            n = normalize_ticket(t.get("messages") or [], t.get("region"), t.get("server_lang"))
            label, source = weak_label(t, n.clean_text)
            stats[(t["kind"], label, source)] += 1
            f.write(json.dumps({
                "ticket_id": t["ticket_id"], "kind": t["kind"], "region": t.get("region"),
                "text": n.clean_text, "rule_refs_raw": t.get("rule_refs_raw") or "",
                "images_count": t.get("images_count") or 0, "created_at": t.get("created_at"),
                "reporter_hash": t.get("reporter_hash"), "reported_hash": t.get("reported_hash"),
                "verdict": t.get("verdict"), "weak_label": label, "lf": source,
            }, ensure_ascii=False) + "\n")
    return stats


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = ["# Weak supervision: покрытие labeling functions\n",
              "Сгенерировано `scripts/prepare_dataset.py`. Только счётчики.\n"]
    for name, src in [("train", RAW / "tickets_train.jsonl.gz"), ("valid", RAW / "tickets_valid.jsonl")]:
        stats = process(src, OUT / f"{name}_weak.jsonl")
        total = sum(stats.values())
        labeled = sum(v for (k, l, s), v in stats.items() if l)
        report.append(f"## {name}: {total} записей, размечено {labeled} ({100 * labeled / total:.1f}%)\n")
        report.append("| kind | label | LF | n |\n|---|---|---|---|")
        for (k, l, s), v in sorted(stats.items(), key=lambda x: (x[0][0], str(x[0][1]), -x[1])):
            report.append(f"| {k} | {l} | {s} | {v} |")
        report.append("")
        print(f"{name}: {total} records, labeled {labeled}", file=sys.stderr)
    (ROOT / "reports" / "weak_supervision.md").write_text("\n".join(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
