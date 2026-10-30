"""Ручная разметка контрольной выборки (200 обращений из tickets_valid.jsonl).

Разметку выполнил разработчик (Claude) вручную по описаниям и примерам taxonomy.json,
читая текст игрока, поле «пункт правил» и ответ администратора. Это НЕ экспертная
разметка Grand Mobile; метрики на ней — оценка, а не результат на истинной разметке.
Разметка сделана до обучения классификатора, чтобы не подстраиваться под его ответы.

Индексы соответствуют data/processed/control_sample.jsonl (стратифицированная выборка,
seed=2026, квоты по полю lang: ru 95, uz 40, lat 20, kk 20→все доступные, und 10, pt 8, en 7).

Политика разметки:
- обжалование собственного наказания / претензия к действиям админа → complaint.admin;
- жалоба без содержательного текста → категория по заявленному пункту правил / сленгу;
- текст не является жалобой (вопрос, баг, бессвязный текст) → complaint.invalid;
- «помогите», «yordam», «+», «help» без деталей → report.offtopic (пример из taxonomy);
- язык: реальный язык текста игрока; украинский → ru (не входит в языки игры);
  только числа/латинский сленг/ссылки → und; кириллический сленг («оск», «дм») → ru.
- приоритет выводится из категории по правилам taxonomy.json (см. app/services/classifier.py
  priority_for) + ручные исключения PRIORITY_OVERRIDES.

Запуск: python scripts/control_labels_v1.py  → data/processed/control_labels.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

C = {
    "ins": "complaint.insult", "cht": "complaint.cheat", "dm": "complaint.dm", "scm": "complaint.scam",
    "rmt": "complaint.rmt_ads", "adm": "complaint.admin", "oth": "complaint.other", "inv": "complaint.invalid",
    "q": "report.question", "hlp": "report.help", "rc": "report.complaint", "acc": "report.account",
    "pay": "report.payment", "off": "report.offtopic",
}

CATEGORY = (
    "ins ins adm oth ins off rc ins ins scm "      # 0-9
    "off oth ins ins off ins off ins off adm "     # 10-19
    "adm ins off off scm ins rc inv ins adm "      # 20-29
    "off ins adm adm adm inv adm dm rc ins "       # 30-39
    "off hlp ins dm ins ins q scm ins oth "        # 40-49
    "oth adm inv dm dm off dm rc dm ins "          # 50-59
    "ins oth adm off ins off adm ins off hlp "     # 60-69
    "rc adm scm adm adm scm dm off off ins "       # 70-79
    "oth off adm dm ins cht ins inv adm inv "      # 80-89
    "dm adm oth rc ins cht adm rc hlp adm "        # 90-99
    "dm ins cht dm inv cht cht dm adm ins "        # 100-109
    "ins adm off dm scm ins off ins ins adm "      # 110-119
    "adm oth adm adm adm hlp cht ins off adm "     # 120-129
    "dm off rc cht ins dm cht dm ins dm "          # 130-139
    "rc adm cht oth hlp ins adm dm cht scm "       # 140-149
    "scm adm inv ins dm dm ins dm ins ins "        # 150-159
    "off adm cht ins dm scm adm q ins hlp "        # 160-169
    "dm ins oth adm ins ins ins hlp dm ins "       # 170-179
    "ins adm ins inv hlp adm dm dm off ins "       # 180-189
    "off adm ins acc adm ins dm dm dm off"         # 190-199
).split()

LANG_UZ = {8, 9, 15, 21, 43, 44, 56, 58, 63, 72, 75, 76, 77, 87, 89, 94, 99, 101, 105, 107, 111, 112, 114, 120,
           122, 130, 139, 145, 147, 149, 151, 155, 160, 170, 176, 179, 181, 187, 188, 191, 195, 196}
LANG_KK = {31, 33, 121, 141, 178, 185}
LANG_EN = {27, 68, 190, 199}
LANG_UND = {1, 3, 11, 13, 16, 30, 37, 40, 48, 55, 65, 67, 92, 95, 116, 126, 135, 137, 143, 171, 182, 186}

# Ручные исключения приоритета (taxonomy: p0 — платежи, доступ к аккаунту, угрозы вне игры).
PRIORITY_OVERRIDES = {87: "p0"}  # жалоба о невозможности оплаты доната → платежи

# Неоднозначные случаи (разметчик не уверен) — отдельно в отчёте.
AMBIGUOUS = {2, 12, 22, 26, 72, 79, 80, 93, 103, 114, 119, 155, 167, 172, 176, 193, 196}


def main() -> None:
    assert len(CATEGORY) == 200, len(CATEGORY)
    sample = [json.loads(l) for l in (ROOT / "data/processed/control_sample.jsonl").read_text(encoding="utf-8").splitlines()]
    out = ROOT / "data/processed/control_labels.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row, code in zip(sample, CATEGORY):
            i = row["i"]
            lang = "uz" if i in LANG_UZ else "kk" if i in LANG_KK else "en" if i in LANG_EN else "und" if i in LANG_UND else "ru"
            f.write(json.dumps({
                "i": i, "ticket_id": row["ticket_id"], "category": C[code], "lang": lang,
                "priority_override": PRIORITY_OVERRIDES.get(i), "ambiguous": i in AMBIGUOUS,
            }, ensure_ascii=False) + "\n")
    print(f"written {out}")


if __name__ == "__main__":
    main()
