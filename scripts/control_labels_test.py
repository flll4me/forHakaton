"""Ручная разметка НЕЗАВИСИМОЙ тестовой выборки: 100 обращений valid (seed=777, без стратификации,
не пересекается с контрольной выборкой v1). Размечена разработчиком (Claude) по той же политике,
что и scripts/control_labels_v1.py, ПОСЛЕ завершения настройки правил и модели и без просмотра
предсказаний. На этой выборке ничего не настраивалось — это основная (честная) оценка.

Запуск: python scripts/control_labels_test.py → data/processed/test_labels.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from control_labels_v1 import C  # noqa: E402  (те же сокращения категорий)

ROOT = Path(__file__).resolve().parents[1]

CATEGORY = (
    "oth cht off adm ins cht inv dm off scm "      # 0-9
    "rc rc ins dm dm adm dm inv ins hlp "          # 10-19
    "adm ins dm q ins ins ins adm ins dm "         # 20-29
    "scm dm oth off ins ins ins adm dm ins "       # 30-39
    "adm inv adm oth off ins dm adm off off "      # 40-49
    "ins rc adm adm ins dm ins adm dm ins "        # 50-59
    "adm hlp dm adm inv dm ins dm ins dm "         # 60-69
    "adm dm ins adm adm off ins ins adm adm "      # 70-79
    "off ins ins off dm off rc adm off ins "       # 80-89
    "adm dm adm ins rc off q ins ins scm"          # 90-99
).split()

LANG_UZ = {15, 16, 60, 67, 75, 87, 95, 98}
LANG_UND = {8, 13, 32, 44, 85}
AMBIGUOUS = {14, 15, 19, 33, 41, 43, 49, 51, 61, 62, 86}


def main() -> None:
    assert len(CATEGORY) == 100, len(CATEGORY)
    sample = [json.loads(l) for l in (ROOT / "data/processed/test_sample.jsonl").read_text(encoding="utf-8").splitlines()]
    out = ROOT / "data/processed/test_labels.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row, code in zip(sample, CATEGORY):
            i = row["i"]
            lang = "uz" if i in LANG_UZ else "und" if i in LANG_UND else "ru"
            f.write(json.dumps({"i": i, "ticket_id": row["ticket_id"], "category": C[code], "lang": lang,
                                "priority_override": None, "ambiguous": i in AMBIGUOUS}, ensure_ascii=False) + "\n")
    print(f"written {out}")


if __name__ == "__main__":
    main()
