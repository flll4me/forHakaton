"""Загрузка rules.json и taxonomy.json, канонические идентификаторы пунктов правил.

В rules.json номера пунктов (`points[].ref`) повторяются в разных разделах (1.1 есть почти
везде), поэтому для цитирования используется `rule_id` = код раздела + ref.
Коды разделов взяты из того, как их пишут администраторы в ответах (частоты по train):
«2.1 ОП» (20k), «1.2 GZ» (1.3k), «1.2.7 ППЖ» (1k), «ПГО», «FW/ФВ», «ОПГ», «ОПЛЗ», «ПГВД».
Для общих правил код пустой — `2.1`, как в taxonomy.json и в контракте задания.
Для правил жалоб используется код `ЖБ` из taxonomy.json (`ЖБ 1.2.7`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings

# category из rules.json → (канонический код, синонимы, которые пишут игроки и админы)
SECTION_CODES: dict[str, tuple[str, tuple[str, ...]]] = {
    "Общие правила игры": ("", ("ОП", "OP", "ОР", "OП", "ОБЩИХ", "ОБЩИЕ")),
    "Правила жалоб": ("ЖБ", ("ЖБ", "ППЖ", "ПЖ", "JB")),
    "Правила нахождения в Зелёной Зоне": ("GZ", ("GZ", "ГЗ", "ЗЗ", "ZZ", "GREEN")),
    "Правила для лидеров и их заместителей": ("ОПЛЗ", ("ОПЛЗ", "ОПЛИЗ", "ПЛ")),
    "Правила государственных организаций": ("ПГО", ("ПГО", "PGO", "НПГО")),
    "Правила государственной волны и департамента": ("ПГВД", ("ПГВД", "ПГВИД", "ГВ")),
    "Правила семейных мероприятий": ("FW", ("FW", "ФВ")),
    "Правила для ОПГ": ("ОПГ", ("ОПГ", "OPG", "ПОПГ")),
    "Правила войны за бизнес": ("BW", ("BW", "ВЗБ")),
    "Правила битвы за фургон": ("FURA", ("FURA", "ФУРА", "БЗФ")),
}
_ALIAS_TO_CODE = {alias.upper(): code for code, aliases in SECTION_CODES.values() for alias in aliases}
_ALIAS_TO_CODE.update({code: code for code, _ in SECTION_CODES.values() if code})

# Сопоставление названия раздела по ключевому слову (устойчиво к пробелам/похожим символам)
_SECTION_KEYWORDS = [
    ("общие правила", ""), ("жалоб", "ЖБ"), ("зелён", "GZ"), ("зелен", "GZ"), ("лидер", "ОПЛЗ"),
    ("государственных организаций", "ПГО"), ("волны", "ПГВД"), ("семейных", "FW"), ("опг", "ОПГ"),
    ("бизнес", "BW"), ("фургон", "FURA"),
]


def section_code(name: str) -> str | None:
    if name in SECTION_CODES:
        return SECTION_CODES[name][0]
    low = " ".join(name.lower().split())
    for kw, code in _SECTION_KEYWORDS:
        if kw in low:
            return code
    return None


@dataclass(frozen=True)
class RulePoint:
    rule_id: str      # канонический идентификатор для цитирования
    ref: str          # как в rules.json
    section: str      # название раздела
    section_code: str
    text: str

    def short(self, n: int = 220) -> str:
        t = self.text.split(" | ")[0]
        return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def make_rule_id(section_code: str, ref: str) -> str:
    return f"{section_code} {ref}".strip()


def _is_header(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(text) < 60 and letters and sum(c.isupper() for c in letters) / len(letters) > 0.7


@lru_cache(maxsize=4)
def load_rules(path: str | None = None) -> tuple[RulePoint, ...]:
    p = Path(path or get_settings().rules_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    out: list[RulePoint] = []
    for sec in data.get("rules", []):
        name = sec.get("category", "")
        code = section_code(name)
        if code is None:
            code = "".join(w[0] for w in name.split()).upper()  # неизвестный раздел: аббревиатура названия
        for pt in sec.get("points") or []:
            ref, text = str(pt.get("ref", "")).strip(), (pt.get("text") or "").strip()
            if not ref or not text or _is_header(text):
                continue  # заголовки подразделов («ОБЩИЕ ПОЛОЖЕНИЯ:») не являются правилами
            out.append(RulePoint(make_rule_id(code, ref), ref, name, code, text))
    return tuple(out)


_TERM_RE = re.compile(r"•\s*([^•(\n]+?)\s*\(сокращенно\s+([^)]+)\)\s*[—-]\s*([^|•\n]+)")


@lru_cache(maxsize=4)
def load_terms(path: str | None = None) -> dict[str, dict]:
    """Глоссарий из раздела без пунктов («RP термины»): {"DM": {"name": "DeathMatch", "definition": ...}}.
    У терминов нет номеров, поэтому они НЕ попадают в cited — только в текст как определение."""
    p = Path(path or get_settings().rules_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    terms: dict[str, dict] = {}
    for sec in data.get("rules", []):
        if sec.get("points"):
            continue
        for name, abbr, definition in _TERM_RE.findall(sec.get("text") or ""):
            terms[abbr.strip().upper()] = {"name": name.strip(), "definition": definition.strip().rstrip("."), "section": sec.get("category", "")}
    return terms


@lru_cache(maxsize=4)
def rules_by_id(path: str | None = None) -> dict[str, RulePoint]:
    return {r.rule_id: r for r in load_rules(path)}


@lru_cache(maxsize=4)
def load_taxonomy(path: str | None = None) -> dict:
    p = Path(path or get_settings().taxonomy_path)
    return json.loads(p.read_text(encoding="utf-8"))


def category_codes() -> list[str]:
    return [c["code"] for c in load_taxonomy()["categories"]]


def category_info(code: str) -> dict:
    for c in load_taxonomy()["categories"]:
        if c["code"] == code:
            return c
    return {}


def expand_taxonomy_rule(ref: str) -> list[str]:
    """taxonomy.json ссылается на правила как «2.1», «ЖБ 1.1.2» или «GZ» (весь раздел)."""
    ref = ref.strip()
    index = rules_by_id()
    if ref in index:
        return [ref]
    code = _ALIAS_TO_CODE.get(ref.upper())
    if code is not None:
        return [r.rule_id for r in load_rules() if r.section_code == code]
    parsed = parse_rule_refs(ref)
    return [x for x in parsed if x in index]


def category_rule_ids(code: str) -> list[str]:
    ids: list[str] = []
    for ref in category_info(code).get("rules", []):
        for rid in expand_taxonomy_rule(ref):
            if rid not in ids:
                ids.append(rid)
    return ids


_REF_RE = re.compile(
    # только точка как разделитель: «3,5,7» у игроков — это перечисление, а не пункт 3.5.7
    r"(?<![\d.,])(\d{1,2}(?:\.\d{1,2}){1,2})(?![\d.,]\d)\s*[-–]?\s*([A-Za-zА-Яа-яЁё]{2,5})?",
)


def parse_rule_refs(raw: str | None) -> list[str]:
    """Извлекает номера пунктов из произвольного текста игрока: «2.1 ОП», «2.1оп», «1.2 gz», «1.2.7 ППЖ».

    Возвращает канонические rule_id В ТОМ ВИДЕ, КАК ЗАЯВИЛ ИГРОК (без проверки существования).
    Номер без кода раздела трактуется как общие правила. Одиночные числа («1», «3», «7»)
    номерами пунктов не считаются.
    """
    if not raw:
        return []
    out: list[str] = []
    for num, suffix in _REF_RE.findall(raw):
        code = ""
        if suffix:
            code = _ALIAS_TO_CODE.get(suffix.upper(), "")
        rid = make_rule_id(code, num)
        if rid not in out:
            out.append(rid)
    return out
