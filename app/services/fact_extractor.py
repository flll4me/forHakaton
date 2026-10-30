"""Извлечение структурированных фактов из обращения.

Правила (ТЗ 5.4 + официальное задание):
1. Никнейм из `reported_nick` имеет приоритет.
2. images_count/attachments > 0 — это наличие вложений, а не подтверждение нарушения.
3. Номера правил из текста и rule_refs_raw — ЗАЯВЛЕННЫЕ игроком; проверяются по базе,
   несуществующие не попадают в rule_refs, а уходят в invalid_rule_refs.
4. Факты не превращаются в вердикты. Отсутствующие сведения → null.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.knowledge_base.rules_index import parse_rule_refs, rules_by_id

# Синтетические ники в выгрузке: Name_Surname12, Bek_Nazarov_, _Anastasia_Wilson_
NICK_RE = re.compile(r"(?<![\w@])_?[A-Z][A-Za-z]{1,20}_[A-Z][A-Za-z]{1,20}\d{0,4}_?(?!\w)")
NICK_CUE_RE = re.compile(r"(?:ник(?:ом|нейм)?|nick(?:name)?|игрок(?:а)?|под\s+ником)\s*[:\-]?\s*([A-Za-z0-9_]{3,24})", re.IGNORECASE)
SERVER_RE = re.compile(
    r"(?:(?:на|с)\s+)?(?:сервер[еа]?|server|serverda|сервері)\s*(?:№|#|n|номер)?\s*(\d{1,2})\b|\b(\d{1,2})\s*(?:-?(?:й|ой|ом|м))?\s*(?:сервер|server)",
    re.IGNORECASE,
)
PLAYER_ID_RE = re.compile(r"(?:\bid\s*|айди\s*|айді\s*|игрок\s*:?\s*)?(?<![\d.])(\d{1,4})(?![\d.])")
EVIDENCE_WORDS_RE = re.compile(r"скрин|фрапс|fraps|видео|video|доказ|доки|запис|isbot|dalil|rasm|videofiks|screenshot|скриншот", re.IGNORECASE)
PUNISH_REQUEST_RE = re.compile(r"(бан|мут|варн|деморган|джайл|ban|mut|blok)\w*\s*(на\s*)?(\d{1,3})\s*(дн|день|дня|мин|час|kun|daqiqa|minut)", re.IGNORECASE)


@dataclass
class Facts:
    nickname: str | None = None
    rule_refs: list[str] = field(default_factory=list)          # существующие в базе, заявленные игроком
    has_evidence: bool = False
    server: str | None = None
    # дополнительные факты (не часть контракта, видны в debug/дашборде)
    invalid_rule_refs: list[str] = field(default_factory=list)  # заявлены, но в правилах нет
    nickname_source: str | None = None
    attachments: int = 0
    links: int = 0
    evidence_mentioned: bool = False
    player_ids: list[str] = field(default_factory=list)
    requested_punishment: str | None = None
    accused_commented: bool = False
    confidence: float = 0.0

    def contract(self) -> dict:
        return {"nickname": self.nickname, "rule_refs": self.rule_refs, "has_evidence": self.has_evidence, "server": self.server}


def _attachments(messages: list[dict], images_count: int | None) -> int:
    from_msgs = sum(int(m.get("attachments") or 0) for m in messages if (m.get("role") or "player") == "player")
    return max(from_msgs, int(images_count or 0))


def extract_facts(
    kind: str,
    text: str,
    messages: list[dict],
    reported_nick: str | None = None,
    rule_refs_raw: str | None = None,
    images_count: int | None = None,
    accused_text: str = "",
) -> Facts:
    f = Facts()
    # 1. никнейм: поле лаунчера → ник в тексте → «ник X»
    if reported_nick:
        f.nickname, f.nickname_source = reported_nick.strip(), "reported_nick"
    else:
        m = NICK_RE.search(text)
        if m:
            f.nickname, f.nickname_source = m.group(0), "text_pattern"
        else:
            m = NICK_CUE_RE.search(text)
            if m and not m.group(1).isdigit():
                f.nickname, f.nickname_source = m.group(1), "text_cue"

    # 2. заявленные пункты правил: поле «пункт правил» + текст
    claimed = []
    for rid in parse_rule_refs(rule_refs_raw) + parse_rule_refs(text):
        if rid not in claimed:
            claimed.append(rid)
    index = rules_by_id()
    f.rule_refs = [r for r in claimed if r in index]
    f.invalid_rule_refs = [r for r in claimed if r not in index]

    # 3. доказательства: вложения и ссылки (а не слова «скрины есть»)
    f.attachments = _attachments(messages, images_count)
    f.links = text.count("<url>")
    f.evidence_mentioned = bool(EVIDENCE_WORDS_RE.search(text))
    f.has_evidence = f.attachments > 0 or f.links > 0

    # 4. сервер
    m = SERVER_RE.search(text)
    if m:
        f.server = m.group(1) or m.group(2)

    # 5. доп. факты
    if kind == "report":
        stripped = parse_rule_refs(text)
        t = text
        for rid in stripped:
            t = t.replace(rid, " ")
        f.player_ids = list(dict.fromkeys(PLAYER_ID_RE.findall(NICK_RE.sub(" ", t))))[:3]
    m = PUNISH_REQUEST_RE.search(text)
    if m:
        f.requested_punishment = m.group(0)
    f.accused_commented = bool(accused_text.strip())

    found = sum([f.nickname is not None, bool(f.rule_refs), f.has_evidence, f.server is not None])
    base = 0.9 if f.nickname_source == "reported_nick" else 0.6
    f.confidence = round(min(1.0, base * (0.5 + 0.125 * found) + (0.1 if not f.invalid_rule_refs else 0.0)), 3)
    return f
