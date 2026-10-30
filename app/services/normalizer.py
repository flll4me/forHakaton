"""Предобработка текста обращения.

Сохраняет два представления: original_text (как пришло, без изменений) и
normalized_text (для ML и RAG). Исходные данные не модифицируются.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Игра работает в cp1251: на kz-серверах казахские буквы набираются похожими символами.
KAZAKH_CP1251_MAP = {
    "ќ": "қ", "Ќ": "Қ",
    "ґ": "ғ", "Ґ": "Ғ",
    "ѕ": "ң", "Ѕ": "Ң",
    "є": "ә", "Є": "Ә",
    "ј": "ө", "Ј": "Ө",
    "ў": "ұ", "Ў": "Ұ",
    "ї": "ү", "Ї": "Ү",
    # расширение таблицы задания: ѓ (cp1251 0x83) визуально неотличима от ґ и встречается в той же роли
    "ѓ": "ғ", "Ѓ": "Ғ",
}
_KK_TABLE = str.maketrans(KAZAKH_CP1251_MAP)

# Символы, которые в реальном тексте встречаются почти только как cp1251-замены казахских букв.
# (є, ї, ґ — также украинские буквы, ў — узбекская, поэтому сами по себе не сигнал.)
_STRONG_KK_SUBSTITUTES = set("ќЌѕЅјЈѓЃ")
_ALL_KK_SUBSTITUTES = set(KAZAKH_CP1251_MAP)
_KK_LETTERS = set("әғқңөұүһӘҒҚҢӨҰҮҺ")

# Частые казахские слова, набранные с заменами или без (признак, что замены надо применять).
_KK_CUE_WORDS = {
    # «мені» не включено: это слово есть и в украинском
    "мен", "сен", "бұл", "осы", "және", "жоқ", "бар", "керек", "маған", "саған",
    "ойыншы", "өтінемін", "көмектесіңіз", "рахмет", "неге", "қалай", "болды", "кетті", "деп",
}

_MASK_TOKENS = {"<url>": " URLTOKEN ", "<email>": " EMAILTOKEN ", "<handle>": " HANDLETOKEN ", "<digits>": " DIGITSTOKEN "}
_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f​-‏⁠﻿]")
_WS_RE = re.compile(r"\s+")
_REPEAT_RE = re.compile(r"(.)\1{3,}")


def fix_kazakh_cp1251(text: str) -> str:
    """Безусловная замена cp1251-подобных символов на казахские буквы (ќ→қ, ґ→ғ, ...)."""
    return text.translate(_KK_TABLE)


def should_fix_kazakh(text: str, region: str | None = None, server_lang: str | None = None) -> bool:
    """Решает, нужно ли применять казахские замены.

    Замены применяются, если есть хотя бы один из признаков:
    - сервер kz (region=kz или server_lang=kk) и в тексте есть символы-заменители;
    - в тексте есть «сильные» заменители (ќ, ѕ, ј), которые в ru/uk/uz не используются;
    - в тексте есть заменители и одновременно настоящие казахские буквы или казахские слова.
    Иначе (украинские є/ї/ґ, узбекская ў) текст не трогаем.
    """
    chars = set(text)
    if not chars & _ALL_KK_SUBSTITUTES:
        return False
    if (region or "").lower() == "kz" or (server_lang or "").lower() == "kk":
        return True
    if chars & _STRONG_KK_SUBSTITUTES:
        return True
    if chars & _KK_LETTERS:
        return True
    words = set(re.findall(r"\w+", fix_kazakh_cp1251(text.lower())))
    return bool(words & _KK_CUE_WORDS)


def clean_text(text: str | None) -> str:
    """Unicode NFC, удаление управляющих символов, схлопывание пробелов."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r", "\n")
    return _WS_RE.sub(" ", text).strip()


def ml_text(text: str) -> str:
    """Текст для ML/RAG: нижний регистр, маски-токены, ё→е, сжатие повторов букв."""
    t = text.lower()
    for k, v in _MASK_TOKENS.items():
        t = t.replace(k, v)
    t = t.replace("ё", "е")
    t = _REPEAT_RE.sub(r"\1\1", t)
    return _WS_RE.sub(" ", t).strip()


@dataclass
class NormalizedTicket:
    original_text: str          # склейка сообщений игрока как есть
    clean_text: str             # очищенный текст (казахские замены применены, если нужно)
    normalized_text: str        # для ML/RAG
    accused_text: str           # комментарий обвиняемого (не используется как слова заявителя)
    kazakh_fixed: bool
    is_empty: bool


def player_messages(messages: list[dict]) -> list[str]:
    """Только сообщения заявителя. reported_player и admin не считаются словами игрока."""
    return [m.get("text") or "" for m in messages if (m.get("role") or "player") == "player"]


def normalize_ticket(messages: list[dict], region: str | None = None, server_lang: str | None = None) -> NormalizedTicket:
    original = "\n".join(player_messages(messages))
    accused = "\n".join(m.get("text") or "" for m in messages if m.get("role") == "reported_player")
    cleaned = clean_text(original)
    fixed = should_fix_kazakh(cleaned, region, server_lang)
    if fixed:
        cleaned = fix_kazakh_cp1251(cleaned)
    normalized = ml_text(cleaned)
    return NormalizedTicket(
        original_text=original,
        clean_text=cleaned,
        normalized_text=normalized,
        accused_text=clean_text(accused),
        kazakh_fixed=fixed,
        is_empty=not re.search(r"\w", cleaned.replace("<url>", "").replace("<digits>", "")),
    )
