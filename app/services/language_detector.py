"""Определение языка обращения: ru | kk | uz | en | es | pt | und.

Готовые детекторы (fastText lid.176, Lingua) были рассмотрены: Lingua не поддерживает
узбекский, а у fastText нет колёс под Python 3.14/Windows. Кроме того, на коротких
игровых сообщениях («дм 248», «yordam», «+») они ошибаются. Поэтому используется
прозрачный скоринг: письменность + специфичные буквы + словари частых слов,
собранные по реальным обращениям train, + слабый приор по серверу (region).

Поле `lang` из выгрузки сюда не передаётся как истина — оно шумное
(украинский текст размечен как kk, транслит — как lat/pt).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SUPPORTED = ("ru", "kk", "uz", "en", "es", "pt", "und")

_KK_ONLY = set("әөүұңһі")          # казахские буквы, которых нет в русском (і есть и в украинском)
_KK_STRONG = set("әөүұңһ")          # без і
_UZ_CYR_ONLY = set("ўҳ")            # узбекская кириллица
_SHARED_KK_UZ = set("қғ")
_UK_ONLY = set("їєґ")               # украинские (если не были казахскими заменами)

_WORD_RE = re.compile(r"[^\W\d_]+(?:[ʻʼ'’`][^\W\d_]+)*", re.UNICODE)

LEXICON: dict[str, set[str]] = {
    "kk": {
        "мен", "сен", "ол", "бұл", "осы", "және", "жоқ", "бар", "керек", "маған", "саған", "мені", "сені", "оны",
        "ойыншы", "көмек", "көмектесіңіз", "рахмет", "неге", "қалай", "қайда", "болды", "кетті", "деп", "емес",
        "берді", "өлтірді", "ұрды", "сөкті", "балағаттады", "күнге", "мына", "анау", "ақша",
        "көлік", "менің", "сенің", "біз", "сіз", "сіздер", "өтінемін", "тұрып", "қалдым", "жатыр", "алды",
        "шығып", "кірдім", "үшін", "ғой", "дейді", "айтты",
    },
    "uz": {
        "men", "sen", "u", "va", "uchun", "bilan", "iltimos", "yordam", "yordam bering", "qildi", "qilib", "qilyapti",
        "berdi", "oldirdi", "o'ldirdi", "oʻldirdi", "sokdi", "so'kdi", "soʻkdi", "sokyapti", "kerak", "yoq", "yo'q",
        "bor", "emas", "nima", "qanday", "nega", "menga", "meni", "mani", "manga", "onadan", "onamni", "akkaunt",
        "mashina", "mashinam", "pul", "edi", "qoldi", "boldi", "bo'ldi", "boʻldi", "chora", "korila", "ko'rilsin",
        "jazo", "admin", "adminlar", "assalomu", "alaykum", "aleykum", "salom", "rahmat", "hozir", "qayerda",
        "ham", "bu", "shu", "kim", "nimaga", "o'yinchi", "oyinchi", "ketti", "qilmoqda", "deb", "bering", "beringlar",
        "qildim", "kelmadi", "kelmayapti", "keldi", "to'lov", "tolov", "to'ladim", "toladim",  # платёжные обращения (аудит 2026-10-30)
        "tashladi", "urdi", "otib", "qoydi", "be", "sabab", "besabab", "mesabab", "sababsiz", "dalil", "shikoyat",
        "kun", "kunga", "butunlay", "butunlayga", "berilar", "ochib", "chiqdi", "tiqilib", "qoldim", "chiqolmayapman",
    },
    "en": {
        "the", "is", "are", "help", "me", "please", "pls", "my", "i", "you", "he", "she", "why", "how", "what",
        "where", "player", "kill", "killed", "stuck", "car", "can", "cant", "can't", "not", "this", "that", "was",
        "for", "no", "reason", "insulted", "cheat", "cheater", "hack", "hacked", "account", "money", "give", "want",
        "it", "to", "and", "of", "in", "with", "him", "her", "they", "have", "has", "did", "do", "admin", "get",
    },
    "es": {
        "ayuda", "por", "favor", "que", "el", "la", "los", "las", "me", "mi", "es", "porque", "como", "jugador",
        "mato", "mató", "coche", "auto", "carro", "esta", "está", "gracias", "hola", "una", "uno", "para", "con",
        "pero", "sin", "razon", "razón", "insulto", "insultó", "dinero", "cuenta", "ayudame", "ayúdame", "atascado",
        "estoy", "y", "de", "en", "lo", "se", "su", "muy", "hay", "yo",
    },
    "pt": {
        "ajuda", "por", "favor", "você", "voce", "vc", "não", "nao", "meu", "minha", "é", "o", "um", "uma",
        "jogador", "matou", "carro", "obrigado", "olá", "ola", "oi", "está", "esta", "para", "com", "mas", "sem",
        "motivo", "xingou", "dinheiro", "conta", "me", "ajudem", "preso", "eu", "de", "em", "do", "da", "que",
        "isso", "muito", "tá", "ta",
    },
    # русский транслитом → ru
    "ru_translit": {
        "pomogite", "pomogi", "pomagite", "menya", "mne", "ubil", "ubili", "pozhaluysta", "pojaluysta", "pozhalujsta",
        "ne", "kak", "gde", "chto", "net", "da", "admin", "adm", "on", "ona", "zastryal", "zastral", "perevernite",
        "mashina", "mashinu", "oskorblyaet", "oskorbil", "bez", "prichiny", "prichini", "igrok", "tut", "vot",
        "sdelayte", "spasibo", "privet", "chel", "blin", "ya", "ty", "vy", "eto", "pochemu", "nakazanie", "ban",
    },
    "uk": {
        "що", "мене", "будь", "ласка", "допоможіть", "поможіть", "адмін", "адміни", "він", "вона", "це", "але",
        "тому", "який", "дуже", "чому", "мені", "потрібно", "щоб", "знав", "дати", "розмовляти", "ви", "його",
    },
}

# Однозначно русские слова: если они есть, украинские/казахские маркеры слабее.
_RU_CUES = {
    "меня", "мне", "помогите", "пожалуйста", "что", "как", "где", "почему", "это", "он", "она", "без", "причины",
    "оскорбил", "убил", "игрок", "жалоба", "админ", "застрял", "переверните", "наказание", "не", "я", "и", "в",
}


@dataclass
class LangResult:
    lang: str
    confidence: float
    scores: dict = field(default_factory=dict)
    note: str = ""


# Игровой сленг латиницей/кириллицей одинаков у всех языковых групп и язык не определяет.
GAME_SLANG = {
    "dm", "db", "tk", "sk", "pg", "mg", "rp", "nrp", "nonrp", "op", "gz", "zz", "osk", "rod", "soft", "fraps",
    "warn", "ban", "mute", "jail", "kop", "cop", "ck", "vip", "id", "admin", "adm", "sos", "fv", "gk", "jb",
    "дм", "дб", "тк", "ск", "пг", "мг", "рп", "нрп", "оп", "гз", "зз", "жб", "фв", "гк", "варн", "мут", "id",
}
_NICK_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b")


def strip_non_linguistic(text: str) -> str:
    """Удаляет маски, синтетические ники (Name_Surname12), номера правил и цифры."""
    t = text
    for tok in ("<url>", "<email>", "<handle>", "<digits>"):
        t = t.replace(tok, " ")
    t = _NICK_RE.sub(" ", t)
    return re.sub(r"\d+", " ", t)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(strip_non_linguistic(text).lower())


def detect_language(text: str, region: str | None = None) -> LangResult:
    """Определяет язык нормализованного (не lower-case) текста игрока."""
    all_words = _words(text)
    words = [w for w in all_words if w not in GAME_SLANG]
    letters = [c for c in "".join(all_words) if c.isalpha()]
    if not all_words or len(letters) < 2:
        return LangResult("und", 0.0, note="нет языковых признаков")
    if not words:
        # только сленг: кириллический сленг почти всегда пишут русскоязычные игроки
        if all(any("Ѐ" <= c <= "ӿ" for c in w) for w in all_words):
            return LangResult("ru", 0.5, note="только игровой сленг кириллицей")
        return LangResult("und", 0.2, note="только игровой сленг")
    letters = [c for c in "".join(words) if c.isalpha()]

    n_cyr = sum(1 for c in letters if "Ѐ" <= c <= "ӿ")
    n_lat = sum(1 for c in letters if c.isascii() or c in "ʻʼ’çãõáéíóúâêôñü")
    chars = set("".join(letters))
    wset = set(words)
    region = (region or "").lower()
    scores: dict[str, float] = {}

    if n_cyr >= n_lat:
        ru_hits = len(wset & _RU_CUES)
        kk_hits = len(wset & LEXICON["kk"])
        uk_hits = len(wset & LEXICON["uk"])
        scores["kk"] = 3.0 * len(chars & _KK_STRONG) + 1.0 * kk_hits + (0.5 if "і" in chars else 0.0)
        if not chars & (_KK_STRONG | _SHARED_KK_UZ) and kk_hits < 2:
            scores["kk"] = min(scores["kk"], 0.5)  # без казахских букв нужен хотя бы двухсловный сигнал
        scores["uz"] = 3.0 * len(chars & _UZ_CYR_ONLY)
        scores["uk"] = 2.0 * len(chars & _UK_ONLY) + 1.0 * uk_hits + (1.0 if "і" in chars and not chars & _KK_STRONG else 0.0)
        scores["ru"] = 1.0 + 1.0 * ru_hits + (0.5 if chars & set("ыэъё") else 0.0)
        if chars & _SHARED_KK_UZ:
            scores["kk"] += 1.0 if region != "uz" else 0.0
            scores["uz"] += 1.0 if region == "uz" else 0.0
        if region == "kz":
            scores["kk"] += 0.5
        best = max(scores, key=scores.get)
        total = sum(max(v, 0) for v in scores.values()) or 1.0
        conf = scores[best] / total
        if best == "uk":
            # Украинский не входит в список языков игры; админы отвечают таким игрокам по-русски.
            return LangResult("ru", round(min(0.9, conf), 3), scores, note="украинский текст → ответ на ru")
        if best == "ru" and scores["ru"] <= 1.0 and len(words) <= 2:
            conf = max(conf, 0.6)  # короткий кириллический текст без специфичных букв — русский по умолчанию
        return LangResult(best, round(min(0.99, conf), 3), scores)

    # латиница
    def hits(lang: str) -> float:
        return float(len(wset & LEXICON[lang]))

    apostrophe_uz = 1.5 if re.search(r"\b(?:o|g)[ʻʼ'’`]", text.lower()) else 0.0
    scores["uz"] = hits("uz") + apostrophe_uz + (1.0 if region == "uz" else 0.0)
    scores["en"] = hits("en")
    scores["es"] = hits("es") + (1.0 if "ñ" in chars else 0.0)
    scores["pt"] = hits("pt") + (1.5 if chars & set("ãõç") else 0.0)
    scores["ru"] = hits("ru_translit")
    best = max(scores, key=scores.get)
    if scores[best] <= 0.0:
        if region == "uz":
            return LangResult("uz", 0.4, scores, note="латиница без маркеров, приор по серверу uz")
        return LangResult("und", 0.2, scores, note="латиница без языковых маркеров")
    total = sum(scores.values()) or 1.0
    conf = scores[best] / total
    # «ban», «admin», «dm» встречаются во всех языках; одно слово — слабый сигнал
    if scores[best] < 1.5 and len(words) <= 2:
        conf = min(conf, 0.6)
    return LangResult(best, round(min(0.99, conf), 3), scores)
