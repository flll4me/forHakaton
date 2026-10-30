"""Словари и правила для категорий taxonomy.json.

Используются в двух местах:
1. scripts/prepare_dataset.py — weak supervision: правила по тексту игрока + поля исхода
   (verdict, violation, punishment, ответ админа) дают слабые метки для обучения.
2. classifier.py во время инференса — высокоточные правила по тексту игрока для
   категорий, где ошибка дорогая (платежи, аккаунт, апелляции), поверх ML-модели.

Словари собраны по частотным формулировкам из tickets_train (ru, uz-латиница, kk).
"""
from __future__ import annotations

import re

from app.knowledge_base.rules_index import parse_rule_refs


def _rx(*parts: str) -> re.Pattern:
    return re.compile("|".join(parts), re.IGNORECASE | re.UNICODE)


# --- безусловная эскалация (официальное задание: платежи, апелляции банов, взлом) ---
PAYMENT_RX = _rx(
    r"донат", r"задонат", r"\bdonat", r"\bdanat", r"оплат", r"оплочен", r"платеж", r"платёж", r"\bto'?lov", r"tolov",
    r"списал[иао]?\b.*(деньг|руб|тг|тенге|сум)", r"деньги\s+(списал|ушл|снял)", r"не\s+приш(ел|ёл|ло|ли)\s+(донат|гк|коин|рубин)",
    r"\bгк\b.*не\s+приш", r"не\s+зачисл", r"возврат\s+(денег|средств)", r"верните\s+деньги", r"refund", r"\bpayment",
    r"uzcard", r"humo", r"visa", r"mastercard", r"kaspi", r"каспи", r"чек\s+об\s+оплате",
)
ACCOUNT_HACK_RX = _rx(
    r"взлом", r"взломал", r"взломали", r"\bhack(ed)?\b", r"buzib", r"buzish", r"бұзып", r"украли\s+акк",
    r"угнали\s+акк", r"не\s+могу\s+(зайти|войти)\s+в\s+акк", r"(сменил|поменял)и?\s+пароль", r"пароль",
    r"\bparol", r"доступ\s+к\s+акк", r"потерял\s+доступ", r"akkauntimga\s+kira\s+olmayapman",
)
ACCOUNT_LOSS_RX = _rx(
    r"пропал[аио]?\s+(машин|тачк|авто|дом|бизнес|деньг|вещ|скин|акс|имущ|кейс|оруж)", r"(машин|дом|деньги|вещи|бизнес)\w*\s+пропал",
    r"исчез(ла|ли|ло)?\s+(машин|дом|деньг|вещ)", r"не\s+появил", r"нет\s+на\s+почте", r"слетел[аио]?\s+(машин|дом|бизнес|прогресс|уровень)",
    r"обнулил", r"откатил", r"yo'?qolib\s+qoldi", r"yo'?qoldi",
)
BAN_APPEAL_RX = _rx(
    r"разбан", r"снимите\s+(бан|наказ|блок|мут|варн|деморган|джайл)", r"сними\s+(бан|наказ|блок)", r"снять\s+(бан|наказ|блок)",
    r"не\s*согла[сш]", r"\bне\s+согл\b", r"несогл", r"ne\s+soglas", r"обжал", r"апелляц", r"амнист",
    r"(выдал|дал|дали|выдали|выдала|дала)\s+мне\s+(бан|мут|варн|деморган|джайл|предупрежд|блокировк|наказ)",
    r"мне\s+(выдал|дал|дали|выдали|выдала|дала)\s+(бан|мут|варн|деморган|джайл|предупрежд|блокировк|наказ)",
    r"^\W*(без\s+причины\s+)?(дали|выдали)\s+(бан|мут|варн|деморган|джайл|предупрежд)", r"не\s+үшін\s+бан", r"неге\s+бан",
    r"(за\s+что|почему)\s+(мне\s+)?(бан|мут|варн|наказ|блок)", r"без\s+причины\s+(дали|выдали|забанил|посадил)",
    r"забанил[аи]?\s+(меня\s+)?(просто|без|ни\s+за)", r"бан\s+(снимите|уберите)", r"когда\s+бан\s+сним",
    r"незаслуж", r"необоснован", r"неверн\w*\s+наказ", r"nohaq", r"noxaq", r"naqonuniy", r"jazoni?\s+(olib|yech)",
    r"bandan\s+(chiqar|och)", r"ban\w*\s+yech", r"mut\w*\s+yech", r"banga\s+norozi", r"blok\w*\s+(och|yech)",
    r"бан\s+не\s+за\s+что", r"\bунбан", r"unban",
    r"меня\s+посадил[аи]?", r"посадили\s+(меня|ещ[её]|без)", r"(варн|бан|мут|деморган|джайл|предупреждение)\s+(мне\s+)?дали",
    r"(это|был[аио]?)\s+(рп|иц|ic|rp)[\s-]*(процесс|ситуац)", r"в\s+каком\s+месте\s+(было|я)", r"ничего\s+не\s+нарушал",
    r"to'?liq\s+rasm", r"toliq\s+rasm",
)
# Явные признаки «другого нарушения» (никнейм, 18+, номера машин, ивенты, правила фракций)
OTHER_RX = _rx(
    r"18\+", r"\bсекс", r"порн", r"наркот", r"никнейм", r"\bник\b\s+с\s+оск", r"номер\w*\s+знак", r"ивент", r"мероприяти",
    r"в\s+рабочей\s+форме", r"\bмз\b", r"заявк\w*\s+на",
)
BUG_OR_QUESTION_RX = _rx(
    r"\bлвл\b", r"уров\w*\s+не\s+(подним|раст)", r"не\s+работает", r"\bбаг\b", r"knopka", r"\bкак\s+(купить|продать|получить)",
)
_OTHER_RULE_IDS = re.compile(r"^(1\.[1-8]|2\.[25]|3\.\d|5\.(1|1\.1|[5-9])|4\.[35-8]|4\.3\.1)$")
ADMIN_RX = _rx(
    r"\bадмин\w*\s+(выдал|дал|забанил|посадил|наказал|не\s+прав|превыш|злоупотреб|оскорб|вмешал)", r"администратор\w*\s+(выдал|дал|забанил|посадил|наказал)",
    r"злоупотреб\w*\s+(полномоч|властью|админ)", r"превыш\w*\s+полномоч", r"жалоба\s+на\s+админ", r"admin\w*\s+(menga\s+)?(nohaq|ban\s+berdi|mut\s+berdi)",
    r"lavozimini\s+suiste", r"әкімші", r"скриншот\s+выдачи\s+наказания",
    # формулировки из жалоб train, на которые админ отвечал как на обжалование / жалобу на администратора
    r"фарм\w*\s+реп", r"пересмотр", r"оспор", r"наказани\w*\s+для\s+админ", r"(неадекватн|некомпетентн)\w*\s+(админ|одм)",
    r"\b(админ|одмен|одмин)\w*\s+(игнор|не\s+помог|не\s+отвеча|неадекват|покрыва|блат|не\s+сле)", r"ответил\s+но\s+не\s+помог",
    r"(?<!его\s)(?<!ему\s)(?<!ей\s)(выдал|выдала|выдали)\s+(мне\s+)?(мут|варн|бан|деморган|джайл|предупрежд)",
    r"(варн|мут|бан|деморган|джайл)\s+(без\s+причин|не\s+за\s+что|ни\s+за\s+что)", r"за\s+что\s+(варн|мут|бан|наказ)",
    r"\bа?дми\w*\s+(meni|mani|menga|manga)", r"admin\w*\s+(nohaq|noto.g.ri|asossiz)",
)

# --- нарушения в жалобах ---
INSULT_RX = _rx(
    r"\bоск", r"оскорб", r"обзыв", r"мат(ом|ерит|ерился|ы|ерн)?\b", r"матер", r"упом\w*\s+род", r"\bродн(и|ю|ых|я)\b", r"\bрод\b",
    r"мамк|мать\s+|маму|матери", r"расизм", r"нацизм", r"нации", r"унижа", r"\bosk\b", r"\bock\b", r"\bosc\b", r"osk\.?rod",
    r"so'?k(di|ish|inish|yapti|vot|indi|kanda|di)", r"soʻk", r"sokdi", r"sokyapti", r"so‘k", r"haqorat", r"xaqorat", r"onadan", r"onamni",
    r"сөкті", r"балағат", r"қорла", r"\bинсульт", r"insult", r"провоцир",
)
CHEAT_RX = _rx(
    r"\bсофт", r"\bчит", r"аимбот", r"\bаим\b", r"\baim", r"\bвх\b", r"\bwh\b", r"сборк", r"\bсб\b", r"\bsoft\b", r"cheat", r"\bhack\b",
    r"летает", r"телепортир\w+\s+по", r"спидхак", r"speedhack", r"через\s+стен", r"сквозь\s+стен", r"сбрил", r"\bбаг\w*\s+исп",
    r"\b5\.2\b", r"бессмерт", r"godmode", r"автоприцел|авто\s*прицел", r"bag\s+q",
)
DM_RX = _rx(
    r"\bдм\b", r"\bdm\b", r"\bдб\b", r"\bdb\b", r"\bтк\b", r"\btk\b", r"\bск\b", r"\bsk\b", r"\bпг\b", r"\bpg\b", r"дмз",
    r"убил\w*", r"убива", r"убийств", r"без\s+причин", r"просто\s+так", r"сбил", r"задавил", r"застрелил", r"стреля",
    r"\bзз\b", r"\bzz\b", r"\bgz\b", r"\bгз\b", r"зел[её]н\w*\s+зон", r"нон\s*рп", r"non\s*rp", r"\bnrp\b", r"\bнрп\b", r"уход\s+от\s+рп",
    r"наручник", r"кишан", r"kishan", r"o'?ldirdi", r"oʻldirdi", r"oldirdi", r"uldr", r"o'?ldir", r"otib\s+(qo|ket|tashla)", r"otyap",
    r"sababsiz", r"be\s*sabab", r"mesabab", r"өлтірді", r"арестов\w*\s+без", r"посадил\w*\s+без", r"верёвк|веревк", r"похищ",
)
SCAM_RX = _rx(
    r"обман", r"\bскам", r"заскам", r"\bscam", r"кинул", r"развел", r"развёл", r"не\s+отдал", r"не\s+вернул", r"вымога",
    r"мошен", r"украл", r"aldab", r"aldadi", r"aldash", r"pul\s+undur", r"\bpora\b", r"кража", r"\b4\.[12]\b",
)
RMT_RX = _rx(
    r"реклам", r"за\s+реал", r"реальные\s+деньги", r"\brmt\b", r"продает\s+акк", r"продаёт\s+акк", r"продажа\s+акк",
    r"тг\s*канал", r"телеграм", r"\btg\b", r"другой\s+сервер", r"другого\s+сервера", r"\b2\.3\b", r"\b4\.4\b",
)

# --- репорты ---
HELP_RX = _rx(
    r"застрял", r"застряла", r"текстур", r"переверн", r"переверт", r"перевёрн", r"флип", r"\bflip", r"вытащ", r"вытяни", r"\bтп\b",
    r"телепорт", r"тепни", r"тпни", r"\bпорт\b", r"\btp\b", r"наручник", r"\bнары\b", r"развяж", r"разв'?яж", r"в\s+воде", r"в\s+речк",
    r"утонул", r"\bбаг\b", r"забаг", r"\bbug", r"не\s+могу\s+(выйти|выбраться|встать|двигат|выехать)", r"выбраться", r"где\s+я",
    r"спавн", r"релог", r"рестарт", r"не\s+заводит", r"почин", r"эвакуатор", r"stuck", r"tiqilib", r"chiqolma", r"ag'?darilib",
    r"тұрып\s+қалдым", r"розв'?яз", r"зав'?яз", r"измерени", r"под\s+карт",
)
REPORT_COMPLAINT_RX = _rx(
    r"\bдм\b", r"\bdm\b", r"\bдб\b", r"\bdb\b", r"\bтк\b", r"\btk\b", r"\bск\b", r"\bsk\b", r"\bсофт", r"\bчит", r"оск", r"оскорб",
    r"мешает", r"приста", r"спамит", r"флудит", r"\bспек\b", r"следите\s+за\s+ним", r"убива", r"убил",
    r"забаньте", r"накажите", r"посади", r"музык", r"\bmg\b", r"\bмг\b", r"\bnrp\b", r"нрп", r"нон\s*рп", r"\bid\s*\d", r"айди\s*\d",
    r"^\s*\d{1,4}\s*$", r"\d{1,4}\s*(дм|дб|тк|ск|оск|софт)", r"(дм|дб|тк|ск|оск|софт|спек)\s*\d{1,4}", r"sokyapti", r"haqorat",
)
QUESTION_RX = _rx(
    r"^\s*(как|где|что|почему|зачем|когда|сколько|можно\s+ли|можно|какой|какая|какие|кто|куда|откуда|чем)\b", r"\?\s*$",
    r"подскажите", r"скажите", r"как\s+(мне\s+)?(купить|продать|получить|устроит|вступить|выйти|сделать|заработать|перевести)",
    r"\b(qanday|qayerda|qachon|nima\s+uchun|nimaga|nega|qancha)\b", r"\b(how|where|what|why|when)\b", r"\b(қалай|қайда|неге|қашан)\b",
)

_FILLER_WORDS = {
    "ау", "ауу", "алло", "ало", "привет", "прив", "здравствуйте", "здрасте", "салам", "салом", "salom", "hi", "hello",
    "ку", "url", "handle", "digits", "email", "эй", "эээ", "ээ", "ok", "ок", "a", "а",
}
_TOKEN_RX = re.compile(r"[^\W\d_]+", re.UNICODE)
BARE_HELP_RX = re.compile(
    r"^\s*(админ\w*|адмін\w*|одмин|адм|admin\w*)?\s*(помоги\w*|помаги\w*|памаги\w*|памоги\w*|помги\w*|поможіт\w*|памагіт\w*|памагі|хелп|help|sos|сос|помощь|спасите|yordam|yordam\s+ber\w*|кӛмек|көмек)\s*(пж|пжл|пожалуйста|плиз|pls|please|iltimos|admin\w*|адми\w*|є)*\s*[!.?]*\s*$",
    re.IGNORECASE,
)

# violation → категория (из шаблона ответа админа; только для weak supervision)
VIOLATION_MAP = [
    (re.compile(r"\b5\.2\b|софт|soft|чит|сборк", re.I), "complaint.cheat"),
    (re.compile(r"\b2\.[146]\b|оск|упом|мат\b", re.I), "complaint.insult"),
    (re.compile(r"\b4\.[12]\b|обман|скам|scam|мошен", re.I), "complaint.scam"),
    (re.compile(r"\b2\.3\b|\b4\.4\b|реклам|rmt", re.I), "complaint.rmt_ads"),
    (re.compile(r"\b5\.[34]\b|\bdm\b|\bдм\b|\bdb\b|\bдб\b|\btk\b|\bтк\b|\bsk\b|\bск\b|\bpg\b|gz|гз|зелен|nonrp|нон\s*рп|nrp|нрп|убийств", re.I), "complaint.dm"),
    (re.compile(r"\b(1\.[1-8]|2\.[25]|3\.\d|5\.[15-9]|4\.[35-8])\b|fw|фв|пго|опг|оплз|пгвд", re.I), "complaint.other"),
]


def has_any(rx: re.Pattern, text: str) -> bool:
    return bool(rx.search(text or ""))


def is_content_free(text: str) -> bool:
    """Только знаки, цифры, маски и приветствия — без содержательных слов."""
    words = [w.lower() for w in _TOKEN_RX.findall(text or "")]
    return all(w in _FILLER_WORDS or len(set(w)) == 1 for w in words)


def complaint_rule_category(text: str, rule_refs_raw: str | None = None) -> tuple[str | None, str]:
    """Правило по тексту игрока (и заявленному пункту) для жалоб. Возвращает (категория|None, причина)."""
    t = text or ""
    refs = rule_refs_raw or ""
    if has_any(BAN_APPEAL_RX, t) or has_any(ADMIN_RX, t):
        return "complaint.admin", "обжалование наказания / действия администратора"
    joined = f"{t} {refs}"
    claimed = parse_rule_refs(joined)
    hits = {
        "complaint.cheat": has_any(CHEAT_RX, joined),
        "complaint.scam": has_any(SCAM_RX, joined),
        "complaint.rmt_ads": has_any(RMT_RX, joined),
        "complaint.insult": has_any(INSULT_RX, joined),
        "complaint.dm": has_any(DM_RX, joined),
    }
    for rid in claimed:
        if rid in {"2.1", "2.4", "2.6"}:
            hits["complaint.insult"] = True
        elif rid == "5.2":
            hits["complaint.cheat"] = True
        elif rid in {"4.1", "4.2"}:
            hits["complaint.scam"] = True
        elif rid in {"2.3", "4.4"}:
            hits["complaint.rmt_ads"] = True
        elif rid in {"5.3", "5.4"} or rid.startswith("GZ"):
            hits["complaint.dm"] = True
    fired = [k for k, v in hits.items() if v]
    if not fired and (has_any(OTHER_RX, joined) or any(" " in rid or _OTHER_RULE_IDS.match(rid) for rid in claimed)):
        # заявлен пункт из «прочих» разделов (ОПГ, FW, ПГО, 3.x, 1.6 ...) или явная тема 18+/никнейм/ивент
        return "complaint.other", "заявлен пункт / тема вне основных категорий"
    if len(fired) == 1:
        return fired[0], "ключевые слова / заявленный пункт"
    if len(fired) > 1:
        return None, "несколько категорий: " + ", ".join(fired)
    if is_content_free(t) and not claimed:
        return "complaint.invalid", "пустой текст жалобы"
    if has_any(QUESTION_RX, t) or has_any(BUG_OR_QUESTION_RX, t) or has_any(PAYMENT_RX, t):
        return "complaint.invalid", "вопрос/баг/платёж вместо жалобы"
    return None, "нет совпадений"


def report_rule_category(text: str) -> tuple[str | None, str]:
    t = text or ""
    if has_any(PAYMENT_RX, t):
        return "report.payment", "платёж/донат"
    if has_any(ACCOUNT_HACK_RX, t) or has_any(ACCOUNT_LOSS_RX, t):
        return "report.account", "доступ/взлом/пропажа имущества"
    if is_content_free(t) or BARE_HELP_RX.match(t):
        return "report.offtopic", "сообщение без содержания"
    if has_any(HELP_RX, t):
        return "report.help", "просьба о помощи (застрял/телепорт/баг)"
    if has_any(REPORT_COMPLAINT_RX, t):
        return "report.complaint", "жалоба на игрока через репорт"
    if has_any(QUESTION_RX, t):
        return "report.question", "вопрос"
    return None, "нет совпадений"


def violation_category(violation: str | None, punishment: str | None = None) -> str | None:
    v = f"{violation or ''} {punishment or ''}".strip()
    if not v:
        return None
    for rx, cat in VIOLATION_MAP:
        if rx.search(v):
            return cat
    return None
