"""Генерация черновика ответа на языке игрока.

Типовые обращения — детерминированные шаблоны (ru, kk, uz, en, es, pt). Шаблоны не
выносят вердиктов и не обещают наказания/разбана/возврата: решение остаётся за админом.
Цитаты берутся ТОЛЬКО из результатов RAG и проходят validate_citations().

Процедурные нормы (срок рассмотрения, требования к доказательствам и к жалобе на
наказание) тоже цитируются через RAG: для каждой нормы выполняется отдельный поиск
по правилам, и пункт цитируется, только если поиск вернул его в top-3.

LLM (если включён) используется для сложных случаев — длинных жалоб и вопросов по
правилам — с JSON-схемой ответа; результат проходит те же проверки, при ошибке —
шаблон (fallback).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.knowledge_base.rules_index import category_rule_ids, load_terms, rules_by_id
from app.services import category_rules as cr
from app.services.llm_provider import LLMProvider, NullProvider, truncate_to_tokens
from app.services.rag import Retrieved, RulesRetriever

TEMPLATE_LANGS = ("ru", "kk", "uz", "en", "es", "pt")
SERVER_DEFAULT_LANG = {"uz": "uz", "kz": "ru", "cis": "ru"}  # на kz-серверах админы отвечают преимущественно по-русски

# процедурная норма → (поисковый запрос к правилам, ожидаемый пункт)
PROCEDURES = {
    "deadline": ("сроки рассмотрения жалоб обрабатываются в течение 48 часов", "ЖБ 1.4.1"),
    "evidence": ("жалоба рассматривается только при наличии объективных доказательств нарушения", "ЖБ 1.1.1"),
    "punish_screenshot": ("жалоба должна содержать скриншот наказания описание доказательства никнейм", "ЖБ 1.3.1"),
    "not_complaint": ("раздел жалоб не предназначен для вопросов технических проблем предложений", "ЖБ 1.1.2"),
}

RULE_NEEDED = {"complaint.insult", "complaint.cheat", "complaint.dm", "complaint.scam", "complaint.rmt_ads",
               "complaint.admin", "complaint.invalid", "complaint.other"}

TOPIC = {
    "complaint.insult": {"ru": "оскорбления", "kk": "қорлау", "uz": "haqorat", "en": "insults", "es": "insultos", "pt": "insultos"},
    "complaint.cheat": {"ru": "стороннее ПО", "kk": "бөгде бағдарламалар", "uz": "uchinchi tomon dasturlari",
                        "en": "third-party software", "es": "software de terceros", "pt": "software de terceiros"},
    "complaint.dm": {"ru": "убийство или урон без причины", "kk": "себепсіз өлтіру немесе зақым келтіру",
                     "uz": "sababsiz o‘ldirish yoki zarar yetkazish", "en": "killing or damage without a reason",
                     "es": "matar o dañar sin motivo", "pt": "matar ou causar dano sem motivo"},
    "complaint.scam": {"ru": "мошенничество", "kk": "алаяқтық", "uz": "firibgarlik", "en": "fraud", "es": "fraude", "pt": "fraude"},
    "complaint.rmt_ads": {"ru": "реклама и сделки за реальные деньги", "kk": "жарнама және нақты ақшаға сауда",
                          "uz": "reklama va real pulga savdo", "en": "advertising and real-money trading",
                          "es": "publicidad y comercio con dinero real", "pt": "publicidade e comércio com dinheiro real"},
}

T = {
    "ru": {
        "hello": "Здравствуйте!",
        "accepted_nick": "Ваша жалоба на игрока {nick} принята.", "accepted": "Ваша жалоба принята.",
        "rules": "Описанная ситуация может относиться к {rules} ({topic}).", "rules_notopic": "Описанная ситуация может относиться к {rules}.",
        "check": "Администратор проверит приложенные доказательства и примет решение в соответствии с правилами.",
        "deadline": "Жалобы обычно рассматриваются в течение 48 часов ({r}).",
        "no_evidence": "Для рассмотрения нужны доказательства — скриншоты или видео ({r}).",
        "admin": "Ваше обращение по действиям администратора принято и будет передано старшей администрации для проверки.",
        "admin_rules": "Жалобы на действия администрации подаются в течение 2 дней с момента наказания ({r}).",
        "punish_screenshot": "Пожалуйста, убедитесь, что приложен скриншот выдачи наказания ({r}).",
        "admin_decision": "Решение о снятии или изменении наказания принимает администрация после проверки.",
        "invalid": "Ваше обращение не похоже на жалобу на нарушение правил.",
        "review": "Ваше обращение принято и передано администратору для проверки.",
        "not_complaint": "Раздел жалоб не предназначен для вопросов и технических проблем ({r}) — для этого есть Discord-сервер игры и техническая поддержка.",
        "question": "Ваш вопрос передан администратору, он ответит вам. С правилами игры можно ознакомиться в лаунчере.",
        "question_rule": "Возможно, вам поможет {rules}: «{snippet}».",
        "help": "Администратор посмотрит ситуацию и постарается помочь.",
        "help_car": "Если машина перевернулась или застряла, можно воспользоваться кнопкой эвакуатора на спидометре.",
        "report_complaint": "Спасибо за сообщение, администратор проверит ситуацию.",
        "report_complaint_id": "Спасибо за сообщение, администратор проверит ситуацию с игроком (ID {pid}).",
        "report_evidence": "Если у вас есть доказательства нарушения, подайте жалобу через лаунчер.",
        "account": "Ваше обращение по доступу к аккаунту или имуществу передано специалисту.",
        "account_safety": "Никому не сообщайте пароль и данные аккаунта.",
        "payment": "Ваше обращение по платежу передано специалисту.",
        "payment_keep": "Пожалуйста, сохраните чек или номер транзакции.",
        "offtopic": "Пожалуйста, уточните ваш вопрос или жалобу, чтобы администратор мог помочь.",
        "rule_word": "п. {id}",
    },
    "kk": {
        "hello": "Сәлеметсіз бе!",
        "accepted_nick": "Сіздің {nick} ойыншысына қатысты шағымыңыз қабылданды.", "accepted": "Сіздің шағымыңыз қабылданды.",
        "rules": "Сипатталған жағдай ережелердің {rules} тармағына ({topic}) қатысты болуы мүмкін.", "rules_notopic": "Сипатталған жағдай ережелердің {rules} тармағына қатысты болуы мүмкін.",
        "check": "Әкімші тіркелген дәлелдерді тексеріп, ережелерге сәйкес шешім қабылдайды.",
        "deadline": "Шағымдар әдетте 48 сағат ішінде қаралады ({r}).",
        "no_evidence": "Шағымды қарау үшін дәлелдер қажет — скриншот немесе бейне ({r}).",
        "admin": "Әкімшінің әрекетіне қатысты өтінішіңіз қабылданды және аға әкімшілікке тексеруге жіберіледі.",
        "admin_rules": "Әкімшіліктің әрекетіне шағым жаза берілген сәттен бастап 2 күн ішінде беріледі ({r}).",
        "punish_screenshot": "Жазаның берілген сәтінің скриншоты тіркелгеніне көз жеткізіңіз ({r}).",
        "admin_decision": "Жазаны алу немесе өзгерту туралы шешімді әкімшілік тексеруден кейін қабылдайды.",
        "invalid": "Өтінішіңіз ереже бұзушылық туралы шағымға ұқсамайды.",
        "review": "Өтінішіңіз қабылданды және тексеру үшін әкімшіге жіберілді.",
        "not_complaint": "Шағымдар бөлімі сұрақтар мен техникалық мәселелерге арналмаған ({r}) — ол үшін ойынның Discord-сервері мен техникалық қолдау бар.",
        "question": "Сұрағыңыз әкімшіге жіберілді, ол сізге жауап береді. Ойын ережелерімен лаунчерде танысуға болады.",
        "question_rule": "Сізге {rules} тармағы көмектесуі мүмкін: «{snippet}».",
        "help": "Әкімші жағдайды қарап, көмектесуге тырысады.",
        "help_car": "Көлік аударылып немесе тұрып қалса, спидометрдегі эвакуатор батырмасын қолдануға болады.",
        "report_complaint": "Хабарламаңыз үшін рахмет, әкімші жағдайды тексереді.",
        "report_complaint_id": "Хабарламаңыз үшін рахмет, әкімші ойыншыға (ID {pid}) қатысты жағдайды тексереді.",
        "report_evidence": "Бұзушылықтың дәлелі болса, лаунчер арқылы шағым беріңіз.",
        "account": "Аккаунтқа кіру немесе мүлікке қатысты өтінішіңіз маманға жіберілді.",
        "account_safety": "Құпия сөзіңізді және аккаунт деректерін ешкімге айтпаңыз.",
        "payment": "Төлемге қатысты өтінішіңіз маманға жіберілді.",
        "payment_keep": "Төлем түбіртегін немесе транзакция нөмірін сақтап қойыңыз.",
        "offtopic": "Әкімші көмектесе алуы үшін сұрағыңызды немесе шағымыңызды нақтылап жазыңыз.",
        "rule_word": "{id}",
    },
    "uz": {
        "hello": "Assalomu alaykum!",
        "accepted_nick": "{nick} o‘yinchisiga nisbatan shikoyatingiz qabul qilindi.", "accepted": "Shikoyatingiz qabul qilindi.",
        "rules": "Tasvirlangan holat qoidalarning {rules}-bandiga ({topic}) taalluqli bo‘lishi mumkin.", "rules_notopic": "Tasvirlangan holat qoidalarning {rules}-bandiga taalluqli bo‘lishi mumkin.",
        "check": "Administrator ilova qilingan dalillarni tekshirib, qoidalarga muvofiq qaror qabul qiladi.",
        "deadline": "Shikoyatlar odatda 48 soat ichida ko‘rib chiqiladi ({r}).",
        "no_evidence": "Shikoyatni ko‘rib chiqish uchun dalillar kerak — skrinshot yoki video ({r}).",
        "admin": "Administrator harakatlari bo‘yicha murojaatingiz qabul qilindi va tekshiruv uchun katta administratsiyaga yuboriladi.",
        "admin_rules": "Administratsiya harakatlari ustidan shikoyat jazo berilgan paytdan boshlab 2 kun ichida beriladi ({r}).",
        "punish_screenshot": "Iltimos, jazo berilgan paytdagi skrinshot ilova qilinganiga ishonch hosil qiling ({r}).",
        "admin_decision": "Jazoni olib tashlash yoki o‘zgartirish haqidagi qarorni administratsiya tekshiruvdan so‘ng qabul qiladi.",
        "invalid": "Murojaatingiz qoidabuzarlik haqidagi shikoyatga o‘xshamaydi.",
        "review": "Murojaatingiz qabul qilindi va tekshiruv uchun administratorga yuborildi.",
        "not_complaint": "Shikoyatlar bo‘limi savollar va texnik muammolar uchun mo‘ljallanmagan ({r}) — buning uchun o‘yinning Discord-serveri va texnik yordam xizmati mavjud.",
        "question": "Savolingiz administratorga yuborildi, u sizga javob beradi. O‘yin qoidalari bilan launcherda tanishishingiz mumkin.",
        "question_rule": "Sizga {rules}-band yordam berishi mumkin: «{snippet}».",
        "help": "Administrator vaziyatni ko‘rib chiqib, yordam berishga harakat qiladi.",
        "help_car": "Agar mashina ag‘darilgan yoki tiqilib qolgan bo‘lsa, spidometrdagi evakuator tugmasidan foydalanishingiz mumkin.",
        "report_complaint": "Xabaringiz uchun rahmat, administrator vaziyatni tekshiradi.",
        "report_complaint_id": "Xabaringiz uchun rahmat, administrator o‘yinchi (ID {pid}) bilan bog‘liq vaziyatni tekshiradi.",
        "report_evidence": "Qoidabuzarlik dalillari bo‘lsa, launcher orqali shikoyat yuboring.",
        "account": "Akkauntga kirish yoki mulk bo‘yicha murojaatingiz mutaxassisga yuborildi.",
        "account_safety": "Parolingiz va akkaunt ma’lumotlarini hech kimga bermang.",
        "payment": "To‘lov bo‘yicha murojaatingiz mutaxassisga yuborildi.",
        "payment_keep": "Iltimos, to‘lov chekini yoki tranzaksiya raqamini saqlab qo‘ying.",
        "offtopic": "Administrator yordam bera olishi uchun savolingiz yoki shikoyatingizni aniqroq yozing.",
        "rule_word": "{id}",
    },
    "en": {
        "hello": "Hello!",
        "accepted_nick": "Your complaint about player {nick} has been received.", "accepted": "Your complaint has been received.",
        "rules": "The situation you describe may fall under rule {rules} ({topic}).", "rules_notopic": "The situation you describe may fall under rule {rules}.",
        "check": "An administrator will review the attached evidence and make a decision according to the rules.",
        "deadline": "Complaints are usually reviewed within 48 hours ({r}).",
        "no_evidence": "Evidence is required for review — screenshots or video ({r}).",
        "admin": "Your request about an administrator's action has been received and will be passed to senior administration for review.",
        "admin_rules": "Complaints about administration actions are filed within 2 days of the punishment ({r}).",
        "punish_screenshot": "Please make sure a screenshot of the punishment being issued is attached ({r}).",
        "admin_decision": "The administration decides whether to lift or change a punishment after the review.",
        "invalid": "Your request does not look like a complaint about a rule violation.",
        "review": "Your request has been received and passed to an administrator for review.",
        "not_complaint": "The complaints section is not intended for questions or technical problems ({r}) — please use the game's Discord server or technical support.",
        "question": "Your question has been passed to an administrator, who will answer you. The game rules are available in the launcher.",
        "question_rule": "Rule {rules} may help: «{snippet}».",
        "help": "An administrator will look into the situation and try to help.",
        "help_car": "If your car has flipped or got stuck, you can use the tow truck button on the speedometer.",
        "report_complaint": "Thank you for the report, an administrator will check the situation.",
        "report_complaint_id": "Thank you for the report, an administrator will check the situation with the player (ID {pid}).",
        "report_evidence": "If you have evidence of a violation, please file a complaint via the launcher.",
        "account": "Your request about account access or property has been passed to a specialist.",
        "account_safety": "Do not share your password or account details with anyone.",
        "payment": "Your payment request has been passed to a specialist.",
        "payment_keep": "Please keep the receipt or transaction number.",
        "offtopic": "Please clarify your question or complaint so that an administrator can help.",
        "rule_word": "{id}",
    },
    "es": {
        "hello": "¡Hola!",
        "accepted_nick": "Hemos recibido tu queja sobre el jugador {nick}.", "accepted": "Hemos recibido tu queja.",
        "rules": "La situación que describes puede corresponder a la regla {rules} ({topic}).", "rules_notopic": "La situación que describes puede corresponder a la regla {rules}.",
        "check": "Un administrador revisará las pruebas adjuntas y tomará una decisión según las reglas.",
        "deadline": "Las quejas suelen revisarse en un plazo de 48 horas ({r}).",
        "no_evidence": "Para revisarla se necesitan pruebas: capturas de pantalla o vídeo ({r}).",
        "admin": "Hemos recibido tu solicitud sobre la acción de un administrador y se enviará a la administración superior para su revisión.",
        "admin_rules": "Las quejas sobre acciones de la administración se presentan dentro de los 2 días siguientes a la sanción ({r}).",
        "punish_screenshot": "Asegúrate de adjuntar una captura de pantalla de la sanción ({r}).",
        "admin_decision": "La administración decide si retira o modifica la sanción después de la revisión.",
        "invalid": "Tu solicitud no parece una queja sobre una infracción de las reglas.",
        "review": "Hemos recibido tu solicitud y se ha enviado a un administrador para su revisión.",
        "not_complaint": "La sección de quejas no está destinada a preguntas ni problemas técnicos ({r}); para eso están el servidor de Discord del juego y el soporte técnico.",
        "question": "Tu pregunta se ha enviado a un administrador, que te responderá. Las reglas del juego están disponibles en el launcher.",
        "question_rule": "Puede ayudarte la regla {rules}: «{snippet}».",
        "help": "Un administrador revisará la situación e intentará ayudarte.",
        "help_car": "Si tu coche se ha volcado o se ha quedado atascado, puedes usar el botón de grúa en el velocímetro.",
        "report_complaint": "Gracias por el aviso, un administrador revisará la situación.",
        "report_complaint_id": "Gracias por el aviso, un administrador revisará la situación con el jugador (ID {pid}).",
        "report_evidence": "Si tienes pruebas de la infracción, presenta una queja a través del launcher.",
        "account": "Tu solicitud sobre el acceso a la cuenta o los bienes se ha enviado a un especialista.",
        "account_safety": "No compartas tu contraseña ni los datos de tu cuenta con nadie.",
        "payment": "Tu solicitud sobre el pago se ha enviado a un especialista.",
        "payment_keep": "Por favor, guarda el recibo o el número de transacción.",
        "offtopic": "Por favor, aclara tu pregunta o queja para que un administrador pueda ayudarte.",
        "rule_word": "{id}",
    },
    "pt": {
        "hello": "Olá!",
        "accepted_nick": "Recebemos sua denúncia sobre o jogador {nick}.", "accepted": "Recebemos sua denúncia.",
        "rules": "A situação descrita pode se enquadrar na regra {rules} ({topic}).", "rules_notopic": "A situação descrita pode se enquadrar na regra {rules}.",
        "check": "Um administrador vai verificar as provas anexadas e tomar uma decisão de acordo com as regras.",
        "deadline": "As denúncias costumam ser analisadas em até 48 horas ({r}).",
        "no_evidence": "Para a análise são necessárias provas: capturas de tela ou vídeo ({r}).",
        "admin": "Recebemos sua solicitação sobre a ação de um administrador e ela será encaminhada à administração superior para verificação.",
        "admin_rules": "Denúncias sobre ações da administração são enviadas em até 2 dias após a punição ({r}).",
        "punish_screenshot": "Verifique se a captura de tela da punição está anexada ({r}).",
        "admin_decision": "A administração decide sobre retirar ou alterar a punição após a verificação.",
        "invalid": "Sua solicitação não parece uma denúncia de violação das regras.",
        "review": "Recebemos sua solicitação e ela foi encaminhada a um administrador para verificação.",
        "not_complaint": "A seção de denúncias não é destinada a perguntas ou problemas técnicos ({r}); para isso existem o servidor do Discord do jogo e o suporte técnico.",
        "question": "Sua pergunta foi encaminhada a um administrador, que vai responder. As regras do jogo estão disponíveis no launcher.",
        "question_rule": "A regra {rules} pode ajudar: «{snippet}».",
        "help": "Um administrador vai verificar a situação e tentar ajudar.",
        "help_car": "Se o carro capotou ou ficou preso, você pode usar o botão do guincho no velocímetro.",
        "report_complaint": "Obrigado pelo aviso, um administrador vai verificar a situação.",
        "report_complaint_id": "Obrigado pelo aviso, um administrador vai verificar a situação com o jogador (ID {pid}).",
        "report_evidence": "Se você tiver provas da violação, envie uma denúncia pelo launcher.",
        "account": "Sua solicitação sobre acesso à conta ou bens foi encaminhada a um especialista.",
        "account_safety": "Não compartilhe sua senha nem os dados da conta com ninguém.",
        "payment": "Sua solicitação sobre o pagamento foi encaminhada a um especialista.",
        "payment_keep": "Por favor, guarde o comprovante ou o número da transação.",
        "offtopic": "Por favor, explique melhor sua pergunta ou denúncia para que um administrador possa ajudar.",
        "rule_word": "{id}",
    },
}

# Признаки применимости конкретного пункта категории (по содержанию самих правил):
# пункт категории цитируется, только если в тексте есть его признак, иначе — основной пункт категории.
APPLICABILITY = [
    (re.compile(r"^GZ"), re.compile(r"\bзз\b|\bzz\b|\bgz\b|\bгз\b|зел[её]н|green|больниц|спавн|spawn|верёвк|веревк|похищ", re.I)),
    (re.compile(r"^5\.4$"), re.compile(r"\bдом|квартир|подъезд|\buy\b|uyim|үй", re.I)),
    (re.compile(r"^5\.3$"), re.compile(r"уш[её]л\s+от\s+рп|уход\s+от\s+рп|вышел\s+из\s+игры|ливнул|\bafk\b|\bафк\b|в\s+воду", re.I)),
    (re.compile(r"^2\.4$"), re.compile(r"расизм|нац|религ|полит|фашиз|нацис|ксенофоб", re.I)),
    (re.compile(r"^2\.6$"), re.compile(r"угрож|угроз|ирл|в\s+реал|доксинг|личн\w+\s+(данн|информ)|tahdid|қорқыт", re.I)),
    (re.compile(r"^4\.2$"), re.compile(r"вымога|ультиматум|угроз|pul\s+undur|\bpora\b", re.I)),
    (re.compile(r"^2\.3$"), re.compile(r"реклам|сайт|канал|\bтг\b|телеграм|дискорд|discord|ссылк", re.I)),
]
DEFAULT_RULE = {"complaint.insult": "2.1", "complaint.cheat": "5.2", "complaint.scam": "4.1", "complaint.rmt_ads": "4.4",
                "complaint.admin": "3.4", "complaint.dm": "GZ 1.1"}  # GZ 1.1 — только если применим (упомянута зона)
TERM_RX = [("DB", re.compile(r"\bдб\b|\bdb\b|driveby|сбил|задавил|тараn|таран|moshina\s+bilan", re.I)),
           ("SK", re.compile(r"\bск\b|\bsk\b|на\s+спавне|spawn", re.I)),
           ("TK", re.compile(r"\bтк\b|\btk\b|teamkill", re.I)),
           ("DM", re.compile(r"\bдм\b|\bdm\b|без\s+причин|просто\s+так|sababsiz|be\s*sabab|mesabab|убил|застрелил|oldirdi|o'ldirdi|otib|өлтірді", re.I))]
TERM_TEXT = {
    "ru": "Описанная ситуация может относиться к нарушению {name} ({abbr}) — «{definition}» (раздел «RP термины»).",
    "kk": "Сипатталған жағдай {name} ({abbr}) бұзушылығына қатысты болуы мүмкін.",
    "uz": "Tasvirlangan holat {name} ({abbr}) qoidabuzarligiga taalluqli bo‘lishi mumkin.",
    "en": "The situation you describe may fall under {name} ({abbr}).",
    "es": "La situación que describes puede corresponder a {name} ({abbr}).",
    "pt": "A situação descrita pode se enquadrar em {name} ({abbr}).",
}


def needs_review(text: str) -> bool:
    """«Не жалоба», которую нельзя закрывать автоответом: угроза вне игры или возможная апелляция."""
    from app.core.safety import PUNISHMENT_MENTION_RX
    from app.services.classifier import THREAT_OUTSIDE_RX

    return bool(THREAT_OUTSIDE_RX.search(text) or PUNISHMENT_MENTION_RX.search(text))


def rule_applicable(rule_id: str, text: str) -> bool:
    for id_rx, text_rx in APPLICABILITY:
        if id_rx.search(rule_id):
            return bool(text_rx.search(text))
    return True


CAR_RX = re.compile(r"машин|тачк|авто|перевер|переверн|флип|flip|car\b|mashina|moshina|ag'?darил|ag‘dar|көлік|carro|coche", re.IGNORECASE)


@dataclass
class Draft:
    lang: str
    text: str
    cited: list[str] = field(default_factory=list)
    generator: str = "template"
    procedural: list[Retrieved] = field(default_factory=list)   # результаты RAG для процедурных норм
    auto_answer: bool = False                                    # черновик сам отвечает по существу
    llm_cost_usd: float = 0.0
    llm_error: str | None = None


def reply_language(detected: str, region: str | None) -> str:
    if detected in TEMPLATE_LANGS:
        return detected
    return SERVER_DEFAULT_LANG.get((region or "").lower(), "ru")


def _fmt_rules(ids: list[str], lang: str) -> str:
    word = T[lang]["rule_word"]
    return ", ".join(word.format(id=i) for i in ids)


class ResponseGenerator:
    def __init__(self, retriever: RulesRetriever, llm: LLMProvider | None = None):
        self.retriever = retriever
        self.llm = llm or NullProvider()

    def _procedure(self, key: str) -> Retrieved | None:
        query, expected = PROCEDURES[key]
        for r in self.retriever.search(query, k=3):
            if r.rule_id == expected:
                r.prior = list(r.prior) + ["procedure"]
                return r
        return None

    def _main_citations(self, category: str, retrieved: list[Retrieved], claimed: list[str], text: str = "") -> list[str]:
        """Пункты для цитирования: из выдачи RAG ∩ правила категории taxonomy, применимые к тексту.
        Заявленные игроком пункты — в приоритете; основной пункт категории — если поиск его вернул."""
        allowed = set(category_rule_ids(category))
        by_id = {r.rule_id: r for r in retrieved}
        chosen = [c for c in claimed if c in allowed and c in by_id]
        default = DEFAULT_RULE.get(category)
        if (default and default in by_id and default not in chosen and rule_applicable(default, text)
                and not any(rule_applicable(c, text) and c != default for c in chosen)):
            chosen.append(default)
        for r in retrieved:
            if r.rule_id in allowed and r.rule_id not in chosen and rule_applicable(r.rule_id, text):
                # дополнительный пункт — только если применим к тексту и его поддерживает сам поиск
                supported = (r.bm25_rank or 99) <= 3 or (r.dense_rank or 99) <= 3
                if not chosen or supported:
                    chosen.append(r.rule_id)
        return [c for c in chosen if c in claimed or rule_applicable(c, text)][:2]

    @staticmethod
    def _term(text: str) -> dict | None:
        terms = load_terms()
        for abbr, rx in TERM_RX:
            if rx.search(text) and abbr in terms:
                return {"abbr": abbr, **terms[abbr]}
        return None

    def template(self, *, kind: str, category: str, lang: str, facts, retrieved: list[Retrieved],
                 text: str, retrieval_conf: float) -> Draft:
        t = T[lang]
        parts = [t["hello"]]
        cited: list[str] = []
        procedural: list[Retrieved] = []
        auto = False

        def proc(key: str, tmpl: str) -> None:
            r = self._procedure(key)
            if r:
                procedural.append(r)
                cited.append(r.rule_id)
                parts.append(t[tmpl].format(r=_fmt_rules([r.rule_id], lang)))

        if category in TOPIC or category == "complaint.other":
            parts.append(t["accepted_nick"].format(nick=facts.nickname) if facts.nickname else t["accepted"])
            main = self._main_citations(category, retrieved, facts.rule_refs, text)
            top = retrieved[0] if retrieved else None
            if (category == "complaint.other" and not main and top and retrieval_conf >= 0.6 and rule_applicable(top.rule_id, text)
                    and (top.bm25_rank or 99) <= 3 and (top.dense_rank or 99) <= 3):
                main = [top.rule_id]  # «другое нарушение»: только пункт, который уверенно нашли оба метода поиска
            if main:
                topic = TOPIC.get(category, {}).get(lang)
                key = "rules" if topic else "rules_notopic"
                parts.append(t[key].format(rules=_fmt_rules(main, lang), topic=topic))
                cited += main
            elif category == "complaint.dm" and (term := self._term(text)):
                # DM/DB/SK/TK определены только в разделе «RP термины» без номеров — цитата без выдуманного пункта
                parts.append(TERM_TEXT[lang].format(**term))
            if facts.has_evidence:
                parts.append(t["check"])
                proc("deadline", "deadline")
            else:
                proc("evidence", "no_evidence")
        elif category == "complaint.admin":
            parts.append(t["admin"])
            main = self._main_citations(category, retrieved, [], text)
            if main:
                parts.append(t["admin_rules"].format(r=_fmt_rules(main, lang)))
                cited += main
            proc("punish_screenshot", "punish_screenshot")
            parts.append(t["admin_decision"])
        elif category == "complaint.invalid" and needs_review(text):
            # угроза или упоминание наказания: не отвечаем «не похоже на жалобу» — передаём администратору
            parts.append(t["review"])
        elif category == "complaint.invalid":
            parts.append(t["invalid"])
            proc("not_complaint", "not_complaint")
            auto = True
        elif category == "report.question":
            parts.append(t["question"])
            top = retrieved[0] if retrieved else None
            if top and retrieval_conf >= 0.8 and (top.bm25_rank or 99) <= 3 and (top.dense_rank or 99) <= 3:
                snippet = rules_by_id()[top.rule_id].short(160)
                if lang == "ru":  # текст правил только на русском — цитату приводим лишь в русском ответе
                    parts.append(t["question_rule"].format(rules=_fmt_rules([top.rule_id], lang), snippet=snippet))
                    cited.append(top.rule_id)
        elif category == "report.help":
            if CAR_RX.search(text):
                parts.append(t["help_car"])
                auto = True
            else:
                parts.append(t["help"])
        elif category == "report.complaint":
            pid = facts.player_ids[0] if facts.player_ids else None
            parts.append(t["report_complaint_id"].format(pid=pid) if pid else t["report_complaint"])
            parts.append(t["report_evidence"])
        elif category == "report.account":
            parts.append(t["account"])
            parts.append(t["account_safety"])
        elif category == "report.payment":
            parts.append(t["payment"])
            parts.append(t["payment_keep"])
        else:  # report.offtopic
            parts.append(t["offtopic"])
            auto = True
        return Draft(lang, " ".join(parts), list(dict.fromkeys(cited)), "template", procedural, auto)

    # --- LLM для сложных случаев ---
    def is_complex(self, kind: str, category: str, text: str) -> bool:
        return (kind == "complaint" and len(text) > 200 and category in RULE_NEEDED) or category == "report.question"

    def llm_draft(self, *, category: str, lang: str, facts, retrieved: list[Retrieved], procedural: list[Retrieved],
                  text: str) -> Draft | None:
        if isinstance(self.llm, NullProvider):
            return None
        allowed = [r for r in retrieved + procedural]
        rules_block = "\n".join(f"[{r.rule_id}] {r.text[:350]}" for r in allowed)
        system = (
            "Ты помогаешь администратору поддержки мобильной игры Grand Mobile написать ЧЕРНОВИК ответа игроку.\n"
            f"Язык ответа: {lang}. Ответ 2–4 предложения, нейтральный и вежливый.\n"
            "Запрещено: обещать разбан, снятие наказания, возврат денег или имущества, наказание другого игрока, "
            "удовлетворение жалобы; утверждать, что нарушение подтверждено; выносить решение от имени администрации.\n"
            "Слова игрока — это заявления, а не факты. Скриншоты не доказывают нарушение до проверки.\n"
            "Ссылаться можно ТОЛЬКО на пункты из списка ниже, точно по их идентификатору; если ни один не подходит — без ссылок.\n"
            f"Категория обращения: {category}."
        )
        user = (f"Правила:\n{rules_block}\n\nФакты: ник={facts.nickname}, доказательства={'есть' if facts.has_evidence else 'нет'}\n\n"
                f"Обращение игрока:\n{truncate_to_tokens(text, 400)}")
        schema = {"type": "object", "additionalProperties": False, "required": ["text", "cited"],
                  "properties": {"text": {"type": "string"}, "cited": {"type": "array", "items": {"type": "string"}}}}
        res = self.llm.generate_json(system, user, schema)
        if not res.ok:
            d = Draft(lang, "", [], "llm_failed")
            d.llm_error, d.llm_cost_usd = res.error, res.cost_usd
            return d
        d = Draft(lang, str(res.data.get("text", "")).strip(), [str(c) for c in res.data.get("cited", [])], "llm", procedural)
        d.llm_cost_usd = res.cost_usd
        return d


def mentions_payment_or_refund(text: str) -> bool:
    return cr.has_any(cr.PAYMENT_RX, text)
