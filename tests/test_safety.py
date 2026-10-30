from app.core import safety
from app.services.llm_provider import LLMProvider, LLMResult
from tests.conftest import needs_data


def test_forbidden_promises_detected():
    for text in ["Игрок Arman_Petrov получит мут на 30 минут.", "Ваш бан снимем завтра", "Мы вернём вам деньги",
                 "Наказание будет снято.", "Нарушение подтверждено.", "You will be unbanned", "O‘yinchi jazolanadi",
                 "Будете разблокированы", "Жалоба будет одобрена"]:
        assert safety.find_forbidden_promise(text), text


def test_templates_have_no_forbidden_promises():
    from app.services.response_generator import T
    for lang, t in T.items():
        for key, text in t.items():
            assert not safety.find_forbidden_promise(text), (lang, key)


def test_mandatory_escalations():
    assert safety.mandatory_escalations("report", "report.question", "оплатил донат не пришло")
    assert safety.mandatory_escalations("report", "report.question", "верните деньги за донат")
    assert safety.mandatory_escalations("complaint", "complaint.other", "требую разбан")
    assert safety.mandatory_escalations("report", "report.help", "мой аккаунт взломали")
    assert safety.mandatory_escalations("report", "report.account", "пропала машина")
    assert not safety.mandatory_escalations("report", "report.help", "переверните машину пж")


def test_mandatory_escalations_free_word_order_and_threats():
    """Найдено сквозными тестами (аудит 2026-10-30): обратный порядок слов и угрозы от третьего лица."""
    assert safety.R_BAN_APPEAL in safety.mandatory_escalations("complaint", "complaint.invalid", "напишите что мой бан снимут сегодня")
    assert safety.R_BAN_APPEAL in safety.mandatory_escalations("report", "report.question", "мут уберите пожалуйста")
    assert safety.R_REFUND in safety.mandatory_escalations("report", "report.offtopic", "когда деньги вернут?")
    assert safety.R_THREAT in safety.mandatory_escalations("complaint", "complaint.invalid", "пишет что найдёт меня в реале")
    assert safety.R_THREAT in safety.mandatory_escalations("complaint", "complaint.insult", "знает где я живу")
    for text in ["уберите машину у банка", "снимите меня с крыши", "верните на спавн", "он вернул машину"]:
        assert not safety.mandatory_escalations("report", "report.help", text), text


@needs_data
def test_short_appeal_classified_invalid_still_escalated(triage):
    """Короткая апелляция, отнесённая к «не жалобе», не должна получать автоответ."""
    for text in ["за что?", "мне же дали бан на 3 дня почему перевыдали на 7??", "как я оказался в дмг? и не знаю кто выдал"]:
        resp, _ = triage(text)
        if resp.category == "complaint.invalid":
            assert resp.needs_human and safety.R_POSSIBLE_APPEAL in resp.reasons, text
        else:
            assert resp.needs_human, text
    resp, _ = triage("как купить дом?")
    assert safety.R_POSSIBLE_APPEAL not in resp.reasons


@needs_data
def test_payment_ban_appeal_hack_always_human(triage):
    for kind, text in [("report", "оплатил донат, деньги списали, гк не пришли"),
                       ("complaint", "не согласен с наказанием, прошу разбан"),
                       ("report", "мой аккаунт взломали, сменили пароль"),
                       ("report", "to'lov qildim donat kelmadi")]:
        resp, _ = triage(text, kind=kind)
        assert resp.needs_human, text
        assert resp.reasons


@needs_data
def test_complaint_draft_never_promises_punishment(triage):
    resp, _ = triage("Arman_Petrov оскорблял меня и мою семью в чате, скрины есть, 2.1 ОП", attachments=2)
    assert not safety.find_forbidden_promise(resp.draft_reply.text)
    assert resp.needs_human
    assert safety.R_PUNISH_DECISION in resp.reasons


class _FailingLLM(LLMProvider):
    name = "failing"

    @property
    def enabled(self):
        return True

    def generate_json(self, system, user, schema):
        return LLMResult(error="таймаут LLM")


class _BadLLM(_FailingLLM):
    def __init__(self, payload):
        super().__init__()
        self.payload = payload

    def generate_json(self, system, user, schema):
        return LLMResult(data=self.payload, input_tokens=100, output_tokens=50)


LONG = "Игрок Ivan_Petrov12 оскорблял меня и мою семью в общем чате целый вечер, писал нецензурные слова про мою мать " * 2


@needs_data
def test_llm_error_falls_back_to_template(pipeline, triage):
    old = pipeline.generator.llm
    try:
        pipeline.generator.llm = _FailingLLM()
        resp, dbg = triage(LONG, attachments=1)
        assert resp.draft_reply.text and dbg.generator == "template"
        assert resp.needs_human and any("LLM" in r for r in resp.reasons)
    finally:
        pipeline.generator.llm = old


@needs_data
def test_llm_hallucinated_rule_or_promise_rejected(pipeline, triage):
    old = pipeline.generator.llm
    try:
        pipeline.generator.llm = _BadLLM({"text": "Здравствуйте! Нарушение п. 9.9 подтверждено.", "cited": ["9.9"]})
        resp, dbg = triage(LONG, attachments=1)
        assert "9.9" not in resp.draft_reply.cited and dbg.generator == "template"
        pipeline.generator.llm = _BadLLM({"text": "Здравствуйте! Игрок Ivan_Petrov12 получит бан на 7 дней.", "cited": ["2.1"]})
        resp, dbg = triage(LONG, attachments=1)
        assert dbg.generator == "template" and not safety.find_forbidden_promise(resp.draft_reply.text)
        pipeline.generator.llm = _BadLLM({"text": "Здравствуйте! Ваша жалоба принята, администратор проверит доказательства по п. 2.1.", "cited": ["2.1"]})
        resp, dbg = triage(LONG, attachments=1)
        assert dbg.generator == "llm" and resp.draft_reply.cited == ["2.1"]
    finally:
        pipeline.generator.llm = old
