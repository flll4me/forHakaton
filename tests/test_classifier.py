import pytest

from app.knowledge_base.rules_index import category_codes
from app.services import category_rules as cr
from app.services.classifier import priority_for
from app.services.fact_extractor import extract_facts
from tests.conftest import needs_data


@needs_data
@pytest.mark.parametrize("kind,text,refs,expected", [
    ("complaint", "оскорбил меня и мою семью в чате", "2.1", "complaint.insult"),
    ("complaint", "летает по воздуху, аимбот через стены", "", "complaint.cheat"),
    ("complaint", "убил меня без причины у банка", "dm", "complaint.dm"),
    ("complaint", "взял деньги за машину и не отдал, обман", "", "complaint.scam"),
    ("complaint", "не согласен с наказанием, админ выдал мне бан без причины", "", "complaint.admin"),
    ("report", "оплатил гк, не пришло", None, "report.payment"),
    ("report", "взломали аккаунт", None, "report.account"),
    ("report", "застрял в текстурах помогите выбраться", None, "report.help"),
    ("report", "помогите", None, "report.offtopic"),
    ("report", "дм 248", None, "report.complaint"),
    ("report", "как устроиться в полицию?", None, "report.question"),
])
def test_category(pipeline, kind, text, refs, expected):
    res = pipeline.classifier.predict(kind, text, refs)
    assert res.category == expected
    assert 0.0 <= res.confidence <= 1.0


@needs_data
@pytest.mark.parametrize("text", [
    "не согласна с решением адм, рядом у человека играла музыка а Мут выдали мне",
    "Выдал Мут за то что я сказал в спек заходишь",
    "пересмотр жб, наказание не заслуженное",
    "репорт написан игрок не наказан, а это фарм репов и игнор нарушения",
    "men Bek_Nazarov_ nohaq ban berdi iltimos ko'rib beringlar adminlar",
])
def test_appeals_and_admin_complaints(pipeline, text):
    assert pipeline.classifier.predict("complaint", text, "").category == "complaint.admin"


def test_appeal_rule_does_not_fire_on_request_to_punish_other_player():
    assert cr.complaint_rule_category("я хочу чтобы вы его дали бан 31 день, сварил меня на машине", "")[0] != "complaint.admin"


@needs_data
def test_categories_strictly_from_taxonomy(pipeline):
    codes = set(category_codes())
    for kind, text in [("complaint", "что-то непонятное абв"), ("report", "абракадабра"), ("complaint", ""), ("report", "")]:
        res = pipeline.classifier.predict(kind, text, None)
        assert res.category in codes and res.category.startswith(kind)


def test_priority_rules_from_taxonomy():
    assert priority_for("report.payment", False, "оплатил").priority == "p0"
    assert priority_for("report.account", False, "взломали").priority == "p0"
    assert priority_for("complaint.cheat", True, "софт").priority == "p1"
    assert priority_for("complaint.insult", True, "оск").priority == "p2"
    assert priority_for("complaint.insult", False, "оск").priority == "p3"  # жалоба без доказательств
    assert priority_for("report.offtopic", False, "ау").priority == "p3"
    assert priority_for("complaint.insult", True, "найду тебя в реале, знаю где ты живешь").priority == "p0"


def test_spam_detection():
    assert cr.is_content_free("+") and cr.is_content_free("...") and cr.is_content_free("<digits>")
    assert not cr.is_content_free("помогите застрял")


@needs_data
def test_fact_extraction_contract_example():
    f = extract_facts("complaint", "Arman_Petrov оскорблял меня и мою семью в чате, скрины есть, 2.1 ОП",
                      [{"role": "player", "attachments": 2}])
    assert f.contract() == {"nickname": "Arman_Petrov", "rule_refs": ["2.1"], "has_evidence": True, "server": None}


@needs_data
def test_fact_extraction_reported_nick_priority_and_invalid_refs():
    f = extract_facts("complaint", "Ivan_Petrov12 оскорбил, пункт 9.9 и 1.2.7 ППЖ", [{"role": "player"}],
                      reported_nick="Dias_Volkov", rule_refs_raw="2.1оп", images_count=0)
    assert f.nickname == "Dias_Volkov" and f.nickname_source == "reported_nick"
    assert f.rule_refs == ["2.1", "ЖБ 1.2.7"]
    assert f.invalid_rule_refs == ["9.9"]
    assert f.has_evidence is False  # слова «скрины» без вложений доказательством не считаются


def test_fact_extraction_server_and_ids():
    f = extract_facts("report", "дм 248 на 5 сервере", [{"role": "player"}])
    assert f.server == "5"
    assert "248" in f.player_ids
