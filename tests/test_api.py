"""Интеграционные тесты HTTP API и пограничные случаи из ТЗ (раздел 13.3)."""
from app.knowledge_base.rules_index import category_codes, rules_by_id
from app.models.schemas import TriageResponse
from tests.conftest import needs_data

CONTRACT_REQUEST = {"kind": "complaint", "messages": [
    {"role": "player", "text": "Arman_Petrov оскорблял меня и мою семью в чате, скрины есть, 2.1 ОП", "attachments": 2}]}


def _post(client, body):
    r = client.post("/triage", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    TriageResponse.model_validate(data)  # строго по схеме контракта
    assert set(data) == {"lang", "category", "priority", "is_spam_or_duplicate", "extracted", "draft_reply",
                         "needs_human", "confidence", "reasons"}
    assert data["category"] in category_codes()
    for rid in data["draft_reply"]["cited"]:
        assert rid in rules_by_id(), f"выдуманный пункт {rid}"
    return data


@needs_data
def test_health_and_version(client):
    h = client.get("/health")
    assert h.status_code == 200 and h.json()["status"] == "ok"
    assert h.json()["components"]["rag_index_points"] > 200
    assert client.get("/version").json()["app_version"]


@needs_data
def test_contract_example(client):
    d = _post(client, CONTRACT_REQUEST)
    assert d["lang"] == "ru" and d["category"] == "complaint.insult" and d["priority"] == "p2"
    assert d["extracted"] == {"nickname": "Arman_Petrov", "rule_refs": ["2.1"], "has_evidence": True, "server": None}
    assert "2.1" in d["draft_reply"]["cited"] and d["draft_reply"]["lang"] == "ru"
    assert d["needs_human"] is True and 0 <= d["confidence"] <= 1


@needs_data
def test_invalid_requests_do_not_crash(client):
    assert client.post("/triage", content="{not json", headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/triage", json={"messages": []}).status_code == 422
    assert client.post("/triage", json={"kind": "spam", "messages": [{"text": "x"}]}).status_code == 422
    assert client.post("/triage", json={"kind": "report"}).status_code == 422
    assert client.get("/health").status_code == 200  # сервис жив


@needs_data
def test_debug_mode(client):
    r = client.post("/triage?debug=true", json=CONTRACT_REQUEST).json()
    assert r["_debug"]["retrieved"] and r["_debug"]["citation_check"]["valid"]


# --- Edge cases (ТЗ 13.3) ---

@needs_data
def test_edge_01_russian(client):
    d = _post(client, {"kind": "complaint", "messages": [{"text": "убил меня без причины у банка", "attachments": 1}]})
    assert d["lang"] == "ru" and d["category"] == "complaint.dm" and d["draft_reply"]["lang"] == "ru"


@needs_data
def test_edge_02_kazakh_cp1251(client):
    d = _post(client, {"kind": "complaint", "region": "kz",
                       "messages": [{"text": "Мына ойыншы мені балаѓаттады, анамды сљкті", "attachments": 1}]})
    assert d["lang"] == "kk" and d["draft_reply"]["lang"] == "kk"


@needs_data
def test_edge_03_uzbek(client):
    d = _post(client, {"kind": "complaint", "messages": [{"text": "Meni haqorat qildi onadan so'kdi", "attachments": 1}]})
    assert d["lang"] == "uz" and d["draft_reply"]["lang"] == "uz" and d["category"] == "complaint.insult"


@needs_data
def test_edge_04_mixed_language(client):
    d = _post(client, {"kind": "complaint", "messages": [{"text": "bu odam мени оскорбил onadan sokdi ban bering", "attachments": 1}]})
    assert d["lang"] in {"uz", "ru"} and d["category"] == "complaint.insult"


@needs_data
def test_edge_05_typos(client):
    d = _post(client, {"kind": "report", "messages": [{"text": "памагите застрял в тикстурах"}]})
    assert d["category"] == "report.help"


@needs_data
def test_edge_06_very_short(client):
    d = _post(client, {"kind": "complaint", "messages": [{"text": "оск", "attachments": 1}]})
    assert d["category"] == "complaint.insult"


@needs_data
def test_edge_07_empty(client):
    d = _post(client, {"kind": "report", "messages": [{"text": ""}]})
    assert d["category"] == "report.offtopic" and d["is_spam_or_duplicate"] is True and d["draft_reply"]["text"]


@needs_data
def test_edge_08_payment(client):
    d = _post(client, {"kind": "report", "messages": [{"text": "оплатил гк, не пришло"}]})
    assert d["category"] == "report.payment" and d["priority"] == "p0" and d["needs_human"] is True


@needs_data
def test_edge_09_ban_appeal(client):
    d = _post(client, {"kind": "complaint", "messages": [{"text": "не согласен с баном, снимите бан, я ничего не нарушал", "attachments": 1}]})
    assert d["category"] == "complaint.admin" and d["needs_human"] is True
    assert "разбан" not in d["draft_reply"]["text"].lower()


@needs_data
def test_edge_10_hack(client):
    d = _post(client, {"kind": "report", "messages": [{"text": "взломали аккаунт, не могу зайти в аккаунт"}]})
    assert d["category"] == "report.account" and d["priority"] == "p0" and d["needs_human"] is True


@needs_data
def test_edge_11_no_evidence(client):
    d = _post(client, {"kind": "complaint", "messages": [{"text": "он меня оскорбил в чате", "attachments": 0}]})
    assert d["extracted"]["has_evidence"] is False and d["priority"] == "p3"
    assert any("доказательств" in r for r in d["reasons"])


@needs_data
def test_edge_12_wrong_rule_number(client):
    d = _post(client, {"kind": "complaint", "rule_refs_raw": "9.9", "messages": [{"text": "оскорбил меня", "attachments": 1}]})
    assert "9.9" not in d["extracted"]["rule_refs"] and "9.9" not in d["draft_reply"]["cited"]
    assert any("9.9" in r for r in d["reasons"])


@needs_data
def test_edge_13_no_matching_rule(client):
    d = _post(client, {"kind": "report", "messages": [{"text": "привет как дела"}]})
    assert d["draft_reply"]["cited"] == []


@needs_data
def test_edge_14_repeated_ticket(client):
    body = {"kind": "report", "reporter_hash": "edge14", "messages": [{"text": "помогите застрял в текстурах"}]}
    results = [_post(client, {**body, "created_at": f"2026-03-14T19:{m:02d}:00"}) for m in (1, 10, 20, 30, 40)]
    assert results[0]["is_spam_or_duplicate"] is False
    assert all(r["is_spam_or_duplicate"] for r in results[1:])


@needs_data
def test_edge_15_llm_error_handled(pipeline, client):
    from tests.test_safety import _FailingLLM
    old = pipeline.generator.llm
    try:
        pipeline.generator.llm = _FailingLLM()
        d = _post(client, {"kind": "report", "messages": [{"text": "как получить лицензию на оружие?"}]})
        assert d["draft_reply"]["text"] and d["needs_human"] is True
    finally:
        pipeline.generator.llm = old
