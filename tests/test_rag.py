from app.knowledge_base.rules_index import load_rules, parse_rule_refs, rules_by_id
from app.services.rag import RulesRetriever, validate_citations
from tests.conftest import needs_data


@needs_data
def test_index_built_from_rules_json():
    rules = load_rules()
    assert len(rules) > 200
    ids = [r.rule_id for r in rules]
    assert len(ids) == len(set(ids)), "rule_id должны быть уникальны"
    assert "2.1" in ids and "ЖБ 1.2.7" in ids and "GZ 1.1" in ids


def test_parse_rule_refs_variants():
    assert parse_rule_refs("2.1 ОП") == ["2.1"]
    assert parse_rule_refs("2.1оп") == ["2.1"]
    assert parse_rule_refs("1.2 gz") == ["GZ 1.2"]
    assert parse_rule_refs("1.2.7 ППЖ") == ["ЖБ 1.2.7"]
    assert parse_rule_refs("3,5,7") == []
    assert parse_rule_refs("дм") == []


@needs_data
def test_hybrid_search_finds_relevant_rules(pipeline):
    r = pipeline.retriever
    assert r.search("оскорбил меня и мою семью", "complaint.insult")[0].rule_id == "2.1"
    assert "5.2" in [x.rule_id for x in r.search("софт аим через стены", "complaint.cheat")]
    assert any(x.rule_id.startswith("GZ") for x in r.search("дм в зелёной зоне", "complaint.dm"))
    assert "2.4" in [x.rule_id for x in r.search("расизм в чате")]


@needs_data
def test_multilingual_dense_search(pipeline):
    if not pipeline.retriever.dense:
        import pytest
        pytest.skip("модель эмбеддингов недоступна")
    res = pipeline.retriever.search("Meni haqorat qildi onadan sokdi", "complaint.insult")
    assert res[0].rule_id == "2.1"


@needs_data
def test_bm25_only_mode():
    r = RulesRetriever(use_embeddings=False)
    assert not r.dense
    assert r.search("реклама сторонних сайтов в чате")[0].rule_id == "2.3"


@needs_data
def test_draft_citations_are_relevant(triage):
    r, _ = triage("убил просто так дм", attachments=1)
    assert not any(c.startswith("GZ") or c == "5.4" for c in r.draft_reply.cited)  # зона/дом не упомянуты
    assert "DeathMatch" in r.draft_reply.text  # определение из раздела «RP термины», без выдуманного номера
    r, _ = triage("дм в зз у больницы", attachments=1)
    assert r.draft_reply.cited[0] == "GZ 1.1"
    r, _ = triage("оск родни", attachments=1)
    assert "2.4" not in r.draft_reply.cited and "2.1" in r.draft_reply.cited


@needs_data
def test_validate_citations(pipeline):
    res = pipeline.retriever.search("оскорбил меня", "complaint.insult")
    ok = validate_citations(["2.1"], res, "complaint.insult")
    assert ok.valid and ok.accepted == ["2.1"]
    bad = validate_citations(["9.9"], res, "complaint.insult")
    assert not bad.valid and "не существует" in bad.rejected["9.9"]
    not_retrieved = validate_citations(["5.2"], res, "complaint.insult")
    assert not not_retrieved.valid and "не найден поиском" in not_retrieved.rejected["5.2"]
    assert "5.2" in rules_by_id()  # пункт существует, но к обращению не относится
