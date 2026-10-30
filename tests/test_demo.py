"""Демо-сценарий (demo/demo_tickets.jsonl) должен проходить целиком — иначе показ сломается."""
from scripts.demo_check import load_demo, run
from tests.conftest import needs_data


@needs_data
def test_demo_scenarios_pass(pipeline):
    results = run(pipeline, load_demo())
    assert len(results) >= 15
    failed = {t["ticket_id"]: errors for t, _, errors in results if errors}
    assert not failed, failed
    pipeline.deduplicator.reset()  # не оставлять демо-серию в состоянии дедупликатора для других тестов


@needs_data
def test_threat_and_short_appeal_not_told_not_a_complaint(triage):
    for text in ["он пишет что найдёт меня в реале", "за что?"]:
        resp, _ = triage(text)
        assert "не похоже на жалобу" not in resp.draft_reply.text, text
        assert resp.needs_human
