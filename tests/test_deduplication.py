from app.services.deduplicator import Deduplicator


def _d():
    return Deduplicator(use_embeddings=False)


def test_five_tickets_within_hour_same_problem():
    d = _d()
    statuses = []
    for i in range(5):
        r = d.check_and_add(f"t{i}", "report", "помогите застрял в текстурах", "rep1", f"2026-03-14T19:{10 + i * 5:02d}:00")
        statuses.append(r.status)
    assert statuses[0] == "new_issue"
    assert statuses[1:] == ["duplicate"] * 4
    assert r.series_count == 5


def test_outside_window_is_new_issue():
    d = _d()
    d.check_and_add("a", "report", "помогите застрял в текстурах", "rep1", "2026-03-14T10:00:00")
    assert d.check_and_add("b", "report", "помогите застрял в текстурах", "rep1", "2026-03-14T12:00:00").status == "new_issue"


def test_different_reporter_not_duplicate():
    d = _d()
    d.check_and_add("a", "report", "помогите застрял в текстурах", "rep1", "2026-03-14T10:00:00")
    assert d.check_and_add("b", "report", "помогите застрял в текстурах", "rep2", "2026-03-14T10:05:00").status == "new_issue"


def test_common_words_are_not_enough():
    d = _d()
    d.check_and_add("a", "report", "как получить лицензию на оружие в правительстве", "rep1", "2026-03-14T10:00:00")
    r = d.check_and_add("b", "report", "переверните машину пожалуйста", "rep1", "2026-03-14T10:05:00")
    assert r.status != "duplicate"


def test_complaints_same_target_vs_other_target():
    d = _d()
    d.check_and_add("a", "complaint", "оск родни", "rep1", "2026-03-14T10:00:00", reported_hash="X")
    assert d.check_and_add("b", "complaint", "оск родни", "rep1", "2026-03-14T10:10:00", reported_hash="X").status == "duplicate"
    assert d.check_and_add("c", "complaint", "оск родни", "rep1", "2026-03-14T10:12:00", reported_hash="Y").status != "duplicate"


def test_insufficient_data():
    d = _d()
    r = d.check_and_add("a", "report", "помогите", None, None)
    assert r.status == "new_issue" and "не проверялся" in r.note
