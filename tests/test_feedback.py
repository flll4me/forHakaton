import pytest

from app.services.feedback import FeedbackStore


def test_feedback_roundtrip(tmp_path):
    s = FeedbackStore(tmp_path / "fb.sqlite")
    s.add(ticket_id="t1", action="edit", predicted_category="complaint.dm", final_category="complaint.insult", priority="p2",
          lang="ru", draft_text="черновик", final_text="исправлено", cited=["2.1"], rating=3)
    s.add(ticket_id="t2", action="send", predicted_category="report.offtopic", final_category="report.offtopic", priority="p3",
          lang="ru", draft_text="a", final_text="a", cited=[], rating=5)
    assert s.status_by_ticket() == {"t1": "edit", "t2": "send"}
    st = s.stats()
    assert st["n"] == 2 and st["recategorized"] == 1 and st["avg_rating"] == 4
    with pytest.raises(Exception):
        s.add(ticket_id="t3", action="delete", predicted_category="", final_category="", priority="p3", lang="ru",
              draft_text="", final_text="", cited=[])
