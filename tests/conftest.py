import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # модель эмбеддингов берётся из локального кэша
os.environ.setdefault("LLM_PROVIDER", "none")  # тесты не ходят во внешние API

RULES = ROOT / "data" / "raw" / "handout" / "rules.json"
needs_data = pytest.mark.skipif(not RULES.exists(), reason="данные кейса не распакованы в data/raw/handout")


@pytest.fixture(scope="session")
def pipeline():
    if not RULES.exists():
        pytest.skip("данные кейса не распакованы")
    from app.services.pipeline import TriagePipeline

    return TriagePipeline()


@pytest.fixture(scope="session")
def client(pipeline):
    from fastapi.testclient import TestClient

    from app.main import app

    app.state.pipeline = pipeline
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def triage(pipeline):
    from app.models.schemas import TriageRequest

    def run(text: str = "", kind: str = "complaint", **extra):
        msg = {"role": "player", "text": text}
        if "attachments" in extra:
            msg["attachments"] = extra.pop("attachments")
        return pipeline.triage(TriageRequest(kind=kind, messages=[msg], **extra))

    return run
