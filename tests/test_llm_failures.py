"""Отказы LLM: выключенный LLM, нет API-ключа, таймаут, ошибки провайдера.

Реальный API Anthropic здесь не вызывается. Провайдеры (официальный SDK `anthropic` и httpx для
OpenAI-совместимого API) направляются на локальный HTTP-сервер, который имитирует зависание,
HTTP-ошибки и битый ответ, — так проверяется настоящий код обработки исключений SDK.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.core import safety
from app.core.config import Settings
from app.services.llm_provider import AnthropicProvider, NullProvider, OpenAICompatibleProvider, get_provider
from tests.conftest import needs_data

SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}, "cited": {"type": "array", "items": {"type": "string"}}}}
LONG = "Игрок Ivan_Petrov12 оскорблял меня и мою семью в общем чате целый вечер, писал нецензурные слова про мою мать " * 2


class _Handler(BaseHTTPRequestHandler):
    mode = "ok"

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("content-length") or 0))
        mode = type(self).mode
        if mode == "hang":
            time.sleep(3)
        status, body = {
            "hang": (200, b"{}"),
            "500": (500, b'{"type":"error","error":{"type":"api_error","message":"internal"}}'),
            "529": (529, b'{"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}'),
            "429": (429, b'{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}'),
            "401": (401, b'{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}'),
            "garbage": (200, b"not json at all"),
        }[mode]
        try:
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # клиент уже ушёл по таймауту


@pytest.fixture(scope="module")
def fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def _settings(monkeypatch, **env) -> Settings:
    base = {"LLM_ENABLED": "true", "LLM_PROVIDER": "anthropic", "LLM_MODEL": "claude-haiku-5-5", "LLM_TIMEOUT_S": "0.5",
            "LLM_PRICE_IN_PER_MTOK": "0.10", "LLM_PRICE_OUT_PER_MTOK": "0.50"}
    for k, v in {**base, **env}.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    return Settings()


def _anthropic(monkeypatch, server, mode) -> AnthropicProvider:
    _Handler.mode = mode
    s = _settings(monkeypatch, ANTHROPIC_API_KEY="test-key-not-real",
                  ANTHROPIC_BASE_URL=f"http://127.0.0.1:{server.server_address[1]}")
    p = get_provider(s)
    assert isinstance(p, AnthropicProvider)
    return p


# --- конфигурация: LLM выключен / не настроен ---

def test_llm_enabled_false_forces_templates(monkeypatch):
    p = get_provider(_settings(monkeypatch, LLM_ENABLED="false", ANTHROPIC_API_KEY="test-key-not-real"))
    assert isinstance(p, NullProvider) and not p.enabled and p.disabled_reason == "LLM_ENABLED=false"


def test_provider_none_is_default(monkeypatch):
    p = get_provider(_settings(monkeypatch, LLM_PROVIDER="none"))
    assert isinstance(p, NullProvider) and p.disabled_reason is None


def test_missing_api_key_falls_back_to_templates(monkeypatch):
    p = get_provider(_settings(monkeypatch, ANTHROPIC_API_KEY=None, ANTHROPIC_AUTH_TOKEN=None))
    assert isinstance(p, NullProvider) and "ANTHROPIC_API_KEY" in p.disabled_reason


def test_missing_model_and_base_url(monkeypatch):
    assert "LLM_MODEL" in get_provider(_settings(monkeypatch, LLM_MODEL="", ANTHROPIC_API_KEY="k")).disabled_reason
    p = get_provider(_settings(monkeypatch, LLM_PROVIDER="openai_compatible", LLM_BASE_URL=""))
    assert isinstance(p, NullProvider) and "LLM_BASE_URL" in p.disabled_reason
    assert "неизвестный" in get_provider(_settings(monkeypatch, LLM_PROVIDER="gpt9")).disabled_reason


def test_key_removed_at_runtime_gives_clear_error(monkeypatch):
    # провайдер создан с ключом, потом ключ пропал (ротация секрета) — понятная ошибка, а не исключение
    s = _settings(monkeypatch, ANTHROPIC_API_KEY=None, ANTHROPIC_AUTH_TOKEN=None, ANTHROPIC_BASE_URL="http://127.0.0.1:9")
    res = AnthropicProvider(s).generate_json("sys", "user", SCHEMA)
    assert not res.ok and res.error == "LLM: не задан API-ключ"


# --- ошибки провайдера на уровне SDK ---

def test_anthropic_timeout(monkeypatch, fake_server):
    p = _anthropic(monkeypatch, fake_server, "hang")
    t0 = time.perf_counter()
    res = p.generate_json("sys", "user", SCHEMA)
    assert res.error == "таймаут LLM" and not res.ok
    assert time.perf_counter() - t0 < 2.0  # без ретраев: ограничено LLM_TIMEOUT_S


@pytest.mark.parametrize("mode,expected", [("500", "LLM HTTP 500"), ("529", "LLM HTTP 529"),
                                           ("429", "LLM rate limit (429)"), ("401", "LLM HTTP 401")])
def test_anthropic_http_errors(monkeypatch, fake_server, mode, expected):
    res = _anthropic(monkeypatch, fake_server, mode).generate_json("sys", "user", SCHEMA)
    assert res.error == expected and res.data is None


def test_anthropic_garbage_response(monkeypatch, fake_server):
    res = _anthropic(monkeypatch, fake_server, "garbage").generate_json("sys", "user", SCHEMA)
    assert not res.ok and res.error


def test_anthropic_network_down(monkeypatch):
    s = _settings(monkeypatch, ANTHROPIC_API_KEY="test-key-not-real", ANTHROPIC_BASE_URL="http://127.0.0.1:9")
    res = AnthropicProvider(s).generate_json("sys", "user", SCHEMA)
    # Windows повторяет SYN на закрытый порт дольше таймаута, поэтому возможен и «таймаут»
    assert res.error in {"LLM недоступен (сеть)", "таймаут LLM"}


def test_budget_precheck_blocks_expensive_call(monkeypatch):
    s = _settings(monkeypatch, ANTHROPIC_API_KEY="test-key-not-real", LLM_PRICE_IN_PER_MTOK="100", LLM_PRICE_OUT_PER_MTOK="100")
    res = AnthropicProvider(s).generate_json("sys", "x" * 3000, SCHEMA)
    assert res.error and "лимита" in res.error  # вызов не выполнялся


@pytest.mark.parametrize("mode", ["hang", "500", "garbage"])
def test_openai_compatible_failures(monkeypatch, fake_server, mode):
    _Handler.mode = mode
    s = _settings(monkeypatch, LLM_PROVIDER="openai_compatible", LLM_MODEL="local-model",
                  LLM_BASE_URL=f"http://127.0.0.1:{fake_server.server_address[1]}/v1")
    p = get_provider(s)
    assert isinstance(p, OpenAICompatibleProvider)
    res = p.generate_json("sys", "user", SCHEMA)
    assert not res.ok and res.error
    if mode == "hang":
        assert res.error == "таймаут LLM"


# --- весь пайплайн при отказе провайдера ---

@needs_data
@pytest.mark.parametrize("mode", ["hang", "500", "429", "garbage"])
def test_pipeline_survives_provider_failure(monkeypatch, fake_server, pipeline, triage, mode):
    old = pipeline.generator.llm
    try:
        pipeline.generator.llm = _anthropic(monkeypatch, fake_server, mode)
        t0 = time.perf_counter()
        resp, dbg = triage(LONG, attachments=1)
        elapsed = time.perf_counter() - t0
        assert dbg.generator == "template" and resp.draft_reply.text
        assert resp.needs_human and any(r.startswith("ошибка LLM") for r in resp.reasons)
        assert not safety.find_forbidden_promise(resp.draft_reply.text)
        assert elapsed < 5.0  # бюджет латентности p95 ≤ 5 с соблюдается и при зависшем провайдере
    finally:
        pipeline.generator.llm = old


@needs_data
def test_pipeline_simple_ticket_does_not_call_llm(monkeypatch, fake_server, pipeline, triage):
    """Короткие типовые обращения обрабатываются шаблоном, недоступность LLM на них не влияет."""
    old = pipeline.generator.llm
    try:
        pipeline.generator.llm = _anthropic(monkeypatch, fake_server, "500")
        resp, dbg = triage("оплатил гк, не пришло", kind="report")
        assert dbg.generator == "template" and not any("LLM" in r for r in resp.reasons)
        assert resp.category == "report.payment" and resp.needs_human
    finally:
        pipeline.generator.llm = old


@needs_data
def test_health_reports_llm_state(client):
    comp = client.get("/health").json()["components"]
    assert comp["llm_provider"] == "none" and "llm_disabled_reason" in comp
    json.dumps(comp)
