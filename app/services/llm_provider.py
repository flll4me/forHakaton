"""Абстракция LLM-провайдера.

- NullProvider          — LLM выключен (LLM_PROVIDER=none): генератор работает на шаблонах.
- AnthropicProvider     — Claude Messages API через официальный SDK `anthropic`,
                          structured output (output_config.format = json_schema).
- OpenAICompatibleProvider — любой сервер с OpenAI-совместимым /chat/completions
                          (локальные модели: Ollama, vLLM, llama.cpp server) через httpx.

Общие требования ТЗ: модель и цены из .env, контроль размера входа (оценка токенов до
вызова), подсчёт токенов и стоимости по фактическому usage, жёсткий таймаут без ретраев
(бюджет латентности p95 ≤ 5 c), отказ от вызова, если оценка стоимости выше лимита на тикет.
Любая ошибка возвращается как LLMResult(error=...) — вызывающий код уходит в fallback.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class LLMResult:
    data: dict | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str = ""
    error: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.data is not None


def estimate_tokens(text: str) -> int:
    """Грубая оценка до вызова: ~3 символа на токен для кириллицы/латиницы смешанного текста."""
    return max(1, len(text) // 3)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    limit = max_tokens * 3
    return text if len(text) <= limit else text[:limit] + "…"


class LLMProvider:
    name = "none"

    def __init__(self, settings: Settings | None = None, disabled_reason: str | None = None):
        self.s = settings or get_settings()
        self.disabled_reason = disabled_reason  # почему LLM не используется (для /health и логов)

    @property
    def enabled(self) -> bool:
        return False

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.s.llm_price_in_per_mtok + output_tokens * self.s.llm_price_out_per_mtok) / 1_000_000

    def _precheck(self, system: str, user: str) -> str | None:
        est_in = estimate_tokens(system) + estimate_tokens(user)
        if est_in > self.s.llm_max_input_tokens:
            return f"вход {est_in} ток. > LLM_MAX_INPUT_TOKENS={self.s.llm_max_input_tokens}"
        est_cost = self.cost(est_in, self.s.llm_max_output_tokens)
        if est_cost > self.s.llm_max_cost_per_ticket:
            return f"оценка стоимости ${est_cost:.5f} > лимита ${self.s.llm_max_cost_per_ticket}"
        return None

    def generate_json(self, system: str, user: str, schema: dict) -> LLMResult:
        return LLMResult(error="LLM отключён (LLM_PROVIDER=none)")


class NullProvider(LLMProvider):
    pass


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings | None = None):
        super().__init__(settings)
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.s.llm_model) and has_anthropic_credentials()

    def _get_client(self):
        if self._client is None:
            import anthropic

            # ключ берётся из ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN; ретраи выключены — бюджет латентности
            self._client = anthropic.Anthropic(timeout=self.s.llm_timeout_s, max_retries=0)
        return self._client

    def generate_json(self, system: str, user: str, schema: dict) -> LLMResult:
        reason = self._precheck(system, user)
        if reason:
            return LLMResult(error=reason, model=self.s.llm_model)
        t0 = time.perf_counter()
        try:
            import anthropic

            resp = self._get_client().messages.create(
                model=self.s.llm_model,
                max_tokens=self.s.llm_max_output_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": os.getenv("LLM_EFFORT", "low"),
                               "format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.APITimeoutError:
            return LLMResult(error="таймаут LLM", latency_ms=(time.perf_counter() - t0) * 1000, model=self.s.llm_model)
        except anthropic.RateLimitError:
            return LLMResult(error="LLM rate limit (429)", latency_ms=(time.perf_counter() - t0) * 1000, model=self.s.llm_model)
        except anthropic.APIStatusError as e:
            return LLMResult(error=f"LLM HTTP {e.status_code}", latency_ms=(time.perf_counter() - t0) * 1000, model=self.s.llm_model)
        except anthropic.APIConnectionError:
            return LLMResult(error="LLM недоступен (сеть)", latency_ms=(time.perf_counter() - t0) * 1000, model=self.s.llm_model)
        except TypeError as e:  # SDK бросает TypeError, если не удалось определить способ аутентификации
            err = "LLM: не задан API-ключ" if "authentication" in str(e) else f"LLM ошибка: {type(e).__name__}"
            return LLMResult(error=err, latency_ms=(time.perf_counter() - t0) * 1000, model=self.s.llm_model)
        except Exception as e:
            return LLMResult(error=f"LLM ошибка: {type(e).__name__}", latency_ms=(time.perf_counter() - t0) * 1000, model=self.s.llm_model)
        latency = (time.perf_counter() - t0) * 1000
        usage = resp.usage
        result = LLMResult(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                           cost_usd=self.cost(usage.input_tokens, usage.output_tokens), latency_ms=latency, model=self.s.llm_model)
        if resp.stop_reason == "refusal":
            result.error = "LLM отказ (refusal)"
            return result
        if resp.stop_reason == "max_tokens":
            result.error = "LLM ответ обрезан (max_tokens)"
            return result
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            result.data = json.loads(text)
        except json.JSONDecodeError:
            result.error = "LLM вернул невалидный JSON"
        return result


class OpenAICompatibleProvider(LLMProvider):
    """Локальная или сторонняя модель через OpenAI-совместимый API (LLM_BASE_URL)."""

    name = "openai_compatible"

    @property
    def enabled(self) -> bool:
        return bool(self.s.llm_model and self.s.llm_base_url)

    def generate_json(self, system: str, user: str, schema: dict) -> LLMResult:
        import httpx

        reason = self._precheck(system, user)
        if reason:
            return LLMResult(error=reason, model=self.s.llm_model)
        t0 = time.perf_counter()
        headers = {}
        if os.getenv("OPENAI_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
        body = {
            "model": self.s.llm_model, "max_tokens": self.s.llm_max_output_tokens,
            "messages": [{"role": "system", "content": system + "\nОтвечай только JSON по схеме: " + json.dumps(schema, ensure_ascii=False)},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        try:
            r = httpx.post(self.s.llm_base_url.rstrip("/") + "/chat/completions", json=body, headers=headers, timeout=self.s.llm_timeout_s)
            r.raise_for_status()
            payload = r.json()
        except httpx.TimeoutException:
            return LLMResult(error="таймаут LLM", latency_ms=(time.perf_counter() - t0) * 1000, model=self.s.llm_model)
        except Exception as e:
            return LLMResult(error=f"LLM ошибка: {type(e).__name__}", latency_ms=(time.perf_counter() - t0) * 1000, model=self.s.llm_model)
        usage = payload.get("usage") or {}
        it, ot = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        result = LLMResult(input_tokens=it, output_tokens=ot, cost_usd=self.cost(it, ot),
                           latency_ms=(time.perf_counter() - t0) * 1000, model=self.s.llm_model)
        try:
            result.data = json.loads(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, json.JSONDecodeError, TypeError):
            result.error = "LLM вернул невалидный JSON"
        return result


def has_anthropic_credentials() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def get_provider(settings: Settings | None = None) -> LLMProvider:
    """Провайдер по настройкам; если LLM выключен или не настроен — NullProvider с причиной (видна в /health)."""
    s = settings or get_settings()
    if not s.llm_enabled:
        return NullProvider(s, "LLM_ENABLED=false")
    providers = {"anthropic": AnthropicProvider, "openai_compatible": OpenAICompatibleProvider}
    cls = providers.get(s.llm_provider.lower())
    if cls is None:
        return NullProvider(s, None if s.llm_provider.lower() == "none" else f"неизвестный LLM_PROVIDER={s.llm_provider}")
    p = cls(s)
    if p.enabled:
        return p
    if not s.llm_model:
        reason = "не задан LLM_MODEL"
    elif cls is AnthropicProvider:
        reason = "не задан ANTHROPIC_API_KEY"
    else:
        reason = "не задан LLM_BASE_URL"
    log.warning("LLM_PROVIDER=%s выключен: %s — используются шаблоны", s.llm_provider, reason)
    return NullProvider(s, reason)
