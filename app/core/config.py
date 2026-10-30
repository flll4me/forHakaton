"""Конфигурация сервиса. Все значения читаются из переменных окружения (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:  # python-dotenv необязателен: переменные можно задать в окружении
    pass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_version: str = "0.1.0"
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT_DIR / "data" / "raw" / "handout"))))
    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", str(ROOT_DIR / "models"))))
    rules_path: Path = field(default_factory=lambda: Path(_env("RULES_PATH", str(ROOT_DIR / "data" / "raw" / "handout" / "rules.json"))))
    taxonomy_path: Path = field(default_factory=lambda: Path(_env("TAXONOMY_PATH", str(ROOT_DIR / "data" / "raw" / "handout" / "taxonomy.json"))))
    feedback_db: Path = field(default_factory=lambda: Path(_env("FEEDBACK_DB", str(ROOT_DIR / "data" / "processed" / "feedback.sqlite"))))

    # Эмбеддинги (RAG + дедупликация). Если модель недоступна — работает лексический fallback.
    embeddings_enabled: bool = field(default_factory=lambda: _env_bool("EMBEDDINGS_ENABLED", True))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))

    # LLM. LLM_ENABLED=false — общий выключатель: шаблоны при любом LLM_PROVIDER.
    llm_enabled: bool = field(default_factory=lambda: _env_bool("LLM_ENABLED", True))
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "none"))  # none | anthropic | openai_compatible
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", ""))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 3.0))
    llm_max_input_tokens: int = field(default_factory=lambda: int(_env_float("LLM_MAX_INPUT_TOKENS", 1500)))
    llm_max_output_tokens: int = field(default_factory=lambda: int(_env_float("LLM_MAX_OUTPUT_TOKENS", 250)))
    llm_price_in_per_mtok: float = field(default_factory=lambda: _env_float("LLM_PRICE_IN_PER_MTOK", 0.0))
    llm_price_out_per_mtok: float = field(default_factory=lambda: _env_float("LLM_PRICE_OUT_PER_MTOK", 0.0))
    llm_max_cost_per_ticket: float = field(default_factory=lambda: _env_float("LLM_MAX_COST_PER_TICKET", 0.002))

    # Пороги эскалации (подбираются на контрольной выборке, см. reports/evaluation_report.md)
    min_category_confidence: float = field(default_factory=lambda: _env_float("MIN_CATEGORY_CONFIDENCE", 0.55))
    min_retrieval_score: float = field(default_factory=lambda: _env_float("MIN_RETRIEVAL_SCORE", 0.25))
    duplicate_window_minutes: int = field(default_factory=lambda: int(_env_float("DUPLICATE_WINDOW_MINUTES", 60)))
    duplicate_similarity: float = field(default_factory=lambda: _env_float("DUPLICATE_SIMILARITY", 0.6))
    related_similarity: float = field(default_factory=lambda: _env_float("RELATED_SIMILARITY", 0.3))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
