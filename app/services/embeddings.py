"""Мультиязычные эмбеддинги (sentence-transformers) с ленивой загрузкой.

Модель по умолчанию: paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 50+ языков,
включая ru/kk/uz/en/es/pt; 118M параметров, работает на CPU).
Если модель недоступна (нет сети/отключено в .env), `available()` возвращает False и
вызывающие модули переходят на лексический fallback — без подмены результатов.
"""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
_lock = threading.Lock()
_failed: set[str] = set()


@lru_cache(maxsize=2)
def _load(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


def get_model(name: str | None = None):
    settings = get_settings()
    name = name or settings.embedding_model
    if not settings.embeddings_enabled or name in _failed:
        return None
    with _lock:
        try:
            return _load(name)
        except Exception as e:  # нет сети, нет файлов модели и т.п.
            log.warning("embedding model %s unavailable: %s", name, e)
            _failed.add(name)
            return None


def available() -> bool:
    return get_model() is not None


def encode(texts: list[str], batch_size: int = 64) -> np.ndarray | None:
    """L2-нормированные эмбеддинги (косинус = скалярное произведение) или None."""
    model = get_model()
    if model is None:
        return None
    vecs = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


class EmbeddingFeatures(BaseEstimator, TransformerMixin):
    """sklearn-трансформер для подхода C. Хранит только имя модели — сериализуется легко."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        model = get_model(self.model_name)
        if model is None:
            raise RuntimeError("embedding model unavailable")
        return model.encode(list(X), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
