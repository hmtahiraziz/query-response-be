"""Gemini embeddings + chat (no Ollama)."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import Settings, get_settings
from app.services.gemini_retry import RetryingEmbeddings


@lru_cache
def get_embeddings_model() -> Embeddings:
    s = get_settings()
    inner = GoogleGenerativeAIEmbeddings(
        model=s.gemini_embed_model,
        google_api_key=s.gemini_api_key,
    )
    return RetryingEmbeddings(
        inner,
        max_retries=s.gemini_max_retries,
        retry_cap_seconds=s.gemini_retry_cap_seconds,
    )


def get_chat_model(settings: Settings | None = None, *, temperature: float = 0.4):
    s = settings or get_settings()
    return ChatGoogleGenerativeAI(
        model=s.gemini_chat_model,
        google_api_key=s.gemini_api_key,
        temperature=temperature,
    )
