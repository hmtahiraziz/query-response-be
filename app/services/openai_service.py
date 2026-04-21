"""OpenAI embeddings + chat."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import Settings, get_settings
from app.services.llm_retry import RetryingEmbeddings


@lru_cache
def get_embeddings_model() -> Embeddings:
    s = get_settings()
    inner = OpenAIEmbeddings(
        model=s.openai_embed_model,
        dimensions=s.openai_embed_dimensions,
        openai_api_key=s.openai_api_key,
    )
    return RetryingEmbeddings(
        inner,
        max_retries=s.openai_max_retries,
        retry_cap_seconds=s.openai_retry_cap_seconds,
    )


def get_chat_model(settings: Settings | None = None, *, temperature: float = 0.4):
    s = settings or get_settings()
    return ChatOpenAI(
        model=s.openai_chat_model,
        openai_api_key=s.openai_api_key,
        temperature=temperature,
    )
