"""Retry OpenAI (and compatible) chat / embedding calls on 429 and quota errors."""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class RetryingEmbeddings(Embeddings):
    """Wraps an Embeddings model with the same retry policy as chat."""

    def __init__(
        self,
        inner: Embeddings,
        *,
        max_retries: int,
        retry_cap_seconds: float,
    ) -> None:
        self._inner = inner
        self._max_retries = max_retries
        self._retry_cap_seconds = retry_cap_seconds

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return run_with_retry(
            lambda: self._inner.embed_documents(texts),
            max_retries=self._max_retries,
            retry_cap_seconds=self._retry_cap_seconds,
            operation="openai_embed_documents",
        )

    def embed_query(self, text: str) -> list[float]:
        return run_with_retry(
            lambda: self._inner.embed_query(text),
            max_retries=self._max_retries,
            retry_cap_seconds=self._retry_cap_seconds,
            operation="openai_embed_query",
        )


_RETRY_IN_SECONDS = re.compile(
    r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
    re.IGNORECASE,
)
# Protobuf-style text dumps sometimes include: retry_delay { seconds: 15 }
_RETRY_DELAY_BRACE = re.compile(r"retry_delay\s*\{\s*seconds:\s*([0-9]+)", re.IGNORECASE)
# OpenAI / HTTP style hints
_RETRY_AFTER_WORD = re.compile(
    r"(?:try again in|retry after)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s(?:ec(?:ond)?s?)?)?",
    re.IGNORECASE,
)


def _is_rate_limit_or_quota(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    if "429" in msg:
        return True
    if "ratelimit" in name or "rate_limit" in name:
        return True
    if "resource_exhausted" in msg or "resource exhausted" in msg:
        return True
    if "quota" in msg and ("exceed" in msg or "exceeded" in msg):
        return True
    if "rate limit" in msg or "too many requests" in msg:
        return True
    if "insufficient_quota" in msg:
        return True
    return False


def parse_retry_after_seconds(exc: BaseException) -> float | None:
    """Best-effort parse of server-suggested wait time from error strings."""
    msg = str(exc)
    m = _RETRY_IN_SECONDS.search(msg)
    if m:
        return max(0.0, float(m.group(1)))
    m = _RETRY_DELAY_BRACE.search(msg)
    if m:
        return max(0.0, float(m.group(1)))
    m = _RETRY_AFTER_WORD.search(msg)
    if m:
        return max(0.0, float(m.group(1)))
    return None


def run_with_retry(
    fn: Callable[[], Any],
    *,
    max_retries: int,
    retry_cap_seconds: float,
    operation: str,
) -> Any:
    """
    Run ``fn``; on rate-limit style errors, sleep and retry.

    Uses API hint when present, else exponential backoff with jitter (capped).
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except BaseException as exc:
            can_retry = _is_rate_limit_or_quota(exc) and attempt < max_retries - 1
            if not can_retry:
                logger.error(
                    "%s failed (attempt %s/%s): %s",
                    operation,
                    attempt + 1,
                    max_retries,
                    exc,
                    exc_info=True,
                )
                raise
            hint = parse_retry_after_seconds(exc)
            if hint is not None:
                delay = min(hint + random.uniform(0.05, 0.35), retry_cap_seconds)
            else:
                base = min(2**attempt, 64)
                delay = min(base + random.uniform(0, 1.0), retry_cap_seconds)
            logger.warning(
                "%s rate limited (attempt %s/%s), sleeping %.1fs",
                operation,
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"{operation}: retry loop exhausted unexpectedly")


def invoke_chat_with_retry(
    llm: BaseChatModel,
    prompt: str | list[BaseMessage],
    *,
    max_retries: int,
    retry_cap_seconds: float,
) -> BaseMessage:
    return run_with_retry(
        lambda: llm.invoke(prompt),
        max_retries=max_retries,
        retry_cap_seconds=retry_cap_seconds,
        operation="openai_chat",
    )
