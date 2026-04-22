"""Shared cover-letter generation + history persistence for HTTP and Slack."""

from __future__ import annotations

import logging
import uuid

from app.core.config import Settings
from app.schemas import CoverLetterResponse
from app.services.cover_letter_history_service import append_entry
from app.services.cover_letter_service import generate_cover_letter
from app.services.manifest_service import list_projects
from app.services.pinecone_errors import pinecone_connection_user_hint

logger = logging.getLogger(__name__)


class CoverLetterGenError(Exception):
    """Maps to HTTP status via `.status_code` and `.detail`."""

    def __init__(self, detail: str, *, status_code: int = 502) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def generate_cover_letter_response(
    query: str,
    k: int | None,
    settings: Settings,
) -> CoverLetterResponse:
    if not list_projects(openai_only=True):
        raise CoverLetterGenError(
            "No OpenAI-indexed projects yet. Upload at least one PDF (legacy Gemini projects are hidden).",
            status_code=400,
        )

    k_eff = k if k is not None else settings.rag_k

    try:
        letter, sources = generate_cover_letter(
            query,
            k=k_eff,
            settings=settings,
        )
    except ValueError as exc:
        raise CoverLetterGenError(str(exc), status_code=400) from exc
    except Exception as exc:
        logger.exception("Cover letter generation failed")
        hint = pinecone_connection_user_hint(exc)
        msg = hint if hint else f"Generation failed: {exc}"
        code = 503 if hint else 502
        raise CoverLetterGenError(msg, status_code=code) from exc

    history_id = str(uuid.uuid4())
    append_entry(
        query=query,
        k=k_eff,
        cover_letter=letter,
        sources=[s.model_dump() for s in sources],
        entry_id=history_id,
    )

    return CoverLetterResponse(cover_letter=letter, sources=sources, history_id=history_id)
