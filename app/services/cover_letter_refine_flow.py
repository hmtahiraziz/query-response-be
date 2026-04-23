"""Shared cover-letter refinement for HTTP and Slack."""

from __future__ import annotations

import logging

from app.core.config import Settings
from app.schemas import RefineCoverLetterResponse
from app.services.cover_letter_service import refine_cover_letter
from app.services.manifest_service import list_projects
from app.services.pinecone_errors import pinecone_connection_user_hint

logger = logging.getLogger(__name__)


class RefineCoverLetterGenError(Exception):
    def __init__(self, detail: str, *, status_code: int = 502) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def refine_cover_letter_response(
    client_query: str,
    cover_letter: str,
    instruction: str,
    *,
    selection: str | None,
    k: int | None,
    settings: Settings,
) -> RefineCoverLetterResponse:
    if not list_projects(openai_only=True):
        raise RefineCoverLetterGenError(
            "No OpenAI-indexed projects yet. Upload at least one PDF (legacy Gemini projects are hidden).",
            status_code=400,
        )

    k_eff = k if k is not None else settings.rag_k

    try:
        letter, sources = refine_cover_letter(
            client_query,
            cover_letter,
            instruction,
            selection=selection,
            k=k_eff,
            settings=settings,
        )
    except ValueError as exc:
        raise RefineCoverLetterGenError(str(exc), status_code=400) from exc
    except Exception as exc:
        logger.exception("Cover letter refinement failed")
        hint = pinecone_connection_user_hint(exc)
        msg = hint if hint else f"Refinement failed: {exc}"
        code = 503 if hint else 502
        raise RefineCoverLetterGenError(msg, status_code=code) from exc

    return RefineCoverLetterResponse(cover_letter=letter, sources=sources)
