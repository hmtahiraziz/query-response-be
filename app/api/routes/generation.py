from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException

from app.api.deps import SettingsDep
from app.schemas import CoverLetterRequest, CoverLetterResponse, RefineCoverLetterRequest, RefineCoverLetterResponse
from app.services.cover_letter_history_service import append_entry
from app.services.cover_letter_service import generate_cover_letter, refine_cover_letter
from app.services.manifest_service import list_projects
from app.services.pinecone_errors import pinecone_connection_user_hint

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate", tags=["generation"])


@router.post("/cover-letter", response_model=CoverLetterResponse)
def post_cover_letter(
    body: CoverLetterRequest,
    settings: SettingsDep,
) -> CoverLetterResponse:
    if not list_projects(openai_only=True):
        raise HTTPException(
            status_code=400,
            detail="No OpenAI-indexed projects yet. Upload at least one PDF (legacy Gemini projects are hidden).",
        )

    k = body.k if body.k is not None else settings.rag_k

    try:
        letter, sources = generate_cover_letter(
            body.query,
            k=k,
            settings=settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Cover letter generation failed")
        hint = pinecone_connection_user_hint(exc)
        raise HTTPException(
            status_code=503 if hint else 502,
            detail=hint if hint else f"Generation failed: {exc}",
        ) from exc

    history_id = str(uuid.uuid4())
    append_entry(
        query=body.query,
        k=k,
        cover_letter=letter,
        sources=[s.model_dump() for s in sources],
        entry_id=history_id,
    )

    return CoverLetterResponse(cover_letter=letter, sources=sources, history_id=history_id)


@router.post("/cover-letter/refine", response_model=RefineCoverLetterResponse)
def post_refine_cover_letter(
    body: RefineCoverLetterRequest,
    settings: SettingsDep,
) -> RefineCoverLetterResponse:
    if not list_projects(openai_only=True):
        raise HTTPException(
            status_code=400,
            detail="No OpenAI-indexed projects yet. Upload at least one PDF (legacy Gemini projects are hidden).",
        )

    k = body.k if body.k is not None else settings.rag_k

    try:
        letter, sources = refine_cover_letter(
            body.client_query,
            body.cover_letter,
            body.instruction,
            selection=body.selection,
            k=k,
            settings=settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Cover letter refinement failed")
        hint = pinecone_connection_user_hint(exc)
        raise HTTPException(
            status_code=503 if hint else 502,
            detail=hint if hint else f"Refinement failed: {exc}",
        ) from exc

    return RefineCoverLetterResponse(cover_letter=letter, sources=sources)
