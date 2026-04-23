from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.deps import SettingsDep
from app.schemas import CoverLetterRequest, CoverLetterResponse, RefineCoverLetterRequest, RefineCoverLetterResponse
from app.services.cover_letter_generation_flow import CoverLetterGenError, generate_cover_letter_response
from app.services.cover_letter_refine_flow import RefineCoverLetterGenError, refine_cover_letter_response

router = APIRouter(prefix="/generate", tags=["generation"])


@router.post("/cover-letter", response_model=CoverLetterResponse)
def post_cover_letter(
    body: CoverLetterRequest,
    settings: SettingsDep,
) -> CoverLetterResponse:
    try:
        return generate_cover_letter_response(
            body.query,
            body.k,
            settings,
        )
    except CoverLetterGenError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post("/cover-letter/refine", response_model=RefineCoverLetterResponse)
def post_refine_cover_letter(
    body: RefineCoverLetterRequest,
    settings: SettingsDep,
) -> RefineCoverLetterResponse:
    try:
        return refine_cover_letter_response(
            body.client_query,
            body.cover_letter,
            body.instruction,
            selection=body.selection,
            k=body.k,
            settings=settings,
        )
    except RefineCoverLetterGenError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
