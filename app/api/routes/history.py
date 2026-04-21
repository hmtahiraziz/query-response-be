from typing import Any

from fastapi import APIRouter, HTTPException

from app.schemas import (
    CoverLetterHistoryDetail,
    CoverLetterHistorySummary,
    CoverLetterHistoryUpdate,
    CoverLetterHistoryVersion,
    SourceSnippet,
)
from app.services.cover_letter_history_service import (
    delete_entry,
    get_entry,
    list_summaries,
    update_entry_cover_letter,
)
from app.services.history_versions import normalize_versions_for_detail

router = APIRouter(prefix="/cover-letters", tags=["cover-letter-history"])


@router.get("/history", response_model=list[CoverLetterHistorySummary])
def get_cover_letter_history() -> list[CoverLetterHistorySummary]:
    raw = list_summaries()
    return [CoverLetterHistorySummary(**row) for row in raw]


@router.get("/history/{entry_id}", response_model=CoverLetterHistoryDetail)
def get_cover_letter_history_entry(entry_id: str) -> CoverLetterHistoryDetail:
    row = get_entry(entry_id)
    if not row:
        raise HTTPException(status_code=404, detail="History entry not found.")
    src_raw = row.get("sources") or []
    sources: list[SourceSnippet] = []
    for item in src_raw:
        if isinstance(item, dict):
            sources.append(SourceSnippet(**item))
    vers_raw = row.get("versions") or []
    versions: list[CoverLetterHistoryVersion] = []
    for item in vers_raw:
        if isinstance(item, dict):
            try:
                versions.append(CoverLetterHistoryVersion(**item))
            except Exception:
                continue
    if not versions:
        for item in normalize_versions_for_detail(row):
            versions.append(CoverLetterHistoryVersion(**item))
    return CoverLetterHistoryDetail(
        id=str(row["id"]),
        created_at=int(row.get("created_at", 0)),
        query=str(row.get("query", "")),
        k=row.get("k"),
        cover_letter=str(row.get("cover_letter", "")),
        sources=sources,
        versions=versions,
    )


@router.patch("/history/{entry_id}")
def patch_cover_letter_history(entry_id: str, body: CoverLetterHistoryUpdate) -> dict[str, Any]:
    src_payload = [s.model_dump() for s in body.sources] if body.sources is not None else None
    if not update_entry_cover_letter(
        entry_id,
        body.cover_letter,
        version_source=body.version_source,
        refine_note=body.refine_note,
        sources=src_payload,
    ):
        raise HTTPException(status_code=404, detail="History entry not found.")
    return {"id": entry_id, "updated": True}


@router.delete("/history/{entry_id}")
def delete_cover_letter_history_entry(entry_id: str) -> dict[str, Any]:
    if not delete_entry(entry_id):
        raise HTTPException(status_code=404, detail="History entry not found.")
    return {"id": entry_id, "removed": True}
