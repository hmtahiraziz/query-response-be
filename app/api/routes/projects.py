from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.deps import SettingsDep
from app.core.config import PDF_DIR
from app.schemas import GenerateProjectSummaryResponse, IngestResponse, ProjectSummary
from app.services.manifest_service import (
    get_project,
    list_projects,
    remove_project,
    set_project_summary,
    upsert_project,
)
from app.services.pdf_service import chunk_pages_to_documents, extract_pages
from app.services.pinecone_service import delete_project_vectors, ingest_documents, reset_index_cache
from app.services.project_summary_service import generate_project_summary

router = APIRouter(tags=["projects"])
logger = logging.getLogger(__name__)

_SAFE_NAME = __import__("re").compile(r"[^a-zA-Z0-9._-]")


def _sanitize_filename(name: str) -> str:
    base = Path(name).name
    cleaned = _SAFE_NAME.sub("_", base) or "upload.pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"
    return cleaned


@router.get("/projects", response_model=list[ProjectSummary])
def get_projects() -> list[ProjectSummary]:
    raw = list_projects(openai_only=True)
    out: list[ProjectSummary] = []
    for pid, row in sorted(raw.items(), key=lambda x: x[1].get("created_at", 0), reverse=True):
        out.append(
            ProjectSummary(
                project_id=pid,
                name=row.get("name", pid),
                filename=row.get("filename", ""),
                chunks=int(row.get("chunks", 0)),
                pages=int(row.get("pages", 0)),
                created_at=int(row.get("created_at", 0)),
                ai_summary=row.get("ai_summary"),
                summary_generated_at=row.get("summary_generated_at"),
                embedding_provider="openai",
            )
        )
    return out


@router.post("/projects/ingest", response_model=IngestResponse)
async def ingest_project(
    settings: SettingsDep,
    file: UploadFile = File(...),
    display_name: str | None = Form(None),
) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    safe_name = _sanitize_filename(file.filename)
    label = (display_name or Path(safe_name).stem).strip() or "Untitled project"
    if len(label) > 240:
        raise HTTPException(status_code=400, detail="display_name is too long (max 240 characters).")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    project_id = str(uuid.uuid4())
    pdf_path = PDF_DIR / f"{project_id}.pdf"
    pdf_path.write_bytes(content)

    pages = extract_pages(str(pdf_path))
    if not pages:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No extractable text in PDF (try a text-based PDF).")

    docs = chunk_pages_to_documents(
        project_id=project_id,
        project_name=label,
        source_filename=safe_name,
        pages=pages,
        settings=settings,
    )
    if not docs:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No chunks produced from PDF.")

    try:
        ingest_documents(docs, settings=settings)
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        reset_index_cache()
        raise HTTPException(status_code=502, detail=f"Pinecone ingest failed: {exc}") from exc

    profile = None
    summary_generated_at: int | None = None
    try:
        profile = generate_project_summary(project_name=label, pages=pages, settings=settings)
    except Exception:
        logger.exception("Project ingest summary generation failed for project_id=%s", project_id)

    upsert_project(
        project_id,
        name=label,
        filename=safe_name,
        pdf_path=str(pdf_path),
        pages=len(pages),
        chunks=len(docs),
        embedding_provider="openai",
    )
    if profile is not None:
        updated = set_project_summary(project_id, profile.model_dump())
        summary_generated_at = int((updated or {}).get("summary_generated_at") or time.time())

    return IngestResponse(
        project_id=project_id,
        name=label,
        filename=safe_name,
        pages=len(pages),
        chunks=len(docs),
        ai_summary=profile,
        summary_generated_at=summary_generated_at,
        embedding_provider="openai",
    )


@router.delete("/projects/{project_id}")
def delete_project(project_id: str) -> dict[str, Any]:
    entry = get_project(project_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Project not found.")

    try:
        delete_project_vectors(project_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Pinecone delete failed: {exc}") from exc

    remove_project(project_id)
    raw_path = entry.get("pdf_path")
    if raw_path:
        Path(str(raw_path)).unlink(missing_ok=True)

    reset_index_cache()
    return {"project_id": project_id, "removed": True}


@router.post("/projects/{project_id}/summary", response_model=GenerateProjectSummaryResponse)
def generate_project_summary_endpoint(project_id: str, settings: SettingsDep) -> GenerateProjectSummaryResponse:
    entry = get_project(project_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Project not found.")
    raw_path = entry.get("pdf_path")
    if not raw_path:
        raise HTTPException(status_code=400, detail="Project has no PDF path.")
    pdf_path = Path(str(raw_path))
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Project PDF file not found on disk.")

    pages = extract_pages(str(pdf_path))
    if not pages:
        raise HTTPException(status_code=400, detail="No extractable text in the project PDF.")
    profile = generate_project_summary(
        project_name=str(entry.get("name") or project_id),
        pages=pages,
        settings=settings,
    )
    updated = set_project_summary(project_id, profile.model_dump())
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found.")
    ts = int(updated.get("summary_generated_at") or 0)
    return GenerateProjectSummaryResponse(
        project_id=project_id,
        ai_summary=profile,
        summary_generated_at=ts,
    )
