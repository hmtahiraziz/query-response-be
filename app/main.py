"""FastAPI app: ingest portfolio PDFs (Pinecone + Gemini) and generate cover letters."""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import PDF_DIR, ensure_data_dirs, get_settings
from app.models import (
    AssistantRulesResponse,
    AssistantRulesUpdate,
    CoverLetterHistoryDetail,
    CoverLetterHistorySummary,
    CoverLetterHistoryUpdate,
    CoverLetterHistoryVersion,
    CoverLetterRequest,
    CoverLetterResponse,
    IngestResponse,
    ProjectSummary,
    RefineCoverLetterRequest,
    RefineCoverLetterResponse,
    SourceSnippet,
)
from app.services.cover_letter_history_service import (
    append_entry,
    delete_entry,
    get_entry,
    list_summaries,
    update_entry_cover_letter,
)
from app.services.history_versions import normalize_versions_for_detail
from app.services.assistant_rules_service import get_assistant_rules, set_assistant_rules
from app.services.cover_letter_service import generate_cover_letter, refine_cover_letter
from app.services.manifest_service import get_project, list_projects, remove_project, upsert_project
from app.services.pdf_service import chunk_pages_to_documents, extract_pages
from app.services.pinecone_errors import pinecone_connection_user_hint
from app.services.pinecone_service import delete_project_vectors, ingest_documents, reset_index_cache

logger = logging.getLogger(__name__)

_SAFE_NAME = __import__("re").compile(r"[^a-zA-Z0-9._-]")


def _sanitize_filename(name: str) -> str:
    base = Path(name).name
    cleaned = _SAFE_NAME.sub("_", base) or "upload.pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"
    return cleaned


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Expose keys to env for LangChain / Pinecone clients that read os.environ."""
    s = get_settings()
    os.environ["PINECONE_API_KEY"] = s.pinecone_api_key
    os.environ.setdefault("GOOGLE_API_KEY", s.gemini_api_key)
    ensure_data_dirs()
    if (s.mongodb_uri or "").strip():
        from app.services.cover_letter_history_mongo import ensure_indexes as history_ensure_indexes
        from app.services.manifest_mongo import ensure_indexes as manifest_ensure_indexes

        history_ensure_indexes()
        manifest_ensure_indexes()
    yield


app = FastAPI(
    title="Portfolio Cover Letter API",
    description="Ingest project PDFs into Pinecone; generate grounded cover letters with Gemini.",
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/server/info")
def server_info() -> dict[str, Any]:
    s = get_settings()
    return {
        "cover_letter_history_backend": "mongodb"
        if (s.mongodb_uri or "").strip()
        else "json_file",
        "assistant_rules_backend": "mongodb"
        if (s.mongodb_uri or "").strip()
        else "json_file",
        "projects_backend": "mongodb"
        if (s.mongodb_uri or "").strip()
        else "json_file",
        "gemini_chat_model": s.gemini_chat_model,
        "gemini_embed_model": s.gemini_embed_model,
        "gemini_max_retries": s.gemini_max_retries,
        "gemini_retry_cap_seconds": s.gemini_retry_cap_seconds,
        "pinecone_index": s.pinecone_index_name,
        "pinecone_namespace": s.pinecone_namespace,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "default_rag_k": s.rag_k,
    }


@app.get("/assistant/rules", response_model=AssistantRulesResponse)
def get_assistant_rules_endpoint() -> AssistantRulesResponse:
    raw = get_assistant_rules()
    return AssistantRulesResponse(
        global_rules=str(raw.get("global_rules", "")),
        chat_rules=str(raw.get("chat_rules", "")),
        updated_at=int(raw.get("updated_at", 0) or 0),
    )


@app.put("/assistant/rules", response_model=AssistantRulesResponse)
def put_assistant_rules_endpoint(body: AssistantRulesUpdate) -> AssistantRulesResponse:
    raw = set_assistant_rules(global_rules=body.global_rules, chat_rules=body.chat_rules)
    return AssistantRulesResponse(
        global_rules=str(raw.get("global_rules", "")),
        chat_rules=str(raw.get("chat_rules", "")),
        updated_at=int(raw.get("updated_at", 0) or 0),
    )


@app.get("/projects", response_model=list[ProjectSummary])
def get_projects() -> list[ProjectSummary]:
    raw = list_projects()
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
            )
        )
    return out


@app.post("/projects/ingest", response_model=IngestResponse)
async def ingest_project(
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

    settings = get_settings()
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

    upsert_project(
        project_id,
        name=label,
        filename=safe_name,
        pdf_path=str(pdf_path),
        pages=len(pages),
        chunks=len(docs),
    )

    return IngestResponse(
        project_id=project_id,
        name=label,
        filename=safe_name,
        pages=len(pages),
        chunks=len(docs),
    )


@app.delete("/projects/{project_id}")
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


@app.post("/generate/cover-letter", response_model=CoverLetterResponse)
def post_cover_letter(
    body: CoverLetterRequest,
) -> CoverLetterResponse:
    if not list_projects():
        raise HTTPException(
            status_code=400,
            detail="No projects ingested yet. Upload at least one PDF first.",
        )

    settings = get_settings()
    k = body.k if body.k is not None else settings.rag_k

    rules = get_assistant_rules()
    gr = str(rules.get("global_rules", ""))
    cr = str(rules.get("chat_rules", ""))

    try:
        letter, sources = generate_cover_letter(
            body.query,
            k=k,
            settings=settings,
            global_rules=gr,
            chat_rules=cr,
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


@app.post("/generate/cover-letter/refine", response_model=RefineCoverLetterResponse)
def post_refine_cover_letter(body: RefineCoverLetterRequest) -> RefineCoverLetterResponse:
    if not list_projects():
        raise HTTPException(
            status_code=400,
            detail="No projects ingested yet. Upload at least one PDF first.",
        )

    settings = get_settings()
    k = body.k if body.k is not None else settings.rag_k

    rules = get_assistant_rules()
    gr = str(rules.get("global_rules", ""))
    cr = str(rules.get("chat_rules", ""))

    try:
        letter, sources = refine_cover_letter(
            body.client_query,
            body.cover_letter,
            body.instruction,
            selection=body.selection,
            k=k,
            settings=settings,
            global_rules=gr,
            chat_rules=cr,
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


@app.get("/cover-letters/history", response_model=list[CoverLetterHistorySummary])
def get_cover_letter_history() -> list[CoverLetterHistorySummary]:
    raw = list_summaries()
    return [CoverLetterHistorySummary(**row) for row in raw]


@app.get("/cover-letters/history/{entry_id}", response_model=CoverLetterHistoryDetail)
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


@app.patch("/cover-letters/history/{entry_id}")
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


@app.delete("/cover-letters/history/{entry_id}")
def delete_cover_letter_history_entry(entry_id: str) -> dict[str, Any]:
    if not delete_entry(entry_id):
        raise HTTPException(status_code=404, detail="History entry not found.")
    return {"id": entry_id, "removed": True}
