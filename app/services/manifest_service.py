"""Portfolio project manifest: MongoDB when MONGODB_URI is set, else data/projects.json."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.core.config import MANIFEST_PATH, ensure_data_dirs, get_settings

logger = logging.getLogger(__name__)


def _use_mongo() -> bool:
    try:
        return bool((get_settings().mongodb_uri or "").strip())
    except Exception:
        return False


def _read_json() -> dict[str, Any]:
    ensure_data_dirs()
    if not MANIFEST_PATH.exists():
        return {}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(data: dict[str, Any]) -> None:
    ensure_data_dirs()
    MANIFEST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def upsert_project(
    project_id: str,
    *,
    name: str,
    filename: str,
    pdf_path: str,
    pages: int,
    chunks: int,
    embedding_provider: str = "openai",
) -> None:
    if _use_mongo():
        from app.services.manifest_mongo import upsert_one as mongo_upsert

        mongo_upsert(
            project_id,
            name=name,
            filename=filename,
            pdf_path=pdf_path,
            pages=pages,
            chunks=chunks,
            embedding_provider=embedding_provider,
        )
        return

    data = _read_json()
    data[project_id] = {
        "project_id": project_id,
        "name": name,
        "filename": filename,
        "pdf_path": pdf_path,
        "pages": pages,
        "chunks": chunks,
        "created_at": int(time.time()),
        "embedding_provider": embedding_provider,
    }
    _write_json(data)


def remove_project(project_id: str) -> dict[str, Any] | None:
    if _use_mongo():
        from app.services.manifest_mongo import delete_one as mongo_delete

        return mongo_delete(project_id)

    data = _read_json()
    entry = data.pop(project_id, None)
    if entry is None:
        return None
    _write_json(data)
    return entry


def list_projects(*, openai_only: bool = False) -> dict[str, Any]:
    if _use_mongo():
        from app.services.manifest_mongo import list_all as mongo_list

        raw = mongo_list()
    else:
        raw = _read_json()
    if not openai_only:
        return raw
    return {
        pid: row
        for pid, row in raw.items()
        if isinstance(row, dict) and row.get("embedding_provider") == "openai"
    }


def get_project(project_id: str) -> dict[str, Any] | None:
    if _use_mongo():
        from app.services.manifest_mongo import get_one as mongo_get

        return mongo_get(project_id)
    return _read_json().get(project_id)


def set_project_summary(project_id: str, summary: dict[str, Any]) -> dict[str, Any] | None:
    if _use_mongo():
        from app.services.manifest_mongo import set_project_summary as mongo_set_summary

        return mongo_set_summary(project_id, summary)

    data = _read_json()
    row = data.get(project_id)
    if not isinstance(row, dict):
        return None
    row["ai_summary"] = dict(summary)
    row["summary_generated_at"] = int(time.time())
    _write_json(data)
    return row


def backfill_legacy_embedding_providers() -> None:
    """Mark pre-tracking projects as gemini; new ingests set openai in upsert_project."""
    if _use_mongo():
        from app.services.manifest_mongo import backfill_missing_embedding_provider

        try:
            backfill_missing_embedding_provider()
        except Exception:
            # Don't block API startup when Mongo is temporarily unavailable.
            logger.warning(
                "Skipping manifest embedding_provider backfill because MongoDB is unavailable.",
                exc_info=True,
            )
        return

    data = _read_json()
    changed = False
    for _pid, row in list(data.items()):
        if not isinstance(row, dict):
            continue
        if row.get("embedding_provider") not in ("gemini", "openai"):
            row["embedding_provider"] = "gemini"
            changed = True
    if changed:
        _write_json(data)
