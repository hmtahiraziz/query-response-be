"""Portfolio project manifest: MongoDB when MONGODB_URI is set, else data/projects.json."""

from __future__ import annotations

import json
import time
from typing import Any

from app.config import MANIFEST_PATH, ensure_data_dirs, get_settings


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


def list_projects() -> dict[str, Any]:
    if _use_mongo():
        from app.services.manifest_mongo import list_all as mongo_list

        return mongo_list()
    return _read_json()


def get_project(project_id: str) -> dict[str, Any] | None:
    if _use_mongo():
        from app.services.manifest_mongo import get_one as mongo_get

        return mongo_get(project_id)
    return _read_json().get(project_id)
