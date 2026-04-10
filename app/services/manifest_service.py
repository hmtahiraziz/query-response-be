"""Local JSON manifest for project metadata (Pinecone holds vectors)."""

from __future__ import annotations

import json
import time
from typing import Any

from app.config import MANIFEST_PATH, ensure_data_dirs


def _read() -> dict[str, Any]:
    ensure_data_dirs()
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write(data: dict[str, Any]) -> None:
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
    data = _read()
    data[project_id] = {
        "project_id": project_id,
        "name": name,
        "filename": filename,
        "pdf_path": pdf_path,
        "pages": pages,
        "chunks": chunks,
        "created_at": int(time.time()),
    }
    _write(data)


def remove_project(project_id: str) -> dict[str, Any] | None:
    data = _read()
    entry = data.pop(project_id, None)
    if entry is None:
        return None
    _write(data)
    return entry


def list_projects() -> dict[str, Any]:
    return _read()


def get_project(project_id: str) -> dict[str, Any] | None:
    return _read().get(project_id)
