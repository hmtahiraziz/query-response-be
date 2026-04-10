"""Persist generated cover letters: MongoDB when MONGODB_URI is set, else local JSON."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from app.config import COVER_LETTER_HISTORY_MAX, COVER_LETTER_HISTORY_PATH, ensure_data_dirs, get_settings
from app.services.history_versions import normalize_versions_for_detail


def _use_mongo() -> bool:
    try:
        return bool((get_settings().mongodb_uri or "").strip())
    except Exception:
        return False


def _read_raw() -> list[dict[str, Any]]:
    ensure_data_dirs()
    if not COVER_LETTER_HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(COVER_LETTER_HISTORY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_raw(rows: list[dict[str, Any]]) -> None:
    ensure_data_dirs()
    COVER_LETTER_HISTORY_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def append_entry(
    *,
    query: str,
    k: int | None,
    cover_letter: str,
    sources: list[dict[str, Any]],
    entry_id: str,
) -> None:
    if _use_mongo():
        from app.services.cover_letter_history_mongo import append_entry as mongo_append

        mongo_append(
            query=query,
            k=k,
            cover_letter=cover_letter,
            sources=sources,
            entry_id=entry_id,
            max_entries=COVER_LETTER_HISTORY_MAX,
        )
        return

    rows = _read_raw()
    now = int(time.time())
    v0_id = f"{entry_id}-initial"
    entry: dict[str, Any] = {
        "id": entry_id,
        "created_at": now,
        "query": query,
        "k": k,
        "cover_letter": cover_letter,
        "sources": sources,
        "versions": [
            {
                "id": v0_id,
                "created_at": now,
                "source": "generate",
                "body": cover_letter,
                "refine_note": None,
            }
        ],
    }
    rows.insert(0, entry)
    rows = rows[:COVER_LETTER_HISTORY_MAX]
    _write_raw(rows)


def list_summaries() -> list[dict[str, Any]]:
    if _use_mongo():
        from app.services.cover_letter_history_mongo import list_summaries as mongo_list

        return mongo_list()

    out: list[dict[str, Any]] = []
    for row in _read_raw():
        q = str(row.get("query", ""))
        preview = q if len(q) <= 200 else q[:197] + "…"
        out.append(
            {
                "id": row.get("id"),
                "created_at": int(row.get("created_at", 0)),
                "query_preview": preview,
                "k": row.get("k"),
            }
        )
    return out


def get_entry(entry_id: str) -> dict[str, Any] | None:
    if _use_mongo():
        from app.services.cover_letter_history_mongo import get_entry as mongo_get

        return mongo_get(entry_id)

    for row in _read_raw():
        if row.get("id") == entry_id:
            r = dict(row)
            r["versions"] = normalize_versions_for_detail(r)
            return r
    return None


def delete_entry(entry_id: str) -> bool:
    if _use_mongo():
        from app.services.cover_letter_history_mongo import delete_entry as mongo_delete

        return mongo_delete(entry_id)

    rows = _read_raw()
    n = len(rows)
    rows = [r for r in rows if r.get("id") != entry_id]
    if len(rows) == n:
        return False
    _write_raw(rows)
    return True


def update_entry_cover_letter(
    entry_id: str,
    cover_letter: str,
    *,
    version_source: str = "manual",
    refine_note: str | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> bool:
    if _use_mongo():
        from app.services.cover_letter_history_mongo import update_cover_letter as mongo_update

        return mongo_update(
            entry_id,
            cover_letter,
            version_source=version_source,
            refine_note=refine_note,
            sources=sources,
        )

    rows = _read_raw()
    now = int(time.time())
    snap: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "created_at": now,
        "source": version_source,
        "body": cover_letter,
    }
    if version_source == "refine" and refine_note:
        rn = refine_note if len(refine_note) <= 500 else refine_note[:497] + "…"
        snap["refine_note"] = rn

    for i, row in enumerate(rows):
        if row.get("id") == entry_id:
            row["cover_letter"] = cover_letter
            if sources is not None:
                row["sources"] = sources
            vers = row.get("versions")
            if not isinstance(vers, list):
                vers = []
            vers.append(snap)
            row["versions"] = vers
            rows[i] = row
            _write_raw(rows)
            return True
    return False
