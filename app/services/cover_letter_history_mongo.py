"""MongoDB persistence for cover letter history."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from app.config import COVER_LETTER_HISTORY_MAX, get_settings
from app.services.history_versions import normalize_versions_for_detail

_client: MongoClient | None = None


def _get_collection() -> Collection:
    global _client
    s = get_settings()
    uri = (s.mongodb_uri or "").strip()
    if not uri:
        raise RuntimeError("MongoDB URI is not configured")
    if _client is None:
        _client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    return _client[s.mongodb_db_name][s.mongodb_collection_cover_letters]


def ensure_indexes() -> None:
    try:
        coll = _get_collection()
        coll.create_index([("created_at", DESCENDING)])
    except PyMongoError:
        pass


def append_entry(
    *,
    query: str,
    k: int | None,
    cover_letter: str,
    sources: list[dict[str, Any]],
    entry_id: str,
    max_entries: int,
) -> None:
    coll = _get_collection()
    now = int(time.time())
    v0_id = f"{entry_id}-initial"
    initial_version: dict[str, Any] = {
        "id": v0_id,
        "created_at": now,
        "source": "generate",
        "body": cover_letter,
        "refine_note": None,
    }
    doc: dict[str, Any] = {
        "_id": entry_id,
        "id": entry_id,
        "created_at": now,
        "query": query,
        "k": k,
        "cover_letter": cover_letter,
        "sources": sources,
        "versions": [initial_version],
    }
    coll.insert_one(doc)

    total = coll.count_documents({})
    if total > max_entries:
        excess = total - max_entries
        oldest = list(
            coll.find({}, {"_id": 1}).sort("created_at", ASCENDING).limit(excess)
        )
        if oldest:
            coll.delete_many({"_id": {"$in": [x["_id"] for x in oldest]}})


def list_summaries() -> list[dict[str, Any]]:
    coll = _get_collection()
    out: list[dict[str, Any]] = []
    for row in coll.find({}).sort("created_at", DESCENDING).limit(COVER_LETTER_HISTORY_MAX):
        q = str(row.get("query", ""))
        preview = q if len(q) <= 200 else q[:197] + "…"
        out.append(
            {
                "id": row.get("id") or row.get("_id"),
                "created_at": int(row.get("created_at", 0)),
                "query_preview": preview,
                "k": row.get("k"),
            }
        )
    return out


def get_entry(entry_id: str) -> dict[str, Any] | None:
    coll = _get_collection()
    row = coll.find_one({"id": entry_id}) or coll.find_one({"_id": entry_id})
    if not row:
        return None
    return _row_to_api(row)


def delete_entry(entry_id: str) -> bool:
    coll = _get_collection()
    res = coll.delete_one({"$or": [{"id": entry_id}, {"_id": entry_id}]})
    return res.deleted_count > 0


def update_cover_letter(
    entry_id: str,
    cover_letter: str,
    *,
    version_source: str = "manual",
    refine_note: str | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> bool:
    """Set latest cover_letter, optional sources (refine), append version snapshot."""
    coll = _get_collection()
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

    set_doc: dict[str, Any] = {"cover_letter": cover_letter}
    if sources is not None:
        set_doc["sources"] = sources

    res = coll.update_one(
        {"$or": [{"id": entry_id}, {"_id": entry_id}]},
        {"$set": set_doc, "$push": {"versions": snap}},
    )
    return res.matched_count > 0


def _row_to_api(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id") or row.get("_id"),
        "created_at": int(row.get("created_at", 0)),
        "query": str(row.get("query", "")),
        "k": row.get("k"),
        "cover_letter": str(row.get("cover_letter", "")),
        "sources": row.get("sources") if isinstance(row.get("sources"), list) else [],
        "versions": normalize_versions_for_detail(row),
    }
