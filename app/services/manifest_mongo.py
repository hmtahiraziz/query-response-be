"""MongoDB persistence for portfolio project manifest (metadata; PDF files stay on disk)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from pymongo import DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from app.config import MANIFEST_PATH, get_settings

logger = logging.getLogger(__name__)

_client: MongoClient | None = None


def _collection() -> Collection:
    global _client
    s = get_settings()
    uri = (s.mongodb_uri or "").strip()
    if not uri:
        raise RuntimeError("MongoDB URI is not configured")
    if _client is None:
        _client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    return _client[s.mongodb_db_name][s.mongodb_collection_projects]


def ensure_indexes() -> None:
    try:
        coll = _collection()
        coll.create_index([("created_at", DESCENDING)])
    except PyMongoError:
        pass
    try:
        migrate_from_json_if_empty()
    except Exception:
        logger.exception("manifest_mongo: migrate_from_json_if_empty failed")


def migrate_from_json_if_empty() -> None:
    """If the collection is empty and ``data/projects.json`` exists, import it once."""
    coll = _collection()
    if coll.count_documents({}) > 0:
        return
    if not MANIFEST_PATH.exists():
        return
    try:
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(raw, dict) or not raw:
        return
    batch: list[dict[str, Any]] = []
    for pid, row in raw.items():
        if not isinstance(row, dict):
            continue
        doc = dict(row)
        doc["_id"] = str(pid)
        doc["project_id"] = str(row.get("project_id") or pid)
        batch.append(doc)
    if batch:
        coll.insert_many(batch)


def list_all() -> dict[str, Any]:
    coll = _collection()
    out: dict[str, Any] = {}
    for row in coll.find({}):
        pid = str(row.get("project_id") or row.get("_id"))
        item = {k: v for k, v in row.items() if k != "_id"}
        item["project_id"] = pid
        out[pid] = item
    return out


def get_one(project_id: str) -> dict[str, Any] | None:
    coll = _collection()
    row = coll.find_one({"$or": [{"_id": project_id}, {"project_id": project_id}]})
    if not row:
        return None
    pid = str(row.get("project_id") or row.get("_id"))
    item = {k: v for k, v in row.items() if k != "_id"}
    item["project_id"] = pid
    return item


def upsert_one(
    project_id: str,
    *,
    name: str,
    filename: str,
    pdf_path: str,
    pages: int,
    chunks: int,
) -> None:
    coll = _collection()
    doc: dict[str, Any] = {
        "_id": project_id,
        "project_id": project_id,
        "name": name,
        "filename": filename,
        "pdf_path": pdf_path,
        "pages": pages,
        "chunks": chunks,
        "created_at": int(time.time()),
    }
    coll.replace_one({"_id": project_id}, doc, upsert=True)


def delete_one(project_id: str) -> dict[str, Any] | None:
    coll = _collection()
    row = coll.find_one_and_delete({"$or": [{"_id": project_id}, {"project_id": project_id}]})
    if not row:
        return None
    pid = str(row.get("project_id") or row.get("_id"))
    item = {k: v for k, v in row.items() if k != "_id"}
    item["project_id"] = pid
    return item
