"""Persist assistant (global + chat) rules in MongoDB."""

from __future__ import annotations

import time
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection

from app.config import get_settings

_client: MongoClient | None = None

DOC_ID = "assistant_rules"


def _collection() -> Collection:
    global _client
    s = get_settings()
    uri = (s.mongodb_uri or "").strip()
    if not uri:
        raise RuntimeError("MongoDB URI is not configured")
    if _client is None:
        _client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    return _client[s.mongodb_db_name][s.mongodb_collection_assistant_rules]


def get_rules() -> dict[str, Any]:
    doc = _collection().find_one({"_id": DOC_ID})
    if not doc:
        return {"global_rules": "", "chat_rules": "", "updated_at": 0}
    return {
        "global_rules": str(doc.get("global_rules", "")),
        "chat_rules": str(doc.get("chat_rules", "")),
        "updated_at": int(doc.get("updated_at", 0) or 0),
    }


def replace_rules(*, global_rules: str, chat_rules: str) -> dict[str, Any]:
    now = int(time.time())
    payload = {
        "_id": DOC_ID,
        "global_rules": global_rules,
        "chat_rules": chat_rules,
        "updated_at": now,
    }
    _collection().replace_one({"_id": DOC_ID}, payload, upsert=True)
    return {"global_rules": global_rules, "chat_rules": chat_rules, "updated_at": now}
