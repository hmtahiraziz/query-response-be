"""Assistant rules: MongoDB when MONGODB_URI is set, else data/assistant_rules.json."""

from __future__ import annotations

import json
import time
from typing import Any

from app.config import ASSISTANT_RULES_PATH, ensure_data_dirs, get_settings


def _use_mongo() -> bool:
    try:
        return bool((get_settings().mongodb_uri or "").strip())
    except Exception:
        return False


def _read_json_file() -> dict[str, Any]:
    ensure_data_dirs()
    if not ASSISTANT_RULES_PATH.exists():
        return {"global_rules": "", "chat_rules": "", "updated_at": 0}
    try:
        data = json.loads(ASSISTANT_RULES_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"global_rules": "", "chat_rules": "", "updated_at": 0}
        return {
            "global_rules": str(data.get("global_rules", "")),
            "chat_rules": str(data.get("chat_rules", "")),
            "updated_at": int(data.get("updated_at", 0) or 0),
        }
    except Exception:
        return {"global_rules": "", "chat_rules": "", "updated_at": 0}


def _write_json_file(global_rules: str, chat_rules: str) -> dict[str, Any]:
    ensure_data_dirs()
    now = int(time.time())
    payload = {"global_rules": global_rules, "chat_rules": chat_rules, "updated_at": now}
    ASSISTANT_RULES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**payload}


def get_assistant_rules() -> dict[str, Any]:
    if _use_mongo():
        from app.services.assistant_rules_mongo import get_rules as mongo_get

        return mongo_get()
    return _read_json_file()


def set_assistant_rules(*, global_rules: str, chat_rules: str) -> dict[str, Any]:
    if _use_mongo():
        from app.services.assistant_rules_mongo import replace_rules as mongo_replace

        return mongo_replace(global_rules=global_rules, chat_rules=chat_rules)
    return _write_json_file(global_rules, chat_rules)
