"""Normalize persisted `versions` on cover letter history rows (API + legacy rows)."""

from __future__ import annotations

import uuid
from typing import Any


def normalize_versions_for_detail(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a non-empty list of version dicts for API serialization."""
    entry_id = str(row.get("id") or row.get("_id") or "")
    raw = row.get("versions")
    cover = str(row.get("cover_letter", ""))
    created = int(row.get("created_at", 0))

    if isinstance(raw, list) and len(raw) > 0:
        out: list[dict[str, Any]] = []
        for v in raw:
            if not isinstance(v, dict):
                continue
            src = v.get("source")
            if src not in ("generate", "refine", "manual"):
                src = "manual"
            out.append(
                {
                    "id": str(v.get("id") or uuid.uuid4()),
                    "created_at": int(v.get("created_at", created)),
                    "source": src,
                    "body": str(v.get("body", "")),
                    "refine_note": v.get("refine_note"),
                }
            )
        if out:
            return out

    return [
        {
            "id": f"{entry_id}-initial" if entry_id else str(uuid.uuid4()),
            "created_at": created,
            "source": "generate",
            "body": cover,
            "refine_note": None,
        }
    ]
