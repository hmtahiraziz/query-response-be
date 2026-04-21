"""Cover-letter assistant rules bundled as JSON in the repository (no UI/DB free text)."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from app.schemas.assistant import CoverLetterAssistantRulesBundle

logger = logging.getLogger(__name__)

_RULES_FILE = Path(__file__).resolve().parent.parent / "rules" / "cover_letter_assistant.default.json"


class AssistantRulesLoadError(RuntimeError):
    pass


@lru_cache
def get_cover_letter_rules_bundle() -> CoverLetterAssistantRulesBundle:
    if not _RULES_FILE.is_file():
        raise AssistantRulesLoadError(f"Assistant rules file missing: {_RULES_FILE}")
    try:
        raw = json.loads(_RULES_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssistantRulesLoadError(f"Invalid assistant rules JSON: {_RULES_FILE}") from exc
    try:
        return CoverLetterAssistantRulesBundle.model_validate(raw)
    except Exception as exc:
        raise AssistantRulesLoadError(f"Assistant rules failed validation: {_RULES_FILE}") from exc


def rules_bundle_relative_path() -> str:
    return "app/rules/cover_letter_assistant.default.json"


def reload_cover_letter_rules_bundle() -> CoverLetterAssistantRulesBundle:
    """Clear cache after tests or hot edits in dev."""
    get_cover_letter_rules_bundle.cache_clear()
    return get_cover_letter_rules_bundle()
