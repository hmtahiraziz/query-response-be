"""Detect bundled-rule violations in draft text (language, anti-template, openings)."""

from __future__ import annotations

import re

from app.schemas.assistant import CoverLetterAssistantRulesBundle


def _term_matches(lower_text: str, term: str) -> bool:
    """Match a forbidden term: phrases by substring; longer stems catch common inflections (e.g. seamless / seamlessly)."""
    t = term.strip().lower()
    if not t:
        return False
    if " " in t or "-" in t:
        return t in lower_text
    # Longer tokens: substring catches inflections (robust/robustly, seamless/seamlessly) without extra stems list
    if len(t) >= 6:
        return t in lower_text
    # Short tokens: word-ish match to reduce false positives (e.g. "help" in "unhelpful")
    return bool(re.search(rf"(?<![\\w'-]){re.escape(t)}(?![\\w'-])", lower_text))


def scan_bundle_violations(draft: str, bundle: CoverLetterAssistantRulesBundle) -> list[str]:
    """Return human-readable violation lines for prompts and logging."""
    if not draft:
        return []
    lower = draft.strip().lower()
    found: list[str] = []
    hc = bundle.hard_constraints

    for s in hc.opening.forbidden_starts:
        t = str(s).strip().lower()
        if t and lower.startswith(t):
            found.append(f"Opening starts like forbidden pattern `{s}`")

    for s in hc.closing.forbidden_phrases:
        t = str(s).strip().lower()
        if t and t in lower:
            found.append(f"Contains forbidden closing phrase `{s}`")

    for s in bundle.anti_template_blocker.forbidden_patterns:
        t = str(s).strip().lower()
        if t and t in lower:
            label = s if len(s) <= 72 else s[:69] + "…"
            found.append(f"Anti-template phrase: {label!r}")

    for s in bundle.language_control.forbidden_words:
        if _term_matches(lower, str(s)):
            found.append(f"Forbidden vocabulary (language_control): `{s}`")

    if "[" in draft and any(x in draft for x in ("[Your", "[Client", "[Contact", "[Name]")):
        found.append("Bracket placeholders like [Your Name]")

    return found


def draft_passes_bundle_rules(draft: str, bundle: CoverLetterAssistantRulesBundle) -> bool:
    return len(scan_bundle_violations(draft, bundle)) == 0
