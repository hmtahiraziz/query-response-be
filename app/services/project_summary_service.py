"""Generate and format structured project summaries from project PDF text."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import Settings, get_settings
from app.schemas import ProjectAIProfile
from app.services.llm_retry import run_with_retry
from app.services.openai_service import get_chat_model

_URL_RE = re.compile(r"https?://[^\s)>\]\"']+", re.IGNORECASE)


def _page_num_and_text(page: object, fallback_index: int) -> tuple[int, str]:
    """Normalize extractor output: supports raw text or (page_num, text) tuples."""
    if isinstance(page, tuple) and len(page) >= 2:
        raw_num, raw_text = page[0], page[1]
        try:
            page_num = int(raw_num)
        except Exception:
            page_num = fallback_index
        return page_num, str(raw_text or "")
    return fallback_index, str(page or "")


def _extract_live_link(pages: list[object]) -> str | None:
    for idx, page in enumerate(pages, start=1):
        _, text = _page_num_and_text(page, idx)
        for raw in _URL_RE.findall(text):
            url = raw.rstrip(".,;:!?")
            lower = url.lower()
            if lower.endswith(".pdf"):
                continue
            if "linkedin.com" in lower:
                continue
            return url
    return None


def _trim_pages_for_prompt(pages: list[object], char_limit: int = 24_000) -> str:
    parts: list[str] = []
    used = 0
    for idx, page in enumerate(pages, start=1):
        page_num, txt = _page_num_and_text(page, idx)
        txt = txt.strip()
        if not txt:
            continue
        chunk = f"[Page {page_num}]\n{txt}\n"
        if used + len(chunk) > char_limit:
            remaining = char_limit - used
            if remaining <= 300:
                break
            chunk = chunk[:remaining]
        parts.append(chunk)
        used += len(chunk)
        if used >= char_limit:
            break
    return "\n".join(parts)


def generate_project_summary(
    *,
    project_name: str,
    pages: list[object],
    settings: Settings | None = None,
) -> ProjectAIProfile:
    s = settings or get_settings()
    content = _trim_pages_for_prompt(pages)
    if not content.strip():
        raise ValueError("Cannot summarize an empty project document.")

    detected_link = _extract_live_link(pages)

    system_msg = SystemMessage(
        content=(
            "You extract a concise but technically rich project summary from project documentation.\n"
            "Return JSON matching this schema exactly:\n"
            "{\n"
            '  "name": string,\n'
            '  "type": string[],\n'
            '  "problem": string,\n'
            '  "solution": string,\n'
            '  "project_brief": string,\n'
            '  "technical_depth": string,\n'
            '  "stack": string[],\n'
            '  "impact": string,\n'
            '  "talking_points": string[],\n'
            '  "live_link": string | null\n'
            "}\n"
            "Rules:\n"
            "- Prefer short, direct phrases.\n"
            "- Use only information grounded in the document text.\n"
            "- project_brief should be about 180-350 words when enough source detail exists.\n"
            "- technical_depth should cover architecture, key components, data flow, and notable implementation choices.\n"
            "- If unknown, keep string fields as empty strings and arrays empty.\n"
            "- live_link should be the project's public URL if present; else null.\n"
        )
    )
    human_msg = HumanMessage(
        content=(
            f"Project name hint: {project_name}\n"
            f"Detected candidate live link: {detected_link or 'none'}\n\n"
            "Project document text:\n"
            f"{content}\n\n"
            "Return the JSON object now."
        )
    )

    llm = get_chat_model(s, temperature=0.1).with_structured_output(ProjectAIProfile)

    def _invoke() -> ProjectAIProfile:
        out = llm.invoke([system_msg, human_msg])
        if not isinstance(out, ProjectAIProfile):
            raise TypeError(f"Expected ProjectAIProfile, got {type(out)}")
        if not out.live_link and detected_link:
            out.live_link = detected_link
        return out

    return run_with_retry(
        _invoke,
        max_retries=s.openai_max_retries,
        retry_cap_seconds=s.openai_retry_cap_seconds,
        operation="openai_project_summary_generate",
    )


def format_project_summaries_context(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    blocks: list[str] = []
    for idx, row in enumerate(items, start=1):
        summary = row.get("ai_summary") or {}
        name = str(summary.get("name") or row.get("name") or "Project")
        ptype = summary.get("type") if isinstance(summary.get("type"), list) else []
        stack = summary.get("stack") if isinstance(summary.get("stack"), list) else []
        talking = summary.get("talking_points") if isinstance(summary.get("talking_points"), list) else []
        lines = [
            f"[Excerpt {idx} | {name}]",
            f"Type: {', '.join(str(x) for x in ptype) if ptype else 'n/a'}",
            f"Problem: {summary.get('problem', '')}",
            f"Solution: {summary.get('solution', '')}",
            f"Project brief: {summary.get('project_brief', '')}",
            f"Technical depth: {summary.get('technical_depth', '')}",
            f"Stack: {', '.join(str(x) for x in stack) if stack else 'n/a'}",
            f"Impact: {summary.get('impact', '')}",
            f"Talking points: {', '.join(str(x) for x in talking) if talking else 'n/a'}",
            f"Live link: {summary.get('live_link') or 'n/a'}",
        ]
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)
