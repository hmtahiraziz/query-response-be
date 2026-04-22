"""Slack Events API + slash commands → same RAG cover-letter flow as POST /generate/cover-letter."""

from __future__ import annotations

import asyncio
import logging
import re

from fastapi import APIRouter, Request
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp

from app.core.config import get_settings
from app.schemas import CoverLetterResponse
from app.services.cover_letter_generation_flow import CoverLetterGenError, generate_cover_letter_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slack", tags=["slack"])

_MENTION_PREFIX = re.compile(r"^<@[A-Z0-9]+>\s*")


def _slash_command_path() -> str:
    s = get_settings()
    name = (s.slack_slash_command or "portfolio-brief").strip().lstrip("/")
    return f"/{name}" if name else "/portfolio-brief"


def _clean_mention_text(text: str) -> str:
    return _MENTION_PREFIX.sub("", text or "").strip()


def _is_im_message_event(event: dict) -> bool:
    if event.get("channel_type") == "im":
        return True
    ch = event.get("channel")
    return isinstance(ch, str) and ch.startswith("D")


def _format_sources_footer(resp: CoverLetterResponse) -> str:
    lines: list[str] = []
    for sn in resp.sources[:12]:
        label = (sn.project_name or sn.project_id).strip() or sn.project_id
        lines.append(f"• {label}")
    if not lines:
        return ""
    return "\n_Sources:_\n" + "\n".join(lines)


def _build_slack_body(resp: CoverLetterResponse) -> str:
    body = resp.cover_letter
    foot = _format_sources_footer(resp)
    if not foot:
        return body
    return f"{body}\n{foot}"


def _sync_generate(query: str) -> CoverLetterResponse:
    return generate_cover_letter_response(query, None, get_settings())


async def _generate_slack_body(query: str) -> str:
    if len(query) < 10:
        return "Your brief must be at least 10 characters (same rule as the web app API)."
    try:
        resp = await asyncio.to_thread(_sync_generate, query)
        body = _build_slack_body(resp)
        if len(body) > 39_000:
            body = body[:39_000] + "\n\n…(truncated for Slack)"
        return body
    except CoverLetterGenError as exc:
        return f"Error ({exc.status_code}): {exc.detail}"
    except Exception as exc:  # pragma: no cover
        logger.exception("Slack generation failed unexpectedly")
        return f"Unexpected error: {exc}"


def create_slack_bolt_app() -> AsyncApp:
    s = get_settings()
    app = AsyncApp(
        token=(s.slack_bot_token or "").strip(),
        signing_secret=(s.slack_signing_secret or "").strip(),
        process_before_response=False,
    )
    cmd = _slash_command_path()

    @app.command(cmd)
    async def slash_cover_letter(ack, client, command):
        await ack()
        channel_id = command.get("channel_id")
        if not channel_id:
            return
        text = (command.get("text") or "").strip()
        if not text:
            usage = (
                f"Usage: `{cmd}` followed by your client brief or job description.\n"
                f"Example: `{cmd} We need a senior engineer for a 6-month contract…`"
            )
            await client.chat_postMessage(channel=channel_id, text=usage)
            return
        msg = await _generate_slack_body(text)
        await client.chat_postMessage(channel=channel_id, text=msg)

    @app.event("app_mention")
    async def on_app_mention(event, say):
        text = _clean_mention_text(event.get("text", ""))
        if not text:
            await say("Paste your brief after mentioning me, or use the slash command.")
            return
        msg = await _generate_slack_body(text)
        await say(msg)

    @app.event("message")
    async def on_message(event, say):
        if not _is_im_message_event(event):
            return
        if event.get("subtype") is not None:
            return
        if event.get("bot_id"):
            return
        text = (event.get("text") or "").strip()
        if not text:
            return
        msg = await _generate_slack_body(text)
        await say(msg)

    return app


_slack_app = create_slack_bolt_app()
_slack_handler = AsyncSlackRequestHandler(_slack_app)


@router.post("/events")
async def slack_events(request: Request):
    return await _slack_handler.handle(request)
