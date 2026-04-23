"""Slack Events API + slash commands → same RAG cover-letter flow as POST /generate/cover-letter."""

from __future__ import annotations

import asyncio
import logging
import re

from fastapi import APIRouter, Request
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp

from app.core.config import get_settings
from app.schemas import CoverLetterResponse, RefineCoverLetterResponse, SourceSnippet
from app.services.cover_letter_generation_flow import CoverLetterGenError, generate_cover_letter_response
from app.services.cover_letter_refine_flow import RefineCoverLetterGenError, refine_cover_letter_response
from app.services.slack_thread_sessions import get_session, put_session, update_session_letter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slack", tags=["slack"])

_MENTION_PREFIX = re.compile(r"^<@[A-Z0-9]+>\s*")
# e.g. "refine: shorter", "refine make it warmer", "!refine: tone"
_REFINE_INSTRUCTION_RE = re.compile(
    r"^\s*!?\s*refine(?:\s*[:\s]+\s*|\s+)(.+)$",
    re.IGNORECASE | re.DOTALL,
)


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


def _is_channel_thread_refine_target(channel_id: str) -> bool:
    """Public / private channel (not DM). Refine-in-thread is channel-only for now."""
    return isinstance(channel_id, str) and len(channel_id) > 0 and channel_id[0] in ("C", "G")


def _format_sources_lines(sources: list[SourceSnippet]) -> str:
    lines: list[str] = []
    for sn in sources[:12]:
        label = (sn.project_name or sn.project_id).strip() or sn.project_id
        lines.append(f"• {label}")
    if not lines:
        return ""
    return "\n_Sources:_\n" + "\n".join(lines)


def _build_slack_body_from_generate(resp: CoverLetterResponse) -> str:
    body = resp.cover_letter
    foot = _format_sources_lines(resp.sources)
    if not foot:
        return body
    return f"{body}\n{foot}"


def _build_slack_body_from_refine(resp: RefineCoverLetterResponse) -> str:
    body = resp.cover_letter
    foot = _format_sources_lines(resp.sources)
    if not foot:
        return body
    return f"{body}\n{foot}"


def _truncate_slack(text: str) -> str:
    if len(text) > 39_000:
        return text[:39_000] + "\n\n…(truncated for Slack)"
    return text


def _sync_generate(query: str) -> CoverLetterResponse:
    return generate_cover_letter_response(query, None, get_settings())


async def _generate_slack_payload(query: str) -> tuple[str, CoverLetterResponse | None]:
    if len(query) < 10:
        return ("Your brief must be at least 10 characters (same rule as the web app API).", None)
    try:
        resp = await asyncio.to_thread(_sync_generate, query)
        body = _build_slack_body_from_generate(resp)
        return (_truncate_slack(body), resp)
    except CoverLetterGenError as exc:
        return (f"Error ({exc.status_code}): {exc.detail}", None)
    except Exception as exc:  # pragma: no cover
        logger.exception("Slack generation failed unexpectedly")
        return (f"Unexpected error: {exc}", None)


def _parse_refine_instruction(text: str) -> str | None:
    m = _REFINE_INSTRUCTION_RE.match(text.strip())
    if not m:
        return None
    instruction = m.group(1).strip()
    if len(instruction) < 3:
        return None
    return instruction


async def _post_in_thread_returning_root(
    client,
    channel_id: str,
    text: str,
    *,
    thread_ts: str | None,
) -> str | None:
    """Post letter in thread; return thread *root* ts for session lookup (refine replies use this)."""
    if thread_ts:
        await client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=text)
        return thread_ts
    root = await client.chat_postMessage(
        channel=channel_id,
        text=":incoming_envelope: *Cover letter* — see thread.",
    )
    ts = root.get("ts")
    if ts:
        await client.chat_postMessage(channel=channel_id, thread_ts=ts, text=text)
        return str(ts)
    await client.chat_postMessage(channel=channel_id, text=text)
    return None


async def _handle_slack_refine(
    say,
    channel_id: str,
    thread_root_ts: str,
    instruction: str,
) -> None:
    session = get_session(channel_id, thread_root_ts)
    if session is None:
        await say(
            ":shrug: No saved draft for this thread. Generate a letter in this channel first "
            "(mention me with a brief or use the slash command), then reply with "
            "`refine: …` or `refine …` in the thread.",
            thread_ts=thread_root_ts,
        )
        return

    def _run() -> RefineCoverLetterResponse:
        return refine_cover_letter_response(
            session.client_query,
            session.cover_letter,
            instruction,
            selection=None,
            k=None,
            settings=get_settings(),
        )

    try:
        resp = await asyncio.to_thread(_run)
    except RefineCoverLetterGenError as exc:
        await say(f"Error ({exc.status_code}): {exc.detail}", thread_ts=thread_root_ts)
        return
    except Exception as exc:  # pragma: no cover
        logger.exception("Slack refine failed unexpectedly")
        await say(f"Unexpected error: {exc}", thread_ts=thread_root_ts)
        return

    update_session_letter(channel_id, thread_root_ts, resp.cover_letter)
    msg = _truncate_slack(_build_slack_body_from_refine(resp))
    await say(msg, thread_ts=thread_root_ts)


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
        msg, resp = await _generate_slack_payload(text)
        existing_thread = (command.get("thread_ts") or "").strip() or None
        root_ts = await _post_in_thread_returning_root(
            client, channel_id, msg, thread_ts=existing_thread
        )
        if (
            resp is not None
            and root_ts
            and _is_channel_thread_refine_target(channel_id)
        ):
            put_session(channel_id, root_ts, text, resp.cover_letter)

    @app.event("app_mention")
    async def on_app_mention(event, say):
        channel_id = event.get("channel") or ""
        thread_ts = event.get("ts")
        parent_thread = event.get("thread_ts")
        text = _clean_mention_text(event.get("text", ""))

        refine_ins = _parse_refine_instruction(text) if parent_thread else None
        if parent_thread and refine_ins:
            if _is_channel_thread_refine_target(channel_id):
                await _handle_slack_refine(say, channel_id, parent_thread, refine_ins)
            return

        if not text:
            await say(
                "Paste your brief after mentioning me, or use the slash command.",
                thread_ts=parent_thread or thread_ts,
            )
            return

        msg, resp = await _generate_slack_payload(text)
        reply_thread_ts = parent_thread or thread_ts
        await say(msg, thread_ts=reply_thread_ts)
        if resp is not None and _is_channel_thread_refine_target(channel_id):
            session_root_ts = parent_thread or thread_ts
            put_session(channel_id, session_root_ts, text, resp.cover_letter)

    @app.event("message")
    async def on_message(event, say):
        if event.get("subtype") is not None:
            return
        if event.get("bot_id"):
            return

        channel_id = event.get("channel") or ""
        text = (event.get("text") or "").strip()
        if not text:
            return

        thread_root = event.get("thread_ts")
        refine_ins = _parse_refine_instruction(text) if thread_root else None
        if (
            thread_root
            and _is_channel_thread_refine_target(channel_id)
            and refine_ins
        ):
            await _handle_slack_refine(say, channel_id, str(thread_root), refine_ins)
            return

        if not _is_im_message_event(event):
            return

        thread_ts = event.get("ts")
        msg, _resp = await _generate_slack_payload(text)
        await say(msg, thread_ts=thread_ts)

    return app


_slack_app = create_slack_bolt_app()
_slack_handler = AsyncSlackRequestHandler(_slack_app)


@router.post("/events")
async def slack_events(request: Request):
    return await _slack_handler.handle(request)
