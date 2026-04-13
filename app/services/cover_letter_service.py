"""Build grounded cover letter from retrieved portfolio chunks."""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)
from app.models import SourceSnippet
from app.services.gemini_retry import invoke_chat_with_retry
from app.services.gemini_service import get_chat_model
from app.services.pinecone_service import retrieve_context


def _extract_llm_text(msg: BaseMessage) -> str:
    """Normalize LangChain message content (str or multi-part blocks) to plain text."""
    raw = getattr(msg, "content", None)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(raw)


def format_context_blocks(docs: list[Document]) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        proj = meta.get("project_name") or meta.get("project_id") or "Project"
        page = meta.get("page")
        page_s = f" · page {page}" if page is not None else ""
        blocks.append(f"[Excerpt {i} | {proj}{page_s}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


def format_optional_rules_block(global_rules: str, chat_rules: str) -> str:
    """Injected into generate/refine prompts when the user has saved rules in the DB."""
    g = (global_rules or "").strip()
    c = (chat_rules or "").strip()
    if not g and not c:
        return ""
    parts = ["## User-configured rules (follow strictly)"]
    if g:
        parts.append("### Global rules\n" + g)
    if c:
        parts.append("### Chat rules\n" + c)
    return "\n\n".join(parts) + "\n\n"


COVER_LETTER_PROMPT_TEMPLATE = """You are writing a persuasive, client-ready response for someone applying to a role or answering a client’s brief. Your job is to **win trust and the next conversation**, not to sound like a compliance checklist.

## Evidence (portfolio)
- The PORTFOLIO CONTEXT below is **your evidence base** for **named projects, stacks, features, and outcomes**. Ground concrete claims in these excerpts (cite project names naturally, e.g. “On [Project], …”).
- **Do not invent** specific facts that would be easy to verify as false: fake employers, fake metrics, or **exact years of experience per technology** unless the excerpts state them. If years aren’t in the docs, speak to **depth and production scope** shown in the work, and offer to clarify timelines on a call—**do not make up numbers**.

## Persuasion vs. reference
- The portfolio is **supporting material**, not a cage. You **may** connect themes from the JD (Tech Lead, AI-first, Next.js, Supabase, etc.) to the strongest adjacent proof in the excerpts, and write in a **confident, warm, senior** voice.
- Where the client asks for **stories** (tight timelines, shortcuts, PR workflow) and the excerpts don’t spell out a personal anecdote, **do not** write long paragraphs saying “the context does not mention.” Instead: give **one short honest bridge** (e.g. happy to walk through specifics live) and **immediately** pivot to **related proof** from the portfolio (e.g. complex delivery, AI-native systems, production constraints) so the answer still sells.
- **Banned tone**: dry disclaimers like “the provided context primarily consists of specifications,” “I cannot answer,” or repeating “not in the provided context” for every bullet. **Preferred**: specific, upbeat, interview-winning prose.

## Structure & format
- If the CLIENT QUERY contains **multiple questions or a JD with bullets**, respond in a **clear structure** (short sections or numbered answers) so it’s easy to scan.
- Map **requirements from the query** to your evidence: “You’re looking for X; my work on [Project] shows Y.”
- Default length: **substantive**—long enough to feel complete (often several short sections), unless the query asks for brevity.
- No markdown code fences. Plain text; bullets allowed.
{optional_rules}CLIENT QUERY:
{query}

PORTFOLIO CONTEXT (ground claims about shipped work and stack here):
{context}

Write the response now."""

REFINE_COVER_LETTER_PROMPT_TEMPLATE = """You are revising a cover letter / client response draft based on the user’s feedback.

## Rules
- Output the **complete revised letter** as a single document (not a diff, not commentary before or after).
- Keep the same general purpose: answer the **ORIGINAL CLIENT BRIEF** below, in a persuasive, professional voice.
- Apply the **USER EDIT REQUEST** faithfully (tone, length, structure, emphasis, wording).
- **Do not invent** verifiable facts (employers, metrics, years of experience, project outcomes) that are not supported by the **PORTFOLIO CONTEXT** or already present in the **CURRENT DRAFT**. You may rephrase and tighten what is already there.
- If the user asks to add technical or outcome claims, ground them in the portfolio excerpts when possible.
- Preserve markdown-friendly structure if the draft uses headings or bullets unless the user asks to change format.
- No markdown code fences in your output.
{optional_rules}## Original client brief
{client_query}

## Current draft
{cover_letter}

{selection_block}

## Portfolio context (use for factual support when expanding or strengthening claims)
{context}

## User edit request
{instruction}

Write the full revised letter now."""


def generate_cover_letter(
    query: str,
    *,
    k: int | None = None,
    settings: Settings | None = None,
    global_rules: str = "",
    chat_rules: str = "",
) -> tuple[str, list[SourceSnippet]]:
    s = settings or get_settings()
    kk = k if k is not None else s.rag_k
    docs = retrieve_context(query, k=kk, settings=s)
    if not docs:
        raise ValueError(
            "No relevant portfolio context was retrieved. Ingest at least one project PDF first."
        )

    context = format_context_blocks(docs)
    optional_rules = format_optional_rules_block(global_rules, chat_rules)
    prompt = COVER_LETTER_PROMPT_TEMPLATE.format(
        query=query.strip(),
        context=context,
        optional_rules=optional_rules,
    )
    # Slightly higher temperature for persuasive, client-facing prose while excerpts anchor facts.
    llm = get_chat_model(s, temperature=0.45)
    msg = invoke_chat_with_retry(
        llm,
        prompt,
        max_retries=s.gemini_max_retries,
        retry_cap_seconds=s.gemini_retry_cap_seconds,
    )
    text = _extract_llm_text(msg).strip()
    if not text:
        logger.error(
            "Gemini returned empty content for generate_cover_letter (model=%s)",
            s.gemini_chat_model,
        )
        raise RuntimeError(
            "The language model returned an empty response. "
            "Verify GEMINI_API_KEY and GEMINI_CHAT_MODEL, check quotas, and try again."
        )

    sources: list[SourceSnippet] = []
    for d in docs:
        meta = d.metadata or {}
        preview = d.page_content.strip().replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + "…"
        sources.append(
            SourceSnippet(
                project_id=str(meta.get("project_id", "")),
                project_name=meta.get("project_name"),
                page=meta.get("page"),
                preview=preview,
            )
        )

    return text, sources


def refine_cover_letter(
    client_query: str,
    cover_letter: str,
    instruction: str,
    *,
    selection: str | None = None,
    k: int | None = None,
    settings: Settings | None = None,
    global_rules: str = "",
    chat_rules: str = "",
) -> tuple[str, list[SourceSnippet]]:
    s = settings or get_settings()
    kk = k if k is not None else s.rag_k
    retrieve_q = (
        f"{client_query.strip()}\n\nRevision goals:\n{instruction.strip()}"
    )
    docs = retrieve_context(retrieve_q, k=kk, settings=s)
    if not docs:
        raise ValueError(
            "No relevant portfolio context was retrieved. Ingest at least one project PDF first."
        )

    context = format_context_blocks(docs)
    if selection and selection.strip():
        selection_block = (
            "## Passage the user highlighted (prioritize revising this; keep the full letter coherent)\n"
            f"{selection.strip()}"
        )
    else:
        selection_block = ""

    optional_rules = format_optional_rules_block(global_rules, chat_rules)
    prompt = REFINE_COVER_LETTER_PROMPT_TEMPLATE.format(
        client_query=client_query.strip(),
        cover_letter=cover_letter.strip(),
        selection_block=selection_block,
        context=context,
        instruction=instruction.strip(),
        optional_rules=optional_rules,
    )
    llm = get_chat_model(s, temperature=0.35)
    msg = invoke_chat_with_retry(
        llm,
        prompt,
        max_retries=s.gemini_max_retries,
        retry_cap_seconds=s.gemini_retry_cap_seconds,
    )
    text = _extract_llm_text(msg).strip()
    if not text:
        logger.error(
            "Gemini returned empty content for refine_cover_letter (model=%s)",
            s.gemini_chat_model,
        )
        raise RuntimeError(
            "The language model returned an empty response. "
            "Verify GEMINI_API_KEY and GEMINI_CHAT_MODEL, check quotas, and try again."
        )

    sources: list[SourceSnippet] = []
    for d in docs:
        meta = d.metadata or {}
        preview = d.page_content.strip().replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + "…"
        sources.append(
            SourceSnippet(
                project_id=str(meta.get("project_id", "")),
                project_name=meta.get("project_name"),
                page=meta.get("page"),
                preview=preview,
            )
        )

    return text, sources
