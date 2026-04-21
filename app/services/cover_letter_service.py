"""Build grounded cover letter from retrieved portfolio chunks."""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)
from app.schemas import AssistantPolicy, CoverLetterStructuredOutput, SourceSnippet
from app.schemas.assistant import CoverLetterAssistantRulesBundle
from app.services.bundled_rules_prompt import format_bundled_assistant_rules_for_prompt
from app.services.code_assistant_rules import get_cover_letter_rules_bundle
from app.services.cover_letter_compliance import scan_bundle_violations
from app.services.llm_retry import run_with_retry
from app.services.manifest_service import list_projects
from app.services.openai_service import get_chat_model
from app.services.pinecone_service import retrieve_context
from app.services.project_summary_service import format_project_summaries_context


def format_context_blocks(docs: list[Document]) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        proj = meta.get("project_name") or meta.get("project_id") or "Project"
        page = meta.get("page")
        page_s = f" · page {page}" if page is not None else ""
        blocks.append(f"[Excerpt {i} | {proj}{page_s}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


def format_structured_policy_block(policy: AssistantPolicy) -> str:
    """Machine-readable policy block for the system message."""
    lines = ["## Structured policy (apply strictly; reflect accurately in self_check)"]
    lines.append(
        f"- **language**: `{policy.language}` — when `uk` or `us`, use spelling and conventions consistent with that locale; `none` means no extra locale constraint from this policy."
    )
    if policy.max_words is not None:
        lines.append(f"- **max_words**: {policy.max_words} — draft_text must not exceed this many words.")
    else:
        lines.append("- **max_words**: not set (no numeric cap from this policy).")
    if policy.must_include:
        lines.append("- **must_include** (each line must appear in draft_text, verbatim or clear paraphrase):")
        for i, m in enumerate(policy.must_include, 1):
            lines.append(f"  {i}. {m}")
    else:
        lines.append("- **must_include**: (none)")
    if policy.must_not_include:
        lines.append("- **must_not_include** (none of these substrings may appear in draft_text):")
        for i, m in enumerate(policy.must_not_include, 1):
            lines.append(f"  {i}. {m}")
    else:
        lines.append("- **must_not_include**: (none)")
    return "\n".join(lines)


def _has_structured_constraints(policy: AssistantPolicy) -> bool:
    return (
        policy.language != "none"
        or policy.max_words is not None
        or bool(policy.must_include)
        or bool(policy.must_not_include)
    )


RULES_PRECEDENCE = """## Precedence (read carefully)
- **Facts about past work** (employers, products, metrics, timelines, tech stacks as deployed fact): use only what is supported by the **Portfolio context** in the user message and the client query. Do not invent verifiable facts.
- **Bundled assistant rules** (the large mandatory section above): control greetings, closings, bullets/markdown, placeholder text, how many projects to cite, anti-template phrases, and overall letter shape. They **override** generic “Structure & format” wording earlier in this system message when the two conflict.
- **Structured policy** (language, max_words, must_include, must_not_include): applies when those fields are set; bundled rules + portfolio grounding still apply.
"""


STRUCTURED_OUTPUT_INSTRUCTIONS = """## Structured output (required)
Respond using the configured structured format (not raw prose in the top-level message):
- **draft_text**: The complete client-facing letter only (plain text; no markdown code fences).
- **claims**: Substantive factual statements about past work taken from the portfolio; each item includes **excerpt_index** equal to the `[Excerpt i | …]` index `i` in the portfolio context (1-based).
- **self_check**: Booleans **global_rules_addressed** (you followed **Bundled assistant rules** — openings, closings, project count, bullets/markdown, anti-template phrases, persona), **chat_rules_addressed** (claims grounded in excerpts; no invented verifiable facts), **structured_policy_met**, **within_max_words**, and optional **notes** — set them honestly relative to your draft and the policies above.
"""


def _generate_compliance_checklist(has_code_rules: bool, policy: AssistantPolicy) -> str:
    lines = [
        "## Before you output",
        "- Populate the structured fields; **draft_text** must read as a natural client-facing letter with no preamble (conversational > polished).",
        "- No markdown code fences in draft_text.",
        "- If **Bundled assistant rules** forbid bullets or markdown, **draft_text** must be continuous prose only.",
    ]
    if has_code_rules:
        lines.append(
            "- **Re-check** every rule under **Bundled assistant rules** (forbidden openings/closings/patterns, max projects, formatting, placeholders)."
        )
    if _has_structured_constraints(policy):
        lines.append(
            "- **Re-check** structured policy (language, max_words, must_include, must_not_include) and set self_check accordingly."
        )
    lines.append(
        "- Ensure factual claims about shipped work remain grounded in the portfolio excerpts in the user message."
    )
    return "\n".join(lines)


COVER_LETTER_SYSTEM_BASE = """You are writing a persuasive, client-ready response for someone applying to a role or answering a client’s brief. Your job is to **win trust and the next conversation**, not to sound like a compliance checklist.

## Evidence (portfolio)
- The **Portfolio context** in the user message is **your evidence base** for **named projects, stacks, features, and outcomes**. Ground concrete claims in those excerpts (cite project names naturally, e.g. “On [Project], …”).
- **Do not invent** specific facts that would be easy to verify as false: fake employers, fake metrics, or **exact years of experience per technology** unless the excerpts state them. If years aren’t in the docs, speak to **depth and production scope** shown in the work, and offer to clarify timelines on a call—**do not make up numbers**.

## Persuasion vs. reference
- The portfolio is **supporting material**, not a cage. You **may** connect themes from the JD (Tech Lead, AI-first, Next.js, Supabase, etc.) to the strongest adjacent proof in the excerpts, and write in a **confident, warm, senior** voice.
- Where the client asks for **stories** (tight timelines, shortcuts, PR workflow) and the excerpts don’t spell out a personal anecdote, **do not** write long paragraphs saying “the context does not mention.” Instead: give **one short honest bridge** (e.g. happy to walk through specifics live) and **immediately** pivot to **related proof** from the portfolio (e.g. complex delivery, AI-native systems, production constraints) so the answer still sells.
- **Banned tone**: dry disclaimers like “the provided context primarily consists of specifications,” “I cannot answer,” or repeating “not in the provided context” for every bullet. **Preferred**: direct, conversational prose that sounds like a strong freelancer typing to a client — **not** corporate marketing, not “sales polish,” not stock phrases like “aligns perfectly,” “robust solution,” or “seamlessly integrate.”

## Structure & format (defaults only — see **Bundled assistant rules** below)
- **Bundled assistant rules** control bullets, markdown, letter openings/closings, and how many projects to cite. When they forbid list formatting, use short paragraphs instead of bullets even if the client asked for many requirements.
- If the client query has **multiple distinct questions**, you may still answer clearly using short paragraphs (no bullets if the bundle forbids them).
- Map **requirements from the query** to your evidence in flowing prose.
- Default length: **substantive** unless **Structured policy** `max_words` or the query demands brevity.
- No markdown code fences in **draft_text**."""


REFINE_SYSTEM_BASE = """You are revising a cover letter / client response draft based on the user’s feedback in the user message.

## Revision rules
- Output the **complete revised letter** as a single document (not a diff, not commentary before or after).
- Keep the same general purpose: answer the **original client brief** in the user message, in a persuasive, professional voice.
- Apply the **user edit request** faithfully (tone, length, structure, emphasis, wording).
- **Bundled assistant rules** (below) override generic defaults: if they forbid bullets, markdown, or formal greetings/closings, the revised **draft_text** must comply even if the current draft used those patterns.
- **Do not invent** verifiable facts (employers, metrics, years of experience, project outcomes) that are not supported by the **portfolio context** or already present in the **current draft**. You may rephrase and tighten what is already there.
- If the user asks to add technical or outcome claims, ground them in the portfolio excerpts when possible.
- No markdown code fences in your output."""


def _build_generate_messages(
    query: str,
    context: str,
    code_rules_markdown: str,
    policy: AssistantPolicy,
    n_excerpts: int,
) -> list[BaseMessage]:
    has_code_rules = bool(code_rules_markdown.strip())
    parts = [COVER_LETTER_SYSTEM_BASE.strip(), format_structured_policy_block(policy)]
    if has_code_rules:
        parts.append(code_rules_markdown.strip())
    parts.append(STRUCTURED_OUTPUT_INSTRUCTIONS.strip())
    parts.append(RULES_PRECEDENCE.strip())
    parts.append(_generate_compliance_checklist(has_code_rules, policy))
    system_content = "\n\n".join(parts)
    human_content = (
        "## Client query\n"
        f"{query.strip()}\n\n"
        "## Portfolio context (ground claims about shipped work and stack here)\n"
        f"{context}\n\n"
        f"There are **{n_excerpts}** excerpts above, numbered 1..{n_excerpts} in `[Excerpt i | …]` headers. "
        "Use **excerpt_index** = i for each portfolio-backed claim.\n\n"
        "Produce the structured response now."
    )
    return [SystemMessage(content=system_content), HumanMessage(content=human_content)]


def _build_refine_messages(
    client_query: str,
    cover_letter: str,
    instruction: str,
    selection_block: str,
    context: str,
    code_rules_markdown: str,
    policy: AssistantPolicy,
    n_excerpts: int,
) -> list[BaseMessage]:
    has_code_rules = bool(code_rules_markdown.strip())
    parts = [REFINE_SYSTEM_BASE.strip(), format_structured_policy_block(policy)]
    if has_code_rules:
        parts.append(code_rules_markdown.strip())
    parts.append(STRUCTURED_OUTPUT_INSTRUCTIONS.strip())
    parts.append(RULES_PRECEDENCE.strip())
    parts.append(_generate_compliance_checklist(has_code_rules, policy))
    system_content = "\n\n".join(parts)
    human_parts = [
        "## Original client brief\n" + client_query.strip(),
        "## Current draft\n" + cover_letter.strip(),
    ]
    if selection_block.strip():
        human_parts.append(selection_block.strip())
    human_parts.extend(
        [
            "## Portfolio context (use for factual support when expanding or strengthening claims)\n" + context,
            "## User edit request\n" + instruction.strip(),
            f"There are **{n_excerpts}** excerpts above, numbered 1..{n_excerpts} in `[Excerpt i | …]` headers. "
            "Use **excerpt_index** = i for each portfolio-backed claim.\n\n"
            "Produce the structured revised letter now.",
        ]
    )
    human_content = "\n\n".join(human_parts)
    return [SystemMessage(content=system_content), HumanMessage(content=human_content)]


COMPLIANCE_REPAIR_ADDENDUM = """## Compliance repair pass (mandatory)
A previous draft failed automated rule checks. Your replacement **draft_text** must:
- Fix **every** issue listed under “Detected issues” in the user message.
- Keep factual claims grounded in the same portfolio excerpts; do not invent new verifiable facts.
- Prefer opening like you are answering **their** situation first (e.g. mirroring the client’s constraint in plain language) before the project name — avoid stiff phrases like “Your need for a … aligns with my experience.”
- Stay in the voice required by **Bundled assistant rules** (direct, human, not corporate brochure copy).
"""


def _build_compliance_repair_messages(
    query: str,
    context: str,
    violations: list[str],
    failed_draft: str,
    code_rules_markdown: str,
    policy: AssistantPolicy,
    n_excerpts: int,
) -> list[BaseMessage]:
    """Second LLM call after deterministic scan finds bundled-rule violations."""
    parts = [
        COVER_LETTER_SYSTEM_BASE.strip(),
        COMPLIANCE_REPAIR_ADDENDUM.strip(),
        format_structured_policy_block(policy),
        code_rules_markdown.strip(),
        STRUCTURED_OUTPUT_INSTRUCTIONS.strip(),
        RULES_PRECEDENCE.strip(),
        _generate_compliance_checklist(True, policy),
    ]
    system_content = "\n\n".join(parts)
    issues_block = "\n".join(f"- {v}" for v in violations) if violations else "- (unspecified — still rewrite strictly per bundled rules)"
    human_content = (
        "## Client query\n"
        f"{query.strip()}\n\n"
        "## Portfolio context (ground claims about shipped work and stack here)\n"
        f"{context}\n\n"
        "## Previous draft (failed automated checks — rewrite completely, do not lightly edit)\n"
        f"{failed_draft.strip()}\n\n"
        "## Detected issues (you must eliminate all of these in the new draft)\n"
        f"{issues_block}\n\n"
        f"There are **{n_excerpts}** excerpts above, numbered 1..{n_excerpts} in `[Excerpt i | …]` headers. "
        "Use **excerpt_index** = i for each portfolio-backed claim.\n\n"
        "Produce the structured response now with a fully compliant **draft_text**."
    )
    return [SystemMessage(content=system_content), HumanMessage(content=human_content)]


def _warn_policy_violations(draft: str, policy: AssistantPolicy) -> None:
    """Best-effort logging when draft_text misses structured checks (no auto-rewrite yet)."""
    if not draft:
        return
    lower = draft.lower()
    for phrase in policy.must_not_include:
        if phrase.lower() in lower:
            logger.warning("Structured policy: must_not_include violated (%r)", phrase)
    if policy.max_words is not None:
        wc = len(re.split(r"\s+", draft.strip()))
        if wc > policy.max_words:
            logger.warning(
                "Structured policy: within_max_words violated (word count %s > max_words %s)",
                wc,
                policy.max_words,
            )
    for phrase in policy.must_include:
        if phrase.lower() not in lower:
            logger.warning("Structured policy: must_include missing (%r)", phrase)


def _warn_bundle_violations(draft: str, bundle: CoverLetterAssistantRulesBundle) -> None:
    """Log likely violations of bundled JSON rules (best-effort; does not block response)."""
    for v in scan_bundle_violations(draft, bundle):
        logger.warning("Bundled rules: %s", v)


def _maybe_repair_bundle_compliance(
    draft: str,
    *,
    query: str,
    context: str,
    code_rules_md: str,
    policy: AssistantPolicy,
    bundle: CoverLetterAssistantRulesBundle,
    n_excerpts: int,
    s: Settings,
    operation: str,
) -> str:
    """If deterministic scan finds violations, run one structured rewrite pass at low temperature."""
    violations = scan_bundle_violations(draft, bundle)
    if not violations:
        return draft
    logger.warning("%s: running bundle compliance repair (%s)", operation, violations)
    repair_messages = _build_compliance_repair_messages(
        query, context, violations, draft, code_rules_md, policy, n_excerpts
    )
    llm_r = get_chat_model(s, temperature=0.18).with_structured_output(CoverLetterStructuredOutput)

    def _repair() -> CoverLetterStructuredOutput:
        out = llm_r.invoke(repair_messages)
        if not isinstance(out, CoverLetterStructuredOutput):
            raise TypeError(f"Expected CoverLetterStructuredOutput, got {type(out)}")
        return out

    repaired = run_with_retry(
        _repair,
        max_retries=s.openai_max_retries,
        retry_cap_seconds=s.openai_retry_cap_seconds,
        operation=f"{operation}_bundle_repair",
    )
    out_text = (repaired.draft_text or "").strip()
    if not out_text:
        logger.error("%s: compliance repair returned empty draft; keeping first draft", operation)
        return draft
    still = scan_bundle_violations(out_text, bundle)
    if still:
        logger.warning("%s: draft still shows violations after repair: %s", operation, still)
    return out_text


def generate_cover_letter(
    query: str,
    *,
    k: int | None = None,
    settings: Settings | None = None,
) -> tuple[str, list[SourceSnippet]]:
    s = settings or get_settings()
    bundle = get_cover_letter_rules_bundle()
    pol = bundle.policy
    code_rules_md = format_bundled_assistant_rules_for_prompt(bundle)
    kk = k if k is not None else s.rag_k
    project_rows = list_projects(openai_only=True)
    summary_rows = [
        row for row in project_rows.values() if isinstance(row, dict) and isinstance(row.get("ai_summary"), dict)
    ]
    docs: list[Document] = []
    if summary_rows:
        context = format_project_summaries_context(summary_rows)
        n_excerpts = len(summary_rows)
    else:
        docs = retrieve_context(query, k=kk, settings=s)
        if not docs:
            raise ValueError(
                "No project summaries or relevant portfolio chunks were found. Ingest at least one project PDF first."
            )
        context = format_context_blocks(docs)
        n_excerpts = len(docs)
    messages = _build_generate_messages(query, context, code_rules_md, pol, n_excerpts)
    llm = get_chat_model(s, temperature=0.32).with_structured_output(CoverLetterStructuredOutput)

    def _invoke() -> CoverLetterStructuredOutput:
        out = llm.invoke(messages)
        if not isinstance(out, CoverLetterStructuredOutput):
            raise TypeError(f"Expected CoverLetterStructuredOutput, got {type(out)}")
        return out

    parsed = run_with_retry(
        _invoke,
        max_retries=s.openai_max_retries,
        retry_cap_seconds=s.openai_retry_cap_seconds,
        operation="openai_cover_generate_structured",
    )
    text = (parsed.draft_text or "").strip()
    if not text:
        logger.error(
            "OpenAI returned empty draft_text for generate_cover_letter (model=%s)",
            s.openai_chat_model,
        )
        raise RuntimeError(
            "The language model returned an empty response. "
            "Verify OPENAI_API_KEY and OPENAI_CHAT_MODEL, check quotas, and try again."
        )
    text = _maybe_repair_bundle_compliance(
        text,
        query=query,
        context=context,
        code_rules_md=code_rules_md,
        policy=pol,
        bundle=bundle,
        n_excerpts=n_excerpts,
        s=s,
        operation="generate_cover_letter",
    )
    _warn_policy_violations(text, pol)
    _warn_bundle_violations(text, bundle)

    sources: list[SourceSnippet] = []
    if summary_rows:
        for row in summary_rows:
            summary = row.get("ai_summary") or {}
            preview = str(
                summary.get("project_brief")
                or summary.get("technical_depth")
                or summary.get("solution")
                or summary.get("problem")
                or row.get("name")
                or ""
            ).strip()
            if len(preview) > 220:
                preview = preview[:220] + "…"
            sources.append(
                SourceSnippet(
                    project_id=str(row.get("project_id", "")),
                    project_name=row.get("name"),
                    page=None,
                    preview=preview,
                )
            )
    else:
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
) -> tuple[str, list[SourceSnippet]]:
    s = settings or get_settings()
    bundle = get_cover_letter_rules_bundle()
    pol = bundle.policy
    code_rules_md = format_bundled_assistant_rules_for_prompt(bundle)
    kk = k if k is not None else s.rag_k
    project_rows = list_projects(openai_only=True)
    summary_rows = [
        row for row in project_rows.values() if isinstance(row, dict) and isinstance(row.get("ai_summary"), dict)
    ]
    docs: list[Document] = []
    if summary_rows:
        context = format_project_summaries_context(summary_rows)
        n_excerpts = len(summary_rows)
    else:
        retrieve_q = (
            f"{client_query.strip()}\n\nRevision goals:\n{instruction.strip()}"
        )
        docs = retrieve_context(retrieve_q, k=kk, settings=s)
        if not docs:
            raise ValueError(
                "No project summaries or relevant portfolio chunks were found. Ingest at least one project PDF first."
            )
        context = format_context_blocks(docs)
        n_excerpts = len(docs)
    if selection and selection.strip():
        selection_block = (
            "## Passage the user highlighted (prioritize revising this; keep the full letter coherent)\n"
            f"{selection.strip()}"
        )
    else:
        selection_block = ""

    messages = _build_refine_messages(
        client_query.strip(),
        cover_letter.strip(),
        instruction.strip(),
        selection_block,
        context,
        code_rules_md,
        pol,
        n_excerpts,
    )
    llm = get_chat_model(s, temperature=0.26).with_structured_output(CoverLetterStructuredOutput)

    def _invoke() -> CoverLetterStructuredOutput:
        out = llm.invoke(messages)
        if not isinstance(out, CoverLetterStructuredOutput):
            raise TypeError(f"Expected CoverLetterStructuredOutput, got {type(out)}")
        return out

    parsed = run_with_retry(
        _invoke,
        max_retries=s.openai_max_retries,
        retry_cap_seconds=s.openai_retry_cap_seconds,
        operation="openai_cover_refine_structured",
    )
    text = (parsed.draft_text or "").strip()
    if not text:
        logger.error(
            "OpenAI returned empty draft_text for refine_cover_letter (model=%s)",
            s.openai_chat_model,
        )
        raise RuntimeError(
            "The language model returned an empty response. "
            "Verify OPENAI_API_KEY and OPENAI_CHAT_MODEL, check quotas, and try again."
        )
    text = _maybe_repair_bundle_compliance(
        text,
        query=client_query,
        context=context,
        code_rules_md=code_rules_md,
        policy=pol,
        bundle=bundle,
        n_excerpts=n_excerpts,
        s=s,
        operation="refine_cover_letter",
    )
    _warn_policy_violations(text, pol)
    _warn_bundle_violations(text, bundle)

    sources: list[SourceSnippet] = []
    if summary_rows:
        for row in summary_rows:
            summary = row.get("ai_summary") or {}
            preview = str(
                summary.get("project_brief")
                or summary.get("technical_depth")
                or summary.get("solution")
                or summary.get("problem")
                or row.get("name")
                or ""
            ).strip()
            if len(preview) > 220:
                preview = preview[:220] + "…"
            sources.append(
                SourceSnippet(
                    project_id=str(row.get("project_id", "")),
                    project_name=row.get("name"),
                    page=None,
                    preview=preview,
                )
            )
    else:
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
