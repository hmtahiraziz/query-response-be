"""Render ``CoverLetterAssistantRulesBundle`` into system-prompt markdown for the LLM."""

from __future__ import annotations

from app.schemas.assistant import CoverLetterAssistantRulesBundle, GenerationCodeRules


def format_legacy_generation_lists(gen: GenerationCodeRules) -> str:
    """Optional extra bullets under ``generation`` in JSON (tone/composition/grounding/formatting lists)."""

    def _section(title: str, bullets: list[str]) -> str | None:
        lines = [f"### {title}"]
        for b in bullets:
            t = str(b).strip()
            if t:
                lines.append(f"- {t}")
        return "\n".join(lines) if len(lines) > 1 else None

    chunks: list[str] = []
    for title, bullets in (
        ("Tone and voice (extra list)", gen.tone_and_voice),
        ("Composition (extra list)", gen.composition),
        ("Factual grounding (extra list)", gen.factual_grounding),
        ("Formatting (extra list)", gen.formatting),
    ):
        block = _section(title, bullets)
        if block:
            chunks.append(block)
    if not chunks:
        return ""
    return "## Additional bullet lists from JSON `generation` field\n\n" + "\n\n".join(chunks)


def format_bundled_assistant_rules_for_prompt(bundle: CoverLetterAssistantRulesBundle) -> str:
    """Full bundled rules as markdown; must be included in the system message for strict compliance."""
    parts: list[str] = [
        "## Bundled assistant rules (mandatory — highest precedence)",
        "",
        "Violating any **hard constraint** or **forbidden** pattern below is unacceptable. "
        "If your first attempt would break a rule, rewrite before returning structured output.",
        "",
        f"- **Bundle version**: {bundle.version}",
        "",
    ]

    p = bundle.persona
    if p.role or p.voice.tense or p.voice.tone:
        parts.append("### Persona")
        if p.role:
            parts.append(f"- **Role**: {p.role}")
        if p.voice.tense or p.voice.tone:
            parts.append(
                f"- **Voice**: tense = `{p.voice.tense or '—'}` · tone = `{p.voice.tone or '—'}`"
            )
        parts.append("")

    hc = bundle.hard_constraints
    if hc.opening.rule or hc.opening.forbidden_starts:
        parts.append("### Hard constraints — opening")
        if hc.opening.rule:
            parts.append(f"- **Rule**: {hc.opening.rule}")
        if hc.opening.forbidden_starts:
            parts.append(
                "- **Forbidden starts** (the first characters of **draft_text** must NOT match or paraphrase these openings; "
                "includes “Dear …”, “Thank you for …”, “I am excited …”, etc.):"
            )
            for s in hc.opening.forbidden_starts:
                parts.append(f"  - `{s}`")
        parts.append("")

    if hc.closing.rule or hc.closing.forbidden_phrases:
        parts.append("### Hard constraints — closing")
        if hc.closing.rule:
            parts.append(f"- **Rule**: {hc.closing.rule}")
        if hc.closing.forbidden_phrases:
            parts.append(
                "- **Forbidden closing phrases** (must not appear anywhere in **draft_text**):"
            )
            for s in hc.closing.forbidden_phrases:
                parts.append(f"  - `{s}`")
        parts.append("")

    parts.append("### Hard constraints — project usage")
    if hc.project_usage.rule:
        parts.append(f"- **Rule**: {hc.project_usage.rule}")
    parts.append(f"- **Max distinct projects to cite as proof**: {hc.project_usage.max_projects}")
    parts.append("")

    parts.append("### Hard constraints — formatting")
    fmt = hc.formatting
    parts.append(f"- **Plain text only**: {fmt.plain_text_only}")
    parts.append(f"- **Markdown** (headings, bold, lists): **{'allowed' if fmt.allow_markdown else 'FORBIDDEN'}**")
    parts.append(f"- **Bullets / numbered lists**: **{'allowed' if fmt.allow_bullets else 'FORBIDDEN'}**")
    if not fmt.allow_bullets:
        parts.append(
            "- Use **continuous prose paragraphs** only; do not use `-`, `*`, `1.` list markers or line-break “fake bullets”."
        )
    parts.append("")

    rf = bundle.response_flow
    if rf.strict_order or rf.definitions:
        parts.append("### Response flow (strict order)")
        if rf.strict_order:
            parts.append(
                "- **Follow this order** in **draft_text** (use short paragraphs; no section headings unless bundled rules allow markdown):"
            )
            for i, step in enumerate(rf.strict_order, start=1):
                parts.append(f"  {i}. `{step}`")
        if rf.definitions:
            parts.append("- **Step definitions**:")
            for key, desc in rf.definitions.items():
                parts.append(f"  - **`{key}`**: {desc}")
        parts.append("")

    at = bundle.anti_template_blocker
    if at.rule or at.forbidden_patterns:
        parts.append("### Anti-template blocker")
        if at.rule:
            parts.append(f"- **Rule**: {at.rule}")
        if at.forbidden_patterns:
            parts.append("- **Forbidden patterns** (must not appear; rewrite if any match):")
            for s in at.forbidden_patterns:
                parts.append(f"  - `{s}`")
        parts.append("")

    lc = bundle.language_control
    if lc.forbidden_words or lc.replacement_style:
        parts.append("### Language control")
        if lc.forbidden_words:
            parts.append(
                "- **Forbidden words / phrases** (do not use these tokens in **draft_text**; use concrete alternatives):"
            )
            for s in lc.forbidden_words:
                parts.append(f"  - `{s}`")
        if lc.replacement_style:
            parts.append("- **Replacement style**:")
            for s in lc.replacement_style:
                parts.append(f"  - {s}")
        parts.append("")

    gs = bundle.generation_strategy
    if gs.rules or gs.structure or gs.writing_style or gs.content_rules:
        parts.append("### Generation strategy")
        if gs.rules:
            parts.append("- **Rules**:")
            for s in gs.rules:
                parts.append(f"  - {s}")
        if gs.structure:
            parts.append("- **Structure**:")
            for s in gs.structure:
                parts.append(f"  - {s}")
        if gs.writing_style:
            parts.append("- **Writing style**:")
            for s in gs.writing_style:
                parts.append(f"  - {s}")
        if gs.content_rules:
            parts.append("- **Content**:")
            for s in gs.content_rules:
                parts.append(f"  - {s}")
        parts.append("")

    pl = bundle.project_linking
    if pl.rule or pl.include_live_link:
        parts.append("### Project linking")
        parts.append(f"- **Include live link when relevant**: {pl.include_live_link}")
        if pl.rule:
            parts.append(f"- **Rule**: {pl.rule}")
        if pl.example:
            parts.append(f"- **Example**: {pl.example}")
        parts.append("")

    pe = bundle.portfolio_extension
    if pe.enabled or pe.rule or pe.text or pe.text_variants:
        parts.append("### Portfolio extension (optional line)")
        parts.append(f"- **Enabled**: {pe.enabled}")
        if pe.rule:
            parts.append(f"- **Rule**: {pe.rule}")
        if pe.text:
            parts.append(f"- **Suggested wording** (only if conditions met): {pe.text}")
        if pe.text_variants:
            parts.append("- **Choose at most one** of these lines when appropriate (match tone to the rest of the letter):")
            for s in pe.text_variants:
                parts.append(f"  - {s}")
        if pe.conditions:
            parts.append("- **Conditions**:")
            for s in pe.conditions:
                parts.append(f"  - {s}")
        parts.append("")

    qe = bundle.quality_enforcement
    if qe.forbidden or qe.required:
        parts.append("### Quality enforcement")
        if qe.forbidden:
            parts.append("- **Forbidden styles / content**:")
            for s in qe.forbidden:
                parts.append(f"  - {s}")
        if qe.required:
            parts.append("- **Required**:")
            for s in qe.required:
                parts.append(f"  - {s}")
        parts.append("")

    val = bundle.validation
    if val.checks or val.on_fail:
        parts.append("### Validation (self-check before output)")
        if val.checks:
            for s in val.checks:
                parts.append(f"- [ ] {s}")
        if val.on_fail:
            parts.append(f"- **If any check fails**: {val.on_fail}")
        parts.append("")

    parts.append("### Placeholder and salutation ban (always)")
    parts.append(
        "- Never use bracket placeholders such as `[Client's Name]`, `[Your Name]`, `[Contact Information]`, "
        "or similar. Write natural sentences; omit unknown names rather than using fake fill-in fields."
    )
    parts.append(
        "- Do not use formal letter headers (“Dear …”, “To whom it may concern”) unless the bundled rules explicitly allow it "
        "(they do not)."
    )
    parts.append("")

    legacy = format_legacy_generation_lists(bundle.generation)
    if legacy:
        parts.append(legacy)
        parts.append("")

    return "\n".join(parts).strip()
