"""Assistant rules models (policy + code-bundled generation rules)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class AssistantPolicy(BaseModel):
    """Machine-readable constraints applied to cover letter generation and refinement."""

    model_config = ConfigDict(extra="ignore")

    language: Literal["none", "uk", "us"] = Field(
        default="none",
        description="Preferred spelling/locale when not 'none'",
    )
    max_words: int | None = Field(
        default=None,
        ge=50,
        le=10_000,
        description="Hard cap on draft length when set",
    )
    must_include: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Up to 5 short phrases that must appear",
    )
    must_not_include: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Up to 5 phrases or words that must not appear",
    )

    @field_validator("must_include", "must_not_include", mode="before")
    @classmethod
    def _normalize_string_lists(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        out: list[str] = []
        for item in v[:5]:
            s = str(item).strip()
            if not s:
                continue
            out.append(s[:200])
        return out


class GenerationCodeRules(BaseModel):
    """Optional extra bullet lists (legacy shape); bundled JSON may also use structured sections below."""

    model_config = ConfigDict(extra="ignore")

    tone_and_voice: list[str] = Field(default_factory=list, max_length=24)
    composition: list[str] = Field(default_factory=list, max_length=24)
    factual_grounding: list[str] = Field(default_factory=list, max_length=24)
    formatting: list[str] = Field(default_factory=list, max_length=16)


# --- Structured sections in ``cover_letter_assistant.default.json`` ---


class PersonaVoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tense: str = ""
    tone: str = ""


class PersonaRules(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = ""
    voice: PersonaVoice = Field(default_factory=PersonaVoice)


class HardOpening(BaseModel):
    model_config = ConfigDict(extra="ignore")

    forbidden_starts: list[str] = Field(default_factory=list)
    rule: str = ""


class HardClosing(BaseModel):
    model_config = ConfigDict(extra="ignore")

    forbidden_phrases: list[str] = Field(default_factory=list)
    rule: str = ""


class HardProjectUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_projects: int = Field(default=1, ge=1, le=20)
    rule: str = ""


class HardFormatting(BaseModel):
    model_config = ConfigDict(extra="ignore")

    plain_text_only: bool = True
    allow_markdown: bool = False
    allow_bullets: bool = False


class HardConstraints(BaseModel):
    model_config = ConfigDict(extra="ignore")

    opening: HardOpening = Field(default_factory=HardOpening)
    closing: HardClosing = Field(default_factory=HardClosing)
    project_usage: HardProjectUsage = Field(default_factory=HardProjectUsage)
    formatting: HardFormatting = Field(default_factory=HardFormatting)


class AntiTemplateBlocker(BaseModel):
    model_config = ConfigDict(extra="ignore")

    forbidden_patterns: list[str] = Field(default_factory=list)
    rule: str = ""


class ResponseFlow(BaseModel):
    """Ordered sections for the letter (e.g. client_hook → project_proof → …)."""

    model_config = ConfigDict(extra="ignore")

    strict_order: list[str] = Field(default_factory=list)
    definitions: dict[str, str] = Field(default_factory=dict)


class LanguageControl(BaseModel):
    """Vocabulary and phrasing constraints."""

    model_config = ConfigDict(extra="ignore")

    forbidden_words: list[str] = Field(default_factory=list, max_length=80)
    replacement_style: list[str] = Field(default_factory=list, max_length=24)


class GenerationStrategy(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rules: list[str] = Field(default_factory=list, max_length=32)
    structure: list[str] = Field(default_factory=list)
    writing_style: list[str] = Field(default_factory=list)
    content_rules: list[str] = Field(default_factory=list)


class ProjectLinking(BaseModel):
    model_config = ConfigDict(extra="ignore")

    include_live_link: bool = False
    rule: str = ""
    example: str = ""


class PortfolioExtension(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    rule: str = ""
    text: str = ""
    text_variants: list[str] = Field(default_factory=list, max_length=12)
    conditions: list[str] = Field(default_factory=list)


class QualityEnforcement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    forbidden: list[str] = Field(default_factory=list)
    required: list[str] = Field(default_factory=list)


class ValidationRules(BaseModel):
    model_config = ConfigDict(extra="ignore")

    checks: list[str] = Field(default_factory=list)
    on_fail: str = ""


class CoverLetterAssistantRulesBundle(BaseModel):
    """Root object in ``app/rules/cover_letter_assistant.default.json``."""

    model_config = ConfigDict(extra="ignore")

    version: int = Field(default=1, ge=1, le=10_000)
    policy: AssistantPolicy = Field(default_factory=AssistantPolicy)
    generation: GenerationCodeRules = Field(default_factory=GenerationCodeRules)
    persona: PersonaRules = Field(default_factory=PersonaRules)
    hard_constraints: HardConstraints = Field(default_factory=HardConstraints)
    response_flow: ResponseFlow = Field(default_factory=ResponseFlow)
    anti_template_blocker: AntiTemplateBlocker = Field(default_factory=AntiTemplateBlocker)
    language_control: LanguageControl = Field(default_factory=LanguageControl)
    generation_strategy: GenerationStrategy = Field(default_factory=GenerationStrategy)
    project_linking: ProjectLinking = Field(default_factory=ProjectLinking)
    portfolio_extension: PortfolioExtension = Field(default_factory=PortfolioExtension)
    quality_enforcement: QualityEnforcement = Field(default_factory=QualityEnforcement)
    validation: ValidationRules = Field(default_factory=ValidationRules)


class AssistantRulesResponse(BaseModel):
    """Read-only view of bundled rules (for UI / API). Top-level fields mirror ``bundle`` for compatibility."""

    source: Literal["code"] = "code"
    rules_path: str = Field(description="Relative path to the bundled JSON inside the backend package")
    bundle: CoverLetterAssistantRulesBundle

    @computed_field  # type: ignore[prop-decorator]
    @property
    def version(self) -> int:
        return self.bundle.version

    @computed_field  # type: ignore[prop-decorator]
    @property
    def policy(self) -> AssistantPolicy:
        return self.bundle.policy

    @computed_field  # type: ignore[prop-decorator]
    @property
    def generation(self) -> GenerationCodeRules:
        return self.bundle.generation
