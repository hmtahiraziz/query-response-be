"""Cover letter generation, refinement, and history API models."""

from typing import Literal

from pydantic import BaseModel, Field


class CoverLetterRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=50_000, description="Job description or client brief")
    k: int | None = Field(default=None, ge=1, le=30, description="Chunks to retrieve (optional)")


class SourceSnippet(BaseModel):
    project_id: str
    project_name: str | None = None
    page: int | None = None
    preview: str


class CoverLetterResponse(BaseModel):
    cover_letter: str
    sources: list[SourceSnippet]
    history_id: str


class RefineCoverLetterRequest(BaseModel):
    client_query: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="Original job description or brief (for tone and requirements)",
    )
    cover_letter: str = Field(..., min_length=20, max_length=100_000, description="Current draft to revise")
    instruction: str = Field(
        ...,
        min_length=3,
        max_length=8000,
        description="What to change (tone, wording, length, emphasis, etc.)",
    )
    selection: str | None = Field(
        default=None,
        max_length=8000,
        description="Optional excerpt the user highlighted—prioritize editing this part",
    )
    k: int | None = Field(default=None, ge=1, le=30, description="Chunks to retrieve for factual support")


class RefineCoverLetterResponse(BaseModel):
    cover_letter: str
    sources: list[SourceSnippet]


class CoverLetterHistorySummary(BaseModel):
    id: str
    created_at: int
    query_preview: str
    k: int | None = None


class CoverLetterHistoryVersion(BaseModel):
    id: str
    created_at: int
    source: Literal["generate", "refine", "manual"]
    body: str
    refine_note: str | None = None


class CoverLetterHistoryDetail(BaseModel):
    id: str
    created_at: int
    query: str
    k: int | None = None
    cover_letter: str
    sources: list[SourceSnippet]
    versions: list[CoverLetterHistoryVersion] = Field(default_factory=list)


class CoverLetterHistoryUpdate(BaseModel):
    cover_letter: str = Field(..., min_length=10, max_length=100_000)
    version_source: Literal["manual", "refine"] = "manual"
    refine_note: str | None = Field(default=None, max_length=4000)
    sources: list[SourceSnippet] | None = None


class ClaimItem(BaseModel):
    """One factual claim tied to a portfolio excerpt index (1-based, order in context)."""

    claim: str = Field(..., max_length=2000)
    excerpt_index: int = Field(..., ge=1, le=500)


class LetterSelfCheck(BaseModel):
    """Model-reported compliance; informational only (not trusted as proof)."""

    global_rules_addressed: bool = True
    chat_rules_addressed: bool = True
    structured_policy_met: bool = True
    within_max_words: bool = True
    notes: str = Field(default="", max_length=1000)


class CoverLetterStructuredOutput(BaseModel):
    """Structured response from the LLM for generate + refine."""

    draft_text: str = Field(..., min_length=10, max_length=100_000)
    claims: list[ClaimItem] = Field(default_factory=list, max_length=40)
    self_check: LetterSelfCheck = Field(default_factory=LetterSelfCheck)
