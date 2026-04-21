"""Project list, ingest, and summary API models."""

from typing import Literal

from pydantic import BaseModel, Field


class ProjectAIProfile(BaseModel):
    name: str = Field(..., min_length=1, max_length=240)
    type: list[str] = Field(default_factory=list, max_length=12)
    problem: str = Field(default="", max_length=2000)
    solution: str = Field(default="", max_length=2000)
    project_brief: str = Field(
        default="",
        max_length=6000,
        description="A few-hundred-word project overview with clear context and outcomes.",
    )
    technical_depth: str = Field(
        default="",
        max_length=4000,
        description="Technical architecture, key engineering decisions, and implementation details.",
    )
    stack: list[str] = Field(default_factory=list, max_length=20)
    impact: str = Field(default="", max_length=2000)
    talking_points: list[str] = Field(default_factory=list, max_length=12)
    live_link: str | None = Field(default=None, max_length=1200)


class ProjectSummary(BaseModel):
    project_id: str
    name: str
    filename: str
    chunks: int
    pages: int
    created_at: int
    ai_summary: ProjectAIProfile | None = None
    summary_generated_at: int | None = None
    embedding_provider: Literal["gemini", "openai"] = Field(
        default="gemini",
        description="Which API produced Pinecone vectors for this project",
    )


class IngestResponse(BaseModel):
    project_id: str
    name: str
    filename: str
    pages: int
    chunks: int
    ai_summary: ProjectAIProfile | None = None
    summary_generated_at: int | None = None
    embedding_provider: Literal["openai"] = "openai"


class GenerateProjectSummaryResponse(BaseModel):
    project_id: str
    ai_summary: ProjectAIProfile
    summary_generated_at: int
