"""Pydantic request/response models (API contracts)."""

from app.schemas.assistant import (
    AssistantPolicy,
    AssistantRulesResponse,
    CoverLetterAssistantRulesBundle,
    GenerationCodeRules,
)
from app.schemas.cover_letter import (
    ClaimItem,
    CoverLetterHistoryDetail,
    CoverLetterHistorySummary,
    CoverLetterHistoryUpdate,
    CoverLetterHistoryVersion,
    CoverLetterRequest,
    CoverLetterResponse,
    CoverLetterStructuredOutput,
    LetterSelfCheck,
    RefineCoverLetterRequest,
    RefineCoverLetterResponse,
    SourceSnippet,
)
from app.schemas.projects import GenerateProjectSummaryResponse, IngestResponse, ProjectAIProfile, ProjectSummary

__all__ = [
    "AssistantPolicy",
    "AssistantRulesResponse",
    "CoverLetterAssistantRulesBundle",
    "GenerationCodeRules",
    "ClaimItem",
    "CoverLetterHistoryDetail",
    "CoverLetterHistorySummary",
    "CoverLetterHistoryUpdate",
    "CoverLetterHistoryVersion",
    "CoverLetterRequest",
    "CoverLetterResponse",
    "CoverLetterStructuredOutput",
    "GenerateProjectSummaryResponse",
    "IngestResponse",
    "LetterSelfCheck",
    "ProjectAIProfile",
    "ProjectSummary",
    "RefineCoverLetterRequest",
    "RefineCoverLetterResponse",
    "SourceSnippet",
]
