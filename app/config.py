"""Environment-driven settings (Gemini + Pinecone only)."""

from dotenv import load_dotenv

load_dotenv()

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
MANIFEST_PATH = DATA_DIR / "projects.json"
COVER_LETTER_HISTORY_PATH = DATA_DIR / "cover_letter_history.json"
ASSISTANT_RULES_PATH = DATA_DIR / "assistant_rules.json"
COVER_LETTER_HISTORY_MAX = 150


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = Field(..., description="Google AI Studio / Gemini API key")
    gemini_chat_model: str = Field(
        default="gemini-2.0-flash",
        description="Model for cover letter generation",
    )
    gemini_embed_model: str = Field(
        default="models/gemini-embedding-001",
        description="Embedding model (3072 dims for gemini-embedding-001)",
    )

    gemini_max_retries: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Attempts per call on 429 / quota (chat + embeddings)",
    )
    gemini_retry_cap_seconds: float = Field(
        default=120.0,
        ge=5.0,
        le=600.0,
        description="Max single sleep between retries (server hint is capped to this)",
    )

    pinecone_api_key: str = Field(..., description="Pinecone API key")
    pinecone_index_name: str = Field(..., description="Dense index name (cosine, dim 3072)")
    pinecone_namespace: str = Field(
        default="portfolio",
        description="Single namespace; vectors carry project_id in metadata",
    )

    chunk_size: int = Field(default=1000, ge=200, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    rag_k: int = Field(default=8, ge=1, le=30, description="Default chunks for cover letter context")
    rag_fetch_k_max: int = Field(default=48, ge=16, le=200)
    rag_mmr_lambda: float = Field(default=0.55, ge=0.0, le=1.0)

    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated origins for CORS (must match the browser address bar exactly)",
    )

    mongodb_uri: str | None = Field(
        default=None,
        description=(
            "MongoDB connection string; if unset, project manifest / history / rules use JSON files under data/"
        ),
    )
    mongodb_db_name: str = Field(
        default="portfolio_cover_letter",
        description="Database name for cover letter history",
    )
    mongodb_collection_cover_letters: str = Field(
        default="cover_letter_history",
        description="Collection for cover letter history documents",
    )
    mongodb_collection_assistant_rules: str = Field(
        default="assistant_rules",
        description="Collection for global + chat assistant rules (single document)",
    )
    mongodb_collection_projects: str = Field(
        default="portfolio_projects",
        description="Collection for ingested project metadata (replaces data/projects.json when MongoDB is set)",
    )

    def cors_origin_list(self) -> list[str]:
        out: list[str] = []
        for o in self.cors_origins.split(","):
            s = o.strip().rstrip("/")
            if s:
                out.append(s)
        return out


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
