"""Application startup / shutdown hooks."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import ensure_data_dirs, get_settings
from app.services.manifest_service import backfill_legacy_embedding_providers


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Expose keys to env for LangChain / Pinecone clients that read os.environ."""
    s = get_settings()
    os.environ["PINECONE_API_KEY"] = s.pinecone_api_key
    os.environ.setdefault("OPENAI_API_KEY", s.openai_api_key)
    ensure_data_dirs()
    backfill_legacy_embedding_providers()
    if (s.mongodb_uri or "").strip():
        from app.services.cover_letter_history_mongo import ensure_indexes as history_ensure_indexes
        from app.services.manifest_mongo import ensure_indexes as manifest_ensure_indexes

        history_ensure_indexes()
        manifest_ensure_indexes()
    yield
