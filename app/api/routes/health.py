from typing import Any

from fastapi import APIRouter

from app.api.deps import SettingsDep

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/server/info")
def server_info(s: SettingsDep) -> dict[str, Any]:
    return {
        "cover_letter_history_backend": "mongodb"
        if (s.mongodb_uri or "").strip()
        else "json_file",
        "assistant_rules_source": "bundled_json",
        "projects_backend": "mongodb"
        if (s.mongodb_uri or "").strip()
        else "json_file",
        "openai_chat_model": s.openai_chat_model,
        "openai_embed_model": s.openai_embed_model,
        "openai_embed_dimensions": s.openai_embed_dimensions,
        "openai_max_retries": s.openai_max_retries,
        "openai_retry_cap_seconds": s.openai_retry_cap_seconds,
        "pinecone_index": s.pinecone_index_name,
        "pinecone_namespace": s.pinecone_namespace,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "default_rag_k": s.rag_k,
    }
