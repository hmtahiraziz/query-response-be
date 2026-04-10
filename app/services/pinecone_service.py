"""Pinecone: ingest, delete by project, and MMR retrieval."""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.config import Settings, get_settings
from app.services.gemini_service import get_embeddings_model
from app.services.pinecone_errors import pinecone_connection_user_hint

_pc: Pinecone | None = None
_index = None


def _client(settings: Settings) -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def get_raw_index(settings: Settings | None = None):
    global _index
    s = settings or get_settings()
    if _index is None:
        _index = _client(s).Index(s.pinecone_index_name)
    return _index


def reset_index_cache() -> None:
    """Test hook: clear cached gRPC index handle."""
    global _index
    _index = None


def get_vectorstore(settings: Settings | None = None) -> PineconeVectorStore:
    s = settings or get_settings()
    return PineconeVectorStore(
        index=get_raw_index(s),
        embedding=get_embeddings_model(),
        namespace=s.pinecone_namespace,
    )


def ingest_documents(documents: List[Document], settings: Settings | None = None) -> None:
    """Upsert chunks into the shared portfolio namespace."""
    s = settings or get_settings()
    if not documents:
        return
    PineconeVectorStore.from_documents(
        documents,
        get_embeddings_model(),
        index_name=s.pinecone_index_name,
        namespace=s.pinecone_namespace,
        pinecone_api_key=s.pinecone_api_key,
    )


def delete_project_vectors(project_id: str, settings: Settings | None = None) -> None:
    s = settings or get_settings()
    idx = get_raw_index(s)
    idx.delete(filter={"project_id": {"$eq": project_id}}, namespace=s.pinecone_namespace)


def retrieve_context(
    query: str,
    *,
    k: int,
    settings: Settings | None = None,
) -> List[Document]:
    """MMR search across all ingested projects in the namespace."""
    s = settings or get_settings()
    fetch_k = min(max(k * 4, 16), s.rag_fetch_k_max)

    def _search() -> List[Document]:
        store = get_vectorstore(s)
        return store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=s.rag_mmr_lambda,
        )

    try:
        return _search()
    except Exception as exc:
        if pinecone_connection_user_hint(exc):
            reset_index_cache()
            return _search()
        raise
