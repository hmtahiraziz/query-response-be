"""Extract text from PDFs and chunk with stable metadata for Pinecone."""

from __future__ import annotations

import re
from typing import List, Tuple

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings


def extract_pages(file_path: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    with fitz.open(file_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append((i + 1, text))
    return pages


def chunk_pages_to_documents(
    *,
    project_id: str,
    project_name: str,
    source_filename: str,
    pages: List[Tuple[int, str]],
    settings: Settings,
    embedding_provider: str = "openai",
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: List[Document] = []
    safe_name = _one_line(project_name) or "Untitled project"
    safe_file = _one_line(source_filename) or "document.pdf"

    for page_num, text in pages:
        chunks = splitter.split_text(text)
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "project_id": project_id,
                        "project_name": safe_name,
                        "page": page_num,
                        "source_filename": safe_file,
                        "embedding_provider": embedding_provider,
                    },
                )
            )
    return docs


def _one_line(s: str) -> str:
    t = (s or "").strip()
    t = re.sub(r"[\r\n]+", " ", t)
    return t[:512] if len(t) > 512 else t
