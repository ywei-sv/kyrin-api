"""RAG service — core business logic for embedding, search, and reranking."""
from __future__ import annotations

from app.services.rag import (
    build_rag_context,
    delete_document,
    search_documents,
    get_collection,
    chunk_text,
    parse_file,
    parse_pdf,
    ingest,
    ingest_text,
    query,
    list_documents,
)

__all__ = [
    "build_rag_context",
    "delete_document",
    "search_documents",
    "get_collection",
    "chunk_text",
    "parse_file",
    "parse_pdf",
    "ingest",
    "ingest_text",
    "query",
    "list_documents",
]
