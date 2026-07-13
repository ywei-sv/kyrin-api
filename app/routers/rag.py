"""
Kyrin RAG Engine API endpoints.
Thin wrappers around app.services.rag — all business logic lives in services.
"""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.services.rag import (
    ingest as _ingest,
    ingest_text as _ingest_text,
    query as _query,
    list_documents as _list_documents,
    delete_document as _delete_document,
    build_rag_context,
)

router = APIRouter()

# ── Config ─────────────────────────────────────────────
UPLOAD_DIR = Path(
    os.environ.get("KYRIN_RAG_UPLOAD_DIR", os.path.expanduser("~/.kyrin/rag/uploads"))
)
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPPORTED_EXT = {".pdf", ".txt", ".md"}
import asyncio


async def warmup_rag():
    """Pre-warm ChromaDB embedding model on startup."""
    try:
        from app.services.rag import get_collection
        coll = get_collection()
        # Dummy query to trigger ONNX model loading
        _ = coll.query(query_texts=["warmup"], n_results=1)
    except Exception:
        pass  # Non-critical — will load on first real query



class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    results: list[dict]


class IngestTextRequest(BaseModel):
    text: str
    filename: str = "paste.txt"


class IngestResponse(BaseModel):
    ingested: int
    filename: str
    error: str | None = None


# ── Endpoints ─────────────────────────────────────────
@router.post("/rag/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a document (PDF, TXT, MD)."""
    ext = Path(file.filename or "file.txt").suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(400, f"Unsupported file type: {ext}. Use: {', '.join(SUPPORTED_EXT)}")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    tmp = UPLOAD_DIR / file.filename
    with open(tmp, "wb") as f:
        f.write(content)

    try:
        result = _ingest(tmp)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")


@router.post("/rag/ingest-text", response_model=IngestResponse)
async def ingest_text_endpoint(req: IngestTextRequest):
    """Ingest raw text directly."""
    try:
        result = _ingest_text(req.text, req.filename)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")


@router.post("/rag/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    """Query the RAG index for relevant chunks."""
    try:
        results = _query(req.query, req.top_k)
        return QueryResponse(results=results)
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")


@router.get("/rag/documents")
async def list_docs():
    """List all ingested documents."""
    return {"documents": _list_documents()}


@router.delete("/rag/documents/{filename}")
async def delete_doc(filename: str):
    """Delete an ingested document."""
    ok = _delete_document(filename)
    if not ok:
        raise HTTPException(404, f"Document '{filename}' not found")

    fp = UPLOAD_DIR / filename
    if fp.exists():
        fp.unlink()
    return {"status": "deleted", "filename": filename}


@router.get("/rag/context")
async def get_rag_context(q: str):
    """Build RAG context string + sources for LLM injection (client-side use)."""
    context, sources = build_rag_context(q)
    return {"context": context, "sources": sources}
