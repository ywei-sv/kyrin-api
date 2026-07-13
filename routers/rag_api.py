"""
RAG API — document upload, ingestion, query endpoint.
"""

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from routers import rag as rag_engine

router = APIRouter()

UPLOAD_DIR = Path(
    os.environ.get("KYRIN_RAG_UPLOAD_DIR", os.path.expanduser("~/.kyrin/rag/uploads"))
)
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPPORTED_EXT = {".pdf", ".txt", ".md"}


# ── Models ────────────────────────────────────────────
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

    # Save to temp then ingest
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    tmp = UPLOAD_DIR / file.filename
    with open(tmp, "wb") as f:
        f.write(content)

    try:
        result = rag_engine.ingest(tmp)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")


@router.post("/rag/ingest-text", response_model=IngestResponse)
async def ingest_text(req: IngestTextRequest):
    """Ingest raw text directly."""
    try:
        result = rag_engine.ingest_text(req.text, req.filename)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")


@router.post("/rag/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    """Query the RAG index for relevant chunks."""
    try:
        results = rag_engine.query(req.query, req.top_k)
        return QueryResponse(results=results)
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")


@router.get("/rag/documents")
async def list_docs():
    """List all ingested documents."""
    return {"documents": rag_engine.list_documents()}


@router.delete("/rag/documents/{filename}")
async def delete_doc(filename: str):
    """Delete an ingested document."""
    ok = rag_engine.delete_document(filename)
    if not ok:
        raise HTTPException(404, f"Document '{filename}' not found")

    # Also clean up upload
    fp = UPLOAD_DIR / filename
    if fp.exists():
        fp.unlink()
    return {"status": "deleted", "filename": filename}


@router.get("/rag/context")
async def get_rag_context(q: str):
    """Build RAG context string + sources for LLM injection (client-side use)."""
    context, sources = rag_engine.build_rag_context(q)
    return {"context": context, "sources": sources}
