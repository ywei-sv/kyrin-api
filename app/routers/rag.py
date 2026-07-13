"""
Kyrin RAG Engine — document ingestion + vector search + API endpoints.
Uses ChromaDB with built-in ONNX embedding (no PyTorch/sentence-transformers needed).
"""

import os
import re
import hashlib
import tempfile
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from app.services import build_rag_context, delete_document, search_documents, get_collection

router = APIRouter()

# ── Config ─────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("KYRIN_RAG_DIR", os.path.expanduser("~/.kyrin/rag")))
CHUNK_SIZE = int(os.environ.get("KYRIN_RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("KYRIN_RAG_OVERLAP", "100"))
TOP_K = int(os.environ.get("KYRIN_RAG_TOP_K", "5"))

os.makedirs(DATA_DIR, exist_ok=True)

# ChromaDB persistent client (uses ONNX embedding by default)
_client = chromadb.PersistentClient(
    path=str(DATA_DIR / "vectordb"),
    settings=ChromaSettings(anonymized_telemetry=False),
)
_collection_name = "kyrin-docs"
_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        try:
            _collection = _client.get_collection(_collection_name)
        except Exception:
            _collection = _client.create_collection(_collection_name)
    return _collection


# ── Chunking ──────────────────────────────────────────
def chunk_text(text: str, filename: str = "") -> list[dict]:
    """Split text into overlapping chunks."""
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        # Try to break at paragraph or sentence boundary
        if end < len(text):
            # Look backwards for paragraph break
            para = text.rfind("\n\n", start, end + CHUNK_OVERLAP)
            if para > start + CHUNK_SIZE // 2:
                end = para
            else:
                # Look for sentence break
                sent = max(
                    text.rfind(". ", start, end + CHUNK_OVERLAP),
                    text.rfind("! ", start, end + CHUNK_OVERLAP),
                    text.rfind("? ", start, end + CHUNK_OVERLAP),
                )
                if sent > start + CHUNK_SIZE // 2:
                    end = sent + 1

        content = text[start:end].strip()
        if content:
            doc_id = hashlib.md5(f"{filename}:{start}".encode()).hexdigest()[:12]
            chunks.append({
                "id": doc_id,
                "content": content,
                "source": filename,
                "chunk_idx": len(chunks),
            })
        start = end - CHUNK_OVERLAP if end < len(text) else len(text)

    return chunks


def parse_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    return "\n\n".join(text_parts)


def parse_file(path: Path) -> str:
    """Parse a document file into plain text."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    elif ext in (".txt", ".md", ".mdoc"):
        return path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Ingest ────────────────────────────────────────────
def ingest(file_path: Path) -> dict:
    """Ingest a document into the vector store."""
    text = parse_file(file_path)
    chunks = chunk_text(text, file_path.name)

    collection = _get_collection()
    existing = set()
    try:
        existing_meta = collection.get(include=["metadatas"])
        if existing_meta and existing_meta["metadatas"]:
            for m in existing_meta["metadatas"]:
                if m.get("source") == file_path.name:
                    existing.add(m["source"])
    except Exception:
        pass

    if file_path.name in existing:
        # Remove old chunks for this file
        collection.delete(where={"source": file_path.name})

    if not chunks:
        return {"ingested": 0, "filename": file_path.name, "error": "No extractable content"}

    ids = [c["id"] for c in chunks]
    texts = [c["content"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_idx": c["chunk_idx"]} for c in chunks]

    collection.add(documents=texts, metadatas=metadatas, ids=ids)

    return {"ingested": len(chunks), "filename": file_path.name}


def ingest_text(text: str, filename: str = "paste.txt") -> dict:
    """Ingest raw text directly."""
    chunks = chunk_text(text, filename)
    collection = _get_collection()

    ids = [c["id"] for c in chunks]
    texts = [c["content"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_idx": c["chunk_idx"]} for c in chunks]

    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    return {"ingested": len(chunks), "filename": filename}


# ── Query ─────────────────────────────────────────────
def query(q: str, top_k: int = TOP_K) -> list[dict]:
    """Search for relevant document chunks."""
    collection = _get_collection()
    results = collection.query(query_texts=[q], n_results=top_k)

    if not results or not results["ids"] or not results["ids"][0]:
        return []

    items = []
    for i in range(len(results["ids"][0])):
        items.append({
            "id": results["ids"][0][i],
            "content": results["documents"][0][i] if results["documents"] else "",
            "source": results["metadatas"][0][i].get("source", "") if results["metadatas"] else "",
            "score": results["distances"][0][i] if results.get("distances") else 0,
        })
    return items


def list_documents() -> list[str]:
    """List unique document sources in the vector store."""
    collection = _get_collection()
    try:
        meta = collection.get(include=["metadatas"])
        if not meta or not meta["metadatas"]:
            return []
        sources = set()
        for m in meta["metadatas"]:
            if m.get("source"):
                sources.add(m["source"])
        return sorted(sources)
    except Exception:
        return []


# ── API Models ────────────────────────────────────────
UPLOAD_DIR = Path(
    os.environ.get("KYRIN_RAG_UPLOAD_DIR", os.path.expanduser("~/.kyrin/rag/uploads"))
)
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPPORTED_EXT = {".pdf", ".txt", ".md"}


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
        result = ingest(tmp)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")


@router.post("/rag/ingest-text", response_model=IngestResponse)
async def ingest_text_endpoint(req: IngestTextRequest):
    """Ingest raw text directly."""
    try:
        result = ingest_text(req.text, req.filename)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")


@router.post("/rag/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    """Query the RAG index for relevant chunks."""
    try:
        results = query(req.query, req.top_k)
        return QueryResponse(results=results)
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")


@router.get("/rag/documents")
async def list_docs():
    """List all ingested documents."""
    return {"documents": list_documents()}


@router.delete("/rag/documents/{filename}")
async def delete_doc(filename: str):
    """Delete an ingested document."""
    ok = delete_document(filename)
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
    context, sources = build_rag_context(q)
    return {"context": context, "sources": sources}
