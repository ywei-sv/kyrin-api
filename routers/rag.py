"""
Kyrin RAG Engine — document ingestion + vector search
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


def build_rag_context(q: str) -> tuple[str, list[dict]]:
    """Query RAG and build context string + sources list for LLM injection."""
    results = query(q)
    if not results:
        return "", []

    context_parts = []
    sources = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[{i}] (from: {r['source']})\n{r['content']}")
        sources.append({
            "index": i,
            "source": r["source"],
            "snippet": r["content"][:100],
        })

    context = (
        "\n\n## 📚 Retrieved Documents\n"
        + "\n\n".join(context_parts)
        + "\n\n**Instructions:** Use the above documents to answer. Cite sources as [1], [2], etc. "
        "If documents don't contain the answer, say so."
    )
    return context, sources


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


def delete_document(filename: str) -> bool:
    """Remove all chunks for a document."""
    collection = _get_collection()
    try:
        collection.delete(where={"source": filename})
        return True
    except Exception:
        return False
