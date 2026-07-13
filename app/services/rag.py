"""
Kyrin RAG Engine — document ingestion + vector search + context building.
Uses ChromaDB with built-in ONNX embedding (no PyTorch/sentence-transformers needed).
Single source of truth — imported by both routers/rag.py and routers/chat.py.
"""

import os
import re
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

# ── Config ─────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("KYRIN_RAG_DIR", settings.kyrin_rag_dir))
CHUNK_SIZE = int(os.environ.get("KYRIN_RAG_CHUNK_SIZE", str(settings.kyrin_rag_chunk_size)))
CHUNK_OVERLAP = int(os.environ.get("KYRIN_RAG_OVERLAP", str(settings.kyrin_rag_overlap)))
TOP_K = int(os.environ.get("KYRIN_RAG_TOP_K", str(settings.kyrin_rag_top_k)))

os.makedirs(DATA_DIR, exist_ok=True)

# ChromaDB persistent client (uses ONNX embedding by default — no PyTorch needed)
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


def get_collection():
    """Alias for external imports (e.g. routers/rag.py)."""
    return _get_collection()


# ── Chunking ──────────────────────────────────────────
def chunk_text(text: str, filename: str = "") -> list[dict]:
    """Split text into overlapping chunks with paragraph/sentence boundary detection."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        # Try to break at paragraph or sentence boundary
        if end < len(text):
            para = text.rfind("\n\n", start, end + CHUNK_OVERLAP)
            if para > start + CHUNK_SIZE // 2:
                end = para
            else:
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
    """Extract text from a PDF file using PyMuPDF."""
    import fitz
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
    """Ingest a document into the vector store. Replaces old chunks for the same file."""
    text = parse_file(file_path)
    chunks = chunk_text(text, file_path.name)
    collection = _get_collection()

    # Remove old chunks for this file
    try:
        collection.delete(where={"source": file_path.name})
    except Exception:
        pass

    if not chunks:
        return {"ingested": 0, "filename": file_path.name, "error": "No extractable content"}

    ids = [c["id"] for c in chunks]
    texts = [c["content"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_idx": c["chunk_idx"]} for c in chunks]

    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    return {"ingested": len(chunks), "filename": file_path.name}


def ingest_text(text: str, filename: str = "paste.txt") -> dict:
    """Ingest raw text directly into the vector store."""
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


def search_documents(query_str: str, n_results: int = 5) -> list[dict]:
    """Search documents and return ranked results with keyword reranking."""
    results = query(query_str, n_results)
    if not results:
        return []

    for r in results:
        r["score"] = round(1.0 - r["score"], 4)

    # Simple keyword overlap rerank for 3+ results
    if len(results) > 3:
        scored = []
        for h in results:
            overlap = sum(1 for w in re.findall(r"\w+", query_str.lower())
                          if w in h.get("content", "").lower())
            boost = 0.1 * (overlap / max(len(re.findall(r"\w+", query_str)), 1))
            scored.append((h["score"] + boost, h))
        scored.sort(key=lambda x: -x[0])
        results = [s[1] for s in scored]
    return results


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


def delete_document(source: str) -> bool:
    """Remove all chunks for a document from the vector store."""
    collection = _get_collection()
    try:
        collection.delete(where={"source": source})
        return True
    except Exception:
        return False


def build_rag_context(q: str) -> tuple[str, list[dict]]:
    """Query RAG and build context string + sources list for LLM injection."""
    results = search_documents(q)
    if not results:
        return "", []

    context_parts = []
    sources = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[{i}] (from: {r['source']})\n{r['content']}")
        sources.append({
            "index": i,
            "source": r["source"],
            "snippet": r["content"][:200],
            "score": r.get("score", 0),
        })

    context = (
        "\n\n## 📚 Retrieved Documents\n"
        + "\n\n".join(context_parts)
        + "\n\n**Instructions:** Use the above documents to answer. Cite sources as [1], [2], etc. "
        "If documents don't contain the answer, say so."
    )
    return context, sources
