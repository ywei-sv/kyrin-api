"""RAG service — core business logic for embedding, search, and reranking."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions


def get_chroma_client() -> chromadb.Client:
    """Get ChromaDB client (ephemeral or persistent)."""
    host = os.environ.get("CHROMA_HOST", "localhost")
    port = int(os.environ.get("CHROMA_PORT", "8000"))
    try:
        return chromadb.HttpClient(host=host, port=port)
    except Exception:
        persist = Path.home() / ".kyrin" / "chroma"
        persist.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(str(persist))


def get_collection():
    """Get or create the RAG collection."""
    client = get_chroma_client()
    name = os.environ.get("CHROMA_COLLECTION", "kyrin-docs")
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    try:
        return client.get_collection(name, embedding_function=emb_fn)
    except Exception:
        return client.create_collection(name, embedding_function=emb_fn)


def search_documents(query: str, n_results: int = 5) -> list[dict]:
    """Search documents and return ranked results."""
    coll = get_collection()
    try:
        results = coll.query(query_texts=[query], n_results=n_results)
    except Exception:
        return []
    hits = []
    if results.get("ids") and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            dist = results["distances"][0][i] if results.get("distances") else 0
            hits.append({
                "id": doc_id,
                "source": meta.get("source", "unknown"),
                "snippet": results["documents"][0][i][:500] if results.get("documents") else "",
                "score": round(1.0 - dist, 4),
            })
    # Optional cross-encoder rerank
    if len(hits) > 3:
        import re
        scored = []
        for h in hits:
            overlap = sum(1 for w in re.findall(r"\w+", query.lower()) if w in h["snippet"].lower())
            boost = 0.1 * (overlap / max(len(re.findall(r"\w+", query)), 1))
            scored.append((h["score"] + boost, h))
        scored.sort(key=lambda x: -x[0])
        hits = [s[1] for s in scored]
    return hits


def build_rag_context(query: str) -> tuple[str, list[dict]]:
    """Search ChromaDB and return (context_string, sources_list)."""
    results = search_documents(query)
    if not results:
        return "", []
    lines = []
    for i, r in enumerate(results):
        lines.append(
            f"[{i + 1}] (from: {r['source']})\n{r['snippet']}"
        )
    context = "\n\n## 📚 Retrieved Documents\n" + "\n\n".join(lines)
    sources = [
        {"index": i + 1, "source": r["source"], "snippet": r["snippet"][:200]}
        for i, r in enumerate(results)
    ]
    return context, sources


def delete_document(source: str) -> dict:
    """Delete a document by source filename."""
    coll = get_collection()
    try:
        results = coll.get(where={"source": source})
        ids = results.get("ids", [])
        if ids:
            coll.delete(ids=ids)
            return {"status": "deleted", "count": len(ids)}
        return {"status": "not_found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
