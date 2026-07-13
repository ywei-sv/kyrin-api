"""
Chat session persistence — saves/loads chats using SQLite (via built-in sqlite3).
Auto-migrates existing JSON files on first run.
Supports metadata column for tier/model info.
"""

import os
import json
import glob
import sqlite3

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.database import get_db

router = APIRouter()

DATA_DIR = os.environ.get("KYRIN_CHATS_DIR", os.path.expanduser("~/.kyrin/chats"))
DB_PATH = os.path.join(os.path.dirname(DATA_DIR), "chats.db")
JSON_DIR = DATA_DIR  # where old JSON files live


@router.on_event("startup")
async def migrate_json_to_sqlite():
    """Migrate existing JSON files to SQLite on first run (once)."""
    if not os.path.isdir(JSON_DIR):
        return
    with get_db() as db:
        count = db.execute("SELECT COUNT(*) as c FROM chats").fetchone()["c"]
        if count > 0:
            return

        files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")), reverse=True)
        migrated = 0
        for fp in files:
            try:
                with open(fp) as f:
                    data = json.load(f)
                sid = data.get("id", os.path.splitext(os.path.basename(fp))[0])
                title = data.get("title", "New chat")
                messages = json.dumps(data.get("messages", []), ensure_ascii=False)
                updated_at = data.get("updatedAt", 0)
                metadata = json.dumps(data.get("metadata", {}), ensure_ascii=False)
                db.execute(
                    "INSERT OR REPLACE INTO chats (id, title, messages, updatedAt, metadata) VALUES (?, ?, ?, ?, ?)",
                    (sid, title, messages, float(updated_at), metadata),
                )
                migrated += 1
            except Exception:
                continue
        db.commit()
        if migrated > 0:
            try:
                os.rename(JSON_DIR, JSON_DIR + "_migrated")
            except OSError:
                pass


class ChatSession(BaseModel):
    id: str
    title: str
    messages: list[dict]
    updatedAt: int | float
    metadata: dict = Field(default_factory=dict)


@router.get("/chats")
async def list_chats():
    """List all saved chat sessions, sorted by updatedAt desc."""
    with get_db() as db:
        rows = db.execute(
            "SELECT id, title, messages, updatedAt, metadata FROM chats ORDER BY updatedAt DESC"
        ).fetchall()
        result = []
        for r in rows:
            try:
                msgs = json.loads(r["messages"])
            except (json.JSONDecodeError, TypeError):
                msgs = []
            try:
                meta = json.loads(r["metadata"])
            except (json.JSONDecodeError, TypeError):
                meta = {}
            result.append({
                "id": r["id"],
                "title": r["title"],
                "messages": msgs,
                "updatedAt": r["updatedAt"],
                "metadata": meta,
            })
        return result


@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Get a single chat session."""
    with get_db() as db:
        row = db.execute(
            "SELECT id, title, messages, updatedAt, metadata FROM chats WHERE id = ?",
            (chat_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Chat not found")
    try:
        msgs = json.loads(row["messages"])
    except (json.JSONDecodeError, TypeError):
        msgs = []
    try:
        meta = json.loads(row["metadata"])
    except (json.JSONDecodeError, TypeError):
        meta = {}
    return {
        "id": row["id"],
        "title": row["title"],
        "messages": msgs,
        "updatedAt": row["updatedAt"],
        "metadata": meta,
    }


@router.post("/chats")
async def save_chat(session: ChatSession):
    """Save (create or overwrite) a chat session."""
    with get_db() as db:
        db.execute(
            "INSERT OR REPLACE INTO chats (id, title, messages, updatedAt, metadata) VALUES (?, ?, ?, ?, ?)",
            (
                session.id,
                session.title,
                json.dumps(session.messages, ensure_ascii=False),
                float(session.updatedAt),
                json.dumps(session.metadata, ensure_ascii=False),
            ),
        )
        db.commit()
    return {"status": "ok", "id": session.id}


@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat session."""
    with get_db() as db:
        db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        db.commit()
    return {"status": "deleted", "id": chat_id}
