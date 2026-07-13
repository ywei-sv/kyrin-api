"""
Chat session persistence — saves/loads chats using SQLite (via built-in sqlite3).
Auto-migrates existing JSON files on first run.
"""

import os
import json
import glob
import sqlite3
import threading
from contextlib import contextmanager

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

DATA_DIR = os.environ.get("KYRIN_CHATS_DIR", os.path.expanduser("~/.kyrin/chats"))
DB_PATH = os.path.join(os.path.dirname(DATA_DIR), "chats.db")
JSON_DIR = DATA_DIR  # where old JSON files live

_local = threading.local()

@contextmanager
def _db():
    """Get thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New chat',
                messages TEXT NOT NULL DEFAULT '[]',
                updatedAt REAL NOT NULL
            )
        """)
        _local.conn.execute("CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(updatedAt DESC)")
        _local.conn.commit()
    try:
        yield _local.conn
    except Exception:
        _local.conn.rollback()
        raise


@router.on_event("startup")
async def migrate_json_to_sqlite():
    """Migrate existing JSON files to SQLite on first run (once)."""
    # Check if JSON dir still has files AND SQLite already has data
    if not os.path.isdir(JSON_DIR):
        return
    with _db() as db:
        count = db.execute("SELECT COUNT(*) as c FROM chats").fetchone()["c"]
        if count > 0:
            return  # already migrated

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
                db.execute(
                    "INSERT OR REPLACE INTO chats (id, title, messages, updatedAt) VALUES (?, ?, ?, ?)",
                    (sid, title, messages, float(updated_at)),
                )
                migrated += 1
            except Exception:
                continue
        db.commit()
        # Rename JSON dir to mark migration done
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


@router.get("/chats")
async def list_chats():
    """List all saved chat sessions, sorted by updatedAt desc."""
    with _db() as db:
        rows = db.execute(
            "SELECT id, title, messages, updatedAt FROM chats ORDER BY updatedAt DESC"
        ).fetchall()
        result = []
        for r in rows:
            try:
                msgs = json.loads(r["messages"])
            except (json.JSONDecodeError, TypeError):
                msgs = []
            result.append({
                "id": r["id"],
                "title": r["title"],
                "messages": msgs,
                "updatedAt": r["updatedAt"],
            })
        return result


@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Get a single chat session."""
    with _db() as db:
        row = db.execute(
            "SELECT id, title, messages, updatedAt FROM chats WHERE id = ?",
            (chat_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Chat not found")
    try:
        msgs = json.loads(row["messages"])
    except (json.JSONDecodeError, TypeError):
        msgs = []
    return {
        "id": row["id"],
        "title": row["title"],
        "messages": msgs,
        "updatedAt": row["updatedAt"],
    }


@router.post("/chats")
async def save_chat(session: ChatSession):
    """Save (create or overwrite) a chat session."""
    with _db() as db:
        db.execute(
            "INSERT OR REPLACE INTO chats (id, title, messages, updatedAt) VALUES (?, ?, ?, ?)",
            (
                session.id,
                session.title,
                json.dumps(session.messages, ensure_ascii=False),
                float(session.updatedAt),
            ),
        )
        db.commit()
    return {"status": "ok", "id": session.id}


@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat session."""
    with _db() as db:
        db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        db.commit()
    return {"status": "deleted", "id": chat_id}
