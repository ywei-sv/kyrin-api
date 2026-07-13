"""
Shared SQLite connection manager for chat persistence.
Provides a thread-local context manager for SQLite access.
"""

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from app.config import settings

DATA_DIR = Path(os.path.expanduser(settings.kyrin_chats_dir))
DB_PATH = DATA_DIR.parent / "chats.db"

_local = threading.local()


@contextmanager
def get_db():
    """Get thread-local SQLite connection with auto-created schema."""
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(DB_PATH.parent, exist_ok=True)
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New chat',
                messages TEXT NOT NULL DEFAULT '[]',
                updatedAt REAL NOT NULL
            )
        """)
        _local.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(updatedAt DESC)"
        )
        _local.conn.commit()
    try:
        yield _local.conn
    except Exception:
        _local.conn.rollback()
        raise
