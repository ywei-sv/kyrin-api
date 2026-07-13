"""
Application configuration via Pydantic Settings.
Loads from .env (project root) and environment variables.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── OpenCode LLM ────────────────────────────────────
    kyrin_api_key: str = ""
    kyrin_base_url: str = "https://opencode.ai/zen/go/v1"
    kyrin_model: str = "deepseek-v4-flash"

    # ── Server ──────────────────────────────────────────
    port: int = 5271

    # ── Web search ──────────────────────────────────────
    searxng_url: str = "http://127.0.0.1:8080"
    web_search_result_count: int = 5
    web_search_domain_filter: str = ""

    # ── Chat persistence ────────────────────────────────
    kyrin_chats_dir: str = "~/.kyrin/chats"

    # ── RAG ─────────────────────────────────────────────
    kyrin_rag_dir: str = "~/.kyrin/rag"
    kyrin_rag_upload_dir: str = "~/.kyrin/rag/uploads"
    kyrin_rag_chunk_size: int = 800
    kyrin_rag_overlap: int = 100
    kyrin_rag_top_k: int = 5

    # ── Anime ───────────────────────────────────────────
    trace_moe_url: str = "https://api.trace.moe"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
