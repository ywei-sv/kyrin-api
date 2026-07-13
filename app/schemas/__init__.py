"""Pydantic models for chat API."""
from __future__ import annotations

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # 'user' | 'assistant' | 'system'
    content: str | list | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "deepseek-v4-flash"
    stream: bool = True
    tier: str = "dawn"


class ChatSession(BaseModel):
    id: str
    title: str
    messages: list[dict]
    updatedAt: int | float
