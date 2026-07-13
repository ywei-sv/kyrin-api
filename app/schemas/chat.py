"""
Pydantic models for the chat completions API.
"""

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict]  # supports vision format


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "deepseek-v4-flash"
    tier: str = "zenith"
    stream: bool = True
    max_tokens: int = 8192


class ChatResponse(BaseModel):
    id: str
    choices: list[dict]
    usage: dict | None = None


class ChatSession(BaseModel):
    id: str
    title: str
    messages: list[dict]
    updatedAt: int | float
