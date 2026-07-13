"""
LLM chat completions — proxies to opencode-go with streaming + vision support.
"""
import os
import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

router = APIRouter()

API_KEY = os.environ.get("KYRIN_API_KEY", "")
BASE_URL = os.environ.get("KYRIN_BASE_URL", "https://opencode.ai/zen/go/v1")
MODEL = os.environ.get("KYRIN_MODEL", "deepseek-v4-flash")

SYSTEM_PROMPTS = {
    "dawn": "You are Kyrin Dawn, a fast & efficient AI assistant. Give clear, well-structured answers with markdown formatting (headings, lists, bold). Be thorough enough but concise — aim for the right depth, not brevity at cost of quality.",
    "zenith": "You are Kyrin Zenith, a balanced AI. Give comprehensive, well-structured answers with markdown (headings, lists, bold, tables). Include examples and details where useful.",
    "dusk": "You are Kyrin Dusk, a deep-reasoning AI. Show step-by-step reasoning for complex topics. Be thorough and detailed. Use markdown formatting for clarity.",
}


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict]  # supports vision format


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = MODEL
    tier: str = "zenith"
    stream: bool = True
    max_tokens: int = 8192


class ChatResponse(BaseModel):
    id: str
    choices: list[dict]
    usage: dict | None = None


async def _stream_proxy(payload: dict) -> AsyncGenerator[bytes, None]:
    """Stream chunks from opencode-go SSE."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=180) as client:
        async with client.stream(
            "POST", f"{BASE_URL}/chat/completions", json=payload, headers=headers
        ) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk


def _inject_system(messages: list[dict], tier: str) -> list[dict]:
    """Prepend system prompt based on model tier."""
    prompt = SYSTEM_PROMPTS.get(tier, SYSTEM_PROMPTS["zenith"])
    return [{"role": "system", "content": prompt}, *messages]


@router.post("/chat/completions")
async def chat_completions(req: ChatRequest):
    """Proxy chat completions to the LLM with optional streaming."""
    raw = [m.model_dump() for m in req.messages]
    msgs = _inject_system(raw, req.tier)

    payload = {
        "model": req.model or MODEL,
        "messages": msgs,
        "max_tokens": req.max_tokens,
        "stream": req.stream,
    }

    if req.stream:
        return StreamingResponse(
            _stream_proxy(payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"{BASE_URL}/chat/completions", json=payload, headers=headers
        )
        if not resp.is_success:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()
