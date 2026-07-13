"""
LLM chat completions — proxies to opencode-go with streaming + vision support.
Auto-injects RAG context when documents are available.
Uses shared business logic from app.services.chat.

Two-round approach:
  Round 1 (non-streaming, with tools) → tool execution if needed
  Round 2 (streaming, no tools) → final response to client
"""

import os
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

from app.services.rag import build_rag_context
from app.services.chat import (
    SYSTEM_PROMPTS,
    TOOLS,
    exec_tool,
    stream_proxy,
    inject_system,
)

router = APIRouter()

API_KEY = os.environ.get("KYRIN_API_KEY", "")
BASE_URL = os.environ.get("KYRIN_BASE_URL", "https://opencode.ai/zen/go/v1")
MODEL = os.environ.get("KYRIN_MODEL", "deepseek-v4-flash")


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


# ── API Endpoint ───────────────────────────────────────


@router.post("/chat/completions")
async def chat_completions(req: ChatRequest):
    """Proxy chat completions with two-round tool calling support.

    Round 1: Non-streaming with tools → detect tool calls, execute them.
    Round 2: Streaming (or not) without tools → forward to client.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")
    raw = [m.model_dump() for m in req.messages]
    msgs = inject_system(raw, req.tier)

    # Auto RAG — query documents and inject context if relevant
    last_user_msg = None
    for m in reversed(raw):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            last_user_msg = m["content"]
            break
    rag_context, rag_sources = build_rag_context(last_user_msg or "")
    if rag_context and rag_sources:
        skip_words = ['kyrin-intro', 'kyrin-devlog', 'kyrin-model', 'kyrin-chat']
        filtered = [
            s for s in rag_sources
            if not any(w in s.get('source', '') for w in skip_words)
        ]
        if filtered:
            context_parts = []
            for i, r in enumerate(filtered, 1):
                context_parts.append(f"[{i}] (from: {r['source']})\n{r['snippet']}")
            rag_context = (
                "\n\n## 📚 Retrieved Documents\n"
                + "\n\n".join(context_parts)
                + "\n\n**Instructions:** Use the above documents to answer. "
                "Cite sources as [1], [2], etc. "
                "If documents don't contain the answer, say so."
            )
            msgs.insert(0, {"role": "system", "content": rag_context})

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=180) as client:

        # ── Round 1: non-streaming with tools ──────────
        payload1 = {
            "model": req.model or MODEL,
            "messages": msgs,
            "max_tokens": req.max_tokens,
            "stream": False,
            "tools": TOOLS,
        }
        resp1 = await client.post(
            f"{BASE_URL}/chat/completions", json=payload1, headers=headers
        )
        if not resp1.is_success:
            raise HTTPException(status_code=resp1.status_code, detail=resp1.text)
        data1 = resp1.json()

        choice = data1.get("choices", [{}])[0]
        msg1 = choice.get("message", {})

        # Execute any tool calls from Round 1
        if msg1.get("tool_calls"):
            for tc in msg1["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                result = await exec_tool(name, args)
                msgs.append(msg1)
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result,
                })

        # ── Round 2: forward to client (streaming or not) ──
        payload2 = {
            "model": req.model or MODEL,
            "messages": msgs,
            "max_tokens": req.max_tokens,
            "stream": req.stream,
        }

        if req.stream:
            return StreamingResponse(
                stream_proxy(payload2, API_KEY, BASE_URL),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        resp2 = await client.post(
            f"{BASE_URL}/chat/completions", json=payload2, headers=headers
        )
        if not resp2.is_success:
            raise HTTPException(status_code=resp2.status_code, detail=resp2.text)
        return resp2.json()
