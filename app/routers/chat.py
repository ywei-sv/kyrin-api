"""
LLM chat completions — proxies to opencode-go with streaming + vision support.
Auto-injects RAG context when documents are available.
Uses shared business logic from app.services.chat.
"""

import os
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

from app.services import build_rag_context
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
    """Proxy chat completions to the LLM with optional streaming and function calling."""
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
        # Filter out Kyrin-self RAG sources that aren't relevant to user queries
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

    payload = {
        "model": req.model or MODEL,
        "messages": msgs,
        "max_tokens": req.max_tokens,
        "stream": req.stream,
    }

    if req.stream:
        return StreamingResponse(
            stream_proxy(payload, API_KEY, BASE_URL),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming with function calling support
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=180) as client:
        # First call with tools
        payload_with_tools = {**payload, "stream": False, "tools": TOOLS}
        resp = await client.post(
            f"{BASE_URL}/chat/completions", json=payload_with_tools, headers=headers
        )
        if not resp.is_success:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()

        # Check for function calling
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                result = await exec_tool(name, args)
                msgs.append(msg)
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result,
                })

            # Second call (no tools) for synthesis
            payload2 = {
                "model": req.model or MODEL,
                "messages": msgs,
                "max_tokens": req.max_tokens,
                "stream": False,
            }
            resp2 = await client.post(
                f"{BASE_URL}/chat/completions", json=payload2, headers=headers
            )
            if not resp2.is_success:
                raise HTTPException(
                    status_code=resp2.status_code, detail=resp2.text
                )
            return resp2.json()

        return data
