"""
LLM chat completions — smart streaming with inline tool execution.
Instead of two sequential LLM calls, we stream the first call (with tools)
and only pause + execute tools if the LLM actually calls one.
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
    inject_system,
)

router = APIRouter()

API_KEY = os.environ.get("KYRIN_API_KEY", "")
BASE_URL = os.environ.get("KYRIN_BASE_URL", "https://opencode.ai/zen/go/v1")
MODEL = os.environ.get("KYRIN_MODEL", "deepseek-v4-flash")


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict]


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = MODEL
    tier: str = "zenith"
    stream: bool = True
    max_tokens: int = 8192


@router.post("/chat/completions")
async def chat_completions(req: ChatRequest):
    """Chat with smart streaming — forwards SSE immediately, executes tools inline."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")
    raw = [m.model_dump() for m in req.messages]
    msgs = inject_system(raw, req.tier)

    # Auto RAG
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

    async def smart_stream():
        """Stream with inline tool execution.

        Phase 1: Stream LLM response with tools. Forward all content/reasoning
        chunks immediately. Buffer tool_calls if any.

        Phase 2: If tool_calls were made, execute tools and stream the
        follow-up response (no tools). The frontend sees a continuous stream.
        """
        async with httpx.AsyncClient(timeout=180) as client:

            # ── Phase 1: stream with tools ─────────────────────
            payload1 = {
                "model": req.model or MODEL,
                "messages": msgs,
                "max_tokens": req.max_tokens,
                "stream": True,
                "tools": TOOLS,
            }

            tool_calls_buffer: dict[int, dict] = {}
            finish_reason: str | None = None

            async with client.stream(
                "POST", f"{BASE_URL}/chat/completions",
                json=payload1, headers=headers,
            ) as resp:
                buf = ""
                async for chunk in resp.aiter_bytes():
                    text = chunk.decode()
                    buf += text
                    lines = buf.split("\n")
                    buf = lines.pop() or ""

                    for line in lines:
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            yield (line + "\n").encode()
                            continue

                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if not choices:
                                yield (line + "\n").encode()
                                continue

                            delta = choices[0].get("delta", {})
                            finish_reason = choices[0].get("finish_reason")

                            # Buffer tool_calls for later execution
                            tool_calls_delta = delta.get("tool_calls")
                            if tool_calls_delta:
                                for tc in tool_calls_delta:
                                    idx = tc.get("index", 0)
                                    if idx not in tool_calls_buffer:
                                        tool_calls_buffer[idx] = {
                                            "id": "",
                                            "function": {"name": "", "arguments": ""},
                                        }
                                    if tc.get("id"):
                                        tool_calls_buffer[idx]["id"] = tc["id"]
                                    if tc.get("function", {}).get("name"):
                                        tool_calls_buffer[idx]["function"]["name"] = tc["function"]["name"]
                                    if tc.get("function", {}).get("arguments"):
                                        tool_calls_buffer[idx]["function"]["arguments"] += tc["function"]["arguments"]

                                # Forward to frontend WITHOUT tool_calls data
                                # (frontend only handles content/reasoning)
                                clean_delta = {
                                    k: v for k, v in delta.items() if k != "tool_calls"
                                }
                                clean_data = {
                                    **data,
                                    "choices": [{**choices[0], "delta": clean_delta}],
                                }
                                yield f"data: {json.dumps(clean_data)}\n\n".encode()
                            else:
                                yield (line + "\n").encode()

                        except json.JSONDecodeError:
                            yield (line + "\n").encode()

            # ── Phase 2: tool execution + follow-up stream ────
            if tool_calls_buffer and finish_reason == "tool_calls":
                # Reconstruct the assistant message with full tool_calls
                tool_calls_list = []
                for idx in sorted(tool_calls_buffer):
                    tc = tool_calls_buffer[idx]
                    tool_calls_list.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    })

                # Append assistant message (with tool calls)
                assistant_msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls_list,
                }
                msgs.append(assistant_msg)

                # Execute each tool and append results
                for tc_data in tool_calls_list:
                    fn = tc_data["function"]
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    result = await exec_tool(fn["name"], args)
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc_data["id"],
                        "content": result,
                    })

                # Follow-up stream (no tools)
                payload2 = {
                    "model": req.model or MODEL,
                    "messages": msgs,
                    "max_tokens": req.max_tokens,
                    "stream": True,
                }
                async with client.stream(
                    "POST", f"{BASE_URL}/chat/completions",
                    json=payload2, headers=headers,
                ) as resp2:
                    async for chunk in resp2.aiter_bytes():
                        yield chunk

    if req.stream:
        return StreamingResponse(
            smart_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming: single call with tools (fast path) ──
    async with httpx.AsyncClient(timeout=180) as client:
        payload = {
            "model": req.model or MODEL,
            "messages": msgs,
            "max_tokens": req.max_tokens,
            "stream": False,
            "tools": TOOLS,
        }
        resp = await client.post(
            f"{BASE_URL}/chat/completions", json=payload, headers=headers
        )
        if not resp.is_success:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        msg = data.get("choices", [{}])[0].get("message", {})

        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                result = await exec_tool(fn.get("name", ""), args)
                msgs.append(msg)
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result,
                })

            # Follow-up non-streaming
            payload2 = {
                "model": req.model or MODEL,
                "messages": msgs,
                "max_tokens": req.max_tokens,
                "stream": False,
            }
            resp2 = await client.post(
                f"{BASE_URL}/chat/completions", json=payload2, headers=headers,
            )
            if not resp2.is_success:
                raise HTTPException(status_code=resp2.status_code, detail=resp2.text)
            return resp2.json()

        return data
