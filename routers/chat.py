"""
LLM chat completions — proxies to opencode-go with streaming + vision support.
Auto-injects RAG context when documents are available.
"""

import os
import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

from routers.rag import build_rag_context

router = APIRouter()

API_KEY = os.environ.get("KYRIN_API_KEY", "")
BASE_URL = os.environ.get("KYRIN_BASE_URL", "https://opencode.ai/zen/go/v1")
MODEL = os.environ.get("KYRIN_MODEL", "deepseek-v4-flash")

SYSTEM_PROMPTS = {
    "dawn": (
        "You are Kyrin Dawn, powered by DeepSeek V4 Flash — fast and precise.\n\n"
        "**Response style:**\n"
        "- Respond in **Thai** with clean, scannable markdown\n"
        "- Use **emoji** per section (🌸🗼🍜🎯📌)\n"
        "- **Bold** for key terms and numbers\n"
        "- Bullet points for lists, short paragraphs (1–3 sentences)\n"
        "- ### for section headings\n\n"
        "**Citations:**\n"
        "- Inline: [source text](url)\n"
        "- End with: 📚 **Sources**  \n"
        "  [1](url) — Description  \n"
        "  [2](url) — Description\n\n"
        "**Quality:** Efficient but complete. Every line adds value. No fluff, no greetings."
    ),
    "zenith": (
        "You are Kyrin Zenith, powered by MiMo V2.5 — a balanced, agent-capable AI.\n\n"
        "**Response style:**\n"
        "- Respond in **Thai** with polished, well-structured markdown\n"
        "- ### emoji headings (🗾 **Overview**, 🏙️ **Cities**, 📊 **Comparison**)\n"
        "- **Bold** highlights, bullet lists, and short paragraphs (2–4 sentences)\n"
        "- Use `---` horizontal rules between major sections\n"
        "- Blockquotes > for tips or key takeaways\n"
        "- Tables for comparisons where useful\n\n"
        "**Citations:**\n"
        "- Inline: [source text](url) with [1][2] markers\n"
        "- End with:\n"
        "  ---\n"
        "  ### 📚 Sources\n"
        "  [1](url) — Description  \n"
        "  [2](url) — Description\n\n"
        "**Quality:** Comprehensive, conversational, and engaging. Think like a knowledgeable travel guide or consultant — warm, detailed, and trustworthy."
    ),
    "dusk": (
        "You are Kyrin Dusk, powered by Qwen 3.7 Plus — a deep-reasoning AI at Claude Opus level.\n\n"
        "**Response style:**\n"
        "- Respond in **Thai** with rich, publication-ready markdown\n"
        "- ### emoji headings with short intros (🗾 **ภาพรวม**, 🌸 **ไฮไลท์**, 📊 **เปรียบเทียบ**)\n"
        "- **Bold** + bullet lists + short paragraphs + `---` separators\n"
        "- > blockquotes for insights or expert tips\n"
        "- Tables for data comparisons\n"
        "- Use `---` between major sections for readability\n\n"
        "**Reasoning:**\n"
        "- For complex questions, break down your thinking:\n"
        "  1. Key factors to consider\n"
        "  2. Compare options with evidence\n"
        "  3. Give a clear recommendation\n"
        "- Be nuanced — acknowledge tradeoffs, pros/cons\n"
        "- Think step-by-step **only when the topic is complex**\n\n"
        "**Citations:**\n"
        "- Inline: [source text](url) with [1][2][3] markers\n"
        "- End with:\n"
        "  ---\n"
        "  ### 📚 Sources\n"
        "  [1](url) — Description  \n"
        "  [2](url) — Description\n\n"
        "**Quality:** Expert-level depth. Read like a well-researched article. Be thorough, nuanced, and insightful. Match the depth a human expert would provide."
    ),
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


# ── Function Calling Tools ────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information. Use when user asks about news, facts, or recent events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crawl_url",
            "description": "Fetch and extract content from a URL. Use when user wants to analyze a webpage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to crawl"}
                },
                "required": ["url"],
            },
        },
    },
]


async def _exec_tool(name: str, args: dict) -> str:
    """Execute a tool and return the result as a string."""
    try:
        if name == "search_web":
            query = args.get("query", "")
            if not query:
                return "Error: No query provided"
            async with httpx.AsyncClient(timeout=15) as client:
                searxng = os.environ.get("SEARXNG_URL", "http://localhost:8080")
                resp = await client.get(f"{searxng}/search", params={"q": query, "format": "json", "language": "th"})
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])[:5]
                if not results:
                    return f"No results found for '{query}'."
                lines = []
                for r in results:
                    title = r.get("title", "")
                    url = r.get("url", "")
                    snippet = (r.get("content", "") or "")[:300]
                    lines.append(f"- {title}\n  URL: {url}\n  {snippet}")
                return f"Web search results for '{query}':\n\n" + "\n\n".join(lines)
        elif name == "crawl_url":
            url = args.get("url", "")
            if not url:
                return "Error: No URL provided"
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                text = resp.text[:8000]
                import re
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return f"Content from {url} (first {len(text)} chars):\n\n{text[:4000]}"
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error executing {name}: {str(e)}"


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

    # Auto RAG — query documents and inject context if found
    last_user_msg = None
    for m in reversed(raw):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            last_user_msg = m["content"]
            break
    rag_context, rag_sources = build_rag_context(last_user_msg or "")
    if rag_context:
        msgs.insert(0, {"role": "system", "content": rag_context})

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
            # Execute tools and append results
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                result = await _exec_tool(name, args)
                msgs.append(msg)
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})

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
                raise HTTPException(status_code=resp2.status_code, detail=resp2.text)
            return resp2.json()

        return data
