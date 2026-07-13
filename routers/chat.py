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
