"""
Chat business logic — system prompts, tool definitions, tool execution, and streaming.
"""

import os
import json
import re
from typing import AsyncGenerator

import httpx

from app.config import settings

API_KEY = os.environ.get("KYRIN_API_KEY", settings.kyrin_api_key)
BASE_URL = os.environ.get("KYRIN_BASE_URL", settings.kyrin_base_url)
MODEL = os.environ.get("KYRIN_MODEL", settings.kyrin_model)

SYSTEM_PROMPTS = {
    "dawn": (
        "You are Kyrin Dawn, powered by DeepSeek V4 Flash — fast and precise.\n\n"
        "**Response style:**\n"
        "- Respond in **Thai** with clean, scannable markdown\n"
        "- Use **emoji** per section (🌸🗼🍜🎯📌)\n"
        "- **Bold** for key terms and numbers\n"
        "- Bullet points for lists, short paragraphs (1–3 sentences)\n"
        "- ### for section headings\n\n"
        "**Citations:** Only if the conversation contains search results or documents. "
        "Otherwise answer from your own knowledge — do NOT mention documents, sources, or citations.\n\n"
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
        "**Citations:** Only if the conversation contains search results or documents. "
        "Otherwise answer from your own knowledge — do NOT mention documents, sources, or citations.\n\n"
        "**Quality:** Comprehensive, conversational, and engaging. Think like a knowledgeable travel guide or consultant."
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
        "**Citations:** Only if the conversation contains search results or documents. "
        "Otherwise answer from your own knowledge — do NOT mention documents, sources, or citations.\n\n"
        "**Quality:** Expert-level depth. Read like a well-researched article. Be thorough, nuanced, and insightful."
    ),
}

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


async def exec_tool(name: str, args: dict) -> str:
    """Execute a tool and return the result as a string."""
    try:
        if name == "search_web":
            query = args.get("query", "")
            if not query:
                return "Error: No query provided"
            async with httpx.AsyncClient(timeout=15) as client:
                searxng = os.environ.get("SEARXNG_URL", settings.searxng_url)
                resp = await client.get(
                    f"{searxng}/search",
                    params={"q": query, "format": "json", "language": "th"},
                )
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
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return f"Content from {url} (first {len(text)} chars):\n\n{text[:4000]}"
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error executing {name}: {str(e)}"


async def stream_proxy(payload: dict) -> AsyncGenerator[bytes, None]:
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


def inject_system(messages: list[dict], tier: str) -> list[dict]:
    """Prepend system prompt based on model tier."""
    prompt = SYSTEM_PROMPTS.get(tier, SYSTEM_PROMPTS["zenith"])
    return [{"role": "system", "content": prompt}, *messages]
