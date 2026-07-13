# Kyrin API — FastAPI backend harness for AI tools
#
# Endpoints:
#   POST /api/chat            — Chat completion (SSE streaming) + tool calling
#   GET  /api/tools/search    — Web search via SearXNG
#   POST /api/tools/crawl     — Crawl a URL via Crawl4AI
#   GET  /api/tools/anime     — Anime screenshot search via trace.moe
#   GET  /api/tools/vision    — Image analysis (uses vision model)
#
# Run: uvicorn main:app --reload --port 5271

import os, json, asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

load_dotenv()

OPENCODE_KEY = os.getenv('OPENCODE_API_KEY', '')
OPENCODE_BASE = os.getenv('OPENCODE_API_BASE', 'https://opencode.ai/zen/go/v1')
SEARXNG_BASE = os.getenv('SEARXNG_BASE', 'http://localhost:9999')
ANIME_API = os.getenv('ANIME_TRACE_API', 'https://api.trace.moe')

# ─── app ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=120.0)
    yield
    await app.state.client.aclose()

app = FastAPI(title='Kyrin API', version='0.1.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# ─── helpers ────────────────────────────────────────────────────────
def sys_msg(text: str) -> dict:
    return {'role': 'system', 'content': text}

def build_headers() -> dict:
    h = {'Content-Type': 'application/json'}
    if OPENCODE_KEY:
        h['Authorization'] = f'Bearer {OPENCODE_KEY}'
    return h

SYSTEM_PROMPT = sys_msg(
    'You are Kyrin, an AI assistant. You have access to tools:\n'
    '- Web search (triggered when user asks "search: <query>")\n'
    '- Anime screenshot search (when user asks about anime scenes)\n'
    '- URL crawling (when user asks to analyze a webpage)\n'
    '- Image analysis (when user sends an image)\n'
    'Be helpful, concise, and accurate. When the user asks in Thai, answer in Thai. '
    'If you receive an image you cannot read, say so. If you can identify an anime screenshot, do so.'
)

# ─── chat (SSE streaming) ───────────────────────────────────────────
@app.post('/api/chat')
async def chat(req: Request):
    body = await req.json()
    messages: list = body.get('messages', [])

    # Inject system prompt
    if not any(m.get('role') == 'system' for m in messages):
        messages.insert(0, SYSTEM_PROMPT)

    # Tool detection — for now: pass-through to OpenCode
    upstream_messages = []
    for m in messages:
        role = m.get('role', 'user')
        content = m.get('content', '')
        if isinstance(content, list):
            # content array (text + images) — pass through for vision models
            upstream_messages.append({'role': role, 'content': content})
        else:
            upstream_messages.append({'role': role, 'content': content})

    # Determine model: if any message has images, use minimax-m3 (vision)
    has_images = any(
        isinstance(m.get('content'), list) and any(p.get('type') == 'image_url' for p in m['content'])
        for m in messages
    )
    model = 'minimax-m3' if has_images else 'deepseek-v4-flash'

    payload = {
        'model': model,
        'messages': upstream_messages,
        'stream': True,
    }

    client: httpx.AsyncClient = req.app.state.client
    upstream_resp = await client.post(
        f'{OPENCODE_BASE}/chat/completions',
        headers=build_headers(),
        json=payload,
    )
    upstream_resp.raise_for_status()

    async def stream():
        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk
        except Exception:
            yield b'data: [DONE]\n\n'

    return StreamingResponse(
        stream(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*',
        },
    )

# ─── tools: web search via SearXNG ──────────────────────────────────
@app.get('/api/tools/search')
async def tool_search(q: str, language: str = 'th', max_results: int = 8):
    client: httpx.AsyncClient = app.state.client
    try:
        resp = await client.get(f'{SEARXNG_BASE}/search', params={
            'q': q, 'format': 'json', 'language': language,
        }, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        results = data.get('results', [])[:max_results]
        return JSONResponse({
            'query': q,
            'results': [{
                'title': r.get('title', ''),
                'url': r.get('url', ''),
                'content': r.get('content', '')[:300],
            } for r in results],
        })
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=502)

# ─── tools: crawl via Crawl4AI ──────────────────────────────────────
@app.post('/api/tools/crawl')
async def tool_crawl(body: dict):
    url = body.get('url', '')
    if not url:
        return JSONResponse({'error': 'url required'}, status_code=400)

    try:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
        from crawl4ai.browser_manager import BrowserConfig
    except ImportError:
        return JSONResponse({'error': 'crawl4ai not installed'}, status_code=501)

    try:
        config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
            result = await crawler.arun(url=url, config=config)
        text = (result.markdown or result.html or '')[:8000]
        return JSONResponse({'url': url, 'content': text, 'length': len(text)})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=502)

# ─── tools: anime screenshot search via trace.moe ───────────────────
@app.get('/api/tools/anime')
async def tool_anime_search(url: str = '', cut_borders: bool = True):
    """Search anime by image URL. Returns top matches."""
    if not url:
        return JSONResponse({'error': 'url parameter required (image URL)'}, status_code=400)

    try:
        client: httpx.AsyncClient = app.state.client
        params = {'url': url, 'cutBorders': str(cut_borders).lower()}
        resp = await client.get(f'{ANIME_API}/search', params=params, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        # Return top 3 results
        results = []
        for r in (data.get('result', [])[:3]):
            anilist_id = r.get('anilist', 0)
            results.append({
                'anilist_id': anilist_id,
                'anime': r.get('filename', '').split('/')[0] if '/' in (r.get('filename', '')) else r.get('filename', ''),
                'episode': r.get('episode'),
                'similarity': round(r.get('similarity', 0) * 100, 1),
                'from': round(r.get('from', 0), 1),
                'to': round(r.get('to', 0), 1),
            })
        return JSONResponse({'results': results, 'raw_count': len(data.get('result', []))})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=502)

# ─── tools: image analysis ──────────────────────────────────────────
@app.post('/api/tools/vision')
async def tool_vision(body: dict):
    """Analyze an image using the vision model."""
    image_url = body.get('image_url', '') or body.get('url', '')
    prompt = body.get('prompt', 'Describe this image in detail.')

    if not image_url:
        return JSONResponse({'error': 'image_url required'}, status_code=400)

    messages = [
        sys_msg('You are an image analysis assistant. Describe images accurately.'),
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': image_url}},
            ],
        },
    ]

    payload = {'model': 'minimax-m3', 'messages': messages, 'stream': False}

    client: httpx.AsyncClient = app.state.client
    try:
        resp = await client.post(
            f'{OPENCODE_BASE}/chat/completions',
            headers=build_headers(),
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        return JSONResponse({'description': content})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=502)

# ─── health ─────────────────────────────────────────────────────────
@app.get('/health')
async def health():
    return {'status': 'ok', 'service': 'kyrin-api', 'version': '0.1.0'}

# ─── entry ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=5271, reload=True)
