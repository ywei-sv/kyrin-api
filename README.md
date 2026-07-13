# Kyrin API

FastAPI backend harness for Kyrin AI agent — connects chat, web search,
web crawling, anime screenshot search, and image analysis.

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit with your keys
uvicorn main:app --reload --port 5271
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Chat completion (SSE stream) with tool calling |
| GET | `/api/tools/search?q=...` | Web search via SearXNG |
| POST | `/api/tools/crawl` | Crawl URL via Crawl4AI |
| GET | `/api/tools/anime?url=...` | Anime screenshot search |
| POST | `/api/tools/vision` | Image analysis |
| GET | `/health` | Health check |

## Frontend

Frontend is at [kyrin-landing](https://github.com/ywei-sv/document-hub).

Vite config proxies `/api/*` → `http://localhost:5271`.
