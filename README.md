# Kyrin API

FastAPI backend for the Kyrin chat system — LLM proxy, web search, chat storage.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat/completions` | Chat completion (SSE stream) with tier system prompts |
| GET | `/api/search` | Web search (SearXNG / DuckDuckGo) |
| GET | `/api/chats` | List saved chats |
| POST | `/api/chats` | Save/update chat |
| DELETE | `/api/chats/{id}` | Delete chat |
| GET | `/api/health` | Health check |

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit with your keys
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Frontend

Frontend at **[kyrin-landing](https://github.com/ywei-sv/kyrin-landing)**.
See the [README](https://github.com/ywei-sv/kyrin-landing/blob/main/README.md) for full architecture & features.
