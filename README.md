# Kyrin API

FastAPI backend for the Kyrin AI chat system — LLM proxy, function calling, web search, RAG pipeline, anime identification, and chat storage.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Frontend   │────▶│  Kyrin API   │────▶│  OpenCode Zen   │
│  (React)    │     │  (FastAPI)   │     │  (LLM Provider) │
│  :5270      │◀────│  :5271       │◀────│  api.opencode.ai │
└─────────────┘     └──────┬───────┘     └─────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌─────────┐ ┌──────────┐ ┌──────────┐
         │SearXNG  │ │ChromaDB  │ │SQLite    │
         │:8080    │ │(vector)  │ │(chats)   │
         └─────────┘ └──────────┘ └──────────┘
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat/completions` | Chat + function calling (SSE streaming) |
| GET | `/api/search?q=...` | Web search (SearXNG) |
| POST | `/api/crawl` | URL crawling |
| GET | `/api/anime-search?url=...` | Anime identification via trace.moe |
| POST | `/api/rag/ingest` | Upload document (PDF/TXT/MD) |
| POST | `/api/rag/query` | Query RAG documents |
| GET | `/api/chats` | List saved chats |
| POST | `/api/chats` | Save/update chat |
| GET | `/api/chats/{id}` | Get chat by ID |
| DELETE | `/api/chats/{id}` | Delete chat |
| GET | `/api/health` | Health check |

## Features

- **3 Model Tiers**: Dawn (deepseek-v4-flash), Zenith (mimo-v2.5), Dusk (qwen3.7-plus)
- **Function Calling**: `search_web(query)`, `crawl_url(url)` — model calls tools automatically
- **RAG Pipeline**: ChromaDB + chunking + embedding + vector search (PDF/TXT/MD)
- **Chat Persistence**: SQLite backend (`~/.kyrin/chats.db`)
- **Web Search**: SearXNG (self-hosted)
- **Anime ID**: trace.moe API integration
- **Image Vision**: base64 image support in chat
- **Auto Tool Execution**: model decides → backend executes → final answer

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ywei-sv/kyrin-api.git
cd kyrin-api

# 2. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 3. Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Start server
uvicorn main:app --host 0.0.0.0 --port 5271 --reload

# 5. Verify
curl http://localhost:5271/api/health
# → {"status":"ok","service":"kyrin-api","version":"1.0.0"}
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Type check
mypy main.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KYRIN_API_KEY` | — | OpenCode API key |
| `KYRIN_BASE_URL` | `https://opencode.ai/zen/go/v1` | LLM API endpoint |
| `KYRIN_MODEL` | `deepseek-v4-flash` | Default model |
| `SEARXNG_URL` | `http://localhost:8080` | SearXNG instance |
| `TRACE_MOE_URL` | `https://api.trace.moe` | Anime search API |
| `PORT` | `5271` | Server port |

## Project Structure

```
kyrin-api/
├── main.py              # FastAPI app + CORS + router mounts (48 lines)
├── requirements.txt     # Python dependencies
├── .env                 # Local environment (gitignored)
├── .env.example         # Environment template
├── routers/
│   ├── chat.py          # Chat completions (function calling, SSE streaming)
│   ├── search.py        # Web search (SearXNG + DuckDuckGo fallback)
│   ├── crawl.py         # URL crawling
│   ├── anime.py         # Anime identification
│   ├── rag.py           # RAG engine (ChromaDB, chunking, embedding)
│   ├── rag_api.py       # RAG API endpoints
│   ├── chats.py         # Chat persistence (SQLite)
│   └── __init__.py
└── README.md
```

## Production

```bash
# Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 5271 --workers 4

# Or using gunicorn with uvicorn workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:5271
```

## License

Private — All rights reserved.
