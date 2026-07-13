# Kyrin API

FastAPI backend for the Kyrin AI chat system — LLM proxy with two-round function calling, web search, RAG pipeline, anime identification, and SQLite chat storage.

## Architecture

```
┌──────────────┐     ┌───────────────┐     ┌─────────────────┐
│  Frontend    │────▶│  Kyrin API    │────▶│  OpenCode Zen   │
│  (React)     │     │  (FastAPI)    │     │  (LLM Provider) │
│  :5270       │◀────│  :5271        │◀────│  api.opencode.ai│
└──────────────┘     └───────┬───────┘     └─────────────────┘
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
| POST | `/api/chat/completions` | Two-round chat: tools then streaming response |
| GET | `/api/search?q=...` | Web search (SearXNG + DuckDuckGo fallback) |
| POST | `/api/crawl` | URL crawling |
| GET/POST | `/api/anime-search` | Anime identification via trace.moe |
| POST | `/api/rag/ingest` | Upload document (PDF/TXT/MD) |
| POST | `/api/rag/ingest-text` | Ingest raw text |
| POST | `/api/rag/query` | Query RAG documents |
| GET | `/api/rag/documents` | List ingested documents |
| DELETE | `/api/rag/documents/{filename}` | Delete a document |
| GET | `/api/chats` | List chats (lightweight: title + count only) |
| GET | `/api/chats/{id}` | Get full chat with messages |
| POST | `/api/chats` | Save/update chat |
| DELETE | `/api/chats/{id}` | Delete chat |
| GET | `/api/youtube/transcript` | YouTube transcript fetcher |
| GET | `/api/health` | Health check |

## Features

- **Two-Round Function Calling**: Round 1 (non-streaming, with tools) → execute tools → Round 2 (streaming, no tools) → final answer
- **3 Model Tiers**: Dawn, Zenith, Dusk — tier-specific system prompts
- **RAG Pipeline**: ChromaDB + chunking + embedding + vector search (PDF/TXT/MD)
- **Chat Persistence**: SQLite backend (`~/.kyrin/chats.db`)
- **Web Search**: SearXNG (primary) + DuckDuckGo (fallback)
- **Anime ID**: trace.moe API integration
- **Image Vision**: base64 image support in chat
- **Smart RAG Filtering**: Skips Kyrin-self documents from context injection

## Project Structure

```
kyrin-api/
├── main.py                 # Entry point (uvicorn)
├── app/
│   ├── main.py             # FastAPI app factory + CORS + router mounts
│   ├── config.py           # Pydantic settings (from .env)
│   ├── database.py         # SQLite connection manager
│   ├── services/
│   │   ├── chat.py         # System prompts, tools, streaming, injection
│   │   └── rag.py          # RAG engine: chunking, parsing, ingest, query
│   └── routers/
│       ├── chat.py         # /api/chat/completions (two-round)
│       ├── chats.py        # /api/chats CRUD (SQLite)
│       ├── search.py       # Web search (SearXNG + DDG)
│       ├── rag.py          # RAG API endpoints (thin wrappers)
│       ├── anime.py        # Anime identification
│       ├── crawl.py        # URL crawling
│       └── youtube.py      # YouTube transcript
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## Quick Start

```bash
# 1. Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit with API keys

# 2. Start
uvicorn app.main:app --host 0.0.0.0 --port 5271 --reload

# 3. Verify
curl http://localhost:5271/api/health
# → {"status":"ok","service":"kyrin-api","version":"1.0.0",...}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KYRIN_API_KEY` | — | OpenCode API key |
| `KYRIN_BASE_URL` | `https://opencode.ai/zen/go/v1` | LLM API endpoint |
| `KYRIN_MODEL` | `deepseek-v4-flash` | Default model |
| `KYRIN_CHATS_DIR` | `~/.kyrin/chats` | SQLite db parent dir |
| `KYRIN_RAG_DIR` | `~/.kyrin/rag` | ChromaDB persistent dir |
| `SEARXNG_URL` | `http://127.0.0.1:8080` | SearXNG instance |
| `TRACE_MOE_URL` | `https://api.trace.moe` | Anime search API |
| `PORT` | `5271` | Server port |

## Two-Round Chat Flow

```
Request: stream=true
  │
  ├─ Round 1 (non-streaming, tools enabled)
  │   POST /chat/completions {stream:false, tools:[search_web,crawl_url]}
  │   │
  │   ├─ LLM responds with text → skip to Round 2
  │   └─ LLM calls search_web/crawl_url → backend executes → append results
  │
  └─ Round 2 (streaming, no tools)
      POST /chat/completions {stream:true}
      │
      └─ Forward SSE stream to client
```

## Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5271 --workers 4
```
