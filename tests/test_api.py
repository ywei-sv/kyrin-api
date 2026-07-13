"""
Basic tests for Kyrin API.
Run with: pytest tests/ -v
"""

import os
import sys
import json
from pathlib import Path

# Ensure app module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test env vars before importing app
os.environ["KYRIN_API_KEY"] = "test-key"
os.environ["KYRIN_MODEL"] = "test-model"
os.environ["SEARXNG_URL"] = "http://localhost:9999"  # won't be reached

from fastapi.testclient import TestClient

# Import app after env is set
from app.main import app

client = TestClient(app)


class TestHealth:
    def test_health_endpoint(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "kyrin-api"
        assert data["version"] == "1.0.0"

    def test_health_model(self):
        resp = client.get("/api/health")
        data = resp.json()
        assert data["model"] == "test-model"


class TestSearchRouter:
    def test_search_empty_query(self):
        resp = client.get("/api/search?q=")
        assert resp.status_code == 400

    def test_search_missing_query(self):
        resp = client.get("/api/search")
        assert resp.status_code == 422  # FastAPI validation

    def test_search_unknown_engine(self):
        resp = client.get("/api/search?q=test&engine=nonexistent")
        assert resp.status_code == 400
        assert "Unknown engine" in resp.text


class TestCrawlRouter:
    def test_crawl_invalid_url(self):
        resp = client.post("/api/crawl", json={"url": ""})
        # Falls through to jina.ai fallback which returns 422 for empty
        assert resp.status_code in (400, 422, 502)

    def test_crawl_missing_url(self):
        resp = client.post("/api/crawl", json={})
        assert resp.status_code == 422  # FastAPI validation


class TestChatRouter:
    def test_chat_validation(self):
        """Messages must be a non-empty array."""
        resp = client.post("/api/chat/completions", json={"messages": []})
        assert resp.status_code == 400
        assert "non-empty" in resp.json()["detail"]

    def test_chat_missing_messages(self):
        resp = client.post("/api/chat/completions", json={})
        assert resp.status_code == 422  # FastAPI validates required field


class TestAnimeRouter:
    def test_anime_no_image(self):
        """Missing both url and file should return 400."""
        resp = client.get("/api/anime-search")
        assert resp.status_code == 400

    def test_anime_empty_url(self):
        resp = client.get("/api/anime-search?url=")
        assert resp.status_code == 400


class TestChatsRouter:
    def test_list_chats_empty(self):
        resp = client.get("/api/chats")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_create_and_get_chat(self):
        chat_id = "test-chat-123"
        resp = client.post("/api/chats", json={
            "id": chat_id,
            "title": "Test Chat",
            "messages": [{"role": "user", "content": "Hello"}],
            "updatedAt": 1000000,
        })
        assert resp.status_code == 200
        assert resp.json()["id"] == chat_id

        # Get it back via detail endpoint
        resp = client.get(f"/api/chats/{chat_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == chat_id
        assert resp.json()["title"] == "Test Chat"
        assert len(resp.json()["messages"]) == 1

        # List should show msgCount, not full messages
        resp = client.get("/api/chats")
        assert resp.status_code == 200
        listed = resp.json()
        found = [c for c in listed if c["id"] == chat_id]
        assert len(found) == 1
        assert found[0]["msgCount"] == 1
        assert "messages" not in found[0] or found[0].get("messages") is None

        # Delete it
        resp = client.delete(f"/api/chats/{chat_id}")
        assert resp.status_code == 200

        # Verify gone
        resp = client.get(f"/api/chats/{chat_id}")
        assert resp.status_code == 404

    def test_chat_with_metadata(self):
        chat_id = "test-chat-meta"
        resp = client.post("/api/chats", json={
            "id": chat_id,
            "title": "Meta Chat",
            "messages": [],
            "updatedAt": 1000000,
            "metadata": {"tier": "zenith"},
        })
        assert resp.status_code == 200

        resp = client.get(f"/api/chats/{chat_id}")
        data = resp.json()
        assert data["metadata"] == {"tier": "zenith"}

        client.delete(f"/api/chats/{chat_id}")


class TestModels:
    def test_models_endpoint(self):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) > 0

    def test_models_fallback(self):
        # Should always return at least the static fallback models
        resp = client.get("/api/models")
        models = resp.json()["models"]
        ids = [m["id"] for m in models]
        assert "deepseek-v4-flash" in ids


class TestCORS:
    def test_cors_headers(self):
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5270",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers


class TestConfig:
    def test_config_loaded(self):
        from app.config import settings
        assert settings.kyrin_api_key == "test-key"
        assert settings.kyrin_model == "test-model"
