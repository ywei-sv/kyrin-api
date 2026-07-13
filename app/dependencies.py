"""Shared FastAPI dependencies."""
from __future__ import annotations

from app.config import settings


def get_settings() -> SettingsProxy:
    """Return a dict-like proxy of current settings."""
    return SettingsProxy()


class SettingsProxy:
    """Read-only view of settings for route injection."""

    @property
    def model(self) -> str:
        return settings.kyrin_model

    @property
    def base_url(self) -> str:
        return settings.kyrin_base_url

    @property
    def api_key(self) -> str:
        return settings.kyrin_api_key
