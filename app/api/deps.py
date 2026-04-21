"""FastAPI dependencies shared by route modules."""

from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings


def _settings() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(_settings)]
