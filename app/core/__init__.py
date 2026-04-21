"""Cross-cutting application configuration and lifecycle."""

from app.core.config import (
    COVER_LETTER_HISTORY_MAX,
    COVER_LETTER_HISTORY_PATH,
    DATA_DIR,
    MANIFEST_PATH,
    PDF_DIR,
    PROJECT_ROOT,
    Settings,
    ensure_data_dirs,
    get_settings,
)

__all__ = [
    "COVER_LETTER_HISTORY_MAX",
    "COVER_LETTER_HISTORY_PATH",
    "DATA_DIR",
    "MANIFEST_PATH",
    "PDF_DIR",
    "PROJECT_ROOT",
    "Settings",
    "ensure_data_dirs",
    "get_settings",
]
