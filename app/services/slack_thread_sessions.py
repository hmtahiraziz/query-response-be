"""In-memory Slack thread → last generate context (channel threads only; single-process)."""

from __future__ import annotations

import time
from dataclasses import dataclass

TTL_SECONDS = 7 * 24 * 3600  # 7 days

_store: dict[str, "SlackThreadSession"] = {}


@dataclass
class SlackThreadSession:
    client_query: str
    cover_letter: str
    updated_at: float


def _key(channel_id: str, thread_root_ts: str) -> str:
    return f"{channel_id}:{thread_root_ts}"


def _prune_expired() -> None:
    now = time.time()
    dead = [k for k, v in _store.items() if now - v.updated_at > TTL_SECONDS]
    for k in dead:
        del _store[k]


def put_session(channel_id: str, thread_root_ts: str, client_query: str, cover_letter: str) -> None:
    _prune_expired()
    _store[_key(channel_id, thread_root_ts)] = SlackThreadSession(
        client_query=client_query,
        cover_letter=cover_letter,
        updated_at=time.time(),
    )


def get_session(channel_id: str, thread_root_ts: str) -> SlackThreadSession | None:
    _prune_expired()
    k = _key(channel_id, thread_root_ts)
    s = _store.get(k)
    if s is None:
        return None
    if time.time() - s.updated_at > TTL_SECONDS:
        del _store[k]
        return None
    return s


def update_session_letter(channel_id: str, thread_root_ts: str, cover_letter: str) -> None:
    k = _key(channel_id, thread_root_ts)
    if k in _store:
        _store[k].cover_letter = cover_letter
        _store[k].updated_at = time.time()
