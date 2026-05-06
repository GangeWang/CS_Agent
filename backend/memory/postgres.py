"""PostgreSQL memory adapter placeholder.

The current backend stores active WebSocket conversation state in an in-process
TTL cache (`backend.tools.memory`). This module reserves the persistence
boundary for future PostgreSQL-backed long-term memory.
"""
from __future__ import annotations

from typing import Iterable

from backend.memory.models import ConversationMessage


class PostgresMemoryStore:
    def save_messages(self, messages: Iterable[ConversationMessage]) -> None:
        raise NotImplementedError("PostgreSQL memory persistence is not configured yet.")
