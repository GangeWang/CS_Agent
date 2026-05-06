"""Persistent memory data models."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class ConversationMessage:
    session_id: str
    role: str
    content: str
    created_at: datetime
