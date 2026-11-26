"""
每個 WebSocket session 的對話歷史。

目前存在行程記憶體（TTLCache）：重啟就消失、多個 worker 之間不共用。資料庫建好之後，
只要換掉這個模組的實作（介面維持 create / get_history / set_history / append_turn / clear / drop），
routers/ws.py 不需要改。
"""
from __future__ import annotations

from typing import Dict, List

from cachetools import TTLCache

from ..config import settings
from .context_manager import is_summary_message

Message = Dict[str, str]

# 最多 1000 個同時進行的 session；一小時沒有任何寫入就過期
_sessions: TTLCache = TTLCache(maxsize=1000, ttl=3600)


def create(session_id: str) -> None:
    _sessions[session_id] = []


def get_history(session_id: str) -> List[Message]:
    """回傳副本，呼叫端改了也不會影響存放的內容。"""
    return list(_sessions.get(session_id, []))


def set_history(session_id: str, history: List[Message]) -> None:
    _sessions[session_id] = list(history)


def append_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """
    加入一輪問答，並套用 HISTORY_MAX_LENGTH 硬上限（只保留最新一則壓縮摘要 + 最近的對話）。
    送給模型的量另外由 context_manager 依 token 預算控制。
    """
    history = get_history(session_id)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})

    summaries = [m for m in history if is_summary_message(m)]
    dialogue = [m for m in history if not is_summary_message(m)]
    limit = max(2, int(settings.history_max_length))
    if len(dialogue) > limit:
        dialogue = dialogue[-limit:]
    _sessions[session_id] = summaries[-1:] + dialogue


def clear(session_id: str) -> None:
    _sessions[session_id] = []


def drop(session_id: str) -> None:
    _sessions.pop(session_id, None)
