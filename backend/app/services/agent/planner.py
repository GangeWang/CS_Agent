from __future__ import annotations

from typing import Dict, List


def plan_steps(user_msg: str) -> List[Dict]:
    # MVP: 簡單 rule-based planner
    if "時間" in user_msg or "time" in user_msg.lower():
        return [
            {"step": 1, "action": "tool", "name": "get_time", "args": {}},
            {"step": 2, "action": "respond"},
        ]
    if "摘要" in user_msg:
        return [
            {"step": 1, "action": "tool", "name": "summarize", "args": {"text": user_msg}},
            {"step": 2, "action": "respond"},
        ]
    return [
        {"step": 1, "action": "tool", "name": "lookup_doc", "args": {"query": user_msg}},
        {"step": 2, "action": "respond"},
    ]
