from __future__ import annotations

from typing import Dict, List


def plan_steps(user_msg: str) -> List[Dict]:
    # MVP: 簡單 rule-based planner
    if "時間" in user_msg or "time" in user_msg.lower():
        return [
            {"step": 1, "action": "tool", "name": "get_time", "args": {}},
            {"step": 2, "action": "respond"},
        ]
    # 其餘問題：文件檢索還沒接上，不呼叫工具，直接回答。
    # （原本會把使用者原文丟給 summarize / lookup_doc，而這兩個 stub 會把原文原樣當成「工具結果」
    #  放進 system 指令，等於讓使用者的話升格成系統指令。）
    return [{"step": 1, "action": "respond"}]
