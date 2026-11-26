from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Callable, Dict
from datetime import datetime, timezone, timedelta
ToolFn = Callable[..., Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: Dict[str, Any]
    fn: ToolFn


_TOOL_REGISTRY: Dict[str, ToolSpec] = {}


def register_tool(name: str, description: str, schema: Dict[str, Any], fn: ToolFn) -> None:
    _TOOL_REGISTRY[name] = ToolSpec(name=name, description=description, schema=schema, fn=fn)


def get_tool(name: str) -> ToolSpec | None:
    return _TOOL_REGISTRY.get(name)


# ---- MVP Tools ----
def get_time() -> str:
    now = datetime.now(timezone(timedelta(hours=8)))
    return f"現在時間為\n{now.strftime('%Y.%m.%d %H:%M:%S')}"


def lookup_doc(query: str) -> str:
    # MVP stub，之後改成文件檢索。不可把 query（使用者原文）放進回傳值：工具結果會進 system 指令
    return "（文件檢索尚未接上，沒有找到相關文件）"


GET_TIME_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

LOOKUP_DOC_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


register_tool("get_time", "取得目前 UTC 時間", GET_TIME_SCHEMA, get_time)
register_tool("lookup_doc", "查詢文件（MVP stub）", LOOKUP_DOC_SCHEMA, lookup_doc)
