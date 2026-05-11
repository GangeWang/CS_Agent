from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict

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


def list_tools() -> Dict[str, ToolSpec]:
    return dict(_TOOL_REGISTRY)


def get_tool(name: str) -> ToolSpec | None:
    return _TOOL_REGISTRY.get(name)


# ---- MVP Tools ----
def get_time() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize(text: str) -> str:
    return text[:200] + ("..." if len(text) > 200 else "")


def lookup_doc(query: str) -> str:
    # MVP stub，之後改成文件檢索
    return f"[doc] matched for query: {query}"


GET_TIME_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

SUMMARIZE_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}

LOOKUP_DOC_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


register_tool("get_time", "取得目前 UTC 時間", GET_TIME_SCHEMA, get_time)
register_tool("summarize", "摘要文本內容（截斷前 200 字）", SUMMARIZE_SCHEMA, summarize)
register_tool("lookup_doc", "查詢文件（MVP stub）", LOOKUP_DOC_SCHEMA, lookup_doc)
