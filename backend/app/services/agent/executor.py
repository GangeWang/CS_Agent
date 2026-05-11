from __future__ import annotations

from typing import Any, Dict, List

from .tools import get_tool


def _validate_args(schema: Dict[str, Any], args: Dict[str, Any]) -> str | None:
    if not isinstance(args, dict):
        return "args must be object"

    required = schema.get("required", [])
    for key in required:
        if key not in args:
            return f"missing required arg: {key}"

    properties = schema.get("properties", {})
    for key, spec in properties.items():
        if key not in args:
            continue
        expected_type = spec.get("type")
        if expected_type == "string" and not isinstance(args[key], str):
            return f"arg {key} must be string"

    if not schema.get("additionalProperties", True):
        extras = [key for key in args.keys() if key not in properties]
        if extras:
            return f"unexpected args: {', '.join(extras)}"

    return None


def execute_plan(plan: List[Dict]) -> List[Dict]:
    results: List[Dict[str, Any]] = []
    for step in plan:
        if step.get("action") != "tool":
            continue
        name = step.get("name")
        args = step.get("args", {})
        tool = get_tool(name) if isinstance(name, str) else None
        if tool is None:
            results.append({"step": step.get("step"), "tool": name, "error": "tool not found"})
            continue

        validation_error = _validate_args(tool.schema, args)
        if validation_error:
            results.append({"step": step.get("step"), "tool": name, "error": validation_error})
            continue

        try:
            output = tool.fn(**args)
            results.append({"step": step.get("step"), "tool": name, "output": output})
        except Exception as e:
            results.append({"step": step.get("step"), "tool": name, "error": str(e)})
    return results
