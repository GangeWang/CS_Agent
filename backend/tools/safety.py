"""Safety tool facade used by the orchestrator.

The guardrail policy loads ML dependencies, so import it lazily to keep the API
module importable in lightweight environments and during syntax checks.
"""
from __future__ import annotations

from typing import Any, Dict

from backend.orchestrator.planner import build_guardrail_instruction


def classify_text(text: str) -> Dict[str, Any]:
    from backend.guard.policy import classify_text as _classify_text

    return _classify_text(text)


__all__ = ["classify_text", "build_guardrail_instruction"]
