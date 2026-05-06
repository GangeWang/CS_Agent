"""LLM tool facade used by the orchestrator."""
from backend.services.llama_client import request_stream_sync

__all__ = ["request_stream_sync"]
