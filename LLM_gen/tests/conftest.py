"""用假的 llama_cpp 模組測 server_llama.py（CI 不需要 GPU，也不需要編譯 llama-cpp-python）。"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeLlama:
    """記錄呼叫參數；串流時逐字吐出 reply，非串流時回傳完整 reply。"""

    reply = "您好，\n1. 第一點\n2. 第二點 world"
    calls: list = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def create_chat_completion(self, messages, max_tokens, stream):
        FakeLlama.calls.append({"messages": messages, "max_tokens": max_tokens, "stream": stream})
        if not stream:
            return {"choices": [{"message": {"role": "assistant", "content": FakeLlama.reply}}]}

        def gen():
            yield {"choices": [{"delta": {"role": "assistant"}}]}
            for piece in ["您好，", "\n", "1. 第一點", "\n", "2. 第二點", " world"]:
                yield {"choices": [{"delta": {"content": piece}}]}

        return gen()


@pytest.fixture()
def server(monkeypatch):
    fake = types.ModuleType("llama_cpp")
    fake.Llama = FakeLlama
    fake.GGML_TYPE_Q8_0 = 8
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    monkeypatch.setenv("LLAMA_N_CTX", "4096")
    monkeypatch.delenv("LLAMA_API_KEY", raising=False)
    FakeLlama.calls = []
    sys.modules.pop("server_llama", None)
    import server_llama

    yield server_llama
    sys.modules.pop("server_llama", None)
