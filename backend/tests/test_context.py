"""上下文預算：設定值與推理服務回報的 n_ctx 取較小者，以及兩種管理策略。"""
from __future__ import annotations

import asyncio

import httpx
import pytest


@pytest.fixture()
def cm(monkeypatch):
    import app.services.context_manager as cm
    import app.services.inference_info as info

    info.reset()
    monkeypatch.setattr(cm.settings, "llama_context_max_tokens", 32768)
    monkeypatch.setattr(cm.settings, "llama_context_reserved_output_tokens", 4096)
    yield cm
    info.reset()


def test_effective_limit_uses_smaller_of_config_and_reported(cm):
    import app.services.inference_info as info

    assert cm.effective_context_max_tokens() == 32768
    info.record_health({"status": "ok", "n_ctx": 8192})
    assert cm.effective_context_max_tokens() == 8192
    info.record_health({"status": "ok", "n_ctx": 131072})
    assert cm.effective_context_max_tokens() == 32768


def test_bad_health_body_is_ignored(cm):
    import app.services.inference_info as info

    for body in (None, "ok", {}, {"n_ctx": "abc"}, {"n_ctx": 0}, {"n_ctx": -1}):
        info.record_health(body)
    assert info.reported_n_ctx() is None


def test_reserve_never_eats_whole_small_context(cm):
    import app.services.inference_info as info

    info.record_health({"n_ctx": 4096})
    # 4096 - min(4096, 2048) = 2048，而不是被壓到最小值 128
    assert cm._context_budget_tokens() == 2048


def _long_history(turns: int, chars: int = 400) -> list[dict]:
    out = []
    for i in range(turns):
        out.append({"role": "user", "content": f"第{i}輪問題" + "問" * chars})
        out.append({"role": "assistant", "content": f"第{i}輪回答" + "答" * chars})
    return out


def test_window_drops_oldest_to_fit(cm):
    import app.services.inference_info as info

    info.record_health({"n_ctx": 8192})
    history = _long_history(20)
    managed, meta = asyncio.run(cm.manage_context(history, [], "新問題", None, "window"))
    assert meta["applied"] and meta["dropped_messages"] > 0
    assert meta["after_tokens"] <= cm._context_budget_tokens()
    assert managed[-1] == history[-1]


def test_short_history_untouched(cm):
    history = _long_history(3, chars=20)
    managed, meta = asyncio.run(cm.manage_context(history, [], "新問題", None, "compress"))
    assert managed == history and not meta["applied"]


def test_compress_falls_back_to_window_when_llm_offline(cm, monkeypatch):
    import app.services.inference_info as info
    import app.services.streamer as streamer

    def offline(request):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(streamer, "_client", httpx.AsyncClient(transport=httpx.MockTransport(offline)))
    info.record_health({"n_ctx": 8192})
    history = _long_history(20)
    managed, meta = asyncio.run(cm.manage_context(history, [], "新問題", None, "compress"))
    assert meta["applied"] and meta["after_tokens"] <= cm._context_budget_tokens()
    assert not any(m["content"].startswith(cm.SUMMARY_PREFIX) for m in managed)


def test_compress_keeps_summary_and_recent(cm, monkeypatch):
    import app.services.inference_info as info
    import app.services.streamer as streamer

    sse = 'data: {"text": "<|channel|>final<|message|>使用者要退貨"}\n\ndata: {"done": true}\n\n'
    monkeypatch.setattr(
        streamer, "_client",
        httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=sse.encode()))),
    )
    info.record_health({"n_ctx": 8192})
    history = _long_history(20)
    managed, meta = asyncio.run(cm.manage_context(history, [], "新問題", None, "compress"))
    assert managed[0]["role"] == "system" and "使用者要退貨" in managed[0]["content"]
    assert managed[-1] == history[-1]
    assert meta["after_tokens"] < meta["before_tokens"]


def test_refresh_if_stale_is_throttled(cm, monkeypatch):
    import app.services.inference_info as info
    import app.services.streamer as streamer

    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"status": "ok", "n_ctx": 16384})

    monkeypatch.setattr(streamer, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    asyncio.run(info.refresh_if_stale())
    asyncio.run(info.refresh_if_stale())
    assert calls == ["/health"]
    assert cm.effective_context_max_tokens() == 16384
