"""WebSocket 端到端：真的 FastAPI app + 真的 httpx 串流，推理服務用 httpx.MockTransport 模擬。

guardrail 模型與推理服務都不需要真的存在（取代原本 backend/tests_smoke.py 要另外起 uvicorn 的整合測試）。
"""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

FINAL = "<|channel|>final<|message|>"


def _sse(*parts: dict) -> bytes:
    return "".join(f"data: {json.dumps(p, ensure_ascii=False)}\n\n" for p in parts).encode("utf-8")


@pytest.fixture()
def make_client(monkeypatch):
    import app.main as main
    import app.routers.ws as ws
    import app.services.streamer as streamer

    seen: dict = {"requests": []}

    def fake_classify(text: str) -> dict:
        label = "ABUSIVE" if "爛" in text else "NORMAL"
        return {"label": label, "stage": "ML", "available": True}

    monkeypatch.setattr(ws, "classify_text", fake_classify)
    monkeypatch.setattr(main, "classify_text", fake_classify)
    monkeypatch.setattr(main.settings, "llama_api_url", "http://inference.test")

    def build(handler):
        def recording_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path != "/health":
                seen["requests"].append(json.loads(request.content or b"{}"))
            return handler(request)

        monkeypatch.setattr(streamer, "_client", httpx.AsyncClient(transport=httpx.MockTransport(recording_handler)))
        return TestClient(main.app), seen

    return build


def _recv_until(ws, types: set[str]) -> list[dict]:
    out = []
    while True:
        msg = json.loads(ws.receive_text())
        out.append(msg)
        if msg.get("type") in types:
            return out


def test_chat_streams_and_keeps_history(make_client):
    def handler(request):
        if request.url.path == "/api/stream":
            return httpx.Response(200, content=_sse({"text": FINAL + "您好，"}, {"text": "請問"}, {"done": True}))
        return httpx.Response(404)

    client, seen = make_client(handler)
    with client, client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"messages": [{"role": "user", "content": "退款怎麼申請"}]}))
        msgs = _recv_until(ws, {"done", "error"})
        assert msgs[0] == {"type": "guardrail", "label": "NORMAL", "degraded": False}
        assert "".join(m.get("text", "") for m in msgs if m["type"] == "delta") == "您好，請問"
        assert msgs[-1]["type"] == "done"

        ws.send_text(json.dumps({"messages": [{"role": "user", "content": "要多久"}]}))
        _recv_until(ws, {"done", "error"})

    second = seen["requests"][-1]
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
    assert second["messages"][1]["content"] == "您好，請問"
    assert "GuardrailLabel=NORMAL" in second["system_prompt"]


def test_abusive_label_reaches_prompt(make_client):
    client, seen = make_client(lambda r: httpx.Response(200, content=_sse({"text": FINAL + "抱歉"}, {"done": True})))
    with client, client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"messages": [{"role": "user", "content": "你們很爛"}]}))
        msgs = _recv_until(ws, {"done", "error"})
    assert msgs[0]["label"] == "ABUSIVE"
    assert "GuardrailLabel=ABUSIVE" in seen["requests"][-1]["system_prompt"]


def test_llm_offline_reports_error_and_connection_survives(make_client):
    def handler(request):
        raise httpx.ConnectError("refused")

    client, _ = make_client(handler)
    with client, client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"messages": [{"role": "user", "content": "你好"}]}))
        msgs = _recv_until(ws, {"done", "error"})
        assert msgs[-1] == {"type": "error", "error": "LLM 連線失敗，請稍後再試。"}
        ws.send_text(json.dumps({"type": "ping"}))
        assert json.loads(ws.receive_text()) == {"type": "pong"}


def test_bad_input_is_rejected(make_client):
    client, _ = make_client(lambda r: httpx.Response(500))
    with client, client.websocket_connect("/ws/chat") as ws:
        ws.send_text("{not json")
        assert json.loads(ws.receive_text()) == {"type": "error", "error": "JSON 格式不正確"}
        ws.send_text(json.dumps({"messages": []}))
        assert json.loads(ws.receive_text()) == {"type": "error", "error": "缺少使用者訊息"}
        ws.send_text("x" * 20000)
        assert json.loads(ws.receive_text())["type"] == "error"


def test_end_conversation_sends_summary(make_client):
    client, seen = make_client(lambda r: httpx.Response(200, content=_sse({"text": FINAL + "摘要內容"}, {"done": True})))
    with client:
        # 結束後伺服器會關閉連線，所以兩個情境各開一條
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "end_conversation"}))
            empty = _recv_until(ws, {"conversation_ended"})
        assert empty[0]["summary"] == "本次對話沒有可摘要的內容。"

        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"messages": [{"role": "user", "content": "我要退貨"}]}))
            _recv_until(ws, {"done", "error"})
            ws.send_text(json.dumps({"type": "end_conversation"}))
            msgs = _recv_until(ws, {"conversation_ended"})
    assert msgs[0] == {"type": "conversation_summary", "reason": "manual", "summary": "摘要內容"}
    assert "我要退貨" in seen["requests"][-1]["prompt"]


def test_agent_mode_streams(make_client):
    client, seen = make_client(lambda r: httpx.Response(200, content=_sse({"text": FINAL + "現在時間"}, {"done": True})))
    with client, client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"mode": "agent", "messages": [{"role": "user", "content": "現在幾點"}]}))
        msgs = _recv_until(ws, {"done", "error"})
    assert msgs[-1]["type"] == "done"
    assert seen["requests"][-1]["messages"][-1]["content"] == "使用者問題：現在幾點"


def test_health_reports_inference_state_and_context(make_client):
    import app.services.inference_info as info

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "n_ctx": 8192})
        return httpx.Response(404)

    info.reset()
    client, _ = make_client(handler)
    with client:
        body = client.get("/health").json()
    info.reset()
    assert body["llama"] == "connected"
    assert body["context_max_tokens"] == 8192


def test_health_reports_disconnected(make_client):
    def handler(request):
        raise httpx.ConnectError("refused")

    client, _ = make_client(handler)
    with client:
        body = client.get("/health").json()
    assert body["status"] == "degraded" and body["llama"] == "disconnected"


def test_non_object_json_is_rejected(make_client):
    client, _ = make_client(lambda r: httpx.Response(500))
    with client, client.websocket_connect("/ws/chat") as ws:
        ws.send_text("[1, 2]")
        assert json.loads(ws.receive_text()) == {"type": "error", "error": "JSON 格式不正確"}
        ws.send_text(json.dumps({"type": "ping"}))
        assert json.loads(ws.receive_text()) == {"type": "pong"}


def test_rate_limit(make_client, monkeypatch):
    import app.routers.ws as ws_mod

    monkeypatch.setattr(ws_mod.settings, "ws_rate_limit_messages", 1)
    client, _ = make_client(lambda r: httpx.Response(200, content=_sse({"text": FINAL + "好"}, {"done": True})))
    with client, client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"messages": [{"role": "user", "content": "一"}]}))
        _recv_until(ws, {"done", "error"})
        ws.send_text(json.dumps({"messages": [{"role": "user", "content": "二"}]}))
        assert json.loads(ws.receive_text()) == {"type": "error", "error": "傳送過於頻繁，請稍候再試。"}


def test_clear_history_forgets_previous_turns(make_client):
    client, seen = make_client(lambda r: httpx.Response(200, content=_sse({"text": FINAL + "好"}, {"done": True})))
    with client, client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"messages": [{"role": "user", "content": "第一句"}]}))
        _recv_until(ws, {"done", "error"})
        ws.send_text(json.dumps({"type": "clear_history"}))
        assert json.loads(ws.receive_text()) == {"type": "history_cleared"}
        ws.send_text(json.dumps({"messages": [{"role": "user", "content": "第二句"}]}))
        _recv_until(ws, {"done", "error"})
    assert [m["content"] for m in seen["requests"][-1]["messages"]] == ["第二句"]


def test_idle_warning_then_timeout(make_client, monkeypatch):
    import app.routers.ws as ws_mod

    client, _ = make_client(lambda r: httpx.Response(200, content=_sse({"text": FINAL + "摘要"}, {"done": True})))
    monkeypatch.setattr(ws_mod, "IDLE_TIMEOUT_SECONDS", 1000)
    monkeypatch.setattr(ws_mod, "IDLE_WARNING_SECONDS_BEFORE_END", 1000)
    with client, client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps({"type": "ping"}))
        warning = json.loads(ws.receive_text())
        assert warning["type"] == "idle_warning" and warning["remaining_seconds"] > 0
        assert json.loads(ws.receive_text()) == {"type": "pong"}
        # 警告只送一次
        ws.send_text(json.dumps({"type": "ping"}))
        assert json.loads(ws.receive_text()) == {"type": "pong"}

        monkeypatch.setattr(ws_mod, "IDLE_TIMEOUT_SECONDS", 0)
        ws.send_text(json.dumps({"type": "ping"}))
        msgs = _recv_until(ws, {"conversation_ended"})
    assert msgs[-1] == {"type": "conversation_ended", "reason": "idle_timeout"}
