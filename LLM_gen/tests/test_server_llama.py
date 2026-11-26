from __future__ import annotations

import json

from fastapi.testclient import TestClient


def _sse_payloads(text: str) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]


def test_health_reports_config_without_inference(server):
    from conftest import FakeLlama

    r = TestClient(server.app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["n_ctx"] == 4096
    assert FakeLlama.calls == []  # 預設不跑推理


def test_health_deep_runs_one_token(server):
    from conftest import FakeLlama

    r = TestClient(server.app).get("/health", params={"deep": "true"})
    assert r.status_code == 200 and r.json()["deep"] is True
    assert FakeLlama.calls[-1]["max_tokens"] == 1


def test_health_503_when_model_missing(server, monkeypatch):
    monkeypatch.setattr(server, "llm", None)
    assert TestClient(server.app).get("/health").status_code == 503


def test_stream_keeps_newlines_and_spaces(server):
    from conftest import FakeLlama

    r = TestClient(server.app).post("/api/stream", json={
        "system_prompt": "你是客服",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 64,
    })
    payloads = _sse_payloads(r.text)
    assert payloads[-1] == {"done": True}
    text = "".join(p.get("text", "") for p in payloads)
    assert text == "您好，\n1. 第一點\n2. 第二點 world"
    call = FakeLlama.calls[-1]
    assert call["stream"] is True and call["max_tokens"] == 64
    assert call["messages"][0] == {"role": "system", "content": "你是客服"}
    assert call["messages"][-1] == {"role": "user", "content": "你好"}


def test_stream_reports_generic_error(server, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory at /secret/path")

    monkeypatch.setattr(server, "generate_stream", boom)
    payloads = _sse_payloads(TestClient(server.app).post("/api/stream", json={"prompt": "hi"}).text)
    assert payloads == [{"error": "generation failed"}]  # 不把內部錯誤細節傳出去


def test_generate_uses_prompt_when_no_messages(server):
    from conftest import FakeLlama

    r = TestClient(server.app).post("/api/generate", json={"prompt": "你好", "max_tokens": 10})
    assert r.json()["text"].startswith("您好")
    assert FakeLlama.calls[-1]["messages"] == [{"role": "user", "content": "你好"}]


def test_api_key_required_when_configured(server, monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "k")
    client = TestClient(server.app)
    assert client.post("/api/generate", json={"prompt": "x"}).status_code == 401
    assert client.post("/api/generate", json={"prompt": "x"}, headers={"X-API-KEY": "k"}).status_code == 200


def test_build_messages_does_not_duplicate_system(server):
    msgs, _ = server.build_messages({
        "system_prompt": "S",
        "messages": [{"role": "system", "content": "already"}, {"role": "user", "content": "u"}, {"bad": 1}],
    })
    assert msgs == [{"role": "system", "content": "already"}, {"role": "user", "content": "u"}]
