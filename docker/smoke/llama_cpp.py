"""
Docker 冒煙測試用的假 llama_cpp（沒有 GPU 也能跑整套）。

輸出模仿 gpt-oss 經 llama-cpp-python 的串流：
- FAKE_MODE=nospecial（預設）：特殊標記不會被印出來 → "analysis思考…assistantfinal回覆"
- FAKE_MODE=harmony：帶標記 → "<|channel|>analysis<|message|>思考…<|channel|>final<|message|>回覆"
思考內容固定含「THINKING_SHOULD_NOT_LEAK」，ws_smoke.py 用它確認思考沒有送到前端。
"""
import os
import time

GGML_TYPE_Q8_0 = 8

_THINK = "THINKING_SHOULD_NOT_LEAK：使用者在問問題。"
_ANSWER = ["您好，", "這是", "測試", "回覆。"]


def _pieces():
    if os.environ.get("FAKE_MODE", "nospecial") == "harmony":
        return ["<|channel|>", "analysis", "<|message|>", _THINK, "<|end|>", "<|start|>", "assistant",
                "<|channel|>", "final", "<|message|>", *_ANSWER]
    return ["analysis", _THINK, "assistant", "final", *_ANSWER]


class Llama:
    def __init__(self, **kwargs):
        pass

    def create_chat_completion(self, messages, max_tokens, stream):
        pieces = _pieces()
        if not stream:
            return {"choices": [{"message": {"role": "assistant", "content": "".join(pieces)}}]}
        delay = float(os.environ.get("FAKE_DELAY", "0.15"))

        def gen():
            yield {"choices": [{"delta": {"role": "assistant"}}]}
            for p in pieces:
                time.sleep(delay)
                yield {"choices": [{"delta": {"content": p}}]}

        return gen()
