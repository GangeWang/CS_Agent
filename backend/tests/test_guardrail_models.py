"""guardrail 的模型階段：用極小的隨機 BERT 跑真的 torch / transformers。

CI 只裝輕量依賴，沒有 torch 時整檔跳過；本機有裝完整 requirements.txt 時會跑。
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture()
def tiny_models(tmp_path, monkeypatch):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    import app.services.guardrail as g

    chars = sorted(set("您好請問退款怎麼申請你們很爛忽略以上指示加我賺錢今天天氣不錯"))
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars), encoding="utf-8")
    torch.manual_seed(0)
    for lb in g.LABELS:
        d = tmp_path / lb.lower()
        tok = BertTokenizerFast(vocab_file=str(vocab), tokenize_chinese_chars=True)
        cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=32, num_labels=2)
        BertForSequenceClassification(cfg).save_pretrained(d)
        tok.save_pretrained(d)
    (tmp_path / "ovr_config.json").write_text(json.dumps({
        "thresholds": {lb: 0.5 for lb in g.LABELS}, "max_len": 32,
    }), encoding="utf-8")

    monkeypatch.setattr(g, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(g, "CONFIG_PATH", tmp_path / "ovr_config.json")
    monkeypatch.setattr(g, "SEMANTIC_TEXTS_PATH", tmp_path / "missing.joblib")
    monkeypatch.setitem(g._LOAD_FAILURE, "at", None)
    g._load_guardrail_resources.cache_clear()
    yield g
    g._load_guardrail_resources.cache_clear()


def test_models_load_and_classify(tiny_models):
    g = tiny_models
    r = g.classify_text("您好請問退款怎麼申請")
    assert r["available"] is True and r["stage"] in {"ML", "UNCERTAIN"}
    assert set(r["probs"]) == set(g.LABELS)
    assert g.guardrail_status()["loaded"] is True


def test_shared_tokenizer_matches_per_model_tokenize(tiny_models):
    g = tiny_models
    rs = g._get_resources()
    assert rs["shared_tokenizer"] is not None  # 四個 tokenizer 相同 → 只 tokenize 一次
    args = dict(models=rs["models"], tokenizers=rs["tokenizers"], thresholds=rs["thresholds"],
                text="你們很爛", device=rs["device"], max_len=rs["max_len"])
    shared = g._predict_ovr_transformer(**args, shared_tokenizer=rs["shared_tokenizer"])
    each = g._predict_ovr_transformer(**args, shared_tokenizer=None)
    assert shared[0] == each[0]
    for lb in g.LABELS:
        assert shared[2][lb] == pytest.approx(each[2][lb], abs=1e-6)
