# -*- coding: utf-8 -*-
"""Step09：清理 guardrail 訓練集 + 合併人工撰寫的優質樣本。

輸入：data_set/train_full_dataset.csv（step08 產物，保留不動）
輸出：
  data_set/train_full_dataset_v2.csv   清理後訓練集（訓練腳本 DATA_PATH 指向此檔）
  data_set/removed_rows_v2.csv         被移除的每一筆 + 原因（供人工複查）
  data_set/hard_eval.csv               困難評估集（不進訓練，重訓後用來驗證誤判；199 句，見 curated_hard_eval.py）

清理規則（2026-11-26 語料審查結論，見 classifcation/README.md「語料審查」）：
  1) label 正規化：小寫 spam → SPAM（原本訓練腳本 isin(LABELS) 會整批丟掉 592 筆）
  2) 翻譯失敗殘留英文（拉丁字母 > 50%）→ 移除（step06 失敗時保留原文）
  3) 翻譯/模板殘留（"I will translate…"、"OpenAI 的內容政策…" 前綴）→ 移除
  4) PROMPT_ATTACK：
     - JailBreakV 的「圖片底部顯示…」多模態題目（缺圖片，多為無害問題）→ 移除
     - GCG 對抗後綴亂碼 → 移除
     - "ChatGPT: … OPPO: …" 雙回答模板殘骸 → 移除
     - 翻譯來源中沒有任何越獄/注入訊號的列（多為仇恨言論或一般有害提問，非 prompt attack）→ 移除
  5) ABUSIVE（Jigsaw 維基討論頁翻譯）：
     - 含維基討論頁語境、> 120 字、或無任何辱罵詞 → 移除
     - 其餘下採樣至 JIGSAW_ABUSIVE_CAP，避免 ABUSIVE 壓倒 NORMAL（原 10,268 vs 5,917）
  6) SPAM：被模板誤組的「合法通知」（若非本人請忽略 / 請備份資料…）→ 移除；> 200 字 → 移除
  7) 以正規化文字（去客套填充詞/標點）做近重複去重；同文不同標籤者全數移除
  8) 合併 curated_seed_data.py 的人工優質樣本（含 curated_seed_extra.py、curated_seed_long.py、curated_seed_targeted.py）；困難評估集與訓練集不得重疊

用法（於 backend/classifcation/ 執行）：
    python data_pipeline/step09_clean_dataset.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import curated_seed_data as seeds  # noqa: E402
from common import norm_key  # noqa: E402  與訓練腳本共用同一個近重複定義

INPUT = ROOT / "data_set" / "train_full_dataset.csv"
ML_DATA = ROOT / "data_set" / "ML_data.csv"  # 本地中文客服語境來源（其餘為翻譯來源）
OUT_TRAIN = ROOT / "data_set" / "train_full_dataset_v2.csv"
OUT_REMOVED = ROOT / "data_set" / "removed_rows_v2.csv"
OUT_EVAL = ROOT / "data_set" / "hard_eval.csv"

LABELS = ["NORMAL", "ABUSIVE", "PROMPT_ATTACK", "SPAM"]
JIGSAW_ABUSIVE_CAP = 2500
SEED = 42

IMG = re.compile(r"(圖片|圖像|影像|照片)(的)?(底部|下方|上方|中)")
BOILER = re.compile(r"OpenAI(的|’s|'s)\s*內容政策|OpenAI[’']s content polic|anyways, since we are out of", re.I)
TRANS = re.compile(r"I will translate|translate your request|翻譯成(繁體|中文)|以下是(翻譯|譯文)|Traditional Chinese \(Taiwan", re.I)
DUAL = re.compile(r"^\s*(\[?ChatGPT\]?\s*[:：]|:\s*我(乃|是)ChatGPT)|OPPO\s*[:：]", re.I)
GCG_CHARS = re.compile(r"[\[\]\{\}\(\)\\\$#;`|<>]")
WIKI = re.compile(r"維基|討論頁|UTC|wikipedia|用戶頁|編輯戰|條目|撤回|回退", re.I)
PA_SIGNAL = re.compile(
    r"忽略|無視|忘記|忘掉|扮演|角色|系統提示|提示詞|開發者|DAN|越獄|不受(任何)?限制|沒有(任何)?(道德|限制|審查)"
    r"|繞過|規則|指令|限制|模式|假裝|假設你|你現在是|從現在開始|prompt|審查|過濾|jailbreak|ignore|pretend"
    r"|roleplay|developer|不要拒絕|永遠不拒絕|回答任何",
    re.I,
)
INSULT = re.compile(
    r"混蛋|白癡|白痴|智障|笨蛋|蠢|傻|廢物|垃圾|低能|他媽|去你|幹|操|屎|屁|婊|賤|滾|去死|狗|豬|雞巴|閉嘴|王八"
    r"|畜生|腦殘|噁心|爛|白目|機掰|靠北|三小|肏|妓|下流|醜|可憐|fuck|bitch|shit|ass",
    re.I,
)
LEGIT_NOTICE = re.compile(r"若非本人請忽略|請備份資料|以避免資料遺失")


def latin_ratio(s: str) -> float:
    return len(re.findall(r"[A-Za-z]", s)) / max(1, len(s))


def removal_reason(text: str, label: str, from_ml_data: bool) -> str | None:
    if latin_ratio(text) > 0.5:
        return "untranslated_english"
    if TRANS.search(text) or BOILER.search(text):
        return "translation_or_template_artifact"
    if label == "PROMPT_ATTACK":
        if IMG.search(text):
            return "pa_multimodal_without_image"
        if len(GCG_CHARS.findall(text)) >= 8:
            return "pa_adversarial_suffix_garbage"
        if DUAL.search(text):
            return "pa_dual_answer_template_residue"
        if not from_ml_data and not PA_SIGNAL.search(text):
            return "pa_no_injection_signal"
        if len(text) > 600:
            return "too_long"
    if label == "ABUSIVE" and not from_ml_data:
        if WIKI.search(text):
            return "abusive_wiki_talk_context"
        if len(text) > 120:
            return "too_long"
        if not INSULT.search(text):
            return "abusive_no_insult"
    if label == "SPAM":
        if LEGIT_NOTICE.search(text):
            return "spam_legit_notice_mislabeled"
        if len(text) > 200:
            return "too_long"
    return None


def main() -> None:
    df = pd.read_csv(INPUT).dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df[(df["text"] != "") & df["label"].isin(LABELS)].reset_index(drop=True)
    n_in = len(df)

    ml_texts = set(pd.read_csv(ML_DATA)["text"].astype(str).str.strip())
    df["from_ml_data"] = df["text"].isin(ml_texts)
    df["reason"] = [removal_reason(t, lb, m) for t, lb, m in zip(df["text"], df["label"], df["from_ml_data"])]

    removed = [df[df["reason"].notna()]]
    kept = df[df["reason"].isna()].copy()

    # Jigsaw ABUSIVE 下採樣
    jig = kept[(kept["label"] == "ABUSIVE") & ~kept["from_ml_data"]]
    if len(jig) > JIGSAW_ABUSIVE_CAP:
        drop_idx = jig.drop(jig.sample(JIGSAW_ABUSIVE_CAP, random_state=SEED).index).index
        d = kept.loc[drop_idx].copy()
        d["reason"] = "abusive_downsampled_for_balance"
        removed.append(d)
        kept = kept.drop(drop_idx)

    # 加入人工優質樣本
    seed_rows = [(t, lb) for lb in LABELS for t in getattr(seeds, lb)]
    seed_df = pd.DataFrame(seed_rows, columns=["text", "label"])
    seed_df["from_ml_data"] = False
    kept = pd.concat([kept, seed_df], ignore_index=True)

    # 近重複 / 標籤衝突
    kept["key"] = kept["text"].map(norm_key)
    conflict_keys = set(kept.groupby("key")["label"].nunique().loc[lambda s: s > 1].index)
    c = kept[kept["key"].isin(conflict_keys)].copy()
    c["reason"] = "conflicting_labels"
    removed.append(c)
    kept = kept[~kept["key"].isin(conflict_keys)]
    dup_mask = kept.duplicated("key", keep="first")
    d = kept[dup_mask].copy()
    d["reason"] = "near_duplicate"
    removed.append(d)
    kept = kept[~dup_mask]

    # 評估集不得出現在訓練集
    eval_df = pd.DataFrame(seeds.HARD_EVAL, columns=["text", "label"])
    eval_keys = set(eval_df["text"].map(norm_key))
    leak = kept["key"].isin(eval_keys)
    d = kept[leak].copy()
    d["reason"] = "overlaps_hard_eval"
    removed.append(d)
    kept = kept[~leak]

    kept[["text", "label"]].sample(frac=1, random_state=SEED).to_csv(OUT_TRAIN, index=False, encoding="utf-8-sig")
    rm = pd.concat(removed, ignore_index=True)[["text", "label", "reason"]]
    rm.to_csv(OUT_REMOVED, index=False, encoding="utf-8-sig")
    eval_df.to_csv(OUT_EVAL, index=False, encoding="utf-8-sig")

    print(f"input rows: {n_in}")
    print(f"removed:    {len(rm)}")
    print(rm["reason"].value_counts().to_string())
    print(f"seed added: {len(seed_df)}")
    print(f"output:     {len(kept)}")
    print(kept["label"].value_counts().to_string())
    print(f"hard eval:  {len(eval_df)}")


if __name__ == "__main__":
    main()
