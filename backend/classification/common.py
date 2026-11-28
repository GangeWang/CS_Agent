# -*- coding: utf-8 -*-
"""六種 guardrail 分類方式共用的資料載入、切分、門檻選擇與評估。

為什麼要共用（2026-11-26 訓練腳本審查）：
- 各腳本原本各自切分與去重（4選1 先去重再切、OvR 先切再只對 train 去重），測試集不同，結果無法比較
- 輸出路徑原本是相對 CWD，run_all_training.ps1 在 classification/ 執行時，模型會寫到 guardrail 讀不到的位置
- OvR 只報告 4 個二元模型各自的分數，上線實際的「過門檻取最高、都沒過 → UNCERTAIN」決策從未被評估
- 門檻搜尋區間 0.30～0.90、平手取最小值：上線 4 個門檻全是 0.30（區間下限），代表沒有真正選過

本模組只依賴 numpy / pandas / scikit-learn，ML 腳本在沒有 torch 的環境也能跑。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent  # backend/classification
DATA_PATH = ROOT / "data_set" / "train_full_dataset_v2.csv"
HARD_EVAL_PATH = ROOT / "data_set" / "hard_eval.csv"
BLIND_EVAL_PATH = ROOT / "data_set" / "blind_eval.csv"  # 盲測集：決定上線模型用，不得照它的錯誤補樣本
REPORT_DIR = ROOT / "reports"

LABELS: List[str] = ["NORMAL", "ABUSIVE", "PROMPT_ATTACK", "SPAM"]
UNCERTAIN = "UNCERTAIN"
SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # 佔 (train+val) 的比例

# 各方法輸出位置：與上線 guardrail / 既有入庫檔案位置一致（不受 CWD 影響）
OUT_PATHS: Dict[str, Path] = {
    "ml_multiclass": ROOT / "backend_ml_guardrail_model.joblib",
    "ml_ovr": ROOT / "ML" / "backend_ml_ovr_models",
    "bilstm_multiclass": ROOT,  # backend_ml_guardrail_bilstm_pytorch.pt / _vocab.joblib / _label_encoder.joblib
    "bilstm_ovr": ROOT / "BILSTM" / "backend_ml_ovr_bilstm_models",
    "transformer_multiclass": ROOT / "transformer" / "backend_ml_guardrail_transformer",
    # 上線模型：backend/app/services/guardrail.py 的 MODEL_DIR
    "transformer_ovr": ROOT / "transformer" / "backend_ml_ovr_transformer_models_clean",
}

_FILLERS = re.compile(r"(不好意思|想請教一下|請告訴我|想了解|可以嗎|請問|我想問|嗎|呢)")


def norm_key(text: str) -> str:
    """近重複比對用的正規化 key：去客套填充詞、標點與空白，轉小寫。"""
    return re.sub(r"[\W_]+", "", _FILLERS.sub("", str(text))).lower()


# ---------------------------------------------------------------- data
def load_dataset(path: Path = DATA_PATH, verbose: bool = True) -> pd.DataFrame:
    """讀取訓練集：標籤轉大寫並過濾、去空白、近重複去重、移除同文不同標籤。"""
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_csv(path).dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    n0 = len(df)
    df = df[(df["text"] != "") & df["label"].isin(LABELS)].copy()
    df["key"] = df["text"].map(norm_key)
    conflict = df.groupby("key")["label"].nunique()
    df = df[~df["key"].isin(conflict[conflict > 1].index)]
    df = df.drop_duplicates("key").reset_index(drop=True)
    if verbose:
        print(f"Loaded {path.name}: {n0} rows -> {len(df)} after label filter / near-dup / conflict removal")
        print(df["label"].value_counts().to_string())
    return df[["text", "label", "key"]]


@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def split_dataset(df: pd.DataFrame, seed: int = SEED, verbose: bool = True) -> Splits:
    """所有方法共用的 train / val / test 切分（分層、固定 seed）。

    load_dataset 已依 norm_key 去重，因此近重複句不會同時出現在 train 與 test。
    """
    X = df["text"].to_numpy(dtype=object)
    y = df["label"].to_numpy(dtype=object)
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE, random_state=seed, stratify=y_tv
    )
    leak = {norm_key(t) for t in X_train} & {norm_key(t) for t in X_test}
    assert not leak, f"train/test near-duplicate leakage: {len(leak)}"
    if verbose:
        print(f"Split: train={len(X_train)} val={len(X_val)} test={len(X_test)} (seed={seed})")
    return Splits(X_train, y_train, X_val, y_val, X_test, y_test)


def load_hard_eval(path: Path = HARD_EVAL_PATH):
    """困難評估集（step09 產出，不進訓練）。不存在時回 (None, None)。"""
    if not path.exists():
        return None, None
    d = pd.read_csv(path).dropna(subset=["text", "label"])
    return d["text"].astype(str).to_numpy(dtype=object), d["label"].astype(str).str.upper().to_numpy(dtype=object)


def load_eval_sets() -> Dict[str, tuple]:
    """訓練集以外的評估集：{"hard_eval": (X, y), "blind_eval": (X, y)}，檔案不存在的略過。

    - hard_eval：困難樣本，已被拿來找弱點補樣本，比較像開發集。
    - blind_eval：盲測集，用來比較 OvR / 4選1 決定上線模型。
    """
    out = {}
    for name, path in (("hard_eval", HARD_EVAL_PATH), ("blind_eval", BLIND_EVAL_PATH)):
        X, y = load_hard_eval(path)
        if X is not None:
            out[name] = (X, y)
    return out


def balanced_class_weights(y: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    """與 sklearn class_weight='balanced' 相同的公式：n_samples / (n_classes * count)。"""
    y = np.asarray(y)
    counts = np.array([max(1, int((y == c).sum())) for c in classes], dtype=float)
    return len(y) / (len(classes) * counts)


# ---------------------------------------------------------------- thresholds / decision
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def find_best_threshold(y_true_bin: Iterable[int], y_prob: Iterable[float], label: str = "") -> float:
    """在 val 上以 F1 選門檻。

    多個門檻 F1 相同（平台）時取平台中位數，而不是最小值；平台過寬時警告
    ——代表 val 太容易分辨（常見原因：近重複洩漏、資料過於模板化），門檻並沒有被真正選出來。
    """
    y_true_bin = np.asarray(list(y_true_bin)).astype(int)
    y_prob = np.asarray(list(y_prob), dtype=float)
    f1s = []
    for t in THRESHOLD_GRID:
        _, _, f1, _ = precision_recall_fscore_support(
            y_true_bin, (y_prob >= t).astype(int), average="binary", zero_division=0
        )
        f1s.append(f1)
    f1s = np.array(f1s)
    best = f1s.max()
    plateau = THRESHOLD_GRID[np.isclose(f1s, best)]
    t = float(np.median(plateau))
    if len(plateau) > len(THRESHOLD_GRID) * 0.5:
        print(f"[WARN] {label} threshold plateau covers {len(plateau)}/{len(THRESHOLD_GRID)} grid points "
              f"(val F1={best:.4f}); val set may be too easy / leaking")
    return round(t, 2)


def ovr_decide(prob_matrix: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """與上線 guardrail._predict_ovr_transformer 相同的決策：過門檻者取機率最高，都沒過 → UNCERTAIN。

    prob_matrix: shape [N, len(LABELS)]，欄位順序同 LABELS。
    """
    th = np.array([float(thresholds[lb]) for lb in LABELS])
    passed = prob_matrix >= th
    masked = np.where(passed, prob_matrix, -1.0)
    best = masked.argmax(axis=1)
    out = np.array([LABELS[i] for i in best], dtype=object)
    out[~passed.any(axis=1)] = UNCERTAIN
    return out


# ---------------------------------------------------------------- reporting
def multiclass_report(y_true: Sequence[str], y_pred: Sequence[str], title: str) -> dict:
    """4 類（+UNCERTAIN）報告與混淆矩陣；回傳可存 json 的 dict 並印出。"""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    # 報告只列 4 個真實類別（UNCERTAIN 預測視為該類 recall 的漏判）；混淆矩陣多一欄 UNCERTAIN
    rep_txt = classification_report(y_true, y_pred, labels=LABELS, digits=4, zero_division=0)
    rep = classification_report(y_true, y_pred, labels=LABELS, digits=4, zero_division=0, output_dict=True)
    cm_labels = LABELS + [UNCERTAIN]
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)[: len(LABELS)]
    acc = float((y_true == y_pred).mean())
    print(f"\n=== {title} ===")
    print(rep_txt)
    print(f"accuracy={acc:.4f}  uncertain_rate={(y_pred == UNCERTAIN).mean():.4f}")
    print("confusion matrix (rows=true, cols=pred):", cm_labels)
    print(cm)
    return {
        "title": title,
        "labels": cm_labels,
        "macro_f1": rep["macro avg"]["f1-score"],
        "accuracy": acc,
        "uncertain_rate": float((y_pred == UNCERTAIN).mean()),
        "report": rep,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_split(texts: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str], title: str) -> dict:
    """multiclass_report + 列出錯誤樣本（hard_eval 時特別有用）。"""
    res = multiclass_report(y_true, y_pred, title)
    res["errors"] = [
        {"text": str(t), "true": str(a), "pred": str(b)} for t, a, b in zip(texts, y_true, y_pred) if a != b
    ]
    if 0 < len(res["errors"]) <= 30:
        print("errors:")
        for e in res["errors"]:
            print(f"  [{e['true']} -> {e['pred']}] {e['text']}")
    return res


def save_report(method: str, payload: dict) -> Path:
    """各方法的評估結果統一存到 classification/reports/<method>.json，方便六種方法並排比較。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{method}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved report => {path}")
    return path


def write_ovr_config(model_dir: Path, thresholds: Dict[str, float], extra: dict) -> None:
    """寫 ovr_config.json；保留既有檔案中本腳本不管理的欄位（如 whitelist_domains、semantic_topk）。"""
    path = model_dir / "ovr_config.json"
    cfg = {}
    if path.exists():
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    cfg.update({"labels": LABELS, "thresholds": thresholds, **extra})
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved config =>", path)
