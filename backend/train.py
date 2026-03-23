# 匯入必要的套件
from pathlib import Path  # 用來處理檔案路徑
import pandas as pd       # 用於資料讀取和處理
import joblib             # 用於模型的存取（儲存與讀取）
from sklearn.model_selection import train_test_split  # 用於將資料分割成訓練集與測試集
from sklearn.pipeline import Pipeline                 # 用於建立 ML 流水線
from sklearn.feature_extraction.text import TfidfVectorizer  # 將文字轉換成 TF-IDF 特徵向量
from sklearn.linear_model import LogisticRegression  # 邏輯迴歸分類器
from sklearn.metrics import classification_report    # 用於評估模型表現


# 指定資料與模型儲存的路徑
DATA_PATH = Path("backend_ml_data_guardrail_augmented_big.csv")  # CSV 資料路徑
MODEL_PATH = Path("backend_ml_guardrail_model.joblib")           # 模型儲存路徑


# 主程式函數
def main():
    # 檢查資料檔案是否存在，若不存在就丟出錯誤
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH.resolve()}")

    # 使用 pandas 讀取 CSV 資料
    df = pd.read_csv(DATA_PATH)

    # 確認 CSV 是否包含必要的欄位 "text" 與 "label"
    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}, got: {list(df.columns)}")

    # ===================== 資料清理 =====================
    # 1. 移除 text 或 label 欄位有缺失值的列
    df = df.dropna(subset=["text", "label"]).copy()

    # 2. 將 text 與 label 轉成字串並去除首尾空白
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    # 3. 移除空字串的列
    df = df[df["text"] != ""]

    # 4. 移除重複的 text-label 組合
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    # 顯示清理後資料的數量與標籤分佈
    print(f"Loaded rows: {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts())

    # ===================== 建立特徵與標籤 =====================
    X = df["text"]  # 文字資料
    y = df["label"] # 對應標籤

    # 將資料拆分成訓練集與測試集
    # test_size=0.2 -> 20% 作為測試集
    # stratify=y -> 保持標籤分佈一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ===================== 建立模型流水線 =====================
    clf = Pipeline([
        # 文字向量化
        ("tfidf", TfidfVectorizer(
            analyzer="char",       # 以字元為單位（而不是單詞）
            ngram_range=(1, 5),    # 取 1~5 個字元組成的 n-gram
            min_df=1,              # 出現次數至少 1 次
            sublinear_tf=True      # TF 權重使用 sublinear scaling (1 + log(tf))
        )),
        # 邏輯迴歸分類器
        ("lr", LogisticRegression(
            max_iter=4000,         # 最大迭代次數，確保收斂
            class_weight="balanced", # 自動平衡不同類別的權重
            C=2.0,                 # 正則化強度，越大越弱正則化
            solver="lbfgs"         # 最適合小/中型資料的求解器
        ))
    ])

    # 訓練模型
    clf.fit(X_train, y_train)

    # 對測試集做預測
    pred = clf.predict(X_test)

    # ===================== 評估模型 =====================
    print("\n=== Classification Report ===")
    # 顯示 Precision、Recall、F1-score、Support
    # digits=4 -> 顯示四位小數
    # zero_division=0 -> 避免除以 0 報錯
    print(classification_report(y_test, pred, digits=4, zero_division=0))

    # ===================== 儲存模型 =====================
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 儲存模型
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved model => {MODEL_PATH.resolve()}")


# 如果這個檔案被執行，而不是被匯入，就執行 main()
if __name__ == "__main__":
    main()