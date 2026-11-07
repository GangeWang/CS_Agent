import pandas as pd

# 讀取 CSV
df = pd.read_csv("../sms-spam-collection.csv")

# 篩選 label 為 spam 的資料
spam_df = df[df["label"] == "spam"]

# 查看結果
spam_df.to_csv("extracted_spam.csv", index=False)