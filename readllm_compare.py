import pandas as pd

# 讀取 TSV 評估結果
df = pd.read_csv("rouge_bert_evaluation.tsv", sep="\t")

# 計算各模型平均評估分數
summary = df.groupby("model")[["rouge1", "rouge2", "rougeL", "bert_f1"]].mean().reset_index()

print("📊 模型平均評估分數：")
print(summary.to_string(index=False, float_format="%.3f"))
