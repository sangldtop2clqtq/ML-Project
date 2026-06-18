import json
import pandas as pd
from pathlib import Path


metrics_path = Path("reports/str_pop/metrics.json")

if not metrics_path.exists():
    print(f"Khong tim thay file: {metrics_path}")
    print("Vui long dam bao da chay huan luyen mo hinh truoc.")
    exit(1)


with open(metrics_path, "r", encoding="utf-8") as f:
    data = json.load(f)

labels = data["labels"]
matrix = data["confusion_matrix"]


df = pd.DataFrame(matrix, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])

print("\n" + "="*50)
print(" CHI SO DO CHINH XAC CHUNG")
print("="*50)
print(f"Accuracy (Do chinh xac tong): {data['accuracy']:.4f}")
print(f"Balanced Accuracy:           {data['balanced_accuracy']:.4f}")
print(f"F1 Macro:                    {data['f1_macro']:.4f}")

print("\n" + "="*50)
print(" MA TRAN NHAM LAN (CONFUSION MATRIX)")
print("="*50)
print(df.to_string())

print("\n" + "="*50)
print(" BAO CAO CHI TIET THEO TUNG NHOM (CLASSIFICATION REPORT)")
print("="*50)
report_df = pd.DataFrame(data["classification_report"]).T

report_df = report_df.loc[labels + ["accuracy", "macro avg", "weighted avg"]]
print(report_df.round(4).to_string())
