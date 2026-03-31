import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── 1. Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv("data.csv")

# ── 2. Parse anomalies → labels ────────────────────────────────────────────────
def parse_col(s):
    try:
        return eval(s, {"array": np.array, "np": np})
    except:
        return None

df["anomalies_parsed"]  = df["anomalies"].apply(parse_col)
df["label_binary"]      = df["anomalies_parsed"].apply(lambda x: int(x["exists"]) if x else 0)
df["label_type"]        = df["anomalies_parsed"].apply(
    lambda x: x["type"] if x and x["exists"] else "normal"
)

# ── 3. Parse statistics → feature matrix ───────────────────────────────────────
df["stats_parsed"] = df["statistics"].apply(parse_col)

def flatten_stats(stats_dict):
    row = {}
    for kpi, stats in stats_dict.items():
        for stat_name, value in stats.items():
            row[f"{kpi}__{stat_name}"] = value
    return row

X = pd.DataFrame(df["stats_parsed"].apply(flatten_stats).tolist())

print(f"Feature matrix shape: {X.shape}")  # (32000, 64) — 16 KPIs × 4 stats
print(f"Features: {X.columns.tolist()}")

# ── 4. Labels ──────────────────────────────────────────────────────────────────
y_binary = df["label_binary"]
y_type   = df["label_type"]

# ── 5. Split ───────────────────────────────────────────────────────────────────
X_train, X_test, yb_train, yb_test, yt_train, yt_test = train_test_split(
    X, y_binary, y_type,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")
print(f"\nLabel distribution (train):\n{yb_train.value_counts()}")

# ── 6. Save ────────────────────────────────────────────────────────────────────
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv",   index=False)
yb_train.to_csv("yb_train.csv", index=False)
yb_test.to_csv("yb_test.csv",   index=False)
yt_train.to_csv("yt_train.csv", index=False)
yt_test.to_csv("yt_test.csv",   index=False)

print("\nDataPrep complete.")