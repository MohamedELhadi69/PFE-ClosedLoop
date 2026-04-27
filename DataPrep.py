import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils import load_or_fetch_dataframe, parse_col


df = load_or_fetch_dataframe()

df["anomalies_parsed"] = df["anomalies"].apply(parse_col)
df["label_binary"] = df["anomalies_parsed"].apply(lambda x: int(x["exists"]) if x else 0)
df["label_type"] = df["anomalies_parsed"].apply(
    lambda x: x["type"] if x and x["exists"] else "normal"
)

df["stats_parsed"] = df["statistics"].apply(parse_col)


def flatten_stats(stats_dict):
    row = {}
    for kpi, stats in stats_dict.items():
        for stat_name, value in stats.items():
            row[f"{kpi}__{stat_name}"] = value
    return row


X = pd.DataFrame(df["stats_parsed"].apply(flatten_stats).tolist())

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

y_binary = df["label_binary"]
y_type = df["label_type"]

X_train, X_test, yb_train, yb_test, yt_train, yt_test = train_test_split(
    X,
    y_binary,
    y_type,
    test_size=0.2,
    random_state=42,
    stratify=y_binary,
)

print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")
print(f"\nLabel distribution (train):\n{yb_train.value_counts()}")

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
yb_train.to_csv("yb_train.csv", index=False)
yb_test.to_csv("yb_test.csv", index=False)
yt_train.to_csv("yt_train.csv", index=False)
yt_test.to_csv("yt_test.csv", index=False)

print("\nDataPrep complete.")
