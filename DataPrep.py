import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

LOCAL_PYTHON_PACKAGES = Path(__file__).with_name(".python_packages")
if LOCAL_PYTHON_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PYTHON_PACKAGES))

from runtime_bootstrap import ensure_repo_python

ensure_repo_python()

import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils import load_or_fetch_dataframe, parse_col
from split_utils import (
    MODEL_TRAIN_SPLIT_NAME,
    TEST_SPLIT_NAME,
    SPLIT_INDEX_FILES,
    save_row_indices,
    save_split_manifest,
)


MODEL_TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
RANDOM_SEED = 42


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
row_indices = pd.Series(df.index, name="row_index")

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

y_binary = df["label_binary"]
y_type = df["label_type"]

X_train, X_test, yb_train, yb_test, yt_train, yt_test, row_train, row_test = train_test_split(
    X,
    y_binary,
    y_type,
    row_indices,
    test_size=TEST_RATIO,
    random_state=RANDOM_SEED,
    stratify=y_binary,
)

print(f"\nModel-train: {len(X_train)} rows | Test: {len(X_test)} rows")
print(f"\nLabel distribution (model-train):\n{yb_train.value_counts()}")
print(f"\nLabel distribution (test):\n{yb_test.value_counts()}")

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
yb_train.to_csv("yb_train.csv", index=False)
yb_test.to_csv("yb_test.csv", index=False)
yt_train.to_csv("yt_train.csv", index=False)
yt_test.to_csv("yt_test.csv", index=False)

save_row_indices(SPLIT_INDEX_FILES[MODEL_TRAIN_SPLIT_NAME], row_train.tolist())
save_row_indices(SPLIT_INDEX_FILES[TEST_SPLIT_NAME], row_test.tolist())
save_split_manifest(
    {
        "random_seed": RANDOM_SEED,
        "ratios": {
            MODEL_TRAIN_SPLIT_NAME: MODEL_TRAIN_RATIO,
            TEST_SPLIT_NAME: TEST_RATIO,
        },
        "counts": {
            MODEL_TRAIN_SPLIT_NAME: int(len(X_train)),
            TEST_SPLIT_NAME: int(len(X_test)),
        },
        "index_files": {name: str(path) for name, path in SPLIT_INDEX_FILES.items()},
    }
)

print(
    "Saved row-index manifests: "
    f"{SPLIT_INDEX_FILES[MODEL_TRAIN_SPLIT_NAME].name}, "
    f"{SPLIT_INDEX_FILES[TEST_SPLIT_NAME].name}"
)
print("\nDataPrep complete.")
