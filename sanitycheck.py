import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

LOCAL_PYTHON_PACKAGES = Path(__file__).with_name(".python_packages")
if LOCAL_PYTHON_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PYTHON_PACKAGES))

from runtime_bootstrap import ensure_repo_python

ensure_repo_python()

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score


model1 = joblib.load("model1_binary.pkl")

X_train_bal = pd.read_csv("X_train_balanced.csv")
yb_train_bal = pd.read_csv("yb_train_balanced.csv").squeeze()

X_test = pd.read_csv("X_test.csv")
yb_test = pd.read_csv("yb_test.csv").squeeze()

yb_train_pred = model1.predict(X_train_bal)
yb_test_pred = model1.predict(X_test)

train_acc = (yb_train_pred == yb_train_bal).mean()
test_acc = (yb_test_pred == yb_test).mean()

print("=== OVERFITTING SANITY CHECK ===")
print(f"Training accuracy : {train_acc:.4f}")
print(f"Test accuracy     : {test_acc:.4f}")
print(f"Gap               : {abs(train_acc - test_acc):.4f}")

if abs(train_acc - test_acc) < 0.02:
    print(">> Gap is tiny â€” model is generalizing well, NOT overfitting")
else:
    print(">> Gap is large â€” possible overfitting, investigate further")

print("\n--- Training set report ---")
print(
    classification_report(
        yb_train_bal,
        yb_train_pred,
        target_names=["Normal", "Anomaly"],
    )
)

print("--- Test set report ---")
print(
    classification_report(
        yb_test,
        yb_test_pred,
        target_names=["Normal", "Anomaly"],
    )
)
