import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score

# ── Load everything ────────────────────────────────────────────────────────────
model1       = joblib.load("model1_binary.pkl")

X_train_bal  = pd.read_csv("X_train_balanced.csv")
yb_train_bal = pd.read_csv("yb_train_balanced.csv").squeeze()

X_test       = pd.read_csv("X_test.csv")
yb_test      = pd.read_csv("yb_test.csv").squeeze()

# ── Compare train vs test performance ─────────────────────────────────────────
yb_train_pred = model1.predict(X_train_bal)
yb_test_pred  = model1.predict(X_test)

train_acc = (yb_train_pred == yb_train_bal).mean()
test_acc  = (yb_test_pred  == yb_test).mean()

print("=== OVERFITTING SANITY CHECK ===")
print(f"Training accuracy : {train_acc:.4f}")
print(f"Test accuracy     : {test_acc:.4f}")
print(f"Gap               : {abs(train_acc - test_acc):.4f}")

if abs(train_acc - test_acc) < 0.02:
    print(">> Gap is tiny — model is generalizing well, NOT overfitting")
else:
    print(">> Gap is large — possible overfitting, investigate further")

# ── Full report on both ────────────────────────────────────────────────────────
print("\n--- Training set report ---")
print(classification_report(yb_train_bal, yb_train_pred, target_names=["Normal", "Anomaly"]))

print("--- Test set report ---")
print(classification_report(yb_test, yb_test_pred, target_names=["Normal", "Anomaly"]))