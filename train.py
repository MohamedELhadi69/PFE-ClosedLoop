import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, roc_auc_score,
                              ConfusionMatrixDisplay, RocCurveDisplay)
import joblib

# ── 1. Load the saved data from DataPrep ───────────────────────────────────────
X_train  = pd.read_csv("X_train.csv")
yb_train = pd.read_csv("yb_train.csv").squeeze()  # squeeze turns it from DataFrame to Series
yt_train = pd.read_csv("yt_train.csv").squeeze()

print("=== BEFORE SMOTE ===")
print(f"Total training rows : {len(X_train)}")
print(f"Normal  (0)         : {(yb_train == 0).sum()}")
print(f"Anomaly (1)         : {(yb_train == 1).sum()}")
print(f"Ratio               : {(yb_train == 0).sum() / (yb_train == 1).sum():.1f}x more normal than anomaly")

# ── 2. Visualize the imbalance before ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left plot — before SMOTE
before_counts = yb_train.value_counts().sort_index()
axes[0].bar(["Normal", "Anomaly"], before_counts.values, color=["steelblue", "tomato"])
axes[0].set_title("Before SMOTE")
axes[0].set_ylabel("Number of samples")
for i, v in enumerate(before_counts.values):
    axes[0].text(i, v + 100, str(v), ha="center", fontweight="bold")

# ── 3. Apply SMOTE ─────────────────────────────────────────────────────────────
# SMOTE = Synthetic Minority Oversampling Technique
# It creates SYNTHETIC new anomaly samples by interpolating between existing ones
# We only apply it to training data — NEVER to test data
print("\nApplying SMOTE... (may take a few seconds)")

smote = SMOTE(random_state=42)
X_train_bal, yb_train_bal = smote.fit_resample(X_train, yb_train)

print("\n=== AFTER SMOTE ===")
print(f"Total training rows : {len(X_train_bal)}")
print(f"Normal  (0)         : {(yb_train_bal == 0).sum()}")
print(f"Anomaly (1)         : {(yb_train_bal == 1).sum()}")

# Right plot — after SMOTE
after_counts = pd.Series(yb_train_bal).value_counts().sort_index()
axes[1].bar(["Normal", "Anomaly"], after_counts.values, color=["steelblue", "tomato"])
axes[1].set_title("After SMOTE")
axes[1].set_ylabel("Number of samples")
for i, v in enumerate(after_counts.values):
    axes[1].text(i, v + 100, str(v), ha="center", fontweight="bold")

plt.suptitle("Class distribution — Model 1 training data", fontweight="bold")
plt.tight_layout()
plt.savefig("smote_distribution.png")
plt.show()
print("\nPlot saved as smote_distribution.png")

# ── 4. Save the balanced data ──────────────────────────────────────────────────
X_train_bal_df = pd.DataFrame(X_train_bal, columns=X_train.columns)
yb_train_bal_s = pd.Series(yb_train_bal, name="label_binary")

X_train_bal_df.to_csv("X_train_balanced.csv", index=False)
yb_train_bal_s.to_csv("yb_train_balanced.csv", index=False)

print("\nSaved: X_train_balanced.csv")
print("Saved: yb_train_balanced.csv")
print("\nStep 1 complete. Ready for model training.")


# ── STEP 2: Train Model 1 (binary classifier) ──────────────────────────────────

# Load test data
X_test   = pd.read_csv("X_test.csv")
yb_test  = pd.read_csv("yb_test.csv").squeeze()

print("\n=== TRAINING MODEL 1 (binary) ===")

# Pipeline: scale features first, then classify
# StandardScaler normalizes all 64 features to the same range
# so no single KPI dominates just because its values are huge (e.g TX_Bytes vs RSRP)
model1 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(
                    n_estimators=200,       # 200 trees — good balance of speed vs accuracy
                    class_weight="balanced", # extra safety on top of SMOTE
                    random_state=42,
                    n_jobs=-1               # use all CPU cores
                ))
])

model1.fit(X_train_bal_df, yb_train_bal_s)
print("Training done.")

# ── Evaluate ───────────────────────────────────────────────────────────────────
print("\n=== MODEL 1 EVALUATION (on real test data) ===")

yb_pred      = model1.predict(X_test)
yb_pred_prob = model1.predict_proba(X_test)[:, 1]  # probability of being anomaly

print(classification_report(yb_test, yb_pred, target_names=["Normal", "Anomaly"]))
print(f"ROC-AUC Score: {roc_auc_score(yb_test, yb_pred_prob):.4f}")

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay.from_predictions(
    yb_test, yb_pred,
    display_labels=["Normal", "Anomaly"],
    ax=axes[0],
    colorbar=False
)
axes[0].set_title("Model 1 — Confusion Matrix")

# ROC curve
RocCurveDisplay.from_predictions(yb_test, yb_pred_prob, ax=axes[1])
axes[1].set_title("Model 1 — ROC Curve")
axes[1].plot([0,1], [0,1], "k--", label="Random baseline")
axes[1].legend()

plt.tight_layout()
plt.savefig("model1_evaluation.png")
plt.show()
print("Plot saved as model1_evaluation.png")

# ── Save Model 1 ───────────────────────────────────────────────────────────────
joblib.dump(model1, "model1_binary.pkl")
print("\nSaved: model1_binary.pkl")
print("\nStep 2 complete. Ready for Model 2.")