import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


X_train = pd.read_csv("X_train.csv")
yb_train = pd.read_csv("yb_train.csv").squeeze()
yt_train = pd.read_csv("yt_train.csv").squeeze()

print("=== BEFORE SMOTE ===")
print(f"Total training rows : {len(X_train)}")
print(f"Normal  (0)         : {(yb_train == 0).sum()}")
print(f"Anomaly (1)         : {(yb_train == 1).sum()}")
print(
    f"Ratio               : "
    f"{(yb_train == 0).sum() / (yb_train == 1).sum():.1f}x more normal than anomaly"
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

before_counts = yb_train.value_counts().sort_index()
axes[0].bar(["Normal", "Anomaly"], before_counts.values, color=["steelblue", "tomato"])
axes[0].set_title("Before SMOTE")
axes[0].set_ylabel("Number of samples")
for i, value in enumerate(before_counts.values):
    axes[0].text(i, value + 100, str(value), ha="center", fontweight="bold")

print("\nApplying SMOTE... (may take a few seconds)")

smote = SMOTE(random_state=42)
X_train_bal, yb_train_bal = smote.fit_resample(X_train, yb_train)

print("\n=== AFTER SMOTE ===")
print(f"Total training rows : {len(X_train_bal)}")
print(f"Normal  (0)         : {(yb_train_bal == 0).sum()}")
print(f"Anomaly (1)         : {(yb_train_bal == 1).sum()}")

after_counts = pd.Series(yb_train_bal).value_counts().sort_index()
axes[1].bar(["Normal", "Anomaly"], after_counts.values, color=["steelblue", "tomato"])
axes[1].set_title("After SMOTE")
axes[1].set_ylabel("Number of samples")
for i, value in enumerate(after_counts.values):
    axes[1].text(i, value + 100, str(value), ha="center", fontweight="bold")

plt.suptitle("Class distribution â€” Model 1 training data", fontweight="bold")
plt.tight_layout()
plt.savefig("smote_distribution.png")
plt.show()
print("\nPlot saved as smote_distribution.png")

X_train_bal_df = pd.DataFrame(X_train_bal, columns=X_train.columns)
yb_train_bal_s = pd.Series(yb_train_bal, name="label_binary")

X_train_bal_df.to_csv("X_train_balanced.csv", index=False)
yb_train_bal_s.to_csv("yb_train_balanced.csv", index=False)

print("\nSaved: X_train_balanced.csv")
print("Saved: yb_train_balanced.csv")
print("\nStep 1 complete. Ready for model training.")

X_test = pd.read_csv("X_test.csv")
yb_test = pd.read_csv("yb_test.csv").squeeze()

print("\n=== TRAINING MODEL 1 (binary) ===")

model1 = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
)

model1.fit(X_train_bal_df, yb_train_bal_s)
print("Training done.")

print("\n=== MODEL 1 EVALUATION (on real test data) ===")

yb_pred = model1.predict(X_test)
yb_pred_prob = model1.predict_proba(X_test)[:, 1]

print(classification_report(yb_test, yb_pred, target_names=["Normal", "Anomaly"]))
print(f"ROC-AUC Score: {roc_auc_score(yb_test, yb_pred_prob):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay.from_predictions(
    yb_test,
    yb_pred,
    display_labels=["Normal", "Anomaly"],
    ax=axes[0],
    colorbar=False,
)
axes[0].set_title("Model 1 â€” Confusion Matrix")

RocCurveDisplay.from_predictions(yb_test, yb_pred_prob, ax=axes[1])
axes[1].set_title("Model 1 â€” ROC Curve")
axes[1].plot([0, 1], [0, 1], "k--", label="Random baseline")
axes[1].legend()

plt.tight_layout()
plt.savefig("model1_evaluation.png")
plt.show()
print("Plot saved as model1_evaluation.png")

joblib.dump(model1, "model1_binary.pkl")
print("\nSaved: model1_binary.pkl")
print("\nStep 2 complete. Ready for Model 2.")
