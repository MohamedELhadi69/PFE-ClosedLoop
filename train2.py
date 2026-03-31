import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ── Load everything ────────────────────────────────────────────────────────────
X_train_orig  = pd.read_csv("X_train.csv")
yb_train_orig = pd.read_csv("yb_train.csv").squeeze()
yt_train      = pd.read_csv("yt_train.csv").squeeze()

X_test        = pd.read_csv("X_test.csv")
yb_test       = pd.read_csv("yb_test.csv").squeeze()
yt_test       = pd.read_csv("yt_test.csv").squeeze()

# ── Filter to anomalous rows only ─────────────────────────────────────────────
anom_mask_train = yb_train_orig == 1
X_train_anom    = X_train_orig[anom_mask_train]
yt_train_anom   = yt_train[anom_mask_train]

print("=== TRAINING MODEL 2 (anomaly type classifier) ===")
print(f"Anomalous training samples : {len(X_train_anom)}")
print(f"\nClass distribution:\n{yt_train_anom.value_counts()}")

# ── Train ──────────────────────────────────────────────────────────────────────
model2 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                ))
])

model2.fit(X_train_anom, yt_train_anom)
print("\nTraining done.")

# ── Evaluate ───────────────────────────────────────────────────────────────────
anom_mask_test = yb_test == 1
X_test_anom    = X_test[anom_mask_test]
yt_test_anom   = yt_test[anom_mask_test]

yt_pred = model2.predict(X_test_anom)

print("\n=== MODEL 2 EVALUATION (anomalous test samples only) ===")
print(f"Anomalous test samples : {len(X_test_anom)}")
print(classification_report(yt_test_anom, yt_pred))

# ── Confusion matrix ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
ConfusionMatrixDisplay.from_predictions(
    yt_test_anom, yt_pred,
    ax=ax,
    colorbar=False,
    xticks_rotation=45
)
ax.set_title("Model 2 — Anomaly Type Confusion Matrix")
plt.tight_layout()
plt.savefig("model2_evaluation.png")
plt.show()
print("Plot saved as model2_evaluation.png")

# ── Save ───────────────────────────────────────────────────────────────────────
joblib.dump(model2, "model2_type.pkl")
print("\nSaved: model2_type.pkl")
print("\nStep 3 complete. Ready for inference pipeline.")