import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_CSV = Path("data.csv")
INPUT_SIZE = 96
HORIZON = 32
VALID_RATIO = 0.2
TEST_RATIO = 0.2
RANDOM_SEED = 42
MODEL_DIR = Path("model3_persistence_model")


def parse_col(value):
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return None
    try:
        return eval(value, {"array": np.array, "np": np})
    except Exception:
        return None


def load_dataframe():
    if DATA_CSV.exists():
        print(f"Loading local dataset from {DATA_CSV}...")
        return pd.read_csv(DATA_CSV)

    print("Local data.csv not found. Falling back to Hugging Face dataset...")
    from datasets import load_dataset

    dataset = load_dataset("AliMaatouk/TelecomTS")
    return dataset["train"].to_pandas()


def build_anomaly_mask(anomaly_info, window_size=INPUT_SIZE + HORIZON):
    mask = np.zeros(window_size, dtype=np.float32)
    if not anomaly_info or not anomaly_info.get("exists"):
        return mask

    duration = anomaly_info.get("anomaly_duration") or {}
    start = int(duration.get("start", 0))
    end = int(duration.get("end", start))
    start = max(0, min(start, window_size - 1))
    end = max(start, min(end, window_size - 1))
    mask[start : end + 1] = 1.0
    return mask


def is_numeric_series(values):
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return isinstance(values, list) and values and all(
        isinstance(v, (int, float, np.integer, np.floating)) for v in values
    )


def trailing_run(values, target_value):
    count = 0
    for value in values[::-1]:
        if int(value) == target_value:
            count += 1
        else:
            break
    return count


def leading_run(values, target_value):
    count = 0
    for value in values:
        if int(value) == target_value:
            count += 1
        else:
            break
    return count


def assign_session_ids(df):
    sorted_df = df.sort_values("start_time").copy()
    previous_end = sorted_df["end_time"].shift(1)
    gap_seconds = (sorted_df["start_time"] - previous_end).dt.total_seconds()
    new_session = previous_end.isna() | (gap_seconds > 0)
    sorted_df["session_id"] = new_session.cumsum().astype(int)
    return sorted_df.sort_index()


def select_tail_sessions(session_summary, target_windows):
    running = 0
    selected = []
    for row in session_summary.iloc[::-1].itertuples():
        selected.append(row.session_id)
        running += row.window_count
        if running >= target_windows:
            break
    return set(selected)


def split_by_session(feature_df, valid_ratio, test_ratio):
    session_summary = (
        feature_df.groupby("session_id")
        .agg(
            session_start=("start_time", "min"),
            session_end=("end_time", "max"),
            window_count=("row_id", "count"),
        )
        .sort_values("session_start")
        .reset_index()
    )

    test_target = max(1, int(round(len(feature_df) * test_ratio)))
    test_sessions = select_tail_sessions(session_summary, test_target)

    remaining_summary = session_summary[~session_summary["session_id"].isin(test_sessions)].copy()
    remaining_df = feature_df[~feature_df["session_id"].isin(test_sessions)].copy()
    valid_target = max(1, int(round(len(remaining_df) * valid_ratio)))
    valid_sessions = select_tail_sessions(remaining_summary, valid_target)

    train_sessions = set(session_summary["session_id"]) - test_sessions - valid_sessions
    if not train_sessions or not valid_sessions or not test_sessions:
        raise ValueError("Session split failed to produce train/valid/test partitions.")

    train_df = feature_df[feature_df["session_id"].isin(train_sessions)].copy()
    valid_df = feature_df[feature_df["session_id"].isin(valid_sessions)].copy()
    test_df = feature_df[feature_df["session_id"].isin(test_sessions)].copy()
    return train_df, valid_df, test_df, session_summary


def build_feature_table(df):
    records = []
    first_kpis = df.iloc[0]["KPIs"]
    kpi_columns = [name for name, values in first_kpis.items() if is_numeric_series(values)]

    anomalous_df = df[df["anomalies"].apply(lambda x: bool(x and x.get("exists")))].copy()
    anomalous_df = anomalous_df.reset_index(drop=True)
    anomalous_df["row_id"] = anomalous_df.index.astype(int)
    anomalous_df["start_time"] = pd.to_datetime(anomalous_df["start_time"], errors="coerce")
    anomalous_df["end_time"] = pd.to_datetime(anomalous_df["end_time"], errors="coerce")
    anomalous_df = assign_session_ids(anomalous_df)

    for row in anomalous_df.itertuples(index=False):
        labels = row.labels
        anomaly = row.anomalies
        anomaly_mask = build_anomaly_mask(anomaly)
        history_mask = anomaly_mask[:INPUT_SIZE]
        future_mask = anomaly_mask[INPUT_SIZE:]
        affected_kpis = anomaly.get("affected_kpis")
        affected_set = set(affected_kpis.tolist()) if hasattr(affected_kpis, "tolist") else set(affected_kpis or [])

        record = {
            "unique_id": f"anom_{int(row.row_id)}",
            "row_id": int(row.row_id),
            "session_id": int(row.session_id),
            "start_time": row.start_time,
            "end_time": row.end_time,
            "anomaly_type": anomaly.get("type", "unknown"),
            "zone": labels.get("zone", "unknown"),
            "application": labels.get("application", "unknown"),
            "mobility": labels.get("mobility", "unknown"),
            "congestion": labels.get("congestion", "unknown"),
            "sampling_rate": float(row.sampling_rate),
            "affected_kpi_count": float(len(affected_set)),
            "history_sum": float(history_mask.sum()),
            "history_mean": float(history_mask.mean()),
            "history_last_active": int(history_mask[-1]),
            "history_last8_sum": float(history_mask[-8:].sum()),
            "history_last16_sum": float(history_mask[-16:].sum()),
            "history_last32_sum": float(history_mask[-32:].sum()),
            "history_trailing_ones": float(trailing_run(history_mask, 1)),
            "history_trailing_zeros": float(trailing_run(history_mask, 0)),
            "history_leading_ones": float(leading_run(history_mask, 1)),
            "history_first_active_idx": float(np.argmax(history_mask) if history_mask.sum() else -1),
            "history_last_active_idx": float(
                INPUT_SIZE - 1 - np.argmax(history_mask[::-1]) if history_mask.sum() else -1
            ),
            "history_transition_count": float(np.abs(np.diff(history_mask)).sum()),
            "future_count": int(future_mask.sum()),
        }

        for kpi_name in kpi_columns:
            series = np.array(row.KPIs[kpi_name], dtype=np.float32)
            history = series[:INPUT_SIZE]
            record[f"{kpi_name}__mean"] = float(history.mean())
            record[f"{kpi_name}__std"] = float(history.std())
            record[f"{kpi_name}__min"] = float(history.min())
            record[f"{kpi_name}__max"] = float(history.max())
            record[f"{kpi_name}__start"] = float(history[0])
            record[f"{kpi_name}__end"] = float(history[-1])
            record[f"{kpi_name}__delta"] = float(history[-1] - history[0])
            record[f"{kpi_name}__trend"] = float((history[-1] - history[0]) / max(len(history) - 1, 1))
            record[f"{kpi_name}__last8_mean"] = float(history[-8:].mean())
            record[f"{kpi_name}__last16_mean"] = float(history[-16:].mean())
            record[f"{kpi_name}__last32_mean"] = float(history[-32:].mean())
            record[f"affected_{kpi_name}"] = float(kpi_name in affected_set)

        records.append(record)

    return pd.DataFrame(records), kpi_columns


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def count_to_regime(count):
    if count == 0:
        return "zero"
    if count == HORIZON:
        return "full"
    return "partial"


def evaluate_predictions(actual, predicted):
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    abs_error = np.abs(actual - predicted)
    actual_regime = np.array([count_to_regime(value) for value in actual])
    predicted_regime = np.array([count_to_regime(value) for value in predicted])

    return {
        "exact_match": float((abs_error == 0).mean()),
        "within_1": float((abs_error <= 1).mean()),
        "within_2": float((abs_error <= 2).mean()),
        "within_4": float((abs_error <= 4).mean()),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "regime_accuracy": float(accuracy_score(actual_regime, predicted_regime)),
        "regime_macro_f1": float(f1_score(actual_regime, predicted_regime, average="macro")),
    }


def print_target_distribution(name, df):
    counts = df["future_count"].value_counts().sort_index()
    top_counts = counts.sort_values(ascending=False).head(10).to_dict()
    print(f"{name} windows               : {len(df)}")
    print(f"{name} sessions              : {df['session_id'].nunique()}")
    print(f"{name} target top counts     : {top_counts}")


class HeuristicBoundaryModel:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.where(X["history_last_active"].to_numpy() >= 0.5, HORIZON, 0).astype(int)


class ExactCountForestModel:
    def __init__(self, estimator):
        self.estimator = estimator
        self.pipeline = None

    def fit(self, X, y, numeric_features, categorical_features):
        self.pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("model", self.estimator),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X).astype(int)


class HybridPersistenceModel:
    def __init__(self):
        self.regime_pipeline = None
        self.partial_pipeline = None
        self.partial_fallback = HORIZON // 2

    def fit(self, X, y, numeric_features, categorical_features):
        regime_y = pd.Series([count_to_regime(value) for value in y], index=y.index)
        self.regime_pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        self.regime_pipeline.fit(X, regime_y)

        partial_mask = (y > 0) & (y < HORIZON)
        if partial_mask.sum() > 0:
            self.partial_fallback = int(np.rint(y[partial_mask].median()))
            self.partial_pipeline = Pipeline(
                [
                    ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=500,
                            random_state=RANDOM_SEED,
                            n_jobs=1,
                        ),
                    ),
                ]
            )
            self.partial_pipeline.fit(X[partial_mask], y[partial_mask])
        return self

    def predict(self, X):
        predicted_regime = self.regime_pipeline.predict(X)
        predictions = np.full(len(X), HORIZON, dtype=int)
        predictions[predicted_regime == "zero"] = 0

        partial_mask = predicted_regime == "partial"
        if partial_mask.any():
            if self.partial_pipeline is None:
                partial_predictions = np.full(partial_mask.sum(), self.partial_fallback, dtype=int)
            else:
                partial_predictions = np.rint(self.partial_pipeline.predict(X[partial_mask])).astype(int)
            predictions[partial_mask] = np.clip(partial_predictions, 1, HORIZON - 1)

        return predictions


def fit_candidate(name, X_train, y_train, numeric_features, categorical_features):
    if name == "heuristic_boundary":
        model = HeuristicBoundaryModel().fit(X_train, y_train)
    elif name == "exact_extra_trees":
        model = ExactCountForestModel(
            ExtraTreesClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=1,
            )
        ).fit(X_train, y_train, numeric_features, categorical_features)
    elif name == "hybrid_boundary_forest":
        model = HybridPersistenceModel().fit(X_train, y_train, numeric_features, categorical_features)
    else:
        raise ValueError(f"Unknown candidate model: {name}")
    return model


def build_leaderboard(candidate_names, X_train, y_train, X_eval, y_eval, numeric_features, categorical_features):
    leaderboard_rows = []
    models = {}
    for name in candidate_names:
        model = fit_candidate(name, X_train, y_train, numeric_features, categorical_features)
        predictions = model.predict(X_eval)
        metrics = evaluate_predictions(y_eval, predictions)
        leaderboard_rows.append({"model": name, **metrics})
        models[name] = model

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(
        ["within_2", "exact_match", "mae"],
        ascending=[False, False, True],
    )
    return leaderboard_df.reset_index(drop=True), models


def print_metric_block(title, metrics):
    print(f"\n=== {title} ===")
    print(f"Exact match           : {metrics['exact_match']:.4%}")
    print(f"Within +-1 timestep   : {metrics['within_1']:.4%}")
    print(f"Within +-2 timesteps  : {metrics['within_2']:.4%}")
    print(f"Within +-4 timesteps  : {metrics['within_4']:.4%}")
    print(f"MAE                   : {metrics['mae']:.4f}")
    print(f"RMSE                  : {metrics['rmse']:.4f}")
    print(f"Regime accuracy       : {metrics['regime_accuracy']:.4%}")
    print(f"Regime macro F1       : {metrics['regime_macro_f1']:.4f}")


def main():
    print("Loading dataset...")
    df = load_dataframe()
    df["KPIs"] = df["KPIs"].apply(parse_col)
    df["anomalies"] = df["anomalies"].apply(parse_col)
    df["labels"] = df["labels"].apply(parse_col)
    df = df[df["KPIs"].notna() & df["anomalies"].notna() & df["labels"].notna()].reset_index(drop=True)

    print("Building persistence feature table...")
    feature_df, kpi_columns = build_feature_table(df)
    train_df, valid_df, test_df, session_summary = split_by_session(feature_df, VALID_RATIO, TEST_RATIO)

    feature_columns = [
        column
        for column in feature_df.columns
        if column not in {"unique_id", "row_id", "session_id", "start_time", "end_time", "future_count"}
    ]
    categorical_features = ["anomaly_type", "zone", "application", "mobility", "congestion"]
    numeric_features = [column for column in feature_columns if column not in categorical_features]

    print(f"Anomalous windows total    : {len(feature_df)}")
    print(f"Total sessions             : {feature_df['session_id'].nunique()}")
    print(f"Numeric KPIs used          : {len(kpi_columns)}")
    print(
        f"Session time span          : "
        f"{session_summary['session_start'].min()} -> {session_summary['session_end'].max()}"
    )
    print_target_distribution("Train", train_df)
    print_target_distribution("Valid", valid_df)
    print_target_distribution("Test", test_df)

    X_train = train_df[feature_columns]
    y_train = train_df["future_count"]
    X_valid = valid_df[feature_columns]
    y_valid = valid_df["future_count"]
    X_test = test_df[feature_columns]
    y_test = test_df["future_count"]

    candidate_names = [
        "heuristic_boundary",
        "exact_extra_trees",
        "hybrid_boundary_forest",
    ]
    validation_leaderboard, _ = build_leaderboard(
        candidate_names,
        X_train,
        y_train,
        X_valid,
        y_valid,
        numeric_features,
        categorical_features,
    )

    print("\n=== VALIDATION LEADERBOARD ===")
    print(validation_leaderboard.to_string(index=False))

    selected_model_name = validation_leaderboard.iloc[0]["model"]
    print(f"\nSelected model            : {selected_model_name}")

    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    X_train_valid = train_valid_df[feature_columns]
    y_train_valid = train_valid_df["future_count"]

    test_rows = []
    saved_model = None
    for candidate_name in candidate_names:
        model = fit_candidate(
            candidate_name,
            X_train_valid,
            y_train_valid,
            numeric_features,
            categorical_features,
        )
        predictions = model.predict(X_test)
        metrics = evaluate_predictions(y_test, predictions)
        test_rows.append({"model": candidate_name, **metrics})

        if candidate_name == selected_model_name:
            saved_model = model
            selected_predictions = predictions

    test_leaderboard = pd.DataFrame(test_rows).sort_values(
        ["within_2", "exact_match", "mae"],
        ascending=[False, False, True],
    )

    print("\n=== TEST COMPARISON ===")
    print(test_leaderboard.to_string(index=False))

    selected_metrics = evaluate_predictions(y_test, selected_predictions)
    print_metric_block("SELECTED MODEL TEST METRICS", selected_metrics)

    actual_regime = np.array([count_to_regime(value) for value in y_test])
    predicted_regime = np.array([count_to_regime(value) for value in selected_predictions])
    regime_labels = ["zero", "partial", "full"]

    print("\n=== REGIME CLASSIFICATION REPORT ===")
    print(classification_report(actual_regime, predicted_regime, labels=regime_labels, zero_division=0))

    regime_matrix = confusion_matrix(actual_regime, predicted_regime, labels=regime_labels)
    regime_matrix_df = pd.DataFrame(regime_matrix, index=regime_labels, columns=regime_labels)
    print("\n=== REGIME CONFUSION MATRIX ===")
    print(regime_matrix_df.to_string())

    predictions_df = test_df[
        ["unique_id", "row_id", "session_id", "anomaly_type", "zone", "application", "history_sum", "future_count"]
    ].copy()
    predictions_df["predicted_future_count"] = selected_predictions.astype(int)
    predictions_df["abs_error"] = (predictions_df["future_count"] - predictions_df["predicted_future_count"]).abs()
    predictions_df["actual_regime"] = predictions_df["future_count"].map(count_to_regime)
    predictions_df["predicted_regime"] = predictions_df["predicted_future_count"].map(count_to_regime)

    per_type_df = (
        predictions_df.groupby("anomaly_type")
        .apply(
            lambda part: pd.Series(
                {
                    "rows": len(part),
                    "exact_match": (part["abs_error"] == 0).mean(),
                    "within_2": (part["abs_error"] <= 2).mean(),
                    "mae": part["abs_error"].mean(),
                }
            )
        )
        .reset_index()
        .sort_values(["within_2", "exact_match", "rows"], ascending=[False, False, False])
    )

    print("\n=== PER-ANOMALY-TYPE TEST METRICS ===")
    print(per_type_df.to_string(index=False))

    worst_errors_df = predictions_df.sort_values(["abs_error", "future_count"], ascending=[False, False]).head(15)
    print("\n=== HARDEST TEST CASES ===")
    print(worst_errors_df.to_string(index=False))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(saved_model, MODEL_DIR / "model.joblib")
    validation_leaderboard.to_csv(MODEL_DIR / "validation_leaderboard.csv", index=False)
    test_leaderboard.to_csv(MODEL_DIR / "test_comparison.csv", index=False)
    predictions_df.to_csv(MODEL_DIR / "test_predictions.csv", index=False)
    per_type_df.to_csv(MODEL_DIR / "per_anomaly_type_metrics.csv", index=False)

    metadata = {
        "selected_model": selected_model_name,
        "input_size": INPUT_SIZE,
        "horizon": HORIZON,
        "random_seed": RANDOM_SEED,
        "train_windows": int(len(train_df)),
        "valid_windows": int(len(valid_df)),
        "test_windows": int(len(test_df)),
        "train_sessions": int(train_df["session_id"].nunique()),
        "valid_sessions": int(valid_df["session_id"].nunique()),
        "test_sessions": int(test_df["session_id"].nunique()),
        "kpi_columns": kpi_columns,
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "selected_model_metrics": selected_metrics,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved model and reports to: {MODEL_DIR}")
    print("Step 4 complete. Persistence model is ready.")


if __name__ == "__main__":
    main()