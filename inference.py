import json
from pathlib import Path
import __main__

import joblib
import numpy as np
import pandas as pd

import train3
from data_utils import load_or_fetch_dataframe, parse_col


MODEL3_DIR = Path("model3_persistence_model")
MODEL3_PATH = MODEL3_DIR / "model.joblib"
MODEL3_METADATA_PATH = MODEL3_DIR / "metrics.json"


# Joblib may need these class names on __main__ when loading a model saved by train3.py.
__main__.HeuristicBoundaryModel = train3.HeuristicBoundaryModel
__main__.ExactCountForestModel = train3.ExactCountForestModel
__main__.HybridPersistenceModel = train3.HybridPersistenceModel

model1 = joblib.load("model1_binary.pkl")
model2 = joblib.load("model2_type.pkl")
model3 = joblib.load(MODEL3_PATH) if MODEL3_PATH.exists() else None
model3_metadata = json.loads(MODEL3_METADATA_PATH.read_text()) if MODEL3_METADATA_PATH.exists() else {}


def build_stage1_features(stats_dict):
    row = {}
    for kpi, stats in stats_dict.items():
        for stat_name, value in stats.items():
            row[f"{kpi}__{stat_name}"] = value
    return pd.DataFrame([row])


def normalize_sample(sample):
    if isinstance(sample, pd.Series):
        sample = sample.to_dict()

    if not isinstance(sample, dict):
        raise TypeError("predict expects either a stats dict or a full sample dict/Series.")

    if "statistics" not in sample and "stats_parsed" not in sample:
        return {"statistics": sample}

    normalized = dict(sample)
    for key in ("statistics", "stats_parsed", "KPIs", "labels", "anomalies"):
        if key in normalized:
            normalized[key] = parse_col(normalized[key])
    return normalized


def extract_statistics(sample):
    stats = sample.get("stats_parsed") or sample.get("statistics")
    if stats is None:
        raise ValueError("Statistics are required for model 1 inference.")
    return stats


def build_stage3_features(sample, predicted_type):
    if model3 is None:
        return None, "model3 artifact not found"

    kpis = sample.get("KPIs")
    labels = sample.get("labels")
    if not isinstance(kpis, dict) or not isinstance(labels, dict):
        return None, "KPIs and labels are required for model 3 inference"

    anomaly_info = sample.get("anomalies")
    anomaly_mask = train3.build_anomaly_mask(anomaly_info) if anomaly_info else np.zeros(
        train3.INPUT_SIZE + train3.HORIZON, dtype=np.float32
    )
    history_mask = anomaly_mask[: train3.INPUT_SIZE]

    selected_kpis = model3_metadata.get("kpi_columns") or [
        name for name, values in kpis.items() if train3.is_numeric_series(values)
    ]
    affected_kpis = anomaly_info.get("affected_kpis") if anomaly_info else []
    affected_set = set(affected_kpis.tolist()) if hasattr(affected_kpis, "tolist") else set(affected_kpis or [])

    record = {
        "anomaly_type": predicted_type,
        "zone": labels.get("zone", "unknown"),
        "application": labels.get("application", "unknown"),
        "mobility": labels.get("mobility", "unknown"),
        "congestion": labels.get("congestion", "unknown"),
        "sampling_rate": float(sample.get("sampling_rate", 0.0) or 0.0),
        "affected_kpi_count": float(len(affected_set)),
        "history_sum": float(history_mask.sum()),
        "history_mean": float(history_mask.mean()),
        "history_last_active": int(history_mask[-1]),
        "history_last8_sum": float(history_mask[-8:].sum()),
        "history_last16_sum": float(history_mask[-16:].sum()),
        "history_last32_sum": float(history_mask[-32:].sum()),
        "history_trailing_ones": float(train3.trailing_run(history_mask, 1)),
        "history_trailing_zeros": float(train3.trailing_run(history_mask, 0)),
        "history_leading_ones": float(train3.leading_run(history_mask, 1)),
        "history_first_active_idx": float(np.argmax(history_mask) if history_mask.sum() else -1),
        "history_last_active_idx": float(
            train3.INPUT_SIZE - 1 - np.argmax(history_mask[::-1]) if history_mask.sum() else -1
        ),
        "history_transition_count": float(np.abs(np.diff(history_mask)).sum()),
    }

    for kpi_name in selected_kpis:
        series = kpis.get(kpi_name)
        if not train3.is_numeric_series(series):
            raise ValueError(f"KPI '{kpi_name}' is missing or invalid for model 3 inference.")

        history = np.array(series[: train3.INPUT_SIZE], dtype=np.float32)
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

    return pd.DataFrame([record]), None


def predict(sample):
    sample = normalize_sample(sample)
    stats_dict = extract_statistics(sample)
    X_stage1 = build_stage1_features(stats_dict)

    anomaly_prob = float(model1.predict_proba(X_stage1)[0][1])
    is_anomaly = anomaly_prob >= 0.5

    result = {
        "anomaly": bool(is_anomaly),
        "confidence": round(anomaly_prob if is_anomaly else 1 - anomaly_prob, 4),
        "type": None,
        "type_confidence": None,
        "type_distribution": None,
        "duration_steps": None,
        "duration_regime": None,
        "duration_model": model3_metadata.get("selected_model"),
        "duration_ready": False,
    }

    if not is_anomaly:
        return result

    anomaly_type = model2.predict(X_stage1)[0]
    type_probs = dict(zip(model2.classes_, model2.predict_proba(X_stage1)[0]))
    top_prob = max(type_probs.values())

    result["type"] = anomaly_type
    result["type_confidence"] = round(float(top_prob), 4)
    result["type_distribution"] = {
        k: round(float(v), 4)
        for k, v in sorted(type_probs.items(), key=lambda x: x[1], reverse=True)
    }

    X_stage3, stage3_error = build_stage3_features(sample, anomaly_type)
    if stage3_error:
        result["duration_reason"] = stage3_error
        return result

    duration_steps = int(model3.predict(X_stage3)[0])
    result["duration_steps"] = duration_steps
    result["duration_regime"] = train3.count_to_regime(duration_steps)
    result["duration_ready"] = True
    return result


def true_future_count(anomaly_info):
    if not anomaly_info or not anomaly_info.get("exists"):
        return 0
    return int(train3.build_anomaly_mask(anomaly_info)[train3.INPUT_SIZE :].sum())


def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def run_pipeline_test():
    df = load_or_fetch_dataframe()

    df["stats_parsed"] = df["statistics"].apply(parse_col)
    df["anomalies_parsed"] = df["anomalies"].apply(parse_col)
    df["labels_parsed"] = df["labels"].apply(parse_col)
    df["kpis_parsed"] = df["KPIs"].apply(parse_col)
    df["true_binary"] = df["anomalies_parsed"].apply(lambda x: int(x["exists"]) if x else 0)
    df["true_type"] = df["anomalies_parsed"].apply(
        lambda x: x["type"] if x and x["exists"] else "normal"
    )
    df["true_future_count"] = df["anomalies_parsed"].apply(true_future_count)

    print("=" * 75)
    print("SEQUENTIAL INFERENCE FULL-DATASET SHOWCASE")
    print("=" * 75)

    total_rows = len(df)
    total_normals = int((df["true_binary"] == 0).sum())
    total_anomalies = int((df["true_binary"] == 1).sum())
    print(f"Total rows           : {total_rows}")
    print(f"Normal rows          : {total_normals}")
    print(f"Anomaly rows         : {total_anomalies}")
    print(f"Model 3 selected     : {model3_metadata.get('selected_model', 'missing')}")

    stage1_correct = 0
    stage2_total = 0
    stage2_correct = 0
    stage12_correct = 0
    duration_total = 0
    duration_exact = 0
    duration_within_1 = 0
    duration_within_2 = 0
    duration_abs_errors = []
    binary_false_positive = 0
    binary_false_negative = 0
    type_failures = []
    duration_failures = []

    for row_idx, row in df.iterrows():
        sample = {
            "statistics": row["stats_parsed"],
            "KPIs": row["kpis_parsed"],
            "labels": row["labels_parsed"],
            "anomalies": row["anomalies_parsed"],
            "sampling_rate": row.get("sampling_rate", 0.0),
        }
        result = predict(sample)

        true_binary = int(row["true_binary"])
        true_type = row["true_type"]
        predicted_binary = int(result["anomaly"])
        predicted_type = result["type"] if result["anomaly"] else "normal"

        stage1_ok = predicted_binary == true_binary
        stage1_correct += int(stage1_ok)

        if true_binary == 0 and predicted_binary == 1:
            binary_false_positive += 1
        if true_binary == 1 and predicted_binary == 0:
            binary_false_negative += 1

        if true_binary == 1 and predicted_binary == 1:
            stage2_total += 1
            stage2_ok = predicted_type == true_type
            stage2_correct += int(stage2_ok)
            stage12_correct += int(stage2_ok)
            if not stage2_ok:
                top_types = list((result.get("type_distribution") or {}).items())[:3]
                type_failures.append(
                    {
                        "row": int(row_idx),
                        "true_type": true_type,
                        "predicted_type": predicted_type,
                        "binary_confidence": result["confidence"],
                        "type_confidence": result["type_confidence"],
                        "top_types": top_types,
                    }
                )
        elif true_binary == 0 and predicted_binary == 0:
            stage12_correct += 1

        if true_binary == 1 and result["duration_ready"]:
            duration_total += 1
            abs_error = abs(int(result["duration_steps"]) - int(row["true_future_count"]))
            duration_abs_errors.append(abs_error)
            duration_exact += int(abs_error == 0)
            duration_within_1 += int(abs_error <= 1)
            duration_within_2 += int(abs_error <= 2)
            if abs_error > 0:
                duration_failures.append(
                    {
                        "row": int(row_idx),
                        "true_type": true_type,
                        "predicted_type": predicted_type,
                        "pred_duration": int(result["duration_steps"]),
                        "true_duration": int(row["true_future_count"]),
                        "abs_error": int(abs_error),
                        "duration_regime": result["duration_regime"],
                    }
                )

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(
        f"Stage 1 binary       : {stage1_correct}/{total_rows} "
        f"({safe_divide(stage1_correct, total_rows):.2%})"
    )
    print(
        f"  False positives    : {binary_false_positive} "
        f"({safe_divide(binary_false_positive, total_normals):.2%} of normals)"
    )
    print(
        f"  False negatives    : {binary_false_negative} "
        f"({safe_divide(binary_false_negative, total_anomalies):.2%} of anomalies)"
    )
    print(
        f"Stage 2 type         : {stage2_correct}/{stage2_total} "
        f"({safe_divide(stage2_correct, stage2_total):.2%})"
    )
    print(
        f"Stage 1 + 2 combined : {stage12_correct}/{total_rows} "
        f"({safe_divide(stage12_correct, total_rows):.2%})"
    )
    if duration_total:
        mean_abs_error = float(np.mean(duration_abs_errors))
        print(
            f"Stage 3 exact        : {duration_exact}/{duration_total} "
            f"({safe_divide(duration_exact, duration_total):.2%})"
        )
        print(
            f"Stage 3 within +-1   : {duration_within_1}/{duration_total} "
            f"({safe_divide(duration_within_1, duration_total):.2%})"
        )
        print(
            f"Stage 3 within +-2   : {duration_within_2}/{duration_total} "
            f"({safe_divide(duration_within_2, duration_total):.2%})"
        )
        print(f"Stage 3 MAE          : {mean_abs_error:.4f}")
    else:
        print("Stage 3 exact        : 0/0")

    print("\n" + "=" * 75)
    print("TROUBLESHOOTING SHOWCASE")
    print("=" * 75)

    if type_failures:
        print("\nTop type-classification misses:")
        for failure in type_failures[:10]:
            print(
                f"  Row {failure['row']}: true={failure['true_type']} "
                f"pred={failure['predicted_type']} "
                f"bin_conf={failure['binary_confidence']:.4f} "
                f"type_conf={failure['type_confidence']:.4f}"
            )
            top_types_text = ", ".join([f"{name}={prob:.4f}" for name, prob in failure["top_types"]])
            if top_types_text:
                print(f"    Top guesses: {top_types_text}")
    else:
        print("\nTop type-classification misses:")
        print("  None")

    if duration_failures:
        duration_failures = sorted(duration_failures, key=lambda item: item["abs_error"], reverse=True)
        print("\nTop duration misses:")
        for failure in duration_failures[:10]:
            print(
                f"  Row {failure['row']}: true_type={failure['true_type']} "
                f"pred_type={failure['predicted_type']} "
                f"true_duration={failure['true_duration']} "
                f"pred_duration={failure['pred_duration']} "
                f"abs_error={failure['abs_error']} "
                f"regime={failure['duration_regime']}"
            )
    else:
        print("\nTop duration misses:")
        print("  None")

    print("=" * 75)


if __name__ == "__main__":
    run_pipeline_test()
