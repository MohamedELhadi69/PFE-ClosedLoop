import joblib
import pandas as pd

from data_utils import load_or_fetch_dataframe, parse_col


model1 = joblib.load("model1_binary.pkl")
model2 = joblib.load("model2_type.pkl")


def predict(stats_dict):
    row = {}
    for kpi, stats in stats_dict.items():
        for stat_name, value in stats.items():
            row[f"{kpi}__{stat_name}"] = value

    X = pd.DataFrame([row])

    anomaly_prob = model1.predict_proba(X)[0][1]
    is_anomaly = anomaly_prob >= 0.5

    if not is_anomaly:
        return {
            "anomaly": False,
            "confidence": round(float(1 - anomaly_prob), 4),
            "type": None,
            "type_confidence": None,
        }

    anomaly_type = model2.predict(X)[0]
    type_probs = dict(zip(model2.classes_, model2.predict_proba(X)[0]))
    top_prob = max(type_probs.values())

    return {
        "anomaly": True,
        "confidence": round(float(anomaly_prob), 4),
        "type": anomaly_type,
        "type_confidence": round(float(top_prob), 4),
        "type_distribution": {
            k: round(float(v), 4)
            for k, v in sorted(type_probs.items(), key=lambda x: x[1], reverse=True)
        },
    }


def run_pipeline_test():
    df = load_or_fetch_dataframe()

    df["stats_parsed"] = df["statistics"].apply(parse_col)
    df["anomalies_parsed"] = df["anomalies"].apply(parse_col)
    df["true_binary"] = df["anomalies_parsed"].apply(lambda x: int(x["exists"]) if x else 0)
    df["true_type"] = df["anomalies_parsed"].apply(
        lambda x: x["type"] if x and x["exists"] else "normal"
    )

    normal_samples = df[df["true_binary"] == 0].sample(5, random_state=42)
    anomaly_samples = df[df["true_binary"] == 1].sample(5, random_state=42)
    test_samples = pd.concat([normal_samples, anomaly_samples])

    print("=" * 65)
    print("INFERENCE PIPELINE TEST")
    print("=" * 65)

    correct = 0
    for _, row in test_samples.iterrows():
        result = predict(row["stats_parsed"])
        true_label = row["true_type"]
        predicted = result["type"] if result["anomaly"] else "normal"

        if row["true_binary"] == 0:
            is_correct = not result["anomaly"]
        else:
            is_correct = result["anomaly"] and result["type"] == true_label

        correct += int(is_correct)
        status = "OK" if is_correct else "X"

        print(f"\n[{status}] True label : {true_label}")
        print(f"    Predicted  : {predicted}")
        print(f"    Confidence : {result['confidence']}")
        if result["anomaly"]:
            print(f"    Type conf  : {result['type_confidence']}")
            print("    Top 3 types:")
            for k, v in list(result["type_distribution"].items())[:3]:
                bar = "#" * int(v * 20)
                print(f"      {bar:<20} {v:.2f}  {k}")

    print("\n" + "=" * 65)
    print(f"Correct: {correct}/10")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline_test()
