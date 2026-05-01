import argparse
import json
import shutil
import subprocess
import time
from typing import Any
from urllib import error, request

import pandas as pd

import inference
from data_utils import load_or_fetch_dataframe, parse_col


DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_OLLAMA_TIMEOUT = 600
OLLAMA_HEALTHCHECK_URL = "http://127.0.0.1:11434/api/tags"
OLLAMA_EXECUTABLE_CANDIDATES = [
    shutil.which("ollama"),
    r"C:\Users\MON PC\AppData\Local\Programs\Ollama\ollama.exe",
]


JSON_OUTPUT_INSTRUCTIONS = (
    "Return only valid JSON with these keys exactly: "
    "severity, urgency, confidence_posture, summary, likely_root_cause, "
    "operational_impact, recommended_actions, verification_steps, escalation_needed, operator_note. "
    "severity must be one of info, low, medium, high, critical. "
    "urgency must be one of monitor, planned, soon, urgent, immediate. "
    "confidence_posture must be one of low, medium, high. "
    "recommended_actions must be an array of 3 to 6 strings. "
    "verification_steps must be an array of 2 to 5 strings. "
    "escalation_needed must be a boolean."
)


def load_dataset_with_parsed_columns():
    df = load_or_fetch_dataframe()
    df["stats_parsed"] = df["statistics"].apply(parse_col)
    df["anomalies_parsed"] = df["anomalies"].apply(parse_col)
    df["labels_parsed"] = df["labels"].apply(parse_col)
    df["kpis_parsed"] = df["KPIs"].apply(parse_col)
    return df


def to_builtin(value: Any):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if hasattr(value, "tolist"):
        return to_builtin(value.tolist())
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def top_kpi_summary(stats_dict, limit=4):
    if not isinstance(stats_dict, dict):
        return {}

    ranked = []
    for kpi_name, stats in stats_dict.items():
        if not isinstance(stats, dict):
            continue
        std_value = float(stats.get("std", 0.0) or 0.0)
        delta_value = abs(float(stats.get("delta", 0.0) or 0.0))
        ranked.append((kpi_name, std_value + delta_value))

    ranked.sort(key=lambda item: item[1], reverse=True)
    selected = [name for name, _ in ranked[:limit]]

    return {
        kpi_name: {
            stat_name: to_builtin(stat_value)
            for stat_name, stat_value in stats_dict[kpi_name].items()
            if stat_name in {"mean", "std", "min", "max", "start", "end", "delta", "trend"}
        }
        for kpi_name in selected
    }


def build_context_payload(row, prediction):
    labels = row.get("labels_parsed") or {}
    anomalies = row.get("anomalies_parsed") or {}
    stats = row.get("stats_parsed") or {}

    affected_kpis = anomalies.get("affected_kpis") if isinstance(anomalies, dict) else []
    if hasattr(affected_kpis, "tolist"):
        affected_kpis = affected_kpis.tolist()

    type_distribution = prediction.get("type_distribution") or {}
    top_type_distribution = dict(list(type_distribution.items())[:3])

    context = {
        "row_index": int(row.name),
        "prediction": {
            "anomaly": bool(prediction["anomaly"]),
            "confidence": prediction["confidence"],
            "anomaly_type": prediction["type"],
            "type_confidence": prediction["type_confidence"],
            "type_distribution_top3": top_type_distribution,
            "predicted_duration_steps": prediction["duration_steps"],
            "duration_regime": prediction["duration_regime"],
            "duration_model": prediction["duration_model"],
            "duration_ready": prediction["duration_ready"],
        },
        "context": {
            "application": labels.get("application"),
            "zone": labels.get("zone"),
            "mobility": labels.get("mobility"),
            "congestion": labels.get("congestion"),
            "sampling_rate": to_builtin(row.get("sampling_rate")),
        },
        "anomaly_details": {
            "known_dataset_label": anomalies.get("type") if anomalies else None,
            "affected_kpis": to_builtin(affected_kpis),
            "has_ground_truth_anomaly": bool(anomalies.get("exists")) if anomalies else False,
        },
        "signal_summary": {
            "top_kpis": top_kpi_summary(stats),
        },
    }
    return context


def build_messages(payload):
    instructions = (
        "You are a telecom operations assistant. "
        "Use the structured model outputs and context to recommend practical operator actions. "
        "Be careful with uncertainty: when confidence is not high, suggest verification before disruptive action. "
        "If anomaly is false, avoid inventing an incident and recommend monitoring only. "
        "Long predicted duration or severe anomaly types should increase urgency. "
        + JSON_OUTPUT_INSTRUCTIONS
    )

    user_prompt = {
        "task": "Produce a troubleshooting recommendation for this telecom event.",
        "requirements": [
            "Base the answer on the prediction and context provided.",
            "Do not restate raw JSON; synthesize it into operational guidance.",
            "Recommended actions should be specific and ordered from most useful to least useful.",
            "Verification steps should help confirm the diagnosis before or during remediation.",
            "Escalation should reflect severity, confidence, and predicted duration.",
        ],
        "event": payload,
    }

    return instructions, user_prompt


def build_prompt(payload):
    instructions, user_prompt = build_messages(payload)
    return f"{instructions}\n\nEVENT:\n{json.dumps(user_prompt, ensure_ascii=True, indent=2)}"


def validate_recommendation(recommendation):
    required_keys = {
        "severity",
        "urgency",
        "confidence_posture",
        "summary",
        "likely_root_cause",
        "operational_impact",
        "recommended_actions",
        "verification_steps",
        "escalation_needed",
        "operator_note",
    }
    missing = required_keys - set(recommendation.keys())
    if missing:
        raise ValueError(f"Recommendation JSON is missing keys: {sorted(missing)}")

    if not isinstance(recommendation["recommended_actions"], list):
        raise ValueError("recommended_actions must be a list.")
    if not isinstance(recommendation["verification_steps"], list):
        raise ValueError("verification_steps must be a list.")
    if not isinstance(recommendation["escalation_needed"], bool):
        raise ValueError("escalation_needed must be a boolean.")


def wait_for_ollama_server(timeout_seconds=30):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with request.urlopen(OLLAMA_HEALTHCHECK_URL, timeout=5) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def ensure_ollama_running():
    if wait_for_ollama_server(timeout_seconds=2):
        return

    executable = next((path for path in OLLAMA_EXECUTABLE_CANDIDATES if path), None)
    if not executable:
        raise RuntimeError("Could not find an Ollama executable on this machine.")

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [executable, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    if not wait_for_ollama_server(timeout_seconds=30):
        raise RuntimeError(
            "Ollama was found but the local server did not start on http://127.0.0.1:11434."
        )


def call_ollama(payload, model=DEFAULT_OLLAMA_MODEL, url=DEFAULT_OLLAMA_URL, timeout=DEFAULT_OLLAMA_TIMEOUT):
    ensure_ollama_running()

    body = {
        "model": model,
        "prompt": build_prompt(payload),
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
        },
    }

    req = request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama error {exc.code}: {error_body}") from exc
    except error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama at http://127.0.0.1:11434 after trying to start it automatically. "
            f"Install/start Ollama and pull a model like '{model}', then try again."
        ) from exc

    if "response" not in response_json:
        raise ValueError(f"Unexpected Ollama response: {response_json}")

    recommendation = json.loads(response_json["response"])
    validate_recommendation(recommendation)
    return recommendation, response_json


def recommend_for_row(row, model=DEFAULT_OLLAMA_MODEL, timeout=DEFAULT_OLLAMA_TIMEOUT):
    sample = {
        "statistics": row["stats_parsed"],
        "KPIs": row["kpis_parsed"],
        "labels": row["labels_parsed"],
        "anomalies": row["anomalies_parsed"],
        "sampling_rate": row.get("sampling_rate", 0.0),
    }
    prediction = inference.predict(sample)
    payload = build_context_payload(row, prediction)
    recommendation, raw_response = call_ollama(payload, model=model, timeout=timeout)
    return {
        "row_index": int(row.name),
        "payload": payload,
        "recommendation": recommendation,
        "prediction": prediction,
        "model": model,
        "mode": "local_ollama_llm",
        "raw_done": raw_response.get("done"),
    }


def print_recommendation(result):
    payload = result["payload"]
    recommendation = result["recommendation"]
    prediction = result["prediction"]

    print("=" * 80)
    print(f"LOCAL LLM RECOMMENDATION FOR ROW {result['row_index']}")
    print("=" * 80)
    print(f"LLM model           : {result['model']}")
    print(f"Application         : {payload['context']['application']}")
    print(f"Zone                : {payload['context']['zone']}")
    print(f"Mobility            : {payload['context']['mobility']}")
    print(f"Congestion          : {payload['context']['congestion']}")
    print(f"Predicted anomaly   : {prediction['anomaly']}")
    print(f"Predicted type      : {prediction['type']}")
    print(f"Type confidence     : {prediction['type_confidence']}")
    print(f"Predicted duration  : {prediction['duration_steps']} ({prediction['duration_regime']})")
    print(f"Severity            : {recommendation['severity']}")
    print(f"Urgency             : {recommendation['urgency']}")
    print(f"Confidence posture  : {recommendation['confidence_posture']}")
    print(f"Summary             : {recommendation['summary']}")
    print(f"Likely root cause   : {recommendation['likely_root_cause']}")
    print(f"Operational impact  : {recommendation['operational_impact']}")
    print(f"Escalation needed   : {recommendation['escalation_needed']}")
    print("\nRecommended actions:")
    for idx, action in enumerate(recommendation["recommended_actions"], start=1):
        print(f"  {idx}. {action}")
    print("\nVerification steps:")
    for idx, step in enumerate(recommendation["verification_steps"], start=1):
        print(f"  {idx}. {step}")
    print(f"\nOperator note       : {recommendation['operator_note']}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 3-stage anomaly pipeline and generate a recommendation with a local LLM via Ollama."
    )
    parser.add_argument("--row-index", type=int, default=0, help="Dataset row to analyze.")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Local Ollama model name.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_OLLAMA_TIMEOUT,
        help="Ollama request timeout in seconds.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save the full payload + recommendation as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_dataset_with_parsed_columns()

    if args.row_index < 0 or args.row_index >= len(df):
        raise IndexError(f"row-index must be between 0 and {len(df) - 1}.")

    row = df.iloc[args.row_index]
    result = recommend_for_row(row, model=args.model, timeout=args.timeout)
    print_recommendation(result)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"Saved JSON result to {args.save_json}")


if __name__ == "__main__":
    main()
