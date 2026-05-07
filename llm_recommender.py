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
REMEDY_CATALOGUE_PATH = "remedy_catalogue.csv"
OLLAMA_HEALTHCHECK_URL = "http://127.0.0.1:11434/api/tags"
OLLAMA_EXECUTABLE_CANDIDATES = [
    shutil.which("ollama"),
    r"C:\Users\MON PC\AppData\Local\Programs\Ollama\ollama.exe",
]


JSON_OUTPUT_INSTRUCTIONS = (
    "Return only valid JSON with these keys exactly: "
    "severity, urgency, confidence_posture, escalation_needed, selected_remedy_ids, recommended_actions, "
    "selected_prevention_ids, prevention_measures, selected_verification_ids, verification_steps, summary, "
    "likely_root_cause, operational_impact, operator_note. "
    "severity must be one of info, low, medium, high, critical. "
    "urgency must be one of monitor, planned, soon, urgent, immediate. "
    "confidence_posture must be one of low, medium, high. "
    "selected_remedy_ids must be an array of 1 to 4 ids chosen only from the provided catalogue options. "
    "recommended_actions must be an array of 3 to 6 strings. "
    "selected_prevention_ids must be an array of 1 to 3 ids chosen only from the provided catalogue options. "
    "prevention_measures must be an array of 1 to 3 strings. "
    "selected_verification_ids must be an array of 1 to 3 ids chosen only from the provided catalogue options. "
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


def load_remedy_catalogue():
    catalogue_df = pd.read_csv(REMEDY_CATALOGUE_PATH)
    return catalogue_df.set_index("anomaly_type")


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


def build_catalogue_payload(catalogue_row):
    def collect(prefix, max_items):
        items = []
        for idx in range(1, max_items + 1):
            value = catalogue_row.get(f"{prefix}_{idx}")
            if isinstance(value, str) and value.strip():
                item_id = f"{prefix[0].upper()}{idx}"
                items.append({"id": item_id, "text": value})
        return items

    return {
        "anomaly_type": catalogue_row.name,
        "affected_kpis": catalogue_row.get("affected_kpis"),
        "signal_signature": catalogue_row.get("signal_signature"),
        "operator_goal": catalogue_row.get("operator_goal"),
        "remedies": collect("remedy", 4),
        "preventions": collect("prevention", 3),
        "verifications": collect("verification", 3),
        "escalation_hint": catalogue_row.get("escalation_hint"),
    }


def choose_catalogue_anomaly_type(prediction):
    if not prediction.get("anomaly"):
        return "normal"
    return prediction.get("type") or "normal"


def build_messages(payload, catalogue_payload):
    instructions = (
        "You are a telecom operations assistant. "
        "Use the structured model outputs, context, and the provided remedy catalogue entry to recommend practical operator actions. "
        "Be careful with uncertainty: when confidence is not high, suggest verification before disruptive action. "
        "If anomaly is false, avoid inventing an incident and recommend monitoring only from the provided normal catalogue. "
        "Long predicted duration or severe anomaly types should increase urgency. "
        "Choose remedies, preventions, and verifications only from the provided catalogue entry. "
        "The selected ids must match the chosen catalogue text exactly. "
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
            "Do not invent remedies outside the provided catalogue entry.",
        ],
        "catalogue_entry": catalogue_payload,
        "event": payload,
    }

    return instructions, user_prompt


def build_prompt(payload, catalogue_payload):
    instructions, user_prompt = build_messages(payload, catalogue_payload)
    return f"{instructions}\n\nEVENT:\n{json.dumps(user_prompt, ensure_ascii=True, indent=2)}"


def validate_recommendation(recommendation, catalogue_payload):
    required_keys = {
        "severity",
        "urgency",
        "confidence_posture",
        "selected_remedy_ids",
        "selected_prevention_ids",
        "selected_verification_ids",
        "prevention_measures",
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
    if not isinstance(recommendation["prevention_measures"], list):
        raise ValueError("prevention_measures must be a list.")
    if not isinstance(recommendation["verification_steps"], list):
        raise ValueError("verification_steps must be a list.")
    if not isinstance(recommendation["escalation_needed"], bool):
        raise ValueError("escalation_needed must be a boolean.")
    if not isinstance(recommendation["selected_remedy_ids"], list):
        raise ValueError("selected_remedy_ids must be a list.")
    if not isinstance(recommendation["selected_prevention_ids"], list):
        raise ValueError("selected_prevention_ids must be a list.")
    if not isinstance(recommendation["selected_verification_ids"], list):
        raise ValueError("selected_verification_ids must be a list.")

    valid_remedy_ids = {item["id"]: item["text"] for item in catalogue_payload["remedies"]}
    valid_prevention_ids = {item["id"]: item["text"] for item in catalogue_payload["preventions"]}
    valid_verification_ids = {item["id"]: item["text"] for item in catalogue_payload["verifications"]}

    for action_id in recommendation["selected_remedy_ids"]:
        if action_id not in valid_remedy_ids:
            raise ValueError(f"Unknown selected remedy id: {action_id}")
    for action_id in recommendation["selected_prevention_ids"]:
        if action_id not in valid_prevention_ids:
            raise ValueError(f"Unknown selected prevention id: {action_id}")
    for action_id in recommendation["selected_verification_ids"]:
        if action_id not in valid_verification_ids:
            raise ValueError(f"Unknown selected verification id: {action_id}")


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


def call_ollama(payload, catalogue_payload, model=DEFAULT_OLLAMA_MODEL, url=DEFAULT_OLLAMA_URL, timeout=DEFAULT_OLLAMA_TIMEOUT):
    ensure_ollama_running()

    body = {
        "model": model,
        "prompt": build_prompt(payload, catalogue_payload),
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
    validate_recommendation(recommendation, catalogue_payload)
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
    catalogue_df = load_remedy_catalogue()
    catalogue_anomaly_type = choose_catalogue_anomaly_type(prediction)
    if catalogue_anomaly_type not in catalogue_df.index:
        catalogue_anomaly_type = "normal"
    catalogue_payload = build_catalogue_payload(catalogue_df.loc[catalogue_anomaly_type])

    if not prediction["anomaly"]:
        recommendation = {
            "severity": "info",
            "urgency": "monitor",
            "confidence_posture": "medium",
            "escalation_needed": False,
            "selected_remedy_ids": ["R1"],
            "recommended_actions": [catalogue_payload["remedies"][0]["text"]],
            "selected_prevention_ids": ["P1"],
            "prevention_measures": [catalogue_payload["preventions"][0]["text"]],
            "selected_verification_ids": ["V1"],
            "verification_steps": [catalogue_payload["verifications"][0]["text"]],
            "summary": "No anomaly was detected for this row.",
            "likely_root_cause": "No anomaly was detected in the dataset label for this window.",
            "operational_impact": "No anomaly was detected, so disruptive remediation is not required.",
            "operator_note": "No anomaly was detected. Continue monitoring and keep the healthy baseline.",
        }
        raw_response = {"done": True, "bypassed_llm": True}
    else:
        recommendation, raw_response = call_ollama(payload, catalogue_payload, model=model, timeout=timeout)
    return {
        "row_index": int(row.name),
        "payload": payload,
        "catalogue_anomaly_type": catalogue_anomaly_type,
        "catalogue_entry": catalogue_payload,
        "recommendation": recommendation,
        "prediction": prediction,
        "model": model,
        "mode": "local_rule_normal" if not prediction["anomaly"] else "local_ollama_llm",
        "raw_done": raw_response.get("done"),
    }


def serialize_json_list(value):
    return json.dumps(value, ensure_ascii=False)


def write_result_to_csv(csv_path, result):
    df = pd.read_csv(csv_path, low_memory=False)
    update_dataframe_with_result(df, result)
    df.to_csv(csv_path, index=False)


def update_dataframe_with_result(df, result):
    row_index = result["row_index"]
    mask = df["row_index"] == row_index
    if not mask.any():
        raise ValueError(f"row_index {row_index} was not found in target dataframe")

    prediction = result["prediction"]
    recommendation = result["recommendation"]
    payload = result["payload"]

    updates = {
        "predicted_anomaly": bool(prediction["anomaly"]),
        "predicted_confidence": prediction["confidence"],
        "predicted_type": prediction["type"],
        "predicted_type_confidence": prediction["type_confidence"],
        "predicted_duration_steps": prediction["duration_steps"],
        "predicted_duration_regime": prediction["duration_regime"],
        "duration_ready": prediction["duration_ready"],
        "recommender_model": result["model"],
        "recommender_mode": result["mode"],
        "recommender_catalogue_anomaly_type": result["catalogue_anomaly_type"],
        "recommender_severity": recommendation["severity"],
        "recommender_urgency": recommendation["urgency"],
        "recommender_confidence_posture": recommendation["confidence_posture"],
        "recommender_escalation_needed": recommendation["escalation_needed"],
        "recommender_selected_remedy_ids": serialize_json_list(recommendation["selected_remedy_ids"]),
        "recommender_recommended_actions": serialize_json_list(recommendation["recommended_actions"]),
        "recommender_selected_prevention_ids": serialize_json_list(recommendation["selected_prevention_ids"]),
        "recommender_prevention_measures": serialize_json_list(recommendation["prevention_measures"]),
        "recommender_selected_verification_ids": serialize_json_list(recommendation["selected_verification_ids"]),
        "recommender_verification_steps": serialize_json_list(recommendation["verification_steps"]),
        "recommender_summary": recommendation["summary"],
        "recommender_likely_root_cause": recommendation["likely_root_cause"],
        "recommender_operational_impact": recommendation["operational_impact"],
        "recommender_operator_note": recommendation["operator_note"],
        "recommender_affected_kpis": serialize_json_list(payload["anomaly_details"].get("affected_kpis") or []),
        "recommender_top_kpis": json.dumps(payload["signal_summary"].get("top_kpis") or {}, ensure_ascii=False),
    }

    for column, value in updates.items():
        if column not in df.columns:
            df[column] = pd.NA
        if column in df.columns and df[column].dtype != "object":
            df[column] = df[column].astype("object")
        df.loc[mask, column] = value


def run_all_rows(
    dataset_df,
    csv_path,
    model=DEFAULT_OLLAMA_MODEL,
    timeout=DEFAULT_OLLAMA_TIMEOUT,
    start_row=0,
    end_row=None,
    resume=True,
    flush_every=1,
):
    target_df = pd.read_csv(csv_path, low_memory=False)
    if "row_index" not in target_df.columns:
        raise ValueError(f"{csv_path} must contain a row_index column")

    if end_row is None:
        end_row = len(dataset_df) - 1

    if start_row < 0 or end_row >= len(dataset_df) or start_row > end_row:
        raise ValueError(f"Invalid row range: start={start_row}, end={end_row}")

    if "recommender_model" not in target_df.columns:
        target_df["recommender_model"] = pd.NA

    processed = 0
    skipped = 0
    failed = 0

    for dataset_index in range(start_row, end_row + 1):
        row_mask = target_df["row_index"] == dataset_index
        if not row_mask.any():
            continue

        existing_value = target_df.loc[row_mask, "recommender_model"].iloc[0]
        if resume and pd.notna(existing_value) and str(existing_value).strip():
            skipped += 1
            continue

        row = dataset_df.iloc[dataset_index]
        try:
            result = recommend_for_row(row, model=model, timeout=timeout)
            update_dataframe_with_result(target_df, result)
            processed += 1
            print(
                f"[OK] row_index={dataset_index} "
                f"type={result['prediction']['type']} "
                f"duration={result['prediction']['duration_steps']}"
            )
        except Exception as exc:
            failed += 1
            if "recommender_error" not in target_df.columns:
                target_df["recommender_error"] = pd.NA
            if target_df["recommender_error"].dtype != "object":
                target_df["recommender_error"] = target_df["recommender_error"].astype("object")
            target_df.loc[row_mask, "recommender_error"] = str(exc)
            print(f"[FAIL] row_index={dataset_index} error={exc}")

        if processed % flush_every == 0 or failed > 0:
            target_df.to_csv(csv_path, index=False)

    target_df.to_csv(csv_path, index=False)
    print(
        f"Finished rows {start_row}..{end_row} | "
        f"processed={processed} skipped={skipped} failed={failed}"
    )


def print_recommendation(result):
    payload = result["payload"]
    catalogue_entry = result["catalogue_entry"]
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
    print(f"Catalogue anomaly   : {result['catalogue_anomaly_type']}")
    print(f"Severity            : {recommendation['severity']}")
    print(f"Urgency             : {recommendation['urgency']}")
    print(f"Confidence posture  : {recommendation['confidence_posture']}")
    print(f"Selected remedies   : {', '.join(recommendation['selected_remedy_ids'])}")
    print(f"Selected prevention : {', '.join(recommendation['selected_prevention_ids'])}")
    print(f"Selected verify     : {', '.join(recommendation['selected_verification_ids'])}")
    print(f"Summary             : {recommendation['summary']}")
    print(f"Likely root cause   : {recommendation['likely_root_cause']}")
    print(f"Operational impact  : {recommendation['operational_impact']}")
    print(f"Escalation needed   : {recommendation['escalation_needed']}")
    print("\nRecommended actions:")
    for idx, action in enumerate(recommendation["recommended_actions"], start=1):
        print(f"  {idx}. {action}")
    print("\nPrevention measures:")
    for idx, step in enumerate(recommendation["prevention_measures"], start=1):
        print(f"  {idx}. {step}")
    print("\nVerification steps:")
    for idx, step in enumerate(recommendation["verification_steps"], start=1):
        print(f"  {idx}. {step}")
    print(f"\nCatalogue operator goal : {catalogue_entry['operator_goal']}")
    print(f"Catalogue escalation    : {catalogue_entry['escalation_hint']}")
    print(f"\nOperator note       : {recommendation['operator_note']}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 3-stage anomaly pipeline and generate a recommendation with a local LLM via Ollama."
    )
    parser.add_argument("--row-index", type=int, default=0, help="Dataset row to analyze.")
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Run the recommender across all rows (or a selected row range) and write results into a CSV.",
    )
    parser.add_argument("--start-row", type=int, default=0, help="Starting row_index for batch mode.")
    parser.add_argument("--end-row", type=int, default=None, help="Ending row_index for batch mode.")
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
    parser.add_argument(
        "--write-csv",
        default=None,
        help="Optional CSV path to update by row_index with prediction fields and recommender comparison columns.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="In batch mode, do not skip rows that already have recommender output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_dataset_with_parsed_columns()

    if args.all_rows:
        if not args.write_csv:
            raise ValueError("--all-rows requires --write-csv")
        run_all_rows(
            dataset_df=df,
            csv_path=args.write_csv,
            model=args.model,
            timeout=args.timeout,
            start_row=args.start_row,
            end_row=args.end_row,
            resume=not args.no_resume,
        )
        return

    if args.row_index < 0 or args.row_index >= len(df):
        raise IndexError(f"row-index must be between 0 and {len(df) - 1}.")

    row = df.iloc[args.row_index]
    result = recommend_for_row(row, model=args.model, timeout=args.timeout)
    print_recommendation(result)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"Saved JSON result to {args.save_json}")

    if args.write_csv:
        write_result_to_csv(args.write_csv, result)
        print(f"Updated CSV row {result['row_index']} in {args.write_csv}")


if __name__ == "__main__":
    main()
