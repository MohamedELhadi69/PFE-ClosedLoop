import csv
import json
import os
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib import error, request

LOCAL_PYTHON_PACKAGES = Path(__file__).with_name(".python_packages")
if LOCAL_PYTHON_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PYTHON_PACKAGES))

import pandas as pd

from data_utils import load_or_fetch_dataframe, parse_col


DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_OLLAMA_TIMEOUT = 600
OLLAMA_HEALTHCHECK_URL = "http://127.0.0.1:11434/api/tags"
REMEDY_CATALOGUE_PATH = Path(__file__).with_name("remedy_catalogue_dataset.csv")

RESPONSE_JSON_INSTRUCTIONS = (
    "Return only valid JSON with these keys exactly: "
    "severity, urgency, confidence_posture, summary, likely_root_cause, "
    "operational_impact, selected_remedy_ids, selected_prevention_ids, "
    "selected_verification_ids, escalation_needed, operator_note. "
    "severity must be one of info, low, medium, high, critical. "
    "urgency must be one of monitor, planned, soon, urgent, immediate. "
    "confidence_posture must be one of low, medium, high. "
    "selected_remedy_ids must be an array of 1 to 6 catalogue remedy IDs. "
    "selected_prevention_ids must be an array of 1 to 3 catalogue prevention IDs. "
    "selected_verification_ids must be an array of 1 to 3 catalogue verification IDs. "
    "escalation_needed must be a boolean. "
    "Do not invent new remedy text, prevention text, or verification text; choose only IDs that exist in the supplied catalogue entry."
)

REQUIRED_RESPONSE_KEYS = {
    "severity",
    "urgency",
    "confidence_posture",
    "summary",
    "likely_root_cause",
    "operational_impact",
    "selected_remedy_ids",
    "selected_prevention_ids",
    "selected_verification_ids",
    "escalation_needed",
    "operator_note",
}
VALID_SEVERITIES = {"info", "low", "medium", "high", "critical"}
VALID_URGENCIES = {"monitor", "planned", "soon", "urgent", "immediate"}
VALID_CONFIDENCE_POSTURES = {"low", "medium", "high"}

CSV_FIELDNAMES = [
    "row_index",
    "mode",
    "status",
    "error_message",
    "model",
    "catalogue_anomaly_type",
    "ground_truth_anomaly_type",
    "predicted_anomaly",
    "predicted_confidence",
    "predicted_type",
    "predicted_type_confidence",
    "predicted_duration_steps",
    "predicted_duration_regime",
    "duration_ready",
    "application",
    "zone",
    "mobility",
    "congestion",
    "start_time",
    "end_time",
    "sampling_rate",
    "severity",
    "urgency",
    "confidence_posture",
    "escalation_needed",
    "action_to_take",
    "selected_remedy_ids",
    "catalogue_remedy_ids",
    "catalogue_remedy_options",
    "recommended_actions",
    "selected_prevention_ids",
    "prevention_measures",
    "selected_verification_ids",
    "verification_steps",
    "summary",
    "likely_root_cause",
    "operational_impact",
    "operator_note",
    "affected_kpis",
    "focus_kpis",
    "top_kpis",
    "description",
    "qna_network",
    "qna_timeseries",
]

def ollama_executable_candidates():
    local_app_data = os.environ.get("LOCALAPPDATA")
    user_profile = os.environ.get("USERPROFILE")
    candidates = [
        os.environ.get("OLLAMA_PATH"),
        shutil.which("ollama"),
    ]

    if local_app_data:
        candidates.append(str(Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"))
    if user_profile:
        candidates.append(str(Path(user_profile) / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"))

    candidates.extend(
        [
            r"C:\Program Files\Ollama\ollama.exe",
            r"C:\Program Files (x86)\Ollama\ollama.exe",
            r"C:\Users\MON PC\AppData\Local\Programs\Ollama\ollama.exe",
        ]
    )
    return [candidate for candidate in candidates if candidate]


def load_dataset_with_parsed_columns():
    df = load_or_fetch_dataframe()
    df["stats_parsed"] = df["statistics"].apply(parse_col)
    df["anomalies_parsed"] = df["anomalies"].apply(parse_col)
    df["labels_parsed"] = df["labels"].apply(parse_col)
    df["kpis_parsed"] = df["KPIs"].apply(parse_col)
    df["qna_parsed"] = df["QnA"].apply(parse_col) if "QnA" in df.columns else None
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


def _collect_catalogue_items(row, prefix, id_prefix):
    items = []
    index = 1
    while f"{prefix}_{index}" in row:
        text = (row.get(f"{prefix}_{index}") or "").strip()
        if text:
            items.append({"id": f"{id_prefix}{index}", "text": text})
        index += 1
    return items


@lru_cache(maxsize=1)
def load_remedy_catalogue():
    catalogue = {}
    with REMEDY_CATALOGUE_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            anomaly_type = (row.get("anomaly_type") or "").strip()
            if not anomaly_type:
                continue

            catalogue[anomaly_type] = {
                "anomaly_type": anomaly_type,
                "affected_kpis": [
                    value.strip()
                    for value in (row.get("affected_kpis") or "").split("|")
                    if value.strip()
                ],
                "signal_signature": (row.get("signal_signature") or "").strip(),
                "operator_goal": (row.get("operator_goal") or "").strip(),
                "remedies": _collect_catalogue_items(row, "remedy", "R"),
                "prevention_measures": _collect_catalogue_items(row, "prevention", "P"),
                "verification_steps": _collect_catalogue_items(row, "verification", "V"),
                "escalation_hint": (row.get("escalation_hint") or "").strip(),
            }

    if "normal" not in catalogue:
        raise ValueError("The remedy catalogue must contain a 'normal' entry.")
    return catalogue


def get_catalogue_entry(anomaly_type):
    key = anomaly_type if anomaly_type in load_remedy_catalogue() else "normal"
    source = load_remedy_catalogue()[key]
    return {
        "anomaly_type": source["anomaly_type"],
        "affected_kpis": list(source["affected_kpis"]),
        "signal_signature": source["signal_signature"],
        "operator_goal": source["operator_goal"],
        "remedies": [dict(item) for item in source["remedies"]],
        "prevention_measures": [dict(item) for item in source["prevention_measures"]],
        "verification_steps": [dict(item) for item in source["verification_steps"]],
        "escalation_hint": source["escalation_hint"],
    }


def catalogue_prompt_view(entry):
    return {
        "anomaly_type": entry["anomaly_type"],
        "affected_kpis": entry["affected_kpis"],
        "signal_signature": entry["signal_signature"],
        "operator_goal": entry["operator_goal"],
        "escalation_hint": entry["escalation_hint"],
        "remedies": entry["remedies"],
        "prevention_measures": entry["prevention_measures"],
        "verification_steps": entry["verification_steps"],
    }


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


def find_ollama_executable():
    for candidate in ollama_executable_candidates():
        try:
            if candidate and Path(candidate).exists():
                return candidate
        except OSError:
            continue
    return None


def get_ollama_tags():
    with request.urlopen(OLLAMA_HEALTHCHECK_URL, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"Ollama health check returned HTTP {response.status}.")
        return json.loads(response.read().decode("utf-8"))


def validate_ollama_setup(model=DEFAULT_OLLAMA_MODEL, require_model=True):
    if not wait_for_ollama_server(timeout_seconds=2):
        executable = find_ollama_executable()
        if not executable:
            raise RuntimeError(
                "Ollama is not installed or not discoverable on this PC. "
                "Install Ollama, or set the OLLAMA_PATH environment variable to ollama.exe."
            )

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.Popen(
                [executable, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Ollama executable was expected at '{executable}' but could not be launched."
            ) from exc

        if not wait_for_ollama_server(timeout_seconds=30):
            raise RuntimeError(
                "Ollama executable was found, but the local server did not start on "
                "http://127.0.0.1:11434."
            )

    tags_response = get_ollama_tags()
    available_models = [
        item.get("name")
        for item in tags_response.get("models", [])
        if isinstance(item, dict) and item.get("name")
    ]
    if require_model and model not in available_models:
        preview = ", ".join(available_models[:10]) if available_models else "none"
        raise RuntimeError(
            f"Ollama is running but model '{model}' is not installed. "
            f"Available models: {preview}. Pull it first with: ollama pull {model}"
        )
    return available_models


def ensure_ollama_running():
    validate_ollama_setup(model=DEFAULT_OLLAMA_MODEL, require_model=False)


def call_ollama(prompt, model=DEFAULT_OLLAMA_MODEL, url=DEFAULT_OLLAMA_URL, timeout=DEFAULT_OLLAMA_TIMEOUT):
    validate_ollama_setup(model=model, require_model=True)

    body = {
        "model": model,
        "prompt": prompt,
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
    return recommendation, response_json


def _validate_id_selection(values, valid_ids, minimum, maximum, field_name):
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list.")
    if not (minimum <= len(values) <= maximum):
        raise ValueError(f"{field_name} must contain between {minimum} and {maximum} IDs.")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} contains duplicate IDs.")
    unknown_ids = [value for value in values if value not in valid_ids]
    if unknown_ids:
        raise ValueError(f"{field_name} contains unknown catalogue IDs: {unknown_ids}")


def _default_severity_for_catalogue(anomaly_type):
    return "info" if anomaly_type == "normal" else "high"


def _default_urgency_for_catalogue(anomaly_type):
    return "monitor" if anomaly_type == "normal" else "urgent"


def _default_confidence_posture_for_catalogue(anomaly_type):
    return "medium" if anomaly_type == "normal" else "high"


def _coerce_catalogue_echo(recommendation, catalogue_entry):
    remedy_ids = [item.get("id") for item in recommendation.get("remedies", []) if item.get("id")]
    prevention_ids = [
        item.get("id") for item in recommendation.get("prevention_measures", []) if item.get("id")
    ]
    verification_ids = [
        item.get("id") for item in recommendation.get("verification_steps", []) if item.get("id")
    ]

    if not (remedy_ids or prevention_ids or verification_ids):
        return recommendation

    anomaly_type = catalogue_entry["anomaly_type"]
    is_normal = anomaly_type == "normal"
    signal_signature = recommendation.get("signal_signature") or catalogue_entry["signal_signature"]
    operator_goal = recommendation.get("operator_goal") or catalogue_entry["operator_goal"]
    escalation_hint = recommendation.get("escalation_hint") or catalogue_entry["escalation_hint"]

    return {
        "severity": _default_severity_for_catalogue(anomaly_type),
        "urgency": _default_urgency_for_catalogue(anomaly_type),
        "confidence_posture": _default_confidence_posture_for_catalogue(anomaly_type),
        "summary": (
            "Reference normal-operation answer selected from the reviewed catalogue."
            if is_normal
            else f"Reference remediation answer selected from the reviewed catalogue for {anomaly_type}."
        ),
        "likely_root_cause": (
            "No confirmed anomaly; continue monitoring and preserve the healthy baseline."
            if is_normal
            else signal_signature
        ),
        "operational_impact": (
            "No immediate corrective action is required unless the next windows degrade."
            if is_normal
            else operator_goal
        ),
        "selected_remedy_ids": remedy_ids[:4] or [item["id"] for item in catalogue_entry["remedies"][:1]],
        "selected_prevention_ids": prevention_ids[:3]
        or [item["id"] for item in catalogue_entry["prevention_measures"][:1]],
        "selected_verification_ids": verification_ids[:3]
        or [item["id"] for item in catalogue_entry["verification_steps"][:1]],
        "escalation_needed": not is_normal,
        "operator_note": escalation_hint or operator_goal or signal_signature,
    }


def _build_catalogue_fallback_recommendation(catalogue_entry, partial=None):
    partial = partial or {}
    anomaly_type = catalogue_entry["anomaly_type"]
    is_normal = anomaly_type == "normal"
    signal_signature = partial.get("signal_signature") or catalogue_entry["signal_signature"]
    operator_goal = partial.get("operator_goal") or catalogue_entry["operator_goal"]
    escalation_hint = partial.get("escalation_hint") or catalogue_entry["escalation_hint"]

    return {
        "severity": partial.get("severity") or _default_severity_for_catalogue(anomaly_type),
        "urgency": partial.get("urgency") or _default_urgency_for_catalogue(anomaly_type),
        "confidence_posture": partial.get("confidence_posture")
        or _default_confidence_posture_for_catalogue(anomaly_type),
        "summary": partial.get("summary")
        or (
            "Reference normal-operation answer selected from the reviewed catalogue."
            if is_normal
            else f"Reference remediation answer selected from the reviewed catalogue for {anomaly_type}."
        ),
        "likely_root_cause": partial.get("likely_root_cause")
        or (
            "No confirmed anomaly; continue monitoring and preserve the healthy baseline."
            if is_normal
            else signal_signature
        ),
        "operational_impact": partial.get("operational_impact")
        or (
            "No immediate corrective action is required unless the next windows degrade."
            if is_normal
            else operator_goal
        ),
        "selected_remedy_ids": partial.get("selected_remedy_ids")
        or [item["id"] for item in catalogue_entry["remedies"][: min(3, len(catalogue_entry["remedies"]))]],
        "selected_prevention_ids": partial.get("selected_prevention_ids")
        or [
            item["id"]
            for item in catalogue_entry["prevention_measures"][
                : min(2, len(catalogue_entry["prevention_measures"]))
            ]
        ],
        "selected_verification_ids": partial.get("selected_verification_ids")
        or [
            item["id"]
            for item in catalogue_entry["verification_steps"][
                : min(2, len(catalogue_entry["verification_steps"]))
            ]
        ],
        "escalation_needed": partial.get("escalation_needed")
        if isinstance(partial.get("escalation_needed"), bool)
        else (not is_normal),
        "operator_note": partial.get("operator_note")
        or escalation_hint
        or operator_goal
        or signal_signature,
    }


def build_default_catalogue_recommendation(
    anomaly_type,
    *,
    summary=None,
    likely_root_cause=None,
    operational_impact=None,
    operator_note=None,
    remedy_limit=3,
    prevention_limit=2,
    verification_limit=2,
):
    entry = get_catalogue_entry(anomaly_type)
    partial = {
        "summary": summary,
        "likely_root_cause": likely_root_cause,
        "operational_impact": operational_impact,
        "operator_note": operator_note,
        "selected_remedy_ids": [item["id"] for item in entry["remedies"][:remedy_limit]],
        "selected_prevention_ids": [
            item["id"] for item in entry["prevention_measures"][:prevention_limit]
        ],
        "selected_verification_ids": [
            item["id"] for item in entry["verification_steps"][:verification_limit]
        ],
    }
    return finalize_catalogue_recommendation(
        _build_catalogue_fallback_recommendation(entry, partial=partial),
        entry,
    )


def finalize_catalogue_recommendation(recommendation, catalogue_entry):
    recommendation = _coerce_catalogue_echo(recommendation, catalogue_entry)

    missing = REQUIRED_RESPONSE_KEYS - set(recommendation.keys())
    if missing:
        recommendation = _build_catalogue_fallback_recommendation(catalogue_entry, partial=recommendation)

    severity = recommendation["severity"]
    urgency = recommendation["urgency"]
    confidence_posture = recommendation["confidence_posture"]
    if severity not in VALID_SEVERITIES:
        raise ValueError(f"Invalid severity: {severity}")
    if urgency not in VALID_URGENCIES:
        raise ValueError(f"Invalid urgency: {urgency}")
    if confidence_posture not in VALID_CONFIDENCE_POSTURES:
        raise ValueError(f"Invalid confidence_posture: {confidence_posture}")
    if not isinstance(recommendation["escalation_needed"], bool):
        raise ValueError("escalation_needed must be a boolean.")

    remedy_lookup = {item["id"]: item["text"] for item in catalogue_entry["remedies"]}
    prevention_lookup = {item["id"]: item["text"] for item in catalogue_entry["prevention_measures"]}
    verification_lookup = {item["id"]: item["text"] for item in catalogue_entry["verification_steps"]}

    _validate_id_selection(
        recommendation["selected_remedy_ids"],
        set(remedy_lookup),
        minimum=1,
        maximum=min(6, len(remedy_lookup)),
        field_name="selected_remedy_ids",
    )
    _validate_id_selection(
        recommendation["selected_prevention_ids"],
        set(prevention_lookup),
        minimum=1,
        maximum=min(3, len(prevention_lookup)),
        field_name="selected_prevention_ids",
    )
    _validate_id_selection(
        recommendation["selected_verification_ids"],
        set(verification_lookup),
        minimum=1,
        maximum=min(3, len(verification_lookup)),
        field_name="selected_verification_ids",
    )

    normalized = {key: recommendation[key] for key in REQUIRED_RESPONSE_KEYS}
    normalized["catalogue_anomaly_type"] = catalogue_entry["anomaly_type"]
    normalized["recommended_actions"] = [
        remedy_lookup[item_id] for item_id in normalized["selected_remedy_ids"]
    ]
    normalized["prevention_measures"] = [
        prevention_lookup[item_id] for item_id in normalized["selected_prevention_ids"]
    ]
    normalized["verification_steps"] = [
        verification_lookup[item_id] for item_id in normalized["selected_verification_ids"]
    ]
    return normalized


def derive_action_to_take(recommendation, prediction=None):
    prediction = prediction or {}
    severity = str(recommendation.get("severity") or "").strip().lower()
    urgency = str(recommendation.get("urgency") or "").strip().lower()
    confidence = str(recommendation.get("confidence_posture") or "").strip().lower()
    anomaly_detected = prediction.get("anomaly")

    if anomaly_detected is False or severity in {"", "info"} or urgency == "monitor":
        return "Monitor only and preserve the current baseline."

    if confidence == "low":
        return "Notify the operations team for manual review before applying remediation."

    if severity == "critical" or urgency == "immediate":
        return "Apply the selected remediation urgently and notify the operations team."

    if severity in {"high", "medium"} or urgency in {"urgent", "soon", "planned"}:
        return "Verify the anomaly, then apply the selected remediation under supervision."

    return "Continue monitoring and prepare remediation if the anomaly persists."


def serialize_for_csv(value):
    value = to_builtin(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def append_csv_row(csv_path, row, fieldnames=None):
    target = Path(csv_path)
    fieldnames = fieldnames or CSV_FIELDNAMES
    target.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not target.exists() or target.stat().st_size == 0

    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow({name: serialize_for_csv(row.get(name)) for name in fieldnames})


def load_processed_row_indices(csv_path):
    target = Path(csv_path)
    if not target.exists():
        return set()

    processed = set()
    with target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("status") or "").strip().lower() != "ok":
                continue
            value = (row.get("row_index") or "").strip()
            if value:
                processed.add(int(value))
    return processed


def build_csv_row(
    row,
    result=None,
    *,
    mode,
    status="ok",
    error_message="",
    include_source_text=False,
):
    labels = row.get("labels_parsed") or {}
    anomalies = row.get("anomalies_parsed") or {}
    qna = row.get("qna_parsed") or {}
    prediction = (result or {}).get("prediction") or {}
    payload = (result or {}).get("payload") or {}
    recommendation = (result or {}).get("recommendation") or {}
    catalogue_entry = (result or {}).get("catalogue_entry") or {}

    return {
        "row_index": int(row.name),
        "mode": mode,
        "status": status,
        "error_message": error_message,
        "model": (result or {}).get("model"),
        "catalogue_anomaly_type": recommendation.get("catalogue_anomaly_type"),
        "ground_truth_anomaly_type": anomalies.get("type") if anomalies.get("exists") else "normal",
        "predicted_anomaly": prediction.get("anomaly"),
        "predicted_confidence": prediction.get("confidence"),
        "predicted_type": prediction.get("type"),
        "predicted_type_confidence": prediction.get("type_confidence"),
        "predicted_duration_steps": prediction.get("duration_steps"),
        "predicted_duration_regime": prediction.get("duration_regime"),
        "duration_ready": prediction.get("duration_ready"),
        "application": labels.get("application"),
        "zone": labels.get("zone"),
        "mobility": labels.get("mobility"),
        "congestion": labels.get("congestion"),
        "start_time": row.get("start_time"),
        "end_time": row.get("end_time"),
        "sampling_rate": row.get("sampling_rate"),
        "severity": recommendation.get("severity"),
        "urgency": recommendation.get("urgency"),
        "confidence_posture": recommendation.get("confidence_posture"),
        "escalation_needed": recommendation.get("escalation_needed"),
        "action_to_take": derive_action_to_take(recommendation, prediction),
        "selected_remedy_ids": recommendation.get("selected_remedy_ids"),
        "catalogue_remedy_ids": [item.get("id") for item in catalogue_entry.get("remedies", [])],
        "catalogue_remedy_options": [item.get("text") for item in catalogue_entry.get("remedies", [])],
        "recommended_actions": recommendation.get("recommended_actions"),
        "selected_prevention_ids": recommendation.get("selected_prevention_ids"),
        "prevention_measures": recommendation.get("prevention_measures"),
        "selected_verification_ids": recommendation.get("selected_verification_ids"),
        "verification_steps": recommendation.get("verification_steps"),
        "summary": recommendation.get("summary"),
        "likely_root_cause": recommendation.get("likely_root_cause"),
        "operational_impact": recommendation.get("operational_impact"),
        "operator_note": recommendation.get("operator_note"),
        "affected_kpis": anomalies.get("affected_kpis") if include_source_text else None,
        "focus_kpis": payload.get("signal_summary", {}).get("focus_kpis"),
        "top_kpis": payload.get("signal_summary", {}).get("top_kpis"),
        "description": row.get("description") if include_source_text else None,
        "qna_network": qna.get("network") if include_source_text else None,
        "qna_timeseries": qna.get("timeseries") if include_source_text else None,
    }


def print_structured_recommendation(result, title):
    payload = result["payload"]
    recommendation = result["recommendation"]
    prediction = result.get("prediction") or {}

    print("=" * 80)
    print(f"{title} FOR ROW {result['row_index']}")
    print("=" * 80)
    print(f"LLM model           : {result['model']}")
    print(f"Application         : {payload['context']['application']}")
    print(f"Zone                : {payload['context']['zone']}")
    print(f"Mobility            : {payload['context']['mobility']}")
    print(f"Congestion          : {payload['context']['congestion']}")
    if prediction:
        print(f"Predicted anomaly   : {prediction.get('anomaly')}")
        print(f"Predicted type      : {prediction.get('type')}")
        print(f"Type confidence     : {prediction.get('type_confidence')}")
        print(
            f"Predicted duration  : {prediction.get('duration_steps')} "
            f"({prediction.get('duration_regime')})"
        )
    print(f"Catalogue type      : {recommendation['catalogue_anomaly_type']}")
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
    print("\nPrevention measures:")
    for idx, action in enumerate(recommendation["prevention_measures"], start=1):
        print(f"  {idx}. {action}")
    print("\nVerification steps:")
    for idx, step in enumerate(recommendation["verification_steps"], start=1):
        print(f"  {idx}. {step}")
    print(f"\nOperator note       : {recommendation['operator_note']}")
    print("=" * 80)
