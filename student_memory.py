import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime_bootstrap import ensure_repo_python

ensure_repo_python()

import argparse
import csv
import json
import re
from functools import lru_cache

import inference
from llm_shared import (
    append_csv_row,
    build_csv_row,
    build_default_catalogue_recommendation,
    get_catalogue_entry,
    load_dataset_with_parsed_columns,
    load_processed_row_indices,
    to_builtin,
    top_kpi_summary,
)
from split_utils import MODEL_TRAIN_SPLIT_NAME, resolve_selected_row_indices


DEFAULT_MEMORY_CSV = Path("llm_student_memory.csv")


def build_recommender_payload(row, prediction):
    labels = row.get("labels_parsed") or {}
    stats = row.get("stats_parsed") or {}

    type_distribution = prediction.get("type_distribution") or {}
    top_type_distribution = dict(list(type_distribution.items())[:3])
    top_kpis = top_kpi_summary(stats)

    return {
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
            "duration_reason": prediction.get("duration_reason"),
        },
        "context": {
            "application": labels.get("application"),
            "zone": labels.get("zone"),
            "mobility": labels.get("mobility"),
            "congestion": labels.get("congestion"),
            "sampling_rate": to_builtin(row.get("sampling_rate")),
        },
        "signal_summary": {
            "top_kpis": top_kpis,
            "focus_kpis": list(top_kpis.keys()),
        },
    }


def choose_catalogue_anomaly_type(prediction):
    if not prediction.get("anomaly"):
        return "normal"
    return prediction.get("type") or "normal"


def _strip(value):
    return str(value).strip() if value is not None else ""


def _parse_bool(value, default=False):
    if isinstance(value, bool):
        return value
    text = _strip(value).lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False if text else default
    return default


def _parse_float(value, default=None):
    text = _strip(value)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _parse_int(value, default=None):
    number = _parse_float(value, default=None)
    if number is None:
        return default
    return int(round(number))


def _parse_json(value, default):
    if isinstance(value, (dict, list)):
        return value
    text = _strip(value)
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def _normalize_text(value, fallback="unknown"):
    text = _strip(value)
    return text if text else fallback


TICKET_SECTION_PATTERNS = {
    "issue": re.compile(r"\*\*Issue:\*\*\s*(.+?)(?=\n-\s*\*\*|\Z)", re.IGNORECASE | re.S),
    "symptoms": re.compile(r"\*\*Symptoms:\*\*\s*(.+?)(?=\n-\s*\*\*|\Z)", re.IGNORECASE | re.S),
    "root_cause": re.compile(r"\*\*Root Cause:\*\*\s*(.+?)(?=\n-\s*\*\*|\Z)", re.IGNORECASE | re.S),
    "resolution": re.compile(r"\*\*Resolution:\*\*\s*(.+?)(?=\n-\s*\*\*|\Z)", re.IGNORECASE | re.S),
}

SEVERITY_BY_ANOMALY = {
    "normal": "info",
    "Co-Channel Interference (Mild)": "medium",
    "High Network Congestion (Gradual Buildup)": "medium",
    "Buffer Overflow (Gradual Buildup)": "medium",
    "Faulty Handover Algorithm (Too Frequent)": "medium",
    "Faulty RF Filters (Temporal)": "high",
    "Doppler Shift (Severe)": "high",
    "High Network Congestion (Sudden Spike)": "high",
    "Resource Allocation Bugs": "high",
    "Antenna Failure": "critical",
    "Co-Channel Interference (Severe)": "high",
    "Jamming": "critical",
}

URGENCY_BY_ANOMALY = {
    "normal": "monitor",
    "Co-Channel Interference (Mild)": "soon",
    "High Network Congestion (Gradual Buildup)": "soon",
    "Buffer Overflow (Gradual Buildup)": "soon",
    "Faulty Handover Algorithm (Too Frequent)": "soon",
    "Faulty RF Filters (Temporal)": "urgent",
    "Doppler Shift (Severe)": "urgent",
    "High Network Congestion (Sudden Spike)": "urgent",
    "Resource Allocation Bugs": "urgent",
    "Antenna Failure": "immediate",
    "Co-Channel Interference (Severe)": "urgent",
    "Jamming": "immediate",
}


def parse_troubleshooting_sections(ticket_text):
    text = _strip(ticket_text)
    if not text:
        return {}

    parsed = {}
    for key, pattern in TICKET_SECTION_PATTERNS.items():
        match = pattern.search(text)
        if match:
            parsed[key] = " ".join(match.group(1).split())
    return parsed


def _ground_truth_anomaly_type(row):
    anomalies = row.get("anomalies_parsed") or {}
    if anomalies and anomalies.get("exists"):
        return anomalies.get("type") or "normal"
    return "normal"


def _prediction_matches_ground_truth(prediction, row):
    expected_type = _ground_truth_anomaly_type(row)
    expected_anomaly = expected_type != "normal"

    if bool(prediction.get("anomaly")) != expected_anomaly:
        return False
    if not expected_anomaly:
        return True
    return (prediction.get("type") or "normal") == expected_type


def _severity_for_row(anomaly_type, prediction):
    severity = SEVERITY_BY_ANOMALY.get(anomaly_type, "high")
    duration_regime = prediction.get("duration_regime")
    duration_steps = prediction.get("duration_steps")
    if anomaly_type != "normal" and duration_regime == "full":
        if severity == "medium":
            return "high"
    if anomaly_type == "normal":
        return "info"
    if duration_steps is not None and int(duration_steps) >= 28 and severity == "high":
        return "critical" if anomaly_type in {"Antenna Failure", "Jamming"} else "high"
    return severity


def _urgency_for_row(anomaly_type, prediction):
    urgency = URGENCY_BY_ANOMALY.get(anomaly_type, "urgent")
    duration_regime = prediction.get("duration_regime")
    if anomaly_type != "normal" and duration_regime == "full":
        if urgency == "soon":
            return "urgent"
    return urgency


def build_memory_result_from_dataset(row, prediction, payload):
    anomaly_type = _ground_truth_anomaly_type(row)
    catalogue_entry = get_catalogue_entry(anomaly_type)
    anomalies = row.get("anomalies_parsed") or {}
    sections = parse_troubleshooting_sections(anomalies.get("troubleshooting_tickets"))
    description = _strip(row.get("description"))

    summary = sections.get("issue") or (
        "No anomaly was detected for this window."
        if anomaly_type == "normal"
        else f"Dataset reference case for {anomaly_type}."
    )
    likely_root_cause = sections.get("root_cause") or catalogue_entry["signal_signature"]
    operational_impact = sections.get("symptoms") or description or catalogue_entry["operator_goal"]
    operator_note = sections.get("resolution") or catalogue_entry["escalation_hint"] or catalogue_entry["operator_goal"]

    recommendation = build_default_catalogue_recommendation(
        anomaly_type,
        summary=summary,
        likely_root_cause=likely_root_cause,
        operational_impact=operational_impact,
        operator_note=operator_note,
        remedy_limit=min(3, len(catalogue_entry["remedies"])) or 1,
        prevention_limit=min(2, len(catalogue_entry["prevention_measures"])) or 1,
        verification_limit=min(2, len(catalogue_entry["verification_steps"])) or 1,
    )

    recommendation.update(
        {
            "severity": _severity_for_row(anomaly_type, prediction),
            "urgency": _urgency_for_row(anomaly_type, prediction),
            "confidence_posture": "medium" if anomaly_type == "normal" else "high",
            "escalation_needed": anomaly_type != "normal"
            and recommendation["urgency"] in {"urgent", "immediate"},
        }
    )
    return {
        "row_index": int(row.name),
        "payload": payload,
        "catalogue_entry": catalogue_entry,
        "recommendation": recommendation,
        "prediction": prediction,
        "model": "dataset_grounded_memory",
        "mode": "dataset_grounded_memory",
        "raw_done": True,
    }


def build_student_memory_csv(
    save_csv,
    start_row=0,
    end_row=None,
    resume=False,
    selected_row_indices=None,
    require_prediction_match=True,
):
    inference.get_loaded_models()
    dataset_df = load_dataset_with_parsed_columns()

    total_rows = len(dataset_df)
    if selected_row_indices is None:
        end_row = total_rows - 1 if end_row is None else min(end_row, total_rows - 1)
        if start_row < 0 or start_row >= total_rows:
            raise IndexError(f"start-row must be between 0 and {total_rows - 1}.")
        if end_row < start_row:
            raise IndexError("end-row must be greater than or equal to start-row.")
        selected_row_indices = list(range(start_row, end_row + 1))

    selected_row_indices = [row_index for row_index in selected_row_indices if 0 <= row_index < total_rows]

    completed = load_processed_row_indices(save_csv) if resume else set()
    target_total = len(selected_row_indices)
    print(f"Building student memory rows into {save_csv}")

    for offset, row_index in enumerate(selected_row_indices, start=1):
        if row_index in completed:
            print(f"[{offset}/{target_total}] Skipping row {row_index} (already present).")
            continue

        row = dataset_df.iloc[row_index]

        try:
            sample = {
                "statistics": row["stats_parsed"],
                "KPIs": row["kpis_parsed"],
                "labels": row["labels_parsed"],
                "sampling_rate": row.get("sampling_rate", 0.0),
            }
            prediction = inference.predict(sample)
            if require_prediction_match and not _prediction_matches_ground_truth(prediction, row):
                csv_row = build_csv_row(
                    row,
                    mode="dataset_grounded_memory",
                    status="error",
                    error_message="prediction signature does not match dataset anomaly label",
                    include_source_text=False,
                )
                append_csv_row(save_csv, csv_row)
                print(f"[{offset}/{target_total}] Skipped row {row_index} (prediction mismatch)")
                continue
            payload = build_recommender_payload(row, prediction)
            result = build_memory_result_from_dataset(row, prediction, payload)
            csv_row = build_csv_row(
                row,
                result,
                mode=result["mode"],
                include_source_text=False,
            )
        except Exception as exc:
            csv_row = build_csv_row(
                row,
                mode="dataset_grounded_memory",
                status="error",
                error_message=str(exc),
                include_source_text=False,
            )

        append_csv_row(save_csv, csv_row)
        print(f"[{offset}/{target_total}] Saved memory row {row_index}")


def _case_anomaly_type(case):
    if not case["predicted_anomaly"]:
        return "normal"
    return case["predicted_type"] or case["catalogue_anomaly_type"] or "normal"


def _top_kpi_names(top_kpis):
    if not isinstance(top_kpis, dict):
        return set()
    return set(top_kpis.keys())


@lru_cache(maxsize=8)
def load_memory_cases(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return []

    cases = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if _strip(row.get("status")).lower() != "ok":
                continue

            predicted_anomaly = _parse_bool(row.get("predicted_anomaly"))
            predicted_type = _strip(row.get("predicted_type")) or ("normal" if not predicted_anomaly else "")
            case = {
                "row_index": _parse_int(row.get("row_index"), default=-1),
                "predicted_anomaly": predicted_anomaly,
                "predicted_confidence": _parse_float(row.get("predicted_confidence")),
                "predicted_type": predicted_type,
                "predicted_type_confidence": _parse_float(row.get("predicted_type_confidence")),
                "predicted_duration_steps": _parse_int(row.get("predicted_duration_steps")),
                "predicted_duration_regime": _strip(row.get("predicted_duration_regime")),
                "catalogue_anomaly_type": _strip(row.get("catalogue_anomaly_type")) or "normal",
                "application": _normalize_text(row.get("application")),
                "zone": _normalize_text(row.get("zone")),
                "mobility": _normalize_text(row.get("mobility")),
                "congestion": _normalize_text(row.get("congestion")),
                "severity": _strip(row.get("severity")),
                "urgency": _strip(row.get("urgency")),
                "confidence_posture": _strip(row.get("confidence_posture")),
                "selected_remedy_ids": _parse_json(row.get("selected_remedy_ids"), []),
                "selected_prevention_ids": _parse_json(row.get("selected_prevention_ids"), []),
                "selected_verification_ids": _parse_json(row.get("selected_verification_ids"), []),
                "recommended_actions": _parse_json(row.get("recommended_actions"), []),
                "prevention_measures": _parse_json(row.get("prevention_measures"), []),
                "verification_steps": _parse_json(row.get("verification_steps"), []),
                "summary": _strip(row.get("summary")),
                "likely_root_cause": _strip(row.get("likely_root_cause")),
                "operational_impact": _strip(row.get("operational_impact")),
                "operator_note": _strip(row.get("operator_note")),
                "affected_kpis": _parse_json(row.get("affected_kpis"), []),
                "focus_kpis": _parse_json(row.get("focus_kpis"), _parse_json(row.get("affected_kpis"), [])),
                "top_kpis": _parse_json(row.get("top_kpis"), {}),
            }
            cases.append(case)

    return cases


def _score_memory_case(case, prediction, payload):
    score = 0.0
    query_type = choose_catalogue_anomaly_type(prediction)
    case_type = _case_anomaly_type(case)
    query_top_kpis = payload["signal_summary"].get("top_kpis") or {}
    query_focus_kpis = payload["signal_summary"].get("focus_kpis") or []

    if case["predicted_anomaly"] == bool(prediction["anomaly"]):
        score += 18.0
    else:
        score -= 12.0

    if case_type == query_type:
        score += 35.0
    elif case["catalogue_anomaly_type"] == query_type:
        score += 22.0

    context = payload["context"]
    if case["application"] == _normalize_text(context.get("application")):
        score += 6.0
    if case["zone"] == _normalize_text(context.get("zone")):
        score += 5.0
    if case["mobility"] == _normalize_text(context.get("mobility")):
        score += 4.0
    if case["congestion"] == _normalize_text(context.get("congestion")):
        score += 4.0

    if case["predicted_duration_regime"] and case["predicted_duration_regime"] == (
        prediction.get("duration_regime") or ""
    ):
        score += 6.0

    predicted_confidence = case.get("predicted_confidence")
    if predicted_confidence is not None and prediction.get("confidence") is not None:
        score += max(0.0, 6.0 - abs(predicted_confidence - float(prediction["confidence"])) * 12.0)

    case_type_conf = case.get("predicted_type_confidence")
    if case_type_conf is not None and prediction.get("type_confidence") is not None:
        score += max(0.0, 4.0 - abs(case_type_conf - float(prediction["type_confidence"])) * 8.0)

    case_duration = case.get("predicted_duration_steps")
    query_duration = prediction.get("duration_steps")
    if case_duration is not None and query_duration is not None:
        score += max(0.0, 6.0 - min(abs(case_duration - int(query_duration)), 6.0))

    top_kpi_overlap = len(_top_kpi_names(case["top_kpis"]) & _top_kpi_names(query_top_kpis))
    score += min(top_kpi_overlap, 3) * 3.0

    focus_overlap = len(set(case.get("focus_kpis") or []) & set(query_focus_kpis))
    score += min(focus_overlap, 3) * 2.0
    return score


def retrieve_similar_cases(memory_cases, prediction, payload, top_k=3, exclude_row_index=None):
    if not memory_cases:
        return []

    query_type = choose_catalogue_anomaly_type(prediction)
    candidates = [
        case
        for case in memory_cases
        if case["row_index"] != exclude_row_index and case["predicted_anomaly"] == bool(prediction["anomaly"])
    ]

    same_type = [case for case in candidates if _case_anomaly_type(case) == query_type]
    if same_type:
        candidates = same_type
    elif not candidates:
        candidates = [case for case in memory_cases if case["row_index"] != exclude_row_index]

    ranked = sorted(
        candidates,
        key=lambda case: _score_memory_case(case, prediction, payload),
        reverse=True,
    )
    return ranked[:max(0, top_k)]


def compact_case_for_prompt(case):
    return {
        "row_index": case["row_index"],
        "structured_signature": {
            "predicted_anomaly": case["predicted_anomaly"],
            "predicted_type": _case_anomaly_type(case),
            "predicted_confidence": case["predicted_confidence"],
            "predicted_duration_steps": case["predicted_duration_steps"],
            "predicted_duration_regime": case["predicted_duration_regime"],
            "application": case["application"],
            "zone": case["zone"],
            "mobility": case["mobility"],
            "congestion": case["congestion"],
            "focus_kpis": case.get("focus_kpis") or [],
            "top_kpis": case["top_kpis"],
        },
        "reference_decision": {
            "severity": case["severity"],
            "urgency": case["urgency"],
            "confidence_posture": case["confidence_posture"],
            "selected_remedy_ids": case["selected_remedy_ids"],
            "selected_prevention_ids": case["selected_prevention_ids"],
            "selected_verification_ids": case["selected_verification_ids"],
            "summary": case["summary"],
            "likely_root_cause": case["likely_root_cause"],
            "operational_impact": case["operational_impact"],
        "operator_note": case["operator_note"],
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a dataset-grounded memory CSV that uses structured 3-model outputs plus "
            "catalogue-grounded reference decisions derived directly from the dataset."
        )
    )
    parser.add_argument(
        "--save-csv",
        default=str(DEFAULT_MEMORY_CSV),
        help="Output CSV that the student recommender can use as structured memory.",
    )
    parser.add_argument("--start-row", type=int, default=0, help="Starting row_index to process.")
    parser.add_argument("--end-row", type=int, default=None, help="Ending row_index to process.")
    parser.add_argument(
        "--split",
        default=MODEL_TRAIN_SPLIT_NAME,
        help="Named split to use when building student memory. Defaults to the model-training split.",
    )
    parser.add_argument(
        "--row-indices-csv",
        default=None,
        help="Optional CSV with a row_index column to override the named split.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip row indices that already exist in the target memory CSV.",
    )
    parser.add_argument(
        "--allow-prediction-mismatch",
        action="store_true",
        help="Keep rows even when the 3-model prediction signature does not match the dataset anomaly label.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_row_indices = resolve_selected_row_indices(
        split_name=args.split,
        row_indices_csv=args.row_indices_csv,
        default_range=(
            range(args.start_row, args.end_row + 1)
            if args.end_row is not None
            else None
        ),
    )
    build_student_memory_csv(
        save_csv=args.save_csv,
        start_row=args.start_row,
        end_row=args.end_row,
        resume=args.resume,
        selected_row_indices=selected_row_indices or None,
        require_prediction_match=not args.allow_prediction_mismatch,
    )


if __name__ == "__main__":
    main()
