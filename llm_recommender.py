import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime_bootstrap import ensure_repo_python

ensure_repo_python()

import argparse
import json

import inference
from llm_shared import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT,
    RESPONSE_JSON_INSTRUCTIONS,
    append_csv_row,
    build_csv_row,
    build_default_catalogue_recommendation,
    call_ollama,
    catalogue_prompt_view,
    finalize_catalogue_recommendation,
    get_catalogue_entry,
    load_dataset_with_parsed_columns,
    load_processed_row_indices,
    print_structured_recommendation,
)
from student_memory import (
    DEFAULT_MEMORY_CSV,
    build_recommender_payload,
    choose_catalogue_anomaly_type,
    compact_case_for_prompt,
    load_memory_cases,
    retrieve_similar_cases,
)
from split_utils import TEST_SPLIT_NAME, resolve_selected_row_indices


def build_messages(payload, catalogue_entry, past_cases=None):
    past_cases = past_cases or []
    instructions = (
        "You are a telecom operations assistant acting as a dataset-grounded remediation model. "
        "For the current event, you do not have access to raw description or QnA. "
        "Use only the structured 3-model prediction, the event context, the reviewed remedy catalogue entry, "
        "and any retrieved past solved cases. "
        "Treat past solved cases as dataset-grounded guidance rather than text to copy blindly, and adapt them to the current confidence, "
        "predicted anomaly type, and predicted duration. "
        "Be conservative with uncertainty: when confidence is not high, recommend verification before "
        "disruptive action. "
        "If no anomaly is predicted, keep the answer monitoring-oriented and non-disruptive. "
        "Long predicted duration or severe anomaly signatures should increase urgency. "
        "Choose only catalogue IDs from the provided entry and do not invent new remedies. "
        + RESPONSE_JSON_INSTRUCTIONS
    )

    user_prompt = {
        "task": "Produce a troubleshooting recommendation for this telecom event.",
        "requirements": [
            "Base the answer on the prediction and context provided.",
            "Use retrieved past cases as dataset-grounded prior knowledge, not as text to copy blindly.",
            "Keep recommended actions ordered from most useful to least useful.",
            "Use verification steps to confirm the diagnosis before or during remediation.",
            "Match the selected IDs to the supplied catalogue entry exactly.",
            "Do not invent remedies, prevention measures, or verification steps outside the catalogue entry.",
        ],
        "event": payload,
        "catalogue_entry": catalogue_prompt_view(catalogue_entry),
        "past_solved_cases": past_cases,
    }

    return instructions, user_prompt


def build_prompt(payload, catalogue_entry, past_cases=None):
    instructions, user_prompt = build_messages(payload, catalogue_entry, past_cases=past_cases)
    return f"{instructions}\n\nEVENT:\n{json.dumps(user_prompt, ensure_ascii=True, indent=2)}"


def build_no_anomaly_result(row, payload):
    recommendation = build_default_catalogue_recommendation(
        "normal",
        summary="No anomaly was detected for this row.",
        likely_root_cause="The pipeline did not detect an anomaly in this window.",
        operational_impact="No disruptive remediation is required unless later windows degrade.",
        operator_note="No anomaly detected. Keep the cell in monitor-only mode and preserve the baseline.",
        remedy_limit=1,
        prevention_limit=1,
        verification_limit=1,
    )
    return {
        "row_index": int(row.name),
        "payload": payload,
        "catalogue_entry": get_catalogue_entry("normal"),
        "recommendation": recommendation,
        "prediction": {
            "anomaly": False,
            "confidence": payload["prediction"]["confidence"],
            "type": None,
            "type_confidence": None,
            "duration_steps": None,
            "duration_regime": None,
            "duration_ready": False,
        },
        "model": None,
        "mode": "deterministic_recommender_no_anomaly",
        "raw_done": True,
    }


def recommend_for_row(
    row,
    model=DEFAULT_OLLAMA_MODEL,
    timeout=DEFAULT_OLLAMA_TIMEOUT,
    memory_cases=None,
    top_k_memory=3,
):
    sample = {
        "statistics": row["stats_parsed"],
        "KPIs": row["kpis_parsed"],
        "labels": row["labels_parsed"],
        "sampling_rate": row.get("sampling_rate", 0.0),
    }
    prediction = inference.predict(sample)
    payload = build_recommender_payload(row, prediction)

    if not prediction["anomaly"]:
        return build_no_anomaly_result(row, payload)

    anomaly_type = choose_catalogue_anomaly_type(prediction)
    catalogue_entry = get_catalogue_entry(anomaly_type)
    retrieved_cases = retrieve_similar_cases(
        memory_cases or [],
        prediction,
        payload,
        top_k=top_k_memory,
        exclude_row_index=int(row.name),
    )
    prompt_cases = [compact_case_for_prompt(case) for case in retrieved_cases]
    raw_recommendation, raw_response = call_ollama(
        build_prompt(payload, catalogue_entry, past_cases=prompt_cases),
        model=model,
        timeout=timeout,
    )
    recommendation = finalize_catalogue_recommendation(raw_recommendation, catalogue_entry)
    return {
        "row_index": int(row.name),
        "payload": payload,
        "catalogue_entry": catalogue_entry,
        "recommendation": recommendation,
        "prediction": prediction,
        "model": model,
        "mode": "local_ollama_llm_recommender",
        "raw_done": raw_response.get("done"),
        "memory_case_row_indices": [case["row_index"] for case in retrieved_cases],
    }


def run_batch(
    df,
    save_csv,
    model,
    timeout,
    start_row=0,
    end_row=None,
    resume=True,
    memory_cases=None,
    top_k_memory=3,
    selected_row_indices=None,
):
    total_rows = len(df)
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
    print(f"Writing recommender rows to {save_csv}")

    for offset, row_index in enumerate(selected_row_indices, start=1):
        if row_index in completed:
            print(f"[{offset}/{target_total}] Skipping row {row_index} (already present).")
            continue

        row = df.iloc[row_index]
        try:
            result = recommend_for_row(
                row,
                model=model,
                timeout=timeout,
                memory_cases=memory_cases,
                top_k_memory=top_k_memory,
            )
            csv_row = build_csv_row(
                row,
                result,
                mode=result["mode"],
                include_source_text=False,
            )
            append_csv_row(save_csv, csv_row)
            predicted_type = result["prediction"].get("type") or "normal"
            duration = result["prediction"].get("duration_steps")
            print(
                f"[{offset}/{target_total}] Saved row {row_index} "
                f"type={predicted_type} duration={duration}"
            )
        except Exception as exc:
            csv_row = build_csv_row(
                row,
                mode="local_ollama_llm_recommender",
                status="error",
                error_message=str(exc),
                include_source_text=False,
            )
            append_csv_row(save_csv, csv_row)
            print(f"[{offset}/{target_total}] Error on row {row_index}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the 3-stage anomaly pipeline and generate a catalogue-grounded recommendation "
            "with a local LLM via Ollama."
        )
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
        help="Optional path to save the full payload, prediction, and recommendation as JSON.",
    )
    parser.add_argument(
        "--write-csv",
        dest="save_csv",
        default=None,
        help="Optional CSV path to append recommendation rows.",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Process all rows or a selected row range and append results to a CSV.",
    )
    parser.add_argument("--start-row", type=int, default=0, help="Starting row_index for batch mode.")
    parser.add_argument("--end-row", type=int, default=None, help="Ending row_index for batch mode.")
    parser.add_argument(
        "--memory-csv",
        default=None,
        help="Optional dataset-grounded memory CSV built from 3-model predictions plus dataset reference rows.",
    )
    parser.add_argument(
        "--top-k-memory",
        type=int,
        default=3,
        help="How many similar past solved cases to inject into the student prompt.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable past-case retrieval even if a memory CSV is available.",
    )
    parser.add_argument(
        "--split",
        default=TEST_SPLIT_NAME,
        help="Named split to use for --all-rows. Defaults to the final test split.",
    )
    parser.add_argument(
        "--row-indices-csv",
        default=None,
        help="Optional CSV with a row_index column to override the named split for --all-rows.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="In batch mode, do not skip rows that already exist in the target CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_dataset_with_parsed_columns()
    default_memory_csv = str(DEFAULT_MEMORY_CSV) if Path(DEFAULT_MEMORY_CSV).exists() else None
    memory_csv = None if args.no_memory else (args.memory_csv or default_memory_csv)
    memory_cases = load_memory_cases(str(Path(memory_csv).resolve())) if memory_csv else []

    if memory_csv:
        print(f"Loaded {len(memory_cases)} student memory cases from {memory_csv}")
    else:
        print("Running recommender without dataset-grounded memory")

    if args.all_rows:
        save_csv = args.save_csv or "llm_recommender_all_rows.csv"
        selected_row_indices = resolve_selected_row_indices(
            split_name=args.split,
            row_indices_csv=args.row_indices_csv,
            default_range=(
                range(args.start_row, args.end_row + 1)
                if args.end_row is not None
                else None
            ),
        )
        run_batch(
            df,
            save_csv=save_csv,
            model=args.model,
            timeout=args.timeout,
            start_row=args.start_row,
            end_row=args.end_row,
            resume=not args.no_resume,
            memory_cases=memory_cases,
            top_k_memory=args.top_k_memory,
            selected_row_indices=selected_row_indices or None,
        )
        return

    if args.row_index < 0 or args.row_index >= len(df):
        raise IndexError(f"row-index must be between 0 and {len(df) - 1}.")

    row = df.iloc[args.row_index]
    result = recommend_for_row(
        row,
        model=args.model,
        timeout=args.timeout,
        memory_cases=memory_cases,
        top_k_memory=args.top_k_memory,
    )
    print_structured_recommendation(result, "LLM RECOMMENDER")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"Saved JSON result to {args.save_json}")

    if args.save_csv:
        append_csv_row(
            args.save_csv,
            build_csv_row(
                row,
                result,
                mode=result["mode"],
                include_source_text=False,
            ),
        )
        print(f"Saved CSV row to {args.save_csv}")


if __name__ == "__main__":
    main()
