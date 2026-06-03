import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

LOCAL_PYTHON_PACKAGES = Path(__file__).with_name(".python_packages")
if LOCAL_PYTHON_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PYTHON_PACKAGES))

from runtime_bootstrap import ensure_repo_python

ensure_repo_python()

import argparse
import csv
import json
import re
from difflib import SequenceMatcher

import pandas as pd


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def parse_json_list(value):
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def normalize_metric_item(value):
    text = safe_text(value).lower()
    return text


def safe_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def tokenize_text(text):
    return TOKEN_PATTERN.findall(safe_text(text).lower())


def token_f1_score(left_text, right_text):
    left_tokens = tokenize_text(left_text)
    right_tokens = tokenize_text(right_text)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0

    left_counts = {}
    right_counts = {}
    for token in left_tokens:
        left_counts[token] = left_counts.get(token, 0) + 1
    for token in right_tokens:
        right_counts[token] = right_counts.get(token, 0) + 1

    overlap = 0
    for token, left_count in left_counts.items():
        overlap += min(left_count, right_counts.get(token, 0))

    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def sequence_similarity(left_text, right_text):
    return SequenceMatcher(None, safe_text(left_text), safe_text(right_text)).ratio()


def set_metrics(predicted_items, reference_items):
    predicted_set = {
        normalize_metric_item(item)
        for item in predicted_items
        if normalize_metric_item(item)
    }
    reference_set = {
        normalize_metric_item(item)
        for item in reference_items
        if normalize_metric_item(item)
    }

    if not predicted_set and not reference_set:
        return {
            "exact": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "jaccard": 1.0,
        }

    intersection = len(predicted_set & reference_set)
    union = len(predicted_set | reference_set)
    precision = intersection / len(predicted_set) if predicted_set else 0.0
    recall = intersection / len(reference_set) if reference_set else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "exact": float(predicted_set == reference_set),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": intersection / union if union else 1.0,
    }


def dedupe_latest_rows(df):
    return df.drop_duplicates(subset=["row_index"], keep="last").copy()


def load_csv(path):
    return pd.read_csv(path, low_memory=False)


def summarize_errors(df, actor_name):
    if "status" not in df.columns:
        return pd.DataFrame(columns=["actor", "error_message", "count"])
    errors = df[df["status"].astype(str).str.strip().str.lower() == "error"].copy()
    if errors.empty:
        return pd.DataFrame(columns=["actor", "error_message", "count"])
    summary = (
        errors.groupby("error_message")
        .size()
        .reset_index(name="count")
        .sort_values(["count", "error_message"], ascending=[False, True])
    )
    summary.insert(0, "actor", actor_name)
    return summary


def build_row_metrics(teacher_ok_df, student_ok_df):
    teacher_map = teacher_ok_df.set_index("row_index")
    student_map = student_ok_df.set_index("row_index")
    common_row_indices = sorted(set(teacher_map.index) & set(student_map.index))

    rows = []
    for row_index in common_row_indices:
        teacher_row = teacher_map.loc[row_index]
        student_row = student_map.loc[row_index]

        remedy_metrics = set_metrics(
            parse_json_list(student_row["recommended_actions"]),
            parse_json_list(teacher_row["recommended_actions"]),
        )
        prevention_metrics = set_metrics(
            parse_json_list(student_row["prevention_measures"]),
            parse_json_list(teacher_row["prevention_measures"]),
        )
        verification_metrics = set_metrics(
            parse_json_list(student_row["verification_steps"]),
            parse_json_list(teacher_row["verification_steps"]),
        )

        severity_exact = float(safe_text(student_row["severity"]) == safe_text(teacher_row["severity"]))
        urgency_exact = float(safe_text(student_row["urgency"]) == safe_text(teacher_row["urgency"]))
        confidence_exact = float(
            safe_text(student_row["confidence_posture"]) == safe_text(teacher_row["confidence_posture"])
        )
        escalation_exact = float(
            normalize_bool(student_row["escalation_needed"]) == normalize_bool(teacher_row["escalation_needed"])
        )
        anomaly_exact = float(
            safe_text(student_row["catalogue_anomaly_type"]) == safe_text(teacher_row["catalogue_anomaly_type"])
        )

        summary_token_f1 = token_f1_score(student_row["summary"], teacher_row["summary"])
        root_cause_token_f1 = token_f1_score(
            student_row["likely_root_cause"],
            teacher_row["likely_root_cause"],
        )
        impact_token_f1 = token_f1_score(
            student_row["operational_impact"],
            teacher_row["operational_impact"],
        )
        note_token_f1 = token_f1_score(student_row["operator_note"], teacher_row["operator_note"])

        row_exact = float(
            severity_exact
            and urgency_exact
            and confidence_exact
            and escalation_exact
            and anomaly_exact
            and remedy_metrics["exact"]
            and prevention_metrics["exact"]
            and verification_metrics["exact"]
        )

        plan_score = (
            0.15 * severity_exact
            + 0.10 * urgency_exact
            + 0.05 * confidence_exact
            + 0.05 * escalation_exact
            + 0.10 * anomaly_exact
            + 0.20 * remedy_metrics["f1"]
            + 0.10 * prevention_metrics["f1"]
            + 0.10 * verification_metrics["f1"]
            + 0.05 * summary_token_f1
            + 0.05 * root_cause_token_f1
            + 0.03 * impact_token_f1
            + 0.02 * note_token_f1
        )

        rows.append(
            {
                "row_index": int(row_index),
                "ground_truth_anomaly_type": safe_text(teacher_row["ground_truth_anomaly_type"]),
                "is_anomaly_row": float(safe_text(teacher_row["ground_truth_anomaly_type"]) != "normal"),
                "catalogue_anomaly_type_teacher": safe_text(teacher_row["catalogue_anomaly_type"]),
                "catalogue_anomaly_type_student": safe_text(student_row["catalogue_anomaly_type"]),
                "severity_exact": severity_exact,
                "urgency_exact": urgency_exact,
                "confidence_posture_exact": confidence_exact,
                "escalation_exact": escalation_exact,
                "catalogue_anomaly_type_exact": anomaly_exact,
                "remedy_exact": remedy_metrics["exact"],
                "remedy_precision": remedy_metrics["precision"],
                "remedy_recall": remedy_metrics["recall"],
                "remedy_f1": remedy_metrics["f1"],
                "remedy_jaccard": remedy_metrics["jaccard"],
                "prevention_exact": prevention_metrics["exact"],
                "prevention_precision": prevention_metrics["precision"],
                "prevention_recall": prevention_metrics["recall"],
                "prevention_f1": prevention_metrics["f1"],
                "prevention_jaccard": prevention_metrics["jaccard"],
                "verification_exact": verification_metrics["exact"],
                "verification_precision": verification_metrics["precision"],
                "verification_recall": verification_metrics["recall"],
                "verification_f1": verification_metrics["f1"],
                "verification_jaccard": verification_metrics["jaccard"],
                "summary_token_f1": summary_token_f1,
                "root_cause_token_f1": root_cause_token_f1,
                "impact_token_f1": impact_token_f1,
                "operator_note_token_f1": note_token_f1,
                "summary_sequence_similarity": sequence_similarity(student_row["summary"], teacher_row["summary"]),
                "root_cause_sequence_similarity": sequence_similarity(
                    student_row["likely_root_cause"],
                    teacher_row["likely_root_cause"],
                ),
                "impact_sequence_similarity": sequence_similarity(
                    student_row["operational_impact"],
                    teacher_row["operational_impact"],
                ),
                "operator_note_sequence_similarity": sequence_similarity(
                    student_row["operator_note"],
                    teacher_row["operator_note"],
                ),
                "full_plan_exact": row_exact,
                "plan_score": plan_score,
            }
        )

    return pd.DataFrame(rows)


def build_field_metrics(row_metrics_df):
    if row_metrics_df.empty:
        return pd.DataFrame(columns=["metric", "mean", "std", "min", "max"])

    field_names = [
        "severity_exact",
        "urgency_exact",
        "confidence_posture_exact",
        "escalation_exact",
        "catalogue_anomaly_type_exact",
        "remedy_exact",
        "remedy_precision",
        "remedy_recall",
        "remedy_f1",
        "remedy_jaccard",
        "prevention_exact",
        "prevention_precision",
        "prevention_recall",
        "prevention_f1",
        "prevention_jaccard",
        "verification_exact",
        "verification_precision",
        "verification_recall",
        "verification_f1",
        "verification_jaccard",
        "summary_token_f1",
        "root_cause_token_f1",
        "impact_token_f1",
        "operator_note_token_f1",
        "summary_sequence_similarity",
        "root_cause_sequence_similarity",
        "impact_sequence_similarity",
        "operator_note_sequence_similarity",
        "full_plan_exact",
        "plan_score",
    ]
    rows = []
    for field_name in field_names:
        rows.append(
            {
                "metric": field_name,
                "mean": float(row_metrics_df[field_name].mean()),
                "std": float(row_metrics_df[field_name].std(ddof=0)),
                "min": float(row_metrics_df[field_name].min()),
                "max": float(row_metrics_df[field_name].max()),
            }
        )
    return pd.DataFrame(rows)


def subset_row_metrics(row_metrics_df, subset_name):
    if subset_name == "all":
        return row_metrics_df.copy()
    if subset_name == "anomaly_only":
        return row_metrics_df[row_metrics_df["ground_truth_anomaly_type"] != "normal"].copy()
    if subset_name == "normal_only":
        return row_metrics_df[row_metrics_df["ground_truth_anomaly_type"] == "normal"].copy()
    raise ValueError(f"Unknown subset name: {subset_name}")


def summarize_metric_subset(row_metrics_df, subset_name):
    subset_df = subset_row_metrics(row_metrics_df, subset_name)
    field_metrics_df = build_field_metrics(subset_df)
    summary = {"rows": int(len(subset_df))}
    if subset_df.empty:
        summary.update(
            {
                "plan_score": 0.0,
                "full_plan_exact": 0.0,
                "remedy_f1": 0.0,
                "prevention_f1": 0.0,
                "verification_f1": 0.0,
                "severity_exact": 0.0,
                "urgency_exact": 0.0,
                "summary_token_f1": 0.0,
                "root_cause_token_f1": 0.0,
                "impact_token_f1": 0.0,
            }
        )
        return summary, field_metrics_df

    metric_lookup = field_metrics_df.set_index("metric")["mean"].to_dict()
    for metric_name in [
        "plan_score",
        "full_plan_exact",
        "remedy_f1",
        "prevention_f1",
        "verification_f1",
        "severity_exact",
        "urgency_exact",
        "summary_token_f1",
        "root_cause_token_f1",
        "impact_token_f1",
    ]:
        summary[metric_name] = float(metric_lookup.get(metric_name, 0.0))
    return summary, field_metrics_df


def build_per_type_metrics(row_metrics_df, teacher_ok_df, student_ok_df):
    teacher_counts = teacher_ok_df.groupby("ground_truth_anomaly_type").size().to_dict()

    rows = []
    for anomaly_type, part in row_metrics_df.groupby("ground_truth_anomaly_type"):
        teacher_total = int(teacher_counts.get(anomaly_type, 0))
        student_total = int(len(part))
        rows.append(
            {
                "ground_truth_anomaly_type": anomaly_type,
                "teacher_ok_rows": teacher_total,
                "student_ok_rows": student_total,
                "pipeline_success_rate_vs_teacher": (
                    student_total / teacher_total if teacher_total else 0.0
                ),
                "rows_scored": int(len(part)),
                "severity_exact_rate": float(part["severity_exact"].mean()),
                "urgency_exact_rate": float(part["urgency_exact"].mean()),
                "confidence_posture_exact_rate": float(part["confidence_posture_exact"].mean()),
                "escalation_exact_rate": float(part["escalation_exact"].mean()),
                "catalogue_type_exact_rate": float(part["catalogue_anomaly_type_exact"].mean()),
                "remedy_f1_mean": float(part["remedy_f1"].mean()),
                "prevention_f1_mean": float(part["prevention_f1"].mean()),
                "verification_f1_mean": float(part["verification_f1"].mean()),
                "summary_token_f1_mean": float(part["summary_token_f1"].mean()),
                "root_cause_token_f1_mean": float(part["root_cause_token_f1"].mean()),
                "impact_token_f1_mean": float(part["impact_token_f1"].mean()),
                "operator_note_token_f1_mean": float(part["operator_note_token_f1"].mean()),
                "full_plan_exact_rate": float(part["full_plan_exact"].mean()),
                "plan_score_mean": float(part["plan_score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["plan_score_mean", "full_plan_exact_rate", "teacher_ok_rows"],
        ascending=[False, False, False],
    )


def write_markdown_report(path, summary, field_metrics_df, per_type_df, student_error_df, teacher_error_df):
    lines = []
    lines.append("# Student Test Evaluation")
    lines.append("")
    lines.append("## Reliability")
    lines.append("")
    lines.append(f"- Reference test rows (deduped): {summary['teacher_total_rows']}")
    lines.append(f"- Reference successful rows: {summary['teacher_ok_rows']}")
    lines.append(f"- Student test rows (deduped): {summary['student_total_rows']}")
    lines.append(f"- Student successful rows: {summary['student_ok_rows']}")
    lines.append(f"- Common scored rows: {summary['common_ok_rows']}")
    lines.append(
        f"- Student pipeline success vs reference-ok rows: {summary['pipeline_success_rate_vs_teacher']:.4%}"
    )
    lines.append(
        f"- End-to-end system score (coverage x content): {summary['system_score']:.4f}"
    )
    lines.append("")
    lines.append("## Scoring Policy")
    lines.append("")
    lines.append(
        f"- Anomaly rows used for remediation-quality scoring: {summary['anomaly_rows_scored']}"
    )
    lines.append(
        f"- Normal rows excluded from remediation-quality scoring: {summary['normal_rows_excluded']}"
    )
    lines.append("")
    lines.append("## Core Quality (Anomaly Rows Only)")
    lines.append("")

    top_metrics = [
        "full_plan_exact",
        "plan_score",
        "remedy_f1",
        "prevention_f1",
        "verification_f1",
        "severity_exact",
        "urgency_exact",
        "summary_token_f1",
        "root_cause_token_f1",
        "impact_token_f1",
    ]
    field_metric_lookup = field_metrics_df.set_index("metric")["mean"].to_dict()
    for metric_name in top_metrics:
        value = field_metric_lookup.get(metric_name)
        if value is None:
            continue
        lines.append(f"- {metric_name}: {value:.4f}")

    lines.append("")
    lines.append("## Per Anomaly Type")
    lines.append("")
    if per_type_df.empty:
        lines.append("No per-type metrics available.")
    else:
        lines.append("| anomaly_type | reference_ok | student_ok | success_rate | plan_score | full_plan_exact | remedy_f1 | prevention_f1 | verification_f1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in per_type_df.itertuples(index=False):
            lines.append(
                "| "
                f"{row.ground_truth_anomaly_type} | "
                f"{row.teacher_ok_rows} | "
                f"{row.student_ok_rows} | "
                f"{row.pipeline_success_rate_vs_teacher:.4f} | "
                f"{row.plan_score_mean:.4f} | "
                f"{row.full_plan_exact_rate:.4f} | "
                f"{row.remedy_f1_mean:.4f} | "
                f"{row.prevention_f1_mean:.4f} | "
                f"{row.verification_f1_mean:.4f} |"
            )

    lines.append("")
    lines.append("## Error Summary")
    lines.append("")
    if student_error_df.empty and teacher_error_df.empty:
        lines.append("No error rows.")
    else:
        if not student_error_df.empty:
            lines.append("### Student Errors")
            lines.append("")
            for row in student_error_df.itertuples(index=False):
                lines.append(f"- `{row.error_message}`: {row.count}")
            lines.append("")
        if not teacher_error_df.empty:
            lines.append("### Reference Errors")
            lines.append("")
            for row in teacher_error_df.itertuples(index=False):
                lines.append(f"- `{row.error_message}`: {row.count}")

    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(student_csv, teacher_csv, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    student_df = dedupe_latest_rows(load_csv(student_csv))
    teacher_df = dedupe_latest_rows(load_csv(teacher_csv))

    student_error_df = summarize_errors(student_df, "student")
    teacher_error_df = summarize_errors(teacher_df, "teacher")

    student_ok_df = student_df[student_df["status"].astype(str).str.strip().str.lower() == "ok"].copy()
    teacher_ok_df = teacher_df[teacher_df["status"].astype(str).str.strip().str.lower() == "ok"].copy()

    row_metrics_df = build_row_metrics(teacher_ok_df, student_ok_df)
    anomaly_row_metrics_df = subset_row_metrics(row_metrics_df, "anomaly_only")
    normal_row_metrics_df = subset_row_metrics(row_metrics_df, "normal_only")
    field_metrics_df = build_field_metrics(anomaly_row_metrics_df)
    per_type_df = build_per_type_metrics(anomaly_row_metrics_df, teacher_ok_df, student_ok_df)
    anomaly_only_summary, anomaly_only_metrics_df = summarize_metric_subset(
        row_metrics_df,
        "anomaly_only",
    )

    teacher_total_rows = int(len(teacher_df))
    teacher_ok_rows = int(len(teacher_ok_df))
    student_total_rows = int(len(student_df))
    student_ok_rows = int(len(student_ok_df))
    common_ok_rows = int(len(row_metrics_df))
    pipeline_success_rate = common_ok_rows / teacher_ok_rows if teacher_ok_rows else 0.0
    mean_plan_score = (
        float(anomaly_row_metrics_df["plan_score"].mean()) if len(anomaly_row_metrics_df) else 0.0
    )
    system_score = pipeline_success_rate * mean_plan_score

    summary = {
        "teacher_total_rows": teacher_total_rows,
        "teacher_ok_rows": teacher_ok_rows,
        "teacher_error_rows": int(len(teacher_df) - len(teacher_ok_df)),
        "student_total_rows": student_total_rows,
        "student_ok_rows": student_ok_rows,
        "student_error_rows": int(len(student_df) - len(student_ok_df)),
        "common_ok_rows": common_ok_rows,
        "pipeline_success_rate_vs_teacher": pipeline_success_rate,
        "mean_plan_score_on_common_ok_rows": mean_plan_score,
        "system_score": system_score,
        "anomaly_only": anomaly_only_summary,
        "anomaly_rows_scored": int(len(anomaly_row_metrics_df)),
        "normal_rows_excluded": int(len(normal_row_metrics_df)),
    }

    row_metrics_df.to_csv(output_dir / "student_eval_row_metrics.csv", index=False)
    field_metrics_df.to_csv(output_dir / "student_eval_field_metrics.csv", index=False)
    anomaly_only_metrics_df.to_csv(
        output_dir / "student_eval_field_metrics_anomaly_only.csv",
        index=False,
    )
    per_type_df.to_csv(output_dir / "student_eval_per_anomaly_type.csv", index=False)
    pd.concat([student_error_df, teacher_error_df], ignore_index=True).to_csv(
        output_dir / "student_eval_error_summary.csv",
        index=False,
    )

    pd.crosstab(
        row_metrics_df["catalogue_anomaly_type_teacher"],
        row_metrics_df["catalogue_anomaly_type_student"],
        rownames=["teacher"],
        colnames=["student"],
    ).to_csv(output_dir / "student_eval_catalogue_type_confusion.csv")

    severity_join = row_metrics_df.merge(
        teacher_ok_df[["row_index", "severity"]].rename(columns={"severity": "teacher_severity"}),
        on="row_index",
        how="left",
    ).merge(
        student_ok_df[["row_index", "severity"]].rename(columns={"severity": "student_severity"}),
        on="row_index",
        how="left",
    )
    pd.crosstab(
        severity_join["teacher_severity"],
        severity_join["student_severity"],
        rownames=["teacher"],
        colnames=["student"],
    ).to_csv(output_dir / "student_eval_severity_confusion.csv")

    urgency_join = row_metrics_df.merge(
        teacher_ok_df[["row_index", "urgency"]].rename(columns={"urgency": "teacher_urgency"}),
        on="row_index",
        how="left",
    ).merge(
        student_ok_df[["row_index", "urgency"]].rename(columns={"urgency": "student_urgency"}),
        on="row_index",
        how="left",
    )
    pd.crosstab(
        urgency_join["teacher_urgency"],
        urgency_join["student_urgency"],
        rownames=["teacher"],
        colnames=["student"],
    ).to_csv(output_dir / "student_eval_urgency_confusion.csv")

    (output_dir / "student_eval_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(
        output_dir / "student_eval_report.md",
        summary,
        field_metrics_df,
        per_type_df,
        student_error_df,
        teacher_error_df,
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation outputs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the student telecom remediation outputs on the held-out test split "
            "against reference outputs."
        )
    )
    parser.add_argument(
        "--student-csv",
        default="llm_recommender_test.csv",
        help="Student recommender CSV on the test split.",
    )
    parser.add_argument(
        "--reference-csv",
        "--teacher-csv",
        dest="teacher_csv",
        default="llm_reference_test.csv",
        help="Reference CSV on the test split.",
    )
    parser.add_argument(
        "--output-dir",
        default="student_test_evaluation",
        help="Directory where evaluation reports and CSVs will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(
        student_csv=Path(args.student_csv),
        teacher_csv=Path(args.teacher_csv),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
