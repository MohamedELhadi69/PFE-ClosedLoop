import csv
import json
from pathlib import Path


MODEL_TRAIN_SPLIT_NAME = "model_train"
TEST_SPLIT_NAME = "test"

SPLIT_INDEX_FILES = {
    MODEL_TRAIN_SPLIT_NAME: Path("row_indices_model_train.csv"),
    TEST_SPLIT_NAME: Path("row_indices_test.csv"),
}
SPLIT_MANIFEST_PATH = Path("dataset_splits.json")


def save_row_indices(path, row_indices):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index"])
        for row_index in row_indices:
            writer.writerow([int(row_index)])


def load_row_indices(path):
    path = Path(path)
    if not path.exists():
        return []

    row_indices = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = (row.get("row_index") or "").strip()
            if value:
                row_indices.append(int(value))
    return row_indices


def get_split_index_path(split_name):
    return SPLIT_INDEX_FILES.get(split_name)


def load_split_row_indices(split_name):
    path = get_split_index_path(split_name)
    if path is None:
        raise ValueError(f"Unknown split name: {split_name}")
    return load_row_indices(path)


def resolve_selected_row_indices(
    *,
    split_name=None,
    row_indices_csv=None,
    default_range=None,
):
    if row_indices_csv:
        return load_row_indices(row_indices_csv)

    if split_name:
        path = get_split_index_path(split_name)
        if path and path.exists():
            return load_row_indices(path)

    return list(default_range or [])


def save_split_manifest(payload, path=SPLIT_MANIFEST_PATH):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
