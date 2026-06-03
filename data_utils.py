import sys
from pathlib import Path

LOCAL_PYTHON_PACKAGES = Path(__file__).with_name(".python_packages")
if LOCAL_PYTHON_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PYTHON_PACKAGES))

import numpy as np
import pandas as pd
from datasets import DownloadMode, load_dataset


DATASET_NAME = "AliMaatouk/TelecomTS"
LOCAL_DATA_CSV = Path("data.csv")


def parse_col(value):
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return None
    try:
        return eval(value, {"array": np.array, "np": np})
    except Exception:
        return None


def load_or_fetch_dataframe(cache_csv: bool = True) -> pd.DataFrame:
    if LOCAL_DATA_CSV.exists():
        print(f"Loading local dataset from {LOCAL_DATA_CSV}...")
        return pd.read_csv(LOCAL_DATA_CSV)

    print(f"Local data.csv not found. Downloading {DATASET_NAME} from Hugging Face...")
    dataset = load_dataset(
        DATASET_NAME,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    df = dataset["train"].to_pandas()

    if cache_csv:
        temp_csv = LOCAL_DATA_CSV.with_suffix(".tmp.csv")
        df.to_csv(temp_csv, index=False)
        temp_csv.replace(LOCAL_DATA_CSV)
        print(f"Saved local cache to {LOCAL_DATA_CSV}")

    return df
