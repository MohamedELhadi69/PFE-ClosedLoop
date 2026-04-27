from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


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
    dataset = load_dataset(DATASET_NAME)
    df = dataset["train"].to_pandas()

    if cache_csv:
        temp_csv = LOCAL_DATA_CSV.with_suffix(".tmp.csv")
        df.to_csv(temp_csv, index=False)
        temp_csv.replace(LOCAL_DATA_CSV)
        print(f"Saved local cache to {LOCAL_DATA_CSV}")

    return df
