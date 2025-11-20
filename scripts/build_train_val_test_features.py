"""
Build per-vehicle feature matrices for train, validation, and test splits.

This script:
- Loads raw SCANIA data (operational, TTE, specifications) from ADLS Gen2
- Uses the feature_engineering module to build per-vehicle features
- Ensures validation/test features are aligned with training feature columns
- Saves:
    - train_vehicle_features.csv
    - validation_vehicle_features.csv
    - test_vehicle_features.csv

Run this script from the repository root, e.g.:

    conda activate azure-pdm
    python scripts/build_train_val_test_features.py
"""

import os
import sys
from typing import Dict, List

import pandas as pd
from adlfs import AzureBlobFileSystem

# Make sure we can import from src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.feature_engineering import (
    build_train_features,
    build_eval_features,
)

# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------

ACCOUNT_NAME = "scaniapdmstorage"
FILE_SYSTEM = "scania-dataset"

# ⚠️ IMPORTANT:
# Use the storage account key for now. Do NOT commit this script with the key filled in.
# Consider reading from an environment variable in the future.
ACCOUNT_KEY = os.environ.get("SCANIA_STORAGE_ACCOUNT_KEY", "tMM0AEo95ysfrScREtwUNLpLj493AN1LfMlj4oIId+SruT7zsvNif48lHhY09ymSC8mXOW2o5FeI+AStLCeMiQ==") #"<PASTE-KEY-HERE>")

if ACCOUNT_KEY == "<PASTE-KEY-HERE>":
    print(
        "WARNING: ACCOUNT_KEY is still the placeholder. "
        "Set the SCANIA_STORAGE_ACCOUNT_KEY environment variable or edit this script."
    )

# Raw data paths inside the filesystem
RAW_PATHS = {
    "train": {
        "oper": f"{FILE_SYSTEM}/train/train_operational_readouts.csv",
        "tte": f"{FILE_SYSTEM}/train/train_tte.csv",
        "spec": f"{FILE_SYSTEM}/train/train_specifications.csv",
    },
    "validation": {
        "oper": f"{FILE_SYSTEM}/validation/validation_operational_readouts.csv",
        "tte": f"{FILE_SYSTEM}/validation/validation_labels.csv",
        "spec": f"{FILE_SYSTEM}/validation/validation_specifications.csv",
    },
    "test": {
        "oper": f"{FILE_SYSTEM}/test/test_operational_readouts.csv",
        "tte": f"{FILE_SYSTEM}/test/test_labels.csv",
        "spec": f"{FILE_SYSTEM}/test/test_specifications.csv",
    },
}

# Column names
VEHICLE_COL = "vehicle_id"
TIME_COL = "time_step"
TARGET_COL = "in_study_repair"

# Counter columns (single-value counters)
COUNTER_COLS: List[str] = [
    "171_0", "666_0", "427_0", "837_0",
    "309_0", "835_0", "370_0", "100_0",
]

# Histogram prefixes (each prefix corresponds to a set of bin columns)
HISTOGRAM_PREFIXES: List[str] = ["167_", "272_", "291_", "158_", "459_", "397_"]


# --------------------------------------------------------------------------------------
# Helper: Read CSV from ADLS
# --------------------------------------------------------------------------------------

def read_csv_from_adls(fs: AzureBlobFileSystem, path: str, **read_csv_kwargs) -> pd.DataFrame:
    """
    Read a CSV file from ADLS Gen2 into a pandas DataFrame.
    `path` is relative to the filesystem root, e.g. 'train/train_operational_readouts.csv'.
    """
    print(f"Loading: {path}")
    with fs.open(path, "rb") as f:
        df = pd.read_csv(f, **read_csv_kwargs)
    print(f"  → shape: {df.shape}")
    return df


# --------------------------------------------------------------------------------------
# Helper: Determine histogram groups based on prefixes and columns
# --------------------------------------------------------------------------------------

def infer_histogram_groups(columns, prefixes) -> Dict[str, List[str]]:
    """
    Build a dict: base_prefix -> [list of columns], e.g. "167" -> ["167_0", ..., "167_9"].
    """
    histogram_groups: Dict[str, List[str]] = {}
    for prefix in prefixes:
        cols = [c for c in columns if c.startswith(prefix)]
        if cols:
            histogram_groups[prefix[:-1]] = cols  # "167_" -> "167"
    return histogram_groups


# --------------------------------------------------------------------------------------
# Main build function
# --------------------------------------------------------------------------------------

def main():
    # Connect to ADLS
    fs = AzureBlobFileSystem(
        account_name=ACCOUNT_NAME,
        account_key=ACCOUNT_KEY,
    )
    print(f"Connected to ADLS: account={ACCOUNT_NAME}, filesystem={FILE_SYSTEM}")

    # -------------------------
    # 1) Load TRAIN raw data
    # -------------------------
    train_oper = read_csv_from_adls(fs, RAW_PATHS["train"]["oper"])
    train_tte = read_csv_from_adls(fs, RAW_PATHS["train"]["tte"])
    train_spec = read_csv_from_adls(fs, RAW_PATHS["train"]["spec"])

    # Infer histogram groups from TRAIN columns
    train_cols = train_oper.columns.tolist()
    histogram_groups = infer_histogram_groups(train_cols, HISTOGRAM_PREFIXES)
    print("Histogram groups inferred from TRAIN:")
    print({k: len(v) for k, v in histogram_groups.items()})

    # Filter counter cols to those actually present in TRAIN
    counter_cols_present = [c for c in COUNTER_COLS if c in train_cols]
    print("Counter columns present:", counter_cols_present)

    # -------------------------
    # 2) Build TRAIN features
    # -------------------------
    print("\nBuilding TRAIN features...")
    train_features, spec_feature_cols, feature_columns = build_train_features(
        df_oper=train_oper,
        df_tte=train_tte,
        df_spec=train_spec,
        counter_cols=counter_cols_present,
        histogram_groups=histogram_groups,
        time_col=TIME_COL,
        vehicle_col=VEHICLE_COL,
        target_col=TARGET_COL,
    )

    print("TRAIN feature matrix shape:", train_features.shape)
    train_out_path = "train_vehicle_features.csv"
    train_features.to_csv(train_out_path, index=False)
    print(f"Saved TRAIN features to: {train_out_path}")

    # -------------------------
    # 3) Build VALIDATION features
    # -------------------------
    print("\nBuilding VALIDATION features...")
    val_oper = read_csv_from_adls(fs, RAW_PATHS["validation"]["oper"])
    val_tte = read_csv_from_adls(fs, RAW_PATHS["validation"]["tte"])
    val_spec = read_csv_from_adls(fs, RAW_PATHS["validation"]["spec"])

    val_features = build_eval_features(
        df_oper=val_oper,
        df_tte=val_tte,
        df_spec=val_spec,
        counter_cols=counter_cols_present,
        histogram_groups=histogram_groups,
        spec_feature_cols=spec_feature_cols,
        feature_columns=feature_columns,
        time_col=TIME_COL,
        vehicle_col=VEHICLE_COL,
        target_col=TARGET_COL,
    )

    print("VALIDATION feature matrix shape:", val_features.shape)
    val_out_path = "validation_vehicle_features.csv"
    val_features.to_csv(val_out_path, index=False)
    print(f"Saved VALIDATION features to: {val_out_path}")

    # -------------------------
    # 4) Build TEST features
    # -------------------------
    print("\nBuilding TEST features...")
    test_oper = read_csv_from_adls(fs, RAW_PATHS["test"]["oper"])
    test_tte = read_csv_from_adls(fs, RAW_PATHS["test"]["tte"])
    test_spec = read_csv_from_adls(fs, RAW_PATHS["test"]["spec"])

    test_features = build_eval_features(
        df_oper=test_oper,
        df_tte=test_tte,
        df_spec=test_spec,
        counter_cols=counter_cols_present,
        histogram_groups=histogram_groups,
        spec_feature_cols=spec_feature_cols,
        feature_columns=feature_columns,
        time_col=TIME_COL,
        vehicle_col=VEHICLE_COL,
        target_col=TARGET_COL,
    )

    print("TEST feature matrix shape:", test_features.shape)
    test_out_path = "test_vehicle_features.csv"
    test_features.to_csv(test_out_path, index=False)
    print(f"Saved TEST features to: {test_out_path}")

    print("\n✅ Done. Feature matrices built for train, validation, and test.")


if __name__ == "__main__":
    main()
