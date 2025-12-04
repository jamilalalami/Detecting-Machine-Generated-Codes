"""
Data loading and conversion utilities for SemEval-2026 Task 13.

This module:
-Loads raw data for Subtasks A, B, and C from data/raw/task_*/ folders
-Handles Parquet files from Kaggle
-Saves train / val / test_sample splits as CSV into data/processed/

Resulting files:
  data/processed/
    subtaskA_train.csv
    subtaskA_val.csv
    subtaskA_test_sample.csv

    subtaskB_train.csv
    subtaskB_val.csv
    subtaskB_test_sample.csv

    subtaskC_train.csv
    subtaskC_val.csv
    subtaskC_test_sample.csv
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import Config


def _load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame, with a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    return pd.read_parquet(path)


def _ensure_processed_dir():
    """Ensure the processed data directory exists."""
    Config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _convert_splits_to_csv(
    subtask: str,
    train_path: Path,
    val_path: Path,
    test_path: Optional[Path],
    output_prefix: Path,
    label_col: str = "label",
):
    """
    Convert train/val/test_sample Parquet files for a given subtask into CSVs
    with consistent naming.

    Args:
        subtask: 'A', 'B', or 'C'
        train_path: path to train parquet
        val_path: path to validation parquet
        test_path: path to test_sample parquet (may or may not include labels)
        output_prefix: base prefix for output CSV names
        label_col: column name for labels (default: 'label')
    """
    print(f"\nPreparing data for Subtask {subtask}")

    #   Train  
    print(f"Loading TRAIN from: {train_path}")
    train_df = _load_parquet(train_path)

    # Basic checks
    expected_cols = {"code", label_col}
    missing = expected_cols.difference(train_df.columns)
    if missing:
        raise KeyError(
            f"Train file {train_path} is missing expected columns {missing}"
        )

    #   Validation  
    print(f"Loading VAL from: {val_path}")
    val_df = _load_parquet(val_path)

    missing_val = expected_cols.difference(val_df.columns)
    if missing_val:
        raise KeyError(
            f"Validation file {val_path} is missing expected columns {missing_val}"
        )

    #   Test sample
    print(f"Loading TEST SAMPLE from: {test_path}")
    test_df = _load_parquet(test_path)

    if "code" not in test_df.columns:
        raise KeyError(f"Test sample file {test_path} must contain a 'code' column.")

    has_test_labels = label_col in test_df.columns
    if has_test_labels:
        print(f"Test sample has labels column '{label_col}'.")
    else:
        print(f"Test sample has NO '{label_col}' column (labels not required).")

    _ensure_processed_dir()

    # Output paths
    train_csv = output_prefix.with_name(output_prefix.name + "_train.csv")
    val_csv = output_prefix.with_name(output_prefix.name + "_val.csv")
    test_csv = output_prefix.with_name(output_prefix.name + "_test_sample.csv")

    print(f"Saving TRAIN ({len(train_df)} rows) to: {train_csv}")
    train_df.to_csv(train_csv, index=False)

    print(f"Saving VAL ({len(val_df)} rows) to: {val_csv}")
    val_df.to_csv(val_csv, index=False)

    print(f"Saving TEST SAMPLE ({len(test_df)} rows) to: {test_csv}")
    test_df.to_csv(test_csv, index=False)

    print(f"Subtask {subtask} conversion complete.")


def prepare_all_subtasks():
    """
    Prepare data for all three subtasks (A, B, C).
    Converts Parquet splits to CSV in data/processed/.
    """

    # Subtask A
    _convert_splits_to_csv(
        subtask="A",
        train_path=Config.RAW_A_TRAIN,
        val_path=Config.RAW_A_VAL,
        test_path=Config.RAW_A_TEST,
        output_prefix=Config.PROCESSED_SUBTASK_A_PREFIX,
        label_col="label",
    )

    # Subtask B
    _convert_splits_to_csv(
        subtask="B",
        train_path=Config.RAW_B_TRAIN,
        val_path=Config.RAW_B_VAL,
        test_path=Config.RAW_B_TEST,
        output_prefix=Config.PROCESSED_SUBTASK_B_PREFIX,
        label_col="label",
    )

    # Subtask C
    _convert_splits_to_csv(
        subtask="C",
        train_path=Config.RAW_C_TRAIN,
        val_path=Config.RAW_C_VAL,
        test_path=Config.RAW_C_TEST,
        output_prefix=Config.PROCESSED_SUBTASK_C_PREFIX,
        label_col="label",
    )


if __name__ == "__main__":
    prepare_all_subtasks()