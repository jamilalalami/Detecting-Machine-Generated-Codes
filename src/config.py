# src/config.py

import torch
from pathlib import Path


class Config:
    """
    Global configuration for the SemEval code detection project.
    Adjust RAW_* paths ONLY if your local folder/file names differ.
    """

    # Paths
    ROOT_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    MODELS_DIR = ROOT_DIR / "models"
    REPORTS_DIR = ROOT_DIR / "reports"

    #   Subtask A raw files  
    RAW_A_TRAIN = RAW_DIR / "task_a" / "task_a_training_set_1.parquet"
    RAW_A_VAL   = RAW_DIR / "task_a" / "task_a_validation_set.parquet"
    RAW_A_TEST  = RAW_DIR / "task_a" / "task_a_test_set_sample.parquet"

    #   Subtask B raw files  
    RAW_B_TRAIN = RAW_DIR / "task_b" / "task_b_training_set.parquet"
    RAW_B_VAL   = RAW_DIR / "task_b" / "task_b_validation_set.parquet"
    RAW_B_TEST  = RAW_DIR / "task_b" / "task_b_test_set_sample.parquet"

    #   Subtask C raw files  
    RAW_C_TRAIN = RAW_DIR / "task_c" / "task_c_training_set_1.parquet"
    RAW_C_VAL   = RAW_DIR / "task_c" / "task_c_validation_set.parquet"
    RAW_C_TEST  = RAW_DIR / "task_c" / "task_c_test_set_sample.parquet"

    # Processed prefixes
    PROCESSED_SUBTASK_A_PREFIX = PROCESSED_DIR / "subtaskA"
    PROCESSED_SUBTASK_B_PREFIX = PROCESSED_DIR / "subtaskB"
    PROCESSED_SUBTASK_C_PREFIX = PROCESSED_DIR / "subtaskC"

    # Model Hyperparams
    MODEL_NAME = "microsoft/graphcodebert-base"
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    LR = 2e-5
    EPOCHS = 2

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
