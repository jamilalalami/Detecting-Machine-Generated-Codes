"""
Train and evaluate baseline TF-IDF + linear models for Subtasks A, B, and C.

- Uses processed CSVs from data/processed/
- Trains one baseline per subtask
- Evaluates on validation split using macro F1
- Saves models into models/ as .joblib files
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.metrics import f1_score, classification_report

from .config import Config
from .model_baseline import create_baseline_model


SubtaskType = Literal["A", "B", "C"]


def _ensure_models_dir():
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _get_split_paths(subtask_prefix: Path):
    """
    Given a prefix like data/processed/subtaskA, build train/val paths.
    """
    train_path = subtask_prefix.with_name(subtask_prefix.name + "_train.csv")
    val_path = subtask_prefix.with_name(subtask_prefix.name + "_val.csv")
    return train_path, val_path


def train_and_eval_baseline_for_subtask(
    subtask: SubtaskType,
    model_type: str = "svm",
) -> None:
    """
    Train and evaluate a baseline model for a given subtask.

    Args:
        subtask: 'A', 'B', or 'C'
        model_type: 'svm' or 'logreg'
    """
    print(f"\nBaseline training for Subtask {subtask} ({model_type})")

    if subtask == "A":
        prefix = Config.PROCESSED_SUBTASK_A_PREFIX
    elif subtask == "B":
        prefix = Config.PROCESSED_SUBTASK_B_PREFIX
    elif subtask == "C":
        prefix = Config.PROCESSED_SUBTASK_C_PREFIX
    else:
        raise ValueError(f"Unknown subtask: {subtask}")

    train_path, val_path = _get_split_paths(prefix)

    print(f"Loading TRAIN from: {train_path}")
    train_df = pd.read_csv(train_path)

    print(f"Loading VAL from:   {val_path}")
    val_df = pd.read_csv(val_path)

    # Sanity checks
    for df_name, df in [("train", train_df), ("val", val_df)]:
        if "code" not in df.columns or "label" not in df.columns:
            raise KeyError(
                f"{df_name} DataFrame must contain 'code' and 'label' columns."
            )

    X_train, y_train = train_df["code"].astype(str), train_df["label"].astype(int)
    X_val, y_val = val_df["code"].astype(str), val_df["label"].astype(int)

    print(f"Training size: {len(X_train)} | Validation size: {len(X_val)}")

    model = create_baseline_model(
        model_type=model_type,
        analyzer="char",
        ngram_range=(3, 5),
        max_features=100_000,
    )

    print("Fitting baseline model:")
    model.fit(X_train, y_train)

    print("Predicting on validation set:")
    y_pred = model.predict(X_val)

    macro_f1 = f1_score(y_val, y_pred, average="macro")
    print(f"[Subtask {subtask}] Macro F1 (val): {macro_f1:.4f}")

    print("\nClassification report (val):")
    print(classification_report(y_val, y_pred, digits=4))

    # Save model
    _ensure_models_dir()
    model_path = Config.MODELS_DIR / f"baseline_subtask{subtask}_{model_type}.joblib"
    print(f"Saving trained model to: {model_path}")
    joblib.dump(model, model_path)

    print(f"Subtask {subtask} baseline training finished.\n")


def main():
    """
    Train baselines for Subtasks A, B, and C using Linear SVM.
    You can change 'svm' to 'logreg' if you want to compare them.
    """
    # SVM baseline
    for subtask in ["A", "B", "C"]:
        train_and_eval_baseline_for_subtask(subtask=subtask, model_type="svm")


if __name__ == "__main__":
    main()
