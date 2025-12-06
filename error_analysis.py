"""
Error analysis for SemEval Task 13 code detectors.

We analyse the *single-task* GraphCodeBERT models for all three subtasks (A, B, C):
- Subtask A: transformer_subtaskA_best.pt (2-class)
- Subtask B: transformer_subtaskB_best.pt (11-class)
- Subtask C: transformer_subtaskC_best.pt (4-class)

For each subtask we:
  - Evaluate on the validation set
  - Compute macro F1 and a classification report
  - Save a CSV of misclassified examples for manual inspection
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from .config import Config
from .datasets import CodeDataset


class TransformerCodeClassifier(nn.Module):
    """
    Simple single-task transformer classifier used for error analysis.

    Must match the architecture used in train_transformer_single_task.py so that
    the saved state_dict can be loaded correctly.
    """

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_repr = self.dropout(cls_repr)
        logits = self.classifier(cls_repr)
        return logits


def _val_paths(subtask: str) -> Path:
    """Return validation CSV path for a given subtask."""
    if subtask == "A":
        return Config.PROCESSED_DIR / "subtaskA_val.csv"
    elif subtask == "B":
        return Config.PROCESSED_DIR / "subtaskB_val.csv"
    elif subtask == "C":
        return Config.PROCESSED_DIR / "subtaskC_val.csv"
    else:
        raise ValueError(f"Unknown subtask: {subtask}")


def _model_path(subtask: str) -> Tuple[int, Path]:
    """
    Return (num_labels, checkpoint_path) for the single-task transformer
    of the given subtask.
    """
    if subtask == "A":
        return 2, Config.MODELS_DIR / "transformer_subtaskA_best.pt"
    elif subtask == "B":
        return 11, Config.MODELS_DIR / "transformer_subtaskB_best.pt"
    elif subtask == "C":
        return 4, Config.MODELS_DIR / "transformer_subtaskC_best.pt"
    else:
        raise ValueError(f"Unknown subtask: {subtask}")


def run_error_analysis(subtask: str, max_examples_to_save: int = 500) -> None:
    print("\n" + "=" * 80)
    print(f"Error analysis for Subtask {subtask}")
    print("=" * 80)

    val_path = _val_paths(subtask)
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found for subtask {subtask}: {val_path}")

    df = pd.read_csv(val_path)
    print(f"[Subtask {subtask}] Loaded VAL: {val_path}  (rows={len(df)})")

    # Basic sanity: expect 'code' and 'label'
    if "code" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected columns 'code' and 'label' in {val_path}, got: {df.columns}")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    dataset = CodeDataset(
        df,
        tokenizer=tokenizer,
        max_length=Config.MAX_LENGTH,
        use_preprocessing=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    num_labels, ckpt_path = _model_path(subtask)
    print(f"[Subtask {subtask}] Loading single-task transformer from: {ckpt_path}")

    model = TransformerCodeClassifier(
        model_name=Config.MODEL_NAME,
        num_labels=num_labels,
        dropout=0.1,
    )
    state = torch.load(ckpt_path, map_location=Config.DEVICE)
    model.load_state_dict(state)
    model.to(Config.DEVICE)
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            labels = batch["labels"].to(Config.DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"[Subtask {subtask}] Single-task transformer Macro F1 (val): {macro_f1:.4f}")

    report = classification_report(y_true, y_pred, digits=4)
    print(f"\nClassification report (Subtask {subtask}, single-task transformer):\n{report}")

    # Build a DataFrame of predictions vs labels for misclassified examples
    # If the VAL CSV has an 'id' column, keep it; otherwise use index
    if "id" in df.columns:
        ids = df["id"].tolist()
    else:
        ids = list(df.index)

    error_df = pd.DataFrame(
        {
            "id": ids,
            "code": df["code"],
            "true_label": y_true,
            "pred_label": y_pred,
        }
    )
    error_df["correct"] = error_df["true_label"] == error_df["pred_label"]

    misclassified = error_df[~error_df["correct"]].copy()
    misclassified = misclassified.head(max_examples_to_save)

    out_dir = Config.REPORTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / f"error_analysis_subtask{subtask}_single_transformer_misclassified.csv"
    misclassified.to_csv(out_path, index=False)
    print(f"[Subtask {subtask}] Saved {len(misclassified)} misclassified examples to: {out_path}")


def main():
    torch.set_grad_enabled(False)
    for subtask in ["A", "B", "C"]:
        run_error_analysis(subtask)


if __name__ == "__main__":
    main()
