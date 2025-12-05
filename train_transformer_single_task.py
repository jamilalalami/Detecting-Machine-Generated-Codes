"""
Train transformer-based classifiers separately for Subtasks A, B, and C.

Uses processed CSVs from data/processed/
Uses CodeDataset from datasets.py
Uses TransformerClassifier from model_transformer.py
Evaluates on validation split using macro F1
Saves best model (by val Macro F1) to models/ as .pt files
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

import time
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from .config import Config
from .datasets import CodeDataset
from .model_transformer import TransformerClassifier


SubtaskType = Literal["A", "B", "C"]


def _get_split_paths(subtask_prefix: Path) -> Tuple[Path, Path]:
    train_path = subtask_prefix.with_name(subtask_prefix.name + "_train.csv")
    val_path = subtask_prefix.with_name(subtask_prefix.name + "_val.csv")
    return train_path, val_path


def _get_num_labels_for_subtask(subtask: SubtaskType) -> int:
    if subtask == "A":
        return 2
    elif subtask == "B":
        return 11
    elif subtask == "C":
        return 4
    else:
        raise ValueError(f"Unknown subtask: {subtask}")


def train_single_subtask(
    subtask: SubtaskType,
    *,
    model_name: str | None = None,
    epochs: int | None = None,
    max_train_samples: int | None = None,
) -> None:
    """
    Train a single-task transformer classifier for a given subtask.

    Args:
        subtask: 'A', 'B', or 'C'
        model_name: HuggingFace model name (default: Config.MODEL_NAME)
        epochs: number of epochs (default: Config.EPOCHS)
        max_train_samples: if set, subsample the training set to this many rows
                           (useful for faster experimentation)
    """
    device = Config.DEVICE
    model_name = model_name or Config.MODEL_NAME
    epochs = epochs or Config.EPOCHS

    print(f"\n=== Transformer training for Subtask {subtask} ===")
    print(f"Using base model: {model_name}")
    print(f"Device: {device}")

    # Determine data paths
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

    # Optional subsampling for faster initial runs
    if max_train_samples is not None and len(train_df) > max_train_samples:
        train_df = train_df.sample(n=max_train_samples, random_state=42).reset_index(drop=True)
        print(f"Subsampled training set to {len(train_df)} rows")

    # Sanity check
    for df_name, df in [("train", train_df), ("val", val_df)]:
        if "code" not in df.columns or "label" not in df.columns:
            raise KeyError(f"{df_name} DataFrame must contain 'code' and 'label' columns.")

    num_labels = _get_num_labels_for_subtask(subtask)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = CodeDataset(
        train_df,
        tokenizer=tokenizer,
        max_length=Config.MAX_LENGTH,
        use_preprocessing=True,
        strip_comments=False,
    )
    val_dataset = CodeDataset(
        val_df,
        tokenizer=tokenizer,
        max_length=Config.MAX_LENGTH,
        use_preprocessing=True,
        strip_comments=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
    )

    model = TransformerClassifier(model_name=model_name, num_labels=num_labels).to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LR)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device == "cuda"))

    
    best_macro_f1 = 0.0
    best_model_path = Config.MODELS_DIR / f"transformer_subtask{subtask}_best.pt"
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        total_loss = 0.0

        epoch_start = time.time()

        train_iter = tqdm(
            train_loader,
            desc=f"Subtask {subtask}, Epoch {epoch}/{epochs} [train]",
            leave=False,
        )

        for batch in train_iter:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=(device == "cuda")):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)

            # Mixed precision backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            train_iter.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_time = time.time() - epoch_start
        avg_train_loss = total_loss / len(train_loader)
        print(
            f"Average training loss: {avg_train_loss:.4f}  |  "
            f"Epoch time: {epoch_time/60:.2f} min"
        )



        #   Validation  
        model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(logits, dim=1)

                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

        all_labels_arr = np.array(all_labels)
        all_preds_arr = np.array(all_preds)

        macro_f1 = f1_score(all_labels_arr, all_preds_arr, average="macro")
        print(f"[Subtask {subtask}] Val Macro F1: {macro_f1:.4f}")

        # Save best model
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to: {best_model_path}")

        print("\nValidation classification report:")
        print(classification_report(all_labels_arr, all_preds_arr, digits=4))

    print(f"\nTraining for Subtask {subtask} finished. Best Val Macro F1: {best_macro_f1:.4f}")
    print(f"Best model stored at: {best_model_path}")
    

def main():
    """
    Train transformer classifiers for all subtasks (A, B, C).

    NOTE: Full training on all data may be heavy depending on your GPU.
    You can optionally pass max_train_samples to train_single_subtask(...)
    for faster initial experiments.
    """

    # Here we call with full data by default:
    train_single_subtask("A")
    train_single_subtask("B")
    train_single_subtask("C")


if __name__ == "__main__":
    main()
