"""
Multi-task training for Subtasks A, B, and C with a shared transformer encoder.

- Uses CodeDataset (transformer-only).
- One shared encoder (GraphCodeBERT by default).
- Three heads:
    * Subtask A: 2 classes (binary)
    * Subtask B: 11 classes (authorship)
    * Subtask C: 4 classes (hybrid/adversarial)
- Joint training: in each epoch, we run through A, then B, then C.
- We track validation Macro F1 separately per task.
- The model is saved when Subtask C's Macro F1 improves (bonus-focused).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler

from .config import Config
from .datasets import CodeDataset
from .model_multitask import MultiTaskClassifier


def _get_split_paths(subtask_prefix: Path) -> Tuple[Path, Path]:
    """
    Given a prefix like data/processed/subtaskA, return
    TRAIN: data/processed/subtaskA_train.csv
    VAL:   data/processed/subtaskA_val.csv
    """
    train_path = subtask_prefix.with_name(subtask_prefix.name + "_train.csv")
    val_path = subtask_prefix.with_name(subtask_prefix.name + "_val.csv")
    return train_path, val_path


def _load_subtask_data(
    subtask: str,
    max_train_samples: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train/val DataFrames for subtask 'A', 'B', or 'C'.
    Optionally subsample training data for faster experimentation.
    """
    if subtask == "A":
        prefix = Config.PROCESSED_SUBTASK_A_PREFIX
    elif subtask == "B":
        prefix = Config.PROCESSED_SUBTASK_B_PREFIX
    elif subtask == "C":
        prefix = Config.PROCESSED_SUBTASK_C_PREFIX
    else:
        raise ValueError(f"Unknown subtask: {subtask}")

    train_path, val_path = _get_split_paths(prefix)

    print(f"[Subtask {subtask}] Loading TRAIN from: {train_path}")
    train_df = pd.read_csv(train_path)

    print(f"[Subtask {subtask}] Loading VAL from:   {val_path}")
    val_df = pd.read_csv(val_path)

    if max_train_samples is not None and len(train_df) > max_train_samples:
        train_df = train_df.sample(n=max_train_samples, random_state=42).reset_index(drop=True)
        print(f"[Subtask {subtask}] Subsampled training set to {len(train_df)} rows")

    # minimal sanity check
    for name, df in [("train", train_df), ("val", val_df)]:
        if "code" not in df.columns or "label" not in df.columns:
            raise KeyError(f"{name} DataFrame for Subtask {subtask} must contain 'code' and 'label' columns.")

    return train_df, val_df


def _build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders (train, val) for a given subtask using CodeDataset.
    """
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
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader


def _evaluate_subtask(
    model: MultiTaskClassifier,
    task: str,
    val_loader: DataLoader,
    device: str,
) -> Tuple[float, str]:
    """
    Evaluate the multi-task model on a single subtask.

    Returns:
        macro_f1, classification_report_str
    """
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                task=task,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)

    macro_f1 = f1_score(labels_arr, preds_arr, average="macro")
    report = classification_report(labels_arr, preds_arr, digits=4)
    return macro_f1, report


def train_multitask(
    model_name: str | None = None,
    epochs: int | None = None,
    max_train_samples_A: int | None = None,
    max_train_samples_B: int | None = None,
    max_train_samples_C: int | None = None,
) -> None:
    """
    Jointly train a multi-task model for Subtasks A, B, and C.

    Args:
        model_name: HF model name (default: Config.MODEL_NAME)
        epochs: number of epochs (default: Config.EPOCHS)
        max_train_samples_A/B/C: optional caps for training data per subtask
    """
    device = Config.DEVICE
    model_name = model_name or Config.MODEL_NAME
    epochs = epochs or Config.EPOCHS

    print("\n=== Multi-task training for Subtasks A, B, C ===")
    print(f"Using base model: {model_name}")
    print(f"Device: {device}")

    #   Load data for each subtask  
    train_A, val_A = _load_subtask_data("A", max_train_samples=max_train_samples_A)
    train_B, val_B = _load_subtask_data("B", max_train_samples=max_train_samples_B)
    train_C, val_C = _load_subtask_data("C", max_train_samples=max_train_samples_C)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_loader_A, val_loader_A = _build_dataloaders(train_A, val_A, tokenizer)
    train_loader_B, val_loader_B = _build_dataloaders(train_B, val_B, tokenizer)
    train_loader_C, val_loader_C = _build_dataloaders(train_C, val_C, tokenizer)

    #   Model, optimizer, scheduler  
    model = MultiTaskClassifier(model_name=model_name).to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LR)

    steps_per_epoch = (
        len(train_loader_A)
        + len(train_loader_B)
        + len(train_loader_C)
    )
    total_steps = steps_per_epoch * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device == "cuda"))

    best_macro_f1_C = 0.0  # we will save model according to Subtask C performance
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = Config.MODELS_DIR / "multitask_ABC_best.pt"

    #   Training loop  
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_end = torch.cuda.Event(enable_timing=True)

        if device == "cuda":
            torch.cuda.synchronize()
            epoch_start.record()

        # Train sequentially on A, then B, then C in this epoch
        for task_name, train_loader in [
            ("A", train_loader_A),
            ("B", train_loader_B),
            ("C", train_loader_C),
        ]:
            task_iter = tqdm(
                train_loader,
                desc=f"Task {task_name} | Epoch {epoch}/{epochs} [multitask train]",
                leave=False,
            )
            for batch in task_iter:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with autocast(enabled=(device == "cuda")):
                    logits = model(
                        task=task_name,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    loss = loss_fn(logits, labels)

                # guard against occasional NaNs (very rare)
                if torch.isnan(loss):
                    print(f"[Warning] NaN loss encountered on task {task_name}, skipping batch.")
                    scaler.update()
                    continue

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss_sum += loss.item()
                epoch_steps += 1
                task_iter.set_postfix({"loss": f"{loss.item():.4f}"})

        if device == "cuda":
            epoch_end.record()
            torch.cuda.synchronize()
            epoch_time_sec = epoch_start.elapsed_time(epoch_end) / 1000.0
        else:
            epoch_time_sec = 0.0

        avg_epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
        print(
            f"Average training loss (all tasks): {avg_epoch_loss:.4f}  |  "
            f"Epoch time: {epoch_time_sec/60:.2f} min"
        )

        #   Validation per subtask  
        macro_A, report_A = _evaluate_subtask(model, "A", val_loader_A, device)
        macro_B, report_B = _evaluate_subtask(model, "B", val_loader_B, device)
        macro_C, report_C = _evaluate_subtask(model, "C", val_loader_C, device)

        print(f"[Multitask] Val Macro F1 - Subtask A: {macro_A:.4f}")
        print(f"[Multitask] Val Macro F1 - Subtask B: {macro_B:.4f}")
        print(f"[Multitask] Val Macro F1 - Subtask C: {macro_C:.4f}")

        # Save according to Subtask C (bonus/novelty)
        if macro_C > best_macro_f1_C:
            best_macro_f1_C = macro_C
            torch.save(model.state_dict(), best_model_path)
            print(f"New best multi-task model saved to: {best_model_path}")

        print("\nValidation classification report (Subtask A, multitask):")
        print(report_A)

        print("\nValidation classification report (Subtask B, multitask):")
        print(report_B)

        print("\nValidation classification report (Subtask C, multitask):")
        print(report_C)

    print(
        f"\nMulti-task training finished. "
        f"Best Subtask C Val Macro F1: {best_macro_f1_C:.4f}"
    )
    print(f"Best multi-task model stored at: {best_model_path}")


def main():
    """
    Entry point for multi-task training.

    NOTE:
    - Full-data training is heavy (same order as training 3 separate transformers).
    - For faster experiments, set max_train_samples_* below.
    """

    train_multitask(
        model_name=Config.MODEL_NAME,
        epochs=Config.EPOCHS,
        max_train_samples_A=None,  
        max_train_samples_B=None,
        max_train_samples_C=None,
    )


if __name__ == "__main__":
    main()
