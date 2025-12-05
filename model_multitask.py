"""
Multi-task classifier for SemEval-style code detection:

Shared encoder (e.g., GraphCodeBERT)
Task-specific heads:
   Subtask A: binary (2 classes)
   Subtask B: 11-way authorship
   Subtask C: 4-way hybrid/adversarial

Forward usage:
    logits = model(task="A", input_ids=..., attention_mask=...)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class MultiTaskClassifier(nn.Module):
    """
    Multi-task classifier with a shared encoder and three task-specific heads.
    """

    def __init__(
        self,
        model_name: str,
        num_labels_A: int = 2,
        num_labels_B: int = 11,
        num_labels_C: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Shared encoder (e.g. microsoft/graphcodebert-base)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Task-specific linear heads
        self.head_A = nn.Linear(hidden_size, num_labels_A)
        self.head_B = nn.Linear(hidden_size, num_labels_B)
        self.head_C = nn.Linear(hidden_size, num_labels_C)

    def forward(
        self,
        task: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            task: "A", "B", or "C"
            input_ids: [B, L]
            attention_mask: [B, L]

        Returns:
            logits: [B, num_labels_task]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [B, H]
        cls_repr = self.dropout(cls_repr)

        if task == "A":
            return self.head_A(cls_repr)
        elif task == "B":
            return self.head_B(cls_repr)
        elif task == "C":
            return self.head_C(cls_repr)
        else:
            raise ValueError(f"Unknown task identifier: {task!r}")
