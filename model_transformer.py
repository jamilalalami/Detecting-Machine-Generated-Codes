"""
Transformer-based classifier for code snippets.

We use a generic encoder (e.g., microsoft/graphcodebert-base) with
a simple classification head on top of the [CLS] token.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class TransformerClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            logits: [batch_size, num_labels]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation (position 0)
        pooled = outputs.last_hidden_state[:, 0, :]  # [B, H]
        logits = self.classifier(self.dropout(pooled))
        return logits
