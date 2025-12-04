"""
Dataset classes for transformer-based models.

- CodeDataset: for transformer-only models (Chunk 5)
- HybridCodeDataset: for hybrid models (transformer + char-CNN + stylometric)
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .preprocessing import preprocess_code
from .features_stylometric import extract_style_vector, STYLE_FEATURE_NAMES


class CodeDataset(Dataset):
    """
    PyTorch Dataset for code classification (transformer only).

    Expects a DataFrame with at least:
    - 'code'
    - 'label'
    and optionally:
    - 'language' (for preprocessing)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        *,
        use_preprocessing: bool = True,
        strip_comments: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_preprocessing = use_preprocessing
        self.strip_comments = strip_comments

        self.has_language = "language" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        code = row["code"]
        lang: Optional[str] = row["language"] if self.has_language else None

        if self.use_preprocessing:
            code = preprocess_code(
                code,
                language=lang,
                normalize_ws=True,
                strip_comments=self.strip_comments,
            )

        enc = self.tokenizer(
            code,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }
        return item


class HybridCodeDataset(Dataset):
    """
    Dataset for the hybrid model:
    - Transformer tokens
    - Char-level sequence (for Char-CNN)
    - Stylometric feature vector

    It precomputes:
    - Preprocessed code (normalized whitespace)
    - Stylometric features via extract_style_vector()
    - Char-level encoded sequences

    This is more expensive to build once, but faster per batch during training.
    """

    # Allowed characters for char-level encoding
    _ALLOWED_CHARS = (
        "\n\t "  # whitespace
        + "abcdefghijklmnopqrstuvwxyz"
        + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        + "0123456789"
        + "[]{}()<>;:.,+-*/=%!&|^~#'\"\\_"
    )

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        *,
        char_max_length: int = 512,
        use_preprocessing: bool = True,
        strip_comments: bool = False,
    ) -> None:
        """
        Args:
            df: DataFrame with at least 'code' and 'label'; optionally 'language'
            tokenizer: HuggingFace tokenizer for the transformer
            max_length: max token length for transformer
            char_max_length: max length for char-level sequence
            use_preprocessing: apply preprocess_code()
            strip_comments: whether to remove comments (for ablation-type runs)
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.char_max_length = char_max_length
        self.use_preprocessing = use_preprocessing
        self.strip_comments = strip_comments

        self.has_language = "language" in self.df.columns

        #   Build char vocabulary  
        self.pad_idx = 0
        # known chars: 1..N
        self.char2idx = {ch: i + 1 for i, ch in enumerate(self._ALLOWED_CHARS)}
        # unknown chars get a dedicated index
        self.unk_idx = len(self.char2idx) + 1
        self.vocab_size = self.unk_idx + 1  # +1 for padding index 0

        # Expose for training scripts / models
        self.char_pad_idx = self.pad_idx
        self.char_vocab_size = self.vocab_size

        #   Precompute processed code, style features, and char ids  
        codes_processed: List[str] = []
        style_list: List[np.ndarray] = []
        char_list: List[np.ndarray] = []

        codes = self.df["code"].astype(str).tolist()
        if self.has_language:
            languages = self.df["language"].astype(str).tolist()
        else:
            languages = [None] * len(codes)

        for code, lang in zip(codes, languages):
            if self.use_preprocessing:
                pre_code = preprocess_code(
                    code,
                    language=lang,
                    normalize_ws=True,
                    strip_comments=self.strip_comments,
                )
            else:
                pre_code = str(code)

            codes_processed.append(pre_code)

            # Stylometric features
            style_vec = extract_style_vector(pre_code, language=lang)
            style_list.append(style_vec)

            # Char-level encoding
            char_ids = self._encode_chars(pre_code)
            char_list.append(char_ids)

        self.codes_processed = codes_processed
        self.style_feats = np.stack(style_list)  # [N, style_dim]
        self.char_ids = np.stack(char_list)      # [N, char_max_length]
        self.style_dim = self.style_feats.shape[1]

    def _encode_chars(self, text: str) -> np.ndarray:
        """
        Encode a string into a fixed-length sequence of character indices.
        """
        arr = np.zeros(self.char_max_length, dtype=np.int64)

        # Truncate to max length
        text = text[: self.char_max_length]

        for i, ch in enumerate(text):
            idx = self.char2idx.get(ch, self.unk_idx)
            arr[i] = idx

        return arr

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        code = self.codes_processed[idx]
        label = int(self.df.iloc[idx]["label"])

        enc = self.tokenizer(
            code,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        char_ids = torch.tensor(self.char_ids[idx], dtype=torch.long)
        style_feats = torch.tensor(self.style_feats[idx], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "char_ids": char_ids,
            "style_feats": style_feats,
            "labels": torch.tensor(label, dtype=torch.long),
        }
