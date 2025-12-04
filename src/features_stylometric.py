"""
Stylometric feature extraction for code snippets.

These features are designed to capture:
- Formatting style (indentation, line lengths)
- Comment usage
- Identifier characteristics
- Rough complexity/structure (AST depth for Python)

They are language-agnostic where possible, and safe to compute
for large datasets.
"""

from __future__ import annotations

import re
import ast
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .preprocessing import normalize_whitespace


#   Helper functions for individual features  

def _split_lines(code: str) -> List[str]:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    return code.split("\n")


def num_lines(code: str) -> int:
    lines = _split_lines(code)
    # count non-empty or all? we use all, to reflect formatting
    return len(lines)


def line_length_stats(code: str) -> Dict[str, float]:
    lines = _split_lines(code)
    if not lines:
        return {"avg_line_len": 0.0, "max_line_len": 0.0}

    lengths = [len(l) for l in lines]
    return {
        "avg_line_len": float(np.mean(lengths)),
        "max_line_len": float(np.max(lengths)),
    }


def comment_stats(code: str, language: str | None = None) -> Dict[str, float]:
    """
    Rough comment statistics based on heuristics:
    '#' for Python
    '//' and '/* */' for C-like languages
    """
    lang = (language or "").lower()
    lines = _split_lines(code)
    if not lines:
        return {"num_comments": 0.0, "comment_density": 0.0}

    num_comments = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "python" in lang or lang == "py":
            if "#" in stripped:
                num_comments += 1
        else:
            # C-like
            if stripped.startswith("//") or "/*" in stripped or "*/" in stripped:
                num_comments += 1
            elif "//" in stripped:
                num_comments += 1

    density = num_comments / len(lines)
    return {"num_comments": float(num_comments), "comment_density": float(density)}


def indent_stats(code: str) -> Dict[str, float]:
    """
    Measure average indentation depth in spaces.
    We treat leading spaces as indentation; tabs are approximated as 4 spaces.
    """
    lines = _split_lines(code)
    if not lines:
        return {"avg_indent": 0.0, "max_indent": 0.0}

    indents = []
    for line in lines:
        # Replace tabs with 4 spaces for counting
        line = line.replace("\t", "    ")
        leading_spaces = len(line) - len(line.lstrip(" "))
        indents.append(leading_spaces)

    return {
        "avg_indent": float(np.mean(indents)),
        "max_indent": float(np.max(indents)),
    }


def identifier_avg_length(code: str) -> float:
    """
    Average length of identifiers (variable/function names) based on a simple regex.
    """
    # Simple identifier regex: letters, digits, underscore, starting with letter/_.
    identifiers = re.findall(r"\b[_A-Za-z][_A-Za-z0-9]*\b", code)
    if not identifiers:
        return 0.0
    lengths = [len(tok) for tok in identifiers]
    return float(np.mean(lengths))


def keyword_density(code: str) -> float:
    """
    Density of common programming keywords across many languages.
    This is a crude but useful stylistic feature.
    """
    common_keywords = {
        "if", "else", "for", "while", "return", "class", "def", "function",
        "public", "private", "static", "void", "int", "float", "double",
        "string", "var", "let", "const", "try", "catch", "finally",
        "switch", "case", "break", "continue", "import", "from", "using",
    }

    tokens = re.findall(r"\b[_A-Za-z][_A-Za-z0-9]*\b", code.lower())
    if not tokens:
        return 0.0

    kw_count = sum(1 for t in tokens if t in common_keywords)
    return float(kw_count / len(tokens))


def python_ast_depth(code: str, language: str | None = None) -> float:
    """
    Approximate AST depth for Python code only.
    For non-Python languages, returns 0.0.

    AST depth is a rough proxy for structural complexity.
    """
    lang = (language or "").lower()
    if "python" not in lang and lang != "py":
        return 0.0

    try:
        tree = ast.parse(code)
    except Exception:
        return 0.0

    max_depth = 0

    def _visit(node, depth):
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            _visit(child, depth + 1)

    _visit(tree, 1)
    return float(max_depth)


#   High-level stylo feature extractor  

STYLE_FEATURE_NAMES = [
    "num_lines",
    "avg_line_len",
    "max_line_len",
    "num_comments",
    "comment_density",
    "avg_indent",
    "max_indent",
    "identifier_avg_len",
    "keyword_density",
    "python_ast_depth",
]


def extract_style_features(code: str, language: str | None = None) -> Dict[str, float]:
    """
    Compute a consistent set of stylometric features for a single snippet.

    Returns:
        dict with keys matching STYLE_FEATURE_NAMES.
    """
    # Normalize whitespace lightly for more stable statistics
    code_norm = normalize_whitespace(code)

    n_lines = num_lines(code_norm)
    line_stats = line_length_stats(code_norm)
    cstats = comment_stats(code_norm, language=language)
    istats = indent_stats(code_norm)
    ident_len = identifier_avg_length(code_norm)
    kw_density = keyword_density(code_norm)
    ast_depth_val = python_ast_depth(code_norm, language=language)

    feats = {
        "num_lines": float(n_lines),
        "avg_line_len": line_stats["avg_line_len"],
        "max_line_len": line_stats["max_line_len"],
        "num_comments": cstats["num_comments"],
        "comment_density": cstats["comment_density"],
        "avg_indent": istats["avg_indent"],
        "max_indent": istats["max_indent"],
        "identifier_avg_len": float(ident_len),
        "keyword_density": float(kw_density),
        "python_ast_depth": float(ast_depth_val),
    }

    return feats


def extract_style_vector(code: str, language: str | None = None) -> np.ndarray:
    """
    Same as extract_style_features but returns a NumPy vector in a fixed order
    defined by STYLE_FEATURE_NAMES.
    """
    feats = extract_style_features(code, language)
    return np.array([feats[name] for name in STYLE_FEATURE_NAMES], dtype=np.float32)


#   Optional: batch extraction for a DataFrame  

def add_style_features_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with at least 'code' and optionally 'language' columns,
    return a new DataFrame with stylometric feature columns appended.

    This does NOT modify the original DataFrame in-place.
    """
    if "code" not in df.columns:
        raise KeyError("DataFrame must contain a 'code' column.")

    has_lang = "language" in df.columns

    feature_rows = []
    for _, row in df.iterrows():
        code = row["code"]
        lang = row["language"] if has_lang else None
        feats = extract_style_features(code, lang)
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows, columns=STYLE_FEATURE_NAMES)
    # Align indices
    feat_df.index = df.index

    return pd.concat([df, feat_df], axis=1)


def demo_on_sample_csv(csv_path: str, n: int = 5) -> None:
    """
    Small helper for manual sanity-check:
    - Loads csv_path
    - Computes stylometric features for first n rows
    - Prints them
    """
    df = pd.read_csv(csv_path).head(n)
    print(f"Loaded {len(df)} rows from {csv_path}")
    df_with_feats = add_style_features_to_df(df)
    print(df_with_feats[["code"] + STYLE_FEATURE_NAMES].head())


if __name__ == "__main__":
    from .config import Config

    sample_path = Config.PROCESSED_SUBTASK_A_PREFIX.with_name(
        Config.PROCESSED_SUBTASK_A_PREFIX.name + "_train.csv"
    )
    print(f"Running stylometric feature demo on: {sample_path}")
    demo_on_sample_csv(str(sample_path), n=5)
