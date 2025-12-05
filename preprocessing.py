"""
Preprocessing utilities for machine-generated code detection.

This module provides:
- Whitespace normalization
- Language-aware comment removal (for ablations)
- A single preprocess_code() function that you can call from datasets/models
"""

from __future__ import annotations

import re
from typing import Literal


LanguageType = Literal["python", "cpp", "c", "java", "js", "csharp", "go", "php", "other"]


def normalize_whitespace(code: str) -> str:
    """
    Normalize whitespace while preserving indentation structure.

    - Converts Windows/Mac newlines to '\n'
    - Strips trailing spaces/tabs at line end
    - Keeps leading spaces (indentation)
    - Removes leading/trailing empty lines
    """
    if not isinstance(code, str):
        code = str(code)

    # Normalize newlines
    code = code.replace("\r\n", "\n").replace("\r", "\n")

    lines = code.split("\n")
    cleaned_lines = []

    for line in lines:
        # Remove trailing whitespace but keep indentation
        cleaned = re.sub(r"[ \t]+$", "", line)
        cleaned_lines.append(cleaned)

    # Remove leading/trailing completely empty lines
    while cleaned_lines and cleaned_lines[0].strip() == "":
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1].strip() == "":
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)


def _remove_python_comments(code: str) -> str:
    """
    Remove single-line (# ...) comments from Python code.
    We do NOT remove docstrings here (that would require AST manipulation).
    """
    lines = code.split("\n")
    new_lines = []
    for line in lines:
        # Very simple heuristic: everything after '#' is comment
        # BUT ignore '#' inside strings (rough heuristic with quotes).
        # For this project, a simple split is usually fine.
        if "#" in line:
            # Try to split only if # is not inside quotes (basic heuristic)
            quote_chars = ("'", '"')
            hash_idx = line.find("#")
            before = line[:hash_idx]
            num_quotes = sum(before.count(q) for q in quote_chars)
            if num_quotes % 2 == 0:
                # '#' is likely starting a comment
                line = before.rstrip()
        new_lines.append(line)
    return "\n".join(new_lines)


def _remove_c_style_comments(code: str) -> str:
    """
    Remove // and /* */ comments used in C/C++/Java/JS/C#/Go-like languages.
    """
    # Remove /* ... */
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    # Remove // ... 
    lines = code.split("\n")
    new_lines = []
    for line in lines:
        if "//" in line:
            idx = line.find("//")
            line = line[:idx].rstrip()
        new_lines.append(line)
    return "\n".join(new_lines)


def remove_comments(code: str, language: str) -> str:
    """
    Remove comments from code depending on language.

    Args:
        code: raw code string
        language: language name from dataset, e.g., 'python', 'cpp', 'c', 'java', 'js', etc.

    Returns:
        Code string with comments removed (best-effort).
    """
    lang = (language or "").lower()

    if "python" in lang or lang == "py":
        return _remove_python_comments(code)

    if lang in {"c", "cpp", "c++", "java", "js", "javascript", "c#", "csharp", "go", "php"}:
        return _remove_c_style_comments(code)

    # Default: return code unchanged if we don't recognize the language
    return code


def preprocess_code(
    code: str,
    language: str | None = None,
    *,
    normalize_ws: bool = True,
    strip_comments: bool = False,
) -> str:
    """
    End-to-end preprocessing for a single code snippet.

    Args:
        code: raw code string
        language: programming language string from dataset (may be None)
        normalize_ws: whether to normalize whitespace
        strip_comments: whether to remove comments (for certain experiments / ablations)

    Returns:
        Preprocessed code string.
    """
    if not isinstance(code, str):
        code = str(code)

    if strip_comments:
        code = remove_comments(code, language or "")

    if normalize_ws:
        code = normalize_whitespace(code)

    return code
