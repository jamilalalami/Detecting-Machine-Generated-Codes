"""
Baseline models for code classification using TF-IDF + linear classifiers.

We use character n-grams as features because they:
- Work well across multiple programming languages
- Capture formatting and token patterns
- Are simple and fast to train

Models:
- Logistic Regression
- Linear SVM (LinearSVC)
"""

from __future__ import annotations

from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


def create_baseline_model(
    model_type: str = "svm",
    *,
    analyzer: str = "char",
    ngram_range: Tuple[int, int] = (3, 5),
    max_features: int = 100_000,
) -> Pipeline:
    """
    Create a TF-IDF + linear classifier pipeline.

    Args:
        model_type: 'svm' or 'logreg'
        analyzer: 'char' or 'word'
        ngram_range: n-gram range for TF-IDF
        max_features: maximum number of features for TF-IDF

    Returns:
        sklearn Pipeline
    """
    if model_type == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            verbose=0,
        )
    elif model_type == "svm":
        clf = LinearSVC()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    tfidf = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
    )

    pipe = Pipeline([
        ("tfidf", tfidf),
        ("clf", clf),
    ])

    return pipe
