"""
Module 4a — Delay Cause Classifier (implements SRS FR-07's cause
classification, using TF-IDF + Logistic Regression instead of DistilBERT)

Scope decision (documented, same convention as Modules 2 and 3):
    The SRS specifies a BERT-based classifier for `reported_text`. Given
    cause_category is a clean, labeled, 7-class problem with ~600-1000
    examples per class, a TF-IDF + Logistic Regression pipeline reaches
    comparable practical accuracy without needing a pretrained transformer
    download (Hugging Face) or PyTorch itself. This is both cheaper to run
    and faster to train (seconds vs minutes-to-hours on CPU), so it is the
    efficient choice, not just the free one. Swappable for a real
    fine-tuned DistilBERT later without changing how this module is used
    by the rest of the pipeline (predict_proba / predict interface stays
    the same either way).
"""

from __future__ import annotations

import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score


class CauseClassifier:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=5000,
                        ngram_range=(1, 2),
                        stop_words="english",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",  # cause categories aren't perfectly even (588-1061)
                    ),
                ),
            ]
        )
        self.classes_ = None

    def train(self, texts: pd.Series, labels: pd.Series, test_size: float = 0.2, random_state: int = 42):
        """Trains on reported_text -> cause_category, holds out a test set,
        and returns an evaluation report dict."""
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        self.pipeline.fit(X_train, y_train)
        self.classes_ = self.pipeline.named_steps["clf"].classes_

        preds = self.pipeline.predict(X_test)
        report = {
            "accuracy": accuracy_score(y_test, preds),
            "macro_f1": f1_score(y_test, preds, average="macro"),
            "classification_report": classification_report(y_test, preds),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        return report

    def predict(self, texts: pd.Series):
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: pd.Series):
        return self.pipeline.predict_proba(texts)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    @classmethod
    def load(cls, path: str) -> "CauseClassifier":
        obj = cls()
        with open(path, "rb") as f:
            obj.pipeline = pickle.load(f)
        obj.classes_ = obj.pipeline.named_steps["clf"].classes_
        return obj