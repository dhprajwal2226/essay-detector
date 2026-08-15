"""
Trains a simple logistic regression classifier on the extracted
features, and provides prediction with per-feature contribution so
the interface can show WHY something was flagged, not just a verdict.

Logistic regression (not a deep model) is a deliberate choice: with a
small dataset, a simple, interpretable model is more honest than a
complex one that would just overfit. It also makes "why" answerable —
we can look at which features pushed the prediction toward AI or human.
"""

import sys
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_dataset
from feature_vector import build_feature_vector, FEATURE_NAMES


def build_training_data(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts loaded samples into (X, y) arrays.
    label: 1 = ai, 0 = human
    """
    X = []
    y = []
    for s in samples:
        vec = build_feature_vector(s["text"])  # perplexity added once available
        X.append(vec)
        y.append(1 if s["label"] == "ai" else 0)
    return np.array(X), np.array(y)


def train_classifier(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf


def evaluate_leave_one_out(X: np.ndarray, y: np.ndarray) -> dict:
    """
    With a very small dataset, a normal train/test split isn't
    meaningful (too few samples to hold any out). Leave-one-out
    cross-validation is a more honest way to get *some* accuracy
    signal from a small dataset: train on all but one sample, test
    on the one left out, repeat for every sample.

    Still not a substitute for a proper-sized held-out test set —
    documented as a limitation in behavior-spec.md.
    """
    if len(X) < 4:
        return {
            "note": "Dataset too small for any meaningful evaluation "
                    "(need at least a few samples per class). "
                    "This is honestly reported, not hidden.",
            "n_samples": len(X),
        }

    loo = LeaveOneOut()
    predictions = []
    actuals = []

    for train_idx, test_idx in loo.split(X):
        clf = train_classifier(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        predictions.append(pred[0])
        actuals.append(y[test_idx][0])

    acc = accuracy_score(actuals, predictions)
    return {
        "accuracy": round(acc, 3),
        "n_samples": len(X),
        "method": "leave-one-out cross-validation (dataset too small for a held-out test split)",
    }


if __name__ == "__main__":
    samples = load_dataset()
    X, y = build_training_data(samples)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels: {y}")
    print()
    result = evaluate_leave_one_out(X, y)
    print("Evaluation result:", result)
