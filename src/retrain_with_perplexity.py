"""
Retrains the classifier using REAL perplexity for every sample in the
dataset, not the placeholder zeros used during sandbox development.

This can only run on a machine with GPT-2 actually working (torch +
transformers installed, model downloaded) — see behavior-spec.md for
why this couldn't be done in the development sandbox.

Run with: python3 src/retrain_with_perplexity.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_dataset
from stylometric_features import split_sentences
from perplexity_features import passage_perplexity_features
from feature_vector import build_feature_vector
from classifier import train_classifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score


def build_training_data_with_real_perplexity(samples: list[dict]):
    """
    Same shape as classifier.build_training_data, but computes real
    perplexity for every sample instead of using placeholder zeros.
    This is slower (each sample requires GPT-2 inference) but only
    needs to run when the dataset changes.
    """
    X = []
    y = []
    for i, s in enumerate(samples):
        print(f"  Computing perplexity for sample {i+1}/{len(samples)}: {s['source']}...")
        sentences = split_sentences(s["text"])
        ppl_features = passage_perplexity_features(sentences)
        vec = build_feature_vector(s["text"], ppl_features)
        X.append(vec)
        y.append(1 if s["label"] == "ai" else 0)
    return np.array(X), np.array(y)


def evaluate_with_real_perplexity(X, y):
    if len(X) < 4:
        return {"note": "dataset too small", "n_samples": len(X)}

    loo = LeaveOneOut()
    predictions, actuals = [], []
    for train_idx, test_idx in loo.split(X):
        clf = train_classifier(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        predictions.append(pred[0])
        actuals.append(y[test_idx][0])

    return {
        "accuracy": round(accuracy_score(actuals, predictions), 3),
        "n_samples": len(X),
        "method": "leave-one-out, WITH real perplexity (not placeholder zeros)",
    }


if __name__ == "__main__":
    samples = load_dataset()
    print(f"Computing real perplexity for {len(samples)} samples — this takes a "
          f"few seconds per sample on CPU, please wait...\n")
    X, y = build_training_data_with_real_perplexity(samples)

    print("\nEvaluating with leave-one-out...")
    result = evaluate_with_real_perplexity(X, y)
    print("\nResult (WITH real perplexity):", result)

    # Save for comparison against the placeholder-zero result
    with open("documents/accuracy-report-with-perplexity.md", "w") as f:
        f.write("# Accuracy Report — WITH real perplexity\n\n")
        f.write(f"Dataset: {len(samples)} samples "
                f"({sum(1 for s in samples if s['label']=='human')} human, "
                f"{sum(1 for s in samples if s['label']=='ai')} AI)\n\n")
        f.write(f"Leave-one-out accuracy: {result.get('accuracy', 'N/A')}\n\n")
        f.write("This uses REAL GPT-2 perplexity for every training sample, "
                "unlike the earlier report which used placeholder zeros "
                "(perplexity wasn't available in the development sandbox). "
                "Compare against documents/accuracy-report.md to see the "
                "actual effect of enabling perplexity.\n")

    print("\nSaved comparison to documents/accuracy-report-with-perplexity.md")