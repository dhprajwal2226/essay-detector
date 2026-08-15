"""
Checks whether the pipeline unfairly flags non-standard or informal
English as AI-written — a documented risk with real detectors (the
brief explicitly warns about this).

Uses leave-one-out: for each human sample, train on every OTHER
sample, then predict on the held-out one. This is the honest way to
test this — testing a sample the model was trained on tells us
nothing about how it treats genuinely new writing.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_dataset
from classifier import train_classifier, build_training_data
from feature_vector import build_feature_vector


def run_bias_check(samples: list[dict]) -> list[dict]:
    """
    For every human sample, retrain excluding it, then predict on it.
    Returns each sample's held-out P(AI) score, so we can see whether
    any particular kind of human writing gets scored suspiciously
    high despite genuinely being human.
    """
    X, y = build_training_data(samples)
    results = []

    for i, sample in enumerate(samples):
        if sample["label"] != "human":
            continue

        train_idx = [j for j in range(len(samples)) if j != i]
        if len(train_idx) < 2 or len(set(y[train_idx])) < 2:
            results.append({
                "source": sample["source"],
                "p_ai": None,
                "note": "not enough remaining data to train a held-out model",
            })
            continue

        clf = train_classifier(X[train_idx], y[train_idx])
        query_vec = np.array([build_feature_vector(sample["text"])])
        proba = clf.predict_proba(query_vec)[0]

        results.append({
            "source": sample["source"],
            "p_ai": round(proba[1], 3),
            "note": None,
        })

    return results


if __name__ == "__main__":
    samples = load_dataset()
    results = run_bias_check(samples)
    print("Bias check (leave-one-out, human samples only):")
    for r in results:
        flag = " <-- FLAGGED AS LIKELY AI, despite being genuine human writing" if (r["p_ai"] and r["p_ai"] > 0.5) else ""
        print(f"  {r['source']}: P(AI) = {r['p_ai']} {r['note'] or ''}{flag}")
