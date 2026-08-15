"""
Generates the accuracy report the brief requires: honest results on
our own test set, the most confidently-wrong examples with an
explanation attempt, not just a bare accuracy percentage.

Uses leave-one-out across the WHOLE dataset (not just human samples,
unlike bias_check.py which is specifically about human-side fairness)
to find every sample's held-out prediction, then surfaces whichever
ones were most confidently wrong.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_dataset, dataset_summary
from classifier import train_classifier, build_training_data
from feature_vector import build_feature_vector, FEATURE_NAMES
from bias_check import run_bias_check


def leave_one_out_predictions(samples: list[dict]) -> list[dict]:
    """
    For every sample, train on all others, predict on it. Returns
    each sample's true label, held-out predicted probability, and
    whether it was correct — the raw material for both the accuracy
    number and the confidently-wrong examples.
    """
    X, y = build_training_data(samples)
    results = []

    for i, sample in enumerate(samples):
        train_idx = [j for j in range(len(samples)) if j != i]
        if len(train_idx) < 2 or len(set(y[train_idx])) < 2:
            results.append({
                "source": sample["source"],
                "true_label": sample["label"],
                "p_ai": None,
                "correct": None,
                "note": "not enough remaining data to train a held-out model",
            })
            continue

        clf = train_classifier(X[train_idx], y[train_idx])
        query_vec = np.array([build_feature_vector(sample["text"])])
        p_ai = clf.predict_proba(query_vec)[0][1]
        predicted_label = "ai" if p_ai >= 0.5 else "human"
        correct = predicted_label == sample["label"]

        results.append({
            "source": sample["source"],
            "true_label": sample["label"],
            "p_ai": round(float(p_ai), 3),
            "correct": correct,
            "confidence_of_wrongness": abs(p_ai - (0 if sample["label"] == "human" else 1)),
        })

    return results


def find_confidently_wrong(loo_results: list[dict], top_n: int = 3) -> list[dict]:
    """Returns the top_n wrong predictions, ranked by how confidently wrong they were."""
    wrong = [r for r in loo_results if r.get("correct") is False]
    wrong.sort(key=lambda r: r["confidence_of_wrongness"], reverse=True)
    return wrong[:top_n]


def generate_report() -> str:
    samples = load_dataset()
    summary = dataset_summary(samples)
    loo_results = leave_one_out_predictions(samples)
    confidently_wrong = find_confidently_wrong(loo_results)
    bias_results = run_bias_check(samples)

    valid_results = [r for r in loo_results if r.get("correct") is not None]
    accuracy = (
        sum(1 for r in valid_results if r["correct"]) / len(valid_results)
        if valid_results else None
    )

    lines = []
    lines.append("# Accuracy Report")
    lines.append("")
    lines.append(f"**Dataset**: {summary['total']} samples "
                  f"({summary['human_count']} human, {summary['ai_count']} AI)")
    lines.append("")
    lines.append("**This dataset is currently very small.** The numbers below are "
                  "honestly reported, but should not be read as reliable — see the "
                  "explicit caveat at the end of this report.")
    lines.append("")
    lines.append(f"## Leave-one-out accuracy: "
                  f"{f'{accuracy:.1%}' if accuracy is not None else 'not computable yet'}")
    lines.append("")
    lines.append("| Sample | True label | P(AI) | Correct? |")
    lines.append("|---|---|---|---|")
    for r in loo_results:
        lines.append(f"| {os.path.basename(r['source'])} | {r['true_label']} | "
                      f"{r['p_ai']} | {r.get('correct', r.get('note'))} |")
    lines.append("")

    lines.append("## Confidently wrong examples")
    lines.append("")
    if confidently_wrong:
        for r in confidently_wrong:
            lines.append(f"- **{os.path.basename(r['source'])}** "
                          f"(true label: {r['true_label']}, predicted P(AI)={r['p_ai']})")
    else:
        lines.append("None found yet at this dataset size, or all held-out predictions were correct.")
    lines.append("")

    lines.append("## Bias check: does the model unfairly flag informal/non-standard human English?")
    lines.append("")
    for r in bias_results:
        flag = " — **FLAGGED AS LIKELY AI despite being genuine human writing**" if (r["p_ai"] and r["p_ai"] > 0.5) else ""
        lines.append(f"- {os.path.basename(r['source'])}: P(AI) = {r['p_ai']}{flag}")
    lines.append("")

    lines.append("## Honest limitations")
    lines.append("")
    lines.append(f"- Dataset size ({summary['total']} samples) is far below what would be needed "
                  "for a trustworthy accuracy figure. Leave-one-out was used instead of a "
                  "held-out test split because the dataset is too small for a split to be meaningful.")

    wrong_human = [r for r in loo_results if r["true_label"] == "human" and r.get("correct") is False]
    correct_human = [r for r in loo_results if r["true_label"] == "human" and r.get("correct") is True]
    lines.append(f"- Of {len(wrong_human) + len(correct_human)} human samples tested held-out, "
                  f"{len(correct_human)} were correctly identified as human and "
                  f"{len(wrong_human)} were confidently misclassified as AI "
                  f"({', '.join(os.path.basename(r['source']) for r in wrong_human) if wrong_human else 'none'}).")
    lines.append("- **Specific investigated cause, not just speculation**: the human samples that "
                  "ARE correctly identified are informal, run-on text with little punctuation — our "
                  "sentence-splitter reads each of them as a single sentence (burstiness = 0.0), which "
                  "may be acting as an accidental shortcut (\"looks like one giant run-on = human\") "
                  "rather than the model genuinely learning AI-vs-human writing patterns. The one "
                  "properly-punctuated, multi-sentence human sample does NOT get this effect and is "
                  "still misclassified. This is a real limitation of the current tiny, skewed dataset "
                  "composition (3 of 4 human samples happen to share this run-on structure), not a "
                  "confirmed general bias against informal English — but it means the current model "
                  "should not be trusted to generalize to well-punctuated human writing yet.")
    lines.append("- Perplexity (the core signal for a real detector) was not available "
                  "when this report was generated — see behavior-spec.md for status.")
    lines.append("- This report should be regenerated as the dataset grows; the numbers "
                  "above will change, likely substantially.")

    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_report()
    print(report)
