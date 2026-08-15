"""
Loads the human/AI essay dataset from data/human/ and data/ai/,
returning texts with their labels. Also tracks basic dataset stats
so we can honestly report dataset size/composition, per the brief's
requirement to document "how much there is, and what it does not cover."
"""

import glob
import os


def load_dataset(data_dir: str | None = None) -> list[dict]:
    """
    Returns a list of {"text": str, "label": "human"|"ai", "source": filename}

    If data_dir isn't given, resolves to the "data" folder next to the
    project root (one level up from this file's src/ folder) — this
    way the loader works correctly regardless of which directory the
    script is actually run from, instead of silently returning nothing.
    """
    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data")

    samples = []

    for path in sorted(glob.glob(os.path.join(data_dir, "human", "*.txt"))):
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            samples.append({"text": text, "label": "human", "source": path})

    for path in sorted(glob.glob(os.path.join(data_dir, "ai", "*.txt"))):
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            samples.append({"text": text, "label": "ai", "source": path})

    return samples


def dataset_summary(samples: list[dict]) -> dict:
    """Honest counts for the accuracy report — no hiding a small dataset."""
    human_count = sum(1 for s in samples if s["label"] == "human")
    ai_count = sum(1 for s in samples if s["label"] == "ai")
    return {
        "total": len(samples),
        "human_count": human_count,
        "ai_count": ai_count,
    }


if __name__ == "__main__":
    samples = load_dataset()
    print(dataset_summary(samples))
    for s in samples:
        print(f"  [{s['label']}] {s['source']} ({len(s['text'])} chars)")
