"""
Stylometric and burstiness feature extraction.

These are signals that don't need any language model — just statistics
about sentence structure, word choice variety, and punctuation patterns.
Human writing tends to be "burstier" (more variation in sentence length
and rhythm); AI-generated text tends to be more uniform.

Every function here operates on plain text and returns numeric features,
which get combined with the perplexity signal (see perplexity_features.py,
which needs the local language model) into the final per-sentence score.
"""

import re
import statistics
from dataclasses import dataclass


def split_sentences(text: str) -> list[str]:
    """
    Sentence splitter: splits on '.', '!', '?' followed by whitespace
    and a capital letter (avoiding common abbreviation false-splits),
    AND treats line breaks as boundaries too — informal writing often
    uses newlines instead of periods to separate thoughts, and without
    this a whole multi-line paragraph gets read as a single sentence,
    which artificially zeroes out the burstiness signal. Found via a
    real investigation, not assumed — see behavior-spec.md.
    """
    text = text.strip()
    if not text:
        return []

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    sentences = []
    for line in lines:
        # Avoid splitting "Mr." "Dr." "e.g." etc. by requiring a capital
        # letter or quote after the punctuation+space.
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', line)
        sentences.extend(p.strip() for p in parts if p.strip())

    return sentences


def word_tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stripping punctuation."""
    return re.findall(r"[a-zA-Z']+", text.lower())


@dataclass
class SentenceStats:
    text: str
    word_count: int
    char_count: int


def sentence_length_stats(sentences: list[str]) -> dict:
    """
    Burstiness signal: human writing varies sentence length a lot
    (short punchy sentences mixed with long complex ones). AI text
    tends to hover around a narrower band of lengths.
    """
    lengths = [len(word_tokenize(s)) for s in sentences]
    lengths = [l for l in lengths if l > 0]

    if len(lengths) < 2:
        return {
            "mean_sentence_length": lengths[0] if lengths else 0,
            "stdev_sentence_length": 0.0,
            "burstiness": 0.0,
        }

    mean_len = statistics.mean(lengths)
    stdev_len = statistics.stdev(lengths)
    # Burstiness coefficient: standardized variance measure.
    # Higher = more variation between sentences = more human-typical.
    burstiness = stdev_len / mean_len if mean_len > 0 else 0.0

    return {
        "mean_sentence_length": round(mean_len, 2),
        "stdev_sentence_length": round(stdev_len, 2),
        "burstiness": round(burstiness, 3),
    }


def lexical_diversity(text: str) -> dict:
    """
    Type-token ratio: unique words / total words. AI text often
    reuses a narrower vocabulary; human text tends to have more
    varied word choice, especially in longer passages.
    """
    words = word_tokenize(text)
    if not words:
        return {"type_token_ratio": 0.0, "word_count": 0}

    unique = len(set(words))
    total = len(words)
    return {
        "type_token_ratio": round(unique / total, 3),
        "word_count": total,
    }


def punctuation_and_structure(text: str, sentences: list[str]) -> dict:
    """
    A few structural signals:
    - Repeated sentence-starter patterns (AI text often starts many
      sentences the same way: "This shows...", "Additionally...")
    - Comma density (AI text sometimes over- or under-uses commas
      in a very consistent way)
    """
    starters = [s.split()[0].lower() for s in sentences if s.split()]
    starter_repetition = 0.0
    if len(starters) > 1:
        most_common_count = max(starters.count(s) for s in set(starters))
        starter_repetition = round(most_common_count / len(starters), 3)

    comma_count = text.count(",")
    word_count = len(word_tokenize(text))
    comma_density = round(comma_count / word_count, 3) if word_count else 0.0

    return {
        "starter_repetition": starter_repetition,
        "comma_density": comma_density,
    }


def extract_stylometric_features(text: str) -> dict:
    """
    Runs all stylometric checks on a passage and returns one combined
    dict of features. This does NOT include the perplexity/burstiness-
    from-language-model signal — that's a separate module, since it
    needs the downloaded GPT-2 model.
    """
    sentences = split_sentences(text)
    features = {}
    features.update(sentence_length_stats(sentences))
    features.update(lexical_diversity(text))
    features.update(punctuation_and_structure(text, sentences))
    features["sentence_count"] = len(sentences)
    return features
