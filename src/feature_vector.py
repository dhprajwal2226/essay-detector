"""
Combines stylometric features (always available) with perplexity
features (available once GPT-2 is set up) into one flat feature
vector for the classifier.

Perplexity features are optional here on purpose: this lets us build
and test the full pipeline structure now using stylometric features
alone, and plug in real perplexity values later without changing this
module's interface.
"""

from stylometric_features import extract_stylometric_features

# Fixed order of feature names -> the classifier always sees features
# in this exact order, whether or not perplexity is available.
FEATURE_NAMES = [
    "mean_sentence_length",
    "stdev_sentence_length",
    "burstiness",
    "type_token_ratio",
    "starter_repetition",
    "comma_density",
    "mean_perplexity",
    "perplexity_burstiness",
]


def build_feature_vector(text: str, perplexity_features: dict | None = None) -> list[float]:
    """
    Returns a fixed-length numeric feature vector for one essay.
    If perplexity_features is None (GPT-2 not available in this
    environment), those two slots are filled with 0.0 as a neutral
    placeholder — NOT a real signal, just keeps the vector shape
    consistent so the same code path works with or without it.
    """
    stylo = extract_stylometric_features(text)

    ppl_mean = 0.0
    ppl_burst = 0.0
    if perplexity_features:
        ppl_mean = perplexity_features.get("mean_perplexity") or 0.0
        ppl_burst = perplexity_features.get("perplexity_burstiness") or 0.0

    return [
        stylo["mean_sentence_length"],
        stylo["stdev_sentence_length"],
        stylo["burstiness"],
        stylo["type_token_ratio"],
        stylo["starter_repetition"],
        stylo["comma_density"],
        ppl_mean,
        ppl_burst,
    ]
