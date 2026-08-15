"""
Perplexity-based detection features, using a small local language model
(GPT-2) to measure how "predictable" text is.

This is the core signal that makes this a real detector rather than a
wrapper around a chat model: we're not asking anything "is this AI?" —
we're computing actual token-level probabilities and deriving statistics
from them ourselves.

The idea: a language model assigns a probability to each word given the
words before it. AI-generated text tends to consistently pick
high-probability (predictable) words, because that's what generation
models are optimized to do. Human writing is less consistently
predictable — people make less "optimal" word choices, go on tangents,
use unexpected phrasing.

Two signals come out of this:
1. Mean perplexity: overall how "surprised" the model is by the text.
   Lower perplexity = more predictable = more AI-typical.
2. Perplexity burstiness: how much the per-token surprise VARIES across
   the passage. Human text tends to have uneven predictability sentence
   to sentence; heavily-AI text tends to be uniformly predictable
   throughout.

NOTE: this module requires the `transformers` and `torch` packages, and
on first run will download the GPT-2 model weights (~500MB) from
HuggingFace. That download must happen with real internet access —
see documents/behavior-spec.md for exactly how this was verified.
"""

import math
import statistics

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

_MODEL_NAME = "gpt2"
_model = None
_tokenizer = None


def _load_model():
    """
    Lazy-loads the model once and reuses it across calls — loading GPT-2
    fresh on every request would be slow. First call downloads the
    weights if they aren't already cached locally.
    """
    global _model, _tokenizer
    if _model is None:
        _tokenizer = GPT2TokenizerFast.from_pretrained(_MODEL_NAME)
        _model = GPT2LMHeadModel.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _model, _tokenizer


def sentence_perplexity(sentence: str) -> float:
    """
    Computes the perplexity of a single sentence under GPT-2.
    Perplexity = exp(average negative log-likelihood per token).
    Lower = more predictable to the model.
    """
    model, tokenizer = _load_model()

    encodings = tokenizer(sentence, return_tensors="pt")
    input_ids = encodings.input_ids

    if input_ids.shape[1] < 2:
        # Too short to compute meaningful next-token probabilities
        return float("nan")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss is the average negative log-likelihood per token
        neg_log_likelihood = outputs.loss.item()

    return math.exp(neg_log_likelihood)


def passage_perplexity_features(sentences: list[str]) -> dict:
    """
    Computes per-sentence perplexity for a whole passage and derives
    the two aggregate signals described above. Also returns the raw
    per-sentence scores so the interface can show exactly which
    sentences were flagged and why (required by the brief — evidence,
    not just a percentage).
    """
    per_sentence = []
    for s in sentences:
        try:
            score = sentence_perplexity(s)
        except Exception:
            score = float("nan")
        per_sentence.append(score)

    valid_scores = [s for s in per_sentence if not math.isnan(s)]

    if len(valid_scores) < 2:
        return {
            "mean_perplexity": valid_scores[0] if valid_scores else None,
            "perplexity_burstiness": None,
            "per_sentence_perplexity": per_sentence,
        }

    mean_ppl = statistics.mean(valid_scores)
    stdev_ppl = statistics.stdev(valid_scores)
    burstiness = stdev_ppl / mean_ppl if mean_ppl > 0 else 0.0

    return {
        "mean_perplexity": round(mean_ppl, 2),
        "perplexity_burstiness": round(burstiness, 3),
        "per_sentence_perplexity": [
            round(s, 2) if not math.isnan(s) else None for s in per_sentence
        ],
    }
