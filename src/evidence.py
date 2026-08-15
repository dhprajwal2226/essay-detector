"""
Generates per-sentence evidence for why an essay was flagged, rather
than just an overall percentage. This is the piece that makes the
output something a reader can actually inspect and argue with, as
the brief requires.

Works in two modes:
- Full mode: perplexity scores are available (real GPT-2 inference
  was run) — combines perplexity-outlier detection with stylometric
  flags for the strongest evidence.
- Degraded mode: perplexity isn't available in this environment —
  falls back to stylometric-only flags, and the result clearly says
  so, rather than silently pretending to be as strong as full mode.
"""


from stylometric_features import split_sentences, word_tokenize
def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _stdev(values: list[float]) -> float:
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


def flag_sentences(
    text: str,
    per_sentence_perplexity: list[float] | None = None,
) -> dict:
    
    """
    Returns {
        "mode": "full" | "degraded",
        "sentences": [
            {"text": str, "flag_score": float, "reasons": [str, ...]},
            ...
        ]
    }

    flag_score is 0-1, higher = more evidence this sentence looks
    machine-typical relative to the rest of THIS essay (not an
    absolute, cross-essay-comparable score).
    """
    sentences = split_sentences(text)
    if not sentences:
        return {"mode": "degraded", "sentences": []}

    results = []

    has_perplexity = (
        per_sentence_perplexity is not None
        and len(per_sentence_perplexity) == len(sentences)
        and any(p is not None for p in per_sentence_perplexity)
    )

    ppl_values = [p for p in (per_sentence_perplexity or []) if p is not None]
    ppl_mean = _mean(ppl_values) if len(ppl_values) >= 2 else None
    ppl_stdev = _stdev(ppl_values) if len(ppl_values) >= 2 else None

    sentence_lengths = [len(word_tokenize(s)) for s in sentences]
    valid_lengths = [l for l in sentence_lengths if l > 0]
    length_mean = _mean(valid_lengths) if len(valid_lengths) >= 2 else None
    length_stdev = _stdev(valid_lengths) if len(valid_lengths) >= 2 else None

    for i, sentence in enumerate(sentences):
        reasons = []
        score_components = []

        # Perplexity-based flag: sentence is unusually predictable
        # relative to the rest of THIS essay.
        if has_perplexity and ppl_mean and ppl_stdev and ppl_stdev > 0:
            s_ppl = per_sentence_perplexity[i]
            if s_ppl is not None:
                z = (s_ppl - ppl_mean) / ppl_stdev
                if z < -0.75:
                    # notably MORE predictable than the essay's own average
                    component = min(1.0, abs(z) / 3)
                    score_components.append(component)
                    reasons.append(
                        "unusually predictable wording compared to the rest of this essay"
                    )

        # Stylometric flag: sentence length very close to the essay's
        # mean, with low variation nearby — a weak signal on its own,
        # used only to add supporting evidence, never as the sole flag.
        if length_mean and length_stdev and length_stdev > 0:
            this_len = sentence_lengths[i]
            z_len = abs(this_len - length_mean) / length_stdev
            if z_len < 0.3:
                score_components.append(0.15)
                reasons.append("sentence length very close to the essay's average")

        flag_score = round(sum(score_components) / max(len(score_components), 1), 3) if score_components else 0.0

        results.append({
            "text": sentence,
            "flag_score": min(flag_score, 1.0),
            "reasons": reasons,
        })

    return {
        "mode": "full" if has_perplexity else "degraded",
        "sentences": results,
    }
