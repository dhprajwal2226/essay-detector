# AI Essay Detector

Detects which parts of a college admissions essay were likely written by AI, with visible evidence per sentence — not a single opaque percentage, and not a wrapper that asks a chat model for a verdict.

## How it works

Two independent signals, combined:

1. **Perplexity** (`src/perplexity_features.py`) — uses a small local language model (GPT-2) to measure how "predictable" each sentence is. AI-generated text tends to consistently choose high-probability wording; human writing is less uniformly predictable.
2. **Stylometric features** (`src/stylometric_features.py`) — sentence-length variation ("burstiness"), lexical diversity, sentence-starter repetition, and comma density. Computed with plain Python, no model needed.

These combine into a feature vector (`src/feature_vector.py`) fed to a logistic regression classifier (`src/classifier.py`), chosen deliberately over a more complex model — with a small dataset, an interpretable model is more honest than one that would just overfit.

Sentence-level evidence (`src/evidence.py`) flags sentences that are unusually predictable *relative to the rest of that same essay*, with the reason shown alongside — this is what the interface actually displays, not just an overall score.

## Setup

```
pip install -r requirements.txt
```

The first time you run anything that uses `perplexity_features.py`, it will download the GPT-2 model weights (~500MB) from HuggingFace — this requires a real internet connection and about 1GB of free disk space, and only happens once.

## Running the interface

```
streamlit run app/app.py
```

Opens in your browser. Paste an essay, click Analyze.

## Running the accuracy report

```
python3 src/report.py
```

Regenerates `documents/accuracy-report.md` using leave-one-out evaluation on the current dataset — re-run this any time the dataset grows.

## Project structure

```
src/
  stylometric_features.py   — sentence-length, lexical diversity, structure signals (no model needed)
  perplexity_features.py    — GPT-2-based predictability signals (needs model download)
  feature_vector.py         — combines both into one fixed-shape vector
  classifier.py             — trains/evaluates the logistic regression model
  evidence.py                — per-sentence flagging with reasons
  bias_check.py              — tests for unfair flagging of non-standard English
  report.py                  — generates the full accuracy report
  dataset.py                  — loads data/human/ and data/ai/
app/
  app.py                     — Streamlit interface
data/
  human/                     — genuine human-written samples (contributed directly, not scraped, to avoid copyright issues)
  ai/                        — AI-generated samples, clearly disclosed as such
documents/
  behavior-spec.md            — full build log: what's built, what's verified, every bug found
  accuracy-report.md          — current honest accuracy report (regenerable)
```

## Honest current limitations

See `documents/behavior-spec.md` and `documents/accuracy-report.md` for the full, current, honest picture — including a real finding that the model currently fails to generalize to unseen human writing at all, due to the small dataset size. This is documented rather than hidden, in line with the project brief's explicit preference for honest limitations over an unbelievable accuracy claim.
