# Accuracy Report

**Dataset**: 11 samples (4 human, 7 AI)

**This dataset is currently very small.** The numbers below are honestly reported, but should not be read as reliable — see the explicit caveat at the end of this report.

## Leave-one-out accuracy: 72.7%

| Sample | True label | P(AI) | Correct? |
|---|---|---|---|
| sample_01.txt | human | 0.987 | False |
| sample_02.txt | human | 0.0 | True |
| sample_03.txt | human | 0.996 | False |
| sample_04.txt | human | 0.0 | True |
| sample_01.txt | ai | 0.565 | True |
| sample_02.txt | ai | 0.453 | False |
| sample_03.txt | ai | 0.508 | True |
| sample_04.txt | ai | 0.907 | True |
| sample_05.txt | ai | 0.94 | True |
| sample_06.txt | ai | 0.941 | True |
| sample_07.txt | ai | 0.933 | True |

## Confidently wrong examples

- **sample_03.txt** (true label: human, predicted P(AI)=0.996)
- **sample_01.txt** (true label: human, predicted P(AI)=0.987)
- **sample_02.txt** (true label: ai, predicted P(AI)=0.453)

## Bias check: does the model unfairly flag informal/non-standard human English?

- sample_01.txt: P(AI) = 0.987 — **FLAGGED AS LIKELY AI despite being genuine human writing**
- sample_02.txt: P(AI) = 0.0
- sample_03.txt: P(AI) = 0.996 — **FLAGGED AS LIKELY AI despite being genuine human writing**
- sample_04.txt: P(AI) = 0.0

## Honest limitations

- Dataset size (11 samples) is far below what would be needed for a trustworthy accuracy figure. Leave-one-out was used instead of a held-out test split because the dataset is too small for a split to be meaningful.
- Of 4 human samples tested held-out, 2 were correctly identified as human and 2 were confidently misclassified as AI (sample_01.txt, sample_03.txt).
- **Specific investigated cause, not just speculation**: the human samples that ARE correctly identified are informal, run-on text with little punctuation — our sentence-splitter reads each of them as a single sentence (burstiness = 0.0), which may be acting as an accidental shortcut ("looks like one giant run-on = human") rather than the model genuinely learning AI-vs-human writing patterns. The one properly-punctuated, multi-sentence human sample does NOT get this effect and is still misclassified. This is a real limitation of the current tiny, skewed dataset composition (3 of 4 human samples happen to share this run-on structure), not a confirmed general bias against informal English — but it means the current model should not be trusted to generalize to well-punctuated human writing yet.
- Perplexity (the core signal for a real detector) was not available when this report was generated — see behavior-spec.md for status.
- This report should be regenerated as the dataset grows; the numbers above will change, likely substantially.
