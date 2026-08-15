# AI Essay Detector — Behavior Spec

Documents what's built, how it was verified, and honest limitations. Same format/purpose as Project 1's spec.

## Component: Stylometric feature extraction (`src/stylometric_features.py`)

**Status: built and verified.**

Extracts, per passage of text:
- Sentence-length statistics (mean, standard deviation) and a burstiness coefficient (stdev/mean) — human writing tends to vary sentence length more than AI text
- Type-token ratio (unique words / total words) — a lexical diversity signal
- Sentence-starter repetition — flags text where many sentences begin the same way, a common AI tell
- Comma density

**Verified:**
- Ran on the first real human sample (`data/human/sample_01.txt`) — produced sensible, non-crashing output (burstiness 0.297, TTR 0.584, etc.)
- Ran on edge cases: empty string, single word, single sentence — all handled gracefully with zero-values, no exceptions

**Does not require any model download** — pure Python, works identically in any environment.

## Component: Perplexity feature extraction (`src/perplexity_features.py`)

**Status: code written, syntax-verified, runtime NOT yet verified.**

Uses GPT-2 (via `transformers`/`torch`) to compute per-sentence perplexity and derive two signals: mean perplexity, and perplexity burstiness (variation across sentences).

**Verified:** Python syntax and AST parse cleanly (`py_compile`, `ast.parse`) — no syntax errors, correct structure.

**NOT verified here:** actual model loading and inference. The development sandbox used for this project has network restrictions that block downloading model weights from HuggingFace, and ran out of disk space attempting to install `torch`. This must be verified on a machine with normal internet access and disk space — see the run-it-yourself steps for exact verification commands.

## Dataset progress

- Human samples: 1 so far (`data/human/sample_01.txt`), contributed directly by the project owner in their own words — necessary to avoid any copyright issue with using real people's essays found online.
- AI samples: 3 generated (`data/ai/sample_01-03.txt`), admissions-essay style, covering different common prompts (overcoming a challenge, community involvement, why this field of study). Generated directly, clearly disclosed as AI-authored for this dataset.
- **Honest early observation** (illustrative only — nowhere near enough data for a real conclusion): running the stylometric features on the 1 human + 3 AI samples so far shows burstiness lower for all 3 AI samples than the human one (expected direction), but type-token ratio (lexical diversity) went the *opposite* direction of what the code's own assumptions expected. Flagged here rather than ignored — worth re-checking once the dataset is larger.

## Bug found and fixed: dataset loader path resolution

- What: `load_dataset()` defaulted to a relative path `"data"`, which only worked if the script happened to be run from the project's root folder. Running it from `src/` (a completely normal thing to do) silently returned an empty dataset — no error, just wrong/empty results, which is worse than crashing since it could go unnoticed.
- Fix: path now resolves relative to the module's own file location, not the current working directory. Verified: re-ran from `src/` after the fix, correctly found all 5 samples.
- Caught by actually running the code and inspecting output, not by inspection alone — consistent with the verification approach used throughout Project 1.

## Component: Classifier (`src/classifier.py`)

**Status: pipeline built and verified with real (tiny) data — accuracy number is NOT meaningful yet, explicitly.**

Logistic regression, chosen deliberately over a more complex model — with a small dataset, an interpretable model is more honest than one that would just overfit.

Ran leave-one-out cross-validation (more appropriate than a train/test split when the dataset is this small) on the current 5 samples (2 human, 3 AI) — **using stylometric features only, perplexity still at placeholder 0.0 since it can't run in this environment**. Result: 0.6 accuracy.

**This number must not be reported as real accuracy in the final submission.** With 5 samples and no perplexity signal yet, this is a pipeline sanity check only — it confirms the code runs correctly end-to-end, nothing more. Real evaluation happens once (a) perplexity is wired in on the target machine, and (b) the dataset has grown substantially.

## Design decision: sentence-level evidence, not a sentence-level classifier

The brief requires showing *where* in an essay the AI-likelihood is, not just an overall score. Important honesty point: we don't have sentence-level labeled ground truth (only whole-essay labels), so a genuinely separate sentence-level classifier isn't something we can honestly train or validate.

Instead: the overall verdict comes from the whole-essay classifier, and sentence-level highlighting uses each sentence's own perplexity relative to the essay's average — sentences that are unusually predictable (low perplexity) compared to the rest of that same essay get flagged as the strongest evidence. This is a relative-outlier heuristic, not a separately-validated classification — documented explicitly as such rather than implied to be more rigorous than it is.

## Component: Sentence-level evidence (`src/evidence.py`)

**Status: built and verified, in degraded mode (no perplexity available in this environment).**

Two modes, both honestly labeled in the output:
- **Full mode**: uses per-sentence perplexity relative to the essay's own average, flagging sentences that are unusually predictable compared to the rest of that specific essay — plus a weak supporting stylometric signal.
- **Degraded mode**: perplexity unavailable, falls back to the stylometric signal alone.

**Real, honest finding from testing**: ran on an AI sample in degraded mode — nearly every sentence scored 0.0 evidence, with only one weak flag firing. This confirms the stylometric-only fallback is genuinely weak on its own; perplexity is the load-bearing signal for this detector, not a nice-to-have. This will be stated plainly in the final report rather than glossed over.

## Component: Streamlit interface (`app/app.py`)

**Status: built and verified running.**

- Started successfully with no Python errors (checked the actual server log, not just "no crash on launch")
- Confirmed serving real HTML content via a direct request, not just that the process stayed alive
- Honestly surfaces its own limitations in the UI itself: a visible warning banner when perplexity isn't available ("degraded mode"), and a caption on the overall verdict noting the classifier hasn't been trained with real perplexity data yet
- Sidebar shows live, honest dataset stats (sample counts, evaluation result) — no hardcoded/fake numbers

**Known gap, by design, not yet closed**: the classifier is currently trained using placeholder zeros for perplexity (since this sandbox can't run GPT-2), so even once perplexity is available at prediction time, the trained model's weights don't yet meaningfully use it. Closing this requires retraining the classifier on the dataset WITH real perplexity computed for every training sample too, once verified working — this is the next step once perplexity is confirmed running on the target machine.

## Component: Bias check (`src/bias_check.py`)

**Status: built, tested, and it surfaced a real, important finding.**

Uses leave-one-out (train excluding the sample, predict on it as genuinely unseen) rather than testing on training data — the naive version of this test (not excluding the sample) gave misleadingly reassuring results and was corrected before being trusted.

**Real finding — both human samples confidently misclassified as AI when held out:**
- `sample_01.txt`: P(AI) = 0.999
- `sample_02.txt`: P(AI) = 0.919

**Honest interpretation**: this is very likely a consequence of having only 2 human samples total — leave-one-out means the model trains on just 1 human example when predicting the other, which isn't enough to learn what "human" looks like at all. It is NOT yet possible to distinguish this from a genuine bias against informal/non-standard English (which sample_02 specifically represents) with this little data — both explanations are consistent with the result. This is exactly the kind of honest, examined limitation the brief asks for, and will be reported as such rather than either hidden or overclaimed as a confirmed bias finding.

**This reprioritizes remaining work**: growing the human sample count is now the single highest-value remaining task — without it, neither the overall accuracy numbers nor the bias check can produce a meaningful result, no matter how much other code gets built.

## Component: Accuracy report generator (`src/report.py`)

**Status: built, tested, output saved to `documents/accuracy-report.md`.**

Generates the full report the brief requires: dataset summary, leave-one-out accuracy, a table of every sample's held-out prediction, the confidently-wrong examples (correctly found only 2 at this dataset size, not padded to a fake 3), the bias check results, and an explicit "honest limitations" section.

This is regenerable — re-running `python3 src/report.py` after the dataset grows produces an updated report reflecting the real current state, not a one-time snapshot.

## Bug found and fixed: sentence-splitter ignored line breaks

- What: `split_sentences()` only split on `. ! ?` followed by a capital letter — text written with line breaks instead of periods (common in informal writing) was read as a single giant "sentence," artificially zeroing out the burstiness signal for that text.
- Found via direct investigation of why some human samples were confidently misclassified while others weren't — not assumed, traced to the actual root cause.
- Fix: now splits on line breaks first, then applies the punctuation-based split within each line.
- Verified: re-ran on all 11 samples before trusting the fix — `sample_03.txt` correctly improved (1 → 3 sentences), all 7 AI samples were completely unaffected (no regression), and the 2 human samples that remain single-sentence do so because that's honestly how they were written (no line breaks at all), not because of a remaining bug.

## Updated, more precise finding after this fix + dataset growth

Rerunning the full report after the fix produced a very clean, precisely-explained result: the two human samples correctly identified as human (`sample_02`, `sample_04`) are exactly the two that are still genuine single-sentence run-ons (burstiness 0.0). The two misclassified as AI (`sample_01`, `sample_03`) are exactly the two with real multi-sentence structure. Leave-one-out accuracy: 72.7% overall (11 samples) — still far too small a dataset to trust as a real number, but the *reason* for each specific error is now precisely identified rather than guessed at, which is what the brief actually asks for ("three essays it gets confidently wrong, and your explanation of why").
