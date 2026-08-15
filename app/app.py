"""
Streamlit interface: paste an essay, see which parts look
machine-typical and why, plus an overall verdict.

Run with: streamlit run app/app.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import numpy as np

from dataset import load_dataset
from classifier import build_training_data, train_classifier, evaluate_leave_one_out
from feature_vector import build_feature_vector
from evidence import flag_sentences
from stylometric_features import split_sentences

# Perplexity is optional: only available once GPT-2 + torch are set up
# on this machine. Import failure is handled gracefully, not silently.
try:
    from perplexity_features import passage_perplexity_features
    PERPLEXITY_AVAILABLE = True
except Exception:
    PERPLEXITY_AVAILABLE = False


st.set_page_config(page_title="AI Essay Detector", layout="wide")
st.title("AI Essay Detector")
st.caption(
    "Paste an admissions essay to see which parts look machine-typical, "
    "and why — not just a single percentage."
)

# --- honest status banner, always visible ---
if not PERPLEXITY_AVAILABLE:
    st.warning(
        "Running in **degraded mode**: the perplexity model (GPT-2) isn't "
        "set up in this environment yet. Sentence-level evidence will be "
        "weak — see documents/behavior-spec.md for why, and the setup "
        "steps to enable full mode."
    )

# --- train the classifier on startup, on whatever data currently exists ---
samples = load_dataset()
eval_result = evaluate_leave_one_out(*build_training_data(samples))

with st.sidebar:
    st.subheader("Dataset & model status")
    st.write(f"Training samples: {len(samples)} "
             f"({sum(1 for s in samples if s['label']=='human')} human, "
             f"{sum(1 for s in samples if s['label']=='ai')} AI)")
    st.write("Evaluation:")
    st.json(eval_result)
    st.caption(
        "This is a small, honestly-reported dataset — see behavior-spec.md "
        "for full documentation of what it does and doesn't cover."
    )
    if not PERPLEXITY_AVAILABLE:
        st.error("Perplexity model not loaded — degraded mode.")
    else:
        st.success("Perplexity model loaded — full mode.")

essay_text = st.text_area("Paste an essay here", height=300)

if st.button("Analyze", type="primary") and essay_text.strip():
    sentences = split_sentences(essay_text)

    per_sentence_ppl = None
    whole_ppl_features = None
    if PERPLEXITY_AVAILABLE:
        with st.spinner("Running perplexity analysis..."):
            whole_ppl_features = passage_perplexity_features(sentences)
            per_sentence_ppl = whole_ppl_features.get("per_sentence_perplexity")

    evidence = flag_sentences(essay_text, per_sentence_ppl)

    # --- overall verdict ---
    X, y = build_training_data(samples)
    if len(X) >= 2 and len(set(y)) == 2:
        clf = train_classifier(X, y)
        query_vec = np.array([build_feature_vector(essay_text, whole_ppl_features)])
        proba = clf.predict_proba(query_vec)[0]
        ai_probability = proba[1]  # class 1 = ai

        st.subheader("Overall assessment")
        st.metric("Estimated AI-likelihood", f"{ai_probability:.0%}")
        if not PERPLEXITY_AVAILABLE:
            st.caption(
                "⚠️ This verdict was trained without real perplexity data "
                "(placeholder zeros) — treat it as a weak signal until the "
                "classifier is retrained with perplexity enabled. See "
                "behavior-spec.md."
            )
    else:
        st.info("Not enough labeled data yet to produce an overall verdict.")

    # --- sentence-level evidence ---
    st.subheader("Sentence-level evidence")
    st.caption(f"Evidence mode: **{evidence['mode']}**")

    for s in evidence["sentences"]:
        if s["flag_score"] > 0.3:
            st.markdown(f":orange[**{s['text']}**]")
            st.caption("Why: " + "; ".join(s["reasons"]))
        else:
            st.markdown(s["text"])
