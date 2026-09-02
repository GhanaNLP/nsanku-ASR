"""Evaluation recipe for facebook/omniASR-CTC-1B-v2 — all languages.

Track: ASR (open models) — Meta Omnilingual ASR, CTC head (omniASR_CTC_1B_v2)
Scope: ALL LANGUAGES. This is the base recipe; per-language recipes
(recipes/facebook_omniASR-CTC-1B-v2__{iso}.py) hold the knobs for individual
languages and are what the run actually reads.

This is the CTC sibling of the omniASR LLM checkpoints: the same wav2vec2-style
speech encoder, but a plain CTC head instead of a transformer decoder. The
weights are downloadable under apache-2.0, so it is scored in the open ASR
track rather than the LLM track.

A CTC head has no language conditioning. The inference pipeline still accepts a
`lang` argument for interface parity with the LLM checkpoints, but the decode is
byte-for-byte identical with and without it, so the per-language recipes leave
LANG_ID as None and the score reflects unconditioned decoding.

It runs on GPU (bf16) in its own virtualenv (.venv-omniasr) because fairseq2
pins torch 2.8 while the main benchmark venv uses torch 2.11:

    .venv-omniasr/bin/python run_omniasr.py --preset ctc-1b

Edit this file and open a pull request at
https://github.com/GhanaNLP/nsanku-ASR to change how this model is evaluated.
"""

MODEL_ID = "facebook/omniASR-CTC-1B-v2"
MODEL_URL = "https://huggingface.co/facebook/omniASR-CTC-1B-v2"
MODEL_CARD = "omniASR_CTC_1B_v2"
OWNER = "facebook"
PARAMS = "1B"
MODEL_CLASS = "non-llm"


def lang_id_for(iso_code):
    """CTC decodes unconditioned — no language hint is used."""
    return None
